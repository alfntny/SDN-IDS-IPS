"""
Ryu Controller — Real-Time IDS + IPS App (Multiclass)
SDN IDS/IPS Project — SRM SIST, Batch B1025
Alfin Tony & Harshaditya Choudhary

Now shows specific attack type in alerts:
[ALERT] 10.0.0.1 → 10.0.0.2 | conf=0.999 | DDoS | BLOCKED

Run:
    ryu-manager ryu_ids_app.py ryu.app.simple_switch_13
"""

import os
import pickle
import time
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ipv4, tcp, udp, icmp
from ryu.lib import hub

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MODELS_DIR          = "models"
DATA_DIR            = "data/processed"
POLL_INTERVAL       = 2
WINDOW_SIZE         = 10
MIN_PKTS = 1    # 1 = original behaviour (skip only empty flows); 20 suppressed blocking in live test
BLOCK_THRESHOLD     = 0.85
RATELIMIT_THRESHOLD = 0.50
BLOCK_TIMEOUT       = 60
LOG_FILE            = "ids_alerts.log"


# ─── BINARY LSTM (for attack/normal decision) ─────────────────────────────────
class LSTMDetector(nn.Module):
    def __init__(self, n_features, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, 2)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.classifier(self.dropout(out))


# ─── MULTICLASS LSTM (for attack type identification) ─────────────────────────
class LSTMMulticlass(nn.Module):
    def __init__(self, n_features, n_classes, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, n_classes)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.classifier(self.dropout(out))


# ─── RYU APP ──────────────────────────────────────────────────────────────────
class IDSIPSApp(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.datapaths     = {}
        self.mac_to_port   = {}
        self.flow_windows  = defaultdict(list)
        self.blocked_flows = set()
        self.alert_count   = 0
        self.mac_to_ip     = {}
        self.port_to_ip    = {}
        self.stats         = {
            "total": 0, "blocked": 0,
            "ratelimited": 0, "allowed": 0,
            "attack_types": defaultdict(int)
        }

        # Load scaler
        with open(f"{DATA_DIR}/scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)

        # Load binary LSTM
        ckpt = torch.load(f"{MODELS_DIR}/lstm_detector.pt",
                          map_location="cpu", weights_only=False)
        self.binary_model = LSTMDetector(
            n_features=ckpt["n_features"],
            hidden_size=ckpt["hidden_size"],
            num_layers=ckpt["num_layers"],
            dropout=ckpt["dropout"],
        )
        self.binary_model.load_state_dict(ckpt["model_state"])
        self.binary_model.eval()
        self.n_features = ckpt["n_features"]

        # Load multiclass LSTM
        mc_ckpt = torch.load(f"{MODELS_DIR}/lstm_multiclass.pt",
                              map_location="cpu", weights_only=False)
        self.multi_model = LSTMMulticlass(
            n_features=mc_ckpt["n_features"],
            n_classes=mc_ckpt["n_classes"],
            hidden_size=mc_ckpt["hidden_size"],
            num_layers=mc_ckpt["num_layers"],
            dropout=mc_ckpt["dropout"],
        )
        self.multi_model.load_state_dict(mc_ckpt["model_state"])
        self.multi_model.eval()
        self.class_names = mc_ckpt["class_names"]

        self.logger.info("=" * 60)
        self.logger.info("  SDN IDS/IPS System Started (Multiclass Mode)")
        self.logger.info(f"  Binary LSTM  | Features: {self.n_features}")
        self.logger.info(f"  Multiclass LSTM | Classes: {self.class_names}")
        self.logger.info("=" * 60)

        self.monitor_thread = hub.spawn(self._monitor)

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        dp      = ev.msg.datapath
        ofproto = dp.ofproto
        parser  = dp.ofproto_parser
        self.datapaths[dp.id] = dp
        self.mac_to_port.setdefault(dp.id, {})
        match   = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                           ofproto.OFPCML_NO_BUFFER)]
        self._add_flow(dp, 0, match, actions)
        self.logger.info(f"[SDN] Switch connected: dpid={dp.id}")

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        msg     = ev.msg
        dp      = msg.datapath
        ofproto = dp.ofproto
        parser  = dp.ofproto_parser
        in_port = msg.match["in_port"]

        pkt     = packet.Packet(msg.data)
        eth_pkt = pkt.get_protocol(ethernet.ethernet)
        if eth_pkt is None:
            return

        dst_mac = eth_pkt.dst
        src_mac = eth_pkt.src
        dpid    = dp.id

        self.mac_to_port[dpid][src_mac] = in_port

        ip_pkt = pkt.get_protocol(ipv4.ipv4)
        src_ip = None
        dst_ip = None

        if ip_pkt:
            src_ip = ip_pkt.src
            dst_ip = ip_pkt.dst
            self.mac_to_ip[src_mac] = src_ip
            self.mac_to_ip[dst_mac] = dst_ip

        if dst_mac in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst_mac]
        else:
            out_port = ofproto.OFPP_FLOOD

        actions = [parser.OFPActionOutput(out_port)]

        if out_port != ofproto.OFPP_FLOOD:
            if ip_pkt:
                match = parser.OFPMatch(
                    in_port=in_port, eth_type=0x0800,
                    ipv4_src=src_ip, ipv4_dst=dst_ip
                )
                self.port_to_ip[(dpid, in_port, out_port)] = (src_ip, dst_ip)
            else:
                match = parser.OFPMatch(
                    in_port=in_port, eth_dst=dst_mac, eth_src=src_mac
                )
            if msg.buffer_id != ofproto.OFP_NO_BUFFER:
                self._add_flow(dp, 1, match, actions, msg.buffer_id)
            else:
                self._add_flow(dp, 1, match, actions)

        data = msg.data if msg.buffer_id == ofproto.OFP_NO_BUFFER else None
        out  = parser.OFPPacketOut(
            datapath=dp, buffer_id=msg.buffer_id,
            in_port=in_port, actions=actions, data=data
        )
        dp.send_msg(out)

    def _monitor(self):
        while True:
            for dp in list(self.datapaths.values()):
                self._request_flow_stats(dp)
            hub.sleep(POLL_INTERVAL)

    def _request_flow_stats(self, dp):
        parser = dp.ofproto_parser
        dp.send_msg(parser.OFPFlowStatsRequest(dp))

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def flow_stats_reply_handler(self, ev):
        dp = ev.msg.datapath
        for stat in ev.msg.body:
            if stat.priority in (0, 100):  # skip table-miss and our own DROP rules
                continue
            self._analyze_flow(dp, stat)

    def _analyze_flow(self, dp, stat):
        duration   = max(stat.duration_sec + stat.duration_nsec * 1e-9, 1e-9)
        pkt_count  = stat.packet_count
        byte_count = stat.byte_count
        if pkt_count < MIN_PKTS:
            return

        # Resolve IPs
        src_ip = stat.match.get("ipv4_src", None)
        dst_ip = stat.match.get("ipv4_dst", None)
        if not src_ip:
            in_port = stat.match.get("in_port", None)
            dpid    = dp.id
            for (d, ip, op), (s, dst) in self.port_to_ip.items():
                if d == dpid and ip == in_port:
                    src_ip = s
                    dst_ip = dst
                    break
        if not src_ip:
            eth_src = stat.match.get("eth_src", None)
            eth_dst = stat.match.get("eth_dst", None)
            if eth_src and eth_src in self.mac_to_ip:
                src_ip = self.mac_to_ip[eth_src]
            if eth_dst and eth_dst in self.mac_to_ip:
                dst_ip = self.mac_to_ip[eth_dst]

        bytes_per_pkt = byte_count / max(pkt_count, 1)
        pkts_per_sec  = pkt_count / duration
        bytes_per_sec = byte_count / duration
        iat_mean      = duration / max(pkt_count, 1)

        features = [
            float(stat.match.get("in_port", 0)),
            0.0, duration,
            float(pkt_count), 0.0,
            float(byte_count), 0.0,
            bytes_per_pkt, 0.0, bytes_per_pkt, 0.0,
            0.0, 0.0, 0.0, 0.0,
            bytes_per_sec, pkts_per_sec,
            iat_mean, 0.0, duration, 0.0,
            duration, iat_mean, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0,
            pkts_per_sec, 0.0,
            0.0, bytes_per_pkt, bytes_per_pkt, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, bytes_per_pkt, bytes_per_pkt, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            float(pkt_count), float(byte_count), 0.0, float(byte_count),
            0.0, 0.0,
            float(pkt_count), 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
        ]

        if len(features) < self.n_features:
            features += [0.0] * (self.n_features - len(features))
        features = features[:self.n_features]

        scaled = self.scaler.transform([features])[0]

        key    = (dp.id, src_ip, dst_ip)
        window = self.flow_windows[key]
        window.append(scaled)
        if len(window) > WINDOW_SIZE:
            window.pop(0)

        if len(window) == WINDOW_SIZE:
            self._run_inference(dp, key, window, src_ip, dst_ip)

    def _run_inference(self, dp, key, window, src_ip, dst_ip):
        x = torch.tensor(
            np.array(window, dtype=np.float32)[np.newaxis],
            dtype=torch.float32
        )

        with torch.no_grad():
            # Binary detection
            bin_logits  = self.binary_model(x)
            bin_probs   = torch.softmax(bin_logits, dim=1)[0]
            attack_prob = bin_probs[1].item()

            # Multiclass identification
            mc_logits = self.multi_model(x)
            mc_probs  = torch.softmax(mc_logits, dim=1)[0]
            mc_idx    = mc_probs.argmax().item()
            mc_label  = self.class_names[mc_idx]
            mc_conf   = mc_probs[mc_idx].item()

        self.stats["total"] += 1

        if attack_prob >= BLOCK_THRESHOLD:
            self._alert(dp, src_ip, dst_ip, attack_prob, mc_label, "BLOCKED")
            if src_ip and dst_ip:
                self._block_flow(dp, src_ip, dst_ip)
            self.stats["blocked"] += 1
            self.stats["attack_types"][mc_label] += 1
        elif attack_prob >= RATELIMIT_THRESHOLD:
            self._alert(dp, src_ip, dst_ip, attack_prob, mc_label, "RATE-LIMITED")
            self.stats["ratelimited"] += 1
            self.stats["attack_types"][mc_label] += 1
        else:
            self.stats["allowed"] += 1

        if self.stats["total"] % 10 == 0:
            attack_summary = dict(self.stats["attack_types"])
            self.logger.info(
                f"[STATS] Total={self.stats['total']} | "
                f"Blocked={self.stats['blocked']} | "
                f"Rate-Limited={self.stats['ratelimited']} | "
                f"Allowed={self.stats['allowed']} | "
                f"Types={attack_summary}"
            )

    def _alert(self, dp, src_ip, dst_ip, confidence, attack_type, action):
        self.alert_count += 1
        ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        src = src_ip if src_ip else "unknown"
        dst = dst_ip if dst_ip else "unknown"
        msg = (f"[ALERT #{self.alert_count}] {ts} | "
               f"dpid={dp.id} | {src} → {dst} | "
               f"conf={confidence:.3f} | {attack_type} | {action}")
        self.logger.warning(msg)
        with open(LOG_FILE, "a") as f:
            f.write(msg + "\n")

    def _block_flow(self, dp, src_ip, dst_ip):
        key = (src_ip, dst_ip)
        if key in self.blocked_flows:
            return
        self.blocked_flows.add(key)
        parser = dp.ofproto_parser
        match  = parser.OFPMatch(
            eth_type=0x0800,
            ipv4_src=src_ip,
            ipv4_dst=dst_ip
        )
        self._add_flow(dp, priority=100, match=match,
                       actions=[], hard_timeout=BLOCK_TIMEOUT)
        self.logger.warning(f"[IPS] ⛔ BLOCKED {src_ip} → {dst_ip} ({BLOCK_TIMEOUT}s)")

    def _add_flow(self, dp, priority, match, actions,
                  buffer_id=None, hard_timeout=0, idle_timeout=0):
        ofproto = dp.ofproto
        parser  = dp.ofproto_parser
        inst    = [parser.OFPInstructionActions(
            ofproto.OFPIT_APPLY_ACTIONS, actions
        )]
        kwargs = dict(
            datapath=dp, priority=priority,
            match=match, instructions=inst,
            hard_timeout=hard_timeout, idle_timeout=idle_timeout,
        )
        if buffer_id and buffer_id != ofproto.OFP_NO_BUFFER:
            kwargs["buffer_id"] = buffer_id
        dp.send_msg(parser.OFPFlowMod(**kwargs))
