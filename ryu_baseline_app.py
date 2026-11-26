"""
Ryu Controller — Packet-rate THRESHOLD baseline IPS (no machine learning)
SDN IDS/IPS Project — SRM SIST, Batch B1025

Same polling, same IP resolution, same DROP rule as ryu_ids_app.py.
The only difference: the decision is a packets-per-second threshold computed
from the DELTA between consecutive flow-stats polls, not a BiLSTM.

    block if  delta_packets / delta_duration >= PPS_THRESHOLD
              for CONSECUTIVE consecutive polls

Alert format (same as the ML app, so the same grep works):
[ALERT #1] 19:06:54 | 10.0.0.1 -> 10.0.0.2 | rate=92000 pkt/s | RATE | BLOCKED

Run:
    ryu-manager ryu_baseline_app.py ryu.app.simple_switch_13
"""

import os
import time
from datetime import datetime
from collections import defaultdict


from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ipv4, tcp, udp, icmp
from ryu.lib import hub

# ─── CONFIG ───────────────────────────────────────────────────────────────────
POLL_INTERVAL       = 2
MIN_PKTS            = 1
PPS_THRESHOLD       = 1000.0   # packets/s, measured between polls
CONSECUTIVE         = 2        # consecutive polls above threshold before blocking
BLOCK_TIMEOUT       = 60
LOG_FILE            = "baseline_alerts.log"


# ─── RYU APP ──────────────────────────────────────────────────────────────────
class BaselineIPSApp(app_manager.RyuApp):
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

        self.prev = {}          # key -> (pkt_count, duration)
        self.hits = defaultdict(int)  # key -> consecutive polls over threshold

        self.logger.info("=" * 60)
        self.logger.info("  SDN IPS Baseline Started (packet-rate threshold)")
        self.logger.info(f"  Threshold: {PPS_THRESHOLD} pkt/s over {CONSECUTIVE} consecutive polls")
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

        key = (dp.id, src_ip, dst_ip)
        p_pkt, p_dur = self.prev.get(key, (0, 0.0))
        d_pkt = pkt_count - p_pkt
        d_dur = duration - p_dur
        self.prev[key] = (pkt_count, duration)
        if d_pkt <= 0 or d_dur <= 0:
            return                      # idle or counter reset: no decision
        pps = d_pkt / d_dur

        self.stats["total"] += 1
        if pps >= PPS_THRESHOLD:
            self.hits[key] += 1
        else:
            self.hits[key] = 0
            self.stats["allowed"] += 1
            return

        if self.hits[key] < CONSECUTIVE:
            self.logger.info(f"[WATCH] {src_ip} -> {dst_ip} | {pps:.0f} pkt/s | "
                             f"{self.hits[key]}/{CONSECUTIVE}")
            return

        self._alert(dp, src_ip, dst_ip, pps, "RATE", "BLOCKED")
        if src_ip and dst_ip:
            self._block_flow(dp, src_ip, dst_ip)
        self.stats["blocked"] += 1

    def _alert(self, dp, src_ip, dst_ip, confidence, attack_type, action):
        self.alert_count += 1
        ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        src = src_ip if src_ip else "unknown"
        dst = dst_ip if dst_ip else "unknown"
        msg = (f"[ALERT #{self.alert_count}] {ts} | "
               f"dpid={dp.id} | {src} → {dst} | "
               f"rate={confidence:.0f} pkt/s | {attack_type} | {action}")
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
