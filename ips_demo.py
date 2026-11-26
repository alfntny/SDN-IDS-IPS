"""
Real-Time IPS Demo — Live Terminal Dashboard
SDN IDS/IPS Project — SRM SIST, Batch B1025
Alfin Tony & Harshaditya Choudhary

Simulates live SDN flow traffic through the trained LSTM model.
Shows real-time detection + automated IPS actions in terminal.

Run:
    python3 ips_demo.py
"""

import numpy as np
import pandas as pd
import pickle
import json
import time
import random
import threading
import os
import sys
from datetime import datetime
from collections import deque

import torch
import torch.nn as nn

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MODELS_DIR   = "models"
DATA_DIR     = "data/processed"
BLOCK_THRESHOLD     = 0.85
RATELIMIT_THRESHOLD = 0.50
FLOW_INTERVAL       = 0.3   # seconds between flows (lower = faster demo)
WINDOW_SIZE         = 10

# ─── COLORS ───────────────────────────────────────────────────────────────────
class C:
    RED    = "\033[91m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    BLUE   = "\033[94m"
    CYAN   = "\033[96m"
    WHITE  = "\033[97m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    RESET  = "\033[0m"

# ─── LSTM MODEL ───────────────────────────────────────────────────────────────
class LSTMDetector(nn.Module):
    def __init__(self, n_features, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.classifier(out)


# ─── LOAD MODELS & DATA ───────────────────────────────────────────────────────
def load_everything():
    # Scaler
    with open(f"{DATA_DIR}/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Label encoder
    with open(f"{DATA_DIR}/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    # Metadata
    with open(f"{DATA_DIR}/metadata.json") as f:
        meta = json.load(f)

    # LSTM
    ckpt = torch.load(f"{MODELS_DIR}/lstm_detector.pt", map_location="cpu")
    model = LSTMDetector(
        n_features=ckpt["n_features"],
        hidden_size=ckpt["hidden_size"],
        num_layers=ckpt["num_layers"],
        dropout=ckpt["dropout"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Load test data to sample real flows from
    X_test     = np.load(f"{DATA_DIR}/X_test.npy")
    y_bin_test = np.load(f"{DATA_DIR}/y_bin_test.npy")
    y_mul_test = np.load(f"{DATA_DIR}/y_multi_test.npy")

    return model, scaler, le, meta, X_test, y_bin_test, y_mul_test


# ─── IPS LOGIC ────────────────────────────────────────────────────────────────
def ips_action(confidence: float) -> tuple:
    """Returns (action_label, color)"""
    if confidence >= BLOCK_THRESHOLD:
        return "BLOCKED    ", C.RED
    elif confidence >= RATELIMIT_THRESHOLD:
        return "RATE-LIMIT ", C.YELLOW
    else:
        return "ALLOWED    ", C.GREEN


def fake_ip():
    return f"{random.randint(10,192)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"


# ─── DASHBOARD ────────────────────────────────────────────────────────────────
def clear():
    os.system("clear")


def print_header():
    print(f"{C.BOLD}{C.CYAN}")
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║       SDN INTRUSION DETECTION & PREVENTION SYSTEM (IDS/IPS)         ║")
    print("║         SRM SIST | Batch B1025 | 21CSP302L Minor Project            ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print(f"{C.RESET}")


def print_stats(stats: dict):
    total = stats["total"]
    blocked = stats["blocked"]
    ratelimited = stats["ratelimited"]
    allowed = stats["allowed"]
    alerts = stats["alerts"]

    bar_len = 30
    attack_ratio = (blocked + ratelimited) / max(total, 1)
    filled = int(bar_len * attack_ratio)
    bar = f"{C.RED}{'█' * filled}{C.DIM}{'░' * (bar_len - filled)}{C.RESET}"

    print(f"{C.BOLD}  Live Statistics{C.RESET}")
    print(f"  {'─'*66}")
    print(f"  Flows Analyzed : {C.WHITE}{C.BOLD}{total:>6}{C.RESET}    "
          f"Attack Rate: [{bar}{C.RESET}] {attack_ratio:.1%}")
    print(f"  {C.RED}⛔ Blocked      : {blocked:>6}{C.RESET}    "
          f"{C.YELLOW}⚠  Rate-Limited : {ratelimited:>6}{C.RESET}    "
          f"{C.GREEN}✓  Allowed      : {allowed:>6}{C.RESET}")
    print(f"  {C.BOLD}🚨 Total Alerts : {alerts:>6}{C.RESET}")
    print(f"  {'─'*66}")


def print_flow_log(log: deque):
    print(f"\n{C.BOLD}  Live Flow Monitor{C.RESET}")
    print(f"  {'─'*66}")
    print(f"  {'Time':<10} {'Src IP':<18} {'Dst IP':<18} {'Confidence':>10}  {'Action':<12} {'Label'}")
    print(f"  {'─'*66}")
    for entry in log:
        ts, src, dst, conf, action, action_color, label = entry
        print(f"  {C.DIM}{ts}{C.RESET}  "
              f"{C.CYAN}{src:<18}{C.RESET}"
              f"{C.CYAN}{dst:<18}{C.RESET}"
              f"{conf:>10.3f}  "
              f"{action_color}{C.BOLD}{action}{C.RESET}  "
              f"{C.DIM}{label}{C.RESET}")
    print(f"  {'─'*66}")


# ─── MAIN LOOP ────────────────────────────────────────────────────────────────
def run():
    print(f"{C.CYAN}Loading models...{C.RESET}")
    model, scaler, le, meta, X_test, y_bin_test, y_mul_test = load_everything()
    n_features = meta["n_features"]
    class_names = meta["class_names"]

    stats = {
        "total": 0, "blocked": 0,
        "ratelimited": 0, "allowed": 0, "alerts": 0
    }

    flow_log   = deque(maxlen=20)   # last 20 flows shown
    window_buf = deque(maxlen=WINDOW_SIZE)

    n_samples = len(X_test)

    print(f"{C.GREEN}✓ Models loaded. Starting live demo...{C.RESET}")
    time.sleep(1)

    try:
        while True:
            # Pick a real flow from test set
            idx = random.randint(0, n_samples - 1)
            flow = X_test[idx]
            true_label_idx = y_mul_test[idx]
            true_label = class_names[true_label_idx]

            # Add to sliding window
            window_buf.append(flow)

            if len(window_buf) < WINDOW_SIZE:
                time.sleep(FLOW_INTERVAL)
                continue

            # Run LSTM inference
            x = torch.tensor([list(window_buf)], dtype=torch.float32)
            with torch.no_grad():
                logits = model(x)
                probs  = torch.softmax(logits, dim=1)[0]
            attack_conf = probs[1].item()

            # IPS decision
            action, action_color = ips_action(attack_conf)

            # Generate fake IPs for display
            src = fake_ip()
            dst = fake_ip()
            ts  = datetime.now().strftime("%H:%M:%S")

            # Update stats
            stats["total"] += 1
            if "BLOCKED" in action:
                stats["blocked"] += 1
                stats["alerts"]  += 1
            elif "RATE" in action:
                stats["ratelimited"] += 1
                stats["alerts"]      += 1
            else:
                stats["allowed"] += 1

            # Add to log
            flow_log.append((ts, src, dst, attack_conf, action, action_color, true_label))

            # Redraw dashboard
            clear()
            print_header()
            print_stats(stats)
            print_flow_log(flow_log)

            print(f"\n  {C.DIM}Press Ctrl+C to stop{C.RESET}")

            time.sleep(FLOW_INTERVAL)

    except KeyboardInterrupt:
        clear()
        print_header()
        print(f"\n{C.BOLD}{C.GREEN}  Demo stopped.{C.RESET}\n")
        print_stats(stats)
        print(f"\n  {C.BOLD}Final Summary:{C.RESET}")
        print(f"  Total flows processed : {stats['total']}")
        print(f"  Attacks detected      : {stats['alerts']}")
        print(f"  Detection rate        : {stats['alerts']/max(stats['total'],1):.1%}\n")


if __name__ == "__main__":
    run()
