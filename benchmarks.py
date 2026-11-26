"""
Step 5: Performance Benchmarks
SDN IDS/IPS Project — SRM SIST, Batch B1025

Measures:
1. LSTM inference latency (per flow, per batch)
2. Throughput (flows per second)
3. CPU and memory usage during inference
4. Detection latency (time from flow to alert)
5. Model size comparison

Run:
    python3 benchmarks.py
"""

import numpy as np
import pickle
import json
import time
import os
import psutil
import threading
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

DATA_DIR   = "data/processed"
MODELS_DIR = "models"
OUT_DIR    = "results"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Models ────────────────────────────────────────────────────────────────────
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


# ── Load everything ───────────────────────────────────────────────────────────
def load_all():
    print("[+] Loading data and models ...")
    X_test = np.load(f"{DATA_DIR}/X_test.npy")
    y_test = np.load(f"{DATA_DIR}/y_bin_test.npy")

    ckpt = torch.load(f"{MODELS_DIR}/lstm_detector.pt",
                      map_location="cpu", weights_only=False)
    lstm = LSTMDetector(
        n_features=ckpt["n_features"],
        hidden_size=ckpt["hidden_size"],
        num_layers=ckpt["num_layers"],
        dropout=ckpt["dropout"],
    )
    lstm.load_state_dict(ckpt["model_state"])
    lstm.eval().to(DEVICE)

    with open(f"{MODELS_DIR}/random_forest.pkl", "rb") as f:
        rf = pickle.load(f)
    with open(f"{MODELS_DIR}/extra_trees.pkl", "rb") as f:
        et = pickle.load(f)
    with open(f"{MODELS_DIR}/svm.pkl", "rb") as f:
        svm = pickle.load(f)

    print(f"    Test samples: {len(X_test)}")
    print(f"    Device: {DEVICE}")
    return X_test, y_test, lstm, rf, et, svm, ckpt["n_features"]


# ── 1. LSTM Inference Latency ─────────────────────────────────────────────────
def benchmark_lstm_latency(lstm, X_test, n_features, seq_len=10):
    print("\n[1/5] LSTM Inference Latency ...")

    # Build one sequence
    seq = X_test[:seq_len]
    x   = torch.tensor(seq[np.newaxis], dtype=torch.float32).to(DEVICE)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            lstm(x)

    # Single flow latency
    N = 1000
    times = []
    for _ in range(N):
        start = time.perf_counter()
        with torch.no_grad():
            lstm(x)
        times.append((time.perf_counter() - start) * 1000)  # ms

    single_mean = np.mean(times)
    single_std  = np.std(times)
    single_p99  = np.percentile(times, 99)

    print(f"    Single flow latency:")
    print(f"      Mean : {single_mean:.3f} ms")
    print(f"      Std  : {single_std:.3f} ms")
    print(f"      P99  : {single_p99:.3f} ms")

    # Batch latency
    batch_results = {}
    for batch_size in [1, 8, 32, 64, 128, 256]:
        n_seqs = min(batch_size, len(X_test) - seq_len)
        seqs   = np.array([X_test[i:i+seq_len] for i in range(n_seqs)],
                           dtype=np.float32)
        xb     = torch.tensor(seqs).to(DEVICE)

        t_list = []
        for _ in range(100):
            start = time.perf_counter()
            with torch.no_grad():
                lstm(xb)
            t_list.append((time.perf_counter() - start) * 1000)

        batch_results[batch_size] = {
            "mean": np.mean(t_list),
            "per_sample": np.mean(t_list) / batch_size
        }
        print(f"    Batch={batch_size:>3}: {np.mean(t_list):.2f}ms total | "
              f"{np.mean(t_list)/batch_size:.3f}ms/sample")

    return {
        "single_mean_ms": single_mean,
        "single_std_ms": single_std,
        "single_p99_ms": single_p99,
        "batch_results": batch_results
    }, times


# ── 2. Throughput ─────────────────────────────────────────────────────────────
def benchmark_throughput(lstm, rf, et, svm, X_test, seq_len=10):
    print("\n[2/5] Throughput (flows/second) ...")
    results = {}

    # LSTM throughput
    n = min(10000, len(X_test) - seq_len)
    seqs = np.array([X_test[i:i+seq_len] for i in range(n)], dtype=np.float32)
    xb   = torch.tensor(seqs).to(DEVICE)

    start = time.perf_counter()
    with torch.no_grad():
        for i in range(0, n, 256):
            lstm(xb[i:i+256])
    elapsed = time.perf_counter() - start
    lstm_tput = n / elapsed
    results["LSTM"] = lstm_tput
    print(f"    LSTM         : {lstm_tput:>10,.0f} flows/sec")

    # RF throughput
    X_sub = X_test[:n]
    start = time.perf_counter()
    rf.predict(X_sub)
    elapsed = time.perf_counter() - start
    rf_tput = n / elapsed
    results["Random Forest"] = rf_tput
    print(f"    Random Forest: {rf_tput:>10,.0f} flows/sec")

    # ET throughput
    start = time.perf_counter()
    et.predict(X_sub)
    elapsed = time.perf_counter() - start
    et_tput = n / elapsed
    results["Extra Trees"] = et_tput
    print(f"    Extra Trees  : {et_tput:>10,.0f} flows/sec")

    # SVM throughput (small subset)
    n_svm = min(2000, n)
    start = time.perf_counter()
    svm.predict(X_test[:n_svm])
    elapsed = time.perf_counter() - start
    svm_tput = n_svm / elapsed
    results["SVM"] = svm_tput
    print(f"    SVM          : {svm_tput:>10,.0f} flows/sec")

    return results


# ── 3. CPU & Memory Usage ─────────────────────────────────────────────────────
def benchmark_cpu_memory(lstm, X_test, seq_len=10):
    print("\n[3/5] CPU & Memory Usage ...")
    process = psutil.Process(os.getpid())

    n = min(5000, len(X_test) - seq_len)
    seqs = np.array([X_test[i:i+seq_len] for i in range(n)], dtype=np.float32)
    xb   = torch.tensor(seqs).to(DEVICE)

    # Baseline
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    cpu_before = psutil.cpu_percent(interval=0.1)

    # Run inference and measure
    cpu_samples = []
    def measure_cpu():
        for _ in range(20):
            cpu_samples.append(psutil.cpu_percent(interval=0.1))

    t = threading.Thread(target=measure_cpu)
    t.start()

    start = time.perf_counter()
    with torch.no_grad():
        for i in range(0, n, 256):
            lstm(xb[i:i+256])
    elapsed = time.perf_counter() - start

    t.join()
    mem_after = process.memory_info().rss / 1024 / 1024

    cpu_during = np.mean(cpu_samples) if cpu_samples else 0
    mem_used   = mem_after - mem_before

    print(f"    Inference time  : {elapsed:.3f}s for {n} flows")
    print(f"    CPU usage       : {cpu_during:.1f}%")
    print(f"    Memory used     : {mem_used:.1f} MB")
    print(f"    Memory baseline : {mem_before:.1f} MB")

    return {
        "inference_time_s": elapsed,
        "cpu_percent": cpu_during,
        "memory_used_mb": mem_used,
        "memory_baseline_mb": mem_before,
        "n_flows": n
    }


# ── 4. Model Size ─────────────────────────────────────────────────────────────
def benchmark_model_size():
    print("\n[4/5] Model Sizes ...")
    sizes = {}
    files = {
        "LSTM (binary)":     f"{MODELS_DIR}/lstm_detector.pt",
        "LSTM (multiclass)": f"{MODELS_DIR}/lstm_multiclass.pt",
        "Random Forest":     f"{MODELS_DIR}/random_forest.pkl",
        "Extra Trees":       f"{MODELS_DIR}/extra_trees.pkl",
        "SVM":               f"{MODELS_DIR}/svm.pkl",
        "Gradient Boosting": f"{MODELS_DIR}/gradient_boosting.pkl",
    }
    for name, path in files.items():
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1024 / 1024
            sizes[name] = size_mb
            print(f"    {name:<22}: {size_mb:.2f} MB")

    return sizes


# ── 5. Detection Delay ────────────────────────────────────────────────────────
def benchmark_detection_delay(lstm, X_test, seq_len=10):
    print("\n[5/5] End-to-End Detection Delay ...")

    delays = []
    for _ in range(500):
        # Simulate: receive flow stat → extract features → normalize → inference
        idx  = np.random.randint(0, len(X_test) - seq_len)
        seq  = X_test[idx:idx+seq_len]

        start = time.perf_counter()
        # Feature normalization (already done, simulate with array copy)
        x = torch.tensor(seq[np.newaxis], dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            logits = lstm(x)
            probs  = torch.softmax(logits, dim=1)[0]
            _      = probs[1].item()  # attack probability
        delay = (time.perf_counter() - start) * 1000
        delays.append(delay)

    mean_delay = np.mean(delays)
    p50        = np.percentile(delays, 50)
    p95        = np.percentile(delays, 95)
    p99        = np.percentile(delays, 99)

    print(f"    Mean delay : {mean_delay:.3f} ms")
    print(f"    P50        : {p50:.3f} ms")
    print(f"    P95        : {p95:.3f} ms")
    print(f"    P99        : {p99:.3f} ms")

    return {"mean_ms": mean_delay, "p50_ms": p50,
            "p95_ms": p95, "p99_ms": p99}, delays


# ── Charts ────────────────────────────────────────────────────────────────────
def plot_latency_dist(times, delays):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.hist(times, bins=50, color="#1a3a5c", alpha=0.8, edgecolor="white")
    ax1.set_xlabel("Latency (ms)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Frequency", fontsize=11, fontweight="bold")
    ax1.set_title("LSTM Single-Flow Inference Latency Distribution",
                  fontsize=12, fontweight="bold")
    ax1.axvline(np.mean(times), color="#e74c3c", linestyle="--",
                label=f"Mean: {np.mean(times):.2f}ms")
    ax1.legend()
    ax1.spines[["top", "right"]].set_visible(False)

    ax2.hist(delays, bins=50, color="#27ae60", alpha=0.8, edgecolor="white")
    ax2.set_xlabel("Detection Delay (ms)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Frequency", fontsize=11, fontweight="bold")
    ax2.set_title("End-to-End Detection Delay Distribution",
                  fontsize=12, fontweight="bold")
    ax2.axvline(np.mean(delays), color="#e74c3c", linestyle="--",
                label=f"Mean: {np.mean(delays):.2f}ms")
    ax2.legend()
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/latency_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [✓] Saved: {OUT_DIR}/latency_distribution.png")


def plot_throughput(throughput):
    names  = list(throughput.keys())
    values = [throughput[n] for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#1a3a5c", "#2980b9", "#27ae60", "#e74c3c"]
    bars   = ax.bar(names, values, color=colors[:len(names)],
                    edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                f"{val:,.0f}", ha="center", fontsize=10, fontweight="bold")

    ax.set_ylabel("Flows per Second", fontsize=12, fontweight="bold")
    ax.set_title("Model Throughput Comparison\n(Higher is better)",
                 fontsize=13, fontweight="bold", pad=15)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/throughput_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [✓] Saved: {OUT_DIR}/throughput_comparison.png")


def plot_model_sizes(sizes):
    names  = list(sizes.keys())
    values = list(sizes.values())

    fig, ax = plt.subplots(figsize=(10, 5))
    colors  = ["#1a3a5c", "#2980b9", "#27ae60", "#e74c3c", "#f39c12", "#8e44ad"]
    bars    = ax.barh(names, values, color=colors[:len(names)],
                      edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                f"{val:.1f} MB", va="center", fontsize=9, fontweight="bold")

    ax.set_xlabel("Model Size (MB)", fontsize=12, fontweight="bold")
    ax.set_title("Model Size Comparison\n(Smaller = more deployment friendly)",
                 fontsize=13, fontweight="bold", pad=15)
    ax.xaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/model_sizes.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [✓] Saved: {OUT_DIR}/model_sizes.png")


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Performance Benchmarks — SDN IDS/IPS")
    print("=" * 55)

    X_test, y_test, lstm, rf, et, svm, n_features = load_all()

    latency_results, latency_times = benchmark_lstm_latency(lstm, X_test, n_features)
    throughput = benchmark_throughput(lstm, rf, et, svm, X_test)
    cpu_mem    = benchmark_cpu_memory(lstm, X_test)
    sizes      = benchmark_model_size()
    delay_results, delay_times = benchmark_detection_delay(lstm, X_test)

    # Charts
    print("\n[+] Generating benchmark charts ...")
    plot_latency_dist(latency_times, delay_times)
    plot_throughput(throughput)
    plot_model_sizes(sizes)

    # Final summary
    print(f"\n{'='*55}")
    print(f"  BENCHMARK SUMMARY")
    print(f"{'='*55}")
    print(f"  LSTM Inference Latency  : {latency_results['single_mean_ms']:.3f} ms (mean)")
    print(f"  LSTM P99 Latency        : {latency_results['single_p99_ms']:.3f} ms")
    print(f"  Detection Delay (E2E)   : {delay_results['mean_ms']:.3f} ms")
    print(f"  LSTM Throughput         : {throughput['LSTM']:,.0f} flows/sec")
    print(f"  RF Throughput           : {throughput['Random Forest']:,.0f} flows/sec")
    print(f"  CPU Usage (inference)   : {cpu_mem['cpu_percent']:.1f}%")
    print(f"  Memory Used             : {cpu_mem['memory_used_mb']:.1f} MB")
    print(f"{'='*55}")

    # Save results
    all_results = {
        "latency": latency_results,
        "throughput": throughput,
        "cpu_memory": cpu_mem,
        "model_sizes": sizes,
        "detection_delay": delay_results,
    }
    with open(f"{MODELS_DIR}/benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[✓] All benchmark results saved → {MODELS_DIR}/benchmark_results.json")
    print(f"[✓] Charts saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
