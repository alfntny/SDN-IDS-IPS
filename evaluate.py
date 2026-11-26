"""
Step 3: Evaluation — Graphs & Charts
SDN IDS/IPS Project — SRM SIST, Batch B1025

Generates:
1. Accuracy comparison bar chart (all models)
2. Training time comparison
3. Confusion matrix (LSTM)
4. ROC curves (top models)
5. F1/Precision/Recall grouped bar chart

Run:
    python3 evaluate.py
"""

import numpy as np
import json
import pickle
import os
import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    ConfusionMatrixDisplay
)

import torch
import torch.nn as nn

DATA_DIR   = "data/processed"
MODELS_DIR = "models"
OUT_DIR    = "results"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Colors ────────────────────────────────────────────────────────────────────
PRIMARY   = "#1a3a5c"
ACCENT    = "#e74c3c"
HIGHLIGHT = "#2ecc71"
COLORS    = ["#1a3a5c","#2980b9","#27ae60","#e74c3c",
             "#f39c12","#8e44ad","#16a085","#d35400","#2c3e50","#c0392b","#7f8c8d"]

# ── LSTM model def ────────────────────────────────────────────────────────────
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


def load_results():
    with open(f"{MODELS_DIR}/all_results.json") as f:
        return json.load(f)


def load_test_data():
    X_test  = np.load(f"{DATA_DIR}/X_test.npy")
    y_test  = np.load(f"{DATA_DIR}/y_bin_test.npy")
    return X_test, y_test


# ── Chart 1: Accuracy Comparison ──────────────────────────────────────────────
def plot_accuracy(results):
    results_sorted = sorted(results, key=lambda x: x["accuracy"], reverse=True)
    names  = [r["model"] for r in results_sorted]
    accs   = [r["accuracy"] * 100 for r in results_sorted]

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(names, accs, color=COLORS[:len(names)], edgecolor="white",
                  linewidth=0.8, zorder=3)

    # Value labels
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{acc:.3f}%", ha="center", va="bottom",
                fontsize=8, fontweight="bold", color="#2c3e50")

    ax.set_ylim(99.7, 100.05)
    ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title("Model Accuracy Comparison — SDN IDS/IPS\n(InSDN Dataset, 343,889 flows)",
                 fontsize=13, fontweight="bold", color=PRIMARY, pad=15)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/accuracy_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [✓] Saved: {OUT_DIR}/accuracy_comparison.png")


# ── Chart 2: Training Time ─────────────────────────────────────────────────────
def plot_train_time(results):
    results_filtered = [r for r in results if r["train_time"] > 0]
    results_sorted   = sorted(results_filtered, key=lambda x: x["train_time"], reverse=True)
    names      = [r["model"] for r in results_sorted]
    times      = [r["train_time"] for r in results_sorted]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.barh(names, times, color=COLORS[:len(names)],
                   edgecolor="white", linewidth=0.8)

    for bar, t in zip(bars, times):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{t:.1f}s", va="center", fontsize=9, fontweight="bold")

    ax.set_xlabel("Training Time (seconds)", fontsize=12, fontweight="bold")
    ax.set_title("Training Time Comparison\n(Slower models trained on 20k sample subset)",
                 fontsize=13, fontweight="bold", color=PRIMARY, pad=15)
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/training_time.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [✓] Saved: {OUT_DIR}/training_time.png")


# ── Chart 3: F1/Precision/Recall Grouped Bar ──────────────────────────────────
def plot_metrics(results):
    results_sorted = sorted(results, key=lambda x: x["f1"], reverse=True)
    names  = [r["model"] for r in results_sorted]
    f1s    = [r["f1"] for r in results_sorted]
    precs  = [r["precision"] for r in results_sorted]
    recs   = [r["recall"] for r in results_sorted]

    x     = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width, precs, width, label="Precision", color="#2980b9", alpha=0.85)
    ax.bar(x,         f1s,   width, label="F1 Score",  color="#27ae60", alpha=0.85)
    ax.bar(x + width, recs,  width, label="Recall",    color="#e74c3c", alpha=0.85)

    ax.set_ylim(0.996, 1.001)
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title("Precision / F1 / Recall Comparison — All Models",
                 fontsize=13, fontweight="bold", color=PRIMARY, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/metrics_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [✓] Saved: {OUT_DIR}/metrics_comparison.png")


# ── Chart 4: Confusion Matrix (Random Forest) ─────────────────────────────────
def plot_confusion_matrix(X_test, y_test):
    print("  [+] Loading Random Forest for confusion matrix ...")
    with open(f"{MODELS_DIR}/random_forest.pkl", "rb") as f:
        rf = pickle.load(f)

    y_pred = rf.predict(X_test)
    cm     = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal", "Attack"]
    )
    disp.plot(ax=ax, colorbar=False, cmap="Blues")

    # Add counts + percentages
    total = cm.sum()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pct = cm[i, j] / total * 100
            ax.text(j, i + 0.3, f"({pct:.2f}%)",
                    ha="center", va="center",
                    fontsize=9, color="white" if cm[i,j] > cm.max()/2 else "black")

    ax.set_title("Confusion Matrix — Random Forest\n(Best Performing Model)",
                 fontsize=12, fontweight="bold", color=PRIMARY, pad=15)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/confusion_matrix_rf.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [✓] Saved: {OUT_DIR}/confusion_matrix_rf.png")


# ── Chart 5: LSTM Confusion Matrix ────────────────────────────────────────────
def plot_lstm_confusion(X_test, y_test):
    print("  [+] Loading LSTM for confusion matrix ...")
    ckpt = torch.load(f"{MODELS_DIR}/lstm_detector.pt",
                      map_location="cpu", weights_only=False)
    model = LSTMDetector(
        n_features=ckpt["n_features"],
        hidden_size=ckpt["hidden_size"],
        num_layers=ckpt["num_layers"],
        dropout=ckpt["dropout"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    seq_len = ckpt["seq_len"]
    n = len(X_test) - seq_len + 1
    preds, labels = [], []

    with torch.no_grad():
        for i in range(0, n, 512):
            batch_x = np.array([X_test[j:j+seq_len] for j in range(i, min(i+512, n))],
                                dtype=np.float32)
            batch_y = y_test[i+seq_len-1:i+seq_len-1+len(batch_x)]
            x = torch.tensor(batch_x)
            p = model(x).argmax(dim=1).numpy()
            preds.extend(p)
            labels.extend(batch_y)

    cm   = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal", "Attack"]
    )
    disp.plot(ax=ax, colorbar=False, cmap="Greens")

    total = cm.sum()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pct = cm[i, j] / total * 100
            ax.text(j, i + 0.3, f"({pct:.2f}%)",
                    ha="center", va="center",
                    fontsize=9, color="white" if cm[i,j] > cm.max()/2 else "black")

    ax.set_title("Confusion Matrix — Bidirectional LSTM\n(Real-Time SDN Detection Model)",
                 fontsize=12, fontweight="bold", color="#1a5c3a", pad=15)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/confusion_matrix_lstm.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [✓] Saved: {OUT_DIR}/confusion_matrix_lstm.png")


# ── Chart 6: ROC Curve (top models) ───────────────────────────────────────────
def plot_roc(X_test, y_test):
    print("  [+] Generating ROC curves ...")
    fig, ax = plt.subplots(figsize=(9, 7))

    # Random Forest
    with open(f"{MODELS_DIR}/random_forest.pkl", "rb") as f:
        rf = pickle.load(f)
    fpr, tpr, _ = roc_curve(y_test, rf.predict_proba(X_test)[:, 1])
    ax.plot(fpr, tpr, color="#1a3a5c", lw=2,
            label=f"Random Forest (AUC = {auc(fpr,tpr):.4f})")

    # Extra Trees
    with open(f"{MODELS_DIR}/extra_trees.pkl", "rb") as f:
        et = pickle.load(f)
    fpr, tpr, _ = roc_curve(y_test, et.predict_proba(X_test)[:, 1])
    ax.plot(fpr, tpr, color="#27ae60", lw=2,
            label=f"Extra Trees (AUC = {auc(fpr,tpr):.4f})")

    # SVM
    with open(f"{MODELS_DIR}/svm.pkl", "rb") as f:
        svm = pickle.load(f)
    fpr, tpr, _ = roc_curve(y_test, svm.predict_proba(X_test)[:, 1])
    ax.plot(fpr, tpr, color="#e74c3c", lw=2,
            label=f"SVM (AUC = {auc(fpr,tpr):.4f})")

    # Gradient Boosting
    with open(f"{MODELS_DIR}/gradient_boosting.pkl", "rb") as f:
        gb = pickle.load(f)
    fpr, tpr, _ = roc_curve(y_test, gb.predict_proba(X_test)[:, 1])
    ax.plot(fpr, tpr, color="#f39c12", lw=2,
            label=f"Gradient Boosting (AUC = {auc(fpr,tpr):.4f})")

    # Diagonal
    ax.plot([0,1], [0,1], "k--", lw=1, alpha=0.5, label="Random Classifier")

    ax.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=12, fontweight="bold")
    ax.set_title("ROC Curves — Top Models\nSDN IDS/IPS (InSDN Dataset)",
                 fontsize=13, fontweight="bold", color=PRIMARY, pad=15)
    ax.legend(fontsize=10, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [✓] Saved: {OUT_DIR}/roc_curves.png")


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 52)
    print("  Evaluation & Graph Generation")
    print("=" * 52)

    results      = load_results()
    X_test, y_test = load_test_data()

    print("\n[1/6] Accuracy comparison chart ...")
    plot_accuracy(results)

    print("\n[2/6] Training time chart ...")
    plot_train_time(results)

    print("\n[3/6] Metrics comparison chart ...")
    plot_metrics(results)

    print("\n[4/6] Random Forest confusion matrix ...")
    plot_confusion_matrix(X_test, y_test)

    print("\n[5/6] LSTM confusion matrix ...")
    plot_lstm_confusion(X_test, y_test)

    print("\n[6/6] ROC curves ...")
    plot_roc(X_test, y_test)

    print(f"\n{'='*52}")
    print(f"  All graphs saved to: {OUT_DIR}/")
    print(f"{'='*52}")
    print("\n  Files generated:")
    for f in os.listdir(OUT_DIR):
        print(f"    - {f}")


if __name__ == "__main__":
    main()
