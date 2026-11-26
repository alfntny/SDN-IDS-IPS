"""
Step 4: Multiclass Attack Detection
SDN IDS/IPS Project — SRM SIST, Batch B1025

Trains models to identify specific attack types:
BFA, BOTNET, DDoS, DoS, Normal, Probe, U2R, Web-Attack

Run:
    python3 multiclass.py
"""

import numpy as np
import json
import pickle
import os
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

DATA_DIR   = "data/processed"
MODELS_DIR = "models"
OUT_DIR    = "results"
RANDOM_STATE = 42
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# ── LSTM Model ────────────────────────────────────────────────────────────────
class LSTMMulticlass(nn.Module):
    def __init__(self, n_features, n_classes, hidden_size=128, num_layers=2, dropout=0.3):
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
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.classifier(self.dropout(out))


def load_data():
    print("[+] Loading preprocessed data ...")
    X_train = np.load(f"{DATA_DIR}/X_train.npy")
    X_test  = np.load(f"{DATA_DIR}/X_test.npy")
    y_train = np.load(f"{DATA_DIR}/y_multi_train.npy")
    y_test  = np.load(f"{DATA_DIR}/y_multi_test.npy")

    with open(f"{DATA_DIR}/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    with open(f"{DATA_DIR}/metadata.json") as f:
        meta = json.load(f)

    class_names = meta["class_names"]
    print(f"    Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"    Classes ({len(class_names)}): {class_names}")
    return X_train, X_test, y_train, y_test, class_names, le


# ── 1. Random Forest Multiclass ───────────────────────────────────────────────
def train_rf_multiclass(X_train, y_train, X_test, y_test, class_names):
    print("\n[+] Training Random Forest (multiclass) ...")
    start = time.time()
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=20,
        n_jobs=-1, random_state=RANDOM_STATE
    )
    rf.fit(X_train, y_train)
    train_time = time.time() - start
    print(f"    Done in {train_time:.1f}s")

    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted")

    print(f"\n{'='*55}")
    print(f"  Random Forest — Multiclass Results")
    print(f"{'='*55}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"\n  Per-Class Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Save model
    with open(f"{MODELS_DIR}/rf_multiclass.pkl", "wb") as f:
        pickle.dump(rf, f)

    # Confusion matrix
    _plot_multiclass_cm(y_test, y_pred, class_names, "Random Forest", "Blues")

    return rf, acc, f1, y_pred


# ── 2. Extra Trees Multiclass ─────────────────────────────────────────────────
def train_et_multiclass(X_train, y_train, X_test, y_test, class_names):
    print("\n[+] Training Extra Trees (multiclass) ...")
    start = time.time()
    et = ExtraTreesClassifier(
        n_estimators=200, max_depth=20,
        n_jobs=-1, random_state=RANDOM_STATE
    )
    et.fit(X_train, y_train)
    train_time = time.time() - start
    print(f"    Done in {train_time:.1f}s")

    y_pred = et.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted")

    print(f"\n{'='*55}")
    print(f"  Extra Trees — Multiclass Results")
    print(f"{'='*55}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"\n  Per-Class Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    with open(f"{MODELS_DIR}/et_multiclass.pkl", "wb") as f:
        pickle.dump(et, f)

    _plot_multiclass_cm(y_test, y_pred, class_names, "Extra Trees", "Greens")

    return et, acc, f1, y_pred


# ── 3. LSTM Multiclass ────────────────────────────────────────────────────────
def train_lstm_multiclass(X_train, y_train, X_test, y_test, class_names):
    print("\n[+] Training LSTM (multiclass) ...")
    torch.manual_seed(RANDOM_STATE); np.random.seed(RANDOM_STATE)
    DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEQ_LEN    = 10
    BATCH_SIZE = 256
    EPOCHS     = 20
    PATIENCE   = 5
    n_features = X_train.shape[1]
    n_classes  = len(class_names)

    # Build sequences
    def make_seq(X, y):
        Xs, ys = [], []
        for i in range(len(X) - SEQ_LEN + 1):
            Xs.append(X[i:i+SEQ_LEN])
            ys.append(y[i+SEQ_LEN-1])
        return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.int64)

    print("    Building sequences ...")
    # Hold out 10% of TRAIN for early stopping; test is used only for final evaluation.
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, stratify=y_train, random_state=RANDOM_STATE
    )
    X_tr_seq,  y_tr_seq  = make_seq(X_tr,  y_tr)
    X_val_seq, y_val_seq = make_seq(X_val, y_val)
    X_te_seq,  y_te_seq  = make_seq(X_test, y_test)
    print(f"    Train {X_tr_seq.shape} | Val {X_val_seq.shape} | Test {X_te_seq.shape}")

    mk = lambda X, y, sh: DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
        batch_size=BATCH_SIZE, shuffle=sh)
    train_loader = mk(X_tr_seq,  y_tr_seq,  True)
    val_loader   = mk(X_val_seq, y_val_seq, False)
    test_loader  = mk(X_te_seq,  y_te_seq,  False)

    model     = LSTMMulticlass(n_features, n_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=2, factor=0.5
    )

    print(f"    Device: {DEVICE} | Parameters: {sum(p.numel() for p in model.parameters()):,}")

    best_f1, patience_cnt, best_state = 0.0, 0, None

    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(yb)
            correct    += (logits.argmax(1) == yb).sum().item()
            total      += len(yb)

        # Validate on the held-out VALIDATION split, not the test split
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                p = model(xb.to(DEVICE)).argmax(1).cpu().numpy()
                preds.extend(p)
                labels.extend(yb.numpy())

        val_f1 = f1_score(labels, preds, average="weighted")
        scheduler.step(val_f1)
        print(f"  Epoch {epoch:02d}/{EPOCHS}  "
              f"loss={total_loss/total:.4f}  "
              f"acc={correct/total:.4f}  "
              f"val_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    model.eval()

    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            p = model(xb.to(DEVICE)).argmax(1).cpu().numpy()
            preds.extend(p)
            labels.extend(yb.numpy())

    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="weighted")

    print(f"\n{'='*55}")
    print(f"  LSTM — Multiclass Results")
    print(f"{'='*55}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"\n  Per-Class Report:")
    print(classification_report(labels, preds, target_names=class_names))

    torch.save({
        "model_state": best_state,
        "n_features": n_features,
        "n_classes": n_classes,
        "seq_len": SEQ_LEN,
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.3,
        "class_names": class_names,
    }, f"{MODELS_DIR}/lstm_multiclass.pt")

    _plot_multiclass_cm(labels, preds, class_names, "LSTM", "Oranges")

    return model, acc, f1, preds


# ── Confusion Matrix Helper ───────────────────────────────────────────────────
def _plot_multiclass_cm(y_true, y_pred, class_names, model_name, cmap):
    cm   = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=False, cmap=cmap, xticks_rotation=45)
    ax.set_title(f"Multiclass Confusion Matrix — {model_name}\n"
                 f"(8 Attack Types + Normal)",
                 fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()
    fname = model_name.lower().replace(" ", "_")
    plt.savefig(f"{OUT_DIR}/multiclass_cm_{fname}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    [✓] Saved: {OUT_DIR}/multiclass_cm_{fname}.png")


# ── Per-class accuracy bar chart ──────────────────────────────────────────────
def plot_perclass(y_test, rf_pred, et_pred, class_names):
    from sklearn.metrics import precision_recall_fscore_support
    _, _, rf_f1, _ = precision_recall_fscore_support(y_test, rf_pred, average=None)
    _, _, et_f1, _ = precision_recall_fscore_support(y_test, et_pred, average=None)

    x     = np.arange(len(class_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width/2, rf_f1, width, label="Random Forest", color="#1a3a5c", alpha=0.85)
    ax.bar(x + width/2, et_f1, width, label="Extra Trees",   color="#27ae60", alpha=0.85)

    ax.set_ylim(0, 1.1)
    ax.set_ylabel("F1 Score", fontsize=12, fontweight="bold")
    ax.set_title("Per-Class F1 Score — Multiclass Attack Detection\n(8 Classes)",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=10)
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    for i, (r, e) in enumerate(zip(rf_f1, et_f1)):
        ax.text(i - width/2, r + 0.02, f"{r:.2f}", ha="center", fontsize=8)
        ax.text(i + width/2, e + 0.02, f"{e:.2f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/multiclass_perclass_f1.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    [✓] Saved: {OUT_DIR}/multiclass_perclass_f1.png")


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Multiclass Attack Detection")
    print("=" * 55)

    X_train, X_test, y_train, y_test, class_names, le = load_data()

    # Train models
    rf, rf_acc, rf_f1, rf_pred = train_rf_multiclass(
        X_train, y_train, X_test, y_test, class_names)

    et, et_acc, et_f1, et_pred = train_et_multiclass(
        X_train, y_train, X_test, y_test, class_names)

    lstm, lstm_acc, lstm_f1, lstm_pred = train_lstm_multiclass(
        X_train, y_train, X_test, y_test, class_names)

    # Per-class chart
    print("\n[+] Generating per-class F1 chart ...")
    plot_perclass(y_test, rf_pred, et_pred, class_names)

    # Summary
    print(f"\n{'='*55}")
    print(f"  MULTICLASS SUMMARY")
    print(f"{'='*55}")
    print(f"  {'Model':<20} {'Accuracy':>10} {'F1':>10}")
    print(f"  {'-'*42}")
    print(f"  {'Random Forest':<20} {rf_acc:>10.4f} {rf_f1:>10.4f}")
    print(f"  {'Extra Trees':<20} {et_acc:>10.4f} {et_f1:>10.4f}")
    print(f"  {'LSTM':<20} {lstm_acc:>10.4f} {lstm_f1:>10.4f}")
    print(f"{'='*55}")

    # Save results
    results = [
        {"model": "Random Forest", "accuracy": rf_acc,   "f1": rf_f1},
        {"model": "Extra Trees",   "accuracy": et_acc,   "f1": et_f1},
        {"model": "LSTM",          "accuracy": lstm_acc, "f1": lstm_f1},
    ]
    with open(f"{MODELS_DIR}/multiclass_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[✓] All multiclass results saved!")
    print(f"[✓] Charts saved to {OUT_DIR}/")
    print(f"\n  Next step: python3 benchmarks.py")


if __name__ == "__main__":
    main()
