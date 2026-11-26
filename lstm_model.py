"""
Step 3: LSTM-Based Anomaly Detection Model
SDN IDS/IPS Project — SRM SIST, Batch B1025

Trains a bidirectional LSTM on sequential SDN flow windows.
Captures temporal patterns in network traffic for zero-day detection.

Run after baseline_models.py:
    python lstm_model.py

Requirements:
    pip install torch
"""

import numpy as np
import json
import os
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from sklearn.model_selection import train_test_split

DATA_DIR  = "data/processed"
MODELS_DIR = "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── HYPERPARAMETERS ───────────────────────────────────────────────────────────
SEQ_LEN     = 10        # number of consecutive flows per window
HIDDEN_SIZE = 128
NUM_LAYERS  = 2
DROPOUT     = 0.3
BATCH_SIZE  = 256
EPOCHS      = 30
LR          = 1e-3
PATIENCE    = 5         # early stopping patience
RANDOM_STATE = 42

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# ─── MODEL ─────────────────────────────────────────────────────────────────────
class LSTMDetector(nn.Module):
    """
    Bidirectional LSTM for SDN flow anomaly detection.
    Input:  (batch, seq_len, n_features)
    Output: (batch, 2)  — binary: Normal / Attack
    """
    def __init__(self, n_features: int, hidden_size: int = HIDDEN_SIZE,
                 num_layers: int = NUM_LAYERS, dropout: float = DROPOUT):
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
            nn.Linear(hidden_size * 2, 64),   # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        out, _ = self.lstm(x)
        # Take last timestep output
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.classifier(out)


# ─── HELPERS ───────────────────────────────────────────────────────────────────
def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    """
    Slide a window of size seq_len over the data.
    Label of window = label of last sample in window.
    """
    Xs, ys = [], []
    for i in range(len(X) - seq_len + 1):
        Xs.append(X[i:i + seq_len])
        ys.append(y[i + seq_len - 1])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.int64)


def load_data():
    print("[+] Loading preprocessed data ...")
    X_train = np.load(f"{DATA_DIR}/X_train.npy")
    X_test  = np.load(f"{DATA_DIR}/X_test.npy")
    y_train = np.load(f"{DATA_DIR}/y_bin_train.npy")
    y_test  = np.load(f"{DATA_DIR}/y_bin_test.npy")
    print(f"    Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def get_loaders(X_train, X_test, y_train, y_test):
    """Hold out 10% of TRAIN as a stratified validation split for early stopping.
    The test split is used only for the final evaluation, never for model selection."""
    print(f"[+] Building sequence windows (seq_len={SEQ_LEN}) ...")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, stratify=y_train, random_state=RANDOM_STATE
    )
    print(f"    Train flows: {len(X_tr)} | Val flows: {len(X_val)} | Test flows: {len(X_test)}")

    X_tr_seq,  y_tr_seq  = make_sequences(X_tr,  y_tr,  SEQ_LEN)
    X_val_seq, y_val_seq = make_sequences(X_val, y_val, SEQ_LEN)
    X_te_seq,  y_te_seq  = make_sequences(X_test, y_test, SEQ_LEN)
    print(f"    Train sequences: {X_tr_seq.shape}")
    print(f"    Val   sequences: {X_val_seq.shape}")
    print(f"    Test  sequences: {X_te_seq.shape}")

    mk = lambda X, y, sh: DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
        batch_size=BATCH_SIZE, shuffle=sh, num_workers=0)
    return mk(X_tr_seq, y_tr_seq, True), mk(X_val_seq, y_val_seq, False), \
           mk(X_te_seq, y_te_seq, False), y_te_seq


def train(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        preds = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total   += len(y_batch)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_loader(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        preds = model(X_batch).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y_batch.numpy())
    return np.array(all_preds), np.array(all_labels)


def run():
    print("=" * 60)
    print("  LSTM Anomaly Detector Training")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    X_train, X_test, y_train, y_test = load_data()
    n_features = X_train.shape[1]

    train_loader, val_loader, test_loader, y_te_seq = get_loaders(
        X_train, X_test, y_train, y_test
    )

    model = LSTMDetector(n_features=n_features).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=2, factor=0.5
    )

    print(f"\n[+] Model parameters: "
          f"{sum(p.numel() for p in model.parameters()):,}")
    print(f"    Training for up to {EPOCHS} epochs (early stop patience={PATIENCE})\n")

    best_f1, patience_counter = 0.0, 0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        loss, acc = train(model, train_loader, optimizer, criterion)
        preds, labels = evaluate_loader(model, val_loader)   # VALIDATION, not test
        val_f1 = f1_score(labels, preds, average="weighted")
        scheduler.step(val_f1)

        print(f"  Epoch {epoch:02d}/{EPOCHS}  "
              f"loss={loss:.4f}  train_acc={acc:.4f}  val_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch}")
                break

    # Load best weights
    model.load_state_dict(best_state)

    # Final evaluation
    preds, labels = evaluate_loader(model, test_loader)
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="weighted")

    print(f"\n{'='*50}")
    print(f"  LSTM Final Results")
    print(f"{'='*50}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(labels, preds,
                                 target_names=["Normal", "Attack"]))
    print(f"  Confusion Matrix:\n{confusion_matrix(labels, preds)}")

    # Save
    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save({
        "model_state": best_state,
        "n_features": n_features,
        "seq_len": SEQ_LEN,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
    }, f"{MODELS_DIR}/lstm_detector.pt")
    print(f"\n[✓] Model saved → {MODELS_DIR}/lstm_detector.pt")

    result = {"model": "LSTM", "accuracy": acc, "f1": f1}
    with open(f"{MODELS_DIR}/lstm_results.json", "w") as f:
        json.dump(result, f, indent=2)

    print("[✓] LSTM training complete. Next step: run ryu_ids_app.py")


if __name__ == "__main__":
    run()
