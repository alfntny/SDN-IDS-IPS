"""
Step 2: Baseline ML Models — Random Forest & SVM
SDN IDS/IPS Project — SRM SIST, Batch B1025

Trains RF and SVM on preprocessed InSDN data.
Evaluates accuracy, precision, recall, F1, confusion matrix.
Saves trained models for comparison with LSTM.

Run after preprocess.py:
    python baseline_models.py
"""

import numpy as np
import json
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)

DATA_DIR = "data/processed"
MODELS_DIR = "models"
RANDOM_STATE = 42


def load_data():
    print("[+] Loading preprocessed data ...")
    X_train = np.load(f"{DATA_DIR}/X_train.npy")
    X_test  = np.load(f"{DATA_DIR}/X_test.npy")
    y_train = np.load(f"{DATA_DIR}/y_bin_train.npy")
    y_test  = np.load(f"{DATA_DIR}/y_bin_test.npy")

    with open(f"{DATA_DIR}/metadata.json") as f:
        meta = json.load(f)

    print(f"    Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test, meta


def evaluate(model, X_test, y_test, name):
    print(f"\n{'='*50}")
    print(f"  Results: {name}")
    print(f"{'='*50}")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred,
                                 target_names=["Normal", "Attack"]))
    cm = confusion_matrix(y_test, y_pred)
    print(f"  Confusion Matrix:\n{cm}")
    return {"model": name, "accuracy": acc, "f1": f1}


def train_random_forest(X_train, y_train):
    print("\n[+] Training Random Forest ...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=2,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    rf.fit(X_train, y_train)
    print("    [✓] Done")
    return rf


def train_svm(X_train, y_train):
    print("\n[+] Training SVM (RBF kernel) ...")
    # Use a subset for speed — SVM is O(n²) on large data
    MAX_SVM_SAMPLES = 20000
    if len(X_train) > MAX_SVM_SAMPLES:
        print(f"    [!] Large dataset — sampling {MAX_SVM_SAMPLES} rows for SVM")
        idx = np.random.RandomState(RANDOM_STATE).choice(
            len(X_train), MAX_SVM_SAMPLES, replace=False
        )
        X_sub, y_sub = X_train[idx], y_train[idx]
    else:
        X_sub, y_sub = X_train, y_train

    svm = SVC(kernel="rbf", C=10, gamma=0.1, probability=True,
              random_state=RANDOM_STATE)
    svm.fit(X_sub, y_sub)
    print("    [✓] Done")
    return svm


def save_model(model, name):
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = f"{MODELS_DIR}/{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"    [✓] Saved → {path}")


def run():
    X_train, X_test, y_train, y_test, meta = load_data()

    results = []

    # Random Forest
    rf = train_random_forest(X_train, y_train)
    r = evaluate(rf, X_test, y_test, "Random Forest")
    results.append(r)
    save_model(rf, "random_forest")

    # SVM
    svm = train_svm(X_train, y_train)
    r = evaluate(svm, X_test, y_test, "SVM")
    results.append(r)
    save_model(svm, "svm")

    # Summary
    print("\n" + "="*50)
    print("  BASELINE SUMMARY")
    print("="*50)
    for r in results:
        print(f"  {r['model']:<20} Acc: {r['accuracy']:.4f}  F1: {r['f1']:.4f}")

    with open(f"{MODELS_DIR}/baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n[✓] Baselines complete. Next step: run lstm_model.py")


if __name__ == "__main__":
    run()
