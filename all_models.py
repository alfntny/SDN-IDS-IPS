"""
Step 2b: All Models — Full Comparison
SDN IDS/IPS Project — SRM SIST, Batch B1025

Trains all models from the project proposal:
- Random Forest (already done, reloaded)
- Gradient Boosting
- Extra Trees
- K-Nearest Neighbors
- Neural Network (MLP)
- Bagging
- Voting Regressor
- Stacking
- Blending

Run:
    python3 all_models.py
"""

import numpy as np
import json
import pickle
import os
import time
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, BaggingClassifier, VotingClassifier,
    StackingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, classification_report
)

DATA_DIR   = "data/processed"
MODELS_DIR = "models"
RANDOM_STATE = 42
os.makedirs(MODELS_DIR, exist_ok=True)


def load_data():
    print("[+] Loading preprocessed data ...")
    X_train = np.load(f"{DATA_DIR}/X_train.npy")
    X_test  = np.load(f"{DATA_DIR}/X_test.npy")
    y_train = np.load(f"{DATA_DIR}/y_bin_train.npy")
    y_test  = np.load(f"{DATA_DIR}/y_bin_test.npy")
    print(f"    Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def evaluate(name, model, X_test, y_test):
    start  = time.time()
    y_pred = model.predict(X_test)
    infer  = time.time() - start

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average="weighted")
    prec = precision_score(y_test, y_pred, average="weighted")
    rec  = recall_score(y_test, y_pred, average="weighted")

    print(f"\n{'='*52}")
    print(f"  {name}")
    print(f"{'='*52}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  Infer time: {infer:.3f}s")

    return {
        "model": name, "accuracy": acc,
        "precision": prec, "recall": rec,
        "f1": f1, "infer_time": infer
    }


def train_and_eval(name, model, X_train, y_train, X_test, y_test, save=True):
    print(f"\n[+] Training {name} ...")
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    print(f"    Done in {train_time:.1f}s")

    result = evaluate(name, model, X_test, y_test)
    result["train_time"] = train_time

    if save:
        fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        with open(f"{MODELS_DIR}/{fname}.pkl", "wb") as f:
            pickle.dump(model, f)
        print(f"    Saved → {MODELS_DIR}/{fname}.pkl")

    return result, model


def run():
    print("=" * 52)
    print("  Full Model Comparison — SDN IDS/IPS")
    print("=" * 52)

    X_train, X_test, y_train, y_test = load_data()

    # Subsample for slow models
    MAX_SLOW = 20000
    idx = np.random.RandomState(RANDOM_STATE).choice(
        len(X_train), MAX_SLOW, replace=False
    )
    X_sub = X_train[idx]
    y_sub = y_train[idx]

    results = []

    # ── 1. Random Forest ──────────────────────────────────────────────────────
    if os.path.exists(f"{MODELS_DIR}/random_forest.pkl"):
        print("\n[+] Loading existing Random Forest ...")
        with open(f"{MODELS_DIR}/random_forest.pkl", "rb") as f:
            rf = pickle.load(f)
        r = evaluate("Random Forest", rf, X_test, y_test)
        r["train_time"] = 0.0
        results.append(r)
    else:
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=20,
            min_samples_split=2, n_jobs=-1,
            random_state=RANDOM_STATE
        )
        r, rf = train_and_eval("Random Forest", rf, X_train, y_train, X_test, y_test)
        results.append(r)

    # ── 2. Gradient Boosting ──────────────────────────────────────────────────
    gb = GradientBoostingClassifier(
        n_estimators=200, max_depth=7,
        learning_rate=0.1, subsample=0.8,
        random_state=RANDOM_STATE
    )
    r, gb = train_and_eval("Gradient Boosting", gb, X_sub, y_sub, X_test, y_test)
    results.append(r)

    # ── 3. Extra Trees ────────────────────────────────────────────────────────
    et = ExtraTreesClassifier(
        n_estimators=200, max_depth=20,
        n_jobs=-1, random_state=RANDOM_STATE
    )
    r, et = train_and_eval("Extra Trees", et, X_train, y_train, X_test, y_test)
    results.append(r)

    # ── 4. K-Nearest Neighbors ────────────────────────────────────────────────
    knn = KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1)
    r, knn = train_and_eval("KNN", knn, X_sub, y_sub, X_test, y_test)
    results.append(r)

    # ── 5. SVM ────────────────────────────────────────────────────────────────
    if os.path.exists(f"{MODELS_DIR}/svm.pkl"):
        print("\n[+] Loading existing SVM ...")
        with open(f"{MODELS_DIR}/svm.pkl", "rb") as f:
            svm = pickle.load(f)
        r = evaluate("SVM", svm, X_test, y_test)
        r["train_time"] = 0.0
        results.append(r)
    else:
        svm = SVC(kernel="rbf", C=10, gamma=0.1,
                  probability=True, random_state=RANDOM_STATE)
        r, svm = train_and_eval("SVM", svm, X_sub, y_sub, X_test, y_test)
        results.append(r)

    # ── 6. MLP Neural Network ─────────────────────────────────────────────────
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),
        activation="relu", solver="adam",
        max_iter=200, random_state=RANDOM_STATE
    )
    r, mlp = train_and_eval("MLP Neural Network", mlp, X_train, y_train, X_test, y_test)
    results.append(r)

    # ── 7. Bagging ────────────────────────────────────────────────────────────
    bagging = BaggingClassifier(
        estimator=RandomForestClassifier(n_estimators=10, random_state=RANDOM_STATE),
        n_estimators=10, n_jobs=-1, random_state=RANDOM_STATE
    )
    r, bagging = train_and_eval("Bagging", bagging, X_sub, y_sub, X_test, y_test)
    results.append(r)

    # ── 8. Voting ─────────────────────────────────────────────────────────────
    voting = VotingClassifier(
        estimators=[
            ("rf",  RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE)),
            ("et",  ExtraTreesClassifier(n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE)),
            ("knn", KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
        ],
        voting="hard", n_jobs=-1
    )
    r, voting = train_and_eval("Voting Ensemble", voting, X_sub, y_sub, X_test, y_test)
    results.append(r)

    # ── 9. Stacking ───────────────────────────────────────────────────────────
    stacking = StackingClassifier(
        estimators=[
            ("rf",  RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE)),
            ("et",  ExtraTreesClassifier(n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE)),
            ("knn", KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
        ],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5, n_jobs=-1
    )
    r, stacking = train_and_eval("Stacking", stacking, X_sub, y_sub, X_test, y_test)
    results.append(r)

    # ── 10. Blending (manual holdout ensemble) ────────────────────────────────
    print("\n[+] Training Blending ensemble ...")
    blend_start = time.time()

    # Use 80/20 split of subsample for blending
    split = int(0.8 * len(X_sub))
    X_bl_train, X_bl_hold = X_sub[:split], X_sub[split:]
    y_bl_train, y_bl_hold = y_sub[:split], y_sub[split:]

    bl_rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE)
    bl_et = ExtraTreesClassifier(n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE)
    bl_gb = GradientBoostingClassifier(n_estimators=50, random_state=RANDOM_STATE)

    bl_rf.fit(X_bl_train, y_bl_train)
    bl_et.fit(X_bl_train, y_bl_train)
    bl_gb.fit(X_bl_train, y_bl_train)

    # Meta features from holdout
    meta_hold = np.column_stack([
        bl_rf.predict_proba(X_bl_hold)[:, 1],
        bl_et.predict_proba(X_bl_hold)[:, 1],
        bl_gb.predict_proba(X_bl_hold)[:, 1],
    ])
    meta_learner = LogisticRegression(max_iter=1000)
    meta_learner.fit(meta_hold, y_bl_hold)

    # Predict on test
    meta_test = np.column_stack([
        bl_rf.predict_proba(X_test)[:, 1],
        bl_et.predict_proba(X_test)[:, 1],
        bl_gb.predict_proba(X_test)[:, 1],
    ])
    y_blend_pred = meta_learner.predict(meta_test)
    blend_time   = time.time() - blend_start

    acc  = accuracy_score(y_test, y_blend_pred)
    f1   = f1_score(y_test, y_blend_pred, average="weighted")
    prec = precision_score(y_test, y_blend_pred, average="weighted")
    rec  = recall_score(y_test, y_blend_pred, average="weighted")

    print(f"\n{'='*52}")
    print(f"  Blending")
    print(f"{'='*52}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  Done in   : {blend_time:.1f}s")

    results.append({
        "model": "Blending", "accuracy": acc,
        "precision": prec, "recall": rec,
        "f1": f1, "train_time": blend_time, "infer_time": 0.0
    })

    # ── LSTM (already trained, load results) ──────────────────────────────────
    if os.path.exists(f"{MODELS_DIR}/lstm_results.json"):
        with open(f"{MODELS_DIR}/lstm_results.json") as f:
            lstm_r = json.load(f)
        lstm_r["precision"] = lstm_r.get("precision", lstm_r["accuracy"])
        lstm_r["recall"]    = lstm_r.get("recall", lstm_r["accuracy"])
        lstm_r["train_time"] = 0.0
        lstm_r["infer_time"] = 0.0
        results.append(lstm_r)
        print(f"\n[+] LSTM results loaded from file")

    # ── FINAL SUMMARY TABLE ───────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  FINAL COMPARISON — ALL MODELS")
    print(f"{'='*72}")
    print(f"  {'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Train(s)':>9}")
    print(f"  {'-'*68}")

    results_sorted = sorted(results, key=lambda x: x["accuracy"], reverse=True)
    for r in results_sorted:
        print(f"  {r['model']:<22} {r['accuracy']:>9.4f} {r['precision']:>10.4f} "
              f"{r['recall']:>8.4f} {r['f1']:>8.4f} {r['train_time']:>9.1f}s")

    print(f"{'='*72}")
    print(f"\n  Best model: {results_sorted[0]['model']} "
          f"(Accuracy: {results_sorted[0]['accuracy']:.4f})")

    # Save all results
    with open(f"{MODELS_DIR}/all_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[✓] All results saved → {MODELS_DIR}/all_results.json")
    print("[✓] Next step: python3 evaluate.py  (graphs + charts)")


if __name__ == "__main__":
    run()
