"""
Step 1: InSDN Dataset Preprocessing Pipeline
SDN IDS/IPS Project — SRM SIST, Batch B1025

Merges Normal_data.csv, OVS.csv, metasploitable-2.csv
then cleans, encodes, normalizes and splits the data.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os
import json
import pickle

# ─── CONFIG ────────────────────────────────────────────────────────────────────
DATA_FILES = [
    "data/Normal_data.csv",
    "data/OVS.csv",
    "data/metasploitable-2.csv",
]
OUTPUT_DIR   = "data/processed"
RANDOM_STATE = 42
TEST_SIZE    = 0.2

DROP_COLS = ["Flow ID", "Src IP", "Src Port", "Dst IP", "Timestamp"]
LABEL_COL = "Label"

# ─── PIPELINE ──────────────────────────────────────────────────────────────────

def load_and_merge() -> pd.DataFrame:
    print("[1/6] Loading and merging CSV files ...")
    dfs = []
    for path in DATA_FILES:
        df = pd.read_csv(path, low_memory=False)
        df.columns = df.columns.str.strip()
        print(f"      {path}: {df.shape}  |  Labels: {df[LABEL_COL].unique()}")
        dfs.append(df)
    merged = pd.concat(dfs, ignore_index=True)
    print(f"      Merged shape: {merged.shape}")
    return merged


def drop_irrelevant(df: pd.DataFrame) -> pd.DataFrame:
    print("[2/6] Dropping irrelevant columns ...")
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    print(f"      Dropped: {cols_to_drop}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    print("[3/6] Cleaning data ...")
    n_before = len(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    print(f"      Removed {n_before - len(df)} rows with NaN/inf")
    print(f"      Remaining: {len(df)} rows")
    return df


def encode_labels(df: pd.DataFrame):
    print("[4/6] Encoding labels ...")
    df["label_binary"] = df[LABEL_COL].apply(
        lambda x: 0 if str(x).strip().lower() == "normal" else 1
    )
    le = LabelEncoder()
    df["label_multi"] = le.fit_transform(df[LABEL_COL].astype(str).str.strip())
    print(f"      Classes: {list(le.classes_)}")
    print(f"      Distribution:\n{df[LABEL_COL].value_counts().to_string()}")
    print(f"      Normal: {(df['label_binary']==0).sum()} | Attack: {(df['label_binary']==1).sum()}")
    return df, le


def split_normalize_save(df: pd.DataFrame, le: LabelEncoder):
    print("[5/6] Splitting and normalizing ...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    feature_cols = [c for c in df.columns
                    if c not in [LABEL_COL, "label_binary", "label_multi"]]

    X = df[feature_cols].values.astype(np.float32)
    y_bin   = df["label_binary"].values
    y_multi = df["label_multi"].values

    X_train, X_test, y_bin_train, y_bin_test, y_multi_train, y_multi_test = \
        train_test_split(X, y_bin, y_multi,
                         test_size=TEST_SIZE,
                         random_state=RANDOM_STATE,
                         stratify=y_bin)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    np.save(f"{OUTPUT_DIR}/X_train.npy",       X_train)
    np.save(f"{OUTPUT_DIR}/X_test.npy",        X_test)
    np.save(f"{OUTPUT_DIR}/y_bin_train.npy",   y_bin_train)
    np.save(f"{OUTPUT_DIR}/y_bin_test.npy",    y_bin_test)
    np.save(f"{OUTPUT_DIR}/y_multi_train.npy", y_multi_train)
    np.save(f"{OUTPUT_DIR}/y_multi_test.npy",  y_multi_test)

    with open(f"{OUTPUT_DIR}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(f"{OUTPUT_DIR}/label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    meta = {
        "n_features":         len(feature_cols),
        "feature_names":      feature_cols,
        "n_classes_multi":    len(le.classes_),
        "class_names":        list(le.classes_),
        "train_size":         int(len(X_train)),
        "test_size":          int(len(X_test)),
        "attack_ratio_train": float(y_bin_train.mean()),
        "attack_ratio_test":  float(y_bin_test.mean()),
    }
    with open(f"{OUTPUT_DIR}/metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"      Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"      Features: {len(feature_cols)}")
    print(f"      Attack ratio (train): {y_bin_train.mean():.2%}")


def main():
    print("=" * 55)
    print("  InSDN Preprocessing Pipeline")
    print("=" * 55)
    df = load_and_merge()
    df = drop_irrelevant(df)
    df = clean(df)
    df, le = encode_labels(df)
    split_normalize_save(df, le)
    print("\n[6/6] Done!")
    print(f"      Outputs saved to {OUTPUT_DIR}/")
    print("\n  Next step: python3 baseline_models.py")


if __name__ == "__main__":
    main()
