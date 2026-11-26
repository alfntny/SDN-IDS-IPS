"""
Generalisation checks for the paper (run from ~/ids_ips with the venv active):  python3 crossfile.py
  1) Cross-file:     train Normal(half)+OVS -> test Normal(half)+Metasploitable-2, and the reverse.
  2) Chronological:  per file, first 80% by Timestamp = train, last 20% = test.
  3) Duplicates:     share of random-split test rows with an exact feature-vector copy in train.
  4) Seeds:          random stratified split repeated for 5 seeds -> mean +/- std.
Trees only (RF, ET). Takes ~10 min. Writes results/crossfile_results.json
"""
import os
import numpy as np, pandas as pd, json
os.makedirs("results", exist_ok=True)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

FILES = {"normal": "data/Normal_data.csv", "ovs": "data/OVS.csv", "meta": "data/metasploitable-2.csv"}
DROP = ["Flow ID", "Src IP", "Src Port", "Dst IP"]
SEED = 42
RES = {}

def load(path):
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    df = df.drop(columns=[c for c in DROP if c in df.columns])
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df["y"] = (df["Label"].astype(str).str.strip().str.lower() != "normal").astype(int)
    df["_t"] = pd.to_datetime(df["Timestamp"], errors="coerce", dayfirst=True) if "Timestamp" in df.columns else pd.NaT
    return df

d = {k: load(v) for k, v in FILES.items()}
for k, df in d.items():
    print(f"{k:7s} rows={len(df):7d} attack_share={df['y'].mean():.3f} timestamp_parsed={df['_t'].notna().mean():.2%} "
          f"labels={sorted(df['Label'].astype(str).str.strip().unique())}")
feat = [c for c in d["ovs"].columns if c not in ("Label", "y", "_t", "Timestamp")]
XY = lambda df: (df[feat].values.astype(np.float32), df["y"].values)
models = lambda: [("RF", RandomForestClassifier(n_estimators=200, max_depth=20, n_jobs=-1, random_state=SEED)),
                  ("ET", ExtraTreesClassifier(n_estimators=200, max_depth=20, n_jobs=-1, random_state=SEED))]

def evaluate(title, tr, te, key):
    Xtr, ytr = XY(tr); Xte, yte = XY(te)
    print(f"\n=== {title}  (train {len(tr)}, test {len(te)}; test attack share {yte.mean():.3f}) ===")
    print("    test labels:", te['Label'].astype(str).str.strip().value_counts().to_dict())
    for mname, m in models():
        m.fit(Xtr, ytr); p = m.predict(Xte)
        tn, fp, fn, tp = confusion_matrix(yte, p, labels=[0, 1]).ravel()
        r = dict(acc=float(accuracy_score(yte, p)), wf1=float(f1_score(yte, p, average="weighted")),
                 attack_recall=float(tp / max(tp + fn, 1)), normal_fpr=float(fp / max(fp + tn, 1)),
                 tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
        RES[f"{key}|{mname}"] = r
        print(f"  {mname}: acc={r['acc']:.4f} wF1={r['wf1']:.4f} attack-recall={r['attack_recall']:.4f} "
              f"normal-FPR={r['normal_fpr']:.4f}  [TN={tn} FP={fp} FN={fn} TP={tp}]")

# 1) cross-file
n_a, n_b = train_test_split(d["normal"], test_size=0.5, random_state=SEED)
evaluate("CROSS-FILE: train Normal(A)+OVS -> test Normal(B)+Metasploitable-2", pd.concat([n_a, d["ovs"]]), pd.concat([n_b, d["meta"]]), "xfile_ovs_to_meta")
evaluate("CROSS-FILE: train Normal(A)+Metasploitable-2 -> test Normal(B)+OVS", pd.concat([n_a, d["meta"]]), pd.concat([n_b, d["ovs"]]), "xfile_meta_to_ovs")

# 2) chronological
if all(d[k]["_t"].notna().mean() > 0.9 for k in d):
    trs, tes = [], []
    for k, df in d.items():
        df = df.sort_values("_t"); cut = int(len(df) * 0.8)
        trs.append(df.iloc[:cut]); tes.append(df.iloc[cut:])
    evaluate("CHRONOLOGICAL: first 80% of each file (by time) -> last 20%", pd.concat(trs), pd.concat(tes), "chrono")
else:
    print("\n[!] Timestamps did not parse for >90% of rows; skipping chronological split. Paste 3 raw Timestamp values and I'll fix the format.")

# 3) duplicates + 4) seeds on the paper's random split
allf = pd.concat([d["normal"], d["ovs"], d["meta"]], ignore_index=True)
tr, te = train_test_split(allf, test_size=0.2, random_state=SEED, stratify=allf["y"])
h = lambda df: pd.util.hash_pandas_object(df[feat].round(6), index=False)
dup = float(h(te).isin(set(h(tr))).mean()); RES["duplicate_share_of_test"] = dup
print(f"\nDuplicate leakage: {dup*100:.2f}% of random-split test rows have an exact feature-vector duplicate in train")

print("\n=== SEEDS: random stratified 80/20, 5 seeds ===")
for mname in ("RF", "ET"):
    accs = []
    for s in range(5):
        a, b = train_test_split(allf, test_size=0.2, random_state=s, stratify=allf["y"])
        m = dict(RF=RandomForestClassifier(n_estimators=200, max_depth=20, n_jobs=-1, random_state=s),
                 ET=ExtraTreesClassifier(n_estimators=200, max_depth=20, n_jobs=-1, random_state=s))[mname]
        m.fit(*XY(a)); accs.append(accuracy_score(b["y"].values, m.predict(XY(b)[0])))
    RES[f"seeds|{mname}"] = dict(mean=float(np.mean(accs)), std=float(np.std(accs)), runs=[float(x) for x in accs])
    print(f"  {mname}: accuracy {100*np.mean(accs):.3f}% +/- {100*np.std(accs):.3f}  runs={[round(100*x,3) for x in accs]}")

json.dump(RES, open("results/crossfile_results.json", "w"), indent=2)
print("\nSaved results/crossfile_results.json")
