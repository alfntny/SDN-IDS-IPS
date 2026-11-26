"""Offline feasibility: detector using only OpenFlow-computable rate/size features.
Run:  python3 of_train.py            (RF/ET + BiLSTM)
      python3 of_train.py --trees-only
"""
import os, json, sys, time, pickle, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from lstm_model import LSTMDetector, make_sequences, train, evaluate_loader, DEVICE

os.makedirs("results", exist_ok=True); os.makedirs("models", exist_ok=True)
D = "data/processed"
WANT = ["Flow Byts/s","Flow Pkts/s","Fwd Pkts/s","Bwd Pkts/s","Pkt Len Mean",
        "Fwd Pkt Len Mean","Flow IAT Mean","Fwd IAT Mean","Pkt Size Avg","Fwd Seg Size Avg"]
meta = json.load(open(f"{D}/metadata.json"))
names = meta["feature_names"]
idx = [i for i, n in enumerate(names) if n.strip() in WANT]
print("[+] Using", len(idx), "features:", [names[i] for i in idx])

Xtr = np.load(f"{D}/X_train.npy")[:, idx]; Xte = np.load(f"{D}/X_test.npy")[:, idx]
ytr = np.load(f"{D}/y_bin_train.npy");     yte = np.load(f"{D}/y_bin_test.npy")

res = {"features": [names[i] for i in idx], "feature_idx": idx}
for nm, M in [("RF", RandomForestClassifier(200, max_depth=20, n_jobs=-1, random_state=42)),
              ("ET", ExtraTreesClassifier(200, max_depth=20, n_jobs=-1, random_state=42))]:
    M.fit(Xtr, ytr); p = M.predict(Xte)
    res[nm] = {"acc": accuracy_score(yte, p), "f1": f1_score(yte, p, average="weighted")}
    print(nm, res[nm]); print(confusion_matrix(yte, p))

if "--trees-only" not in sys.argv:
    torch.manual_seed(42); np.random.seed(42)
    Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.1, stratify=ytr, random_state=42)
    mk = lambda X, y, sh: DataLoader(TensorDataset(*[torch.from_numpy(a) for a in make_sequences(X, y, 10)]),
                                     batch_size=256, shuffle=sh)
    tl, vl, el = mk(Xa, ya, True), mk(Xv, yv, False), mk(Xte, yte, False)
    m = LSTMDetector(len(idx)).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), 1e-3); crit = nn.CrossEntropyLoss()
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=2)
    best, bs, bad = 0, None, 0
    for ep in range(30):
        tr = train(m, tl, opt, crit); p, l = evaluate_loader(m, vl); a = accuracy_score(l, p)
        sch.step(1 - a); print(f"ep{ep+1} train_acc={tr[1]:.4f} val_acc={a:.4f}")
        if a > best: best, bs, bad = a, {k: v.cpu().clone() for k, v in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 5: break
    m.load_state_dict(bs); p, l = evaluate_loader(m, el)
    res["BiLSTM"] = {"acc": accuracy_score(l, p), "f1": f1_score(l, p, average="weighted")}
    print("BiLSTM", res["BiLSTM"]); print(confusion_matrix(l, p))
    torch.save({"model_state": bs, "n_features": len(idx), "seq_len": 10, "hidden_size": 128,
                "num_layers": 2, "dropout": 0.3, "feature_idx": idx}, "models/lstm_detector_of.pt")

json.dump(res, open("results/of_feasibility.json", "w"), indent=2)
print("[✓] saved results/of_feasibility.json")
