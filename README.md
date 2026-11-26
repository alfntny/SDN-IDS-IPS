# Anomaly-Based Intrusion Detection and Prevention in SDN

Code for the paper *"Anomaly-Based Intrusion Detection and Prevention in SDN: What Survives Cross-File Testing and Controller Deployment"*.

A BiLSTM intrusion detector trained on InSDN and deployed inside a Ryu controller that installs time-limited OpenFlow DROP rules. The paper audits how far the offline scores transfer: across capture files, across a deployable feature subset, and against a non-learned baseline running in the same controller.

**Authors:** Alfin Tony, Harshaditya Choudhary, Sakthipriya S. — SRMIST, Chennai

---

## Results summary

| Evaluation | Random Forest | BiLSTM |
|---|---|---|
| Random split, binary | 99.994% | 99.974% |
| Random split, 8-class (macro-F1) | 0.9992 (0.94) | 0.9976 (0.83) |
| Cross-file, OVS → Metasploitable-2 | 96.15% | — |
| Cross-file, Metasploitable-2 → OVS | 83.43% | — |
| 10 OpenFlow-derivable features only | 98.82% | 97.23% |

37.3% of random-split test rows have an exact feature-vector duplicate in the training set.

---

## Environment

Two separate virtualenvs are required. Ryu needs Python 3.8; PyTorch does not run well there.

```bash
# Training / analysis — Python 3.11
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Controller — Python 3.8, separate venv
python3.8 -m venv venv38
source venv38/bin/activate
pip install "setuptools<58"
pip install --no-build-isolation ryu==4.34 eventlet==0.30.2
pip install routes webob msgpack six netaddr tinyrpc oslo.config ovs psutil
```

Also needed: Mininet 2.2.2, Open vSwitch 2.13.8, hping3. Tested on Ubuntu 20.04 under WSL2.

`setup.sh` installs the system packages, but the two-venv split above supersedes its pip section.

---

## Dataset

InSDN is not included. Download `Normal_data.csv`, `OVS.csv` and `metasploitable-2.csv` from
https://aseados.ucd.ie/datasets/SDN/ and place them in `data/`.

343,889 flows, 78 features after dropping the five identifier columns, 8 classes.

---

## Pipeline

```bash
python3 preprocess.py          # → data/processed/  (stratified 80/20, seed 42, z-score)
python3 baseline_models.py     # RF + SVM
python3 all_models.py          # 11 classifiers, binary
python3 lstm_model.py          # BiLSTM, binary
python3 multiclass.py          # RF / ET / BiLSTM, 8 classes
python3 benchmarks.py          # latency, throughput, CPU, memory, model size
python3 evaluate.py            # charts
```

Audit scripts, which produce the paper's Sections V-F to V-H:

```bash
python3 crossfile.py           # cross-file split, duplicate leakage, 5-seed variance
python3 of_train.py            # 10 OpenFlow-derivable features only
```

---

## Live evaluation

Terminal 1 — controller:

```bash
source venv38/bin/activate
ryu-manager ryu_ids_app.py ryu.app.simple_switch_13 2>&1 | tee results/run.log
```

Terminal 2 — network:

```bash
sudo mn -c
sudo mn --topo single,3 --controller remote
mininet> pingall
mininet> sh date +%T.%N
mininet> h1 timeout 10 hping3 -S --flood -p 80 h2
mininet> h1 ping h2 -c 5
```

Terminal 3 — check the installed rules:

```bash
sudo ovs-ofctl -O OpenFlow13 dump-flows s1 | grep drop
```

Wait 65 s for the hard timeout, then ping again to confirm recovery.

For the non-learned baseline of Section V-H, substitute `ryu_baseline_app.py`, which keeps the
same polling, feature extraction, IP resolution and DROP rule but replaces the BiLSTM with a
packets-per-second threshold.

---

## Detector behaviour

| Attack probability | Action |
|---|---|
| < 0.50 | allow, no action |
| 0.50 – 0.85 | logged as suspicious, no rule installed |
| ≥ 0.85 | OpenFlow DROP rule, priority 100, 60 s hard timeout, once per IP pair |

No meter or rate limiting is implemented. The eight-class model supplies the attack label in the
alert only; it does not affect the decision.

---

## Known limitations

These are reported in the paper rather than fixed, and are the obvious places to start if you
want to build on this.

- **Feature-extraction defect.** Statistics entries without an `ipv4_src` match field are resolved
  through the controller's port and MAC tables, and several entries can resolve to the same
  `(switch, src, dst)` key. Both the BiLSTM and the threshold baseline produced the same
  false-positive block because of it. Keying per-flow state on the flow match should fix it.
- **Train/live feature gap.** The controller derives about 10 features from OpenFlow counters and
  zero-fills the other 68. The models were trained on CICFlowMeter bidirectional flow records.
- **No re-blocking.** `blocked_flows` is never cleared, so an attack that outlives the 60 s timeout
  is not blocked again within the same controller run.
- **Class imbalance.** U2R has 17 flows total and 4 in the test split, so its per-class F1 is noise.

---

## Files

| File | Purpose |
|---|---|
| `preprocess.py` | merge, clean, encode, split, scale InSDN |
| `baseline_models.py` | Random Forest, SVM |
| `all_models.py` | 11 classifiers, binary task |
| `lstm_model.py` | BiLSTM, binary task |
| `multiclass.py` | RF / ET / BiLSTM, 8 classes |
| `benchmarks.py` | inference latency, throughput, CPU, memory, model size |
| `evaluate.py` | charts |
| `crossfile.py` | cross-file split, duplicate leakage, seed variance |
| `of_train.py` | OpenFlow-derivable feature subset |
| `ryu_ids_app.py` | Ryu IDS/IPS application |
| `ryu_baseline_app.py` | packet-rate threshold baseline, same pipeline |
| `ips_demo.py` | offline terminal demo, replays test flows |
| `setup.sh` | system package install |

---

## Citing the dataset

M. S. Elsayed, N.-A. Le-Khac, and A. D. Jurcut, "InSDN: A novel SDN intrusion dataset,"
*IEEE Access*, vol. 8, pp. 165263–165284, 2020.
