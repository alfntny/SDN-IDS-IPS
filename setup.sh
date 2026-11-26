#!/bin/bash
# =============================================================================
# SDN IDS/IPS Project — Ubuntu Environment Setup
# SRM SIST | Batch B1025 | 21CSP302L
# Run this ONCE on a fresh Ubuntu 22.04 install (or WSL2)
# Usage: chmod +x setup.sh && ./setup.sh
# =============================================================================

set -e  # exit on any error

echo "============================================="
echo "  SDN IDS/IPS Project — Environment Setup"
echo "============================================="

# ── 1. System packages ────────────────────────────────────────────────────────
echo ""
echo "[1/6] Installing system packages ..."
sudo apt-get update -qq
sudo apt-get install -y \
    python3 python3-pip python3-venv \
    net-tools curl git \
    openvswitch-switch openvswitch-testcontroller \
    mininet \
    iperf3 tcpdump wireshark-common \
    build-essential

echo "  [✓] System packages done"

# ── 2. Python virtual environment ────────────────────────────────────────────
echo ""
echo "[2/6] Creating Python virtual environment ..."
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip -q

echo "  [✓] Virtualenv created at ./venv"

# ── 3. Python ML packages ─────────────────────────────────────────────────────
echo ""
echo "[3/6] Installing Python ML packages ..."
pip install -q \
    pandas numpy scikit-learn \
    torch torchvision \
    matplotlib seaborn \
    imbalanced-learn \
    joblib tqdm

echo "  [✓] ML packages done"

# ── 4. Ryu controller ─────────────────────────────────────────────────────────
echo ""
echo "[4/6] Installing Ryu SDN controller ..."
# Ryu needs an older eventlet
pip install -q \
    ryu \
    eventlet==0.30.2 \
    oslo.config \
    tinyrpc \
    ovs

echo "  [✓] Ryu installed"

# ── 5. Project folder structure ───────────────────────────────────────────────
echo ""
echo "[5/6] Creating project folder structure ..."
mkdir -p data/processed
mkdir -p models
mkdir -p logs

echo "  [✓] Folders created"

# ── 6. Verify everything ──────────────────────────────────────────────────────
echo ""
echo "[6/6] Verifying installations ..."

python3 -c "import pandas; print('  pandas      ✓', pandas.__version__)"
python3 -c "import numpy; print('  numpy       ✓', numpy.__version__)"
python3 -c "import sklearn; print('  sklearn     ✓', sklearn.__version__)"
python3 -c "import torch; print('  torch       ✓', torch.__version__)"
python3 -c "import ryu; print('  ryu         ✓')"

# Check Mininet
if command -v mn &> /dev/null; then
    echo "  mininet     ✓ $(mn --version 2>&1)"
else
    echo "  mininet     ✗ not found — try: sudo apt install mininet"
fi

# Check OVS
if command -v ovs-vsctl &> /dev/null; then
    echo "  ovs         ✓ $(ovs-vsctl --version | head -1)"
else
    echo "  ovs         ✗ not found"
fi

echo ""
echo "============================================="
echo "  Setup complete!"
echo ""
echo "  NEXT STEPS:"
echo "  1. Activate venv:       source venv/bin/activate"
echo "  2. Get InSDN dataset:   3 CSVs into data/ — see README.md"
echo "  3. Run preprocessing:   python3 preprocess.py"
echo "  4. Train baselines:     python baseline_models.py"
echo "  5. Train LSTM:          python lstm_model.py"
echo "  6. Start Mininet:       sudo mn --topo single,3 --controller remote"
echo "  7. Start Ryu IDS/IPS:   ryu-manager ryu_ids_app.py"
echo "============================================="
