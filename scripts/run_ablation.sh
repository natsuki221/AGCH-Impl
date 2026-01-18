#!/usr/bin/env bash
# Ablation Experiment Script for GCN Normalization
# Compares: Baseline (normalize=false) vs Paper Fix (normalize=true)
# AGCH-Impl - January 2026

set -e

echo "=========================================="
echo "🔬 GCN Normalization Ablation Experiment"
echo "=========================================="

# Configuration
EPOCHS=50
BITS=16
DATA_DIR="./data"
LOG_BASE="./logs/ablation"

# --- Experiment A: Baseline (normalize=false) ---
echo ""
echo "[Exp A] Running Baseline (gcn_normalize=false)..."
python src/train.py \
    model.hash_code_len=${BITS} \
    model.gcn_normalize=false \
    trainer.max_epochs=${EPOCHS} \
    hydra.run.dir="${LOG_BASE}/baseline_bits${BITS}"

echo "[Exp A] Baseline complete. Logs at: ${LOG_BASE}/baseline_bits${BITS}"

# --- Experiment B: Paper Fix (normalize=true) ---
echo ""
echo "[Exp B] Running Paper Fix (gcn_normalize=true)..."
python src/train.py \
    model.hash_code_len=${BITS} \
    model.gcn_normalize=true \
    trainer.max_epochs=${EPOCHS} \
    hydra.run.dir="${LOG_BASE}/paperfix_bits${BITS}"

echo "[Exp B] Paper Fix complete. Logs at: ${LOG_BASE}/paperfix_bits${BITS}"

# --- Summary ---
echo ""
echo "=========================================="
echo "🎉 Ablation Experiment Complete!"
echo "Compare results using TensorBoard:"
echo "  tensorboard --logdir=${LOG_BASE}"
echo "=========================================="
