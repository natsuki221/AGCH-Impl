#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

cd "$PROJECT_ROOT"

# Optional overrides (export before running)
ALPHA=${ALPHA:-}
BETA=${BETA:-}
GAMMA=${GAMMA:-}

alpha_arg=""
beta_arg=""
gamma_arg=""

if [[ -n "$ALPHA" ]]; then
  alpha_arg="alpha=$ALPHA"
fi
if [[ -n "$BETA" ]]; then
  beta_arg="beta=$BETA"
fi
if [[ -n "$GAMMA" ]]; then
  gamma_arg="gamma=$GAMMA"
fi

python src/train.py \
  model.hash_code_length=16 \
  hydra.run.dir="logs/agch_bits16" \
  $alpha_arg $beta_arg $gamma_arg

python src/train.py \
  model.hash_code_length=32 \
  hydra.run.dir="logs/agch_bits32" \
  $alpha_arg $beta_arg $gamma_arg

python src/train.py \
  model.hash_code_length=64 \
  hydra.run.dir="logs/agch_bits64" \
  $alpha_arg $beta_arg $gamma_arg
