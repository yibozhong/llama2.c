#!/bin/bash
set -euo pipefail
mkdir -p results
trap 'tar -czf results.tar.gz results' EXIT
export NETRC="$PWD/.netrc" WANDB_DIR="$PWD/results" OMP_NUM_THREADS=1
python render_local_lr.py
