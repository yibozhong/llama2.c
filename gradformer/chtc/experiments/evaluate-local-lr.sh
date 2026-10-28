#!/bin/bash
set -euo pipefail
GRADFORMER_WORKDIR="$PWD"
mkdir -p results
trap 'tar -czf "$GRADFORMER_WORKDIR/results.tar.gz" -C "$GRADFORMER_WORKDIR" results' EXIT
export NETRC="$PWD/.netrc" WANDB_DIR="$PWD/results" WANDB_PROJECT=nanochat WANDB_ENTITY=yibozhong657-none
export OMP_NUM_THREADS=1 CC=/usr/bin/gcc CXX=/usr/bin/g++
export TRITON_CACHE_DIR="$PWD/triton-cache"
tar -xzf llama2-local-lr-eval-v1.tar.gz
tar -xzf cce-3de376c.tar.gz
tar -xf dataset-v1.tar
tar -xf local-lr-selected-models.tar
export PYTHONPATH="$PWD/cce-source${PYTHONPATH:+:$PYTHONPATH}"
cp /opt/gradformer-environment.txt results/container-environment.txt
cp /opt/gradformer-locks.txt results/container-locks.txt
sha256sum llama2-local-lr-eval-v1.tar.gz local-lr-selected-models.tar cce-3de376c.tar.gz > results/input-hashes.txt
cd llama2.c
python -m gradformer.evaluate_local_lr --data ../dataset --selection ../selected/selection.json --output ../results
