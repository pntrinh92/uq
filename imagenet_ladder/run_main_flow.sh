#!/usr/bin/env bash
# Flow as the main model (2026-11-27, Prof. Nghia's direction): blocks 1-8 of ViT-B/16 run exactly, the FlowNet
# replaces blocks 9-12 (4 Euler steps) and its logits are the model's prediction. Uncertainty is to come from the
# flow itself afterwards (last-layer Laplace + CoMP-style coherent propagation), not part of this script.
#
# Both arms fine-tune ode8_full120k (10k test: 83.5 agree / 77.3 acc; teacher 81.7 acc) for 20k more steps with a
# fresh warm-up + cosine schedule (lr 1e-4), 16 images rolled out per step with activation checkpointing
# (ode8_full120k: 4), and change only lambda_CE:
#   bash run_main_flow.sh ce     # ode8_main_ce:   + 0.3 * CE(rollout logits, ImageNet label)
#   bash run_main_flow.sh noce   # ode8_main_noce: label-free control (extra steps + larger rollout batch only)
# The control separates the effect of the labels from that of training longer with a larger rollout batch.
# Peak memory about 4.3 GB (plus the CUDA context); the job waits for a card with 5 GB free. GPU_ALLOW_SHARED=1 lets
# it use a card one of our other jobs holds (GPU 3 stays excluded). Run inside tmux "pnt".
set -uo pipefail
cd "$(dirname "$0")"
source /ephemeral/trinhpn_uq/env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True LADDER=imagenet COMPILE_DEPTH=6 GPU_ALLOW_SHARED=1
LOGS=/ephemeral/trinhpn_uq/logs/full
CK=/ephemeral/trinhpn_uq/ckpt
mkdir -p "$LOGS"

retry_gpu() {   # retry_gpu <log name> <command...>   (as run_arms.sh)
  local name=$1; shift
  for attempt in $(seq 1 300); do
    echo "[$(date +%H:%M:%S)] $name attempt $attempt"
    NUM_THREADS=8 "$@" 2>&1 | grep --line-buffered -v -i "warn" | tee -a "$LOGS/$name.log"
    local rc=${PIPESTATUS[0]}
    if [ "$rc" -eq 0 ]; then echo "[$(date +%H:%M:%S)] $name OK"; return 0; fi
    if tail -n 40 "$LOGS/$name.log" | grep -qE "OutOfMemoryError|no GPU with|CUDA error: out of memory"; then
      echo "[$(date +%H:%M:%S)] $name: GPU memory busy, retrying in 120 s"; sleep 120
    else
      echo "[$(date +%H:%M:%S)] $name FAILED (rc=$rc)"; return 1
    fi
  done
  return 1
}

arm() {   # arm <tag> <lam_ce>
  local tag=$1
  export COMPILE_TAG=$tag COMPILE_FROM=8 COMPILE_INIT=$CK/ode8_full120k_flownet.pt COMPILE_ADAPTER_RANK=0
  export COMPILE_DATA=train,train2,train3a,train3b,train3c COMPILE_LR=1e-4 COMPILE_ROLLOUT_START=0
  export COMPILE_LAM_KL=0.1 COMPILE_LAM_CE=$2 COMPILE_ROLL_BATCH=16 COMPILE_ROLL_CKPT=1 COMPILE_SNAPSHOTS=10000
  export COMPILE_MIN_FREE_MB=5000
  retry_gpu train_$tag $PY compile_flow.py train 20000 || return 1
  unset COMPILE_MIN_FREE_MB
  [ -f $CK/${tag}_step10000_flownet.pt ] && COMPILE_EVAL_CKPT=$CK/${tag}_step10000_flownet.pt retry_gpu final_${tag}_10000 $PY compile_flow.py final
  retry_gpu final_$tag $PY compile_flow.py final || return 1
}

case "${1:-}" in
  ce)   arm ode8_main_ce 0.3 && echo MAIN_CE_DONE || echo MAIN_CE_FAILED ;;
  noce) arm ode8_main_noce 0 && echo MAIN_NOCE_DONE || echo MAIN_NOCE_FAILED ;;
  *) echo "usage: run_main_flow.sh ce|noce"; exit 2 ;;
esac
