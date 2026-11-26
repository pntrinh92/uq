#!/usr/bin/env bash
# Fidelity arms for the last-4-block flow (2026-11-21). Same recipe as run_full.sh last4 (ode8_full: init
# ode8_flownet.pt, all 1.28M images, lr 2e-4, cosine schedule over the run, depth-6 field, rollout from step 0),
# changing one thing each:
#   bash run_arms.sh A   # ode8_full120k: 120k steps with its own schedule (ode8_full had 60k)
#   bash run_arms.sh B   # ode8_full_kl1: 60k steps, rollout-KL weight 1.0 (ode8_full had 0.1)
# Every snapshot and the final field are scored by `compile_flow.py final` (top-1/top-5 agreement, KL, accuracy on
# the same 10k test images as ode8_full: 80.7 / 96.1 / 0.417 / 75.2). Run inside tmux "pnt".
set -uo pipefail
cd "$(dirname "$0")"
source /ephemeral/trinhpn_uq/env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True LADDER=imagenet COMPILE_DEPTH=6
LOGS=/ephemeral/trinhpn_uq/logs/full
CK=/ephemeral/trinhpn_uq/ckpt
mkdir -p "$LOGS"

retry_gpu() {   # retry_gpu <log name> <command...>   (as run_full.sh)
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

arm() {   # arm <tag> <steps> <lam_kl> <snapshot steps, comma-separated>
  local tag=$1 steps=$2
  export COMPILE_TAG=$tag COMPILE_FROM=8 COMPILE_INIT=$CK/ode8_flownet.pt COMPILE_ADAPTER_RANK=0
  export COMPILE_DATA=train,train2,train3a,train3b,train3c COMPILE_LR=2e-4 COMPILE_ROLLOUT_START=0
  export COMPILE_LAM_KL=$3 COMPILE_SNAPSHOTS=$4
  retry_gpu train_$tag $PY compile_flow.py train $steps || return 1
  for st in ${4//,/ }; do
    [ -f $CK/${tag}_step${st}_flownet.pt ] && COMPILE_EVAL_CKPT=$CK/${tag}_step${st}_flownet.pt retry_gpu final_${tag}_$st $PY compile_flow.py final
  done
  retry_gpu final_$tag $PY compile_flow.py final || return 1
}

case "${1:-}" in
  A) arm ode8_full120k 120000 0.1 30000,60000,90000 && echo ARM_A_DONE || echo ARM_A_FAILED ;;
  B) arm ode8_full_kl1 60000 1.0 15000,30000,45000 && echo ARM_B_DONE || echo ARM_B_FAILED ;;
  *) echo "usage: run_arms.sh A|B"; exit 2 ;;
esac
