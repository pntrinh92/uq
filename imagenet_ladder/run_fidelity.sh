#!/usr/bin/env bash
# Compile-fidelity options on ImageNet (run in tmux "pnt"). Every arm continues from the trained
# shared field of the ladder (ckpt/ode8_flownet.pt or ckpt/ode_flownet.pt):
#   cont   shared field, 100k images, +6k steps          (control)
#   adapt  + per-block rank-32 adapters, 100k, +6k        (option 1)
#   data   shared field, 100k + 218k images, +15k steps   (option 2)
#   both   adapters + more data + longer                  (options 1 and 2)
# Each arm is scored by `compile_flow.py final` with top-1/top-5 agreement, KL and accuracy (option 3).
#   bash run_fidelity.sh last4      # base + cont + adapt + data + both for the last 4 blocks
#   bash run_fidelity.sh all12_a    # base + cont + adapt for all 12 blocks
#   bash run_fidelity.sh all12_b    # extract train2, then data + both for all 12 blocks
set -uo pipefail
cd "$(dirname "$0")"
source /ephemeral/trinhpn_uq/env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True LADDER=imagenet COMPILE_DEPTH=6
LOGS=/ephemeral/trinhpn_uq/logs/fidelity
CK=/ephemeral/trinhpn_uq/ckpt
mkdir -p "$LOGS"

retry_gpu() {   # retry_gpu <log name> <command...>
  local name=$1; shift
  for attempt in $(seq 1 80); do
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

arm() {   # arm <tag> <K0> <init ckpt> <adapter rank> <data sets> <steps>
  export COMPILE_TAG=$1 COMPILE_FROM=$2 COMPILE_INIT=$3 COMPILE_ADAPTER_RANK=$4 COMPILE_DATA=$5
  export COMPILE_LR=2e-4 COMPILE_ROLLOUT_START=0
  retry_gpu train_$1 $PY compile_flow.py train $6 && retry_gpu final_$1 $PY compile_flow.py final
}

base_final() {   # base_final <tag> <K0>: score the existing ladder field with the same metrics
  export COMPILE_TAG=$1 COMPILE_FROM=$2 COMPILE_ADAPTER_RANK=0 COMPILE_INIT= COMPILE_DATA=train
  retry_gpu final_$1 $PY compile_flow.py final
}

wait_train2() { until [ -f /ephemeral/trinhpn_uq/features/train2_logits.npy ]; do sleep 60; done; }

case "$1" in
  last4)
    base_final ode8 8
    arm ode8_cont 8 $CK/ode8_flownet.pt 0 train 6000
    arm ode8_adapt 8 $CK/ode8_flownet.pt 32 train 6000
    wait_train2
    arm ode8_data 8 $CK/ode8_flownet.pt 0 train,train2 15000
    arm ode8_both 8 $CK/ode8_flownet.pt 32 train,train2 15000
    ;;
  all12_a)
    base_final ode 0
    arm ode_cont 0 $CK/ode_flownet.pt 0 train 6000
    arm ode_adapt 0 $CK/ode_flownet.pt 32 train 6000
    ;;
  all12_b)
    until grep -q "^done" /ephemeral/trinhpn_uq/logs/download_extra.log; do sleep 60; done
    retry_gpu extract_train2 $PY extract_features.py train2 || exit 1
    arm ode_data 0 $CK/ode_flownet.pt 0 train,train2 15000
    arm ode_both 0 $CK/ode_flownet.pt 32 train,train2 15000
    ;;
  *) echo "usage: run_fidelity.sh last4|all12_a|all12_b"; exit 2 ;;
esac
