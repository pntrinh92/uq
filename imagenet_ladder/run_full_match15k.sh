#!/usr/bin/env bash
# Step-matched arm for the data-versus-steps attribution: last 4 blocks, all 1.28M images, 15k steps with the cosine
# schedule over 15k steps, the same init and learning rate as the 318k-image arm (ode8_data). Run inside tmux "pnt".
set -uo pipefail
cd "$(dirname "$0")"
source /ephemeral/trinhpn_uq/env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True LADDER=imagenet COMPILE_DEPTH=6
LOGS=/ephemeral/trinhpn_uq/logs/full
CK=/ephemeral/trinhpn_uq/ckpt

retry_gpu() {   # retry_gpu <log name> <command...>
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

export COMPILE_TAG=ode8_full15k COMPILE_FROM=8 COMPILE_INIT=$CK/ode8_flownet.pt COMPILE_ADAPTER_RANK=0
export COMPILE_DATA=train,train2,train3a,train3b,train3c COMPILE_LR=2e-4 COMPILE_ROLLOUT_START=0
retry_gpu train_ode8_full15k $PY compile_flow.py train 15000 || exit 1
retry_gpu final_ode8_full15k $PY compile_flow.py final || exit 1
echo MATCH15K_DONE
