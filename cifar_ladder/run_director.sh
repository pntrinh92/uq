#!/usr/bin/env bash
# All-12-block compile of ViT-B/16 on CIFAR under DIRECTOR's training setting (2026-11-26); see compile_director.py.
#   bash run_director.sh <director|ours> <cifar10|cifar100> [...]    # run inside tmux "pnt"
# Tags: ode_dset_dirloss (DIRECTOR's loss: 0.8 mean matching + 0.2 label CE) and ode_dset_ourloss (our label-free
# objective), same budget, data, augmentation, optimizer, schedule and best-held-out-accuracy selection.
# Per dataset: train -> compile_flow.py final (10k test: agreement, top-5, KL, acc) -> sweep -> rollout -> ladder_eval.
# Prints DSET_<loss>_<ladder>_DONE / _FAILED per dataset.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source /ephemeral/trinhpn_uq/env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True COMPILE_FROM=0 COMPILE_DEPTH=6 COMPILE_ADAPTER_RANK=0
export COMPILE_DATA=train COMPILE_INIT="" COMPILE_LAM_KL=0.1 COMPILE_ROLLOUT_START=1000
LOSS=${1:?usage: run_director.sh director|ours cifar10|cifar100 ...}; shift
export DIR_LOSS=$LOSS
TAG=ode_dset_$([ "$LOSS" = "director" ] && echo dirloss || echo ourloss)
export COMPILE_TAG=$TAG

retry_gpu() {   # retry_gpu <log name> <command...>   (as run_cifar.sh; resumes from the state file after OOM)
  local name=$1; shift
  for attempt in $(seq 1 300); do
    echo "[$(date +%H:%M:%S)] $LADDER $name attempt $attempt"
    NUM_THREADS=8 "$@" 2>&1 | grep --line-buffered -v -i "warn" | tee -a "$LOGS/$name.log"
    local rc=${PIPESTATUS[0]}
    if [ "$rc" -eq 0 ]; then echo "[$(date +%H:%M:%S)] $LADDER $name OK"; return 0; fi
    if tail -n 40 "$LOGS/$name.log" | grep -qE "OutOfMemoryError|no GPU with|CUDA error: out of memory"; then
      echo "[$(date +%H:%M:%S)] $name: GPU memory busy, retrying in 120 s"; sleep 120
    else
      echo "[$(date +%H:%M:%S)] $name FAILED (rc=$rc), see $LOGS/$name.log"; return 1
    fi
  done
  return 1
}

one() {   # one <ladder>
  export LADDER=$1
  LOGS=/ephemeral/trinhpn_uq/logs/$LADDER
  mkdir -p "$LOGS"
  local other=$([ "$LADDER" = "cifar10" ] && echo cifar100 || echo cifar10)
  cd "$HERE"
  retry_gpu train_$TAG $PY compile_director.py train || return 1
  cd "$HERE/../imagenet_ladder"
  retry_gpu final_$TAG $PY compile_flow.py final || return 1
  COMPILE_EVAL_CKPT=/ephemeral/trinhpn_uq/$LADDER/ckpt/${TAG}_last_flownet.pt retry_gpu final_${TAG}_last $PY compile_flow.py final
  retry_gpu sweep_$TAG $PY compile_flow.py sweep || return 1
  retry_gpu rollout_$TAG $PY compile_flow.py rollout train val shift svhn $other lsun dtd noise || return 1
  NUM_THREADS=32 $PY ladder_eval.py $TAG 2>&1 | grep --line-buffered -v -i warn | tee "$LOGS/ladder_$TAG.log"
  return "${PIPESTATUS[0]}"
}

for L in "$@"; do
  if one "$L"; then echo "DSET_${LOSS}_${L}_DONE"; else echo "DSET_${LOSS}_${L}_FAILED"; fi
done
