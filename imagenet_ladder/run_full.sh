#!/usr/bin/env bash
# Full ImageNet-1k train set (1,281,167 images) for the compiled flow. Run inside tmux "pnt".
#   bash run_full.sh extract           # x0 caches of shards 73-293 (train3a/b/c); shards 0-72 are cached already
#   bash run_full.sh last4 [steps]     # ode8_full: continue the ladder field of the last 4 blocks on all images
#   bash run_full.sh all12 [steps]     # ode_full: the same for all 12 blocks
# Each compile keeps snapshots at 15k/30k/45k steps; every snapshot and the final field are scored by
# `compile_flow.py final` (top-1/top-5 agreement, KL, accuracy on 10k test images, as run_fidelity.sh),
# then the final field gets rollouts and the full ladder (ladder_eval, partitioned and conformal scaffolds).
set -uo pipefail
cd "$(dirname "$0")"
source /ephemeral/trinhpn_uq/env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True LADDER=imagenet COMPILE_DEPTH=6
LOGS=/ephemeral/trinhpn_uq/logs/full
CK=/ephemeral/trinhpn_uq/ckpt
mkdir -p "$LOGS"

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

full_chain() {   # full_chain <tag> <K0> <init ckpt> <steps>
  local tag=$1
  export COMPILE_TAG=$tag COMPILE_FROM=$2 COMPILE_INIT=$3 COMPILE_ADAPTER_RANK=0
  export COMPILE_DATA=train,train2,train3a,train3b,train3c COMPILE_LR=2e-4 COMPILE_ROLLOUT_START=0
  export COMPILE_SNAPSHOTS=15000,30000,45000
  retry_gpu train_$tag $PY compile_flow.py train $4 || return 1
  for st in 15000 30000 45000; do
    [ -f $CK/${tag}_step${st}_flownet.pt ] && COMPILE_EVAL_CKPT=$CK/${tag}_step${st}_flownet.pt retry_gpu final_${tag}_$st $PY compile_flow.py final
  done
  retry_gpu final_$tag $PY compile_flow.py final || return 1
  retry_gpu sweep_$tag $PY compile_flow.py sweep || return 1
  retry_gpu rollout_$tag $PY compile_flow.py rollout train val shift imagenet_o dtd noise || return 1
  NUM_THREADS=32 $PY ladder_eval.py $tag 2>&1 | grep --line-buffered -v -i warn | tee "$LOGS/ladder_$tag.log"
  NUM_THREADS=32 $PY partitioned_regen.py $tag 2>&1 | grep --line-buffered -v -i warn | tee "$LOGS/partitioned_$tag.log"
  NUM_THREADS=32 $PY conformal_scaffold.py $tag 2>&1 | grep --line-buffered -v -i warn | tee "$LOGS/conformal_$tag.log"
}

case "$1" in
  extract)
    until [ "$(ls /ephemeral/trinhpn_uq/imagenet_full/data/*.parquet 2>/dev/null | wc -l)" -ge 221 ]; do sleep 120; done
    for part in train3a train3b train3c; do
      [ -f /ephemeral/trinhpn_uq/features/${part}_logits.npy ] || retry_gpu extract_$part $PY extract_features.py $part || exit 1
    done
    ;;
  last4)
    until [ -f /ephemeral/trinhpn_uq/features/train3c_logits.npy ]; do sleep 120; done
    full_chain ode8_full 8 $CK/ode8_flownet.pt ${2:-60000} ;;
  all12)
    until [ -f /ephemeral/trinhpn_uq/features/train3c_logits.npy ]; do sleep 120; done
    full_chain ode_full 0 $CK/ode_flownet.pt ${2:-60000} ;;
  *) echo "usage: run_full.sh extract|last4|all12 [steps]"; exit 2 ;;
esac
