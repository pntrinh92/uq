#!/usr/bin/env bash
# Score an already-trained last-4 (K0 = 8) compiled field with the same post-training chain as run_full.sh's
# full_chain: Euler-step sweep, rollouts of every set, then the ladder (ladder_eval, partitioned and conformal
# scaffolds). Run inside tmux "pnt".
#   bash run_score.sh ode8_full120k      # 2026-11-26: Arm A of run_arms.sh (120k steps)
# GPU stages use pick_gpu, which skips GPU 3 (GPU_EXCLUDE, default "3").
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

tag=${1:?usage: run_score.sh <tag>}
[ -f $CK/${tag}_flownet.pt ] || { echo "missing $CK/${tag}_flownet.pt"; exit 1; }
export COMPILE_TAG=$tag COMPILE_FROM=8 COMPILE_ADAPTER_RANK=0
retry_gpu sweep_$tag $PY compile_flow.py sweep || exit 1
retry_gpu rollout_$tag $PY compile_flow.py rollout train val shift imagenet_o dtd noise || exit 1
NUM_THREADS=32 $PY ladder_eval.py $tag 2>&1 | grep --line-buffered -v -i warn | tee "$LOGS/ladder_$tag.log"
NUM_THREADS=32 $PY partitioned_regen.py $tag 2>&1 | grep --line-buffered -v -i warn | tee "$LOGS/partitioned_$tag.log"
NUM_THREADS=32 $PY conformal_scaffold.py $tag 2>&1 | grep --line-buffered -v -i warn | tee "$LOGS/conformal_$tag.log"
echo "SCORE_${tag}_DONE"
