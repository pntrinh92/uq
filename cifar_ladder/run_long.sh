#!/usr/bin/env bash
# Longer last-4-block compile on CIFAR (2026-11-26). Same recipe as run_cifar.sh's `compile_chain 8 6000`
# (fresh field from ViT block 6, 45k fine-tune images, lr 5e-4, cosine over the run, rollout from step 1000,
# lam_KL 0.1), changing only the number of steps. Tag ode8_<N/1000>k, so the 6k `ode8` artifacts stay untouched.
#   bash run_long.sh cifar100 60000     # run inside tmux "pnt"
# Outputs (results_<ladder>/): compile_final_ode8.json (the 6k field, same metrics, for comparison),
# compile_final_<tag>[_stepN].json, compile_step_sweep_<tag>.json, ladder_<tag>.json.
# Snapshots at 1/4, 1/2, 3/4 of the run are mid-schedule (higher lr), so they are not equivalent to shorter runs.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source /ephemeral/trinhpn_uq/env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LADDER=${1:?usage: run_long.sh cifar10|cifar100 <steps>}
STEPS=${2:?usage: run_long.sh cifar10|cifar100 <steps>}
TAG=ode8_$((STEPS / 1000))k
LOGS=/ephemeral/trinhpn_uq/logs/$LADDER
CK=/ephemeral/trinhpn_uq/$LADDER/ckpt
mkdir -p "$LOGS"
OTHER=$([ "$LADDER" = "cifar10" ] && echo cifar100 || echo cifar10)
SETS="train val shift svhn $OTHER lsun dtd noise"
SNAPS="$((STEPS / 4)),$((STEPS / 2)),$((3 * STEPS / 4))"

retry_gpu() {   # retry_gpu <log name> <command...>   (as run_cifar.sh, more attempts: GPUs are shared)
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

cd "$HERE/../imagenet_ladder"
export COMPILE_FROM=8 COMPILE_DEPTH=6 COMPILE_ADAPTER_RANK=0 COMPILE_DATA=train COMPILE_INIT=""
export COMPILE_LR=5e-4 COMPILE_ROLLOUT_START=1000 COMPILE_LAM_KL=0.1

# Baseline: score the existing 6k field with the same final metrics (top-1/top-5 agreement, KL, accuracy).
[ -f "$HERE/results_$LADDER/compile_final_ode8.json" ] || COMPILE_TAG=ode8 retry_gpu final_ode8 $PY compile_flow.py final

export COMPILE_TAG=$TAG COMPILE_SNAPSHOTS=$SNAPS
retry_gpu train_$TAG $PY compile_flow.py train $STEPS || { echo LONG_FAILED; exit 1; }
for st in ${SNAPS//,/ }; do
  [ -f $CK/${TAG}_step${st}_flownet.pt ] && COMPILE_EVAL_CKPT=$CK/${TAG}_step${st}_flownet.pt retry_gpu final_${TAG}_$st $PY compile_flow.py final
done
retry_gpu final_$TAG $PY compile_flow.py final || { echo LONG_FAILED; exit 1; }
retry_gpu sweep_$TAG $PY compile_flow.py sweep || { echo LONG_FAILED; exit 1; }
retry_gpu rollout_$TAG $PY compile_flow.py rollout $SETS || { echo LONG_FAILED; exit 1; }
NUM_THREADS=32 $PY ladder_eval.py $TAG 2>&1 | grep --line-buffered -v -i warn | tee "$LOGS/ladder_$TAG.log"
[ "${PIPESTATUS[0]}" -eq 0 ] && echo LONG_DONE || echo LONG_FAILED
