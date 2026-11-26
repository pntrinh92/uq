#!/usr/bin/env bash
# ImageNet ablation ladder driver. Meant to run inside tmux (session "pnt").
#   bash run_ladder.sh gpu K0 STEPS  # rung 2: compile (K0=0 full, K0=8 last 4 blocks), sweep, rollouts, ODE-backbone eval
#   bash run_ladder.sh fm       # FM-backbone evaluation (rungs 1, 3, 4 and baselines; CPU only)
# GPU stages retry when the shared GPUs are full; compile training resumes from its last state.
set -uo pipefail
cd "$(dirname "$0")"
source /ephemeral/trinhpn_uq/env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOGS=/ephemeral/trinhpn_uq/logs

retry_gpu() {   # retry_gpu <log name> <command...>
  local name=$1; shift
  for attempt in $(seq 1 60); do
    echo "[$(date +%H:%M:%S)] $name attempt $attempt"
    NUM_THREADS=8 "$@" 2>&1 | grep --line-buffered -v -i "warn" | tee -a "$LOGS/$name.log"
    local rc=${PIPESTATUS[0]}
    if [ "$rc" -eq 0 ]; then echo "[$(date +%H:%M:%S)] $name OK"; return 0; fi
    if tail -n 40 "$LOGS/$name.log" | grep -qE "OutOfMemoryError|no GPU with|CUDA error: out of memory"; then
      echo "[$(date +%H:%M:%S)] $name: GPU memory busy, retrying in 120 s"; sleep 120
    else
      echo "[$(date +%H:%M:%S)] $name FAILED (rc=$rc), see $LOGS/$name.log"; return 1
    fi
  done
  return 1
}

wait_for_dropout() {
  # The baseline writes its selection json last; wait for that file (a process-name check
  # also matches other shells on this shared account).
  until [ -f results/dropout_selection.json ]; do sleep 30; done
}

case "${1:-}" in
  gpu)   # bash run_ladder.sh gpu <K0: 0 = compile all 12 blocks, 8 = compile blocks 9..12> <steps>
    K0=${2:-0}; STEPS=${3:-15000}; export COMPILE_FROM=$K0 COMPILE_DEPTH=6
    TAG=$([ "$K0" = "0" ] && echo ode || echo "ode$K0")
    retry_gpu compile_train_$TAG $PY compile_flow.py train $STEPS || exit 1
    retry_gpu compile_sweep_$TAG $PY compile_flow.py sweep || exit 1
    retry_gpu compile_rollout_$TAG $PY compile_flow.py rollout train val shift imagenet_o dtd noise || exit 1
    NUM_THREADS=32 $PY ladder_eval.py $TAG 2>&1 | grep --line-buffered -v -i warn | tee "$LOGS/ladder_$TAG.log"
    ;;
  fm)
    wait_for_dropout || echo "dropout baseline missing; FM evaluation runs without it"
    NUM_THREADS=32 $PY ladder_eval.py fm 2>&1 | grep --line-buffered -v -i warn | tee "$LOGS/ladder_fm.log"
    ;;
  *)
    echo "usage: run_ladder.sh gpu|fm"; exit 2 ;;
esac
