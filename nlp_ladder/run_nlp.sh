#!/usr/bin/env bash
# NLP ladder driver (DIRECTOR text benchmarks, Qwen2.5-0.5B). Run inside tmux session "pnt".
#   bash run_nlp.sh gpu <task>     # 5 seeds: fine-tune, features, dropout baseline; then ensemble caches
#   bash run_nlp.sh cpu <task>     # per seed: ladder (FM backbone), partitioned regeneration, conformal scaffold
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
LADDER_DIR="$HERE/../imagenet_ladder"
source /ephemeral/trinhpn_uq/env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export LADDER=${2:?usage: run_nlp.sh gpu|cpu imdb|cola|sst2}
LOGS=/ephemeral/trinhpn_uq/logs/nlp/$LADDER
mkdir -p "$LOGS"

retry_gpu() {   # retry_gpu <log name> <command...>
  local name=$1; shift
  for attempt in $(seq 1 200); do
    echo "[$(date +%H:%M:%S)] $LADDER $name attempt $attempt"
    NUM_THREADS=8 "$@" 2>&1 | grep --line-buffered -v -i "warn" | tee -a "$LOGS/$name.log"
    local rc=${PIPESTATUS[0]}
    if [ "$rc" -eq 0 ]; then echo "[$(date +%H:%M:%S)] $LADDER $name OK"; return 0; fi
    if tail -n 40 "$LOGS/$name.log" | grep -qE "OutOfMemoryError|no GPU with|CUDA error: out of memory|CUBLAS_STATUS_ALLOC_FAILED"; then
      echo "[$(date +%H:%M:%S)] $name: GPU memory busy, retrying in 120 s"; sleep 120
    else
      echo "[$(date +%H:%M:%S)] $name FAILED (rc=$rc), see $LOGS/$name.log"; return 1
    fi
  done
  return 1
}

case "$1" in
  gpu)
    cd "$HERE"
    for s in 0 1 2 3 4; do
      export NLP_SEED=$s
      CK=/ephemeral/trinhpn_uq/nlp/$LADDER/seed$s/ckpt
      [ -f $CK/head.pt ] || retry_gpu finetune_s$s $PY nlp_prepare.py finetune || exit 1
      [ -f /ephemeral/trinhpn_uq/nlp/$LADDER/seed$s/features/noise_z.npy ] || retry_gpu extract_s$s $PY nlp_prepare.py extract || exit 1
      [ -f "$HERE/results_$LADDER/seed$s/dropout_selection.json" ] || retry_gpu dropout_s$s $PY nlp_prepare.py dropout || exit 1
    done
    for s in 0 1 2 3 4; do NLP_SEED=$s $PY nlp_prepare.py ensemble; done
    echo "ENSEMBLE_OK" > "$LOGS/gpu_done"
    ;;
  cpu)
    until [ -f "$LOGS/gpu_done" ]; do sleep 60; done
    cd "$LADDER_DIR"
    for s in 0 1 2 3 4; do
      export NLP_SEED=$s
      NUM_THREADS=16 $PY ladder_eval.py fm 2>&1 | grep --line-buffered -v -i warn | tee "$LOGS/ladder_fm_s$s.log"
      NUM_THREADS=16 $PY partitioned_regen.py fm 2>&1 | grep --line-buffered -v -i warn | tee "$LOGS/partitioned_fm_s$s.log"
      [ -f conformal_scaffold.py ] && NUM_THREADS=16 $PY conformal_scaffold.py fm 2>&1 | grep --line-buffered -v -i warn | tee "$LOGS/conformal_fm_s$s.log"
    done
    ;;
  *) echo "usage: run_nlp.sh gpu|cpu <task>"; exit 2 ;;
esac
