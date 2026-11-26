#!/usr/bin/env bash
# Rebuild SST-2 after fixing the last-token format (nlp_prepare.build_data). The fine-tuned checkpoints are kept:
# tokenize, then for 5 seeds features, dropout, ensemble, ladder, scaffolds, DIRECTOR metrics and the OOD flag.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source /ephemeral/trinhpn_uq/env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_DISABLE_IMPLICIT_TOKEN=1 LADDER=sst2
LOGS=/ephemeral/trinhpn_uq/logs/nlp/sst2_fix
mkdir -p "$LOGS"

retry_gpu() {   # retry_gpu <log name> <command...>
  local name=$1; shift
  for attempt in $(seq 1 200); do
    NUM_THREADS=8 "$@" 2>&1 | grep --line-buffered -v -i "warn" | tee -a "$LOGS/$name.log"
    local rc=${PIPESTATUS[0]}
    if [ "$rc" -eq 0 ]; then echo "[$(date +%H:%M:%S)] $name OK"; return 0; fi
    if tail -n 40 "$LOGS/$name.log" | grep -qE "OutOfMemoryError|no GPU with|CUDA error: out of memory|CUBLAS_STATUS_ALLOC_FAILED"; then
      sleep 120
    else
      echo "[$(date +%H:%M:%S)] $name FAILED (rc=$rc), see $LOGS/$name.log"; return 1
    fi
  done
  return 1
}

cd "$HERE"
$PY nlp_prepare.py data 2>&1 | grep -v -i warn | tee "$LOGS/data.log" || exit 1
for s in 0 1 2 3 4; do
  export NLP_SEED=$s
  retry_gpu extract_s$s $PY nlp_prepare.py extract || exit 1
  retry_gpu dropout_s$s $PY nlp_prepare.py dropout || exit 1
done
for s in 0 1 2 3 4; do NLP_SEED=$s $PY nlp_prepare.py ensemble; done

cd "$HERE/../imagenet_ladder"
for s in 0 1 2 3 4; do
  export NLP_SEED=$s
  export NUM_THREADS=16
  $PY ladder_eval.py fm 2>&1 | grep -v -i warn > "$LOGS/ladder_s$s.log" || exit 1
  $PY partitioned_regen.py fm 2>&1 | grep -v -i warn > "$LOGS/partitioned_s$s.log" || exit 1
  $PY conformal_scaffold.py fm 2>&1 | grep -v -i warn > "$LOGS/conformal_s$s.log" || exit 1
  $PY director_eval.py eval 2>&1 | grep -v -i warn | tail -1
  for base in fm ts; do
    LTT_MODE=fst SCAFFOLD_BASE=$base $PY conformal_scaffold.py fm 2>&1 | grep -v -i warn > "$LOGS/conformal_s${s}_$base.log" || exit 1
    grep "^base=" "$LOGS/conformal_s${s}_$base.log"
  done
  $PY conformal_ood.py 2>&1 | grep -v -i warn | tail -7
done
$PY director_eval.py table > /dev/null
$PY conformal_ood.py table > /dev/null
echo SST2_FIX_DONE
