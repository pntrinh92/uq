#!/usr/bin/env bash
# Like-for-like comparison in DIRECTOR's metrics (director_eval.py). CPU only. Run inside tmux "pnt".
#   bash run_director.sh vision    # imagenet, cifar100, cifar10
#   bash run_director.sh nlp       # cola, imdb, sst2, 5 seeds each, waiting for each task's GPU stage
# Each ladder / seed: baselines (FM, TS, MCD, ensemble), then the certified scaffold on the raw FM and on
# the A1-tempered FM; the markdown tables are rebuilt after every step.
set -uo pipefail
cd "$(dirname "$0")"
source /ephemeral/trinhpn_uq/env.sh
LOGS=/ephemeral/trinhpn_uq/logs/director
mkdir -p "$LOGS"

one() {   # one <tag>   (LADDER and NLP_SEED already exported)
  local tag=$1
  NUM_THREADS=16 $PY director_eval.py eval 2>&1 | grep -v -i warn | tee "$LOGS/eval_$tag.log" | tail -1
  for base in fm ts; do
    LTT_MODE=fst SCAFFOLD_BASE=$base NUM_THREADS=16 $PY conformal_scaffold.py fm 2>&1 | grep -v -i warn \
      > "$LOGS/conformal_${tag}_$base.log"
    grep "^base=" "$LOGS/conformal_${tag}_$base.log"
  done
  $PY director_eval.py table > /dev/null
}

case "$1" in
  vision)
    for L in cifar100 cifar10 imagenet; do export LADDER=$L; unset NLP_SEED; one $L; done ;;
  nlp)
    for T in cola imdb sst2; do
      until [ -f /ephemeral/trinhpn_uq/logs/nlp/$T/gpu_done ]; do sleep 120; done
      for s in 0 1 2 3 4; do export LADDER=$T NLP_SEED=$s; one ${T}_s$s; done
    done ;;
  *) echo "usage: run_director.sh vision|nlp"; exit 2 ;;
esac
echo "DIRECTOR comparison $1 done"
