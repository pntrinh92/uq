#!/usr/bin/env bash
# CIFAR ladder driver (DIRECTOR setting). Run inside tmux session "pnt".
#   bash run_cifar.sh gpu cifar10    # fine-tune, features, dropout baseline, compiles (last 4, all 12) + their evaluation
#   bash run_cifar.sh cpu cifar10    # FM-backbone evaluation and partitioned regeneration (after the dropout baseline)
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
LADDER_DIR="$HERE/../imagenet_ladder"
source /ephemeral/trinhpn_uq/env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LADDER=${2:?usage: run_cifar.sh gpu|cpu cifar10|cifar100}
LOGS=/ephemeral/trinhpn_uq/logs/$LADDER
RES="$HERE/results_$LADDER"
mkdir -p "$LOGS" "$RES"
OTHER=$([ "$LADDER" = "cifar10" ] && echo cifar100 || echo cifar10)
SETS="train val shift svhn $OTHER lsun dtd noise"
CTAR=$([ "$LADDER" = "cifar10" ] && echo /ephemeral/trinhpn_uq/logs/c10c.log || echo /ephemeral/trinhpn_uq/logs/c100c.log)
CTAG=$([ "$LADDER" = "cifar10" ] && echo C10C_OK || echo C100C_OK)

retry_gpu() {   # retry_gpu <log name> <command...>
  local name=$1; shift
  for attempt in $(seq 1 60); do
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

compile_chain() {   # compile_chain <K0> <steps>
  export COMPILE_FROM=$1 COMPILE_DEPTH=6
  local tag=$([ "$1" = "0" ] && echo ode || echo "ode$1")
  cd "$LADDER_DIR"
  retry_gpu compile_train_$tag $PY compile_flow.py train $2 || return 1
  retry_gpu compile_sweep_$tag $PY compile_flow.py sweep || return 1
  retry_gpu compile_rollout_$tag $PY compile_flow.py rollout $SETS || return 1
  NUM_THREADS=32 $PY ladder_eval.py $tag 2>&1 | grep --line-buffered -v -i warn | tee "$LOGS/ladder_$tag.log"
}

case "$1" in
  gpu)
    cd "$HERE"
    [ -f /ephemeral/trinhpn_uq/$LADDER/ckpt/finetuned.pt ] || retry_gpu finetune $PY cifar_prepare.py finetune 3 || exit 1
    retry_gpu extract $PY cifar_prepare.py extract val train svhn $OTHER lsun dtd noise || exit 1
    until grep -q "$CTAG" "$CTAR" 2>/dev/null; do sleep 60; done
    retry_gpu extract_shift $PY cifar_prepare.py extract shift || exit 1
    cd "$LADDER_DIR" && retry_gpu dropout $PY dropout_baseline.py || exit 1
    compile_chain 8 6000
    compile_chain 0 10000
    ;;
  cpu)
    until [ -f "$RES/dropout_selection.json" ]; do sleep 60; done
    cd "$LADDER_DIR"
    NUM_THREADS=32 $PY ladder_eval.py fm 2>&1 | grep --line-buffered -v -i warn | tee "$LOGS/ladder_fm.log"
    NUM_THREADS=32 $PY partitioned_regen.py fm 2>&1 | grep --line-buffered -v -i warn | tee "$LOGS/partitioned_fm.log"
    ;;
  *) echo "usage: run_cifar.sh gpu|cpu cifar10|cifar100"; exit 2 ;;
esac
