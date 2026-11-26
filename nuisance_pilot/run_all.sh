#!/bin/bash
# Pilot 2 over SST-2 and CoLA x 5 seeds: extract (GPU, skipped if done), then eval (CPU).
source /ephemeral/trinhpn_uq/env.sh
cd "$(dirname "$0")"
for t in cola sst2; do
  for s in 0 1 2 3 4; do
    d=/ephemeral/trinhpn_uq/nuisance_pilot2/$t/seed$s
    [ -f $d/EXTRACT_DONE ] || LADDER=$t NLP_SEED=$s $PY pilot2.py extract 2>&1 | grep -v -i "warning\|Loading weights\|deprecated" | tail -3
    LADDER=$t NLP_SEED=$s OMP_NUM_THREADS=32 $PY pilot2.py eval 2>&1 | grep -v -i warning > $d/eval.log
    echo "done $t s$s"; grep "^==\|maha\|msp\|energy" $d/eval.log | head -30
  done
done
echo ALL_DONE
