#!/usr/bin/env bash
# Reproduce every pilot experiment and figure. CPU only; ~1-2 h on a busy shared box.
# Set PY to a Python with torch, scikit-learn, scipy and matplotlib.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-/data/nvidia/conda_envs/dailam/bin/python}
export NUM_THREADS=${NUM_THREADS:-8}

# 0. Train and cache 5 splits x 5 ensemble members (member 0 is "the FM").
for s in 0 1 2 3 4; do $PY -c "from common import get_models; get_models($s)" & done; wait

# Analytic / toy checks (no trained model needed).
$PY q2_drift_correction.py > results/q2.log
$PY q3_eq6_cost.py > results/q3.log

# Checks on the trained digits ViT.
$PY exp0_variance_budget.py > results/exp0.log
$PY q4_eq8_penalty.py > results/q4.log
for s in 0 1 2 3 4; do
  $PY exp1_langevin_sweep.py final $s > results/exp1_final_split$s.log &
  $PY exp1_langevin_sweep.py all $s > results/exp1_all_split$s.log &
done; wait
$PY exp1_report.py > results/exp1_report.log

# Proposed regeneration scaffold, then the report figures (figures4papers house style).
$PY exp3_regeneration.py > results/exp3.log
(cd ../../figures4papers/figure_Thrust1UQ && $PY plot_report.py)
