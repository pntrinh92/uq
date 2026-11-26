#!/usr/bin/env bash
# Copy the project files that live outside this folder into the repo before committing:
#   director_repro_runner/  <- /ephemeral/trinhpn_uq/director_repro (runner, job list, aggregation, result tables;
#                              not the TransDiff clone, the per-job state/logs or the paper's arXiv source)
#   figures/figure_Thrust1UQ <- ../figures4papers/figure_Thrust1UQ (plot scripts and figures; figures4papers itself
#                              is a clone of github.com/ChenLiu-1996/figures4papers)
set -euo pipefail
cd "$(dirname "$0")"
SRC=/ephemeral/trinhpn_uq/director_repro
mkdir -p director_repro_runner figures
rsync -a --delete --exclude '__pycache__' --exclude '*.bak_*' \
  --include 'gen_jobs.py' --include 'runner.py' --include 'aggregate.py' --include 'requeue_hung.sh' \
  --include 'jobs.json' --include 'tables.md' --include 'summary.json' --exclude '*' \
  "$SRC/" director_repro_runner/
rsync -a --delete --exclude '__pycache__' ../figures4papers/figure_Thrust1UQ/ figures/figure_Thrust1UQ/
echo "synced: $(ls director_repro_runner | wc -l) runner files, $(find figures -type f | wc -l) figure files"
