#!/usr/bin/env bash
# Queue worker for the DIRECTOR-setting compiles (2026-11-26). Each worker pops "<loss> <ladder>" lines from the
# queue file under a lock and runs `run_director.sh <loss> <ladder>`, so a fast GPU takes more jobs.
#   bash run_director_queue.sh <queue file> <worker name>      # GPU_EXCLUDE in the environment pins the worker
# Started with QUEUE lines: "ours cifar10", "director cifar10", "ours cifar100", "director cifar100".
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
Q=${1:?usage: run_director_queue.sh <queue file> <worker name>}
W=${2:?usage: run_director_queue.sh <queue file> <worker name>}
while true; do
  job=$(flock "$Q.lock" bash -c 'l=$(head -n 1 "$1"); [ -n "$l" ] && sed -i 1d "$1"; echo "$l"' _ "$Q")
  [ -z "$job" ] && break
  echo "[$(date '+%m-%d %H:%M')] worker $W (GPU_EXCLUDE=${GPU_EXCLUDE:-3}) starts: $job"
  bash "$HERE/run_director.sh" $job
done
echo "WORKER_${W}_EXIT"
