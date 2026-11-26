#!/bin/bash
# Re-queue the 13 CIFAR jobs that deadlocked on 2026-11-02 13:47 (DataLoader workers stuck because /tmp on the
# full root disk; they started before sitecustomize set TMPDIR=/ephemeral/trinhpn_uq/tmp).
# Kills them by PID (never pkill -f), then deletes the .fail markers the runner writes for them and the
# "dependency failed" markers it cascades to their dependents, so everything goes back to pending.
set -u
cd /ephemeral/trinhpn_uq/director_repro
HUNG="200302 210398 609985 785322 794976 1017217 1286653 1847698 2344556 2449287 2766096 3006825 3375192"
touch state/.kill_stamp
for p in $HUNG; do
  # only touch processes that are still children of the runner
  [ "$(ps -o ppid= -p $p 2>/dev/null | tr -d ' ')" = "131803" ] || { echo "skip $p (not a runner child)"; continue; }
  kill $p $(pgrep -P $p)
done
sleep 20
for p in $HUNG; do kill -0 $p 2>/dev/null && kill -9 $p $(pgrep -P $p) 2>/dev/null; done
sleep 70   # give the runner (30 s loop) time to reap them and cascade the skips
for i in 1 2 3; do
  find state -name '*.fail' -newer state/.kill_stamp -print -delete
  sleep 35
done
echo "re-queued; watch: tail -f runner.log"
