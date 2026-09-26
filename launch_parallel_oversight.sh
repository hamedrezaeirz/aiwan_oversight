#!/bin/bash
# launch_parallel_oversight.sh
# ------------------------------
# Splits config_oversight.SEEDS = [0..19] across 2 processes (10 seeds
# each), matching this project's confirmed decision to use only 2 of
# this VPS's 4 vCPUs (leaving headroom, per the user's own stated
# concern about quota/throttling issues with heavier CPU usage on this
# provider). Each process writes its own
# results/results_oversight_seeds_<...>.csv, so there's no write
# conflict -- merge afterwards with merge_results_oversight.py.
#
# Based on time_probe.py's real measurements on this VPS (net_size=[16]:
# 884 steps/sec, [256]: 843, [2048]: 603 steps/sec), the full 9-cell
# grid's ~21 trainings per cell comes to roughly 14.2 hours per seed;
# with 10 seeds per process running sequentially, that's roughly 142
# hours (~6 days) per process. Both processes run in parallel, so total
# wall-clock time should be roughly 6 days, not 12 -- still, run this
# inside tmux/screen or with nohup so it survives an SSH disconnect,
# and expect this to take several days, not hours.
#
#   tmux new -s aiwan_oversight
#   bash launch_parallel_oversight.sh
#   [Ctrl+b, d to detach -- reattach later with: tmux attach -t aiwan_oversight]
#
# or:
#
#   nohup bash launch_parallel_oversight.sh > launch.log 2>&1 &
#
# Check progress any time with:
#   tail -f logs/seeds_0-9.log      (and logs/seeds_10-19.log)
#   wc -l results/results_oversight_seeds_*.csv   (row counts so far, per process)
#
# IMPORTANT: per this project's own hard-learned lesson (see
# HANDOFF_STATUS_v4.md), a matching row COUNT after the run finishes is
# NOT sufficient to confirm completeness -- always run
# check_completeness_oversight.py (after merging) before trusting the
# results.

set -e
cd "$(dirname "$0")"
source venv/bin/activate
mkdir -p results logs

echo "Starting 2 parallel processes across seeds 0-19 (10 seeds each)..."

python3 -u run_experiment_oversight.py --seeds 0 1 2 3 4 5 6 7 8 9 \
    > logs/seeds_0-9.log 2>&1 &
PID1=$!
python3 -u run_experiment_oversight.py --seeds 10 11 12 13 14 15 16 17 18 19 \
    > logs/seeds_10-19.log 2>&1 &
PID2=$!

echo "Launched PIDs: $PID1 $PID2"
echo "Waiting for both to finish (this is the ~multi-day part)..."

wait $PID1 $PID2

echo "Both processes finished. Run: python3 merge_results_oversight.py"
echo "Then verify completeness: python3 check_completeness_oversight.py results/results_oversight.csv"
