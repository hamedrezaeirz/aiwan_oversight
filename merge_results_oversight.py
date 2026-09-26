"""
merge_results_oversight.py
-----------------------------
Merges every results/results_oversight_seeds_*.csv produced by a
parallel `--seeds` run (see launch_parallel_oversight.sh) into the
canonical results/results_oversight.csv. Mirrors the original project's
merge_results.py.

Usage (after both launch_parallel_oversight.sh processes have finished):
    python3 merge_results_oversight.py

IMPORTANT: this does NOT deduplicate or verify completeness -- run
check_completeness_oversight.py on the merged file afterward, always.
A matching total row count is NOT sufficient on its own (see this
project's own hard-learned lesson, recorded in HANDOFF_STATUS_v4.md): a
duplicate cell can silently mask a missing one.
"""

import glob
import pandas as pd

from config_oversight import RESULTS_DIR, RESULTS_CSV_PATH

if __name__ == "__main__":
    files = sorted(glob.glob(f"{RESULTS_DIR}/results_oversight_seeds_*.csv"))
    if not files:
        raise SystemExit(
            f"No results_oversight_seeds_*.csv files found in {RESULTS_DIR}/ -- "
            f"nothing to merge yet."
        )
    print(f"Merging {len(files)} files:")
    for f in files:
        print(f"  {f}")

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df.to_csv(RESULTS_CSV_PATH, index=False)
    print(f"\nWrote {len(df)} total rows to {RESULTS_CSV_PATH}")
    print(f"\nNow run: python3 check_completeness_oversight.py {RESULTS_CSV_PATH}")
