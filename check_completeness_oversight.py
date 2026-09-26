"""
check_completeness_oversight.py
----------------------------------
Verifies a results CSV from this experiment has exactly one row per
(architecture, network_size, training_timesteps, seed, condition) --
mirrors the original project's hard-learned lesson: a raw row count
matching expectations is NOT enough to confirm completeness, since a
duplicate cell can silently offset a missing one. Always use this (or
the equivalent groupby check) before trusting any results file,
especially after a run that appended to an existing file.

Usage:
    python check_completeness_oversight.py results/results_oversight_mini_test.csv
    python check_completeness_oversight.py results/results_oversight.csv   # the real sweep, once it exists
"""

import sys
import pandas as pd

KEY_COLS = ["architecture", "network_size", "training_timesteps", "seed", "condition"]


def check(path):
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows from {path}")

    counts = df.groupby(KEY_COLS).size()
    duplicates = counts[counts > 1]
    if len(duplicates):
        print(f"\nFOUND {len(duplicates)} DUPLICATED key(s) -- these rows appear more "
              f"than once, which can silently mask a missing cell elsewhere even when "
              f"the total row count looks fine:")
        print(duplicates)
    else:
        print("\nNo duplicate keys found -- every (architecture, network_size, "
              "training_timesteps, seed, condition) combination appears exactly once.")

    print(f"\nUnique cells present: {len(counts)}")
    print("Architectures present:", sorted(df["architecture"].unique()))
    print("Seeds present:", sorted(df["seed"].unique()))
    print("Conditions present:", sorted(df["condition"].unique()))
    print("Network sizes present:", sorted(df["network_size"].unique()))
    print("Training budgets present:", sorted(df["training_timesteps"].unique()))

    # Basic sanity range checks -- not a substitute for reading the actual
    # numbers, but catches obviously broken data (NaN, out-of-range values).
    for col in ("success_rate", "violation_rate"):
        bad = df[(df[col] < 0) | (df[col] > 1) | df[col].isna()]
        if len(bad):
            print(f"\nWARNING: {len(bad)} rows have an out-of-range or NaN {col}:")
            print(bad[KEY_COLS + [col]])
        else:
            print(f"\n{col}: all values in [0, 1], no NaNs -- OK.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python check_completeness_oversight.py <path_to_csv>")
    check(sys.argv[1])
