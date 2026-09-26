"""
plot_results_oversight.py
----------------------------
Generates the final figures + statistical tests for the oversight
experiment, from results/results_oversight.csv (the merged output of
merge_results_oversight.py). Mirrors the original project's
plot_gate_results.py structure closely, adapted to this experiment's
violation_rate terminology, 9-cell grid (3 network sizes x 3 training
budgets), and 20 seeds.

Produces:
  - Figure O1: violation-rate gap (A - C) vs. network size and vs.
    training budget, at the highest override weight (the condition
    where "resistance under pressure" is actually tested).
  - Figure O2: violation rate vs. override weight, A/B/C, at the
    largest (network_size, budget) cell -- shows the erosion pattern
    directly.
  - Paired t-tests for the gap's collapse across budget and across
    network size, mirroring the original project's exact statistical
    approach (so the numbers here are directly comparable in kind to
    the original §7.3 report).

Deduplicates and verifies completeness FIRST (via the same logic as
check_completeness_oversight.py) before trusting the data enough to
plot it -- a duplicate row can silently inflate or distort an average
even if the plot itself looks reasonable, exactly this project's own
repeated lesson.

Usage:
    python3 plot_results_oversight.py [path_to_csv]
    (defaults to results/results_oversight.csv)

Requires: pandas, matplotlib, scipy, numpy
"""

import sys
import ast
import itertools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

DEFAULT_CSV_PATH = "results/results_oversight.csv"
NETWORK_SIZES = [16, 256, 2048]
TRAINING_BUDGETS = [10_000, 100_000, 500_000]
SEEDS = list(range(20))
OVERRIDE_WEIGHTS = [0, 1, 2, 4, 8]


def load_deduped(path):
    df = pd.read_csv(path)
    df["net"] = df["network_size"].apply(lambda x: ast.literal_eval(x)[0])

    key_cols = ["architecture", "net", "training_timesteps", "seed", "condition"]
    before = len(df)
    df = df.drop_duplicates(subset=key_cols, keep="first")
    removed = before - len(df)
    if removed:
        print(f"Removed {removed} duplicate rows (same architecture/net/budget/"
              f"seed/condition key, kept the first). This should be rare -- if "
              f"it's not zero, double-check how the merge happened.")

    expected_cells = set(itertools.product(NETWORK_SIZES, TRAINING_BUDGETS))
    missing = []
    for seed in SEEDS:
        present = set(df[df.seed == seed].groupby(["net", "training_timesteps"]).size().index)
        gap = expected_cells - present
        if gap:
            missing.append((seed, gap))
    if missing:
        raise SystemExit(f"Incomplete data -- missing cells: {missing}. "
                          f"Fix before generating figures (this project's own "
                          f"repeated lesson: a duplicate elsewhere can make the "
                          f"total row count look fine while a real cell is "
                          f"silently absent).")
    print(f"Verified complete: {len(SEEDS)} seeds x {len(expected_cells)} cells, "
          f"{len(df)} rows.")
    return df


def figure_gap_vs_power(df, condition, out_path):
    """Violation-rate gap (A - C) vs network size and training budget,
    mirroring the original project's figure_gap_vs_power exactly, with
    'violation_rate' in place of 'crash_rate'."""
    sub = df[df.condition == condition]
    means = sub.groupby(["architecture", "net", "training_timesteps"])["violation_rate"].mean()
    stds = sub.groupby(["architecture", "net", "training_timesteps"])["violation_rate"].std()
    n = len(SEEDS)

    a_mean = means.xs("A", level="architecture")
    c_mean = means.xs("C", level="architecture")
    a_std = stds.xs("A", level="architecture")
    c_std = stds.xs("C", level="architecture")

    gap_mean = (a_mean - c_mean).rename("gap_mean")
    gap_se = (np.sqrt(a_std**2 + c_std**2) / np.sqrt(n)).rename("gap_se")
    gap_df = pd.concat([gap_mean, gap_se], axis=1).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for budget, g in gap_df.groupby("training_timesteps"):
        g = g.sort_values("net")
        axes[0].errorbar(g["net"], g["gap_mean"], yerr=g["gap_se"], marker="o",
                          label=f"budget={budget}")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("network size (hidden units)")
    axes[0].set_ylabel("violation-rate gap (A - C)")
    axes[0].set_title(f"Gap (A-C) vs network size ({condition})")
    axes[0].axhline(0, color="gray", linewidth=0.8)
    axes[0].legend(fontsize="small")

    for net, g in gap_df.groupby("net"):
        g = g.sort_values("training_timesteps")
        axes[1].errorbar(g["training_timesteps"], g["gap_mean"], yerr=g["gap_se"],
                          marker="o", label=f"net_size={net}")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("training budget (timesteps)")
    axes[1].set_ylabel("violation-rate gap (A - C)")
    axes[1].set_title(f"Gap (A-C) vs training budget ({condition})")
    axes[1].axhline(0, color="gray", linewidth=0.8)
    axes[1].legend(fontsize="small")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def figure_violation_vs_w_largest(df, out_path):
    """Violation rate vs override weight w, A vs B vs C, at the largest
    (network_size, budget) cell tested."""
    override = df[df.condition.str.startswith("override_w")].copy()
    override["w"] = override["condition"].str.replace("override_w", "").astype(float)

    largest_net = max(NETWORK_SIZES)
    largest_budget = max(TRAINING_BUDGETS)
    cell = override[(override.net == largest_net) & (override.training_timesteps == largest_budget)]

    fig, ax = plt.subplots(figsize=(7, 5))
    styles = {"A": ("--", "#d64545", "A (single-loop)"),
              "B": ("-.", "#2f6fab", "B (hardcoded gate)"),
              "C": ("-", "#3ba55d", "C (learned gate)")}
    for arch, (ls, color, label) in styles.items():
        s = cell[cell.architecture == arch].groupby("w")["violation_rate"].mean().reset_index().sort_values("w")
        ax.plot(s["w"], s["violation_rate"], ls, marker="o", color=color, label=label)

    ax.set_xlabel("override / conflict weight (w)")
    ax.set_ylabel("violation rate")
    ax.set_title(f"Violation rate vs conflict weight, largest g tested\n"
                 f"(net_size={largest_net}, budget={largest_budget})")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize="small")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def report_stats(df):
    """Paired t-tests for the gap's collapse across budget and across
    network size, mirroring the original project's report_stats
    exactly (same statistical approach, applied here)."""
    largest_w = max(OVERRIDE_WEIGHTS)
    ow = df[df.condition == f"override_w{largest_w}"]

    def paired_gap_test(group_col, lo_val, hi_val):
        lo = ow[(ow[group_col] == lo_val) & (ow.architecture.isin(["A", "C"]))]
        hi = ow[(ow[group_col] == hi_val) & (ow.architecture.isin(["A", "C"]))]
        lo_p = lo.pivot_table(index="seed", columns="architecture", values="violation_rate")
        hi_p = hi.pivot_table(index="seed", columns="architecture", values="violation_rate")
        lo_gap = lo_p["A"] - lo_p["C"]
        hi_gap = hi_p["A"] - hi_p["C"]
        t, p = stats.ttest_rel(lo_gap, hi_gap)
        return lo_gap.mean(), hi_gap.mean(), t, p

    lo_b, hi_b = min(TRAINING_BUDGETS), max(TRAINING_BUDGETS)
    lo_g, hi_g, t, p = paired_gap_test("training_timesteps", lo_b, hi_b)
    n = len(SEEDS)
    print(f"\nBudget {lo_b} vs {hi_b} gap (A-C), override_w{largest_w}: "
          f"{lo_g:.4f} -> {hi_g:.4f}  paired t({n-1})={t:.2f}, p={p:.4g}")

    lo_n, hi_n = min(NETWORK_SIZES), max(NETWORK_SIZES)
    lo_g, hi_g, t, p = paired_gap_test("net", lo_n, hi_n)
    print(f"Net {lo_n} vs {hi_n} gap (A-C), override_w{largest_w}: "
          f"{lo_g:.4f} -> {hi_g:.4f}  paired t({n-1})={t:.2f}, p={p:.4g}")


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV_PATH
    df = load_deduped(csv_path)

    largest_w_condition = f"override_w{max(OVERRIDE_WEIGHTS)}"
    figure_gap_vs_power(df, largest_w_condition, "results/gap_AC_vs_power_oversight.png")
    figure_violation_vs_w_largest(df, "results/violation_rate_vs_w_ABC_largest.png")
    report_stats(df)
