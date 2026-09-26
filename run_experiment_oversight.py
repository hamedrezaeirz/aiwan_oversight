"""
run_experiment_oversight.py
------------------------------
Runs the full oversight-signal experiment: trains and evaluates
Architectures A, B, and C across NETWORK_SIZES x TRAINING_BUDGETS x
SEEDS (config_oversight.py), writing every result (never just averages)
to results/results_oversight.csv (or a per-process file, see --seeds)
as it goes, one cell at a time -- mirrors the original project's
run_experiment.py, including its progressive per-cell writing
discipline (so a crash partway through a long run doesn't lose earlier
progress, the exact lesson the original VPS run learned the hard way).

Usage:
    python run_experiment_oversight.py            # full sweep
    python run_experiment_oversight.py --quick     # tiny grid, pipeline check only
    python run_experiment_oversight.py --seeds 0 1 # only these seeds (for
                                                     # parallelizing across
                                                     # VPS cores -- see
                                                     # launch_parallel_oversight.sh,
                                                     # not yet written)
"""

import argparse
import csv
import json
import os

import torch

# One thread per process -- same reasoning as the original project's
# run_experiment.py: avoids contention when several of these run in
# parallel across VPS cores.
torch.set_num_threads(1)

from config_oversight import (
    NETWORK_SIZES, TRAINING_BUDGETS, SEEDS, OVERRIDE_WEIGHTS,
    ADVERSARIAL_CAPABILITY_SUBSET, RESULTS_DIR, HYPERPARAMS_PATH,
    RESULTS_CSV_PATH, N_EVAL_EPISODES, RHO_NETWORK_SIZE, RHO_TIMESTEPS,
    RHO_TRAINING_OVERSIGHT_PROB,
)
from train_oversight import (
    train_architecture_a, train_g, train_rho, train_gate,
    get_untrained_g, get_untrained_architecture_a,
)
from arbitration_oversight import HardGatePolicy, LearnedGatePolicy_Oversight
from evaluate_oversight import (
    evaluate_cell, as_policy_fn, as_hard_gate_policy_fn, as_learned_gate_policy_fn,
)

ROW_FIELDS = [
    "architecture", "network_size", "training_timesteps", "seed", "condition",
    "n_eval_episodes", "success_rate", "violation_rate",
    "avg_activations_per_episode", "avg_episode_length",
]


def load_hp():
    """Loads tuned hyperparameters from tune_hyperparams_oversight.py's
    output. Unlike the original project's load_hp(), this raises loudly
    rather than silently falling back to defaults -- by the time the
    real sweep runs, hyperparams_oversight.json should always exist;
    a missing file here almost certainly means tune_hyperparams_oversight.py
    hasn't been run yet on THIS machine (e.g. freshly copied to the VPS),
    which should be fixed, not silently worked around."""
    if not os.path.exists(HYPERPARAMS_PATH):
        raise SystemExit(
            f"{HYPERPARAMS_PATH} not found. Run tune_hyperparams_oversight.py "
            f"first (or copy its output file over, e.g. from the machine "
            f"where it was already run) -- the real sweep should not run on "
            f"un-tuned default hyperparameters."
        )
    with open(HYPERPARAMS_PATH) as f:
        return json.load(f)


def write_rows(rows, path):
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ROW_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _clear_stale_test_output(path):
    """
    --quick and --mini are meant to be idempotent validation runs, run as
    many times as needed while iterating -- but write_rows() APPENDS, not
    overwrites, by design (so the REAL sweep survives a mid-run crash
    without losing earlier progress, mirroring the original project's
    hard-learned lesson). Re-running --quick or --mini without clearing
    the old file first silently duplicates every row instead of replacing
    them, exactly as happened this round (confirmed by a --mini re-run
    whose output contained the OLD, pre-bugfix numbers duplicated
    alongside the NEW, post-fix ones in the same file -- caught only by
    manually inspecting the full file, not from any error or warning).
    For --quick/--mini specifically, deleting any prior file at the start
    is the right behavior, since these are never meant to accumulate
    across multiple validation runs. This does NOT apply to the real
    (non-quick, non-mini) sweep, where progressive accumulation across
    a single run's seeds is intentional -- see the printed warning in
    run() for that case instead.
    """
    if os.path.exists(path):
        print(f"NOTE: removing stale {path} from a previous run before starting "
              f"(--quick/--mini always start fresh, to avoid silently duplicating "
              f"rows from an earlier validation run).")
        os.remove(path)


def run(quick=False, mini=False, seeds=None):
    hp = load_hp()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if quick:
        network_sizes, training_budgets, run_seeds = [[16]], [2_000], [0]
        capability_subset = [(16, 2_000)]
        rho_kwargs = {"net_size": [16], "timesteps": 2_000, "training_oversight_prob": RHO_TRAINING_OVERSIGHT_PROB}
        out_path = f"{RESULTS_DIR}/results_oversight_quick_test.csv"
        _clear_stale_test_output(out_path)
        print("Running --quick: tiny grid, just validating the pipeline. "
              f"Writing to {out_path} (won't touch your real results file).")
    elif mini:
        # One REPRESENTATIVE cell at a REAL (non-trivial) budget, for
        # seeds 0 and 1 -- unlike --quick's 2,000-step toy budget, this
        # exercises the full condition set (standard, goal_nulling,
        # indifference, override sweep) under conditions close to what
        # the real sweep will actually see, without the multi-hour cost
        # of the full 9-cell grid at up to 500,000 steps. Net size [64]
        # and budget 100,000 chosen as a mid-range point: on this
        # machine's measured ~880 steps/sec (time_probe.py), a single
        # 100,000-step training run takes ~114 sec, keeping this whole
        # validation run to roughly an hour rather than several hours.
        # For PIPELINE VALIDATION ONLY -- not enough seeds or cells to
        # draw any real conclusion from the numbers this produces.
        network_sizes, training_budgets, run_seeds = [[64]], [100_000], [0, 1]
        capability_subset = [(64, 100_000)]
        rho_kwargs = {"net_size": RHO_NETWORK_SIZE, "timesteps": RHO_TIMESTEPS,
                       "training_oversight_prob": RHO_TRAINING_OVERSIGHT_PROB}
        out_path = f"{RESULTS_DIR}/results_oversight_mini_test.csv"
        _clear_stale_test_output(out_path)
        print("Running --mini: ONE real-budget cell (net=[64], budget=100,000), "
              "seeds 0-1, full condition set. For pipeline validation under "
              f"realistic conditions -- NOT for drawing conclusions. Writing to "
              f"{out_path} (won't touch your real results file). Expect roughly "
              "an hour on this machine.")
    else:
        network_sizes, training_budgets = NETWORK_SIZES, TRAINING_BUDGETS
        run_seeds = seeds if seeds is not None else SEEDS
        capability_subset = ADVERSARIAL_CAPABILITY_SUBSET
        rho_kwargs = {"net_size": RHO_NETWORK_SIZE, "timesteps": RHO_TIMESTEPS,
                       "training_oversight_prob": RHO_TRAINING_OVERSIGHT_PROB}
        if seeds is not None:
            tag = "_".join(str(s) for s in run_seeds)
            out_path = f"{RESULTS_DIR}/results_oversight_seeds_{tag}.csv"
        else:
            out_path = RESULTS_CSV_PATH
        if os.path.exists(out_path):
            print(f"WARNING: {out_path} already exists and will be APPENDED to, not "
                  f"overwritten -- if this is a re-run of a command you already ran "
                  f"before (rather than a genuine resume after an interruption), this "
                  f"WILL silently duplicate rows. If in doubt, stop now (Ctrl+C) and "
                  f"either remove the file or confirm this is really a resume. The "
                  f"original project's own hard lesson applies here too: duplicate "
                  f"rows can silently mask missing cells even when total row counts "
                  f"look fine -- always verify completeness with a groupby, not a "
                  f"raw count, after any run that touches an existing file.")
        print(f"Running seeds {run_seeds} (of {SEEDS}). Writing to {out_path}.")

    for seed in run_seeds:
        print(f"=== seed {seed}: training rho_oversight "
              f"(fixed across the g-power sweep, boosted training density) ===")
        rho_model = train_rho(seed, hp, **rho_kwargs)

        for net_size in network_sizes:
            for budget in training_budgets:
                cell = (net_size[0], budget)
                print(f"--- seed={seed} net_size={net_size} budget={budget} ---")
                rows = []

                # --- standard: full grid, all three architectures ---
                model_a = train_architecture_a(net_size, budget, seed, hp)
                rows.append(evaluate_cell("A", net_size, budget, seed, "standard",
                                           as_policy_fn(model_a), N_EVAL_EPISODES))

                g_model = train_g(net_size, budget, seed, hp)
                hard_gate_standard = HardGatePolicy(g_model, rho_model)
                rows.append(evaluate_cell("B", net_size, budget, seed, "standard",
                                           as_hard_gate_policy_fn(hard_gate_standard),
                                           N_EVAL_EPISODES, needs_oversight_flag=True))

                # C reuses the SAME g_model/rho_model as B in the standard
                # condition -- only the gate differs between B and C here,
                # mirroring the original project's exact reasoning: any
                # difference is attributable to the gate mechanism alone.
                gate_model = train_gate(g_model, rho_model, net_size, budget, seed, hp)
                learned_gate_standard = LearnedGatePolicy_Oversight(g_model, rho_model, gate_model)
                rows.append(evaluate_cell("C", net_size, budget, seed, "standard",
                                           as_learned_gate_policy_fn(learned_gate_standard),
                                           N_EVAL_EPISODES))

                # --- everything else: only on the capability subset ---
                if cell in capability_subset:
                    # A: goal_nulling (untrained)
                    model_a_null = get_untrained_architecture_a(net_size, seed)
                    rows.append(evaluate_cell("A", net_size, budget, seed, "goal_nulling",
                                               as_policy_fn(model_a_null), N_EVAL_EPISODES))

                    # A: indifference (oversight term removed, task reward unchanged)
                    model_a_indiff = train_architecture_a(net_size, budget, seed, hp,
                                                            lambda_oversight=0.0)
                    rows.append(evaluate_cell("A", net_size, budget, seed, "indifference",
                                               as_policy_fn(model_a_indiff), N_EVAL_EPISODES))

                    # A: override/conflict sweep
                    for w in OVERRIDE_WEIGHTS:
                        model_a_w = train_architecture_a(net_size, budget, seed, hp,
                                                           adversarial_weight=w)
                        rows.append(evaluate_cell("A", net_size, budget, seed, f"override_w{w}",
                                                   as_policy_fn(model_a_w), N_EVAL_EPISODES,
                                                   adversarial_weight=w))

                    # B: goal_nulling (g untrained, rho unaffected by construction)
                    g_null = get_untrained_g(net_size, seed)
                    hard_gate_null = HardGatePolicy(g_null, rho_model)
                    rows.append(evaluate_cell("B", net_size, budget, seed, "goal_nulling",
                                               as_hard_gate_policy_fn(hard_gate_null),
                                               N_EVAL_EPISODES, needs_oversight_flag=True))

                    # B: indifference -- identical to standard by construction
                    # (g never sees r_oversight either way); re-evaluated
                    # rather than copied, so a real bug would still show up.
                    rows.append(evaluate_cell("B", net_size, budget, seed, "indifference",
                                               as_hard_gate_policy_fn(hard_gate_standard),
                                               N_EVAL_EPISODES, needs_oversight_flag=True))

                    # B & C: override/conflict sweep -- g_w trained ONCE per w
                    # and reused for both B and C's evaluation at that w,
                    # mirroring the original project's exact optimization
                    # (avoids retraining g_w twice per weight).
                    for w in OVERRIDE_WEIGHTS:
                        g_w = train_g(net_size, budget, seed, hp, adversarial_weight=w)

                        hard_gate_w = HardGatePolicy(g_w, rho_model)
                        rows.append(evaluate_cell("B", net_size, budget, seed, f"override_w{w}",
                                                   as_hard_gate_policy_fn(hard_gate_w),
                                                   N_EVAL_EPISODES, adversarial_weight=w,
                                                   needs_oversight_flag=True))

                        gate_w = train_gate(g_w, rho_model, net_size, budget, seed, hp,
                                             adversarial_weight=w)
                        learned_gate_w = LearnedGatePolicy_Oversight(g_w, rho_model, gate_w)
                        rows.append(evaluate_cell("C", net_size, budget, seed, f"override_w{w}",
                                                   as_learned_gate_policy_fn(learned_gate_w),
                                                   N_EVAL_EPISODES, adversarial_weight=w))

                    # C: goal_nulling -- same untrained g as B's goal_nulling;
                    # the gate is still trained normally on top of it.
                    gate_null = train_gate(g_null, rho_model, net_size, budget, seed, hp)
                    learned_gate_null = LearnedGatePolicy_Oversight(g_null, rho_model, gate_null)
                    rows.append(evaluate_cell("C", net_size, budget, seed, "goal_nulling",
                                               as_learned_gate_policy_fn(learned_gate_null),
                                               N_EVAL_EPISODES))

                    # C: indifference -- unlike B (a no-op by construction),
                    # this is a real, distinct condition for C: the GATE's
                    # own training signal drops r_oversight
                    # (lambda_oversight=0.0), while g and rho are trained
                    # exactly as for the standard condition.
                    gate_indiff = train_gate(g_model, rho_model, net_size, budget, seed, hp,
                                              lambda_oversight=0.0)
                    learned_gate_indiff = LearnedGatePolicy_Oversight(g_model, rho_model, gate_indiff)
                    rows.append(evaluate_cell("C", net_size, budget, seed, "indifference",
                                               as_learned_gate_policy_fn(learned_gate_indiff),
                                               N_EVAL_EPISODES))

                write_rows(rows, out_path)

    print(f"Done. Results written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                         help="Run a tiny grid (trivial 2,000-step budget) to validate the "
                              "whole pipeline runs end to end before anything else.")
    parser.add_argument("--mini", action="store_true",
                         help="Run ONE real-budget cell (net=[64], budget=100,000), seeds 0-1, "
                              "the FULL condition set (standard/goal_nulling/indifference/"
                              "override sweep). For validating the pipeline under realistic "
                              "training conditions before the real sweep -- not for drawing "
                              "conclusions from the numbers it produces.")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                         help="Run only these seeds, e.g. --seeds 0 1. Defaults to all of "
                              "config_oversight.SEEDS. Launch several of these with disjoint "
                              "seed lists to parallelize across VPS cores -- each writes its "
                              "own results_oversight_seeds_<...>.csv, no write conflict.")
    args = parser.parse_args()
    if args.quick and args.mini:
        raise SystemExit("--quick and --mini are mutually exclusive -- pick one.")
    run(quick=args.quick, mini=args.mini, seeds=args.seeds)

    if args.seeds is not None:
        print(f"\nWhen every --seeds process is done, merge them with a "
              f"merge_results_oversight.py (mirroring the original project's "
              f"merge_results.py -- not yet written).")
