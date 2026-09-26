"""
tune_hyperparams_oversight.py
-------------------------------
Systematic hyperparameter search for the oversight experiment, mirroring
the original project's tune_hyperparams.py and its stated discipline
(§7): nothing here is chosen after looking at final/main results, and
uses VALIDATION_SEEDS, disjoint from the real experiment's SEEDS (0-19),
so no run in the final sweep ever contributes to choosing its own
hyperparameters.

Why this exists now, specifically: the manually-found values used
throughout this project's iterative sanity-check debugging
(learning_rate=3e-4, gamma=0.99, log_std_init=-1.0, lambda_oversight_a=1.0)
were each picked by hand, one at a time, to fix a specific symptom as it
came up (see NEXT_STEP_oversight_signal_v2.md's history) -- not searched
systematically or jointly. In particular, diagnose_A_at_full_budget.py
showed Architecture A failing to meaningfully reduce violation_rate even
at 500,000 real-rate steps, WHILE lambda_oversight_a (A's relative
weighting of the oversight penalty against the goal reward) was still
just the untuned default of 1.0 -- exactly mirroring how the original
project's lambda_battery_a needed its own dedicated search rather than a
hand-picked value, this needs the same treatment before A's apparent
"architectural ceiling" can be trusted as a real finding rather than an
artifact of an unfavorable, unsearched reward weighting.

Two search passes, mirroring the original project's structure:
  1. learning_rate x gamma x log_std_init on Architecture A (the harder,
     combined-objective task -- same fairness logic as the original
     project's choice to tune on A rather than g or rho, so neither
     architecture is handicapped by settings chosen on an easier task).
  2. lambda_oversight_a on top of the chosen (lr, gamma, log_std_init),
     searched over a wider range than the original project's
     lambda_battery_a search (0.5-5.0) -- extended up to 20.0 here,
     since diagnose_A_at_full_budget.py's result suggests oversight
     compliance may need substantially more relative weight than goal
     reward to be learnable at all for a single combined-objective
     policy.

No gate-parameter search here: unlike the original project's B
(DualLoopPolicy, tuned via GATE_K/GATE_B_MID), this experiment's
Architecture B (HardGatePolicy) is a fixed if/else with no tunable
parameters -- see arbitration_oversight.py. Nothing to search there.

rho_oversight's own settings (training_oversight_prob=0.5,
CHECK_TIMESTEPS_RHO=200,000) were derived from a dedicated diagnostic
(diagnose_rho_instability.py), not a grid search -- they're a design
resolution to a specific identified problem (near-zero gradient signal
outside oversight-active windows), not treated as free hyperparameters
here.

Usage:
    python tune_hyperparams_oversight.py

Expected runtime: rough estimate only -- this runs
    len(LR_CANDIDATES) x len(GAMMA_CANDIDATES) x len(LOG_STD_INIT_CANDIDATES)
    x len(VALIDATION_SEEDS)
    + len(LAMBDA_CANDIDATES) x len(VALIDATION_SEEDS)
separate training runs at VALIDATION_TIMESTEPS each. With the defaults
below (3 x 2 x 3 x 3 = 54, plus 6 x 3 = 18, total 72 runs at 50,000
steps each) and this machine's measured ~880 steps/sec (see
time_probe.py), that's roughly 72 x 50,000 / 880 ~= 4,100 seconds ~=
68 minutes. Run time_probe.py again first if in doubt, rather than
trusting this estimate blindly -- this project has already been burned
once by a hand-derived time estimate that was off by several times over
(see HANDOFF_STATUS_v2.md's corrected-timing-lesson section).
"""

import json
import os

import numpy as np

from env_oversight import ContinuousNavOversightEnv, RewardWrapperA_Oversight
from train_oversight import train_architecture_a
from sanity_check_oversight import rollout, as_policy_fn

RESULTS_DIR = "results"
HYPERPARAMS_PATH = f"{RESULTS_DIR}/hyperparams_oversight.json"

VALIDATION_NET_SIZE = [64]
VALIDATION_TIMESTEPS = 200_000  # RAISED from 50,000: the first search attempt
                                 # showed nearly every candidate scoring -1.0
                                 # (disqualified -- essentially zero success
                                 # across all 3 validation seeds), not because
                                 # the candidates were bad but because 50,000
                                 # is far too small a budget for THIS
                                 # environment/algorithm combination to show
                                 # meaningful success at all -- confirmed by
                                 # diagnose_A_at_full_budget.py, where success_rate
                                 # only started climbing past ~400,000 steps.
                                 # 200,000 is still a compromise (not the full
                                 # 500,000 sweep budget), chosen together with
                                 # a SMALLER candidate grid below to keep total
                                 # search time reasonable -- see the runtime
                                 # estimate in the module docstring, which no
                                 # longer applies exactly with these new sizes
                                 # and should be recomputed via time_probe.py
                                 # before trusting a specific ETA.
VALIDATION_SEEDS = [100, 101, 102]  # disjoint from the real experiment's SEEDS (0-19)

# Grid deliberately SHRUNK relative to the first attempt, to keep total
# search time affordable at the larger VALIDATION_TIMESTEPS above: gamma
# fixed at 0.99 (consistently outperformed 0.95 in every comparable pair
# from the first search attempt, e.g. lr=0.001: 0.95->-1.0 vs 0.99->-0.578;
# lr=0.0003: 0.95->-1.0 vs 0.99->-0.311 -- not conclusive on its own given
# how many candidates were simply disqualified, but the ONLY consistent
# signal available from that run, so fixing it here is a reasonable way to
# afford a bigger budget elsewhere rather than a blind guess).
LR_CANDIDATES = [1e-3, 3e-4]
GAMMA_CANDIDATES = [0.99]
LOG_STD_INIT_CANDIDATES = [-2.0, -1.0, -0.5]  # replaced 0.0 (consistently the
                                                # worst performer, never once
                                                # scoring above -0.578 in the
                                                # first attempt) with an
                                                # intermediate -0.5, since the
                                                # only candidate that ever
                                                # passed the floor at all used
                                                # -2.0 -- worth checking whether
                                                # something between -2.0 and
                                                # -1.0 does even better with
                                                # more budget to prove it.
LAMBDA_CANDIDATES = [0.5, 1.0, 2.0, 5.0]  # trimmed from 6 to 4 candidates for
                                            # the same time-affordability reason;
                                            # 10.0/20.0 both scored -1.0 in the
                                            # first attempt (though that attempt's
                                            # budget was too small to trust that
                                            # result either -- dropped here mainly
                                            # for time, not because they're
                                            # confidently ruled out).


def _score(row):
    """
    success_rate + (1 - violation_rate) alone is EXPOSED as gameable: a
    policy that simply never moves gets success_rate~=0 (never reaches
    the goal) AND violation_rate~=0 (can't violate oversight if it never
    moves), scoring ~1.0 -- indistinguishable from a genuinely competent
    policy that reaches the goal often AND complies with oversight.
    Confirmed happening in practice: the first run of this search picked
    log_std_init=-2.0 with score=1.000, the exact hyperparameter value
    this project's own manual debugging had already identified as
    producing a frozen, non-functional policy (success_rate collapsed to
    ~0 across every architecture when this was tried during sanity-check
    iteration -- see NEXT_STEP_oversight_signal_v2.md's history). Left
    uncaught, this would have silently biased every downstream choice
    toward degenerate "do nothing" policies for Architecture A
    specifically, artificially widening the A-vs-C gap the real
    experiment is supposed to measure honestly -- exactly the kind of
    bias the user explicitly asked NOT to introduce.

    Fix, two parts:
      1. A low floor (MIN_SUCCESS_RATE=0.05) disqualifies candidates with
         essentially-zero success outright -- catches the literal frozen-
         policy case without being so strict it disqualifies every
         candidate at this search's modest VALIDATION_TIMESTEPS=50,000
         budget (where even reasonable candidates showed fairly low raw
         success_rate in earlier manual runs, e.g. g reaching ~0.10-0.27).
      2. success_rate is weighted 3x relative to (1 - violation_rate), so
         that even candidates passing the floor can't win purely by
         being cautious/inactive -- a candidate has to actually attempt
         (and sometimes reach) the goal to score competitively, not just
         avoid violating oversight by doing nothing.
    Neither part is a validated final choice -- this is a first pass at
    preventing the specific degenerate solution already observed; revisit
    the exact floor/weight if the chosen candidates still look suspicious
    (e.g. success_rate barely above the floor, or violation_rate exactly
    0.0 alongside barely-passing success).
    """
    MIN_SUCCESS_RATE = 0.05
    if row["success_rate"] < MIN_SUCCESS_RATE:
        return -1.0  # worse than any valid candidate's score (valid range is roughly [0, 4])
    return 3.0 * row["success_rate"] + (1.0 - row["violation_rate"])


def _mean_score_over_seeds(build_and_eval_fn):
    return float(np.mean([_score(build_and_eval_fn(s)) for s in VALIDATION_SEEDS]))


def _evaluate_a(net_size, timesteps, seed, hp, lambda_oversight=None):
    model = train_architecture_a(net_size, timesteps, seed, hp, lambda_oversight=lambda_oversight)
    metrics = rollout(as_policy_fn(model), n_episodes=30, seed=seed * 1000)
    return metrics


def tune_lr_gamma_logstd():
    print("Tuning learning_rate / gamma / log_std_init on Architecture A...")
    best_score, best_hp = -np.inf, None
    for lr in LR_CANDIDATES:
        for gamma in GAMMA_CANDIDATES:
            for log_std_init in LOG_STD_INIT_CANDIDATES:
                hp = {"learning_rate": lr, "gamma": gamma, "log_std_init": log_std_init,
                      "lambda_oversight_a": 1.0}

                def build_and_eval(seed, hp=hp):
                    return _evaluate_a(VALIDATION_NET_SIZE, VALIDATION_TIMESTEPS, seed, hp)

                score = _mean_score_over_seeds(build_and_eval)
                print(f"  lr={lr} gamma={gamma} log_std_init={log_std_init} -> score={score:.3f}")
                if score > best_score:
                    best_score, best_hp = score, {
                        "learning_rate": lr, "gamma": gamma, "log_std_init": log_std_init,
                    }
    if best_hp is None or best_score <= -1.0:
        raise SystemExit(
            "Every candidate was disqualified by _score's MIN_SUCCESS_RATE floor -- "
            "none reached even minimal task competence at VALIDATION_TIMESTEPS. This "
            "likely means VALIDATION_TIMESTEPS is too small for ANY candidate to show "
            "real success at this net_size, not that every hyperparameter combination "
            "is bad. Raise VALIDATION_TIMESTEPS (or lower MIN_SUCCESS_RATE cautiously, "
            "re-reading _score's docstring first) rather than trusting a result chosen "
            "this way -- do not proceed to tune_lambda_oversight_a with best_hp=None."
        )
    print(f"Chosen: {best_hp} (score={best_score:.3f})")
    return best_hp


def tune_lambda_oversight_a(base_hp):
    print("Tuning lambda_oversight_a on Architecture A...")
    best_score, best_lambda = -np.inf, None
    for lam in LAMBDA_CANDIDATES:
        hp = {**base_hp, "lambda_oversight_a": lam}

        def build_and_eval(seed, hp=hp, lam=lam):
            return _evaluate_a(VALIDATION_NET_SIZE, VALIDATION_TIMESTEPS, seed, hp,
                                lambda_oversight=lam)

        score = _mean_score_over_seeds(build_and_eval)
        print(f"  lambda_oversight_a={lam} -> score={score:.3f}")
        if score > best_score:
            best_score, best_lambda = score, lam
    print(f"Chosen: lambda_oversight_a={best_lambda} (score={best_score:.3f})")
    return best_lambda


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    base_hp = tune_lr_gamma_logstd()
    lambda_oversight_a = tune_lambda_oversight_a(base_hp)

    final_hp = {**base_hp, "lambda_oversight_a": lambda_oversight_a}
    with open(HYPERPARAMS_PATH, "w") as f:
        json.dump(final_hp, f, indent=2)
    print(f"\nSaved to {HYPERPARAMS_PATH}: {final_hp}")
    print("\nNOTE: this does NOT re-validate rho_oversight's own settings "
          "(training_oversight_prob, CHECK_TIMESTEPS_RHO) -- those came from "
          "diagnose_rho_instability.py's targeted fix, not this grid search. "
          "Nor does it search anything for Architecture B (HardGatePolicy has "
          "no tunable parameters) or the gate's own net_size/timesteps (which "
          "are tied to whichever g cell they're paired with, per the confirmed "
          "design decision mirrored from the original project's train_gate).")
