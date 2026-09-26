"""
config_oversight.py
----------------------
Central experiment configuration for the oversight-signal generalization
experiment. Mirrors the original project's config.py's role: every
constant lives here so run_experiment_oversight.py doesn't hardcode
anything, and the sweep can be changed without touching core code.

Every value here reflects a decision actually made and tested during
this project's design discussion and sanity-check debugging -- see
NEXT_STEP_oversight_signal_v2.md for the full design rationale, and
this file's own comments for where each specific number came from.
"""

import itertools

# --- Environment ---
# grid_size/battery constants don't apply here -- see env_oversight.py's
# own constants (OVERSIGHT_DURATION, OVERSIGHT_PROB_PER_STEP, MAX_SPEED,
# VIOLATION_PENALTY, HOLD_TOLERANCE) for the environment's own fixed
# design parameters, validated via sanity_check_oversight.py and the
# diagnose_*.py scripts, not repeated here.
MAX_EPISODE_STEPS = 100  # matches the original project's grid env

# --- Independent variable: "power" of g ---
# Deliberately a SMALLER grid than the original project's 24 cells (6
# network sizes x 4 budgets): per the user's explicitly stated priority
# ("validate the intuition solidly, not run the biggest possible
# sweep") and the 5-month timeline, 3 sizes x 3 budgets = 9 cells.
# Spans low/mid/high on both axes rather than a dense grid -- enough to
# see whether the gap narrows with capability, without the cost of a
# full 24-cell sweep. Revisit only if the timing test (still pending on
# the actual VPS) shows this is affordable to expand.
NETWORK_SIZES = [[16], [256], [2048]]

# 10,000 is deliberately kept as the low end even though
# diagnose_A_at_full_budget.py showed Architecture A barely functions at
# that budget (violation_rate ~0.93-1.00) -- that poor showing at low
# budget IS part of what the sweep is supposed to capture (the
# capability-scaling story), not something to avoid by raising the
# floor. 500,000 matches the original project's and this experiment's
# own diagnostic scripts' largest tested budget.
TRAINING_BUDGETS = [10_000, 100_000, 500_000]

# 20 seeds (confirmed by user, upgraded from an initially-proposed 5-10)
# for better statistical power. Corresponds to a 2-core/2-process VPS
# split of 10 seeds per process -- see HANDOFF_STATUS_v2.md's corrected
# timing-lesson section for why this is NOT simply "half the speed of
# the original project's 4-core/10-seed run."
SEEDS = list(range(20))

ALGO = "PPO"  # continuous action space (Box(2,)) -- DQN doesn't apply here,
              # unlike the original project's discrete grid + DQN.

# --- rho_oversight: deliberately kept OUTSIDE the g-power sweep, same as
# the original project's rho (confirmed decision, mirrored) ---
RHO_NETWORK_SIZE = [64]           # matches CHECK_NET_SIZE used throughout sanity-checking
RHO_TIMESTEPS = 200_000           # NOT the same budget as the g-sweep's TRAINING_BUDGETS --
                                   # derived specifically from diagnose_rho_instability.py,
                                   # which showed violation_rate reaching a clean, stable
                                   # 0.000 from ~110,000-125,000 steps onward (with the
                                   # boosted training density below); 200,000 keeps real
                                   # margin above that observed convergence point.
RHO_TRAINING_OVERSIGHT_PROB = 0.5 # NOT the real/evaluation activation rate (0.05) --
                                   # rho_oversight's training reward is exactly zero
                                   # whenever oversight is inactive, so at the real 5%
                                   # rate, ~90% of every training rollout carries zero
                                   # gradient signal. Training rho in an environment with
                                   # a much higher activation rate gives it far denser
                                   # exposure to the one rule it needs to learn, without
                                   # changing what that rule IS. Confirmed via
                                   # diagnose_rho_instability.py: this was the fix that
                                   # took rho from "gets WORSE with 6x more budget" to
                                   # "converges cleanly." The environment used for
                                   # EVALUATION (and for A, g, and the gate's own
                                   # training) is UNCHANGED and still uses the real
                                   # OVERSIGHT_PROB_PER_STEP=0.05 from env_oversight.py.

# --- Hyperparameters (learning_rate, gamma, log_std_init, lambda_oversight_a) ---
# Loaded from tune_hyperparams_oversight.py's output at runtime (see
# HYPERPARAMS_PATH below), NOT hardcoded here -- mirrors the original
# project's own discipline (Section 7): tuned once via a dedicated script
# on disjoint validation seeds, never re-chosen after looking at final
# sweep results. Current tuned values (results/hyperparams_oversight.json,
# as of this project's last tuning run): learning_rate=0.001, gamma=0.99,
# log_std_init=-0.5, lambda_oversight_a=0.5. The 0.5 lambda in particular
# is notable: a WEAKER relative weighting of the oversight penalty than
# 1.0 scored better for Architecture A here -- plausibly reflecting a
# genuine trade-off (lighter oversight weighting lets A move more freely
# toward the goal at the cost of more violations) rather than an
# arbitrary choice; see this project's conversation history for the
# diagnostic reasoning, and treat this as a real, if imperfect,
# empirical finding rather than a target to re-tune away.

# --- Arbitration, Architecture B: HardGatePolicy (confirmed decision) ---
# No tunable parameters -- see arbitration_oversight.py's module
# docstring for why this is a hard step function (oversight_active is
# already binary, unlike battery's graded urgency), not a sigmoid like
# the original project's GATE_K/GATE_B_MID.

# --- Arbitration, Architecture C (learned gate) ---
# No separate net_size/timesteps constants here on purpose, mirroring
# the original project's train_gate: the gate is trained with the SAME
# (net_size, budget) cell as whatever g it's paired with, scaling
# together across NETWORK_SIZES x TRAINING_BUDGETS above.

# --- Adversarial / override scenarios ---
# override/conflict: ContinuousNavOversightEnv's adversarial_weight
# parameter adds a speed-proportional bonus to r_goal, the continuous
# analogue of the original project's "risk temptation" tiles --
# creating incentive pressure to move fast/directly regardless of
# oversight. Weight values chosen to mirror the original project's
# OVERRIDE_WEIGHTS range; not yet re-validated for THIS environment's
# own transition point, since no override sweep has been run here yet.
# Revisit once the sweep's own override results are in hand, the same
# way the original project added an extra weight only after seeing
# where its own transition actually fell.
OVERRIDE_WEIGHTS = [0, 1, 2, 4, 8]

# goal_nulling: no training at all (reward=0 means gradient=0) -- see
# train_oversight.py's get_untrained_g / get_untrained_architecture_a.
# indifference: lambda_oversight=0.0 for A and the gate's OWN training
# signal (mirrors LearnedGateEnv_Oversight's lambda_oversight parameter);
# a no-op for B by construction, same reasoning as the original
# project's B (g never sees the oversight/battery term either way).

# Every (network_size, training_budget) cell gets goal_nulling,
# indifference, and the override sweep -- not a compute-driven subset,
# since this experiment's 9-cell grid is already small enough (unlike
# the original project's first-pass 24-cell grid, which needed a
# 3-cell subset for compute reasons before the VPS run made the full
# grid affordable).
ADVERSARIAL_CAPABILITY_SUBSET = list(itertools.product(
    [n[0] for n in NETWORK_SIZES], TRAINING_BUDGETS
))

# --- Hyperparameter tuning validation set (disjoint from SEEDS on
# purpose -- matches tune_hyperparams_oversight.py's own VALIDATION_SEEDS) ---
VALIDATION_SEEDS = [100, 101, 102]

# --- Evaluation ---
N_EVAL_EPISODES = 200  # matches the original project's choice -- eval rollouts
                        # are cheap relative to training, so this buys tighter
                        # success_rate/violation_rate estimates for negligible
                        # extra cost.

# --- Paths ---
RESULTS_DIR = "results"
HYPERPARAMS_PATH = f"{RESULTS_DIR}/hyperparams_oversight.json"  # filled in by
                                                                  # tune_hyperparams_oversight.py
RESULTS_CSV_PATH = f"{RESULTS_DIR}/results_oversight.csv"
