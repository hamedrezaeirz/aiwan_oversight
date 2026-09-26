"""
diagnose_A_at_full_budget.py
------------------------------
Fairness check requested by the user: Architecture A's sanity-check
learning curve (in sanity_check_oversight.py) showed a noisy, high
violation_rate at 50,000 steps, while rho_oversight only converged to
0.00 with a BOOSTED training-time oversight rate (0.5, vs. the real 0.05)
plus a much larger budget (200,000). Architecture A gets no such boost --
it trains on the real environment at the real activation rate, same as
the actual sweep will -- so it's not yet known whether A's poor showing
at 50k reflects a genuine architectural disadvantage (the paper's
claim) or simply that 50k isn't enough exposure to the same rare-signal
problem rho had, before any boost was applied.

This script trains ONLY Architecture A, at the REAL (unboosted)
oversight_prob_per_step=0.05 -- deliberately NOT using the density trick
used for rho_oversight, since A needs to be evaluated under the exact
conditions the real sweep will use, not an artificially easier one.
Runs up to 500,000 steps (the real sweep's largest training budget) to
see whether A eventually catches up given enough budget, or genuinely
plateaus below what C/B achieve.

This is a one-off diagnostic, not part of the final pipeline.

Usage:
    python diagnose_A_at_full_budget.py

Expected runtime: ~9-10 minutes on this machine, based on the ~880
steps/sec measured by time_probe.py for a similarly-sized model.
"""

from env_oversight import ContinuousNavOversightEnv, RewardWrapperA_Oversight
from train_oversight import _make_ppo
from sanity_check_oversight import rollout, as_policy_fn

TOTAL_TIMESTEPS = 500_000
CHECKPOINTS = (
    list(range(5_000, 50_001, 5_000)) +      # matches the sanity check's own resolution up to 50k
    list(range(75_000, 500_001, 25_000))     # coarser beyond that, out to the sweep's max budget
)
NET_SIZE = [64]
SEED = 0
N_EVAL_EPISODES = 50


if __name__ == "__main__":
    import json
    import os

    HP_PATH = "results/hyperparams_oversight.json"
    if os.path.exists(HP_PATH):
        with open(HP_PATH) as f:
            hp = json.load(f)
        print(f"NOTE: loaded TUNED hyperparameters from {HP_PATH}: {hp}\n")
    else:
        hp = {"learning_rate": 3e-4, "gamma": 0.99, "lambda_oversight_a": 1.0}
        print(f"NOTE: {HP_PATH} not found -- using SB3/PPO default placeholders "
              f"({hp}). Run tune_hyperparams_oversight.py first for tuned values.\n")

    # Deliberately the PLAIN environment -- real activation rate, no boost.
    env = RewardWrapperA_Oversight(ContinuousNavOversightEnv(), lambda_oversight=hp["lambda_oversight_a"])
    model = _make_ppo(env, NET_SIZE, SEED, hp)

    print(f"Training Architecture A alone, REAL oversight rate (0.05), "
          f"up to {TOTAL_TIMESTEPS} timesteps, evaluating at {len(CHECKPOINTS)} checkpoints...\n")

    steps_trained = 0
    results = []
    for target in CHECKPOINTS:
        chunk = target - steps_trained
        model.learn(total_timesteps=chunk, reset_num_timesteps=False)
        steps_trained = target

        metrics = rollout(as_policy_fn(model), n_episodes=N_EVAL_EPISODES, seed=7000)
        results.append((target, metrics["violation_rate"], metrics["success_rate"]))
        print(f"  step={target:>7}  violation_rate={metrics['violation_rate']:.3f}  "
              f"success_rate={metrics['success_rate']:.3f}")

    print("\n--- Summary ---")
    early = [v for s, v, _ in results if s <= 50_000]
    late = [v for s, v, _ in results if s > 400_000]
    print(f"Mean violation_rate, steps <= 50k:   {sum(early)/len(early):.3f}")
    print(f"Mean violation_rate, steps > 400k:   {sum(late)/len(late):.3f}")
    if sum(late) / len(late) < 0.1:
        print("-> A DOES eventually reach a low violation_rate given enough real-rate "
              "budget -- this means the 50k-step sanity check simply caught A mid-struggle "
              "with the same rare-signal problem rho had, not a hard architectural ceiling. "
              "The real sweep's budget range (10k-500k) should reveal this as a "
              "budget-dependent narrowing gap, same shape as the original battery result -- "
              "worth checking the SHAPE of A's improvement against C's, not just comparing "
              "a single fixed budget.")
    elif sum(late) / len(late) < sum(early) / len(early) - 0.2:
        print("-> A improves with more budget but doesn't fully close to ~0 even at 500k -- "
              "a partial narrowing. Worth reporting honestly as-is, not over- or "
              "under-stating how much gap remains at the largest budget tested.")
    else:
        print("-> A does NOT meaningfully improve even at 500k real-rate steps -- this "
              "looks like a genuine architectural ceiling, not a rare-signal exposure "
              "problem alone. This would need to be distinguished carefully from any "
              "remaining unboosted-training confound before treating it as confirming "
              "the paper's prediction.")
