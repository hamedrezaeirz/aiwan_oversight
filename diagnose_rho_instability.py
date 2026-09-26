"""
diagnose_rho_instability.py
-----------------------------
Focused diagnostic, NOT part of the final pipeline -- isolates ONE
question before any more hand-tuning happens: is rho_oversight's rising
violation_rate (seen in the last sanity_check_oversight.py run --
starts around 0.13 at 5k steps, ends around 0.97 at 50k steps) a
"hasn't converged yet" problem that more training steps fixes, or a
"genuinely getting worse" instability that more steps won't fix?

Trains ONLY rho_oversight (the simplest, most isolated component -- no
g, no gate, no combined objective) for a MUCH larger budget than the
sanity check's 50k, logging more frequently at first to see the early
trend clearly, then less often to save time over the long tail.

Usage:
    python diagnose_rho_instability.py

This is deliberately a throwaway diagnostic script, not something that
becomes part of the final experiment pipeline -- delete it once the
question it's asking has been answered.
"""

from env_oversight import ContinuousNavOversightEnv, RewardWrapperOversightOnly
from train_oversight import _make_ppo
from sanity_check_oversight import rollout, as_policy_fn

TOTAL_TIMESTEPS = 300_000
# Log more often early (where the instability was first visible), less
# often later (to save time) -- a fixed interval throughout would either
# waste time on excessive early logging or miss the early trend.
CHECKPOINTS = (
    list(range(5_000, 50_001, 5_000)) +      # every 5k up to 50k (matches the sanity check's own resolution)
    list(range(75_000, 300_001, 25_000))     # every 25k from 75k to 300k
)
NET_SIZE = [64]
SEED = 0
N_EVAL_EPISODES = 50  # more than the sanity check's 30, since this run is About
                        # getting a clear trend, not a quick check


if __name__ == "__main__":
    hp = {"learning_rate": 3e-4, "gamma": 0.99}

    env = RewardWrapperOversightOnly(
        ContinuousNavOversightEnv(oversight_prob_per_step=0.5)  # denser training signal -- see train_rho's docstring
    )
    model = _make_ppo(env, NET_SIZE, SEED, hp)

    print(f"Training rho_oversight alone for up to {TOTAL_TIMESTEPS} timesteps, "
          f"evaluating at {len(CHECKPOINTS)} checkpoints...\n")

    steps_trained = 0
    results = []
    for target in CHECKPOINTS:
        chunk = target - steps_trained
        model.learn(total_timesteps=chunk, reset_num_timesteps=False)
        steps_trained = target

        metrics = rollout(as_policy_fn(model), n_episodes=N_EVAL_EPISODES, seed=6000)
        results.append((target, metrics["violation_rate"], metrics["success_rate"]))
        print(f"  step={target:>7}  violation_rate={metrics['violation_rate']:.3f}  "
              f"(success_rate={metrics['success_rate']:.3f}, irrelevant for rho_oversight "
              f"but logged for completeness)")

    print("\n--- Summary ---")
    early = [v for s, v, _ in results if s <= 50_000]
    late = [v for s, v, _ in results if s > 200_000]
    print(f"Mean violation_rate, steps <= 50k:  {sum(early)/len(early):.3f}")
    print(f"Mean violation_rate, steps > 200k:  {sum(late)/len(late):.3f}")
    if sum(late) / len(late) < sum(early) / len(early) - 0.1:
        print("-> Looks like a CONVERGENCE-SPEED issue: violation_rate trends down "
              "with enough budget. 50k was simply too short; consider raising "
              "CHECK_TIMESTEPS in sanity_check_oversight.py, or accept that the real "
              "sweep's larger training budgets (up to 500k) will resolve this on their own.")
    else:
        print("-> Does NOT look like a simple convergence-speed issue: violation_rate "
              "isn't meaningfully better with 6x the budget. This points to a genuine "
              "instability (reward scale, learning rate, or a remaining design issue) "
              "rather than 'just needs more time' -- worth a hyperparameter-tuning pass "
              "(tune_hyperparams_oversight.py) or a closer look at the reward shaping "
              "before trying more hand-picked values.")
