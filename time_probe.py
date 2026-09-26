"""
time_probe.py (v2 -- all network sizes)
------------------------------------------
Measures real training speed on THIS machine for EVERY network size in
the real sweep's grid (16, 256, 2048), not just one -- training cost is
known (from this project's own history) to scale much more steeply with
network size than with training budget, so a single-size measurement
isn't enough to plan VPS core count, RAM, or rental duration.

Run this FIRST THING after provisioning any VPS, before committing to
the full sweep or even to a specific rental duration -- this project has
already been burned more than once by a hand-derived time estimate that
was off by several times over. If the numbers below suggest the full
grid won't fit in the planned rental window, it's much cheaper to trim
NETWORK_SIZES/TRAINING_BUDGETS in config_oversight.py NOW than to
discover this mid-sweep.

Usage:
    python time_probe.py
"""

import time

from env_oversight import ContinuousNavOversightEnv, RewardWrapperOversightOnly
from train_oversight import _make_ppo

PROBE_STEPS = 10_000  # per network size -- small enough to finish quickly,
                       # large enough to average out start-up overhead
NET_SIZES_TO_PROBE = [[16], [256], [2048]]  # matches config_oversight.NETWORK_SIZES


def probe_one_size(net_size):
    hp = {"learning_rate": 3e-4, "gamma": 0.99}
    env = RewardWrapperOversightOnly(ContinuousNavOversightEnv())
    model = _make_ppo(env, net_size, seed=0, hp=hp)

    start = time.time()
    model.learn(total_timesteps=PROBE_STEPS)
    elapsed = time.time() - start

    return PROBE_STEPS / elapsed


if __name__ == "__main__":
    print(f"Measuring real training speed for each network size in the sweep "
          f"({PROBE_STEPS} steps each)...\n")

    results = {}
    for net_size in NET_SIZES_TO_PROBE:
        label = str(net_size)
        print(f"Probing net_size={label}...")
        steps_per_sec = probe_one_size(net_size)
        results[label] = steps_per_sec
        print(f"  {steps_per_sec:.1f} steps/sec\n")

    print("=" * 60)
    print("Summary -- steps/sec by network size:")
    for label, sps in results.items():
        print(f"  net_size={label:>8}: {sps:>8.1f} steps/sec")

    print("\nEstimated time for a single training run at each (net_size, budget) "
          "combination in the real sweep's grid:")
    for label, sps in results.items():
        for budget in (10_000, 100_000, 500_000):
            est_min = (budget / sps) / 60
            print(f"  net_size={label:>8}, budget={budget:>7}: {est_min:>7.1f} minutes")

    print("\nNOTE: this measures ONE model in isolation (rho_oversight's own "
          "architecture, but net_size/algorithm are shared across A/g/gate too, "
          "so this is a reasonable proxy for all of them). It does NOT account "
          "for the number of models trained per cell (standard + override sweep "
          "+ goal_nulling + indifference -- roughly 20+ separate trainings per "
          "cell once the full condition set is counted), nor for CPU contention "
          "when 2 (or more) processes run in parallel on a multi-core VPS -- "
          "multiply carefully, and prefer measuring a full --mini-style run on "
          "the actual VPS over extrapolating purely from this number if in doubt.")
