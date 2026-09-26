"""
sanity_check_oversight.py
---------------------------
Run this BEFORE committing to any multi-day VPS sweep, and BEFORE the
VPS timing test the user has asked for -- this is what actually tells
us whether OVERSIGHT_PROB_PER_STEP=0.05 is a reasonable rate, and
whether PPO is learning real behavior on this new environment at all,
not an artifact. Mirrors sanity_check.py and sanity_check_gate.py's
structure and reasoning closely, combined into one file since this
experiment's whole pipeline (env + PPO + hard gate + learned gate) is
new all at once, unlike the original project where A/B were validated
first and C was added later on already-trusted ground.

What this checks, in order:

  1. Activation-rate diagnostic (NEW relative to the original project):
     with OVERSIGHT_PROB_PER_STEP=0.05 and max_episode_steps=100, how
     many activations per episode actually occur under a random policy,
     and what fraction of total steps is oversight-active? This has no
     analogue in the original sanity_check.py because battery urgency
     was continuous, not a tunable stochastic rate -- this diagnostic
     is what tells us whether 0.05 needs adjusting BEFORE spending any
     training compute on it.
  2. Two reference baselines: a random policy, and a hand-coded
     "greedy toward goal, ignore oversight entirely" heuristic --
     same role as sanity_check.py's baselines: a sense of what "good"
     success_rate should approach, and what violation_rate looks like
     when oversight is completely ignored on purpose.
  3. Learning curves for Architecture A, g, and rho_oversight -- same
     chunked train-then-evaluate loop as sanity_check.py, using this
     experiment's own rollout().
  4. Learning curve + routing diagnostic for the gate (Architecture C),
     mirroring sanity_check_gate.py: does the composed system's
     success/violation rate climb, and does the gate's g-vs-rho choice
     actually vary with oversight_active rather than collapsing onto a
     state-independent rule?

Usage:
    python sanity_check_oversight.py

NOTE: needs stable-baselines3 + torch installed (same as the rest of
this experiment's training code) -- not runnable in an environment
without those, which is why this could only be syntax-checked, not
executed, before being handed off. Run this on the VPS (or wherever
train_oversight.py's own smoke test was already confirmed to run)
before trusting anything it reports.
"""

import numpy as np
import matplotlib.pyplot as plt

from env_oversight import (
    ContinuousNavOversightEnv,
    RewardWrapperA_Oversight,
    RewardWrapperGoalOnly_Oversight,
    RewardWrapperOversightOnly,
    LearnedGateEnv_Oversight,
    OVERSIGHT_PROB_PER_STEP,
    OVERSIGHT_DURATION,
)
from train_oversight import _make_ppo, train_g, train_rho
from arbitration_oversight import HardGatePolicy, LearnedGatePolicy_Oversight

CHECK_TIMESTEPS = 50_000     # for A / g / gate -- these are the variables the
                              # real experiment actually sweeps over (including
                              # small budgets like 10,000), so their own
                              # convergence behavior across budgets is part of
                              # what's being measured, not something to paper
                              # over with a bigger check budget here.
CHECK_TIMESTEPS_RHO = 200_000 # rho_oversight is NOT part of the capability
                              # sweep -- like the original project's rho, it's
                              # trained once, fixed, and reused across every
                              # (net_size, budget) cell of g. It therefore
                              # needs to be reliably well-trained on its own
                              # terms, with margin above whatever budget it
                              # actually needs. diagnose_rho_instability.py
                              # (run with the density-boosted training env,
                              # see train_oversight.train_rho's
                              # training_oversight_prob) showed violation_rate
                              # reaching a clean, stable 0.000 from ~125,000
                              # steps onward -- 200,000 keeps a real margin
                              # above that observed convergence point.
CHECK_INTERVAL = 5_000
CHECK_NET_SIZE = [64]        # a middle-of-the-road size, matching the original project's convention
CHECK_SEED = 0
N_EVAL_EPISODES = 30
N_ACTIVATION_CHECK_EPISODES = 200
N_ROUTING_EPISODES = 50


# ---------------------------------------------------------------------------
# Rollout helper (this experiment's equivalent of evaluate.py's rollout(),
# not yet split into its own file -- kept here since only this script and,
# later, evaluate_oversight.py will need it; consider extracting if a
# separate evaluate_oversight.py is written later, to avoid duplication).
# ---------------------------------------------------------------------------

def rollout(policy_fn, n_episodes, seed=None, adversarial_weight=0.0, needs_oversight_flag=False):
    """
    Runs n_episodes on a fresh ContinuousNavOversightEnv and aggregates
    metrics. Mirrors evaluate.py's rollout() closely.

    policy_fn signature: (obs, oversight_active) -> action if
    needs_oversight_flag else (obs) -> action -- HardGatePolicy needs the
    raw flag (see arbitration_oversight.py's docstring on why it's passed
    explicitly rather than inferred from obs); plain PPO models and the
    learned gate do not.
    """
    env = ContinuousNavOversightEnv(adversarial_weight=adversarial_weight)
    successes = violations = 0
    episode_activations, lengths = [], []

    for ep in range(n_episodes):
        ep_seed = None if seed is None else seed + ep
        obs, _ = env.reset(seed=ep_seed)
        steps = 0
        n_activations_this_ep = 0
        prev_active = False
        info = {}
        for _ in range(env.max_episode_steps):
            if needs_oversight_flag:
                action = policy_fn(obs, env.oversight_active)
            else:
                action = policy_fn(obs)
            obs, _, terminated, truncated, info = env.step(action)
            if info["oversight_active"] and not prev_active:
                n_activations_this_ep += 1
            prev_active = info["oversight_active"]
            steps += 1
            if terminated or truncated:
                break
        successes += int(info.get("reached_goal", False))
        violations += int(info.get("violated", False))
        episode_activations.append(n_activations_this_ep)
        lengths.append(steps)

    return {
        "n_eval_episodes": n_episodes,
        "success_rate": successes / n_episodes,
        "violation_rate": violations / n_episodes,
        "avg_activations_per_episode": float(np.mean(episode_activations)),
        "avg_episode_length": float(np.mean(lengths)),
    }


def as_policy_fn(sb3_model):
    def fn(obs):
        action, _ = sb3_model.predict(obs, deterministic=True)
        return action
    return fn


def as_hard_gate_policy_fn(hard_gate_policy):
    def fn(obs, oversight_active):
        return hard_gate_policy.predict(obs, oversight_active, deterministic=True)
    return fn


def as_learned_gate_policy_fn(learned_gate_policy):
    def fn(obs):
        return learned_gate_policy.predict(obs, deterministic=True)
    return fn


# ---------------------------------------------------------------------------
# 1. Activation-rate diagnostic
# ---------------------------------------------------------------------------

def activation_rate_diagnostic(n_episodes=N_ACTIVATION_CHECK_EPISODES):
    """
    Runs a RANDOM policy (not trained) purely to characterize how often
    oversight actually fires under OVERSIGHT_PROB_PER_STEP -- this needs
    no learning at all, so it's the cheapest possible check and should
    run first, before spending any training compute.
    """
    env = ContinuousNavOversightEnv()
    total_active_steps = total_steps = 0
    activations_per_ep = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=9000 + ep)
        prev_active = False
        n_act = 0
        for _ in range(env.max_episode_steps):
            action = env.action_space.sample()
            obs, _, terminated, truncated, info = env.step(action)
            total_steps += 1
            if info["oversight_active"]:
                total_active_steps += 1
                if not prev_active:
                    n_act += 1
            prev_active = info["oversight_active"]
            if terminated or truncated:
                break
        activations_per_ep.append(n_act)

    frac_active = total_active_steps / total_steps
    print(f"Activation-rate diagnostic (OVERSIGHT_PROB_PER_STEP={OVERSIGHT_PROB_PER_STEP}, "
          f"duration={OVERSIGHT_DURATION}):")
    print(f"  fraction of all steps oversight-active: {frac_active:.3f}")
    print(f"  avg activations per episode: {np.mean(activations_per_ep):.2f} "
          f"(min={np.min(activations_per_ep)}, max={np.max(activations_per_ep)})")
    if frac_active < 0.05:
        print("  NOTE: oversight is active less than 5% of all steps -- may be too rare "
              "for a policy to learn a reliable response; consider raising "
              "OVERSIGHT_PROB_PER_STEP before the real sweep.")
    elif frac_active > 0.5:
        print("  NOTE: oversight is active more than half of all steps -- episodes may be "
              "dominated by HOLD, leaving little room for goal-directed behavior; consider "
              "lowering OVERSIGHT_PROB_PER_STEP before the real sweep.")
    else:
        print("  Activation rate looks reasonable (neither dominated by HOLD nor too rare "
              "to learn from) -- no adjustment indicated by this check alone.")


# ---------------------------------------------------------------------------
# 2. Reference baselines
# ---------------------------------------------------------------------------

def random_baseline(n_episodes=100):
    env = ContinuousNavOversightEnv()
    return rollout(lambda obs: env.action_space.sample(), n_episodes, seed=2000)


def greedy_ignore_oversight_baseline(n_episodes=100):
    """
    Hand-coded, ignoring oversight ENTIRELY on purpose: always move
    directly toward the goal at full speed. This is a rough ceiling for
    success_rate and, deliberately, a floor for violation_rate --
    a trained g (which never sees r_oversight, same as the original
    project's g) should behave similarly to this when isolated, which
    is what makes rho_oversight's job of overriding it meaningful.
    """
    def policy_fn(obs):
        agent_x, agent_y, goal_x, goal_y = obs[0], obs[1], obs[2], obs[3]
        direction = np.array([goal_x - agent_x, goal_y - agent_y])
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return np.zeros(2, dtype=np.float32)
        return (direction / norm).astype(np.float32)

    return rollout(policy_fn, n_episodes, seed=3000)


# ---------------------------------------------------------------------------
# 3. Learning curves for A, g, rho_oversight
# ---------------------------------------------------------------------------

def learning_curve(build_env_fn, hp, label, track_violations=False, total_timesteps=CHECK_TIMESTEPS):
    """Mirrors sanity_check.py's learning_curve() exactly, adapted to
    this experiment's rollout()/metrics names (violation_rate instead
    of crash_rate). total_timesteps is now a parameter (default
    CHECK_TIMESTEPS) rather than hardcoded, so rho_oversight's own
    check can use the larger CHECK_TIMESTEPS_RHO budget it actually
    needs (see that constant's definition above) without changing A/g's
    budget, which is deliberately kept at the smaller, real-sweep-like
    CHECK_TIMESTEPS."""
    env = build_env_fn()
    model = _make_ppo(env, CHECK_NET_SIZE, CHECK_SEED, hp)

    steps_done, success_rates, violation_rates = [], [], []
    n_chunks = total_timesteps // CHECK_INTERVAL
    for i in range(n_chunks):
        model.learn(total_timesteps=CHECK_INTERVAL, reset_num_timesteps=False)
        metrics = rollout(as_policy_fn(model), n_episodes=N_EVAL_EPISODES, seed=1000 + i)
        steps_done.append((i + 1) * CHECK_INTERVAL)
        success_rates.append(metrics["success_rate"])
        violation_rates.append(metrics["violation_rate"])
        print(f"  [{label}] step={steps_done[-1]:>7} "
              f"success_rate={metrics['success_rate']:.2f} "
              f"violation_rate={metrics['violation_rate']:.2f}")

    return steps_done, success_rates, violation_rates


# ---------------------------------------------------------------------------
# 4. Gate learning curve + routing diagnostic
# ---------------------------------------------------------------------------

def gate_learning_curve(g_model, rho_model, hp):
    """Mirrors sanity_check_gate.py's gate_learning_curve() exactly."""
    env = LearnedGateEnv_Oversight(ContinuousNavOversightEnv(), g_model, rho_model)
    gate_model = _make_ppo(env, CHECK_NET_SIZE, CHECK_SEED, hp)

    steps_done, success_rates, violation_rates = [], [], []
    n_chunks = CHECK_TIMESTEPS // CHECK_INTERVAL
    for i in range(n_chunks):
        gate_model.learn(total_timesteps=CHECK_INTERVAL, reset_num_timesteps=False)
        composed = LearnedGatePolicy_Oversight(g_model, rho_model, gate_model)
        metrics = rollout(as_learned_gate_policy_fn(composed), n_episodes=N_EVAL_EPISODES, seed=1000 + i)
        steps_done.append((i + 1) * CHECK_INTERVAL)
        success_rates.append(metrics["success_rate"])
        violation_rates.append(metrics["violation_rate"])
        print(f"  [gate] step={steps_done[-1]:>7} "
              f"success_rate={metrics['success_rate']:.2f} "
              f"violation_rate={metrics['violation_rate']:.2f}")

    return steps_done, success_rates, violation_rates, gate_model


def routing_diagnostic(g_model, rho_model, gate_model, n_episodes=N_ROUTING_EPISODES):
    """
    Mirrors sanity_check_gate.py's routing_diagnostic(), but bucketed by
    oversight_active (0 or 1) instead of battery-level buckets, since
    the signal here is binary rather than graded. Reports the fraction
    of steps routed to rho_oversight in each bucket -- if oversight_active=1
    doesn't route to rho_oversight noticeably more than oversight_active=0
    does, the gate hasn't learned to use the signal at all, whatever the
    aggregate violation_rate says.
    """
    env = ContinuousNavOversightEnv()
    bucket_counts = {0: [0, 0], 1: [0, 0]}  # [n_rho, n_total]

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=4000 + ep)
        for _ in range(env.max_episode_steps):
            gate_action, _ = gate_model.predict(obs, deterministic=True)
            gate_action = int(gate_action)
            bucket = int(env.oversight_active)
            bucket_counts[bucket][1] += 1
            bucket_counts[bucket][0] += gate_action  # action==1 means "defer to rho_oversight"
            sub_model = rho_model if gate_action == 1 else g_model
            real_action, _ = sub_model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(real_action)
            if terminated or truncated:
                break

    print("\nRouting diagnostic -- fraction of steps the gate deferred to rho_oversight, by oversight_active:")
    fractions = {}
    for bucket in (0, 1):
        n_rho, n_total = bucket_counts[bucket]
        frac = n_rho / n_total if n_total > 0 else float("nan")
        fractions[bucket] = frac
        print(f"  oversight_active={bucket}: n={n_total:>5}  fraction routed to rho_oversight = {frac:.2f}")

    spread = fractions[1] - fractions[0]
    if spread < 0.2:
        print(f"  WARNING: fraction routed to rho_oversight barely differs between "
              f"oversight_active=0 and =1 (spread={spread:.2f}) -- the gate may not have "
              f"learned to use the signal at all. Worth a closer look before the real sweep.")
    else:
        print(f"  Spread between buckets: {spread:.2f} -- the gate clearly routes "
              f"differently when oversight is active vs. not.")

    return fractions


if __name__ == "__main__":
    import json
    import os

    # Load the tuned hyperparameters if tune_hyperparams_oversight.py has
    # been run (results/hyperparams_oversight.json); otherwise fall back
    # to the same SB3/PPO-default placeholder as before. This is the
    # first time this script uses the TUNED values rather than hand-picked
    # placeholders -- read the printed success_rate/violation_rate below
    # carefully, since the tuning search only reported a combined score,
    # not these raw numbers, and the winning lambda_oversight_a=0.5 could
    # reflect a real success/violation trade-off worth seeing explicitly.
    HP_PATH = "results/hyperparams_oversight.json"
    if os.path.exists(HP_PATH):
        with open(HP_PATH) as f:
            hp = json.load(f)
        print(f"NOTE: loaded TUNED hyperparameters from {HP_PATH}: {hp}\n")
    else:
        hp = {"learning_rate": 3e-4, "gamma": 0.99, "lambda_oversight_a": 1.0}
        print(f"NOTE: {HP_PATH} not found -- using SB3/PPO default placeholders "
              f"({hp}). Run tune_hyperparams_oversight.py first for tuned values.\n")

    os.makedirs("results", exist_ok=True)

    print("=" * 70)
    print("STEP 1: Activation-rate diagnostic (no training required)")
    print("=" * 70)
    activation_rate_diagnostic()

    print("\n" + "=" * 70)
    print("STEP 2: Reference baselines (no learning involved)")
    print("=" * 70)
    rand = random_baseline()
    print(f"  random policy:            success_rate={rand['success_rate']:.2f} "
          f"violation_rate={rand['violation_rate']:.2f}")
    greedy = greedy_ignore_oversight_baseline()
    print(f"  greedy-ignore-oversight:   success_rate={greedy['success_rate']:.2f} "
          f"violation_rate={greedy['violation_rate']:.2f} "
          f"(ignores oversight on purpose -- a HIGH violation_rate here is expected and fine)")

    print("\n" + "=" * 70)
    print("STEP 3: Learning curves, Architecture A / g / rho_oversight")
    print("=" * 70)
    print("\nArchitecture A:")
    steps_a, succ_a, viol_a = learning_curve(
        lambda: RewardWrapperA_Oversight(ContinuousNavOversightEnv(), lambda_oversight=hp["lambda_oversight_a"]),
        hp, "A")

    print("\ng (goal-only):")
    steps_g, succ_g, _ = learning_curve(
        lambda: RewardWrapperGoalOnly_Oversight(ContinuousNavOversightEnv()), hp, "g")

    print("\nrho_oversight (oversight-only, boosted training density + larger budget -- "
          f"see CHECK_TIMESTEPS_RHO and train_rho's docstring):")
    # "success" isn't meaningful for rho_oversight (it never sees the goal) --
    # what matters is violation_rate dropping toward 0. Uses
    # RewardWrapperOversightOnly with a DENSER training-only oversight rate
    # (see train_oversight.train_rho's training_oversight_prob) -- this
    # local build_env_fn intentionally does NOT match train_g/RewardWrapperA's
    # plain ContinuousNavOversightEnv(), for exactly that reason.
    from env_oversight import ContinuousNavOversightEnv as _CNE
    steps_r, _, viol_r = learning_curve(
        lambda: RewardWrapperOversightOnly(_CNE(oversight_prob_per_step=0.5)),
        hp, "rho_oversight", total_timesteps=CHECK_TIMESTEPS_RHO)

    print("\n" + "=" * 70)
    print("STEP 4: Gate learning curve + routing diagnostic (Architecture C)")
    print("=" * 70)
    print(f"\nTraining g and rho_oversight fully (g: {CHECK_TIMESTEPS} steps, "
          f"rho_oversight: {CHECK_TIMESTEPS_RHO} steps with boosted training "
          f"density -- see above -- both held fixed for the gate)...")
    g_model = train_g(CHECK_NET_SIZE, CHECK_TIMESTEPS, CHECK_SEED, hp)
    rho_model = train_rho(CHECK_SEED, hp, net_size=CHECK_NET_SIZE, timesteps=CHECK_TIMESTEPS_RHO)

    print("\nEvaluating Architecture B (hard gate) on this SAME g/rho_oversight, for comparison:")
    hard_gate = HardGatePolicy(g_model, rho_model)
    b_metrics = rollout(as_hard_gate_policy_fn(hard_gate), n_episodes=100, seed=5000,
                         needs_oversight_flag=True)
    print(f"  B: success_rate={b_metrics['success_rate']:.2f} "
          f"violation_rate={b_metrics['violation_rate']:.2f}")

    print("\nLearning curve, gate (Architecture C):")
    steps_c, succ_c, viol_c, gate_model = gate_learning_curve(g_model, rho_model, hp)

    routing_diagnostic(g_model, rho_model, gate_model)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(steps_a, succ_a, marker="o", label="A")
    axes[0].plot(steps_g, succ_g, marker="o", label="g")
    axes[0].plot(steps_c, succ_c, marker="o", label="C (learned gate)")
    axes[0].axhline(b_metrics["success_rate"], color="tab:orange", linestyle="-.",
                     label="B (hard gate, same g/rho)")
    axes[0].axhline(rand["success_rate"], color="gray", linestyle="--", label="random baseline")
    axes[0].axhline(greedy["success_rate"], color="black", linestyle=":", label="greedy-ignore-oversight")
    axes[0].set_xlabel("training steps")
    axes[0].set_ylabel("success_rate")
    axes[0].set_title("Does success_rate actually climb?")
    axes[0].legend(fontsize="small")

    axes[1].plot(steps_a, viol_a, marker="o", label="A")
    axes[1].plot(steps_r, viol_r, marker="o", label="rho_oversight")
    axes[1].plot(steps_c, viol_c, marker="o", label="C (learned gate)")
    axes[1].axhline(b_metrics["violation_rate"], color="tab:orange", linestyle="-.",
                     label="B (hard gate, same g/rho)")
    axes[1].axhline(rand["violation_rate"], color="gray", linestyle="--", label="random baseline")
    axes[1].set_xlabel("training steps")
    axes[1].set_ylabel("violation_rate")
    axes[1].set_title("Does violation_rate actually drop?")
    axes[1].legend(fontsize="small")

    fig.tight_layout()
    fig.savefig("results/sanity_check_oversight_learning_curves.png", dpi=150)
    print("\nSaved results/sanity_check_oversight_learning_curves.png")
