"""
evaluate_oversight.py
------------------------
Rollout + metrics for the oversight experiment. Extracted from
sanity_check_oversight.py's inline rollout()/as_*_policy_fn() helpers
(previously duplicated there) so run_experiment_oversight.py can share
the exact same evaluation code path, mirroring the original project's
evaluate.py.

rollout() doesn't care which architecture it's evaluating -- it only
needs a policy_fn(obs) -> action callable, EXCEPT for HardGatePolicy
(Architecture B), which additionally needs the raw oversight_active
flag (see arbitration_oversight.py's docstring on why that's passed
explicitly rather than inferred from obs). as_policy_fn(),
as_hard_gate_policy_fn(), and as_learned_gate_policy_fn() adapt each
kind of model/policy to rollout()'s interface.
"""

import numpy as np

from env_oversight import ContinuousNavOversightEnv


def as_policy_fn(sb3_model):
    """Adapts a plain SB3 model (Architecture A, or a lone untrained g
    for the goal-nulling scenario) to the (obs) -> action interface."""
    def fn(obs):
        action, _ = sb3_model.predict(obs, deterministic=True)
        return action
    return fn


def as_hard_gate_policy_fn(hard_gate_policy):
    """Adapts a HardGatePolicy (Architecture B) to the
    (obs, oversight_active) -> action interface -- note this one keeps
    the second argument, unlike the other two adapters, because
    HardGatePolicy's routing rule needs the raw signal directly (see
    arbitration_oversight.py)."""
    def fn(obs, oversight_active):
        return hard_gate_policy.predict(obs, oversight_active, deterministic=True)
    return fn


def as_learned_gate_policy_fn(learned_gate_policy):
    """Adapts a LearnedGatePolicy_Oversight (Architecture C) to the
    (obs) -> action interface -- the learned gate gets everything it
    needs through obs alone, unlike HardGatePolicy."""
    def fn(obs):
        return learned_gate_policy.predict(obs, deterministic=True)
    return fn


def rollout(policy_fn, n_episodes, seed=None, adversarial_weight=0.0, needs_oversight_flag=False):
    """
    Runs n_episodes on a fresh ContinuousNavOversightEnv and aggregates
    metrics. Mirrors the original project's evaluate.rollout() closely.

    seed, if given, makes the evaluation batch reproducible: episode i is
    reset with seed + i, so the same call always evaluates the same set
    of goal/agent layouts -- same reproducibility rationale as the
    original project's methodology note (§7).

    needs_oversight_flag=True switches policy_fn's expected signature
    from (obs) -> action to (obs, oversight_active) -> action, for
    HardGatePolicy (Architecture B) specifically -- see
    as_hard_gate_policy_fn()'s docstring.
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


def evaluate_cell(architecture, network_size, timesteps, seed, condition,
                   policy_fn, n_episodes, adversarial_weight=0.0, needs_oversight_flag=False):
    """
    Runs rollout() and packages the result into one row matching this
    experiment's results CSV schema. Doesn't train anything -- training
    happens in run_experiment_oversight.py; this only evaluates an
    already-trained policy_fn under a given condition. Mirrors the
    original project's evaluate.evaluate_cell() exactly.
    """
    metrics = rollout(policy_fn, n_episodes, seed=seed, adversarial_weight=adversarial_weight,
                       needs_oversight_flag=needs_oversight_flag)
    return {
        "architecture": architecture,
        "network_size": str(network_size),
        "training_timesteps": timesteps,
        "seed": seed,
        "condition": condition,
        **metrics,
    }
