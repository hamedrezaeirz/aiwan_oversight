"""
train_oversight.py
-------------------
Training functions for the oversight-signal generalization experiment.
Mirrors train.py's structure and naming exactly (train_architecture_a,
train_g, train_rho, train_gate, get_untrained_*) so run_experiment.py's
logic can be reused with minimal changes -- only the env module, the
wrapper classes, and the algorithm differ.

Algorithm: PPO, not DQN. The oversight env's action space is continuous
(Box(2,), see env_oversight.py) -- DQN requires discrete actions, so it
doesn't apply here. PPO is chosen over SAC for this first pass because
it's simpler to get working reliably on a low-dimensional continuous
control task like this one (no replay buffer / off-policy tuning to get
right), per SB3's own guidance; SAC remains a fallback if PPO proves
hard to tune. This choice should be revisited if PPO's sanity-check
learning curve (next step after this file) looks poor.
"""

from stable_baselines3 import PPO
from env_oversight import (
    ContinuousNavOversightEnv,
    RewardWrapperA_Oversight,
    RewardWrapperGoalOnly_Oversight,
    RewardWrapperOversightOnly,
    LearnedGateEnv_Oversight,
)


def _make_ppo(env, net_arch, seed, hp):
    """
    Shared PPO settings for every training run in this experiment --
    defined once here, mirroring train.py's _make_dqn. Reads
    learning_rate and gamma from hp (to come from an oversight-specific
    tune_hyperparams pass, not hand-picked), same discipline as the
    original project's methodology note (§7): nothing here should be
    chosen after looking at final results.

    log_std_init: now read from hp (default -1.0), the value manually
    found to work reasonably during this project's iterative sanity-check
    debugging (log_std_init=-2.0 was tried first and over-corrected,
    crushing exploration everywhere and collapsing success_rate to ~0).
    Exposed as a proper hyperparameter here so
    tune_hyperparams_oversight.py can search it systematically instead
    of it staying a hand-picked constant found by manual trial and error.

    n_steps/batch_size use SB3's PPO defaults (2048/64) for this first
    pass -- revisit only if the sanity-check learning curve shows
    instability or clearly poor sample efficiency; per-hyperparameter
    tuning of these two specifically is not yet in scope.
    """
    return PPO(
        "MlpPolicy",
        env,
        policy_kwargs={"net_arch": net_arch, "log_std_init": hp.get("log_std_init", -1.0)},
        learning_rate=hp.get("learning_rate", 3e-4),
        gamma=hp.get("gamma", 0.99),
        seed=seed,
        verbose=0,
    )


def train_architecture_a(net_size, timesteps, seed, hp,
                          adversarial_weight=0.0, lambda_oversight=None):
    """
    Architecture A: single policy, R = r_goal + lambda_oversight * r_oversight.

    lambda_oversight=None -> comes from hp -> standard condition
    lambda_oversight=0.0  -> indifference scenario
    adversarial_weight>0  -> override/conflict scenario

    Mirrors train.py's train_architecture_a exactly, one function
    covering all three conditions.
    """
    lam = lambda_oversight if lambda_oversight is not None else hp.get("lambda_oversight_a", 1.0)
    env = RewardWrapperA_Oversight(
        ContinuousNavOversightEnv(adversarial_weight=adversarial_weight),
        lambda_oversight=lam,
    )
    model = _make_ppo(env, net_size, seed, hp)
    model.learn(total_timesteps=timesteps)
    return model


def train_g(net_size, timesteps, seed, hp, adversarial_weight=0.0):
    """
    g in Architectures B/C -- sees r_goal only, never r_oversight.
    Indifference changes nothing for g by construction, exactly
    mirroring train.py's train_g.
    """
    env = RewardWrapperGoalOnly_Oversight(
        ContinuousNavOversightEnv(adversarial_weight=adversarial_weight)
    )
    model = _make_ppo(env, net_size, seed, hp)
    model.learn(total_timesteps=timesteps)
    return model


def train_rho(seed, hp, net_size=None, timesteps=None, training_oversight_prob=0.5):
    """
    rho_oversight in Architecture B/C -- trained once per seed, independent
    of the g-power sweep, same as the original train_rho.

    training_oversight_prob=0.5 (vs. the environment's "public"/evaluation
    default of OVERSIGHT_PROB_PER_STEP=0.05): rho_oversight's training
    reward is EXACTLY ZERO whenever oversight is inactive, regardless of
    action -- by the policy gradient theorem, those transitions contribute
    no gradient at all, so at the default 5% activation rate, ~90% of every
    training rollout is uninformative and the ~10% that IS informative is
    split into short 3-step bursts. diagnose_rho_instability.py confirmed
    this makes training genuinely unstable (violation_rate got WORSE, not
    better, with 6x the timestep budget) rather than merely slow.

    Training rho_oversight in an environment with a much higher activation
    rate gives it far denser exposure to the one thing it needs to learn
    ("if oversight_active, output ~zero"), without changing what the RULE
    is -- the rule itself doesn't depend on how often it's triggered, only
    on being triggered often enough during training to learn reliably.
    This mirrors a standard curriculum/domain-randomization-density trick,
    not a change to the task's semantics. The "public" environment used
    everywhere else (evaluation, A, g, the gate) is UNCHANGED and still
    uses the validated OVERSIGHT_PROB_PER_STEP=0.05.

    net_size/timesteps must be passed explicitly (no config_oversight.py
    with defaults exists yet).
    """
    if net_size is None or timesteps is None:
        raise ValueError(
            "train_rho (oversight) currently requires explicit net_size/timesteps "
            "-- no config_oversight.py exists yet with RHO_NETWORK_SIZE/RHO_TIMESTEPS "
            "defaults. Add one before wiring this into run_experiment.py, mirroring "
            "config.py's RHO_NETWORK_SIZE / RHO_TIMESTEPS."
        )
    env = RewardWrapperOversightOnly(
        ContinuousNavOversightEnv(oversight_prob_per_step=training_oversight_prob)
    )
    model = _make_ppo(env, net_size, seed, hp)
    model.learn(total_timesteps=timesteps)
    return model


def train_gate(g_model, rho_model, net_size, timesteps, seed, hp,
               adversarial_weight=0.0, lambda_oversight=None):
    """
    Architecture C (learned gate) for the oversight signal. g_model and
    rho_model are already-trained, fixed policies. Only the gate is
    optimized here, via LearnedGateEnv_Oversight's binary action space,
    on the combined objective -- mirrors train.py's train_gate exactly,
    including the confirmed design rule that the gate's net_size/timesteps
    must scale WITH g's power (same cell), not be fixed independently,
    or the test is foregone by construction (same reasoning as the
    original project's train_gate docstring).

    Note: the gate here is still trained with PPO on a DISCRETE 2-action
    space (0=defer to g, 1=defer to rho_oversight) -- LearnedGateEnv_Oversight
    overrides action_space to Discrete(2), same pattern as the original
    LearnedGateEnv. PPO handles discrete action spaces fine; only the
    underlying g/rho policies need continuous-action support.
    """
    lam = lambda_oversight if lambda_oversight is not None else 1.0
    env = LearnedGateEnv_Oversight(
        ContinuousNavOversightEnv(adversarial_weight=adversarial_weight),
        g_model, rho_model, lambda_oversight=lam,
    )
    model = _make_ppo(env, net_size, seed, hp)
    model.learn(total_timesteps=timesteps)
    return model


def get_untrained_g(net_size, seed, log_std_init=-0.5):
    """
    goal-nulling scenario: reward == 0 means gradient == 0, so training
    is meaningless -- randomly initialize and evaluate that, mirroring
    train.py's get_untrained_g exactly.

    log_std_init defaults to -0.5 (matching the tuned value) purely for
    defensive consistency with the trained models -- under this
    project's convention of always evaluating with deterministic=True,
    log_std has NO effect on the actual evaluated behavior (it only
    matters for stochastic action sampling during training, and this
    model is never trained). Confirmed via the --quick smoke test: an
    untrained network's near-zero-initialized mean output makes it act
    essentially inert regardless of log_std_init. Kept as a parameter
    anyway so this doesn't silently diverge from the trained models if
    evaluation conventions ever change.
    """
    env = RewardWrapperGoalOnly_Oversight(ContinuousNavOversightEnv())
    model = PPO("MlpPolicy", env, policy_kwargs={"net_arch": net_size, "log_std_init": log_std_init},
                seed=seed, verbose=0)
    return model  # no model.learn() call


def get_untrained_architecture_a(net_size, seed, log_std_init=-0.5):
    """
    goal-nulling for Architecture A: mirrors train.py's
    get_untrained_architecture_a exactly. See get_untrained_g's
    docstring for why log_std_init is included here defensively even
    though it has no effect under deterministic evaluation.
    """
    env = RewardWrapperA_Oversight(ContinuousNavOversightEnv(), lambda_oversight=0.0)
    model = PPO("MlpPolicy", env, policy_kwargs={"net_arch": net_size, "log_std_init": log_std_init},
                seed=seed, verbose=0)
    return model  # no model.learn() call


if __name__ == "__main__":
    # Quick manual smoke test with the smallest possible combination --
    # verifies the whole PPO + env_oversight pipeline runs end to end
    # before anything else is built on top of it:
    #   python train_oversight.py
    #
    # NOTE: this only checks the pipeline RUNS without error. It does
    # NOT check that OVERSIGHT_PROB_PER_STEP is well-tuned, or that PPO
    # actually learns a good policy at a real budget -- that's the next
    # step (sanity_check_oversight.py), deliberately not done here.
    hp = {"learning_rate": 3e-4, "gamma": 0.99, "lambda_oversight_a": 1.0}

    print("--- Architecture A (smoke test only, 2_000 timesteps) ---")
    model_a = train_architecture_a(net_size=[16], timesteps=2_000, seed=0, hp=hp)
    env = RewardWrapperA_Oversight(ContinuousNavOversightEnv(), lambda_oversight=1.0)
    obs, _ = env.reset(seed=0)
    total_reward, info = 0.0, {}
    for _ in range(100):
        action, _ = model_a.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    print(f"reached_goal={info['reached_goal']} violated={info['violated']} "
          f"total_reward={total_reward:.2f}")

    print("\n--- g + rho_oversight (smoke test only) ---")
    g_model = train_g(net_size=[16], timesteps=2_000, seed=0, hp=hp)
    rho_model = train_rho(seed=0, hp=hp, net_size=[16], timesteps=2_000)
    print("g_model and rho_model trained without error.")

    print("\n--- Architecture C gate (smoke test only) ---")
    gate_model = train_gate(g_model, rho_model, net_size=[16], timesteps=2_000, seed=0, hp=hp)
    print("gate_model trained without error.")

    print("\nAll smoke tests passed -- pipeline runs end to end. "
          "This does NOT confirm good learning at a real budget; "
          "see sanity_check_oversight.py (next step) for that.")
