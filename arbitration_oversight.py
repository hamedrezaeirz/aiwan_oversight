"""
arbitration_oversight.py
-------------------------
Combines g and rho_oversight into a single action at rollout/evaluation
time, for the oversight-signal experiment. Mirrors arbitration.py's
structure exactly, with one deliberate structural difference from the
original: Architecture B's gate here is a HARD STEP FUNCTION, not a
sigmoid.

Why the gate is different from the original DualLoopPolicy:
Per NEXT_STEP_oversight_signal.md's confirmed design decision, the
oversight signal is binary (on/off), not graded like battery level.
A human "stop now" command is a sudden, discrete event -- there's no
principled analogue of lambda(b) = sigmoid(k*(b - b_mid)) here, because
there's no continuous "how urgent is it" quantity to feed a sigmoid;
oversight_active is already exactly 0 or 1. So Architecture B's
arbitration rule is simply:

    execute rho_oversight's action IF oversight_active ELSE g's action

This is what a "hardcoded gate that reads the raw signal directly, with
no learning involved" means for THIS signal type -- the fact that it's
a step function rather than a sigmoid is a consequence of the signal
being binary, not a weaker or different design choice than the original
Architecture B. Architecture C (LearnedGatePolicy_Oversight) is
otherwise structurally identical to the original: the gate still SEES
the full observation (not just oversight_active) and is still free to
learn a decision boundary that isn't purely oversight-driven -- exactly
the property this architecture exists to test for fold-back.
"""

import numpy as np


class HardGatePolicy:
    """
    Architecture B for the oversight signal: a hardcoded, non-learned
    step-function gate. Deterministic given oversight_active -- no
    randomness, unlike the original DualLoopPolicy's probabilistic
    lambda(b) selection. This is the correct analogue for a binary
    signal (see module docstring): there is nothing to sample
    stochastically when the underlying signal is already 0/1.
    """

    def __init__(self, g_model, rho_model):
        self.g_model = g_model
        self.rho_model = rho_model

    def predict(self, obs, oversight_active, deterministic=True):
        """
        oversight_active must be passed in as a plain bool/0-1 value,
        read directly from the environment's info dict or its own
        internal state -- NOT inferred from obs, to keep this gate
        transparently "hardcoded" (reads the raw signal, no learned
        interpretation of obs), mirroring how DualLoopPolicy took
        battery_level as an explicit argument rather than parsing it out
        of obs itself.
        """
        sub_model = self.rho_model if oversight_active else self.g_model
        action, _ = sub_model.predict(obs, deterministic=deterministic)
        return action


class LearnedGatePolicy_Oversight:
    """
    Architecture C: arbitration is a learned 2-action policy
    (gate_model, trained via env_oversight.LearnedGateEnv_Oversight /
    train_oversight.train_gate) choosing between g's and rho_oversight's
    action at each step, instead of HardGatePolicy's hardcoded
    if/else. Structurally identical to the original LearnedGatePolicy:
    the gate sees the full observation (including but not limited to
    oversight_active) and was optimized on the combined objective, so it
    can in principle learn a decision boundary that isn't purely
    oversight-driven -- exactly the design property this architecture
    exists to test for fold-back generalization.
    """

    def __init__(self, g_model, rho_model, gate_model):
        self.g_model = g_model
        self.rho_model = rho_model
        self.gate_model = gate_model

    def predict(self, obs, deterministic=True):
        gate_action, _ = self.gate_model.predict(obs, deterministic=deterministic)
        gate_action = int(gate_action)
        sub_model = self.rho_model if gate_action == 1 else self.g_model
        action, _ = sub_model.predict(obs, deterministic=deterministic)
        return action


if __name__ == "__main__":
    # Quick manual smoke test -- verifies HardGatePolicy's if/else
    # routing logic in isolation, without needing trained models or
    # SB3/torch at all (uses trivial stand-in "models").
    #   python arbitration_oversight.py

    class _StubModel:
        """Minimal stand-in with the same .predict(obs, deterministic)
        interface as an SB3 model, so this test needs no SB3/torch."""
        def __init__(self, tag):
            self.tag = tag

        def predict(self, obs, deterministic=True):
            return self.tag, None

    g_stub = _StubModel("g_action")
    rho_stub = _StubModel("rho_action")
    gate = HardGatePolicy(g_stub, rho_stub)

    dummy_obs = np.zeros(6, dtype=np.float32)
    assert gate.predict(dummy_obs, oversight_active=False) == "g_action"
    assert gate.predict(dummy_obs, oversight_active=True) == "rho_action"
    assert gate.predict(dummy_obs, oversight_active=0) == "g_action"
    assert gate.predict(dummy_obs, oversight_active=1) == "rho_action"
    print("HardGatePolicy routing test passed: "
          "oversight_active correctly selects rho_action vs g_action, "
          "including truthy/falsy int forms (0/1), not just bool True/False.")
