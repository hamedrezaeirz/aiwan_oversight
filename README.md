# aiwan_oversight

Oversight-signal generalization experiment for **AIWAN: AI With an Artificial Need**
(Rezaei, 2026). Reported as §7.6 of the paper.

## What this tests

AIWAN's §7.3 result shows that a *learned* arbitration gate between a
goal-seeking policy (g) and a regulatory policy (ρ) loses its safety
advantage as g's optimization power grows — the gate learns to fold
the regulatory signal (battery level) back into its own objective
("fold-back"). §7.5 predicts this generalizes beyond battery level to
*any* externally-imposed constraint on g, including human oversight
signals (corrections, shutdown commands). This repository tests that
prediction directly, on a binary oversight signal (active 5% of steps,
lasting 3 steps once triggered) standing in for a human stop command,
using the same continuous-navigation environment and the same
(network size × training budget) grid as §7.3's battery replication.

**Result (§7.6 of the paper):** partial confirmation. Fold-back
generalizes along the training-budget axis (robust, p<0.0001 at every
network size tested) but *not* along the network-capacity axis (no
consistent narrowing; a slight, borderline-significant reversal at the
largest budget). Even at the largest budget and network size jointly,
the gap does not close to zero, unlike §7.3's battery result.

## Architectures

- **A** — single-loop: `R = r_goal + λ · r_oversight`, reward-shaping baseline.
- **B** — hardcoded gate: defers to ρ whenever the oversight signal is
  active, to g otherwise. A *step function*, not §3.2's sigmoid — the
  correct analogue for a binary signal (there's no continuous
  "urgency" to weigh; see `arbitration_oversight.py`'s docstring).
- **C** — learned gate: a small policy, trained on the combined
  objective, choosing between g's and ρ's action at every step.

## Files

| File | Role |
|---|---|
| `env_oversight.py` | Continuous-navigation environment + binary oversight signal + reward wrappers |
| `arbitration_oversight.py` | `HardGatePolicy` (B) and `LearnedGatePolicy_Oversight` (C) |
| `train_oversight.py` | Training routines for A, g, ρ, and the learned gate |
| `evaluate_oversight.py` | Rollout + metrics, shared by every script below |
| `config_oversight.py` | Grid (3 network sizes × 3 budgets × 20 seeds), override weights, paths |
| `run_experiment_oversight.py` | Orchestrates the full sweep (`--quick` / `--mini` / `--seeds` / full) |
| `tune_hyperparams_oversight.py` | Disjoint-seed hyperparameter search (learning rate, γ, log_std_init, λ) |
| `sanity_check_oversight.py` | Learning curves + gate routing diagnostic, run before committing to the full sweep |
| `diagnose_rho_instability.py`, `diagnose_A_at_full_budget.py` | One-off diagnostics from the project's build history |
| `merge_results_oversight.py` | Merges parallel `--seeds` output shards |
| `check_completeness_oversight.py` | Verifies no missing/duplicated (architecture, network_size, budget, seed, condition) cells |
| `plot_results_oversight.py` | Produces Figures 9–10 and the paired t-tests reported in §7.6 |
| `launch_parallel_oversight.sh`, `backup_aiwan_oversight.sh` | VPS orchestration (2-process split, remote backup) |
| `results/results_oversight.csv` | Final merged results: 20 seeds × 9 cells × 24 rows = 4,320 rows |
| `results/hyperparams_oversight.json` | Tuned hyperparameters used for the real sweep |
| `results/gap_AC_vs_power_oversight.png`, `results/violation_rate_vs_w_ABC_largest.png` | Figures 9–10 |

## Reproducing

```bash
python3 -m venv venv && source venv/bin/activate
pip install stable-baselines3 gymnasium torch matplotlib pandas scipy numpy

# pipeline smoke test (seconds)
python3 run_experiment_oversight.py --quick

# one real-budget cell, full condition set (sanity check before committing to the full sweep)
python3 run_experiment_oversight.py --mini

# full sweep (20 seeds, 9 cells) -- split across processes if desired:
python3 run_experiment_oversight.py --seeds 0 1 2 3 4 5 6 7 8 9
python3 run_experiment_oversight.py --seeds 10 11 12 13 14 15 16 17 18 19

python3 merge_results_oversight.py
python3 check_completeness_oversight.py results/results_oversight.csv
python3 plot_results_oversight.py results/results_oversight.csv
```

## Related repositories

- [`AIWAN-Code`](https://github.com/hamedrezaeirz/AIWAN-Code) — main
  discrete-grid battery/crash experiment (§5–§6 of the paper).
- [`aiwan_scaling_learned_gate`](https://github.com/hamedrezaeirz/aiwan_scaling_learned_gate) —
  §7.3's continuous-environment capability-scaling test on the
  battery signal, of which this repository is the direct
  oversight-signal analogue.

## Citation

If you use this code, please cite the AIWAN paper (Rezaei, 2026) and,
for the theoretical framework, its companion paper AIWBN (Rezaei,
2026b).
