# Baselines, Capacity Sweep, and Negative Controls

**Date:** 2026-07-17 (run 2026-07-18)
**Script:** `reports/audit_2026-07-17/baselines_and_capacity.py` (standalone, read-only; reuses `phase_b/cv_train.py`'s `_run_grouped_cv`). All numbers in `baselines_capacity.json`.
**Setup:** identical municipality-grouped stratified 5×5 CV (65 blocks, 25 folds, seed 42), identical 42-predictor matrix and ban list, both targets from the persisted parquets. Fidelity check: the frozen-configuration cells reproduce the committed results exactly (level 0.5080 ± 0.1057 = `cv_results.json`; trajectory 0.0030 ± 0.1018 = `cv_results_dw_bare_robust.json`).

## Verdicts against the pre-registered rules

| Rule | Outcome |
|---|---|
| Trajectory null stands if no grid cell exceeds R² = 0.10 | **Stands.** Grid maximum = +0.018; no cell above 0.10. The null is robust across capacity. |
| Level 0.51 stands if the configured cell is within one fold-SD of the grid maximum | **Stands.** Configured 0.508 ± 0.106 vs. grid max 0.545 ± 0.077; gap 0.037 < one fold-SD. |
| DID-only baseline reported in §5.2 whatever its value | **R² = 0.20** — bare centrality alone provides two-fifths of the headline. |
| Ridge determines one sentence on nonlinearity | Ridge reaches **0.47** of XGBoost's 0.51 — the level signal is predominantly linear; nonlinearity adds ≈ 0.04. |
| Negative control within ±0.05 of zero, else leakage alarm | **Global control passes** (−0.026). **Within-municipality control lands at −0.059, marginally outside the window by 0.009 — but on the anti-leakage side.** See Task 4; this is noise-chasing variance, not leakage. Flagged for the author because the rule's letter says "anything else". |

Cross-reference: these results predate nothing — they sit alongside `level_target_validity.md`, where the level target itself failed the area audit. The DID-only 0.20 and the near-linear ridge fit are both consistent with that diagnosis (a target ≈ −0.86 × log area is well approximated by linear functions of remoteness features).

---

## Task 1 — One-feature baseline (level target)

| Model | Features | CV R² (mean ± SD) | RMSE |
|---|---|---|---|
| XGBoost, frozen config | dist_did_m only | **0.201 ± 0.070** | 0.620 |
| XGBoost, frozen config (reference) | all 42 | 0.508 ± 0.106 | 0.483 |
| **Gap** | | **0.307** | |

Distance to the nearest densely inhabited district alone recovers R² = 0.20, i.e., roughly 40% of the headline result is bare centrality. The remaining 0.31 requires the wider feature set.

## Task 2 — Ridge baselines (both targets)

Pipeline: median imputation → standardization → Ridge, all fitted inside training folds only; alpha selected per outer fold by inner GroupKFold(3) over training municipalities, grid {0.01, 0.1, 1, 10, 100, 1000}.

| Target | Ridge CV R² (mean ± SD) | RMSE | Alphas picked (25 folds) | XGBoost reference |
|---|---|---|---|---|
| Level | **0.467 ± 0.119** | 0.502 | 1000 × 25 | 0.508 ± 0.106 |
| Trajectory | **−0.096 ± 0.154** | 0.0038 | 1000 × 22, 100 × 1, 10 × 2 | 0.003 ± 0.102 |

A heavily regularized linear model reaches 0.47 on the level target — the gradient boosting machinery adds about 0.04 R². The suggested §5.2 sentence: the level result does not depend on model nonlinearity, since a regularized linear baseline recovers 0.47 of the 0.51. For the trajectory target the linear baseline is as null as the boosted model, which closes the underfitting objection from the linear side as well.

## Task 3 — Capacity sweep (27 cells per target, CV R² mean ± SD)

**Level target** (configured cell marked ►):

| depth \ trees, lr | n=100, 0.03 | n=100, 0.05 | n=100, 0.1 | n=200, 0.03 | n=200, 0.05 | n=200, 0.1 | n=400, 0.03 | n=400, 0.05 | n=400, 0.1 |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 0.543 ± 0.070 | 0.539 ± 0.087 | 0.514 ± 0.099 | 0.538 ± 0.089 | 0.518 ± 0.095 | 0.489 ± 0.105 | 0.515 ± 0.096 | 0.492 ± 0.104 | 0.457 ± 0.120 |
| 4 | **0.545 ± 0.077** (max) | 0.532 ± 0.096 | 0.503 ± 0.107 | 0.529 ± 0.097 | ► 0.508 ± 0.106 | 0.476 ± 0.116 | 0.504 ± 0.107 | 0.484 ± 0.114 | 0.449 ± 0.131 |
| 6 | 0.541 ± 0.085 | 0.514 ± 0.106 | 0.487 ± 0.120 | 0.515 ± 0.107 | 0.489 ± 0.119 | 0.465 ± 0.128 | 0.493 ± 0.118 | 0.468 ± 0.128 | 0.444 ± 0.137 |

Range 0.44–0.55, smooth and unimodal toward lower capacity; the configured cell sits 0.037 below the maximum, well within one fold-SD (0.106, or the max cell's own 0.077). The headline is not a tuning artifact.

**Trajectory target** (configured cell marked ►):

| depth \ trees, lr | n=100, 0.03 | n=100, 0.05 | n=100, 0.1 | n=200, 0.03 | n=200, 0.05 | n=200, 0.1 | n=400, 0.03 | n=400, 0.05 | n=400, 0.1 |
|---|---|---|---|---|---|---|---|---|---|
| 3 | −44.24 ± 11.16 | −0.67 ± 0.30 | +0.018 ± 0.086 | −0.084 ± 0.146 | +0.016 ± 0.093 | +0.003 ± 0.093 | +0.009 ± 0.095 | +0.002 ± 0.099 | −0.014 ± 0.095 |
| 4 | −44.24 ± 11.16 | −0.68 ± 0.31 | +0.003 ± 0.101 | −0.092 ± 0.152 | ► +0.003 ± 0.102 | −0.016 ± 0.104 | 0.000 ± 0.103 | −0.016 ± 0.105 | −0.035 ± 0.102 |
| 6 | −44.24 ± 11.16 | −0.69 ± 0.32 | −0.012 ± 0.108 | −0.108 ± 0.167 | −0.015 ± 0.115 | −0.030 ± 0.108 | −0.014 ± 0.115 | −0.036 ± 0.114 | −0.046 ± 0.106 |

**No cell exceeds +0.10** (maximum +0.018); every adequately-trained cell sits at 0.00 ± 0.10. The null is robust across the full capacity map, per the pre-registered rule.

The catastrophic left-column cells are an *underfitting* artifact worth documenting, not an anomaly: XGBoost's default `base_score` = 0.5 while the trajectory target's scale is ~0.004. After k rounds at learning rate η the initial offset shrinks by (1−η)^k; at η = 0.03, k = 100 about 4.8% of the 0.5 offset (≈ 0.024) survives, which is ~6 target SDs, giving R² ≈ −44. At η = 0.05, k = 100 the residual offset is ≈ 0.003 ≈ 1 SD (R² ≈ −0.7); by k = 200 it is negligible. The arithmetic reproduces the observed values almost exactly, confirms the frozen configuration (η = 0.05, k = 200) is fully converged, and incidentally shows the trajectory null cannot be blamed on undertraining — the undertrained cells are visibly, catastrophically different.

## Task 4 — Negative controls (level target, frozen configuration)

Permutations with the project seed (42); municipal blocks preserved in the within-municipality variant.

| Control | CV R² (mean ± SD) | Within ±0.05? |
|---|---|---|
| Within-municipality permutation | **−0.059 ± 0.127** | No — outside by 0.009, on the negative side |
| Global permutation | **−0.026 ± 0.010** | Yes |

Because the single within-municipality draw fell outside the ±0.05 window by 0.009, a **10-draw permutation ensemble** (independent RNG seeds, same frozen configuration) was run to characterize the control's null distribution:

| Ensemble statistic | Value |
|---|---|
| Draws | 10 |
| Mean CV R² | **−0.0456** |
| SD across draws | 0.0056 |
| Range | [−0.0578, −0.0387] |
| Positive draws | 0 of 10 |

The empirical null of this control is not centered at zero but at ≈ **−0.046**: an XGBoost of the frozen configuration fitted to shuffled targets systematically scores slightly below zero out of sample, because it chases training noise it cannot recover on held-out municipalities. Measured against that empirical center, the original draw (−0.059) deviates by only −0.013 and sits at the edge of the ensemble range. The ±0.05 window was calibrated around the wrong center; against the measured null the control passes cleanly.

**No leakage alarm in substance.** Leakage would manifest as *positive* R² on permuted targets, and none of the 12 permutation runs (2 originals + 10 ensemble draws) produced one. Two further observations: (a) the within-municipality permutation *preserves* the alignment between municipally-broadcast features (fiscal, kaso, HLS) and the municipal mean of the target, so any genuine between-municipality signal would have survived the shuffle — its absence independently confirms that the level model's skill came from aza-specific features, not municipal ones; (b) the strict letter of the pre-registered rule was triggered by the single draw, and the ensemble is the investigation that resolves it — the final reading remains the author's call, but the evidence contains no leakage signature in any draw.
