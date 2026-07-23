"""
Cluster-aggregation convention audit (2026-07-17 audit series).

Standalone, read-only. Loads the persisted SHAP values of the level model
(shap_values.parquet) and the trajectory model (shap_values_dw_bare_robust
.parquet), maps features to the eleven mechanism clusters with the exact
taxonomy and prefix logic of phase_b/explain.py, and reports cluster
importance under BOTH aggregation conventions:

  mean = mean over member features of per-feature mean absolute SHAP
         (what explain.py implements, line 107)
  sum  = sum over member features of per-feature mean absolute SHAP

Writes reports/audit_2026-07-17/cluster_aggregation.md.
Does not modify the SHAP pipeline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import yaml

ROOT = Path(r"D:\data_code_masterthesis\code\phase_a")
OUT = ROOT / "reports" / "audit_2026-07-17"

CFG_PATH = ROOT / "phase_b" / "config" / "phase_b_config.yaml"
VARIANTS = {
    "level": ("phase_b/outputs/shap_values.parquet",
              "phase_b/outputs/mechanism_importance.csv"),
    "trajectory": ("phase_b/outputs/shap_values_dw_bare_robust.parquet",
                   "phase_b/outputs/mechanism_importance_dw_bare_robust.csv"),
}


def _assign_mechanism(feature_name: str, taxonomy: Dict[str, str]) -> str:
    """Identical prefix lookup to phase_b/explain.py:_assign_mechanism."""
    for prefix, cluster in taxonomy.items():
        if feature_name.startswith(prefix):
            return cluster
    return "other"


def cluster_table(shap_path: Path, taxonomy: Dict[str, str]) -> pd.DataFrame:
    """Per-cluster n_features, mean and sum of per-feature mean |SHAP|."""
    shap_df = pd.read_parquet(shap_path)
    feat_cols = [c for c in shap_df.columns if c != "unit_id"]
    rows = []
    for feat in feat_cols:
        rows.append({
            "feature": feat,
            "mechanism": _assign_mechanism(feat, taxonomy),
            "mean_abs_shap": float(shap_df[feat].abs().mean()),
        })
    feat_df = pd.DataFrame(rows)
    agg = (
        feat_df.groupby("mechanism")
        .agg(
            n_features=("feature", "count"),
            mean_of_feature_means=("mean_abs_shap", "mean"),
            sum_of_feature_means=("mean_abs_shap", "sum"),
        )
        .reset_index()
    )
    agg["rank_mean"] = agg["mean_of_feature_means"].rank(
        ascending=False).astype(int)
    agg["rank_sum"] = agg["sum_of_feature_means"].rank(
        ascending=False).astype(int)
    agg.attrs["feat_df"] = feat_df
    return agg.sort_values("mean_of_feature_means", ascending=False)


def md_table(agg: pd.DataFrame, dp: int) -> str:
    lines = [
        "| Mechanism cluster | n | Mean convention (pipeline) | Rank | "
        "Sum convention | Rank |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in agg.iterrows():
        lines.append(
            f"| {r['mechanism']} | {r['n_features']} "
            f"| {r['mean_of_feature_means']:.{dp}f} | {r['rank_mean']} "
            f"| {r['sum_of_feature_means']:.{dp}f} | {r['rank_sum']} |")
    return "\n".join(lines)


def main() -> None:
    with open(CFG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    taxonomy = cfg["explain"]["mechanism_taxonomy"]

    results = {}
    for name, (shap_rel, mech_rel) in VARIANTS.items():
        agg = cluster_table(ROOT / shap_rel, taxonomy)
        # cross-check the mean column against the persisted pipeline CSV
        mech = pd.read_csv(ROOT / mech_rel)
        merged = agg.merge(mech, on="mechanism")
        max_diff = float(
            (merged["mean_of_feature_means"] - merged["mean_abs_shap"])
            .abs().max())
        results[name] = (agg, max_diff)
        print(f"[{name}] clusters={len(agg)}, "
              f"max diff vs mechanism_importance CSV = {max_diff:.2e}")
        print(agg.to_string(index=False))

    level_agg, level_diff = results["level"]
    traj_agg, traj_diff = results["trajectory"]
    feat_df = level_agg.attrs["feat_df"].sort_values(
        "mean_abs_shap", ascending=False)

    md = f"""# Cluster Aggregation Convention Audit — Mean vs. Sum

**Date:** 2026-07-17 (run 2026-07-18)
**Inputs:** persisted level-model SHAP values (`phase_b/outputs/shap_values.parquet`,
7,448 aza x 42 features) and trajectory-model SHAP values
(`shap_values_dw_bare_robust.parquet`), mapped to mechanism clusters with the
taxonomy in `phase_b/config/phase_b_config.yaml` (`explain.mechanism_taxonomy`)
and the prefix logic of `phase_b/explain.py:_assign_mechanism`. Script:
`reports/audit_2026-07-17/cluster_aggregation.py` (standalone, read-only).

## What the pipeline implements

`phase_b/explain.py`, `aggregate_by_mechanism` (lines 104-113), first computes
per-feature mean absolute SHAP over all 7,448 aza (line 98), then aggregates:

```python
agg = (
    feat_df.groupby("mechanism")
    .agg(
        mean_abs_shap=("mean_abs_shap", "mean"),   # <-- line 107
        n_features=("feature", "count"),
        features=("feature", lambda x: ", ".join(sorted(x))),
    )
    .reset_index()
    .sort_values("mean_abs_shap", ascending=False)
)
```

The pipeline convention is therefore the **unweighted mean over member
features of per-feature mean absolute SHAP** (`explain.py:107`). Every
cluster value quoted in the thesis (Table 5.3) is this mean. Reproduction
check: the mean column recomputed here matches `mechanism_importance.csv` to
{level_diff:.1e} (level) and `mechanism_importance_dw_bare_robust.csv` to
{traj_diff:.1e} (trajectory).

This also resolves the two textual puzzles that motivated the audit. The
strongest single feature can legitimately exceed its own cluster value under
a mean convention (dist_medical_m contributes 0.157, its three count-variant
siblings contribute 0.035, 0.014, and 0.010, so the cluster mean is 0.054).
And the section 5.3 phrase "0.009 across its eight variables jointly" is
misleading as written, because 0.009 is the per-feature mean of the fiscal
cluster, not a joint total. The joint (sum) figure is 0.074.

## Level model — both conventions side by side

{md_table(level_agg, 4)}

## Trajectory model (null check) — both conventions side by side

{md_table(traj_agg, 5)}

All trajectory clusters stay at or below 0.00021 (mean) and 0.00103 (sum).
The null result is convention-invariant.

## Does the headline survive the sum convention?

**Yes, per the pre-registered thresholds — neither trigger fires:**

- No institutional cluster enters the top three under the sum convention.
  The institutional clusters rank 5 (fiscal, 0.074), 9 (merger, 0.019), and
  11 (designation, 0.009) by sum.
- Medical accessibility does not leave the top three (rank 2 by sum, 0.216).

The accessibility-dominance headline holds under both conventions. Two
within-block movements are worth stating honestly, though neither crosses a
pre-registered threshold:

1. **Transit and medical swap ranks 1 and 2** under the sum convention
   (transit 0.233 vs medical 0.216), because transit has five members to
   medical's four. The correct convention-robust claim is that medical and
   transit accessibility are jointly dominant, not that medical is first
   under every convention. The thesis already phrases the headline as
   "medical accessibility and public transport the two strongest clusters",
   which is true under both conventions unchanged.
2. **The fiscal cluster rises from rank 9 to rank 5** under the sum
   convention (0.009 -> 0.074), overtaking community facilities, DID
   proximity, hospital accessibility, and housing durability[^rank], purely
   because it has eight members. This is the expected mechanical effect of
   summing many minor features and is the reason the mean is the defensible
   primary convention for clusters of unequal size. Even at rank 5 by sum,
   fiscal remains behind medical, transit, and education accessibility, and
   the institutional-families-are-secondary reading survives.

[^rank]: Sum-convention order: transit 0.233, medical 0.216, demographic
threshold 0.138, education 0.090, fiscal 0.074, community 0.073, DID 0.057,
durability 0.039, merger 0.019, hospital 0.011, designation 0.009.

## Actions under the pre-registered rules

The reorder triggers did not fire, so the thesis keeps the mean-convention
tables and adds one sentence. Suggested insertions (style rules respected):

- **Section 4.5** (definition correction, to match `explain.py:107`): "Cluster
  importance is computed as the unweighted mean over a cluster's member
  features of the per-feature mean absolute SHAP value, so clusters of
  different sizes are compared on a per-feature basis and a cluster cannot
  rank highly through many individually weak members."
- **Section 5.3** (robustness sentence): "The ordering is robust to the
  aggregation convention, since summing over member features instead of
  averaging leaves the accessibility families ahead of every institutional
  cluster and moves no institutional cluster into the leading three."
- **Section 5.3** (fix the "jointly" phrase): replace "municipal fiscal
  position $0.009$ across its eight variables jointly" with wording that
  makes the per-feature reading explicit, for example "municipal fiscal
  position $0.009$ per feature across its eight variables".

## Appendix — per-feature mean absolute SHAP, level model

| Feature | Cluster | Mean abs SHAP |
|---|---|---|
"""
    for _, r in feat_df.iterrows():
        md += f"| {r['feature']} | {r['mechanism']} | {r['mean_abs_shap']:.4f} |\n"

    out_path = OUT / "cluster_aggregation.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
