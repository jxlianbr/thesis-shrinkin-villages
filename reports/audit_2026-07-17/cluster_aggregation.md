# Cluster Aggregation Convention Audit — Mean vs. Sum

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
9.7e-17 (level) and `mechanism_importance_dw_bare_robust.csv` to
8.8e-17 (trajectory).

This also resolves the two textual puzzles that motivated the audit. The
strongest single feature can legitimately exceed its own cluster value under
a mean convention (dist_medical_m contributes 0.157, its three count-variant
siblings contribute 0.035, 0.014, and 0.010, so the cluster mean is 0.054).
And the section 5.3 phrase "0.009 across its eight variables jointly" is
misleading as written, because 0.009 is the per-feature mean of the fiscal
cluster, not a joint total. The joint (sum) figure is 0.074.

## Level model — both conventions side by side

| Mechanism cluster | n | Mean convention (pipeline) | Rank | Sum convention | Rank |
|---|---|---|---|---|---|
| accessibility_medical | 4 | 0.0539 | 1 | 0.2157 | 2 |
| accessibility_transit | 5 | 0.0465 | 2 | 0.2326 | 1 |
| demographic_threshold | 4 | 0.0344 | 3 | 0.1376 | 3 |
| accessibility_did | 2 | 0.0283 | 4 | 0.0567 | 7 |
| accessibility_community | 4 | 0.0181 | 5 | 0.0726 | 6 |
| accessibility_education | 5 | 0.0180 | 6 | 0.0898 | 4 |
| accessibility_hospital | 1 | 0.0111 | 7 | 0.0111 | 10 |
| institutional_merger | 2 | 0.0096 | 8 | 0.0192 | 9 |
| institutional_fiscal | 8 | 0.0092 | 9 | 0.0735 | 5 |
| durability_housing | 5 | 0.0078 | 10 | 0.0392 | 8 |
| institutional_policy | 2 | 0.0045 | 11 | 0.0091 | 11 |

## Trajectory model (null check) — both conventions side by side

| Mechanism cluster | n | Mean convention (pipeline) | Rank | Sum convention | Rank |
|---|---|---|---|---|---|
| durability_housing | 5 | 0.00021 | 1 | 0.00103 | 1 |
| accessibility_transit | 5 | 0.00015 | 2 | 0.00074 | 2 |
| institutional_policy | 2 | 0.00006 | 3 | 0.00012 | 8 |
| institutional_fiscal | 8 | 0.00006 | 4 | 0.00047 | 3 |
| accessibility_did | 2 | 0.00006 | 5 | 0.00012 | 9 |
| demographic_threshold | 4 | 0.00005 | 6 | 0.00022 | 4 |
| accessibility_medical | 4 | 0.00005 | 7 | 0.00021 | 5 |
| accessibility_hospital | 1 | 0.00004 | 8 | 0.00004 | 11 |
| accessibility_community | 4 | 0.00004 | 9 | 0.00018 | 6 |
| institutional_merger | 2 | 0.00003 | 10 | 0.00007 | 10 |
| accessibility_education | 5 | 0.00003 | 11 | 0.00016 | 7 |

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
| dist_medical_m | accessibility_medical | 0.1574 |
| n_bus_stop_1000m | accessibility_transit | 0.0872 |
| elderly_flag_severely | demographic_threshold | 0.0862 |
| dist_bus_stop_m | accessibility_transit | 0.0835 |
| dist_did_m | accessibility_did | 0.0481 |
| household_size | demographic_threshold | 0.0479 |
| dist_bus_route_m | accessibility_transit | 0.0438 |
| dist_community_facility_m | accessibility_community | 0.0430 |
| dist_elementary_m | accessibility_education | 0.0393 |
| n_medical_1000m | accessibility_medical | 0.0349 |
| dist_school_m | accessibility_education | 0.0343 |
| fin_fiscal_strength_index | institutional_fiscal | 0.0276 |
| n_community_facility_5000m | accessibility_community | 0.0186 |
| hls_pre1981_ratio | durability_housing | 0.0180 |
| years_since_merger | institutional_merger | 0.0168 |
| hls_vacancy_rate | durability_housing | 0.0145 |
| n_medical_3000m | accessibility_medical | 0.0137 |
| fin_std_fiscal_revenue | institutional_fiscal | 0.0136 |
| fin_local_alloc_tax | institutional_fiscal | 0.0121 |
| dist_hospital_m | accessibility_hospital | 0.0111 |
| n_bus_stop_5000m | accessibility_transit | 0.0104 |
| fin_local_tax | institutional_fiscal | 0.0102 |
| n_medical_5000m | accessibility_medical | 0.0097 |
| in_did | accessibility_did | 0.0085 |
| n_school_3000m | accessibility_education | 0.0085 |
| kaso_type | institutional_policy | 0.0085 |
| n_bus_stop_3000m | accessibility_transit | 0.0077 |
| fin_real_balance | institutional_fiscal | 0.0068 |
| n_community_facility_3000m | accessibility_community | 0.0067 |
| n_school_5000m | accessibility_education | 0.0043 |
| n_community_facility_1000m | accessibility_community | 0.0043 |
| hls_vacancy_other_rate | durability_housing | 0.0042 |
| elderly_flag_shrinking | demographic_threshold | 0.0034 |
| n_school_1000m | accessibility_education | 0.0034 |
| hls_total_dwellings | durability_housing | 0.0025 |
| merged_flag | institutional_merger | 0.0024 |
| fin_total_revenue | institutional_fiscal | 0.0015 |
| fin_total_expenditure | institutional_fiscal | 0.0010 |
| fin_std_fiscal_need | institutional_fiscal | 0.0006 |
| kaso_flag | institutional_policy | 0.0006 |
| household_size_small | demographic_threshold | 0.0001 |
| hls_missing | durability_housing | 0.0000 |
