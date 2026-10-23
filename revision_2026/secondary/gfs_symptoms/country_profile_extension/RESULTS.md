# Country attitude profiles and later anxiety/depressive symptoms

Executed post-primary exploratory extension; protocol saved before its outcome fits. These are new analyses of the existing GFS cohort, not independent replication. Manuscript integration is pending.

Nine countries; 17,186 paired women for design-aware country means and 16,926 complete cases for prediction and adjusted bereavement models.

The main feature is the historical female no-education minus higher-education difference in acceptance, averaged across five reasons (percentage points). This is between-group variation in published margins, not respondent variability. Primary outcomes are follow-up 2024 GAD-2 and PHQ-2 scores (0–6); country sample size is nine.

## Main country associations

| Outcome | Spearman rho | Exact permutation p | Holm p (2 tests) | Country-deletion rho range |
|---|---:|---:|---:|---|
| gad2 | -0.167 | 0.6777 | 1.0000 | -0.524 to 0.024 |
| phq2 | 0.183 | 0.6436 | 1.0000 | -0.143 to 0.333 |

Country means and intervals use longitudinal weights and the full adult survey design before the women-domain subset. These rank tests give countries equal weight and do not propagate uncertainty in country means or DHS contrasts. Secondary baseline/change/other-feature correlations form a separate 16-test family in `country_correlations.csv`; do not select one as a new primary finding.

## Held-out-country prediction

Each outcome model includes its baseline symptom score and individual covariates. Each context model adds only one country feature. Fixed ridge alpha=10; all preprocessing is training-only, all women are used, and training countries have equal total longitudinal weight. MAE is weighted within held-out country and averaged equally across countries. Negative MAE change means improvement.

| Outcome | Added feature | MAE change | Countries improved / 9 |
|---|---|---:|---:|
| gad2 | education_gap | +0.0151 | 3 |
| gad2 | national_acceptance | +0.0014 | 4 |
| gad2 | rural_gap | +0.0087 | 1 |
| phq2 | education_gap | +0.0060 | 2 |
| phq2 | national_acceptance | +0.0023 | 0 |
| phq2 | rural_gap | +0.0035 | 3 |

| Outcome | Reference/model | Mean MAE | Mean R2 |
|---|---|---:|---:|
| gad2 | baseline_persistence | 1.6381 | -0.4465 |
| gad2 | education_gap | 1.5322 | 0.0034 |
| gad2 | individual | 1.5170 | 0.0208 |
| gad2 | national_acceptance | 1.5185 | 0.0203 |
| gad2 | rural_gap | 1.5257 | 0.0078 |
| gad2 | training_mean | 1.6234 | -0.0817 |
| phq2 | baseline_persistence | 1.6257 | -0.5162 |
| phq2 | education_gap | 1.4336 | 0.0183 |
| phq2 | individual | 1.4276 | 0.0258 |
| phq2 | national_acceptance | 1.4299 | 0.0233 |
| phq2 | rural_gap | 1.4311 | 0.0214 |
| phq2 | training_mean | 1.4981 | -0.0508 |

Fold training samples overlap. These differences are descriptive; there is no fold-level significance test. This is retrospective transfer to a withheld country, not a deployment-time forecast. A small average change is not evidence of useful prediction without examining countries and baselines.

## Secondary bereavement diagnostic

Country-specific follow-up symptom coefficients adjust for the matching baseline score and measured individual covariates. The table correlates those estimated bereavement coefficients with the education contrast. This is a two-stage exploratory diagnostic, not a definitive pooled interaction test. Coefficient uncertainty is shown in the figure and source table but is not propagated through the rank test.

| Outcome | Spearman rho | Exact p | Holm p (2 tests) | Country-deletion rho range |
|---|---:|---:|---:|---|
| gad2 | 0.050 | 0.9116 | 0.9867 | -0.357 to 0.310 |
| phq2 | 0.267 | 0.4933 | 0.9867 | -0.048 to 0.619 |

## What these results support

The primary education-contrast correlations provide no clear evidence of a country-level relationship with later anxiety or depressive symptoms. Every tested one-feature context addition worsens average held-out-country MAE relative to its corresponding individual model. The secondary bereavement diagnostic is also inconclusive. These findings do not support making a positive country-norms-to-mental-health prediction claim. They do not establish absence of an individual relationship, or rule out other well-measured contextual features. With only nine country units and historical, heterogeneous context measurements, the extension is best treated as a transparent exploratory boundary analysis.

## Interpretation limits

- Positive country associations cannot show that an individual who accepts violence has worse mental health. GFS has no linked individual attitude or IPV measurement here.
- COVID exposure is reported bereavement, not infection or a dated event. This cannot estimate pandemic-caused anxiety, depression, or clinical diagnoses.
- Historical DHS source years differ; Egypt covers ever-married women and the other source inventories all women aged 15–49. GFS covers adult women, with different populations, translations and follow-up selection.
- Baseline symptom adjustment concerns residual follow-up differences and may condition on symptoms that already reflect bereavement. It does not estimate the total causal effect.
- Employment/marital contrasts with unavailable categories were excluded before fitting, not filled with substitute categories. No community or respondent variability is inferred from published margins.
- Only nine selected country units support the country-level question. Deep embeddings cannot increase that number. These analyses cannot by themselves establish journal suitability.

## Reproduction

Run from the repository root with the recorded Python/R environments:

```bash
python revision_2026/04_mental_health/gfs_public/country_profile_extension/run_extension.py prepare
Rscript revision_2026/04_mental_health/gfs_public/country_profile_extension/estimate_country_outcomes.R
python revision_2026/04_mental_health/gfs_public/country_profile_extension/run_extension.py analyze
```

Do not include respondent-level `design_input.csv` in a review archive. All outputs in this directory are country-level summaries or code; the input manifest records local source hashes.
