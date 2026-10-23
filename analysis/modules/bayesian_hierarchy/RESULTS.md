# Bayesian national mean and justification-profile prediction

This is a completed retrospective exploratory comparison of national female component percentages. It does not model individual endorsers or experienced violence. The author requested execution after the separate ENDES Ising test; the frozen start record documents that order.

## Primary comparisons

| Target domain | Countries | B0 component MAE | B1 component MAE | B2 component MAE | B1−B0 raw mean MAE [conditional interval] | B2−B1 raw profile MAE [conditional interval] |
|---|---:|---:|---:|---:|---|---|
| Strict country/time | 42 | 11.653 | 8.958 | 8.916 | -2.893 [-4.563, -1.206] | -0.068 [-0.250, +0.118] |
| External MICS | 19 | 12.607 | 7.242 | 7.140 | -5.586 [-8.539, -2.356] | -0.054 [-0.372, +0.256] |

Component MAE columns use predictions clipped to 0–100 percentage points. Mean/profile contrasts use exact raw posterior means to separate learned context effects from clipping. Negative loss differences favor the candidate. Intervals resample fixed country errors; they are not posterior parameter intervals, survey-design confidence intervals or prospective confirmation.

B0 has calendar and partially pooled donor-country effects. B1 adds the four structural predictors to the national mean only. B2 additionally models four fixed orthonormal centered-profile coordinates. B0/B1 raw profiles and orderings are identical by construction; any bounded-profile difference comes from clipping. B1/B2 share identical raw mean predictions.

## All fixed donor/prior sensitivities

| Donor case | Residual prior | Target domain | B1−B0 raw mean MAE | B2−B1 raw profile MAE | B2−B1 raw ordering accuracy |
|---|---|---|---|---|---|
| questionnaire_compatible | primary | chronological42 | -2.893 [-4.563, -1.206] | -0.068 [-0.250, +0.118] | -0.024 [-0.048, -0.000] |
| questionnaire_compatible | primary | external19 | -5.586 [-8.539, -2.356] | -0.054 [-0.372, +0.256] | +0.021 [-0.028, +0.081] |
| questionnaire_compatible | smaller_variance_prior | chronological42 | -2.770 [-4.421, -1.111] | -0.065 [-0.247, +0.123] | -0.022 [-0.044, +0.000] |
| questionnaire_compatible | smaller_variance_prior | external19 | -5.472 [-8.413, -2.234] | -0.076 [-0.386, +0.221] | +0.015 [-0.031, +0.064] |
| questionnaire_compatible | larger_variance_prior | chronological42 | -2.975 [-4.681, -1.261] | -0.071 [-0.247, +0.115] | -0.017 [-0.039, +0.005] |
| questionnaire_compatible | larger_variance_prior | external19 | -5.680 [-8.660, -2.413] | -0.039 [-0.399, +0.309] | +0.033 [-0.000, +0.074] |
| omit_india | primary | chronological42 | -2.923 [-4.684, -1.239] | -0.048 [-0.237, +0.135] | -0.029 [-0.055, -0.005] |
| omit_india | primary | external19 | -5.618 [-8.555, -2.459] | -0.055 [-0.390, +0.263] | +0.026 [-0.023, +0.089] |
| omit_india | smaller_variance_prior | chronological42 | -2.792 [-4.531, -1.151] | -0.044 [-0.232, +0.142] | -0.025 [-0.051, +0.000] |
| omit_india | smaller_variance_prior | external19 | -5.500 [-8.410, -2.371] | -0.073 [-0.398, +0.233] | +0.021 [-0.033, +0.086] |
| omit_india | larger_variance_prior | chronological42 | -3.023 [-4.835, -1.302] | -0.047 [-0.231, +0.134] | -0.024 [-0.048, -0.002] |
| omit_india | larger_variance_prior | external19 | -5.725 [-8.683, -2.521] | -0.039 [-0.417, +0.321] | +0.023 [-0.021, +0.072] |

All fixed settings are retained. Differences across priors are sensitivity descriptions, not a search for a favorable result. India omission changes donor pools, not targets.

## Predictive uncertainty and inference

The Gaussian likelihood treats each published round as one observation, with repeated-country covariance. It does not invent respondent sample sizes, binomial counts or standard errors. The new country’s random effect, residual variance, coefficient uncertainty and prespecified finite variance-ratio prior are integrated. Exact normal–inverse-gamma algebra and finite mixtures produce point means; no MCMC or ADVI is used.

| Target domain | Model | Marginal 95% scenario coverage | Mean width (pp) | Raw predictive mass outside 0–100 |
|---|---|---:|---:|---:|
| chronological42 | B0 | 0.976 | 78.6 | 0.224 |
| chronological42 | B1 | 0.976 | 57.9 | 0.156 |
| chronological42 | B2 | 0.976 | 57.8 | 0.157 |
| external19 | B0 | 0.958 | 78.8 | 0.224 |
| external19 | B1 | 0.989 | 57.7 | 0.227 |
| external19 | B2 | 0.989 | 57.6 | 0.228 |

These are model-conditional intervals for a new reported aggregate round under Gaussian/basis/prior assumptions. They do not recover national survey-design uncertainty. Coverage is retrospective and computed across correlated held-out evaluations. Gaussian support permits impossible percentages; out-of-range mass and raw/bounded losses are disclosed. Independent coordinate priors make results depend on the fixed Helmert basis; no psychological factor structure is established.

Saved execution contains 1098 target/model/prior fits, 5490 coordinate blocks and 27450 conditional variance-ratio blocks. Fixed seeded predictive draws are discarded after aggregate interval/order summaries. Of 1098 interval precision checks, 0 remain above the declared endpoint tolerance; 30 required the prespecified larger simulation. Point means and all loss comparisons are analytic and unaffected by this Monte Carlo precision check.

## Limits and manuscript role

All targets were previously inspected; this is an exploratory extension. Questionnaire equivalence, national population coverage and source publication availability remain limited. The hierarchy is Gaussian with independent mean/contrast blocks and a finite prior on variance ratios. Earlier weighted-ridge references use identical training rows/targets but a different fitting assumption. Any useful profile result must survive raw-output checks and fixed sensitivities and remain separate from individual attitudes, causal influence or mental-health mechanisms.

The active manuscript and independent final export are preserved. This study is analysis-only pending scientific interpretation; no favorable result or integration is assumed.

See [protocol](PROTOCOL.md), [complete model summary](results/model_summary.csv), [all comparisons](results/paired_comparisons.csv), [coverage](results/predictive_coverage.csv), [fixed hierarchy identities](results/HIERARCHY_INVARIANTS.json) and independent review for exact verification scope.
