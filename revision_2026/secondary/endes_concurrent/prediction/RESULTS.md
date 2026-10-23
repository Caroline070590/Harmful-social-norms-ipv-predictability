# ENDES: individual attitudes, IPV and concurrent depression prediction

20 October 2026. Executed post-primary exploratory analysis. This study uses the official public INEI ENDES2023 release, not the historical Kaggle aggregate table. Source preparation and prediction have separate frozen protocols and independent QC.

## Population and target

The eligible observed linked cohort contains **10,267 women aged18–49**, across **3,116 PSUs** and **239 released strata**. They completed the linked individual/health/violence interviews, were ever married/cohabiting, and had the required valid attitude, PHQ and recent-IPV responses and native covariates. Eligibility and all exclusions are in the preparation ledgers.

The health and violence modules use separate selection mechanisms. No official joint-intersection population weight was verified. Primary error therefore gives each included woman equal weight and targets these observed complete respondents. It is not national prevalence, a population effect or nationally representative prediction performance. Released individual/health weights were never multiplied or substituted. Religion was entirely unavailable; the unused emotional-insult field was not imputed. The adult scope differs from the original aggregate women15–49 target.

The PHQ-9 sum is concurrent14-day depressive symptom burden, range0–27, not a diagnosis. Recent emotional/physical/sexual IPV concern the preceding12months. These cross-sectional observations cannot show that attitudes preceded abuse or depression, identify causality, or estimate a pandemic effect.

## Fixed comparison

M0 includes age, age squared and native education, wealth, current employment, marital status, residence and region. M1 adds three recent-IPV indicators. M2 adds the0–5 justification total. M3 adds four fixed orthonormal scenario contrasts, retaining the five original answers without a redundant copy of the total. M4 tests attitude information before IPV adjustment.

Every comparison uses the same women and fivefold PSU holdouts under three fixed assignment seeds. All women in a PSU stay together. Preprocessing is fitted on training respondents only. Ridge alpha10 is the primary algorithm; a fixed shallow histogram gradient booster is secondary. No hyperparameters were selected using evaluation outcomes. The primary prediction averages each woman's three genuinely out-of-fold raw predictions before bounding to0–27. All raw/bounded and assignment-specific results remain available.

## Main result

Recent IPV improved the primary ridge model's RMSE from **4.133349 to 4.001252 PHQ-9 points**; the paired difference was **-0.132096**, with descriptive interval **-0.158795 to -0.105047**.

Adding the full individual attitude information changed RMSE from **4.001252 to 4.001376**. The primary M3-minus-M1 difference was **+0.000124**, with descriptive interval **-0.002459 to +0.002634**. This does not establish incremental predictive value. The fixed boosting comparison also shows a very small difference; overall endorsement and scenario-pattern additions remain unconvincing in this cohort.

| Direction | Interpretation |
|---|---|
| Negative paired error difference | Lower error after adding the block |
| Positive paired error difference | Higher error after adding the block |

| Feature block | Ridge RMSE | Boosting RMSE | Ridge MAE | Boosting MAE |
|---|---:|---:|---:|---:|
| Demographics | 4.133349 | 4.134369 | 2.999252 | 3.002064 |
| + Recent IPV | 4.001252 | 4.004459 | 2.870196 | 2.874007 |
| + Endorsement total | 4.001414 | 4.004762 | 2.870365 | 2.874496 |
| + Scenario pattern | 4.001376 | 4.004268 | 2.871482 | 2.874317 |
| Demographics + attitudes | 4.132555 | 4.133391 | 2.998976 | 3.000793 |

| Algorithm | Added block comparison | RMSE difference | 95% descriptive interval |
|---|---|---:|---|
| ridge | M3 minus M1 | +0.000124 | -0.002459 to +0.002634 |
| ridge | M2 minus M1 | +0.000162 | -0.000070 to +0.000399 |
| ridge | M3 minus M2 | -0.000038 | -0.002606 to +0.002468 |
| ridge | M1 minus M0 | -0.132096 | -0.158795 to -0.105047 |
| ridge | M4 minus M0 | -0.000794 | -0.003985 to +0.002134 |
| hist_gradient_boosting | M3 minus M1 | -0.000191 | -0.001947 to +0.001509 |
| hist_gradient_boosting | M2 minus M1 | +0.000303 | -0.000882 to +0.001420 |
| hist_gradient_boosting | M3 minus M2 | -0.000494 | -0.001666 to +0.000657 |
| hist_gradient_boosting | M1 minus M0 | -0.129910 | -0.156957 to -0.103904 |
| hist_gradient_boosting | M4 minus M0 | -0.000979 | -0.003380 to +0.001206 |

Intervals resample observed PSUs within released strata while holding the existing out-of-fold predictions fixed. They are descriptive conditional intervals, do not include full refitting uncertainty, and do not substitute for joint-module population weights. No significance-only or causal conclusion is made.

## Why the null attitude increment is bounded

**10,098/10,267 women endorsed none of the justifications; only 169 endorsed at least one.** This restricted endorsement variation is a central measurement/generalization limitation. A negligible average prediction increment here is not evidence that personal beliefs or social norms are unimportant, nor that the result generalizes to settings with higher endorsement. No post-result subgroup search was used to rescue prediction.

The aggregate attitude study and this individual symptom task have different targets, units, populations and validation designs. Their R2 values must not be compared as if they quantify the same problem. A country-similarity GNN does not provide an individual psychological mechanism. The separately benchmarked GNNs are supporting analyses in the supplement.

## Reproducibility

`FIT_INPUT_MANIFEST.json` freezes cohort, protocol/config, preparation QC and fitting code. `results/INDEPENDENT_QC.json` records the exact scope of source/fold/model/metric/bootstrap verification. Respondent IDs, fold membership, fitted model binaries and individual predictions remain only in excluded `data_local/`.

Every tabulated value binds to CSV/filter/column/hash in `SOURCE_VALUES.json`. The figure uses the saved numerical tables and is exported as vectorPDF/SVG plus600-dpiPNG. This is a bounded completed prediction study, not a national prevalence analysis or submission-readiness certificate.
