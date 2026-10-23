# ENDES: fixed training in 2023 and testing in 2024

20 October 2026. Executed post-primary exploratory replication. **The primary later-year comparison does not establish a clear incremental prediction gain from attitude information. A small error difference is not evidence of equivalence or absence of psychological relevance.**

## What was tested

Models trained on **10,259 women with both actual interviews in 2023** predict concurrent 14-day PHQ-9 symptoms in **9,847 eligible women interviewed in 2024**. The official INEI sources contain same-person attitude, recent-IPV and symptom responses. This is a later-year repeated-cross-section prediction test; it is not a within-person symptom forecast or a Kaggle microdata analysis.

The date audit excluded **8 later-interview records** from the older release before fitting. The original 2023 release-based analyses and their cohort remain unchanged. Training and preprocessing use only the strict earlier-year respondents. The separate source review documents Public/ODbL access, linkage, source item definitions, actual dates and the preserved source hashes.

No joint DV–health intersection population weight was verified. Errors therefore give each observed complete eligible woman equal weight. They do not estimate national performance, prevalence or a population effect. Source PSU panels are documented; the full later-year test retains shared contexts. The fixed stricter sensitivity retains only PSUs absent from every 2023 released household PSU. Public annual person keys can recur; personal identity and movements are not established by those keys. Neither full-sample nor new-PSU results certify universal individual independence.

| Test domain | Women | PSUs | Strata | Any endorsers |
|---|---:|---:|---:|---:|
| All 2024 respondents | 9,847 | 3,072 | 239 | 122 |
| New PSUs | 3,250 | 1,025 | 210 | 45 |
| No endorsement | 9,725 | 3,065 | 239 | 0 |
| Any endorsement | 122 | 111 | 60 | 122 |

## Primary result and fixed sensitivity

In all eligible 2024 women, primary ridge RMSE changes from **3.972419** with demographics/recent IPV to **3.972847 PHQ-9 points** with all five attitude responses. The paired M3−M1 difference is **+0.000428**; its 95% conditional interval is **-0.002161 to +0.002890**. Negative differences favour the added information.

In the new-PSU domain the corresponding difference is **-0.003188**, with interval **-0.008085 to +0.000983**. This fixed sensitivity tests a stricter change of context and remains a selected observed subset.

The measured recent-IPV addition changes ridge RMSE from **4.084572** to **3.972419**; its M1−M0 difference is **-0.112153**, with interval **-0.138828 to -0.086165**. Prediction accuracy does not establish exposure causality or an intervention effect.

## Fixed methods and interpretation

M0 contains age, age squared and native education, wealth, work, marital status, residence and region. M1 adds measured preceding 12-month emotional/physical/sexual IPV. M2 adds the justification total; M3 adds four fixed scenario contrasts, preserving all five responses; M4 adds attitude information to demographics without IPV. Ridge and the secondary gradient booster retain the original locked settings; the training-mean reference is reported alongside every model. **10 outcome models** were fitted once; later outcomes were never used for tuning.

The outcome is a 0–27 symptom sum, not a clinical diagnosis. Selection into the observed complete cohort is restrictive; religion and unused emotional item D103C are entirely unavailable. Source-compatible coding does not establish psychometric invariance. Endorsement is sparse: **122** later-year women endorse any justification, including **45** in the new-PSU subset. The fixed endorsement-domain checks describe performance where attitude variation is present; they are not subgroup rescue searches or proof of effect modification. In the no-endorsement domain, every attitude response is zero; any change between predictions reflects parameters learned from the full training cohort, not informative attitude differences within that domain. A tiny favourable conditional difference there is not a substantive within-group attitude benefit. No conclusion that norms are psychologically irrelevant follows from a weak increment.

Intervals use **2,000** whole-test-PSU resamples within released strata with seed **20261021**, holding training models and predictions fixed. The full test has **19 singleton strata** in the resampling units. Domain interval readiness uses the pre-fit minimum counts; unavailable intervals are left unavailable. The intervals omit model-refitting, module-selection and population-weight uncertainty and are explicitly conditional on these observed respondents. They are not prospectively calibrated clinical prediction intervals.

## All fixed performance results

The primary presentation bounds predictions to the symptom scale; raw error and clipping are retained. Every raw/bounded RMSE, MAE, bias and R² is available in `results/metrics.csv`. Aggregate attitude errors and these respondent symptom errors address different units and targets and must not be directly compared.

### All 2024 respondents

| Algorithm / feature block | Bounded RMSE | Raw RMSE | Bounded MAE | Bias | R² | Clipped rows |
|---|---:|---:|---:|---:|---:|---:|
| Training mean | 4.155659 | 4.155659 | 3.070833 | +0.082361 | -0.000393 | 0 |
| Ridge, M0 | 4.084572 | 4.084573 | 2.993963 | +0.094240 | 0.033540 | 1 |
| Ridge, M1 | 3.972419 | 3.972420 | 2.873451 | +0.086460 | 0.085885 | 2 |
| Ridge, M2 | 3.972433 | 3.972434 | 2.873273 | +0.086129 | 0.085879 | 2 |
| Ridge, M3 | 3.972847 | 3.972901 | 2.874243 | +0.085464 | 0.085688 | 4 |
| Ridge, M4 | 4.085451 | 4.085546 | 2.994987 | +0.092352 | 0.033124 | 3 |
| Gradient boosting, M0 | 4.080291 | 4.080291 | 2.997734 | +0.095957 | 0.035565 | 0 |
| Gradient boosting, M1 | 3.967590 | 3.967590 | 2.871497 | +0.075956 | 0.088106 | 0 |
| Gradient boosting, M2 | 3.965872 | 3.965872 | 2.870280 | +0.072987 | 0.088896 | 0 |
| Gradient boosting, M3 | 3.965661 | 3.965661 | 2.869676 | +0.072676 | 0.088993 | 0 |
| Gradient boosting, M4 | 4.077957 | 4.077957 | 2.994761 | +0.092022 | 0.036668 | 0 |

### New PSUs

| Algorithm / feature block | Bounded RMSE | Raw RMSE | Bounded MAE | Bias | R² | Clipped rows |
|---|---:|---:|---:|---:|---:|---:|
| Training mean | 4.299229 | 4.299229 | 3.118902 | -0.051509 | -0.000144 | 0 |
| Ridge, M0 | 4.209999 | 4.209999 | 3.034541 | -0.029632 | 0.040941 | 0 |
| Ridge, M1 | 4.074789 | 4.074789 | 2.904889 | -0.011076 | 0.101555 | 0 |
| Ridge, M2 | 4.074789 | 4.074789 | 2.904678 | -0.011176 | 0.101555 | 0 |
| Ridge, M3 | 4.071601 | 4.071681 | 2.903485 | -0.011435 | 0.102960 | 1 |
| Ridge, M4 | 4.205850 | 4.206051 | 3.032810 | -0.030590 | 0.042831 | 1 |
| Gradient boosting, M0 | 4.203691 | 4.203691 | 3.037456 | -0.037046 | 0.043813 | 0 |
| Gradient boosting, M1 | 4.072871 | 4.072871 | 2.907926 | -0.022397 | 0.102401 | 0 |
| Gradient boosting, M2 | 4.069645 | 4.069645 | 2.904643 | -0.027127 | 0.103822 | 0 |
| Gradient boosting, M3 | 4.067945 | 4.067945 | 2.903596 | -0.026691 | 0.104571 | 0 |
| Gradient boosting, M4 | 4.196999 | 4.196999 | 3.033458 | -0.039659 | 0.046855 | 0 |

### No endorsement

| Algorithm / feature block | Bounded RMSE | Raw RMSE | Bounded MAE | Bias | R² | Clipped rows |
|---|---:|---:|---:|---:|---:|---:|
| Training mean | 4.141875 | 4.141875 | 3.064832 | +0.096842 | -0.000547 | 0 |
| Ridge, M0 | 4.071293 | 4.071295 | 2.988321 | +0.107619 | 0.033263 | 1 |
| Ridge, M1 | 3.958311 | 3.958312 | 2.868107 | +0.095329 | 0.086175 | 2 |
| Ridge, M2 | 3.958264 | 3.958265 | 2.867603 | +0.093794 | 0.086196 | 2 |
| Ridge, M3 | 3.958068 | 3.958070 | 2.866581 | +0.090651 | 0.086286 | 2 |
| Ridge, M4 | 4.070985 | 4.070987 | 2.986333 | +0.101641 | 0.033410 | 1 |
| Gradient boosting, M0 | 4.066715 | 4.066715 | 2.991841 | +0.108864 | 0.035436 | 0 |
| Gradient boosting, M1 | 3.953426 | 3.953426 | 2.865925 | +0.084824 | 0.088429 | 0 |
| Gradient boosting, M2 | 3.952498 | 3.952498 | 2.864618 | +0.079042 | 0.088856 | 0 |
| Gradient boosting, M3 | 3.952750 | 3.952750 | 2.863753 | +0.077338 | 0.088740 | 0 |
| Gradient boosting, M4 | 4.066375 | 4.066375 | 2.988521 | +0.098921 | 0.035597 | 0 |

### Any endorsement

| Algorithm / feature block | Bounded RMSE | Raw RMSE | Bounded MAE | Bias | R² | Clipped rows |
|---|---:|---:|---:|---:|---:|---:|
| Training mean | 5.136795 | 5.136795 | 3.549180 | -1.071928 | -0.045528 | 0 |
| Ridge, M0 | 5.031540 | 5.031540 | 3.443682 | -0.972194 | -0.003121 | 0 |
| Ridge, M1 | 4.969802 | 4.969802 | 3.299399 | -0.620537 | 0.021345 | 0 |
| Ridge, M2 | 4.973675 | 4.973675 | 3.325255 | -0.524903 | 0.019820 | 0 |
| Ridge, M3 | 5.012640 | 5.015999 | 3.484992 | -0.328010 | 0.004401 | 2 |
| Ridge, M4 | 5.108462 | 5.114472 | 3.684874 | -0.648127 | -0.034026 | 2 |
| Gradient boosting, M0 | 5.046319 | 5.046319 | 3.467474 | -0.932892 | -0.009022 | 0 |
| Gradient boosting, M1 | 4.968406 | 4.968406 | 3.315687 | -0.631009 | 0.021895 | 0 |
| Gradient boosting, M2 | 4.916228 | 4.916228 | 3.321621 | -0.409656 | 0.042331 | 0 |
| Gradient boosting, M3 | 4.886307 | 4.886307 | 3.341767 | -0.298945 | 0.053953 | 0 |
| Gradient boosting, M4 | 4.914113 | 4.914113 | 3.492213 | -0.457952 | 0.043155 | 0 |

## All fixed paired RMSE comparisons

| Domain | Algorithm | Comparison | Δ RMSE | 95% conditional interval |
|---|---|---|---:|---|
| All 2024 respondents | Ridge | M3 − M1 | +0.000428 | -0.002161 to +0.002890 |
| All 2024 respondents | Ridge | M2 − M1 | +0.000014 | -0.000273 to +0.000264 |
| All 2024 respondents | Ridge | M3 − M2 | +0.000414 | -0.002046 to +0.002892 |
| All 2024 respondents | Ridge | M1 − M0 | -0.112153 | -0.138828 to -0.086165 |
| All 2024 respondents | Ridge | M4 − M0 | +0.000879 | -0.002499 to +0.004164 |
| All 2024 respondents | Gradient boosting | M3 − M1 | -0.001929 | -0.003968 to +0.000032 |
| All 2024 respondents | Gradient boosting | M2 − M1 | -0.001719 | -0.003244 to -0.000204 |
| All 2024 respondents | Gradient boosting | M3 − M2 | -0.000211 | -0.001803 to +0.001288 |
| All 2024 respondents | Gradient boosting | M1 − M0 | -0.112701 | -0.138215 to -0.087411 |
| All 2024 respondents | Gradient boosting | M4 − M0 | -0.002335 | -0.005112 to +0.000295 |
| New PSUs | Ridge | M3 − M1 | -0.003188 | -0.008085 to +0.000983 |
| New PSUs | Ridge | M2 − M1 | -0.000001 | -0.000354 to +0.000352 |
| New PSUs | Ridge | M3 − M2 | -0.003188 | -0.007964 to +0.000898 |
| New PSUs | Ridge | M1 − M0 | -0.135210 | -0.184520 to -0.087593 |
| New PSUs | Ridge | M4 − M0 | -0.004149 | -0.010208 to +0.001288 |
| New PSUs | Gradient boosting | M3 − M1 | -0.004926 | -0.009013 to -0.001314 |
| New PSUs | Gradient boosting | M2 − M1 | -0.003226 | -0.006080 to -0.000795 |
| New PSUs | Gradient boosting | M3 − M2 | -0.001700 | -0.004497 to +0.000944 |
| New PSUs | Gradient boosting | M1 − M0 | -0.130821 | -0.175894 to -0.087050 |
| New PSUs | Gradient boosting | M4 − M0 | -0.006692 | -0.012277 to -0.002060 |
| No endorsement | Ridge | M3 − M1 | -0.000242 | -0.000490 to -0.000015 |
| No endorsement | Ridge | M2 − M1 | -0.000047 | -0.000090 to -0.000003 |
| No endorsement | Ridge | M3 − M2 | -0.000196 | -0.000426 to +0.000013 |
| No endorsement | Ridge | M1 − M0 | -0.112983 | -0.140249 to -0.086542 |
| No endorsement | Ridge | M4 − M0 | -0.000308 | -0.000573 to -0.000044 |
| No endorsement | Gradient boosting | M3 − M1 | -0.000676 | -0.001916 to +0.000672 |
| No endorsement | Gradient boosting | M2 − M1 | -0.000928 | -0.002046 to +0.000222 |
| No endorsement | Gradient boosting | M3 − M2 | +0.000251 | -0.001143 to +0.001539 |
| No endorsement | Gradient boosting | M1 − M0 | -0.113289 | -0.139292 to -0.088246 |
| No endorsement | Gradient boosting | M4 − M0 | -0.000340 | -0.002021 to +0.001296 |
| Any endorsement | Ridge | M3 − M1 | +0.042838 | -0.113961 to +0.233128 |
| Any endorsement | Ridge | M2 − M1 | +0.003872 | -0.012797 to +0.023470 |
| Any endorsement | Ridge | M3 − M2 | +0.038966 | -0.117296 to +0.215952 |
| Any endorsement | Ridge | M1 − M0 | -0.061738 | -0.275811 to +0.115841 |
| Any endorsement | Ridge | M4 − M0 | +0.076921 | -0.132476 to +0.307250 |
| Any endorsement | Gradient boosting | M3 − M1 | -0.082099 | -0.171109 to +0.013012 |
| Any endorsement | Gradient boosting | M2 − M1 | -0.052178 | -0.111756 to +0.010909 |
| Any endorsement | Gradient boosting | M3 − M2 | -0.029921 | -0.079415 to +0.019512 |
| Any endorsement | Gradient boosting | M1 − M0 | -0.077913 | -0.282626 to +0.081804 |
| Any endorsement | Gradient boosting | M4 − M0 | -0.132207 | -0.262304 to +0.002297 |

The saved paired table also retains MAE differences and conditional intervals. No new p-value, equivalence or clinical-usefulness claim is added.

## Contribution and limits

This extension tests whether the individual attitude/IPV question reproduces across an actual later calendar year and a fixed set of new PSUs. Its strength is original respondent measurements, strict temporal separation and transparent incremental prediction. It is not a new GNN architecture, a causal pathway or a demonstration that model complexity improves the paper. These results should remain a bounded secondary extension of the primary cross-national attitude study. Journal suitability still depends on the psychological question and the magnitude and generalizability of the evidence.

`FIT_INPUT_MANIFEST.json` freezes the earlier/later inputs, date rules, preparation QC, protocol, configuration and fitting code. `results/INDEPENDENT_QC.json` records exact model/metric/interval verification. Respondent records, identifiers, trained binary models and individual predictions remain in excluded`data_local/`. `SOURCE_VALUES.json` binds report and figure values to source tables/JSON and hashes. This analysis does not alter the existing manuscript or earlier archives.
