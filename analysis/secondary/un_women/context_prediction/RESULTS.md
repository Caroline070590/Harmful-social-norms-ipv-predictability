# UN Women country-context prediction: exploratory results

Executed under a protocol saved before these fits; earlier GFS and Colombian results were already known. No main-manuscript integration has been performed.

## Sample and question

This compares retrospective prediction of Phase II worry (0–3) and depressive-item sums (0–6), using different women in separate countries. The modified since-COVID recall is not a standard GAD/PHQ scale or a measurement of symptom change. Historical national attitudes are contextual features, not individual violence exposure.

- `national_8`: 9,898 complete women, 8 countries: Albania, Bangladesh, Colombia, Cote d'Ivoire/Ivory Coast, Jordan, Kyrgyz Republic, Morocco, Nigeria.
- `complete_profiles_6`: 7,524 complete women, 6 countries: Albania, Bangladesh, Colombia, Cote d'Ivoire/Ivory Coast, Morocco, Nigeria.

The same women are used for both outcomes and all models within each cohort. Cohorts differ, so compare models within a cohort. Supplied weights are used for training/evaluation; countries have equal total training mass and equal weight in the reported mean errors. All category fitting/scaling is restricted to training countries. Fixed ridge alpha=10; no tuning or embeddings.

## Out-of-country error

| Cohort | Outcome | Model | Mean MAE | Mean R2 |
|---|---|---|---:|---:|
| complete_profiles_6 | depressive_item_sum | education_contrast_pp | 1.7026 | -0.3806 |
| complete_profiles_6 | depressive_item_sum | individual | 1.5733 | -0.1677 |
| complete_profiles_6 | depressive_item_sum | national_female_acceptance | 1.6322 | -0.2612 |
| complete_profiles_6 | depressive_item_sum | rural_contrast_pp | 1.7375 | -0.4538 |
| complete_profiles_6 | depressive_item_sum | training_mean | 1.6073 | -0.1667 |
| complete_profiles_6 | worry_item | education_contrast_pp | 1.0084 | -0.2688 |
| complete_profiles_6 | worry_item | individual | 0.9583 | -0.1019 |
| complete_profiles_6 | worry_item | national_female_acceptance | 0.9810 | -0.1845 |
| complete_profiles_6 | worry_item | rural_contrast_pp | 1.0186 | -0.3059 |
| complete_profiles_6 | worry_item | training_mean | 0.9760 | -0.0834 |
| national_8 | depressive_item_sum | individual | 1.5654 | -0.1834 |
| national_8 | depressive_item_sum | national_female_acceptance | 1.6362 | -0.2927 |
| national_8 | depressive_item_sum | training_mean | 1.5869 | -0.1808 |
| national_8 | worry_item | individual | 0.9529 | -0.0829 |
| national_8 | worry_item | national_female_acceptance | 0.9748 | -0.1495 |
| national_8 | worry_item | training_mean | 0.9778 | -0.0890 |

## Increment from country context

| Cohort | Outcome | Added country feature | Mean MAE change | Countries improved |
|---|---|---|---:|---:|
| complete_profiles_6 | depressive_item_sum | education_contrast_pp | +0.1294 | 2/6 |
| complete_profiles_6 | depressive_item_sum | national_female_acceptance | +0.0589 | 1/6 |
| complete_profiles_6 | depressive_item_sum | rural_contrast_pp | +0.1642 | 3/6 |
| complete_profiles_6 | worry_item | education_contrast_pp | +0.0500 | 3/6 |
| complete_profiles_6 | worry_item | national_female_acceptance | +0.0227 | 2/6 |
| complete_profiles_6 | worry_item | rural_contrast_pp | +0.0603 | 4/6 |
| national_8 | depressive_item_sum | national_female_acceptance | +0.0707 | 2/8 |
| national_8 | worry_item | national_female_acceptance | +0.0219 | 1/8 |

Negative MAE change means lower error. These selected-country results and overlapping training folds do not provide a precise population effect, a causal estimate, or a percentage of symptoms explained by country conditions. All country-level changes are saved; do not select only a favorable feature or country.

## Interpretation limits

- This is prediction of cross-sectional reported scores in withheld countries, not a forecast of future depression/anxiety or a test of COVID causing symptoms.
- Historical country estimates have differing years and populations, including ever-married women in Bangladesh/Jordan and differing source age ranges. UN Women includes adult women aged 18 and over. Individual age categories are retained, not silently matched to DHS age bands.
- Work activity and other individual predictors retain exact UN Women categories; no equivalence with DHS cash/kind employment groups is assumed. Generic married and spouse-residence categories remain distinct. Unknown held-out categories are logged.
- Country counts are eight and six. Country profiles cannot explain the Colombia-only direct-IPV coefficient or test how it varies by national context. Neither a favorable nor unfavorable prediction increment establishes an individual attitude–IPV–mental-health pathway.
- Full original PSU/stratum information and religion are unavailable in this release. No inferential design-based confidence intervals, p values, clinical cutoffs or causal effects are estimated here.
- Ridge outputs are not clipped. Any out-of-range predictions, negative R2 values and failures to outperform the training-mean reference remain in the tables.
- This is a narrowly specified exploratory check. A favorable result is not enough to justify a high-capacity country embedding or a claim of clinical utility.

## Reproduction

From the repository root:

```bash
python revision_2026/04_mental_health/women_covid_vaw/context_prediction/run_comparison.py
python revision_2026/04_mental_health/women_covid_vaw/context_prediction/check_comparison.py
```

Respondent inputs stay in `data_local/`. This directory contains aggregate outputs and code. Read `INDEPENDENT_QC.json` before treating results as numerically verified.
