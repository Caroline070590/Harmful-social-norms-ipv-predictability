# Exploratory country-profile and later symptom extension

Written 19 October 2026, before fitting this extension. This is post-primary exploratory work, not preregistration, an independent replication, or a confirmatory test. The author requested both country-profile associations and whether profiles modify the COVID-bereavement association; the former is the main question. Earlier PHQ-4 context results and demographic contrasts have already been inspected.

## Questions and measurements

1. Do historical between-group differences in acceptance of wife beating correlate with later country mean anxiety and depression symptom scores?
2. Do these features improve retrospective held-out-country prediction of women's follow-up scores beyond baseline symptoms and individual covariates?
3. As a secondary diagnostic, do countries with different profiles have different adjusted COVID-bereavement associations with follow-up symptoms?

Use the existing nine overlapping countries and the independently verified historical official DHS aggregate contrasts. Define the primary feature as women's **no-education minus higher-education acceptance**, in percentage points, averaged equally across all five separate standardized reasons. This reverses the sign of the existing higher-minus-none contrast. These overlapping published margins describe between-group contrasts, not respondent variability, individual attitudes, a national composite prevalence, or sampling variance. Secondary features are women's rural-minus-urban contrast averaged over those same five reasons, and the separately published national female any-reason percentage. Do not manufacture national totals by averaging margins. Use each country's original survey consistently for these three features; preserve dates, populations and category definitions. Newer national totals remain a previously executed separate sensitivity; newer totals must not silently be combined with older contrasts.

All nine countries have the two selected contrasts. Employment and marital contrasts are excluded from this small model set before fitting: employment is missing for Turkey and the never-married comparison for Egypt. No category substitution, merging of non-equivalent labels or outcome-guided feature selection is permitted. Save the full availability ledger. Egypt's DHS context covers ever-married women aged 15–49; other source inventories cover all women aged 15–49. GFS respondents are a separate adult cohort.

Outcomes: continuous GAD-2 anxiety and PHQ-2 depressive symptom scores, each 0–6. Follow-up interviews occurred in 2024; baseline occurred in 2022–24. These are screening scores, not diagnosed disorders or post-infection outcomes. COVID exposure is baseline reported death of a family member or close friend, without a verified event date. There is no pre-pandemic comparison and no estimate of mental illness caused by coronavirus.

## Country-level association analysis

Estimate baseline and follow-up score means, and mean follow-up-minus-baseline differences, in the same women with complete four-item scores at both waves and positive longitudinal weights. Construct the survey design on all eligible adult longitudinal-weight records before selecting women and country domains. Use longitudinal weights, country-specific PSU/strata, documented singleton adjustments and t-based intervals. Do not claim national representativeness free of attrition bias or measurement differences.

Two primary exploratory tests: education contrast versus follow-up GAD-2 and versus follow-up PHQ-2. Give each country equal weight in Spearman correlations, enumerate all 9! outcome-rank permutations for exact two-sided permutation p-values, and Holm-adjust across these two tests. A separate secondary family contains all other feature/time/outcome combinations (16 tests). Report country-deletion ranges. Country means' design uncertainty is shown separately; the rank permutation analysis does not propagate DHS measurement error, uncertainty in country mean estimates or translation differences. Do not claim general population-of-countries inference from a selected nine-country sample. No bootstrap correlation interval is planned with only nine countries.

The cross-country sample size is **nine**, regardless of the number of GFS respondents. Follow-up levels may reflect baseline differences. Paired changes and baseline levels are descriptive sensitivity analyses, not randomized change estimates.

## Prediction

Use the existing complete longitudinal model cohort, anticipating 16,926 women. Nine outer folds each exclude one country in full. Predict follow-up GAD-2 and PHQ-2 separately. Fixed ridge alpha=10; fit encoding, imputation and scaling on training countries only. Use all training women, normalize longitudinal training weights to equal total country mass, and use longitudinal-weighted metrics in each held-out country; summarize countries equally. No country identity dummy, target-derived country feature, held-out-label fitting, tuning, or CEBRA is added.

Compare training-mean and baseline-score-persistence references, individual covariates plus the corresponding baseline symptom score, and the identical individual model plus **one** of education contrast, rural contrast, or national acceptance. Covariates are age, squared age, education, employment, marital status, income difficulty, urbanicity, religious attendance and COVID bereavement, using original GFS categories. Report country-specific MAE/RMSE/R2 and paired changes in MAE, not a significance claim across overlapping training folds. Main incremental comparison is education contrast versus individual baseline for each outcome. Negative context-minus-baseline MAE means improvement. This is retrospective country transfer using observed follow-up labels in other countries, not a deployment-time forecast or independent external validation. Do not select a winning feature after inspection as a confirmatory finding.

## Secondary bereavement diagnostic

Within each country fit a design-aware linear model of each follow-up symptom score on baseline reported COVID bereavement, the matching baseline symptom score, and the same individual covariates. Keep all full-adult-design records before subsetting women/complete cases. Save each country's bereavement coefficient, design-aware SE, t interval, residual df, sample size, convergence and any failure. No country coefficient should be treated as known without error.

Correlate these nine estimated coefficients with the primary education contrast, using the same exact rank-permutation procedure and Holm adjustment across the two symptom outcomes. This is a two-stage exploratory effect-modification diagnostic, not a definitive pooled interaction test; rank tests do not account for coefficient-estimation uncertainty or residual country confounding. The small country count prevents a credible high-dimensional interaction model. Plot coefficient intervals and country-deletion diagnostics. Adjustment for baseline symptoms addresses residual later differences, not the total causal consequence of bereavement.

## Reporting and reproducibility

Save code, input hashes, feature provenance, exclusions, weights/SE diagnostics, country estimates, numerical results, environment, and vector PDF/SVG plus 600-dpi PNG figures. No respondent rows or individual predictions go into review outputs. Keep this extension separate from previous results and the manuscript. Report unsupported/imprecise findings as such and retain all planned comparisons. Associations cannot establish that individual attitudes, violence exposure or national attitudes cause mental illness.
