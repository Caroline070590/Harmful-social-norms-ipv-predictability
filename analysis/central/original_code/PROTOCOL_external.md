# External survey-programme evaluation: fixed exploratory protocol

Saved on 20 October 2026 before external model fitting and expanded outcome extraction. The author authorized this evaluation together with a measurement audit and shared-context comparison. Earlier readiness work already inspected Bangladesh and Jamaica outcomes; all results remain retrospective exploratory evaluation, not blinded confirmation or preregistration.

## Question and fixed targets

Does structural context improve a DHS-trained model's predictions of national female attitude percentages in independently administered MICS surveys? Does a common shift in all five percentages account for any gain, compared with item-specific contextual coefficients?

Use the fixed latest-wave MICS candidate inventory from `external_replication_readiness/results/mics_candidate_inventory.csv`. The source-audit protocol governs deterministic search order, coverage restrictions and report extraction. Do not choose countries, alternate rounds or estimates by prediction performance. The primary unit is a country context, using at most one latest eligible inventory survey per country. Bangladesh 2019 remains a source check and is not a second independent primary target when its eligible 2025 report is available.

Require report-verified national estimates for all women aged 15–49 for the five common scenarios: burns food, argues, goes out without telling the husband, neglects children and refuses sex. Retain exact wording and non-equivalence flags. Extra scenarios and any-reason union percentages are excluded from the five-component target; the union cannot be substituted for a missing component. The five-component mean is not the proportion endorsing any reason or an individual psychological score. Record weighted denominator labels without claiming unweighted sample sizes.

Require a source-verified fieldwork start month and end month. Use the first day of the start month for donor exclusion and context-year selection. Treat donor fieldwork recorded in month precision as ending at the end of that month; only donors completed before the target's first possible start date are eligible. Log any stricter exclusion relative to the earlier month-start convention. A report publication year is not a fieldwork date. Unresolved age/universe, scenario, source or date eligibility causes a clean missing-data/readiness exclusion, not an invented estimate.

## Training and fixed models

Use only the existing 169 compatible DHS rounds in 62 countries (`joint_country_time/prepared/rounds.csv`). For each MICS target, exclude all DHS rounds in its country, even when DHS and MICS have different dates. Exclude donors whose fieldwork was not strictly completed before the MICS start. MICS outcomes never enter training, preprocessing, model selection or tuning. Save the full donor/exclusion ledger. If fewer than two donor countries or no finite values for a required training feature remain, stop that target cleanly.

Use the existing four structural series and releases: HDI, electoral democracy, liberal democracy and women's political empowerment. Select the latest available observation in the three calendar years strictly preceding fieldwork-start year. Do not substitute future years. These are retrospectively released sources; lagged observation dates do not certify historical release availability. Calendar is fieldwork-start year plus (month−1)/12. This is retrospective target-date evaluation, not a forecast issued at a prior survey or a COVID effect.

Models are fixed to the previous country/time specification: R0 has item-specific intercepts and calendar slopes; R1 additionally has item-specific coefficients on the four structural predictors. Alpha is 10, with an unpenalized intercept. Donor country total weight is one, split equally among its eligible rounds; weights are not rescaled to the row count. Impute missing features with training-only weighted means and standardize by training-only weighted population SD; constant features use scale one. Target missing features use the same training means and are logged. All missing-context targets remain in the primary eligible cohort; report the prespecified complete-context subset separately.

S is the common-context-shift ablation, with separate item intercepts/calendar slopes and one shared coefficient per structural predictor. In the summed five-output objective, the shared context penalty is 5×alpha. For complete equally weighted output vectors, its unbounded predictions equal `mean(R1_raw) + R0_raw − mean(R0_raw)`. An independent weighted stacked normal-equation implementation must verify this identity and its predictions. S is an interpretation ablation, not an independent psychological construct or additional replication sample.

Primary predictions are bounded to [0,100] identically for all models. Save unbounded values, clipping flags and raw diagnostics; bounding can alter the common-shift identity.

## Evaluation and fixed reporting

Primary comparison: R1−R0 in mean absolute error, first averaging the five absolute component errors within country, then weighting countries equally. Save every target/item prediction, country error, signed bias, level/profile MAE and exact MSE decomposition. S−R0 and R1−S are fixed secondary ablations. Keep overlapping and entirely new DHS country contexts identified, without selecting the better-performing subset.

Report country-specific errors and equal-country means in every eligible cohort. With at least ten eligible country contexts, use 10,000 paired country bootstrap resamples with seed 20261020 for descriptive percentile intervals on mean differences. With fewer than ten, report all country errors and differences without bootstrap intervals. This rule does not make ten contexts a representative global sample. No survey sampling SEs, inferential p-values, calibrated predictive intervals or causal conclusions are generated.

Fixed sensitivities: (1) remove the previously flagged Nepal 2011 and Benin 2012 donor rounds, without asserting that their estimates are wrong; (2) complete target structural context; (3) targets outside the entire 62-country DHS training inventory. Report eligibility/count changes. A separate questionnaire-based comparable-survey sensitivity may be added only from the outcome-independent audit classifications documented before that sensitivity is fitted; retain the primary evaluation unchanged.

## Verification and deliverables

Save protocol/configuration hashes, report inventory and exact source fragments, verified country/item cells, structural-source/year ledger, all training/exclusion memberships, model/scaler parameters, missingness, predictions, errors, fixed comparisons and environment metadata. Verify original source hashes and no overlap of country/date boundaries. Independently reconstruct at least the primary models using weighted normal equations, compare every prediction, and re-read report totals against extracted rows. Generate figures from saved CSVs in vector PDF/SVG and 600-dpi PNG with matching black 11-point panel labels. Retain unsuccessful predictions and source discrepancies.

If available sources support only a small or heterogeneous set, label the result a limited external evaluation and report all unverified candidates. Do not claim all inventory candidates were fully adjudicated. No respondent data, restricted files or placeholder results are created.
