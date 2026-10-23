# Fixed later-year ENDES replication of attitude/IPV symptom prediction

20 October2026. Saved before the first temporal outcome fit. This is a post-primary exploratory external-year test motivated by the existing2023 findings; it is not preregistered or prospective confirmation. Preserve all previous2023 results, primary attitude analyses and active manuscript files.

## Question and estimand

Do individual attitudes justifying IPV add prediction of concurrent PHQ-9 symptom burden beyond demographics and measured recent IPV when models trained on actual2023 respondents are applied unchanged to actual2024 respondents? The primary comparison is ridgeM3 versus ridgeM1. A fixed secondary comparison asks whether measured recent IPV adds prediction beyond demographics(M1 versusM0). All five attitude responses, measured experienced IPV and symptoms remain distinct constructs. No violence-causes-depression, norms-cause-victimization, COVID effect, clinical diagnosis or neural mechanism is inferred.

The target is equal-person prediction among observed complete linked adult women18–49 who were ever married/cohabiting and passed the documented individual/health/DV/privacy/item/covariate eligibility rules. No valid joint DV–health population weight has been established. The test is not national prevalence, nationally weighted predictive performance or a population effect. Module weights are neither multiplied nor substituted.

## Source, date and contextual dependence gates

`PREPARATION_PROTOCOL.md`, `SOURCE_AND_DESIGN_REVIEW.md`, `SOURCE_INVENTORY.json`, `READINESS.json`, `MODEL_READY.json` and independent original-source preparationQC must pass before fitting. Archive and respondent rows remain in excluded`data_local/`.

Train only records in the frozen2023 prepared table whose **actual individual and actual selected-health interview years both equal2023**. Exclude the eight2024interviews from that release; retain10,259training women. This date amendment was decided before fitting from source dates alone; original2023release analyses remain unchanged. Both actual interview years equal2024 for every9,847test women. This predicts concurrent2024 symptoms from2024 predictors using2023learned parameters; it does not forecast later symptoms of the same people.

The official technical sheet explicitly identifies same-number`HV001` clusters as panel clusters across2023/2024 and describes different selected dwellings within retained clusters. The primary full2024test retains shared contexts and acknowledges contextual dependence. A fixed secondary domain retains only2024PSUs absent from **every2023released household PSU**, using private hashed`new_psu_vs_all_2023_households` flags. This domain contains3,250women/1,025PSUs. Do not substitute the weaker absence-from-analytic-training definition. Annual household/person keys can recur; reported birth month/year agreement is only a diagnostic, not verified personal identity. Guaranteed individual non-overlap and tracking movers are not claimed.

## Locked feature blocks and algorithms

Use exactly the existing2023 demographic/IPV/attitude blocks and frozen hyperparameters; no2024outcome-driven feature, architecture or regularization selection:

- M0: age, age squared, native education, wealth quintile, current work, marital status, residence and region.
- M1: M0plus verified recent emotional, physical and sexual IPV indicators.
- M2: M1plus overall five-item attitude endorsement total.
- M3: M2plus four fixed orthonormal Helmert contrasts preserving the five-scenario pattern.
- M4: M0plus attitude total and four contrasts, omitting measuredIPV.

Ridge uses alpha10. The fixed secondary histogram gradient booster uses200iterations, learning rate0.05, seven leaves, minimum40observations per leaf, L2regularization10 and no early stopping. Train scaling and categorical encoding entirely on the strict2023training rows; record unseen test categories. PHQoutcome items cannot enter predictors. Also retain the2023training-mean reference. Fit each block once on all stricttraining rows; the2024test is never used for refitting, tuning or threshold selection.

The five fixed feature comparisons are M3−M1(primary attitudes beyondIPV), M2−M1(total endorsement), M3−M2(pattern beyond total), M1−M0(IPV increment), and M4−M0(attitudes beforeIPV adjustment). Boosting comparisons are secondary algorithm sensitivities. No optimal-method or novel-algorithm claim is assumed.

## Outcomes, domains and uncertainty

Use the complete nine14-day PHQitems scored0–3 and summed0–27, preserving the source labels. Main metric: equal-personRMSE with predictions bounded0–27. Also retain raw/unboundedRMSE, MAE, bias, out-of-sampleR2 and clipping frequency. Comparisons always use the same women and identical outcome definitions. Never compare these errors directly with aggregate attitude percentage-point errors.

Evaluate four fixed domains: all2024women(primary); the strict new-PSU domain; all2024women with no attitude endorsement; and all2024women endorsing any justification. The endorsement domains are a prespecified diagnostic for rarity and do not establish causal effect modification. Export actual women/PSU/stratum counts and score support. Retain point metrics for small domains but do not invent intervals when the domain contains fewer than50women or30observedPSUs. Do not select domains or report only favourable comparisons.

For conditional descriptive uncertainty, resample whole observed testPSUs within released strata using2,000draws and fixed seed20261021, holding all trained models and test predictions fixed. Preserve and count singleton strata. For each defined domain, keep whole selectedPSUs and the domain's observed women. Save paired differences and percentileintervals with their exact direction(lowererror is better). These intervals omit model-refitting and source-selection uncertainty, are conditional on the observed complete testcohort, and are not certified national-design inference or complete prospective uncertainty. No p-value search, equivalence claim, null-effect proof or significance-only conclusion is planned.

## Verification, reporting and preservation

Freeze source/cohort/config/code/protocol hashes before fitting. Independently reconstruct original-source linkage, dates, covariates/items/scores, all-source new-PSU flags and strict training eligibility. Independently check design separation, exact feature construction, ridge normal-equation predictions, boosting tree traversal where supported, all metric tables, domain counts and fixed-prediction resampling distributions. Distinguish stored-state checks from complete optimizer reoptimization.

Save row-level training/test tables, predictions, keys and potentially sample-sensitive model states only in excluded`data_local/`. Export aggregateCSV/JSON, actual environment metadata, source hashes, explicit limitations, and standard vectorPDF/SVG plus600-dpiPNG figures if the executed results justify a display. Generate every reported numerical finding from saved tables. Do not integrate this extension into the primary manuscript or overwrite earlier packages during the analysis stage; first provide the author a concrete, reviewable result and its limits.
