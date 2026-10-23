# Post-primary GFS follow-up attrition sensitivity

Recorded 20 October 2026 **before fitting the models below**. This is a retrospective exploratory sensitivity of already inspected GFS outcomes, not a preregistration, a new cohort or a new primary outcome. The original released-longitudinal-weight result is retained unchanged.

## Question, cohort and target

Does adjustment for measured baseline predictors of observed follow-up change the interpretation of the baseline-symptom-adjusted association between reported COVID-related bereavement and later PHQ-4?

Use the existing non-sensitive public GFS wave-2 wide release and the verified `analysis/design_input.csv`. The analysis domain is the 32,788 baseline-complete women in the same nine countries. Retention means complete follow-up PHQ-4 and a positive released longitudinal weight; the availability audit found 16,926 retained cases, exactly the original outcome-model cohort. Also tabulate symptom availability separately from weight availability; no missing outcome is imputed. Restricting to baseline complete cases does not resolve baseline missingness.

The target of the new weighting analysis is the baseline-complete women's distribution under the **baseline** survey weights, normalized by full valid-adult baseline weight totals within each country before domain selection. This matches the earlier baseline pooling convention; it is not equal weight per woman-country domain, a world-population estimate, or exactly the target induced by released longitudinal weights. Keep that estimand distinction explicit.

## Fixed models and comparisons

Fit one survey-weighted logistic retention model on all baseline-complete women:

`retained ~ covid_loss + phq4_w1 + country + age10 + age10_sq + education + employment + marital + income + urbanicity + attendance`.

Use the original categories as factors, age/decade and its square, and linear baseline PHQ-4. No selection, interactions, tuning or outcome-driven changes. The selection model is a conditional missing-at-random sensitivity, not a test that missing at random holds. Predictions must be finite and strictly between zero and one. Nonconvergence, rank deficiency or nonfinite weights stop the affected computation with a machine-readable report; no substitute favorable model is fitted.

The unchanged outcome is continuous follow-up PHQ-4 (0–12), with the same right-hand-side covariates. Report four fixed specifications on exactly the same 16,926 retained women:

1. Original reference using released longitudinal weights normalized over full valid adult longitudinal-weight totals per country. Reproduce the stored coefficient/SE before treating this as the original reference.
2. Baseline-weighted complete-case reference, without response adjustment.
3. **Primary new sensitivity:** baseline weights divided by predicted retention probability, uncapped.
4. Fixed positivity sensitivity: baseline weights divided by `max(predicted probability, .05)`. Report all capped cases; no data-driven cap selection.

The released longitudinal weights already include the study's attrition adjustment. The new baseline-weight/IPW estimates are alternative, analysis-specific sensitivity estimates, not the first correction of previously unadjusted longitudinal data. Do **not** multiply released longitudinal weights by additional inverse-retention weights. No country recalibration, stabilization multiplier, new predictors, subgroup searches or p-value-based model selection. Weight normalizations are fixed descriptive scaling constants, as in the earlier pooled analyses.

## Uncertainty and diagnostics

Preserve baseline PSU and strata nested within country, construct the valid full adult design, then select the baseline-complete women's domain. Use the existing explicit singleton adjustment (`survey.lonely.psu='adjust'`, `survey.adjust.domain.lonely=TRUE`). For the two new inverse-retention estimates, propagate fitted logistic-model uncertainty with stacked estimating equations for retention and outcome, and Taylor linearization of their influence contributions through the released PSU/stratum design. For a capped prediction the derivative of the response-weight function is zero below .05. Report t-based 95% intervals with design degrees of freedom minus outcome-model rank plus one, and identify this as an asymptotic design-based approximation. Retain the ordinary conditional-weight variance as a diagnostic, not as the principal interval. No finite-population correction is supplied.

Save retention counts and baseline-weighted retention by country; range/quantiles of fitted probabilities; numbers below .05; response-factor and final-weight summaries; Kish effective sample sizes; and weighted baseline means/indicator proportions before selection, among retained women and after each response weighting. These assess observed overlap/balance, not unobserved selection. Report all fixed results and note that capped weights can leave residual selection bias. No confirmatory hypothesis-test family or p-value claim is added; intervals describe sensitivity uncertainty.

## Verification and preservation

Before fitting, hash the protocol, derived input, public source and existing original longitudinal outputs. Record exact sample identities locally but never export respondent rows in public reports or archives. Independently verify inverse-weighted outcome coefficients by weighted normal equations and the stacked derivative by finite perturbation. Where feasible independently reproduce PSU/stratum covariance; state the exact verification scope. Save source hashes, warnings, counts, model coefficients, numerical diagnostics, environment and a machine-readable QC report. All new outputs stay in this directory; any respondent-level diagnostics stay in `data_local/` and are excluded from archives. The active manuscript and all original analyses remain untouched.

## Interpretation

Baseline bereavement is a reported COVID-related death of a family member or close friend, without a known event date. Follow-up PHQ-4 measures symptoms, not diagnoses. Baseline symptoms may already follow bereavement, so adjustment does not estimate its total causal effect. Response weighting addresses only measured selection under its model and positivity assumptions; neither a stable nor a changed coefficient establishes absence of attrition bias, pandemic causality, an individual attitude/IPV pathway or cross-language measurement equivalence.

Method implementation reference: the survey package documents variance linearization from estimating functions in [svyrecvar](https://r-survey.r-forge.r-project.org/survey/html/svyrecvar.html). Public-release policy: [GFS access data](https://www.cos.io/gfs-access-data).
