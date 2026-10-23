# COVID-bereavement follow-up: measured-attrition sensitivity

Executed 20 October 2026 after saving `PROTOCOL.md`. This is an additional retrospective exploratory sensitivity of the existing GFS cohort, not a new primary analysis or an independent replication. The original longitudinal result is retained.

**Finding:** the uncapped analysis-specific response-weight estimate is 0.199 PHQ-4 points (95% CI -0.119 to 0.517). The fixed probability-floor estimate is 0.199 (-0.119 to 0.517). Report these alongside the original result; adjustment for measured retention does not by itself identify a causal bereavement effect or prove attrition bias absent.

The baseline-complete cohort contains 32,788 women in nine countries; 16,926 have complete follow-up symptoms and positive released longitudinal weights (51.6% unweighted retention). Symptom availability and positive longitudinal-weight availability were checked separately. The exact retained cohort matches the original outcome model.

| Specification | Adjusted PHQ-4 difference | 95% CI | Kish effective n |
|---|---:|---|---:|
| released longitudinal reference | 0.196 | -0.171 to 0.563 | 4032 |
| baseline complete case | 0.184 | -0.079 to 0.446 | 7038 |
| ipw uncapped | 0.199 | -0.119 to 0.517 | 4705 |
| ipw floor 005 | 0.199 | -0.119 to 0.517 | 4705 |

Kish effective sample sizes describe weight concentration only; they do not account for clustering or equal the design degrees of freedom. All outcome models use the same 16,926 women and the same adjustment set: baseline PHQ-4, country, age/decade and its square, education, employment, marital status, income difficulty, urbanicity and religious-service attendance. PHQ-4 is a 0–12 symptom score, not a diagnosis.

## Why this is an alternative sensitivity

The released longitudinal weights already include the GFS study's attrition adjustment. The new estimates start from baseline survey weights and use a fixed analysis-specific logistic model of observed follow-up. They do **not** multiply additional retention weights onto the released longitudinal weights. The baseline-weighted reference separates changing the base weights from applying the new response model. The alternative target is the baseline-complete women's distribution under baseline full-adult country-normalized weights; it can differ from the released-longitudinal-weight target and is not a world-population estimate.

Predicted retention probabilities ranged from 0.1746 to 0.7716. 0 baseline women and 0 retained women had predictions below .05. The fixed floor therefore does not change retained response weights. Among the modeled non-country baseline columns, the maximum absolute standardized mean difference from the baseline target is 0.0401 before response adjustment and 0.0097 after it. This is a descriptive balance check, not a measurement of bias or a validation of the missing-at-random assumption. See `retention_by_country.csv`, `probability_quantiles.csv`, `weight_diagnostics.csv`, `baseline_balance.csv` and `balance_summary.csv`; no overlap or balance result establishes absence of selection on unmeasured variables.

## Uncertainty and verification

The main new intervals propagate estimation of retention probabilities using stacked logistic/outcome estimating equations and Taylor linearization over the released PSU/strata, with explicit singleton adjustment. Conditional-weight standard errors are retained only as diagnostics. The t degrees of freedom are the baseline women's-domain design degrees of freedom minus outcome rank plus one. Fixed country weight-normalization constants are treated as known, as in the earlier analyses. There is no finite-population correction. Uncertainty remains conditional on the sampled nine countries and the fitted response model.

An independent Python implementation recoded all used symptom, exposure, covariate, weight and design variables from the public source, reconstructed the logistic estimates, reproduced both outcome coefficients by normal equations, and independently reproduced the PSU/stratum covariance (including the documented singleton rule). Central finite differences separately checked the stacked cross-derivative. The original longitudinal reference coefficient and SE reproduce to less than 1e-10. `INDEPENDENT_QC.json` states the numerical tolerances and preservation checks. Implementation corrections and the independent check that detected a provisional covariance error are documented in `EXECUTION_NOTES.md`; no planned analysis was changed in response to outcome findings.

## Limits

Response weighting requires conditional missing at random given the measured baseline predictors, adequate positivity and an adequate response model. It cannot correct missing-not-at-random selection, original nonresponse, missing baseline covariates or measurement non-equivalence. Bereavement dates are unknown; baseline symptoms may already follow the reported loss, so baseline adjustment does not estimate its total causal effect. These data do not measure an individual attitude-to-IPV-to-symptom pathway or a causal effect of the pandemic. The positive baseline screening associations, inconclusive conditional follow-up and contextual prediction failures remain part of the full evidence.

Public source policy: [GFS access data](https://www.cos.io/gfs-access-data). Survey variance method: [survey estimating-function variance documentation](https://r-survey.r-forge.r-project.org/survey/html/svyrecvar.html). All respondent diagnostics stay in `data_local/`, excluded from review/code archives.
