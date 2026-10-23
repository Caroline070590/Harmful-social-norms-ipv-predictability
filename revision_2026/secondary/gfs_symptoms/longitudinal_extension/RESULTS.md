# Exploratory follow-up results

Executed 19 October 2026 under the separately saved exploratory amendment. This is a new analysis of the existing GFS cohort, not an independent replication. Source tables and code remain separate from the baseline analyses.

**Main finding:** among 16,926 women across nine countries, the adjusted follow-up difference associated with baseline reported COVID bereavement was **0.196 PHQ-4 points**, 95% CI **-0.171 to 0.563**, p=0.296. This interval includes zero and does not provide clear evidence of a residual follow-up difference after baseline symptoms and measured covariates are accounted for. It is not evidence that bereavement has no mental-health consequences.

| Specification | Women | Difference in PHQ-4 points | 95% CI | Unadjusted p |
|---|---:|---:|---|---:|
| Primary | 16,926 | 0.196 | -0.171 to 0.563 | 0.296 |
| Add interview interval | 16,926 | 0.191 | -0.176 to 0.558 | 0.308 |
| Categorical baseline PHQ-4 | 16,926 | 0.206 | -0.152 to 0.564 | 0.260 |
| Ages 18–49 | 14,439 | 0.236 | -0.152 to 0.623 | 0.233 |

The primary endpoint is continuous follow-up PHQ-4 (0–12), not a diagnosis or incident disease. Positive coefficients indicate higher follow-up symptoms among women reporting bereavement. Models account for baseline PHQ-4, country, age and squared age, education, employment, marital status, income difficulty, urbanicity and religious attendance. The other three estimates are pre-specified-in-the-amendment sensitivities, not separate discoveries or independent replications. The amendment itself is post-primary, not a preregistration.

Paired follow-up before covariate exclusions contained 17,186 women; incomplete baseline model covariates excluded another 260. Baseline interviews in the primary analysis range from 2022-11-21 to 2024-01-05; follow-up interviews range from 2024-06-19 to 2024-12-31. All primary-model intervals are observed and positive.

The analysis uses released longitudinal weights and the full adult survey design before selecting the women's domain. Pooling normalizes full adult weight totals within each country; this is not a world-population-weighted effect. Singleton-stratum warnings were handled by the documented survey adjustment, retained in the run log and counted in QC. All fitted outcome models converged. See `sample_coverage.csv`, `all_coefficients.csv` and `R_sessionInfo.txt`.

Retention of observed symptoms plus valid longitudinal weight differs substantially across countries. The adjusted baseline bereavement association with that retention indicator was OR 1.131, 95% CI 0.998–1.281, p=0.054, in 32,788 complete baseline cases. This does not establish the absence of selective follow-up. `baseline_retention_comparison.csv` supplies baseline-weighted age, symptom and bereavement summaries with denominators for retained versus other baseline-eligible women.

**Interpretation:** this result limits claims about additional or persistent symptom differences conditional on baseline symptoms. Baseline symptoms may already reflect an earlier bereavement, whose date is unknown. Adjusting for them does not estimate the total causal effect. Attrition, unmeasured confounding and cross-language measurement comparability remain unresolved. The positive baseline screening associations and inconclusive conditional follow-up result address different questions and should both be reported. No individual attitude/IPV exposure was added or inferred.
