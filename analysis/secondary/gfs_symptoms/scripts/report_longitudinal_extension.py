"""Check and report the separate exploratory longitudinal analysis from saved tables."""
from pathlib import Path
import hashlib, json, re
import numpy as np
import pandas as pd
from scipy.stats import t

BASE = Path(__file__).resolve().parents[1]
OUT = BASE/'longitudinal_extension'


def main():
    results = pd.read_csv(OUT/'followup_bereavement_associations.csv')
    coverage = pd.read_csv(OUT/'sample_coverage.csv')
    retained = pd.read_csv(OUT/'retention_bereavement_association.csv').iloc[0]
    assert coverage.country.nunique() == 9
    assert coverage.paired_with_weight.sum() == 17186
    assert results.converged.all()
    assert (results.lower <= results.coefficient_phq4_points).all()
    assert (results.upper >= results.coefficient_phq4_points).all()
    assert results.p.between(0, 1).all() and (results.SE > 0).all()
    critical = t.ppf(.975, results.residual_df)
    assert np.allclose(results.lower, results.coefficient_phq4_points-critical*results.SE)
    assert np.allclose(results.upper, results.coefficient_phq4_points+critical*results.SE)
    primary = results.set_index('specification').loc['primary']
    assert primary.n == coverage.complete_longitudinal_model.sum()
    assert primary.n < coverage.paired_with_weight.sum()
    log = (BASE/'longitudinal_extension_run.log').read_text()
    warnings = re.findall(r'Warning[^\n]*(?:\n[^\n]*)?', log)
    unknown = [w for w in warnings if 'onestrat' not in w or 'only one PSU at stage 1' not in w]
    assert not unknown, unknown[:3]
    assert not re.search(r'Error|Execution halted|did not converge|non-finite', log, re.I)
    sources = [BASE/'analysis/design_input.csv', BASE/'LONGITUDINAL_AMENDMENT_2026_10_19.md',
               BASE/'scripts/longitudinal_extension.R']
    qc = {'all_checks_passed': True, 'status': 'post-primary exploratory extension',
          'country_count': int(coverage.country.nunique()),
          'paired_before_covariate_exclusions': int(coverage.paired_with_weight.sum()),
          'primary_model_n': int(primary.n),
          'covariate_exclusions_from_paired': int(coverage.paired_with_weight.sum()-primary.n),
          'all_models_converged': True, 't_based_intervals_recomputed': True,
          'expected_singleton_warning_count': len(warnings), 'unexpected_warnings': unknown,
          'singleton_handling': 'survey.lonely.psu=adjust; survey.adjust.domain.lonely=TRUE',
          'full_adult_design_before_women_subpopulation': True,
          'weights': 'Released longitudinal weights; full adult country totals normalized before pooling',
          'baseline_protocol_overwritten': False, 'causal_or_diagnostic_claim': False,
          'source_sha256': {str(p.relative_to(BASE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    (OUT/'QC.json').write_text(json.dumps(qc, indent=2)+'\n')
    names = {'primary': 'Primary', 'add_interview_interval': 'Add interview interval',
             'categorical_baseline': 'Categorical baseline PHQ-4', 'age18_49': 'Ages 18–49'}
    table = '\n'.join(f'| {names[r.specification]} | {r.n:,} | {r.coefficient_phq4_points:.3f} | {r.lower:.3f} to {r.upper:.3f} | {r.p:.3f} |' for r in results.itertuples())
    report = f'''# Exploratory follow-up results

Executed 19 October 2026 under the separately saved exploratory amendment. This is a new analysis of the existing GFS cohort, not an independent replication. Source tables and code remain separate from the baseline analyses.

**Main finding:** among {int(primary.n):,} women across nine countries, the adjusted follow-up difference associated with baseline reported COVID bereavement was **{primary.coefficient_phq4_points:.3f} PHQ-4 points**, 95% CI **{primary.lower:.3f} to {primary.upper:.3f}**, p={primary.p:.3f}. This interval includes zero and does not provide clear evidence of a residual follow-up difference after baseline symptoms and measured covariates are accounted for. It is not evidence that bereavement has no mental-health consequences.

| Specification | Women | Difference in PHQ-4 points | 95% CI | Unadjusted p |
|---|---:|---:|---|---:|
{table}

The primary endpoint is continuous follow-up PHQ-4 (0–12), not a diagnosis or incident disease. Positive coefficients indicate higher follow-up symptoms among women reporting bereavement. Models account for baseline PHQ-4, country, age and squared age, education, employment, marital status, income difficulty, urbanicity and religious attendance. The other three estimates are pre-specified-in-the-amendment sensitivities, not separate discoveries or independent replications. The amendment itself is post-primary, not a preregistration.

Paired follow-up before covariate exclusions contained {int(coverage.paired_with_weight.sum()):,} women; incomplete baseline model covariates excluded another {int(coverage.paired_with_weight.sum()-primary.n):,}. Baseline interviews in the primary analysis range from {primary.baseline_min} to {primary.baseline_max}; follow-up interviews range from {primary.followup_min} to {primary.followup_max}. All primary-model intervals are observed and positive.

The analysis uses released longitudinal weights and the full adult survey design before selecting the women's domain. Pooling normalizes full adult weight totals within each country; this is not a world-population-weighted effect. Singleton-stratum warnings were handled by the documented survey adjustment, retained in the run log and counted in QC. All fitted outcome models converged. See `sample_coverage.csv`, `all_coefficients.csv` and `R_sessionInfo.txt`.

Retention of observed symptoms plus valid longitudinal weight differs substantially across countries. The adjusted baseline bereavement association with that retention indicator was OR {retained.OR:.3f}, 95% CI {retained.lower:.3f}–{retained.upper:.3f}, p={retained.p:.3f}, in {int(retained.n):,} complete baseline cases. This does not establish the absence of selective follow-up. `baseline_retention_comparison.csv` supplies baseline-weighted age, symptom and bereavement summaries with denominators for retained versus other baseline-eligible women.

**Interpretation:** this result limits claims about additional or persistent symptom differences conditional on baseline symptoms. Baseline symptoms may already reflect an earlier bereavement, whose date is unknown. Adjusting for them does not estimate the total causal effect. Attrition, unmeasured confounding and cross-language measurement comparability remain unresolved. The positive baseline screening associations and inconclusive conditional follow-up result address different questions and should both be reported. No individual attitude/IPV exposure was added or inferred.
'''
    (OUT/'RESULTS.md').write_text(report)
    print(json.dumps(qc, indent=2))


if __name__ == '__main__':
    main()
