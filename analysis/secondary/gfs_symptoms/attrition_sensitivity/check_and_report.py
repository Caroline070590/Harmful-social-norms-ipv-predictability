"""Independently reconstruct fixed response weighting, normal equations and PSU variance."""
from pathlib import Path
import hashlib
import json
import platform
import sys
import warnings
import numpy as np
import pandas as pd
import scipy
from scipy.special import expit
from scipy.stats import t

OUT = Path(__file__).resolve().parent
BASE = OUT.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def design_matrix(d, names):
    vectors = []
    numeric = ['covid_loss', 'phq4_w1', 'age10', 'age10_sq']
    factors = ['country', 'education', 'employment', 'marital', 'income', 'urbanicity', 'attendance']
    for name in names:
        if name == '(Intercept)':
            v = np.ones(len(d))
        elif name in numeric:
            v = d[name].to_numpy(float)
        else:
            prefix = next(k for k in factors if name.startswith(k))
            category = name[len(prefix):]
            v = d[prefix].eq(category if prefix == 'country' else float(category)).to_numpy(float)
        vectors.append(v)
    return np.column_stack(vectors)


def main():
    d = pd.read_csv(BASE/'analysis/design_input.csv', low_memory=False)
    d = d[d.valid_design & d.historical_norm.notna()].copy()
    d['w'] = d.weight / d.groupby('country').weight.transform('sum')
    c = d[d.model_complete].copy()
    retained = (c.phq4_w2.notna() & c.weight_l2.gt(0)).to_numpy()
    assert len(c) == 32788 and retained.sum() == 16926 and c.country.nunique() == 9
    assert np.array_equal(retained, c.phq4_w2.notna())
    terms = pd.read_csv(OUT/'model_terms.csv')
    names = terms.term.tolist()
    X = design_matrix(c, names)
    w = c.w.to_numpy(float)
    r = retained.astype(float)
    # Independently fit weighted logistic equations using Newton iterations.
    theta = np.zeros(X.shape[1])
    for iteration in range(100):
        p = expit(X@theta)
        bread = X.T @ (X*(w*p*(1-p))[:, None])
        score = X.T @ (w*(r-p))
        step = np.linalg.solve(bread, score)
        theta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    else:
        raise AssertionError('Independent logistic fit did not converge')
    p = expit(X@theta)
    theta_error = float(np.max(np.abs(theta-terms.retention_coefficient.to_numpy())))
    assert theta_error < 1e-7, theta_error
    local = pd.read_csv(OUT/'data_local/response_weight_diagnostics.csv')
    assert np.array_equal(local.ID, c.ID)
    p_error = float(np.max(np.abs(p-local.retention_probability.to_numpy())))
    assert p_error < 1e-9, p_error
    coefs = pd.read_csv(OUT/'outcome_coefficients.csv')
    summary = pd.read_csv(OUT/'followup_attrition_sensitivity.csv').set_index('specification')
    Z = X[retained]
    y = c.loc[retained, 'phq4_w2'].to_numpy(float)
    U1 = X*(r-p)[:, None]
    A11 = X.T @ (X*(w*p*(1-p))[:, None])
    # Keep original adult design PSU counts, including PSUs with zero domain rows.
    full_counts = d.groupby(['country', 'STRATA']).PSU.nunique().to_dict()
    domain_indices = c.groupby(['country', 'STRATA']).indices
    independent = []
    for spec, floor in [('ipw_uncapped', 0), ('ipw_floor_005', .05)]:
        multiplier = 1/np.maximum(p, floor)
        ww = w[retained]*multiplier[retained]
        A22 = Z.T @ (Z*ww[:, None])
        beta = np.linalg.solve(A22, Z.T@(ww*y))
        bsave = coefs[coefs.specification.eq(spec)].set_index('term').loc[names, 'estimate'].to_numpy()
        beta_error = float(np.max(np.abs(beta-bsave)))
        assert beta_error < 1e-7, beta_error
        error = y-Z@beta
        U2 = np.zeros_like(X)
        U2[retained] = Z*(multiplier[retained]*error)[:, None]
        derivative = np.where(p>floor, (1-p)/p, 0)
        A21 = Z.T @ (X[retained]*(w[retained]*derivative[retained]*error)[:, None])
        influence = np.linalg.solve(A22, (U2-U1@np.linalg.solve(A11, A21.T)).T).T
        weighted_influence = influence*w[:, None]
        covariance = np.zeros((X.shape[1], X.shape[1]))
        lonely = 0
        for key, idx in domain_indices.items():
            psu = c.iloc[idx].PSU.to_numpy()
            unit = pd.DataFrame(weighted_influence[idx]).assign(psu=psu).groupby('psu').sum().to_numpy()
            n_observed = len(unit)
            n_design = full_counts[key]
            assert n_observed <= n_design
            if n_observed < n_design:
                unit = np.vstack([unit, np.zeros((n_design-n_observed, X.shape[1]))])
            if n_observed > 1:
                unit -= unit.mean(axis=0)
            else:
                lonely += 1
            scale = n_design/(n_design-1) if n_design>1 else 1
            covariance += scale*(unit.T@unit)
        saved_cov = pd.read_csv(OUT/f'{spec}_covariance.csv', index_col=0).to_numpy()
        covariance_error = float(np.max(np.abs(covariance-saved_cov)))
        # Covariances on differently scaled terms vary; relative and exposure SE checks both retained.
        relative_cov_error = covariance_error/max(1, float(np.max(np.abs(saved_cov))))
        assert relative_cov_error < 1e-7, (covariance_error, relative_cov_error)
        j = names.index('covid_loss')
        se = np.sqrt(covariance[j, j])
        assert abs(se-summary.loc[spec, 'SE']) < 1e-8
        independent.append({'specification': spec, 'max_coefficient_error': beta_error,
                            'max_covariance_error': covariance_error,
                            'relative_covariance_error': relative_cov_error,
                            'independent_SE': float(se), 'lonely_domain_strata': lonely})

    # Rebuild the exact analysis-source variables from the public source, not the derived CSV.
    fields = ['ID','COUNTRY','GENDER','AGE_Y1','PSU','STRATA','ANNUAL_WEIGHT_C1','ANNUAL_WEIGHT_L2','COVID_DEATH_Y1']
    covmap = {'education':('EDUCATION_3',3),'employment':('EMPLOYMENT',8),'marital':('MARITAL_STATUS',6),
              'income':('INCOME_FEELINGS',4),'attendance':('ATTEND_SVCS',5),'urbanicity':('URBAN_RURAL',4)}
    fields += [v[0]+'_Y1' for v in covmap.values()]
    items = ['DEPRESSED','INTEREST','FEEL_ANXIOUS','CONTROL_WORRY']
    fields += [f'{item}_Y{wave}' for wave in [1,2] for item in items]
    raw = pd.read_csv(BASE/'data/gfs_all_countries_wave2.csv', usecols=fields, low_memory=False)
    raw = raw.set_index('ID').loc[d.ID]
    assert np.array_equal(raw.index, d.ID)
    raw_checks = {'ID_order': True}
    def check(name, actual, expected):
        assert np.allclose(np.asarray(actual,float), np.asarray(expected,float), equal_nan=True), name
        raw_checks[name] = True
    for wave in [1,2]:
        a = raw[[f'{item}_Y{wave}' for item in items]].apply(pd.to_numeric, errors='coerce')
        a = 4-a.where(a.ge(1)&a.le(4))
        check(f'PHQ4_wave_{wave}', a.sum(axis=1,min_count=4), d[f'phq4_w{wave}'])
    for variable,(field,hi) in covmap.items():
        a = pd.to_numeric(raw[field+'_Y1'],errors='coerce')
        check(variable,a.where(a.between(1,hi)),d[variable])
    loss = pd.to_numeric(raw.COVID_DEATH_Y1,errors='coerce').map({1:1,2:0})
    check('covid_bereavement',loss,d.covid_loss)
    check('PSU',raw.PSU,d.PSU);check('STRATA',raw.STRATA,d.STRATA)
    for field,var in [('ANNUAL_WEIGHT_C1','weight'),('ANNUAL_WEIGHT_L2','weight_l2')]:
        a = pd.to_numeric(raw[field],errors='coerce')
        check(var,a.where(a.gt(0)&np.isfinite(a)),d[var])
    check('female',pd.to_numeric(raw.GENDER,errors='coerce').eq(2).astype(int),d.female)
    age = pd.to_numeric(raw.AGE_Y1,errors='coerce')
    check('age',age.where(age.between(18,99)),d.age)

    # Recompute every reported interval and the preservation guard.
    z = summary
    assert np.allclose(z.lower, z.coefficient_phq4_points-t.ppf(.975,z.residual_df)*z.SE)
    assert np.allclose(z.upper, z.coefficient_phq4_points+t.ppf(.975,z.residual_df)*z.SE)
    manifest = json.loads((OUT/'PRE_FIT_MANIFEST.json').read_text())
    preserved = []
    root = BASE.parents[2]
    for row in manifest['files']:
        path = root/row['file']
        if path.name == 'run_attrition.R':
            continue # documented design-matrix/finite-difference implementation fixes
        same = sha(path) == row['sha256']
        # Root may be integrating manuscript concurrently; unchanged original analysis remains mandatory.
        manuscript = '07_manuscript/latex/' in row['file']
        if not manuscript:
            assert same, row['file']
        preserved.append({'file':row['file'],'unchanged':same,'manuscript_concurrent_integration':manuscript})
    warns = pd.read_csv(OUT/'warnings.csv').warning.tolist()
    unknown = [v for v in warns if 'has only one PSU at stage' not in v]
    assert not unknown, unknown
    checks = pd.read_csv(OUT/'estimating_equation_checks.csv')
    assert (checks.glm_vs_normal_equation_max_error<1e-8).all()
    assert (checks.stacked_derivative_relative_error<1e-7).all()
    log = (OUT/'run.log').read_text()
    assert 'Execution halted' not in log and 'Error' not in log
    ret = pd.read_csv(OUT/'retention_by_country.csv')
    prob = pd.read_csv(OUT/'probability_quantiles.csv').set_index('quantity')
    diag = pd.read_csv(OUT/'weight_diagnostics.csv')
    ess = diag[diag.country.eq('All_nine_countries')].set_index('specification')
    balance = pd.read_csv(OUT/'baseline_balance.csv')
    balance = balance[balance.country.eq('All_nine_countries') & ~balance.variable.str.startswith('country')].copy()
    balance['absolute_standardized_difference'] = balance.standardized_difference.abs()
    balance_summary = balance.groupby('specification').absolute_standardized_difference.agg(['count','mean','max'])
    balance_summary.rename(columns={'count':'baseline_covariate_columns','mean':'mean_absolute_standardized_difference',
                                    'max':'maximum_absolute_standardized_difference'}).to_csv(OUT/'balance_summary.csv')
    qc = {'all_checks_passed':True,'status':'post-primary retrospective exploratory attrition sensitivity',
          'baseline_complete_women':len(c),'retained_women':int(retained.sum()),'countries':int(c.country.nunique()),
          'independent_logistic_iterations':iteration+1,'independent_logistic_coefficient_max_error':theta_error,
          'independent_probability_max_error':p_error,'independent_point_and_PSU_variance_checks':independent,
          'raw_source_recoding_checks':raw_checks,'pre_fit_preservation':preserved,
          'current_code_sha256':sha(OUT/'run_attrition.R'),'protocol_sha256':sha(OUT/'PROTOCOL.md'),
          'expected_singleton_warnings':len(warns),'unexpected_warnings':unknown,
          'released_longitudinal_weights_already_attrition_adjusted':True,
          'weights_multiplied_by_released_longitudinal_weights':False,
          'IPW_assumptions':'conditional missing at random given modeled baseline covariates; adequate positivity; correctly specified retention model',
          'variance_scope':'stacked estimated-response/outcome Taylor linearization with released PSU/strata; conditional on normalized baseline country weight constants',
          'causal_claim':False,'Python':sys.version,'platform':platform.platform(),
          'packages':{'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__}}
    (OUT/'INDEPENDENT_QC.json').write_text(json.dumps(qc,indent=2)+'\n')
    table = '\n'.join(f'| {name.replace("_"," ")} | {row.coefficient_phq4_points:.3f} | {row.lower:.3f} to {row.upper:.3f} | {ess.loc[name,"kish_effective_n"]:.0f} |'
                      for name,row in summary.iterrows())
    ipw = summary.loc['ipw_uncapped']
    floor = summary.loc['ipw_floor_005']
    report = f'''# COVID-bereavement follow-up: measured-attrition sensitivity

Executed 20 October 2026 after saving `PROTOCOL.md`. This is an additional retrospective exploratory sensitivity of the existing GFS cohort, not a new primary analysis or an independent replication. The original longitudinal result is retained.

**Finding:** the uncapped analysis-specific response-weight estimate is {ipw.coefficient_phq4_points:.3f} PHQ-4 points (95% CI {ipw.lower:.3f} to {ipw.upper:.3f}). The fixed probability-floor estimate is {floor.coefficient_phq4_points:.3f} ({floor.lower:.3f} to {floor.upper:.3f}). Report these alongside the original result; adjustment for measured retention does not by itself identify a causal bereavement effect or prove attrition bias absent.

The baseline-complete cohort contains {len(c):,} women in nine countries; {int(retained.sum()):,} have complete follow-up symptoms and positive released longitudinal weights ({retained.mean()*100:.1f}% unweighted retention). Symptom availability and positive longitudinal-weight availability were checked separately. The exact retained cohort matches the original outcome model.

| Specification | Adjusted PHQ-4 difference | 95% CI | Kish effective n |
|---|---:|---|---:|
{table}

Kish effective sample sizes describe weight concentration only; they do not account for clustering or equal the design degrees of freedom. All outcome models use the same {int(retained.sum()):,} women and the same adjustment set: baseline PHQ-4, country, age/decade and its square, education, employment, marital status, income difficulty, urbanicity and religious-service attendance. PHQ-4 is a 0–12 symptom score, not a diagnosis.

## Why this is an alternative sensitivity

The released longitudinal weights already include the GFS study's attrition adjustment. The new estimates start from baseline survey weights and use a fixed analysis-specific logistic model of observed follow-up. They do **not** multiply additional retention weights onto the released longitudinal weights. The baseline-weighted reference separates changing the base weights from applying the new response model. The alternative target is the baseline-complete women's distribution under baseline full-adult country-normalized weights; it can differ from the released-longitudinal-weight target and is not a world-population estimate.

Predicted retention probabilities ranged from {prob.loc['retention_probability_all','min']:.4f} to {prob.loc['retention_probability_all','max']:.4f}. {int(ret.below_probability_005.sum()):,} baseline women and {int(ret.retained_below_probability_005.sum()):,} retained women had predictions below .05. The fixed floor therefore {'changes' if ret.retained_below_probability_005.sum() else 'does not change'} retained response weights. Among the modeled non-country baseline columns, the maximum absolute standardized mean difference from the baseline target is {balance_summary.loc['baseline_complete_case','max']:.4f} before response adjustment and {balance_summary.loc['ipw_uncapped','max']:.4f} after it. This is a descriptive balance check, not a measurement of bias or a validation of the missing-at-random assumption. See `retention_by_country.csv`, `probability_quantiles.csv`, `weight_diagnostics.csv`, `baseline_balance.csv` and `balance_summary.csv`; no overlap or balance result establishes absence of selection on unmeasured variables.

## Uncertainty and verification

The main new intervals propagate estimation of retention probabilities using stacked logistic/outcome estimating equations and Taylor linearization over the released PSU/strata, with explicit singleton adjustment. Conditional-weight standard errors are retained only as diagnostics. The t degrees of freedom are the baseline women's-domain design degrees of freedom minus outcome rank plus one. Fixed country weight-normalization constants are treated as known, as in the earlier analyses. There is no finite-population correction. Uncertainty remains conditional on the sampled nine countries and the fitted response model.

An independent Python implementation recoded all used symptom, exposure, covariate, weight and design variables from the public source, reconstructed the logistic estimates, reproduced both outcome coefficients by normal equations, and independently reproduced the PSU/stratum covariance (including the documented singleton rule). Central finite differences separately checked the stacked cross-derivative. The original longitudinal reference coefficient and SE reproduce to less than 1e-10. `INDEPENDENT_QC.json` states the numerical tolerances and preservation checks. Implementation corrections and the independent check that detected a provisional covariance error are documented in `EXECUTION_NOTES.md`; no planned analysis was changed in response to outcome findings.

## Limits

Response weighting requires conditional missing at random given the measured baseline predictors, adequate positivity and an adequate response model. It cannot correct missing-not-at-random selection, original nonresponse, missing baseline covariates or measurement non-equivalence. Bereavement dates are unknown; baseline symptoms may already follow the reported loss, so baseline adjustment does not estimate its total causal effect. These data do not measure an individual attitude-to-IPV-to-symptom pathway or a causal effect of the pandemic. The positive baseline screening associations, inconclusive conditional follow-up and contextual prediction failures remain part of the full evidence.

Public source policy: [GFS access data](https://www.cos.io/gfs-access-data). Survey variance method: [survey estimating-function variance documentation](https://r-survey.r-forge.r-project.org/survey/html/svyrecvar.html). All respondent diagnostics stay in `data_local/`, excluded from review/code archives.
'''
    (OUT/'RESULTS.md').write_text(report)
    print(summary[['coefficient_phq4_points','SE','lower','upper']].to_string())
    print('INDEPENDENT_QC passed')


if __name__ == '__main__':
    main()
