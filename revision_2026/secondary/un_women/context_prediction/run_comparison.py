"""Fixed, post-primary out-of-country contextual prediction comparison."""
from pathlib import Path
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/violence-mpl')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import hashlib
import json
import sys
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
REV = BASE.parents[1]
CATS = ['AgeCat', 'Education', 'Locality', 'Marital_Status', 'Employment']
TARGETS = ['worry_item', 'depressive_item_sum']
CONTEXT = ['national_female_acceptance', 'education_contrast_pp', 'rural_contrast_pp']
SPECS = {'national_8': [CONTEXT[0]], 'complete_profiles_6': CONTEXT}
LABELS = {CONTEXT[0]: 'National acceptance', CONTEXT[1]: 'Education contrast', CONTEXT[2]: 'Rural–urban contrast'}
INPUTS = {
    'coded_input': BASE/'data_local/pilot_coded_input.csv',
    'raw_release': BASE/'data_local/unwomen_vaw_rga_public.xlsx',
    'crosswalk': BASE/'country_profile_linkage/country_crosswalk_and_feature_availability.csv',
    'national_source': REV/'02_aggregate_landscape/female_primary/official_female_national_inventory.csv',
    'contrast_source': REV/'02_aggregate_landscape/original_paper_extension/results/demographic_country_contrasts.csv',
    'protocol': OUT/'PROTOCOL.md',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def js(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def prepare():
    d = pd.read_csv(INPUTS['coded_input'])
    employment = pd.read_excel(INPUTS['raw_release'], usecols=['Country', 'Respondent_Serial', 'Employment'])
    d = d.merge(employment, on=['Country', 'Respondent_Serial'], validate='one_to_one')
    assert len(d) == 16154 and not d.duplicated(['Country', 'Respondent_Serial']).any()
    ledger = []
    for col in CATS:
        d[col] = d[col].astype('string').str.strip()
        for val, n in d[col].fillna('<missing>').value_counts().items():
            exclude = val in ['<missing>', 'Refused', "Don't know", 'Not applicable']
            ledger.append(dict(field=col, original_category=val, women=int(n), action='exclude' if exclude else 'retain_exact_category'))
        d.loc[d[col].isin(['Refused', "Don't know", 'Not applicable']), col] = pd.NA
    assert d.worry_item.dropna().between(0, 3).all()
    assert d.depressive_item_sum.dropna().between(0, 6).all()
    d['complete_symptoms'] = d[TARGETS].notna().all(axis=1)
    d['complete_predictors'] = d[CATS].notna().all(axis=1)
    d['positive_weight'] = np.isfinite(d.weight) & d.weight.gt(0)
    cross = pd.read_csv(INPUTS['crosswalk'])
    d = d.merge(cross[['survey_country']+CONTEXT], left_on='Country', right_on='survey_country', how='left', validate='many_to_one')
    pd.DataFrame(ledger).to_csv(OUT/'category_ledger.csv', index=False)
    coverage = []
    for country, group in d.groupby('Country'):
        valid = group.complete_symptoms & group.complete_predictors & group.positive_weight
        coverage.append(dict(country=country, released_women=len(group),
            missing_symptoms=int((~group.complete_symptoms).sum()),
            complete_symptoms_missing_predictors=int((group.complete_symptoms & ~group.complete_predictors).sum()),
            otherwise_complete_invalid_weight=int((group.complete_symptoms & group.complete_predictors & ~group.positive_weight).sum()),
            complete_women=int(valid.sum()), national_context=bool(group[CONTEXT[0]].notna().all()),
            all_contexts=bool(group[CONTEXT].notna().all().all())))
    cov = pd.DataFrame(coverage)
    assert (cov.released_women == cov.missing_symptoms + cov.complete_symptoms_missing_predictors + cov.otherwise_complete_invalid_weight + cov.complete_women).all()
    cov.to_csv(OUT/'sample_coverage.csv', index=False)
    d = d[d.complete_symptoms & d.complete_predictors & d.positive_weight].copy()
    cohorts = {}
    for name, features in SPECS.items():
        required = [CONTEXT[0]] if name == 'national_8' else CONTEXT
        z = d[d[required].notna().all(axis=1)].copy()
        assert z.Country.nunique() == (8 if name == 'national_8' else 6)
        cohorts[name] = z
    return cohorts


def fit(cohorts):
    metrics, coefficients, preprocessing, unknowns, splits = [], [], [], [], []
    for cohort_name, data in cohorts.items():
        for held in sorted(data.Country.unique()):
            train, test = data[data.Country.ne(held)], data[data.Country.eq(held)]
            assert held not in set(train.Country)
            assert set(zip(train.Country, train.Respondent_Serial)).isdisjoint(zip(test.Country, test.Respondent_Serial))
            w = (train.weight/train.groupby('Country').weight.transform('sum')).to_numpy()
            w *= len(w)/w.sum()
            wt = test.weight.to_numpy()
            yt = test[TARGETS].to_numpy()
            for cn, group in train.groupby('Country'):
                mass = float(w[train.Country.eq(cn)].sum())
                assert np.isclose(mass, len(train)/train.Country.nunique())
                splits.append(dict(cohort=cohort_name, held_out_country=held, training_country=cn, women=len(group), normalized_weight_mass=mass))
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            a_cat, b_cat = encoder.fit_transform(train[CATS]), encoder.transform(test[CATS])
            cat_names = list(encoder.get_feature_names_out(CATS))
            for col, levels in zip(CATS, encoder.categories_):
                unknowns.append(dict(cohort=cohort_name, held_out_country=held, field=col,
                    unknown_test_women=int((~test[col].isin(levels)).sum()),
                    unknown_test_labels=' | '.join(sorted(set(test[col])-set(levels)))))

            def record(model, pred):
                assert pred.shape == yt.shape and np.isfinite(pred).all()
                for j, target in enumerate(TARGETS):
                    upper = 3 if target == 'worry_item' else 6
                    metrics.append(dict(cohort=cohort_name, held_out_country=held, outcome=target, model=model,
                        train_n=len(train), test_n=len(test),
                        MAE=mean_absolute_error(yt[:, j], pred[:, j], sample_weight=wt),
                        RMSE=np.sqrt(mean_squared_error(yt[:, j], pred[:, j], sample_weight=wt)),
                        R2=r2_score(yt[:, j], pred[:, j], sample_weight=wt),
                        prediction_weighted_mean=np.average(pred[:, j], weights=wt),
                        prediction_min=pred[:, j].min(), prediction_max=pred[:, j].max(),
                        predictions_outside_bounds=int(((pred[:, j]<0) | (pred[:, j]>upper)).sum())))

            y = train[TARGETS].to_numpy()
            record('training_mean', np.tile(np.average(y, weights=w, axis=0), (len(test), 1)))
            for spec in ['individual']+SPECS[cohort_name]:
                a, b, names = a_cat, b_cat, cat_names.copy()
                if spec != 'individual':
                    scaler = StandardScaler()
                    a_num = scaler.fit_transform(train[[spec]])
                    b_num = scaler.transform(test[[spec]])
                    assert scaler.var_[0] > 0
                    a, b = np.column_stack([a_cat, a_num]), np.column_stack([b_cat, b_num])
                    names.append(spec)
                    preprocessing.append(dict(cohort=cohort_name, held_out_country=held, feature=spec,
                        training_mean=float(scaler.mean_[0]), training_scale=float(scaler.scale_[0])))
                model = Ridge(alpha=10).fit(a, y, sample_weight=w)
                record(spec, model.predict(b))
                for j, target in enumerate(TARGETS):
                    for term, value in zip(['intercept']+names, [model.intercept_[j]]+list(model.coef_[j])):
                        coefficients.append(dict(cohort=cohort_name, held_out_country=held, outcome=target,
                            model=spec, term=term, coefficient=float(value)))
    pd.DataFrame(coefficients).to_csv(OUT/'model_coefficients.csv', index=False)
    pd.DataFrame(preprocessing).to_csv(OUT/'training_preprocessing.csv', index=False)
    pd.DataFrame(unknowns).to_csv(OUT/'held_out_category_ledger.csv', index=False)
    pd.DataFrame(splits).to_csv(OUT/'split_ledger.csv', index=False)
    result = pd.DataFrame(metrics)
    result.to_csv(OUT/'country_metrics.csv', index=False)
    summary = result.groupby(['cohort', 'outcome', 'model'], as_index=False).agg(
        countries=('held_out_country', 'nunique'), women=('test_n', 'sum'), macro_MAE=('MAE', 'mean'),
        macro_RMSE=('RMSE', 'mean'), macro_R2=('R2', 'mean'), median_R2=('R2', 'median'),
        predictions_outside_bounds=('predictions_outside_bounds', 'sum'))
    summary.to_csv(OUT/'prediction_summary.csv', index=False)
    individual = result[result.model.eq('individual')][['cohort', 'held_out_country', 'outcome', 'MAE']].rename(columns={'MAE': 'individual_MAE'})
    increment = result[result.model.isin(CONTEXT)].merge(individual, on=['cohort', 'held_out_country', 'outcome'], validate='many_to_one')
    increment['MAE_change'] = increment.MAE-increment.individual_MAE
    increment['improved'] = increment.MAE_change.lt(0)
    increment.to_csv(OUT/'context_increment_by_country.csv', index=False)
    incsum = increment.groupby(['cohort', 'outcome', 'model'], as_index=False).agg(
        countries=('held_out_country', 'nunique'), mean_MAE_change=('MAE_change', 'mean'),
        countries_improved=('improved', 'sum'), min_MAE_change=('MAE_change', 'min'), max_MAE_change=('MAE_change', 'max'))
    incsum.to_csv(OUT/'context_increment_summary.csv', index=False)
    return result, summary, increment, incsum


def figure(increment):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.linewidth': 1,
        'axes.edgecolor': 'black', 'axes.spines.top': False, 'axes.spines.right': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})
    countries = sorted(increment.held_out_country.unique())
    colors = dict(zip(countries, ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#D55E00', '#6A51A3', '#555555']))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey='row')
    for row, target in enumerate(TARGETS):
        for col, cohort in enumerate(SPECS):
            ax = axes[row, col]
            subset = increment[increment.cohort.eq(cohort) & increment.outcome.eq(target)]
            for i, spec in enumerate(SPECS[cohort]):
                group = subset[subset.model.eq(spec)].sort_values('held_out_country')
                for jitter, record in zip(np.linspace(-.14, .14, len(group)), group.itertuples()):
                    ax.scatter(i+jitter, record.MAE_change, color=colors[record.held_out_country], s=35,
                        edgecolors='black', linewidths=.6, zorder=3)
                mean = group.MAE_change.mean()
                ax.plot([i-.21, i+.21], [mean, mean], color='black', lw=2, zorder=4)
            ax.axhline(0, color='black', ls='--', lw=.8, zorder=0)
            ax.set_xticks(range(len(SPECS[cohort])))
            ax.set_xticklabels([LABELS[f] for f in SPECS[cohort]], fontsize=8)
            ax.set_xlim(-.5, len(SPECS[cohort])-.5)
            title = ('Worry item' if row == 0 else 'Depressive-item sum')
            ax.set_title(f"{'abcd'[2*row+col]}   {title} · {8 if col == 0 else 6} countries", loc='left', weight='bold', fontsize=10)
            if col == 0:
                ax.set_ylabel('Context − individual MAE (points)')
    handles = [plt.Line2D([], [], ls='', marker='o', markerfacecolor=colors[c], markeredgecolor='black',
        label=c.replace('/Ivory Coast', ''), markersize=5) for c in countries]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(.5, .035))
    fig.text(.5, .005, 'Dots: held-out countries; bars: equal-country means. Negative values indicate lower error. Exploratory comparison.',
        ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .13, 1, 1))
    folder = OUT/'figures'
    folder.mkdir(exist_ok=True)
    for ext in ['pdf', 'svg', 'png']:
        fig.savefig(folder/f'CP_UN_country_context.{ext}', dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def report(cohorts, summary, incsum):
    lines = ['# UN Women country-context prediction: exploratory results', '',
        'Executed under a protocol saved before these fits; earlier GFS and Colombian results were already known. No main-manuscript integration has been performed.', '',
        '## Sample and question', '',
        'This compares retrospective prediction of Phase II worry (0–3) and depressive-item sums (0–6), using different women in separate countries. The modified since-COVID recall is not a standard GAD/PHQ scale or a measurement of symptom change. Historical national attitudes are contextual features, not individual violence exposure.', '']
    for name, data in cohorts.items():
        lines += [f'- `{name}`: {len(data):,} complete women, {data.Country.nunique()} countries: '+', '.join(sorted(data.Country.unique()))+'.']
    lines += ['', 'The same women are used for both outcomes and all models within each cohort. Cohorts differ, so compare models within a cohort. Supplied weights are used for training/evaluation; countries have equal total training mass and equal weight in the reported mean errors. All category fitting/scaling is restricted to training countries. Fixed ridge alpha=10; no tuning or embeddings.', '',
        '## Out-of-country error', '', '| Cohort | Outcome | Model | Mean MAE | Mean R2 |', '|---|---|---|---:|---:|']
    for r in summary.itertuples():
        lines.append(f'| {r.cohort} | {r.outcome} | {r.model} | {r.macro_MAE:.4f} | {r.macro_R2:.4f} |')
    lines += ['', '## Increment from country context', '',
        '| Cohort | Outcome | Added country feature | Mean MAE change | Countries improved |', '|---|---|---|---:|---:|']
    for r in incsum.itertuples():
        lines.append(f'| {r.cohort} | {r.outcome} | {r.model} | {r.mean_MAE_change:+.4f} | {r.countries_improved}/{r.countries} |')
    lines += ['', 'Negative MAE change means lower error. These selected-country results and overlapping training folds do not provide a precise population effect, a causal estimate, or a percentage of symptoms explained by country conditions. All country-level changes are saved; do not select only a favorable feature or country.', '',
        '## Interpretation limits', '',
        '- This is prediction of cross-sectional reported scores in withheld countries, not a forecast of future depression/anxiety or a test of COVID causing symptoms.',
        '- Historical country estimates have differing years and populations, including ever-married women in Bangladesh/Jordan and differing source age ranges. UN Women includes adult women aged 18 and over. Individual age categories are retained, not silently matched to DHS age bands.',
        '- Work activity and other individual predictors retain exact UN Women categories; no equivalence with DHS cash/kind employment groups is assumed. Generic married and spouse-residence categories remain distinct. Unknown held-out categories are logged.',
        '- Country counts are eight and six. Country profiles cannot explain the Colombia-only direct-IPV coefficient or test how it varies by national context. Neither a favorable nor unfavorable prediction increment establishes an individual attitude–IPV–mental-health pathway.',
        '- Full original PSU/stratum information and religion are unavailable in this release. No inferential design-based confidence intervals, p values, clinical cutoffs or causal effects are estimated here.',
        '- Ridge outputs are not clipped. Any out-of-range predictions, negative R2 values and failures to outperform the training-mean reference remain in the tables.',
        '- This is a narrowly specified exploratory check. A favorable result is not enough to justify a high-capacity country embedding or a claim of clinical utility.', '',
        '## Reproduction', '', 'From the repository root:', '', '```bash',
        'python revision_2026/04_mental_health/women_covid_vaw/context_prediction/run_comparison.py',
        'python revision_2026/04_mental_health/women_covid_vaw/context_prediction/check_comparison.py',
        '```', '', 'Respondent inputs stay in `data_local/`. This directory contains aggregate outputs and code. Read `INDEPENDENT_QC.json` before treating results as numerically verified.']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    (OUT/'FIGURE_LEGENDS.md').write_text('# Figure CP-UN\n\nCountry-context prediction increments in the UN Women Phase II survey. Panels a/c: eight-country national-acceptance comparison; panels b/d: six-country complete-profile comparisons, each adding one feature. Colored dots give held-out-country supplied-weight MAE changes relative to the matched individual-only model; horizontal black bars give equal-country means. Negative values indicate lower error. Worry is one modified-recall item (0–3); depressive-item sum uses two modified-recall items (0–6). These are not standard GAD-2/PHQ-2 scales, prospective changes, causal effects or clinical predictions. Exact categories and fixed ridge alpha=10; preprocessing never uses held-out-country outcomes. Countries differ between columns. No confidence intervals are implied by the spread of dots.\n')


def main():
    js('INPUT_MANIFEST.json', {'saved_utc': datetime.now(timezone.utc).isoformat(), 'files': {
        name: {'path': str(path.relative_to(REV)), 'sha256': sha(path)} for name, path in INPUTS.items()},
        'analysis_script_sha256': sha(Path(__file__)), 'seed': 2026, 'alpha': 10,
        'status': 'post-primary exploratory; protocol saved before this extension fits'})
    np.random.seed(2026)
    cohorts = prepare()
    result, summary, increment, incsum = fit(cohorts)
    assert len(result) == 108 and len(incsum) == 8
    assert len(result[result.model.ne('training_mean')]) == 80
    assert all(result[result.cohort.eq(c)].groupby(['outcome','model']).held_out_country.nunique().eq(8 if c=='national_8' else 6).all() for c in SPECS)
    figure(increment)
    report(cohorts, summary, incsum)
    for item in json.loads((OUT/'INPUT_MANIFEST.json').read_text())['files'].values():
        assert sha(REV/item['path']) == item['sha256']
    js('QC.json', {'status': 'passed', 'cohort_women': {k: len(v) for k, v in cohorts.items()},
        'cohort_country_units': {k: int(v.Country.nunique()) for k, v in cohorts.items()},
        'outcome_specific_ridge_fits': 80, 'country_metric_rows': len(result), 'context_summary_rows': len(incsum),
        'input_hashes_verified': True, 'country_and_person_separation_asserted': True,
        'respondent_rows_exported': False, 'confidence_intervals_or_p_values_generated': False,
        'independent_check': 'run check_comparison.py separately'})
    js('python_environment.json', {'python': sys.version, 'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__})
    print(incsum.to_string(index=False))
    print(json.dumps({k: len(v) for k, v in cohorts.items()}))


if __name__ == '__main__':
    main()
