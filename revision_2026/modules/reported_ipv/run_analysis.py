"""New retrospective ENDES IPV classification; writes owned paths only.

Private rows, graphs, transformations, predictions and states remain data_local.
Public outputs contain no individual identities or neighbour memberships.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import scipy
from scipy.special import expit
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

BASE = Path(__file__).resolve().parent
LOCAL = BASE / 'data_local'
OUT = BASE / 'results'
FROZEN = ['PROTOCOL.md', 'CONFIG.json', 'core.py', 'run_analysis.py']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def js(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def config():
    return js(BASE / 'CONFIG.json')


def cohort(temporal=False):
    c = config()
    prefix = 'temporal' if temporal else 'source'
    path = (BASE / c[prefix + '_cohort']).resolve()
    ready = js(BASE / c[prefix + '_ready'])
    assert sha(path) == ready['summary']['prepared_cohort_sha256']
    categories = c['categorical'] + ['row_id', 'psu', 'stratum']
    f = pd.read_csv(path, dtype={x: str for x in categories}, keep_default_na=False)
    assert len(f) == ready['summary']['prepared_records'] and f.row_id.is_unique
    assert f[c['target_items']].isin([0, 1, 2, 3]).all().all()
    f['y'] = f[c['target_items']].isin([1, 2]).any(axis=1).astype(int)
    assert np.array_equal(f.y, f.physical_recent.astype(bool) | f.sexual_recent.astype(bool))
    assert f[c['attitudes']].isin([0, 1]).all().all()
    assert f.age.between(18, 49).all()
    if temporal:
        flags_path = BASE / c['temporal_flags']
        assert sha(flags_path) == ready['temporal_validation']['test_flags_sha256']
        flags = pd.read_csv(flags_path, dtype={'row_id': str})
        assert flags.row_id.is_unique and set(flags.row_id) == set(f.row_id)
        f = f.merge(flags, on='row_id', how='left', validate='one_to_one', sort=False)
        assert f[c['new_psu_flag']].isin([0, 1]).all()
        assert f.interview_year.eq(2024).all() and f.health_interview_year.eq(2024).all()
    return f


def assignments(f):
    c = config()
    units = np.asarray(sorted(f.psu.unique()), dtype=object)
    permuted = np.random.default_rng(c['fold_seed']).permutation(units)
    mapping = {str(psu): fold for fold, group in enumerate(np.array_split(permuted, c['folds'])) for psu in group}
    fold = f.psu.map(mapping).to_numpy(int)
    assert len(set(fold)) == c['folds']
    return fold


def transform(train, test):
    c = config()
    a, b = train[c['numeric']].to_numpy(float), test[c['numeric']].to_numpy(float)
    mean, sd = a.mean(axis=0), a.std(axis=0)
    sd[sd < 1e-12] = 1.
    pieces, queries = [(a - mean) / sd], [(b - mean) / sd]
    graph, graph_queries = [pieces[0]], [queries[0]]
    columns = list(c['numeric']); levels, unknown = {}, {}
    for name in c['categorical']:
        values = sorted(train[name].unique())
        levels[name] = values
        one = np.column_stack([train[name].eq(v).to_numpy(float) for v in values])
        two = np.column_stack([test[name].eq(v).to_numpy(float) for v in values])
        pieces.append(one); queries.append(two)
        graph.append(one / np.sqrt(2)); graph_queries.append(two / np.sqrt(2))
        columns.extend(name + '=' + v for v in values)
        unknown[name] = int((~test[name].isin(values)).sum())
    x, q = np.column_stack(pieces), np.column_stack(queries)
    gx, gq = np.column_stack(graph), np.column_stack(graph_queries)
    forbidden = ('D10', 'recent', 'lifetime', 'phq', 'QS7', 'row_id', 'psu', 'stratum')
    assert not any(any(v in col for v in forbidden) for col in columns)
    return x, q, gx, gq, dict(numeric=c['numeric'], mean=mean.tolist(), scale=sd.tolist(),
        categories=levels, columns=columns, unknown_test_levels=unknown,
        categorical_graph_scale=float(1 / np.sqrt(2)), donor_only=True)


def neighbours(query, donor, donor_ids, query_psu, donor_psu):
    c = config()
    order = np.asarray(sorted(range(len(donor)), key=lambda i: hashlib.sha256(
        (str(c['tie_seed']) + ':' + str(donor_ids[i])).encode()).hexdigest()), dtype=int)
    reference = donor[order]; norms = np.sum(reference ** 2, axis=1)
    result = np.empty((len(query), c['neighbours']), dtype=np.int32)
    for start in range(0, len(query), c['distance_block']):
        x = query[start:start + c['distance_block']]
        d = np.sum(x ** 2, axis=1)[:, None] + norms - 2 * (x @ reference.T)
        np.maximum(d, 0, out=d)
        np.round(d, c['distance_decimals'], out=d)
        same = query_psu[start:start + len(x), None] == donor_psu[order][None, :]
        d[same] = np.inf
        chosen = np.argsort(d, axis=1, kind='stable')[:, :c['neighbours']]
        assert np.isfinite(np.take_along_axis(d, chosen, axis=1)).all()
        result[start:start + len(x)] = order[chosen]
    assert not (query_psu[:, None] == donor_psu[result]).any()
    return result


def random_graph(query_psu, donor_psu, seed):
    c = config(); rng = np.random.default_rng(seed)
    result = np.empty((len(query_psu), c['neighbours']), dtype=np.int32)
    # Cache possible donor rows for each query PSU; no target information enters.
    eligible = {v: np.flatnonzero(donor_psu != v) for v in np.unique(query_psu)}
    for i, psu in enumerate(query_psu):
        result[i] = rng.choice(eligible[psu], c['neighbours'], replace=False)
    assert not (query_psu[:, None] == donor_psu[result]).any()
    return result


def prepare_fold(name, train, test, ordinal):
    c = config()
    x, q, gx, gq, info = transform(train, test)
    donor_psu, query_psu = train.psu.to_numpy(), test.psu.to_numpy()
    real_train = neighbours(gx, gx, train.row_id.to_numpy(), donor_psu, donor_psu)
    real_test = neighbours(gq, gx, train.row_id.to_numpy(), query_psu, donor_psu)
    random_train = random_graph(donor_psu, donor_psu, c['random_graph_seed'] + ordinal * 2)
    random_test = random_graph(query_psu, donor_psu, c['random_graph_seed'] + ordinal * 2 + 1)
    data = dict(X_train=x, X_test=q, attitudes_train=train[c['attitudes']].to_numpy(float),
        attitudes_test=test[c['attitudes']].to_numpy(float), y_train=train.y.to_numpy(float),
        y_test=test.y.to_numpy(float), real_train=real_train, real_test=real_test,
        random_train=random_train, random_test=random_test)
    np.savez_compressed(LOCAL / 'folds' / (name + '.npz'), **data)
    train[['row_id', 'psu', 'stratum']].to_csv(LOCAL / 'folds' / (name + '_train.csv.gz'), index=False)
    test[['row_id', 'psu', 'stratum']].to_csv(LOCAL / 'folds' / (name + '_test.csv.gz'), index=False)
    info.update(name=name, ordinal=ordinal, train_rows=len(train), test_rows=len(test),
        arrays={k: array_hash(v) for k, v in data.items()}, attitude_columns=c['attitudes'])
    save(LOCAL / 'folds' / (name + '.json'), info)
    return dict(fold=name, train_women=len(train), test_women=len(test),
        train_positive=int(train.y.sum()), test_positive=int(test.y.sum()),
        train_PSUs=int(train.psu.nunique()), test_PSUs=int(test.psu.nunique()),
        shared_PSUs=len(set(train.psu) & set(test.psu)), demo_dimensions=x.shape[1],
        attitude_dimensions=x.shape[1] + len(c['attitudes']), neighbours=c['neighbours'],
        self_or_ownPSU_donors=0, heldout_donors=0)


def prepare():
    assert not (BASE / 'FREEZE.json').exists(), 'Do not change frozen preparation'
    assert not (LOCAL / 'PREPARED.json').exists(), 'Preserve previously prepared inputs'
    for folder in (LOCAL / 'folds', LOCAL / 'states', OUT):
        folder.mkdir(parents=True, exist_ok=True)
    source, temporal = cohort(), cohort(True)
    fold = assignments(source)
    source[['row_id', 'psu', 'stratum', 'y']].assign(fold=fold).to_csv(LOCAL / 'primary_cohort.csv.gz', index=False)
    rows = []
    for k in range(config()['folds']):
        train, test = source.loc[fold != k], source.loc[fold == k]
        assert not set(train.psu) & set(test.psu)
        rows.append(prepare_fold('primary_' + str(k), train, test, k))
        print('Prepared primary fold', k + 1, flush=True)
    strict = source.interview_year.eq(2023) & source.health_interview_year.eq(2023)
    train = source.loc[strict]
    assert len(train) == 10259 and len(temporal) == 9847
    new = temporal[config()['new_psu_flag']].eq(1)
    assert int(new.sum()) == 3250 and not set(temporal.loc[new, 'psu']) & set(train.psu)
    temporal[['row_id', 'psu', 'stratum', 'y', config()['new_psu_flag']]].to_csv(LOCAL / 'temporal_cohort.csv.gz', index=False)
    rows.append(prepare_fold('temporal', train, temporal, config()['folds']))
    pd.DataFrame(rows).to_csv(OUT / 'fold_inventory.csv', index=False)
    counts = []
    for name, frame in [('primary', source), ('strict_training_2023', train), ('all_2024', temporal), ('new_psu_2024', temporal.loc[new])]:
        counts.append(dict(domain=name, women=len(frame), positive=int(frame.y.sum()),
            negative=int((1 - frame.y).sum()), PSUs=int(frame.psu.nunique()), strata=int(frame.stratum.nunique()),
            any_attitude=int(frame[config()['attitudes']].any(axis=1).sum()),
            actual_individual_years=';'.join(map(str, sorted(frame.interview_year.unique()))),
            actual_health_years=';'.join(map(str, sorted(frame.health_interview_year.unique())))))
    pd.DataFrame(counts).to_csv(OUT / 'class_counts.csv', index=False)
    save(LOCAL / 'PREPARED.json', dict(status='prepared_no_fits',
        files={str(p.relative_to(BASE)): sha(p) for p in sorted(LOCAL.rglob('*')) if p.is_file()},
        source_files={str((BASE / config()[k]).resolve()): sha(BASE / config()[k]) for k in
            ['source_cohort', 'source_ready', 'temporal_cohort', 'temporal_ready', 'temporal_flags']}))


def freeze():
    assert not (BASE / 'FREEZE.json').exists(), 'Never overwrite fit freeze'
    assert js(LOCAL / 'PREPARED.json')['status'] == 'prepared_no_fits'
    source_gate = BASE / 'source_review/SOURCE_GATE.json'
    temporal_gate = BASE / 'source_review/TEMPORAL_AND_FOLD_GATE.json'
    graph_gate = BASE / 'source_review/GRAPH_INPUT_GATE.json'
    assert js(source_gate)['status'] == 'passed', 'Independent source gate required before fitting'
    assert js(temporal_gate)['status'] == 'passed', 'Independent temporal/fold gate required before fitting'
    assert js(graph_gate)['status'] == 'passed', 'Independent graph-input gate required before fitting'
    prepared = js(LOCAL / 'PREPARED.json')
    assert all(sha(BASE / p) == h for p, h in prepared['files'].items())
    assert all(sha(p) == h for p, h in prepared['source_files'].items())
    save(BASE / 'FREEZE.json', dict(status='frozen_before_fitting', retrospective_exploratory=True,
        inspected_labels_before_freeze=True, client_date='2026-10-22',
        recorded_execution_utc=datetime.now(timezone.utc).isoformat(),
        producer_files={p: sha(BASE / p) for p in FROZEN},
        prepared_manifest_sha256=sha(LOCAL / 'PREPARED.json'), independent_source_sha256=sha(source_gate),
        independent_temporal_fold_sha256=sha(temporal_gate), independent_graph_input_sha256=sha(graph_gate)))


def check_frozen():
    frozen = js(BASE / 'FREEZE.json')
    assert all(sha(BASE / p) == h for p, h in frozen['producer_files'].items())
    assert sha(LOCAL / 'PREPARED.json') == frozen['prepared_manifest_sha256']
    for field, name in [('independent_source_sha256', 'SOURCE_GATE.json'),
                        ('independent_temporal_fold_sha256', 'TEMPORAL_AND_FOLD_GATE.json'),
                        ('independent_graph_input_sha256', 'GRAPH_INPUT_GATE.json')]:
        assert sha(BASE / 'source_review' / name) == frozen[field]
    prepared = js(LOCAL / 'PREPARED.json')
    assert all(sha(BASE / p) == h for p, h in prepared['files'].items())
    assert all(sha(p) == h for p, h in prepared['source_files'].items())


def neural_inputs(data, block, arm):
    if block == 'D':
        x, q = data['X_train'], data['X_test']
    else:
        x = np.column_stack([data['X_train'], data['attitudes_train']])
        q = np.column_stack([data['X_test'], data['attitudes_test']])
    if arm == 'self':
        mx, mq = x, q
    else:
        mx, mq = x[data[arm + '_train']].mean(axis=1), x[data[arm + '_test']].mean(axis=1)
    return np.column_stack([x, mx]), np.column_stack([q, mq])


def fit_models():
    check_frozen()
    assert not (OUT / 'FIT_COMPLETION.json').exists(), 'Preserve completed fit run'
    assert not list((LOCAL / 'states').glob('*')), 'Partial fits require explicit review; do not overwrite'
    import core
    import torch
    started = time.monotonic(); c = config(); records = []; predictions = []
    for file in sorted((LOCAL / 'folds').glob('*.npz')):
        fold_name = file.stem; data = dict(np.load(file))
        test = pd.read_csv(LOCAL / 'folds' / (fold_name + '_test.csv.gz'), dtype=str)
        record_pred = test[['row_id', 'psu', 'stratum']].copy()
        record_pred['y'] = data['y_test']; record_pred['fold'] = fold_name
        record_pred['prevalence'] = np.mean(data['y_train'])
        for block in ['D', 'A']:
            x = data['X_train'] if block == 'D' else np.column_stack([data['X_train'], data['attitudes_train']])
            q = data['X_test'] if block == 'D' else np.column_stack([data['X_test'], data['attitudes_test']])
            for algorithm in ['logistic', 'boosting']:
                model = LogisticRegression(**c['logistic']) if algorithm == 'logistic' else HistGradientBoostingClassifier(**c['boosting'])
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    model.fit(x, data['y_train'])
                assert not caught, [(str(v.message), v.category.__name__) for v in caught]
                key = fold_name + '__' + algorithm + '_' + block
                assert np.isfinite(model.predict_proba(q)).all()
                record_pred[algorithm + '_' + block] = model.predict_proba(q)[:, 1]
                joblib.dump(model, LOCAL / 'states' / (key + '.joblib'), compress=3)
                records.append(dict(fold=fold_name, algorithm=algorithm, block=block, state=key + '.joblib',
                    training_rows=len(x), test_rows=len(q), features=x.shape[1],
                    iterations=np.asarray(model.n_iter_).astype(int).tolist(), warnings=[]))
            for arm in c['neural_arms']:
                tx, tq = neural_inputs(data, block, arm)
                for seed in c['initializations']:
                    key = fold_name + '__' + arm + '_' + block + '__' + str(seed)
                    _, test_logit, state, diagnostics = core.fit(tx, data['y_train'], tq, seed, c)
                    np.savez_compressed(LOCAL / 'states' / (key + '.npz'), **state)
                    initial = diagnostics.pop('initial')
                    diagnostics['initial_sha256'] = {k: array_hash(v) for k, v in initial.items()}
                    diagnostics.update(fold=fold_name, algorithm=arm, block=block, seed=seed,
                        state=key + '.npz', training_rows=len(tx), test_rows=len(tq), features=tx.shape[1])
                    records.append(diagnostics)
                    record_pred[arm + '_' + block + '__' + str(seed)] = expit(test_logit)
                record_pred[arm + '_' + block] = record_pred[
                    [arm + '_' + block + '__' + str(seed) for seed in c['initializations']]].mean(axis=1)
            print('Fitted', fold_name, block, 'elapsed_s', round(time.monotonic() - started, 1), flush=True)
        predictions.append(record_pred)
    pd.concat(predictions).to_csv(LOCAL / 'predictions.csv.gz', index=False)
    save(LOCAL / 'fit_records.json', records)
    check_frozen()
    save(OUT / 'FIT_COMPLETION.json', dict(status='fitted_pending_independent_model_review',
        neural_fits=sum(v['algorithm'] in c['neural_arms'] for v in records),
        logistic_fits=sum(v['algorithm'] == 'logistic' for v in records),
        boosting_fits=sum(v['algorithm'] == 'boosting' for v in records),
        elapsed_seconds=time.monotonic() - started, fixed_updates=c['epochs'],
        freeze_sha256=sha(BASE / 'FREEZE.json'), predictions_sha256=sha(LOCAL / 'predictions.csv.gz')))
    save(BASE / 'ENVIRONMENT.json', dict(python=platform.python_version(), numpy=np.__version__,
        pandas=pd.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
        torch=torch.__version__, joblib=joblib.__version__, cpu_threads=1,
        platform=platform.platform(), no_new_installation=True, no_download=True))


def metric(y, p):
    c = config(); y = np.asarray(y, dtype=int); p = np.asarray(p, dtype=float)
    assert set(np.unique(y)) <= {0, 1} and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    bounded = np.clip(p, 1e-15, 1 - 1e-15); pred = p >= c['threshold']
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    tn = int((~pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
    sens = tp / (tp + fn) if tp + fn else None
    spec = tn / (tn + fp) if tn + fp else None
    return dict(Brier=float(np.mean((p - y) ** 2)),
        log_loss=float(np.mean(-y * np.log(bounded) - (1 - y) * np.log1p(-bounded))),
        ROC_AUC=float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        average_precision=float(average_precision_score(y, p)) if y.sum() else None,
        accuracy=(tp + tn) / len(y), balanced_accuracy=(sens + spec) / 2 if sens is not None and spec is not None else None,
        sensitivity=sens, specificity=spec, PPV=tp / (tp + fp) if tp + fp else None,
        TP=tp, FP=fp, TN=tn, FN=fn, women=len(y), positives=int(y.sum()),
        prevalence=float(y.mean()), mean_prediction=float(p.mean()), threshold=c['threshold'])


def comparisons():
    pairs = [(a + '_A', a + '_D') for a in ['logistic', 'boosting', 'real', 'self', 'random']]
    pairs += [('real_' + b, ref + '_' + b) for b in ['D', 'A'] for ref in ['logistic', 'self', 'random']]
    pairs += [('boosting_A', 'logistic_A')]
    pairs += [(a + '_' + b, 'prevalence') for a in ['logistic', 'boosting', 'real', 'self', 'random'] for b in ['D', 'A']]
    return pairs


def aggregate():
    check_frozen(); assert (OUT / 'FIT_COMPLETION.json').exists()
    c = config(); p = pd.read_csv(LOCAL / 'predictions.csv.gz', dtype={'row_id': str, 'psu': str, 'stratum': str})
    primary = pd.read_csv(LOCAL / 'primary_cohort.csv.gz', dtype={'row_id': str, 'psu': str, 'stratum': str})
    temporal = pd.read_csv(LOCAL / 'temporal_cohort.csv.gz', dtype={'row_id': str, 'psu': str, 'stratum': str})
    primary = primary[['row_id', 'fold']].rename(columns={'fold': 'assignment'}).merge(
        p.loc[p.fold.str.startswith('primary')], on='row_id', validate='one_to_one', sort=False)
    temporal = temporal[['row_id', c['new_psu_flag']]].merge(p.loc[p.fold.eq('temporal')], on='row_id', validate='one_to_one', sort=False)
    assert len(primary) == 10267 and len(temporal) == 9847
    models = ['prevalence'] + [a + '_' + b for a in ['logistic', 'boosting', 'real', 'self', 'random'] for b in ['D', 'A']]
    seed_columns = [a + '_' + b + '__' + str(s) for a in c['neural_arms'] for b in ['D', 'A'] for s in c['initializations']]
    metrics = []; initialization = []; folds = []; calibration = []; contrasts = []; draws = []
    domains = [('primary', primary), ('all_2024', temporal),
               ('new_psu_2024', temporal.loc[temporal[c['new_psu_flag']].eq(1)])]
    for domain, f in domains:
        y = f.y.to_numpy(int)
        for name in models:
            metrics.append(dict(domain=domain, model=name, **metric(y, f[name])))
            prob = f[name].to_numpy(float)
            bin_id = np.minimum((prob * 10).astype(int), 9)
            for binno in range(10):
                choose = bin_id == binno
                calibration.append(dict(domain=domain, model=name, bin=binno, lower=binno / 10,
                    upper=(binno + 1) / 10, women=int(choose.sum()),
                    mean_prediction=float(prob[choose].mean()) if choose.any() else None,
                    observed_prevalence=float(y[choose].mean()) if choose.any() else None))
        for name in seed_columns:
            initialization.append(dict(domain=domain, model=name.split('__')[0], seed=int(name.split('__')[1]), **metric(y, f[name])))
        if domain == 'primary':
            for fold, group in f.groupby('fold', sort=True):
                for name in models:
                    folds.append(dict(domain=domain, fold=fold, model=name, **metric(group.y, group[name])))
        units = sorted(f.psu.unique()); which = pd.Categorical(f.psu, categories=units).codes
        number = np.bincount(which, minlength=len(units))
        rng = np.random.default_rng(c['bootstrap_seed'])
        count = rng.multinomial(len(units), np.ones(len(units)) / len(units), size=c['bootstrap_draws'])
        denominator = count @ number
        for loss_name in ['Brier', 'log_loss']:
            loss = []
            for name in models:
                prob = np.clip(f[name].to_numpy(float), 1e-15, 1 - 1e-15)
                values = (f[name].to_numpy(float) - y) ** 2 if loss_name == 'Brier' else -y * np.log(prob) - (1 - y) * np.log1p(-prob)
                loss.append(np.bincount(which, weights=values, minlength=len(units)))
            meanloss = (count @ np.column_stack(loss)) / denominator[:, None]
            for model, reference in comparisons():
                delta = meanloss[:, models.index(model)] - meanloss[:, models.index(reference)]
                observed = next(r[loss_name] for r in metrics if r['domain'] == domain and r['model'] == model) - next(
                    r[loss_name] for r in metrics if r['domain'] == domain and r['model'] == reference)
                lo, hi = np.quantile(delta, [.025, .975])
                contrasts.append(dict(domain=domain, model=model, reference=reference, metric=loss_name,
                    difference=observed, lower=float(lo), upper=float(hi), draws=c['bootstrap_draws'],
                    primary=bool((model, reference) in [tuple(c['primary_attitude_contrast'])] + [tuple(v) for v in c['primary_graph_contrasts']]),
                    interval_scope='paired observed-PSU bootstrap; fixed predictions/training/graph; not population or full-refit inference'))
                draws.extend(dict(domain=domain, model=model, reference=reference, metric=loss_name,
                    draw=i, difference=float(value)) for i, value in enumerate(delta))
    for name, rows in [('metrics', metrics), ('initialization_metrics', initialization), ('fold_metrics', folds),
                       ('calibration', calibration), ('paired_comparisons', contrasts), ('bootstrap_draws', draws)]:
        filename = name + ('.csv.gz' if name == 'bootstrap_draws' else '.csv')
        pd.DataFrame(rows).to_csv(OUT / filename, index=False)
    save(OUT / 'AGGREGATE_COMPLETION.json', dict(status='produced_pending_independent_review',
        rows=dict(metrics=len(metrics), initialization_metrics=len(initialization), fold_metrics=len(folds),
            calibration=len(calibration), paired_comparisons=len(contrasts), bootstrap_draws=len(draws)),
        bootstrap=dict(draws=c['bootstrap_draws'], seed=c['bootstrap_seed'],
            resampling='global whole observed PSUs per domain, paired across all model predictions', fixed_predictions=True),
        source_files={p.name: sha(p) for p in OUT.iterdir() if p.name.endswith(('.csv', '.csv.gz'))}))
    save(BASE / 'SOURCE_VALUES.json', dict(metrics=metrics, paired_comparisons=contrasts,
        class_counts=pd.read_csv(OUT / 'class_counts.csv').to_dict('records'),
        fold_inventory=pd.read_csv(OUT / 'fold_inventory.csv').to_dict('records'),
        primary_metric='Brier', lower_loss_better=True, exploratory=True,
        average_precision_definition='stepwise precision-recall summary; not trapezoidal PR-AUC',
        cohort_scope='equal-person observed complete linked respondents; no validated joint population weight',
        graph_scope='analyst-constructed demographic similarity, not an observed social network',
        source_files={p.name: sha(p) for p in OUT.iterdir() if p.name.endswith(('.csv', '.csv.gz'))}))
    print(pd.DataFrame(metrics)[['domain', 'model', 'Brier', 'log_loss', 'ROC_AUC', 'average_precision', 'sensitivity']].to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--fit', action='store_true')
    parser.add_argument('--aggregate', action='store_true')
    args = parser.parse_args()
    if args.prepare: prepare()
    if args.freeze: freeze()
    if args.fit: fit_models()
    if args.aggregate: aggregate()
