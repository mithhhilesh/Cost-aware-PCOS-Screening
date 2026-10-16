"""Nested, paired internal evaluation. Does not modify dashboard models or old results."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LOW = ['Age (yrs)', 'BMI', 'Waist:Hip Ratio', 'Cycle(R/I)', 'Cycle length(days)',
       'Weight gain(Y/N)', 'hair growth(Y/N)', 'Skin darkening (Y/N)',
       'Hair loss(Y/N)', 'Pimples(Y/N)', 'Fast food (Y/N)', 'Reg.Exercise(Y/N)']
LABS = ['FSH(mIU/mL)', 'LH(mIU/mL)', 'TSH (mIU/L)', 'AMH(ng/mL)',
        'PRL(ng/mL)', 'PRG(ng/mL)']
ULTRASOUND = ['Follicle No. (L)', 'Follicle No. (R)', 'Avg. F size (L) (mm)',
              'Avg. F size (R) (mm)', 'Endometrium (mm)']
STAGES = {'A0_full': LOW + ['Vit D3 (ng/mL)'] + LABS + ULTRASOUND,
          'A1_no_ultrasound': LOW + ['Vit D3 (ng/mL)'] + LABS,
          'A2_low_vitamin_d': LOW + ['Vit D3 (ng/mL)'], 'A4_low_only': LOW}
# Historical ablation set, not a declaration that each item is a diagnostic criterion.
OVERLAP = ['Cycle(R/I)', 'hair growth(Y/N)', 'Skin darkening (Y/N)'] + ULTRASOUND
BINARY = LOW[5:]
TARGET = 'PCOS (Y/N)'


class FixedCleaning(BaseEstimator, TransformerMixin):
    """Historical fixed rules only; all estimated imputations stay inside CV."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        clean = X.copy().astype(float)
        if np.isinf(clean.to_numpy()).any():
            raise ValueError('Infinite predictor value; correct the source data.')
        for name, limit in {'FSH(mIU/mL)': 100, 'LH(mIU/mL)': 150,
                            'AMH(ng/mL)': 20, 'Vit D3 (ng/mL)': 150}.items():
            if name in clean:
                clean.loc[clean[name] > limit, name] = np.nan
        for name in ['Cycle length(days)', 'Vit D3 (ng/mL)', 'Avg. F size (L) (mm)',
                     'Avg. F size (R) (mm)', 'Endometrium (mm)']:
            if name in clean:
                clean.loc[clean[name] == 0, name] = np.nan
        if 'Cycle(R/I)' in clean:
            clean['Cycle(R/I)'] = clean['Cycle(R/I)'].replace({5: np.nan})
            if not clean['Cycle(R/I)'].dropna().isin([2, 4]).all():
                raise ValueError('Unexpected cycle regularity code.')
            clean['Cycle(R/I)'] = clean['Cycle(R/I)'].map({2: 0, 4: 1})
        for name in BINARY:
            if name in clean and not clean[name].dropna().isin([0, 1]).all():
                raise ValueError(f'Unexpected binary code: {name}')
        return clean


def make_pipeline(features, estimator):
    categorical = [f for f in features if f in BINARY + ['Cycle(R/I)']]
    numeric = [f for f in features if f not in categorical]
    numeric_steps = Pipeline([('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
                              ('scale', StandardScaler())])
    preprocess = ColumnTransformer([
        ('numeric', numeric_steps, numeric),
        ('categorical', SimpleImputer(strategy='most_frequent', keep_empty_features=True), categorical)])
    return Pipeline([('fixed_cleaning', FixedCleaning()), ('preprocess', preprocess),
                     ('classifier', clone(estimator))])


def candidates(seed, include_rf=False):
    models = {f'LR_C={c:g}': LogisticRegression(C=c, max_iter=3000, random_state=seed)
              for c in [0.1, 1.0, 10.0]}
    if include_rf:
        models['RF_depth4_leaf5'] = RandomForestClassifier(
            n_estimators=200, max_depth=4, min_samples_leaf=5, random_state=seed, n_jobs=1)
    return models


def choose_threshold(y, score, target_sensitivity):
    """Maximize specificity subject to INNER-OOF empirical sensitivity constraint."""
    y, score = np.asarray(y), np.asarray(score)
    if not 0 < target_sensitivity <= 1 or set(np.unique(y)) != {0, 1}:
        raise ValueError('Need target sensitivity in (0, 1] and both target classes.')
    if not np.isfinite(score).all() or ((score < 0) | (score > 1)).any():
        raise ValueError('Invalid probability scores.')
    # Largest positive-class cutoff meeting the sensitivity constraint. Raising the
    # threshold cannot decrease specificity; ties use the largest feasible cutoff.
    positive_scores = np.sort(score[y == 1])[::-1]
    required = int(np.ceil(target_sensitivity * len(positive_scores)))
    return float(positive_scores[required - 1])


def metrics(y, score, prediction):
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return {'sensitivity': float(sensitivity), 'specificity': float(specificity),
            'accuracy': float(accuracy_score(y, prediction)),
            'balanced_accuracy': float((sensitivity + specificity) / 2),
            'f1': float(f1_score(y, prediction, zero_division=0)),
            'roc_auc': float(roc_auc_score(y, score)),
            'average_precision': float(average_precision_score(y, score)),
            'brier': float(brier_score_loss(y, score)),
            'screen_positive_rate': float(np.mean(prediction)),
            'ppv': float(tp / (tp + fp)) if tp + fp else None,
            'npv': float(tn / (tn + fn)) if tn + fn else None,
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}


def make_splits(X, y, groups, n_splits, seed):
    splitter = (StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                if groups is None else
                StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed))
    splits = list(splitter.split(X, y, groups))
    for train, valid in splits:
        if set(y.iloc[train].unique()) != {0, 1} or set(y.iloc[valid].unique()) != {0, 1}:
            raise ValueError('A fold lacks a class. Reduce folds or obtain more independent patients.')
        if groups is not None and set(groups.iloc[train]) & set(groups.iloc[valid]):
            raise AssertionError('Patient group crossed a fold boundary.')
    return splits


def evaluate(X, y, target_sensitivity, *, groups=None, outer_folds=5, inner_folds=3,
             seed=42, include_rf=False, analyses=('Primary', 'Sensitivity')):
    """One outer OOF prediction per row per strategy; identical outer splits throughout."""
    outer = make_splits(X, y, groups, outer_folds, seed)
    models = candidates(seed, include_rf)
    predictions, selected, tuning = [], [], []
    for fold, (train, valid) in enumerate(outer):
        X_train, y_train = X.iloc[train], y.iloc[train]
        train_groups = None if groups is None else groups.iloc[train]
        inner = make_splits(X_train, y_train, train_groups, inner_folds, seed + fold + 1)
        for analysis in analyses:
            for stage, original_features in STAGES.items():
                features = [f for f in original_features if analysis == 'Primary' or f not in OVERLAP]
                ranked = []
                for model_name, estimator in models.items():
                    pipeline = make_pipeline(features, estimator)
                    oof_score = cross_val_predict(pipeline, X_train[features], y_train,
                                                 cv=inner, method='predict_proba', n_jobs=1)[:, 1]
                    threshold = choose_threshold(y_train, oof_score, target_sensitivity)
                    inner_metrics = metrics(y_train, oof_score, (oof_score >= threshold).astype(int))
                    tuning.append({'analysis': analysis, 'stage': stage, 'outer_fold': fold,
                                   'candidate': model_name, 'threshold': threshold, **inner_metrics})
                    ranked.append((inner_metrics['specificity'], inner_metrics['roc_auc'],
                                   model_name, threshold, pipeline))
                # Stable order favors simpler/stronger-regularized LR on exact ties.
                best = max(ranked, key=lambda item: (item[0], item[1]))
                _, _, model_name, threshold, pipeline = best
                pipeline.fit(X_train[features], y_train)
                score = pipeline.predict_proba(X.iloc[valid][features])[:, 1]
                selected.append({'analysis': analysis, 'stage': stage, 'outer_fold': fold,
                                 'candidate': model_name, 'threshold': threshold,
                                 'n_features': len(features), 'train_n': len(train), 'test_n': len(valid)})
                for row, probability in zip(valid, score):
                    predictions.append({'row_position': int(row), 'analysis': analysis,
                                        'stage': stage, 'outer_fold': fold, 'y': int(y.iloc[row]),
                                        'score': float(probability),
                                        'prediction': int(probability >= threshold)})
                print(f'Fold {fold + 1}/{outer_folds} | {analysis} | {stage} | {model_name}', flush=True)
    return pd.DataFrame(predictions), pd.DataFrame(selected), pd.DataFrame(tuning)


def summarize(predictions, groups=None, *, bootstrap=2000, seed=42):
    """Paired conditional bootstrap; never a noninferiority/equivalence decision."""
    blocks = {(a, s): frame.sort_values('row_position').reset_index(drop=True)
              for (a, s), frame in predictions.groupby(['analysis', 'stage'])}
    first = next(iter(blocks.values()))
    positions = first.row_position.to_numpy()
    for frame in blocks.values():
        if not np.array_equal(positions, frame.row_position) or not np.array_equal(first.y, frame.y):
            raise ValueError('Strategies must contain aligned predictions for the same records.')
    if groups is None:
        units = [np.array([i]) for i in range(len(first))]
    else:
        labels = np.asarray(groups)[positions]
        units = [np.flatnonzero(labels == g) for g in pd.unique(labels)]
    point = {key: metrics(frame.y, frame.score, frame.prediction) for key, frame in blocks.items()}
    names = ['sensitivity', 'specificity', 'roc_auc', 'average_precision', 'brier',
             'screen_positive_rate', 'ppv', 'npv']
    samples = {(key, name): [] for key in blocks for name in names}
    deltas = {(key, name): [] for key in blocks if key[1] != 'A0_full' for name in names}
    rng = np.random.default_rng(seed)
    accepted = 0
    for _ in range(bootstrap):
        index = np.concatenate([units[i] for i in rng.integers(0, len(units), len(units))])
        if first.y.iloc[index].nunique() < 2:
            continue
        accepted += 1
        replicate = {}
        for key, frame in blocks.items():
            m = metrics(frame.y.iloc[index], frame.score.iloc[index], frame.prediction.iloc[index])
            replicate[key] = m
            for name in names:
                if m[name] is not None:
                    samples[key, name].append(m[name])
        for key, name in deltas:
            a, b = replicate[key][name], replicate[key[0], 'A0_full'][name]
            if a is not None and b is not None:
                deltas[key, name].append(a - b)
    def interval(values):
        return list(map(float, np.quantile(values, [0.025, 0.975]))) if values else [None, None]
    summary, comparisons = [], []
    for key, m in point.items():
        for name in names:
            summary.append({'analysis': key[0], 'stage': key[1], 'metric': name,
                            'estimate': m[name], 'ci_low': interval(samples[key, name])[0],
                            'ci_high': interval(samples[key, name])[1],
                            'valid_bootstrap_replicates': len(samples[key, name])})
        if key[1] != 'A0_full':
            for name in names:
                a, b = m[name], point[key[0], 'A0_full'][name]
                comparisons.append({'analysis': key[0], 'stage': key[1], 'reference': 'A0_full',
                                    'metric': name, 'difference': None if a is None or b is None else a-b,
                                    'ci_low': interval(deltas[key, name])[0],
                                    'ci_high': interval(deltas[key, name])[1],
                                    'valid_bootstrap_replicates': len(deltas[key, name]),
                                    'interpretation': 'Exploratory paired difference; no equivalence claim'})
    return pd.DataFrame(summary), pd.DataFrame(comparisons), point, accepted


def load_data(path, group_column=None):
    frame = pd.read_csv(path)
    frame.columns = frame.columns.str.strip()
    if frame.columns.duplicated().any():
        raise ValueError('Duplicate column names after whitespace normalization.')
    required = STAGES['A0_full'] + [TARGET]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    # Malformed tokens are surfaced, not silently replaced with guessed values.
    # The historical numeric coercion of AMH is retained and counted explicitly.
    X = frame[STAGES['A0_full']].copy()
    amh = pd.to_numeric(X['AMH(ng/mL)'], errors='coerce')
    malformed_amh = int((X['AMH(ng/mL)'].notna() & amh.isna()).sum())
    X['AMH(ng/mL)'] = amh
    X = X.apply(pd.to_numeric, errors='raise')
    y = pd.to_numeric(frame[TARGET], errors='raise')
    if y.isna().any() or set(y.unique()) != {0, 1}:
        raise ValueError('Target must be completely observed and encoded 0/1.')
    groups = None
    if group_column:
        if group_column in required:
            raise ValueError('Grouping column must not be a predictor or target.')
        groups = frame[group_column.strip()]
        if groups.isna().any():
            raise ValueError('Patient grouping column has missing values.')
    clean = FixedCleaning().fit_transform(X)
    audit = {'rows': len(X), 'predictors': len(X.columns), 'positive_count': int(y.sum()),
             'negative_count': int((y == 0).sum()), 'malformed_amh_to_missing': malformed_amh,
             'raw_missing_by_feature': {k: int(v) for k, v in X.isna().sum().items()},
             'cleaned_missing_by_feature': {k: int(v) for k, v in clean.isna().sum().items()},
             'identical_predictor_rows': int(X.duplicated().sum()),
             'patient_grouping': group_column,
             'independence_assumption': ('Patient groups kept together' if groups is not None else
                                       'One independent patient per row; source verification required')}
    return X, y.astype(int), groups, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--target-sensitivity', type=float, required=True,
                        help='Experimental inner-OOF target; not a clinical guarantee')
    parser.add_argument('--outer-folds', type=int, default=5)
    parser.add_argument('--inner-folds', type=int, default=3)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--include-rf', action='store_true')
    parser.add_argument('--group-column')
    args = parser.parse_args()
    if not 0 < args.target_sensitivity <= 1 or min(args.outer_folds, args.inner_folds) < 2:
        parser.error('Invalid sensitivity target or fold count.')
    if args.bootstrap < 100:
        parser.error('Use at least 100 bootstrap replicates; 2000 is the study default.')
    if args.output.exists():
        parser.error('Output directory already exists. Choose a new run name to preserve results.')
    X, y, groups, audit = load_data(args.data, args.group_column)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.pop('data')  # Do not publish the local patient-data path.
    config['data_sha256'] = hashlib.sha256(args.data.read_bytes()).hexdigest()
    config['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    config['versions'] = {'python': platform.python_version(), **{
        p: version(p) for p in ['numpy', 'pandas', 'scipy', 'scikit-learn']}}
    result = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                            cwd=Path(__file__).parent)
    config['base_commit'] = result.stdout.strip() if result.returncode == 0 else None
    config['status'] = 'Exploratory internal validation; protocol chosen after earlier data exploration'
    config['calibration'] = 'No post-hoc calibrator fitted; Brier score assesses raw model probabilities'
    args.output.mkdir(parents=True)
    (args.output / 'run_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    (args.output / 'data_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    predictions, selection, tuning = evaluate(
        X, y, args.target_sensitivity, groups=groups, outer_folds=args.outer_folds,
        inner_folds=args.inner_folds, seed=args.seed, include_rf=args.include_rf)
    # Row-level predictions stay local. No patient IDs or feature values are exported.
    predictions.to_csv(args.output / 'PRIVATE_oof_predictions.csv', index=False)
    selection.to_csv(args.output / 'selected_by_fold.csv', index=False)
    tuning.to_csv(args.output / 'inner_tuning.csv', index=False)
    summary, differences, point, accepted = summarize(
        predictions, groups, bootstrap=args.bootstrap, seed=args.seed)
    summary.to_csv(args.output / 'metrics.csv', index=False)
    differences.to_csv(args.output / 'paired_differences.csv', index=False)
    counts = [{'analysis': a, 'stage': s, **m} for (a, s), m in point.items()]
    pd.DataFrame(counts).to_csv(args.output / 'confusion_and_metrics.csv', index=False)
    text = ['# Corrected exploratory internal evaluation', '',
            'Each record has one outer-fold prediction per strategy. Inner folds alone select',
            'the model and threshold. All feature groups share outer and inner splits.', '',
            f'Experimental inner sensitivity target: {args.target_sensitivity:.0%}.',
            'This is not a guaranteed held-out sensitivity or a clinical recommendation.', '',
            '| Analysis | Stage | Sensitivity | Specificity | AUROC | False negatives |',
            '|---|---|---:|---:|---:|---:|']
    for (analysis, stage), m in point.items():
        text.append(f"| {analysis} | {stage} | {m['sensitivity']:.3f} | {m['specificity']:.3f} | "
                    f"{m['roc_auc']:.3f} | {m['fn']} |")
    text += ['', '## Interpretation limits', '',
             '- Confidence intervals resample paired outer-OOF predictions by patient/row.',
             '  They are conditional on fitted cross-validation models and do not capture',
             '  all training/selection variability or eliminate cross-fold dependence.',
             '- No repeated-fold significance test, equivalence claim, or cost-effective winner.',
             '- PPV/NPV reflect this sample prevalence, not a general screening population.',
             '- Sensitivity analysis removes a historical overlap set, not only formal diagnostic criteria.',
             '- No external validation, prospective validation, economic analysis, or calibrated-risk claim.',
             '- Fixed cleaning cutoffs remain historical assumptions requiring clinical review.',
             '- PRIVATE_oof_predictions.csv is record-level health-related output; keep it local.',
             f'- Accepted two-class bootstrap replicates: {accepted}/{args.bootstrap}.']
    (args.output / 'REPORT.md').write_text('\n'.join(text) + '\n', encoding='utf-8')
    print(f'Completed. Aggregate report: {args.output / "REPORT.md"}')


if __name__ == '__main__':
    main()
