"""Shared notebook/dashboard inference; no training or evaluation on import."""
from pathlib import Path
import json
import platform
from importlib.metadata import version
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone

RANKING = ['recall_mean', 'f1_mean', 'roc_auc_mean', 'accuracy_mean']
LIMITS = {'FSH(mIU/mL)': 100, 'LH(mIU/mL)': 150,
          'AMH(ng/mL)': 20, 'Vit D3 (ng/mL)': 150}
ZERO_MISSING = ['Cycle length(days)', 'Vit D3 (ng/mL)',
                'Avg. F size (L) (mm)', 'Avg. F size (R) (mm)', 'Endometrium (mm)']


def rank_candidates(results):
    required = ['analysis', 'feature_set', 'model'] + RANKING
    if not set(required).issubset(results):
        raise ValueError('Run the seven-model hybrid comparison first.')
    primary = results.loc[results.analysis == 'Primary'].copy()
    if primary.empty or not np.isfinite(primary[RANKING].to_numpy()).all():
        raise ValueError('Primary CV results are missing or non-finite.')
    return primary.sort_values(RANKING + ['model'], ascending=[False]*4 + [True], kind='stable')


def prepare_patient(frame):
    """Mirror the original pre-split numeric cleaning, retaining column spelling/order."""
    clean = frame.copy().astype(float)
    for name, limit in LIMITS.items():
        if name in clean:
            clean.loc[clean[name] > limit, name] = np.nan
    for name in ZERO_MISSING:
        if name in clean:
            clean.loc[clean[name] == 0, name] = np.nan
    if 'Cycle(R/I)' in clean:
        clean.loc[clean['Cycle(R/I)'] == 5, 'Cycle(R/I)'] = np.nan
    return clean


def feature_schema(X, cost_tiers):
    schema = {}
    for name in X:
        vals = X[name].dropna()
        choices = sorted(float(v) for v in vals.unique())
        category = choices if name == 'Cycle(R/I)' or set(choices).issubset({0., 1.}) else None
        schema[name] = {'label': name.strip(), 'choices': category,
                        'min_observed': float(vals.min()), 'max_observed': float(vals.max()),
                        'tier': next((g for g, fs in cost_tiers.items() if name in fs), 'Other')}
    return schema


def export_bundle(X, y, results, estimators, stages, cost_tiers, output_dir):
    """CV selects the model; full-data refit is solely for future inference."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ranked = rank_candidates(results)
    winners = ranked.groupby('feature_set', sort=False).head(1)
    if set(winners.feature_set) != set(stages):
        raise ValueError('Every primary feature stage must have comparison results.')
    if not X.index.equals(y.index) or set(y.unique()) != {0, 1}:
        raise ValueError('X/y alignment or binary target encoding is invalid.')
    schema = feature_schema(X, cost_tiers)
    manifest = {'schema_version': 1, 'selection_rule': 'Recall > F1 > ROC-AUC > accuracy; alphabetical tie-break',
                'analysis': 'Primary', 'threshold': 0.5, 'groups': {},
                'versions': {'python': platform.python_version(), **{p: version(p) for p in
                    ['numpy', 'pandas', 'scikit-learn', 'shap', 'streamlit', 'xgboost', 'lightgbm', 'joblib']}}}
    fitted_by_features = {}
    for stage, features in stages.items():
        features = list(features)
        row = winners.loc[winners.feature_set == stage].iloc[0]
        model_name = row['model']
        cache_key = (model_name, tuple(features))
        if cache_key not in fitted_by_features:
            fitted_by_features[cache_key] = clone(estimators[model_name]).fit(X[features], y)
        bundle = {'model': fitted_by_features[cache_key], 'features': features,
                  'model_name': model_name, 'stage': stage, 'threshold': 0.5,
                  'schema': {f: schema[f] for f in features},
                  'background': X[features].sample(min(30, len(X)), random_state=42).copy()}
        joblib.dump(bundle, output_dir / (stage + '.joblib'), compress=3)
        manifest['groups'][stage] = {'model': model_name, 'n_features': len(features),
                                    'features': features,
                                    'cv_metrics': {m: float(row[m]) for m in RANKING}}
    results.to_csv(output_dir / 'model_comparison.csv', index=False)
    winners.to_csv(output_dir / 'selected_models.csv', index=False)
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return winners, manifest


def probability_function(model, features):
    positive = list(model.classes_).index(1)
    def predict(values):
        frame = pd.DataFrame(values, columns=features)
        return model.predict_proba(frame)[:, positive]
    return predict


def explain_probability(model, background, patients, permutations=10):
    """Explain the COMPLETE fitted pipeline/ensemble, in positive-class probability units."""
    import shap
    features = list(background.columns)
    if list(patients.columns) != features:
        raise ValueError('Explanation input columns must match training order exactly.')
    predict = probability_function(model, features)
    masker = shap.maskers.Independent(background, max_samples=min(30, len(background)))
    explainer = shap.PermutationExplainer(predict, masker, feature_names=features, seed=42)
    explanations = explainer(patients, max_evals=(2*len(features)+1)*permutations,
                             silent=True)
    reconstructed = explanations.base_values + explanations.values.sum(axis=1)
    if not np.allclose(reconstructed, predict(patients), atol=1e-6):
        raise RuntimeError('SHAP additivity check failed; do not show this explanation.')
    return explanations


def contribution_table(explanation, patient):
    effects = np.asarray(explanation.values)
    table = pd.DataFrame({'Feature': [f.strip() for f in patient.index],
                          'Entered / cleaned value': patient.to_numpy(),
                          'SHAP change (percentage points)': effects * 100,
                          'Direction': np.where(effects > 1e-9, 'Raises model score',
                                      np.where(effects < -1e-9, 'Lowers model score', 'Negligible'))})
    return table.iloc[np.argsort(-np.abs(effects))].reset_index(drop=True)


def patient_explanation(explanation, patient, score, threshold=0.5):
    table = contribution_table(explanation, patient)
    baseline = float(explanation.base_values)
    lines = [f'The model score is {score:.1%}; its reference score is {baseline:.1%}.',
             f'At the fixed {threshold:.0%} threshold, this is a ' +
             ('positive' if score >= threshold else 'negative') + ' screening prediction.']
    for direction in ['Raises model score', 'Lowers model score']:
        rows = table[table.Direction == direction].head(3)
        if not rows.empty:
            verb = 'raised' if direction.startswith('Raises') else 'lowered'
            lines.append('Inputs that ' + verb + ' the score relative to the reference: ' +
                         '; '.join(f"{r['Feature']} ({r['Entered / cleaned value']:g}): "
                                   f"{abs(r['SHAP change (percentage points)']):.1f} percentage points"
                                   for _, r in rows.iterrows()) + '.')
    lines += ['These contributions explain this model output, not the medical cause of PCOS. '
              'A feature can act differently at another value or in another patient.',
              'This research score is not a calibrated individual disease risk or a diagnosis. '
              'A negative screen does not rule out PCOS; discuss symptoms and the result with a clinician.']
    return '\n\n'.join(lines)
