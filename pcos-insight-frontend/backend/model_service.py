"""Read-only bridge to trusted notebook exports. Never returns reference patient rows."""
from pathlib import Path
from importlib.metadata import version
import hashlib
import json
import threading

import joblib
import numpy as np
import pandas as pd
from pcos_support import prepare_patient, probability_function, explain_probability

LABELS = {'A0_full': 'Full feature set', 'A1_no_very_high': 'Symptoms + hormonal labs',
          'A2_no_very_high_high': 'Low-cost + vitamin D', 'A4_low_only': 'Low-cost only'}


def display_name(name):
    return {'Cycle length(days)': 'Menstrual bleeding duration (days)',
            'Age (yrs)': 'Age (years)', 'Cycle(R/I)': 'Cycle regularity',
            'Weight gain(Y/N)': 'Weight gain', 'hair growth(Y/N)': 'Excess hair growth',
            'Skin darkening (Y/N)': 'Skin darkening', 'Hair loss(Y/N)': 'Hair loss',
            'Pimples(Y/N)': 'Pimples', 'Fast food (Y/N)': 'Fast food intake',
            'Reg.Exercise(Y/N)': 'Regular exercise'}.get(name.strip(), name.strip())


class ArtifactStore:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        manifest_path = self.directory / 'manifest.json'
        if not manifest_path.is_file():
            raise ValueError(f'No manifest.json in {self.directory}. Pass the exact notebook artifacts folder.')
        self.manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        self.lock = threading.RLock()
        self.bundles = {}
        self.stages = []
        digest = hashlib.sha256(manifest_path.read_bytes())
        seen = set()
        for stage, group in self.manifest['groups'].items():
            if not stage.replace('_', '').isalnum():
                raise ValueError('Invalid stage name in local manifest.')
            key = (group['model'], tuple(group['features']))
            if key in seen:
                continue
            seen.add(key)
            path = self.directory / (stage + '.joblib')
            digest.update(path.read_bytes())
            bundle = joblib.load(path)  # Trusted local notebook export only.
            if list(bundle['features']) != group['features']:
                raise ValueError(f'Feature order differs between manifest and bundle: {stage}')
            threshold = float(bundle.get('threshold', self.manifest.get('threshold', 0.5)))
            if not 0 <= threshold <= 1:
                raise ValueError('Invalid exported threshold.')
            fields = [{**bundle['schema'][f], 'name': f, 'label': display_name(f)} for f in bundle['features']]
            self.bundles[stage] = bundle
            self.stages.append({'id': stage, 'title': LABELS.get(stage, stage), 'model': bundle['model_name'],
                                'n_features': len(fields), 'fields': fields, 'threshold': threshold,
                                'cv_metrics': group.get('cv_metrics', {})})
        self.stages.sort(key=lambda s: s['n_features'])
        self.fingerprint = digest.hexdigest()[:16]

    def schema(self):
        mismatch = []
        for package in ['numpy', 'pandas', 'scikit-learn', 'joblib', 'shap']:
            expected = self.manifest.get('versions', {}).get(package)
            actual = version(package)
            if expected and actual != expected:
                mismatch.append(f'{package}: export {expected}; running {actual}. Use the export environment for compatibility.')
        return {'stages': self.stages, 'evaluation_label': self.manifest.get('evaluation_label',
                'Exploratory exported notebook metrics; see original validation documentation.'),
                'selection_rule': self.manifest.get('selection_rule', ''),
                'versions': self.manifest.get('versions', {}), 'version_warnings': mismatch,
                'fingerprint': self.fingerprint,
                'scope': 'Exploratory notebook export. These are not the corrected nested-evaluation results. '
                         'Scores are not calibrated individual disease risks or diagnoses.'}

    def comparisons(self):
        path = self.directory / 'model_comparison.csv'
        if not path.is_file():
            return []
        frame = pd.read_csv(path)
        for metric in ['accuracy', 'recall', 'specificity', 'f1', 'roc_auc', 'balanced_accuracy', 'average_precision']:
            if metric + '_mean' not in frame and metric in frame:
                frame[metric + '_mean'] = frame[metric]
        return json.loads(frame.to_json(orient='records'))

    def patient(self, payload):
        if not isinstance(payload, dict) or payload.get('stage') not in self.bundles:
            raise ValueError('Choose a feature group supplied by the connected model.')
        if payload.get('fingerprint') != self.fingerprint:
            raise ValueError('Model export changed. Reload this page before screening again.')
        stage = payload['stage']
        bundle = self.bundles[stage]
        values = payload.get('values')
        if not isinstance(values, dict) or set(values) != set(bundle['features']):
            raise ValueError('Supply exactly the required inputs for the selected feature group.')
        for feature in bundle['features']:
            value = values[feature]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(f'Enter a finite numeric value for {display_name(feature)}.')
            if value < 0:
                raise ValueError(f'{display_name(feature)} cannot be negative.')
            options = bundle['schema'][feature].get('choices')
            if options is not None and value not in options:
                raise ValueError(f'Choose a listed option for {display_name(feature)}.')
        raw = pd.DataFrame([values], columns=bundle['features'], dtype=float)
        clean = prepare_patient(raw)
        invalid = [display_name(f) for f in clean if clean[f].isna().any()]
        if invalid:
            raise ValueError('Values rejected by exported cleaning rules: ' + ', '.join(invalid))
        warnings = [f'Outside the observed training range: {display_name(f)}.' for f in clean if not
                    bundle['schema'][f]['min_observed'] <= clean[f].iloc[0] <= bundle['schema'][f]['max_observed']]
        return stage, bundle, clean, warnings

    def predict(self, payload):
        stage, bundle, clean, warnings = self.patient(payload)
        with self.lock:
            score = float(probability_function(bundle['model'], bundle['features'])(clean)[0])
        threshold = next(s['threshold'] for s in self.stages if s['id'] == stage)
        return {'stage': stage, 'model': bundle['model_name'], 'score': score, 'threshold': threshold,
                'screening_class': 'Positive screen' if score >= threshold else 'Negative screen',
                'warnings': warnings, 'fingerprint': self.fingerprint}

    def explain(self, payload):
        _, bundle, clean, _ = self.patient(payload)
        with self.lock:
            exp = explain_probability(bundle['model'], bundle['background'], clean)[0]
        contributions = [{'feature': display_name(f), 'value': float(clean[f].iloc[0]),
                          'contribution_pp': float(v * 100)} for f, v in zip(bundle['features'], exp.values)]
        contributions.sort(key=lambda c: -abs(c['contribution_pp']))
        return {'base_score': float(exp.base_values), 'score': float(exp.base_values + np.sum(exp.values)),
                'contributions': contributions, 'fingerprint': self.fingerprint}
