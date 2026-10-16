"""Integration checks against a trusted local export, using synthetic inputs only."""
import argparse
import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
from model_service import ArtifactStore
from pcos_support import probability_function
from server import Handler

EXAMPLES = {'Age (yrs)': 27, 'BMI': 23.4, 'Waist:Hip Ratio': 0.8, 'Cycle(R/I)': 2,
            'Cycle length(days)': 5, 'Vit D3 (ng/mL)': 30, 'FSH(mIU/mL)': 6,
            'LH(mIU/mL)': 5, 'TSH (mIU/L)': 2, 'AMH(ng/mL)': 3,
            'PRL(ng/mL)': 12, 'PRG(ng/mL)': 1, 'Follicle No. (L)': 6,
            'Follicle No. (R)': 7, 'Avg. F size (L) (mm)': 8,
            'Avg. F size (R) (mm)': 8, 'Endometrium (mm)': 8}


def example(store, stage):
    return {'stage': stage, 'fingerprint': store.fingerprint,
            'values': {f: EXAMPLES.get(f.strip(), 0) for f in store.bundles[stage]['features']}}


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = ArtifactStore(ARTIFACTS)
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.server.store = cls.store
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f'http://127.0.0.1:{cls.server.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def post(self, route, payload, origin=None):
        headers = {'Content-Type': 'application/json'}
        if origin:
            headers['Origin'] = origin
        request = Request(self.base + '/api/' + route, json.dumps(payload).encode(), headers)
        with urlopen(request) as response:
            return json.load(response)

    def test_schema_matches_models_without_patient_reference_rows(self):
        with urlopen(self.base + '/api/schema') as response:
            schema = json.load(response)
        self.assertEqual(len(schema['stages']), 4)
        self.assertEqual(sorted(s['n_features'] for s in schema['stages']), [12, 13, 19, 24])
        self.assertNotIn('background', json.dumps(schema))
        bleeding = next(f for f in schema['stages'][0]['fields'] if f['name'] == 'Cycle length(days)')
        self.assertEqual(bleeding['label'], 'Menstrual bleeding duration (days)')

    def test_all_groups_match_direct_pipeline_inference(self):
        for stage in self.store.bundles:
            payload = example(self.store, stage)
            response = self.post('predict', payload)
            _, bundle, clean, _ = self.store.patient(payload)
            expected = probability_function(bundle['model'], bundle['features'])(clean)[0]
            self.assertAlmostEqual(response['score'], expected, places=12)
            self.assertEqual(response['screening_class'], 'Positive screen' if expected >= response['threshold'] else 'Negative screen')

    def test_shap_additivity_for_synthetic_patient(self):
        payload = example(self.store, self.store.stages[0]['id'])
        prediction = self.post('predict', payload)
        explanation = self.post('explain', payload)
        reconstructed = explanation['base_score'] + sum(c['contribution_pp'] for c in explanation['contributions']) / 100
        self.assertTrue(np.isclose(reconstructed, prediction['score'], atol=1e-6))

    def test_incomplete_nonfinite_and_stale_requests_are_rejected(self):
        stage = self.store.stages[0]['id']
        for mode in ['missing', 'nan', 'stale', 'category']:
            payload = example(self.store, stage)
            if mode == 'missing':
                payload['values'].pop(next(iter(payload['values'])))
            elif mode == 'nan':
                payload['values']['BMI'] = float('nan')
            elif mode == 'category':
                payload['values']['Cycle(R/I)'] = 99
            else:
                payload['fingerprint'] = 'old'
            with self.assertRaises(HTTPError) as error:
                self.post('predict', payload)
            self.assertEqual(error.exception.code, 422)

    def test_remote_origin_is_rejected(self):
        with self.assertRaises(HTTPError) as error:
            self.post('predict', example(self.store, self.store.stages[0]['id']), 'https://example.com')
        self.assertEqual(error.exception.code, 403)

    def test_comparisons_return_real_export_rows(self):
        with urlopen(self.base + '/api/comparisons') as response:
            rows = json.load(response)
        self.assertTrue(rows)
        self.assertTrue(all('analysis' in r and 'feature_set' in r and 'model' in r for r in rows))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifacts', required=True)
    args = parser.parse_args()
    ARTIFACTS = args.artifacts
    unittest.main(argv=['test_service.py'])
