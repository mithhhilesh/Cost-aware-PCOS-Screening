"""Tests of isolation, patient grouping, threshold selection and paired evaluation."""
import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from evaluate import (choose_threshold, make_pipeline, make_splits,
                                 summarize, evaluate, STAGES)


class EvaluationTests(unittest.TestCase):
    def test_threshold_meets_target_and_is_maximal_with_ties(self):
        y = np.array([1, 1, 1, 1, 0, 0])
        score = np.array([0.9, 0.8, 0.8, 0.1, 0.85, 0.2])
        threshold = choose_threshold(y, score, 0.75)
        self.assertEqual(threshold, 0.8)
        self.assertGreaterEqual(np.mean(score[y == 1] >= threshold), 0.75)
        self.assertLess(np.mean(score[y == 1] >= np.nextafter(threshold, 1)), 0.75)
        with self.assertRaises(ValueError):
            choose_threshold(y, score, 0)

    def test_imputation_learns_only_from_training(self):
        X = pd.DataFrame({'BMI': [20., 22., np.nan, 24.],
                          'Fast food (Y/N)': [0., 0., 1., np.nan]})
        pipeline = make_pipeline(list(X), LogisticRegression())
        pipeline.fit(X, [0, 1, 0, 1])
        prep = pipeline.named_steps['preprocess']
        median = prep.named_transformers_['numeric'].named_steps['impute'].statistics_.copy()
        mode = prep.named_transformers_['categorical'].statistics_.copy()
        pipeline.predict_proba(pd.DataFrame({'BMI': [999., np.nan], 'Fast food (Y/N)': [1., 1.]}))
        np.testing.assert_array_equal(median, [22.])
        np.testing.assert_array_equal(mode, [0.])
        np.testing.assert_array_equal(median, prep.named_transformers_['numeric'].named_steps['impute'].statistics_)

    def test_groups_never_cross_either_cv_level(self):
        groups = pd.Series(np.repeat(np.arange(30), 2))
        y = pd.Series(np.repeat(np.arange(30) % 2, 2))
        X = pd.DataFrame({'BMI': np.arange(60)})
        for train, test in make_splits(X, y, groups, 3, 42):
            self.assertFalse(set(groups.iloc[train]) & set(groups.iloc[test]))
            for a, b in make_splits(X.iloc[train], y.iloc[train], groups.iloc[train], 3, 43):
                self.assertFalse(set(groups.iloc[train].iloc[a]) & set(groups.iloc[train].iloc[b]))

    def test_paired_identical_strategies_have_zero_difference(self):
        rows = []
        for stage in ['A0_full', 'A4_low_only']:
            for i in range(20):
                y = i % 2
                rows.append({'analysis': 'Primary', 'stage': stage, 'row_position': i,
                             'y': y, 'score': 0.8 if y else 0.2, 'prediction': y})
        _, differences, _, _ = summarize(pd.DataFrame(rows), bootstrap=100)
        np.testing.assert_array_equal(differences['difference'], 0.)
        np.testing.assert_array_equal(differences.ci_low, 0.)
        np.testing.assert_array_equal(differences.ci_high, 0.)

    def test_nested_run_has_exactly_one_outer_prediction_per_record(self):
        rng = np.random.default_rng(7)
        X = pd.DataFrame({f: rng.normal(3, 0.2, 40) for f in STAGES['A0_full']})
        from evaluate import BINARY
        for f in BINARY:
            X[f] = rng.integers(0, 2, len(X))
        X['Cycle(R/I)'] = rng.choice([2, 4], len(X))
        y = pd.Series([0, 1] * 20)
        predictions, selection, _ = evaluate(X, y, 0.9, outer_folds=2, inner_folds=2,
                                             analyses=('Primary',))
        self.assertEqual(len(selection), 8)
        self.assertTrue((predictions.groupby(['analysis', 'stage', 'row_position']).size() == 1).all())
        for _, block in predictions.groupby('stage'):
            self.assertEqual(set(block.row_position), set(range(len(X))))
        self.assertTrue(predictions.groupby('row_position').outer_fold.nunique().eq(1).all())


if __name__ == '__main__':
    unittest.main()
