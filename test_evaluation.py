import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation import evaluate_detector, label_ground_truth, precision_recall_curve_points


class TestLabelGroundTruth(unittest.TestCase):
    def test_marks_rows_inside_the_cluster_window(self):
        rows = pd.DataFrame([
            {"ticker": "X", "date": "2024-01-10"},   # inside window
            {"ticker": "X", "date": "2024-02-01"},   # outside window
            {"ticker": "Y", "date": "2024-01-10"},   # different ticker, same date
        ])
        ground_truth = pd.DataFrame([
            {"ticker": "X", "cluster_start": "2024-01-08", "cluster_end": "2024-01-11", "jump_date": "2024-01-15"},
        ])
        labels = label_ground_truth(rows, ground_truth)
        self.assertTrue(labels.iloc[0])
        self.assertFalse(labels.iloc[1])
        self.assertFalse(labels.iloc[2])

    def test_uses_cluster_end_when_jump_date_is_missing(self):
        rows = pd.DataFrame([{"ticker": "X", "date": "2024-01-11"}])
        ground_truth = pd.DataFrame([
            {"ticker": "X", "cluster_start": "2024-01-08", "cluster_end": "2024-01-11", "jump_date": None},
        ])
        labels = label_ground_truth(rows, ground_truth)
        self.assertTrue(labels.iloc[0])

    def test_empty_ground_truth_labels_everything_false(self):
        rows = pd.DataFrame([{"ticker": "X", "date": "2024-01-10"}])
        empty_gt = pd.DataFrame(columns=["ticker", "cluster_start", "cluster_end", "jump_date"])
        labels = label_ground_truth(rows, empty_gt)
        self.assertFalse(labels.iloc[0])


class TestEvaluateDetector(unittest.TestCase):
    def test_perfect_detector_scores_precision_and_recall_of_one(self):
        rows = pd.DataFrame([
            {"ticker": "X", "date": "2024-01-10", "is_anomaly": True, "score": 1.0},
            {"ticker": "X", "date": "2024-01-11", "is_anomaly": True, "score": 1.0},
            {"ticker": "X", "date": "2024-02-01", "is_anomaly": False, "score": 0.0},
            {"ticker": "X", "date": "2024-02-02", "is_anomaly": False, "score": 0.0},
        ])
        ground_truth = pd.DataFrame([
            {"ticker": "X", "cluster_start": "2024-01-10", "cluster_end": "2024-01-11", "jump_date": "2024-01-11"},
        ])
        result = evaluate_detector(rows, ground_truth)
        self.assertAlmostEqual(result["precision"], 1.0)
        self.assertAlmostEqual(result["recall"], 1.0)
        self.assertAlmostEqual(result["f1"], 1.0)

    def test_detector_that_flags_nothing_true_scores_zero_precision(self):
        rows = pd.DataFrame([
            {"ticker": "X", "date": "2024-01-10", "is_anomaly": True, "score": 1.0},  # false positive
            {"ticker": "X", "date": "2024-02-01", "is_anomaly": False, "score": 0.0},  # the actual cluster day, missed
        ])
        ground_truth = pd.DataFrame([
            {"ticker": "X", "cluster_start": "2024-02-01", "cluster_end": "2024-02-01", "jump_date": "2024-02-01"},
        ])
        result = evaluate_detector(rows, ground_truth)
        self.assertAlmostEqual(result["precision"], 0.0)
        self.assertAlmostEqual(result["recall"], 0.0)

    def test_returns_none_when_no_ground_truth_positives_exist(self):
        rows = pd.DataFrame([{"ticker": "X", "date": "2024-01-10", "is_anomaly": True, "score": 1.0}])
        empty_gt = pd.DataFrame(columns=["ticker", "cluster_start", "cluster_end", "jump_date"])
        result = evaluate_detector(rows, empty_gt)
        self.assertIsNone(result["precision"])
        self.assertIsNone(result["recall"])

    def test_excludes_nan_scored_rows_from_evaluation(self):
        # Simulates isolation_forest_detect_walkforward's unscored
        # warm-up period -- those rows should not count as (correct)
        # negatives just because they were never given a score.
        rows = pd.DataFrame([
            {"ticker": "X", "date": "2024-01-01", "is_anomaly": False, "score": float("nan")},  # unscored
            {"ticker": "X", "date": "2024-01-10", "is_anomaly": True, "score": 1.0},
            {"ticker": "X", "date": "2024-02-01", "is_anomaly": False, "score": 0.0},
        ])
        ground_truth = pd.DataFrame([
            {"ticker": "X", "cluster_start": "2024-01-10", "cluster_end": "2024-01-10", "jump_date": "2024-01-10"},
        ])
        result = evaluate_detector(rows, ground_truth)
        self.assertEqual(result["n_scored_rows"], 2)  # not 3

    def test_empty_detections_returns_none_metrics_not_a_crash(self):
        empty = pd.DataFrame(columns=["ticker", "date", "is_anomaly", "score"])
        ground_truth = pd.DataFrame(columns=["ticker", "cluster_start", "cluster_end", "jump_date"])
        result = evaluate_detector(empty, ground_truth)
        self.assertIsNone(result["precision"])


class TestPrecisionRecallCurvePoints(unittest.TestCase):
    def test_returns_none_without_a_positive_class(self):
        rows = pd.DataFrame([{"ticker": "X", "date": "2024-01-10", "is_anomaly": True, "score": 1.0}])
        empty_gt = pd.DataFrame(columns=["ticker", "cluster_start", "cluster_end", "jump_date"])
        self.assertIsNone(precision_recall_curve_points(rows, empty_gt))

    def test_returns_arrays_with_a_positive_class_present(self):
        rows = pd.DataFrame([
            {"ticker": "X", "date": "2024-01-10", "is_anomaly": True, "score": 0.9},
            {"ticker": "X", "date": "2024-02-01", "is_anomaly": False, "score": 0.1},
        ])
        ground_truth = pd.DataFrame([
            {"ticker": "X", "cluster_start": "2024-01-10", "cluster_end": "2024-01-10", "jump_date": "2024-01-10"},
        ])
        result = precision_recall_curve_points(rows, ground_truth)
        self.assertIsNotNone(result)
        precision, recall = result
        self.assertGreater(len(precision), 0)
        self.assertEqual(len(precision), len(recall))


if __name__ == "__main__":
    unittest.main()
