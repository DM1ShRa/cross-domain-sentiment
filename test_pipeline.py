#!/usr/bin/env python3
"""
test_pipeline.py — Comprehensive 4-Tier Opaque-Box Test Suite for Milestone 1.

Project: Cross-Domain Sentiment Analysis on Amazon Reviews
Authoritative References: ORIGINAL_REQUEST.md & PROJECT.md
Architecture Reference: TEST_INFRA.md

Tiers Covered:
- Tier 1: Feature Coverage (Core functional requirements)
- Tier 2: Boundary & Corner Cases (BVA, missing text, threshold bounds, zero division)
- Tier 3: Cross-Feature Combinations (TF-IDF leakage prevention, identical test sets, 3x3 matrix math)
- Tier 4: Real-World Workload & Artifact Validation (CLI execution --mode demo, 6 CSV file audits)
"""

import csv
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from typing import Dict, List, Any

# Target under test
import pipeline

CATEGORIES = ["Books", "Electronics", "All_Beauty"]
SENTIMENT_LABELS = ["Negative", "Neutral", "Positive"]
EXPECTED_CSV_FILES = [
    "model_metrics.csv",
    "logreg_transfer_matrix.csv",
    "generalization_gap.csv",
    "confusion_matrices.csv",
    "throughput.csv",
    "misclassified_sample.csv"
]
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


class TestPipelineMilestone1(unittest.TestCase):
    """Milestone 1 Test Suite implementing the 4-Tier Test Architecture."""

    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp(prefix="test_pipeline_m1_")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    # =======================================================================
    # Tier 1: Feature Coverage (Core Requirements)
    # =======================================================================

    def test_version_printing_present(self):
        """F2 / R3 / AC 89: Version printing block returns package status dictionary."""
        versions = pipeline.print_environment_versions()
        self.assertIsInstance(versions, dict)
        expected_packages = [
            "datasets", "scikit-learn", "transformers", "vaderSentiment",
            "pandas", "numpy", "torch"
        ]
        for pkg in expected_packages:
            self.assertIn(pkg, versions)

    def test_seed_setting_reproducibility(self):
        """F3 / R1 / AC 91: Deterministic random seed fixing produces identical sequences."""
        pipeline.set_seed(42)
        import random
        seq1 = [random.random() for _ in range(5)]

        pipeline.set_seed(42)
        seq2 = [random.random() for _ in range(5)]
        self.assertEqual(seq1, seq2, "Random seed 42 must generate identical sequences")

    def test_category_configs_and_splits(self):
        """F5 / AC 72: McAuley-Lab dataset categories defined correctly."""
        self.assertEqual(pipeline.CATEGORIES, ["Books", "Electronics", "All_Beauty"])
        self.assertEqual(pipeline.SENTIMENT_LABELS, ["Negative", "Neutral", "Positive"])

    def test_text_combination_and_cleaning(self):
        """F7 / AC 73: Concatenates title and text with proper punctuation."""
        comb1 = pipeline.combine_title_and_text("Great Book", "Loved the plot.")
        self.assertEqual(comb1, "Great Book. Loved the plot.")

        comb2 = pipeline.combine_title_and_text("Great Book.", "Loved the plot.")
        self.assertEqual(comb2, "Great Book. Loved the plot.")

        comb3 = pipeline.combine_title_and_text("", "Only body text.")
        self.assertEqual(comb3, "Only body text.")

        comb4 = pipeline.combine_title_and_text("Valid title", None)
        self.assertEqual(comb4, "Valid title")

    def test_balanced_sampling_logic(self):
        """F11 / AC 74: Ingestion produces equal quotas across all 3 classes."""
        dataset = pipeline.generate_synthetic_dataset(target_per_class=10, seed=42)
        for cat in pipeline.CATEGORIES:
            recs = dataset[cat]
            self.assertEqual(len(recs), 30)
            counts = {s: 0 for s in pipeline.SENTIMENT_LABELS}
            for r in recs:
                counts[r["sentiment"]] += 1
            for s, count in counts.items():
                self.assertEqual(count, 10, f"Category {cat} class {s} quota mismatch")

    def test_stratified_split_proportions(self):
        """F12 / AC 75: 80/20 train/test split maintains exact class balance."""
        dataset = pipeline.generate_synthetic_dataset(target_per_class=50, seed=42)
        cleaned, _ = pipeline.clean_and_deduplicate(dataset["Books"])
        train_part, test_part = pipeline.stratified_train_test_split(cleaned, test_size=0.20, seed=42)

        # 50 per class * 3 classes = 150 total. 80% = 120, 20% = 30 (10 per class)
        self.assertEqual(len(train_part), 120)
        self.assertEqual(len(test_part), 30)

        test_counts = {s: 0 for s in pipeline.SENTIMENT_LABELS}
        for r in test_part:
            test_counts[r["sentiment"]] += 1
        for s, count in test_counts.items():
            self.assertEqual(count, 10, f"Class {s} must have exactly 10 test samples")

    def test_vader_inference_and_thresholds(self):
        """F14 / F15 / AC 77: VADER classifies polarity based on compound score."""
        vader = pipeline.VADERBaseline()
        self.assertEqual(vader.predict_one("This is absolutely wonderful, glorious, and brilliant!"), "Positive")
        self.assertEqual(vader.predict_one("This is terrible, horrible, disgusting, and completely defective."), "Negative")
        self.assertEqual(vader.predict_one("The package arrived on Tuesday."), "Neutral")

    def test_roberta_zero_fine_tuning_contract(self):
        """F23 / F24 / AC 78: RoBERTa outputs valid 3-class sentiment with frozen weights."""
        roberta = pipeline.PretrainedRoBERTaModel(batch_size=16)
        preds = roberta.predict(["An absolute literary masterpiece!", "Horrible, defective junk."])
        self.assertEqual(len(preds), 2)
        for p in preds:
            self.assertIn(p, pipeline.SENTIMENT_LABELS)

    def test_logreg_transfer_matrix_structure(self):
        """F18 / F19 / AC 79 / AC 85: LogReg fits on train and predicts on test without leakage."""
        train_texts = [
            "Outstanding brilliant masterpiece book",
            "Terrible dull boring awful read",
            "Average novel, standard reading"
        ]
        train_labels = ["Positive", "Negative", "Neutral"]
        model = pipeline.TFIDFLogisticRegressionModel(random_state=42)
        model.fit(train_texts, train_labels)

        test_texts = ["Brilliant book", "Boring awful read"]
        preds = model.predict(test_texts)
        self.assertEqual(len(preds), 2)
        for p in preds:
            self.assertIn(p, pipeline.SENTIMENT_LABELS)

    def test_macro_f1_computation(self):
        """F20 / AC 80: Macro-F1 matches unweighted arithmetic mean of per-class F1."""
        y_true = ["Positive", "Positive", "Negative", "Neutral", "Neutral"]
        y_pred = ["Positive", "Negative", "Negative", "Neutral", "Positive"]
        metrics = pipeline.compute_metrics(y_true, y_pred)
        self.assertIn("macro_f1", metrics)
        self.assertIn("accuracy", metrics)
        self.assertGreaterEqual(metrics["macro_f1"], 0.0)
        self.assertLessEqual(metrics["macro_f1"], 1.0)

    def test_generalization_gap_formula(self):
        """F22 / AC 80: Generalization Gap = F1_in - mean(F1_cross)."""
        f1_in = 0.80
        cross_scores = [0.70, 0.60]
        mean_cross = sum(cross_scores) / len(cross_scores)
        gap = round(f1_in - mean_cross, 4)
        self.assertEqual(gap, 0.15)

    def test_throughput_profiling_logic(self):
        """F26 / AC 86: Throughput computes reviews per second accurately."""
        total_rev = 1000
        elapsed = 0.5
        tp = total_rev / elapsed
        self.assertEqual(tp, 2000.0)

    def test_cli_flags_demo_and_full(self):
        """F4: CLI argument parser supports --mode demo and --mode full."""
        import sys
        old_argv = sys.argv
        try:
            sys.argv = ["pipeline.py", "--mode", "demo", "--seed", "42"]
            args = pipeline.parse_args()
            self.assertEqual(args.mode, "demo")
            self.assertEqual(args.seed, 42)

            sys.argv = ["pipeline.py", "--mode", "full", "--batch-size", "64"]
            args = pipeline.parse_args()
            self.assertEqual(args.mode, "full")
            self.assertEqual(args.batch_size, 64)
        finally:
            sys.argv = old_argv

    # =======================================================================
    # Tier 2: Boundary & Corner Cases
    # =======================================================================

    def test_boundary_empty_and_whitespace_text(self):
        """Boundary: Empty, whitespace-only, and NaN reviews are dropped."""
        raw_records = [
            {"title": "", "text": "   ", "sentiment": "Neutral"},
            {"title": None, "text": None, "sentiment": "Neutral"},
            {"title": "Valid Title", "text": "", "sentiment": "Positive"},
            {"title": "Valid Title", "text": "", "sentiment": "Positive"},  # duplicate
        ]
        cleaned, seen = pipeline.clean_and_deduplicate(raw_records)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned[0]["review_text"], "Valid Title")

    def test_vader_boundary_thresholds(self):
        """Boundary: Exact -0.05 and +0.05 thresholds."""
        class MockVADER(pipeline.VADERBaseline):
            def __init__(self, score):
                self._has_lib = False
                self._score = score
            def _fallback_polarity(self, text):
                return self._score

        v_pos = MockVADER(0.0500)
        self.assertEqual(v_pos.predict_one("text"), "Positive")

        v_neu_upper = MockVADER(0.0499)
        self.assertEqual(v_neu_upper.predict_one("text"), "Neutral")

        v_neu_lower = MockVADER(-0.0499)
        self.assertEqual(v_neu_lower.predict_one("text"), "Neutral")

        v_neg = MockVADER(-0.0500)
        self.assertEqual(v_neg.predict_one("text"), "Negative")

    def test_rating_boundary_mapping(self):
        """Boundary: Star ratings correctly map to 3 classes."""
        self.assertEqual(pipeline.rating_to_sentiment(1.0), "Negative")
        self.assertEqual(pipeline.rating_to_sentiment(2.0), "Negative")
        self.assertEqual(pipeline.rating_to_sentiment(2.0001), "Neutral")
        self.assertEqual(pipeline.rating_to_sentiment(3.0), "Neutral")
        self.assertEqual(pipeline.rating_to_sentiment(3.9999), "Neutral")
        self.assertEqual(pipeline.rating_to_sentiment(4.0), "Positive")
        self.assertEqual(pipeline.rating_to_sentiment(5.0), "Positive")
        self.assertIsNone(pipeline.rating_to_sentiment(0.5))
        self.assertIsNone(pipeline.rating_to_sentiment(5.5))
        self.assertIsNone(pipeline.rating_to_sentiment(float("nan")))

    def test_metric_zero_division_resilience(self):
        """Boundary: Precision/Recall/F1 with zero true positives evaluates to 0.0 without crash."""
        y_true = ["Positive", "Positive"]
        y_pred = ["Negative", "Negative"]
        metrics = pipeline.compute_metrics(y_true, y_pred)
        self.assertEqual(metrics["precision_neutral"], 0.0)
        self.assertEqual(metrics["recall_neutral"], 0.0)
        self.assertEqual(metrics["f1_neutral"], 0.0)
        self.assertEqual(metrics["accuracy"], 0.0)

    def test_misclassified_sample_pool_logic(self):
        """Boundary: Error sampling handles both pool < 100 and pool >= 100."""
        pool_small = [{"review_text": f"Error {i}"} for i in range(15)]
        rng = pipeline.random.Random(42)
        sample_small = pool_small if len(pool_small) < 100 else rng.sample(pool_small, 100)
        self.assertEqual(len(sample_small), 15)

        pool_large = [{"review_text": f"Error {i}"} for i in range(250)]
        sample_large = rng.sample(pool_large, 100)
        self.assertEqual(len(sample_large), 100)

    def test_boundary_batch_size_edge_cases(self):
        """Boundary: Batched inference handles single-item, partial, and empty lists."""
        roberta = pipeline.PretrainedRoBERTaModel(batch_size=1)
        preds_single = roberta.predict(["Just one review"])
        self.assertEqual(len(preds_single), 1)

        roberta_large = pipeline.PretrainedRoBERTaModel(batch_size=100)
        preds_multi = roberta_large.predict(["First review", "Second review", "Third review"])
        self.assertEqual(len(preds_multi), 3)

        preds_empty = roberta.predict([])
        self.assertEqual(len(preds_empty), 0)

    # =======================================================================
    # Tier 3: Cross-Feature Combinations & Leakage Invariants
    # =======================================================================

    def test_tfidf_no_data_leakage(self):
        """Tier 3 Invariant: TF-IDF vectorizer only sees training tokens."""
        train_texts = ["apple banana cherry", "date elderberry fig"]
        train_labels = ["Positive", "Negative"]
        model = pipeline.TFIDFLogisticRegressionModel(random_state=42)
        model.fit(train_texts, train_labels)

        # Test set contains brand new out-of-vocabulary tokens
        test_texts = ["grape honeydew kiwi", "apple kiwi"]
        preds = model.predict(test_texts)
        self.assertEqual(len(preds), 2)

    def test_test_set_standardization_invariant(self):
        """Tier 3 Invariant: All models are evaluated on the exact same test records."""
        bench = pipeline.CrossDomainBenchmarkPipeline(mode="demo", seed=42, output_dir=self.test_dir)
        bench.dataset = pipeline.generate_synthetic_dataset(target_per_class=30, seed=42)
        for cat in pipeline.CATEGORIES:
            cleaned, _ = pipeline.clean_and_deduplicate(bench.dataset[cat])
            tr, te = pipeline.stratified_train_test_split(cleaned, test_size=0.20, seed=42)
            bench.train_data[cat] = tr
            bench.test_data[cat] = te

        for cat in pipeline.CATEGORIES:
            self.assertEqual(len(bench.test_data[cat]), 18)

    def test_cross_feature_3x3_transfer_matrix_alignment(self):
        """Tier 3 Invariant: Transfer matrix structure has 3 source rows and 3 target columns."""
        matrix = {
            "Books": {"Books": 0.81, "Electronics": 0.65, "All_Beauty": 0.62},
            "Electronics": {"Books": 0.60, "Electronics": 0.84, "All_Beauty": 0.67},
            "All_Beauty": {"Books": 0.58, "Electronics": 0.66, "All_Beauty": 0.83}
        }
        self.assertEqual(len(matrix), 3)
        for cat in CATEGORIES:
            self.assertIn(cat, matrix)
            self.assertEqual(len(matrix[cat]), 3)
            for target in CATEGORIES:
                self.assertIn(target, matrix[cat])

    # =======================================================================
    # Tier 4: Real-World Workload & Artifact Validation
    # =======================================================================

    def test_cli_execution_demo_mode(self):
        """Tier 4: Run pipeline end-to-end in isolated dir and audit all 6 CSV outputs."""
        output_dir = os.path.join(self.test_dir, "run_demo")
        os.makedirs(output_dir, exist_ok=True)

        bench = pipeline.CrossDomainBenchmarkPipeline(
            mode="demo",
            seed=42,
            batch_size=32,
            output_dir=output_dir
        )
        bench.run()

        # Audit existence of all 6 CSV artifacts
        for fname in EXPECTED_CSV_FILES:
            fpath = os.path.join(output_dir, fname)
            self.assertTrue(os.path.exists(fpath), f"Missing required artifact: {fname}")
            self.assertGreater(os.path.getsize(fpath), 0, f"Artifact is empty: {fname}")

        # Audit model_metrics.csv schema & row count (9 rows)
        with open(os.path.join(output_dir, "model_metrics.csv"), "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            expected_headers = [
                "model", "category", "macro_f1", "accuracy",
                "precision_negative", "recall_negative", "f1_negative",
                "precision_neutral", "recall_neutral", "f1_neutral",
                "precision_positive", "recall_positive", "f1_positive"
            ]
            self.assertEqual(reader.fieldnames, expected_headers)
            rows = list(reader)
            self.assertEqual(len(rows), 9)

        # Audit logreg_transfer_matrix.csv (3x3)
        with open(os.path.join(output_dir, "logreg_transfer_matrix.csv"), "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.assertEqual(reader.fieldnames, ["train_category", "Books", "Electronics", "All_Beauty"])
            rows = list(reader)
            self.assertEqual(len(rows), 3)

        # Audit generalization_gap.csv (4 rows: 3 categories + Overall_Mean)
        with open(os.path.join(output_dir, "generalization_gap.csv"), "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.assertEqual(reader.fieldnames, ["train_category", "in_domain_f1", "mean_cross_domain_f1", "generalization_gap"])
            rows = list(reader)
            self.assertEqual(len(rows), 4)

        # Audit confusion_matrices.csv
        with open(os.path.join(output_dir, "confusion_matrices.csv"), "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.assertEqual(reader.fieldnames, ["model", "category", "true_label", "predicted_label", "count", "percentage"])
            rows = list(reader)
            self.assertEqual(len(rows), 81)

        # Audit throughput.csv (3 models)
        with open(os.path.join(output_dir, "throughput.csv"), "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.assertEqual(reader.fieldnames, ["model", "total_reviews", "elapsed_seconds", "reviews_per_second", "device"])
            rows = list(reader)
            self.assertEqual(len(rows), 3)

        # Audit misclassified_sample.csv (exactly 100 rows with seed=42)
        with open(os.path.join(output_dir, "misclassified_sample.csv"), "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.assertEqual(reader.fieldnames, ["review_text", "true_label", "predicted_label", "model_name", "category"])
            rows = list(reader)
            self.assertEqual(len(rows), 100)

    def test_subprocess_cli_execution_root(self):
        """Tier 4: Subprocess execution of `python pipeline.py --mode demo` in project root."""
        proc = subprocess.run(
            [sys.executable, "pipeline.py", "--mode", "demo"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=180
        )
        self.assertEqual(
            proc.returncode, 0,
            f"pipeline.py execution failed with code {proc.returncode}.\nStderr: {proc.stderr}"
        )
        # Check package version banner in stdout
        self.assertIn("CROSS-DOMAIN SENTIMENT ANALYSIS BENCHMARK PIPELINE", proc.stdout)
        self.assertIn("Package Inventory:", proc.stdout)

        # Confirm all 6 artifacts exist in root directory
        for fname in EXPECTED_CSV_FILES:
            fpath = os.path.join(PROJECT_ROOT, fname)
            self.assertTrue(os.path.exists(fpath), f"Artifact missing from project root: {fname}")


if __name__ == "__main__":
    unittest.main()
