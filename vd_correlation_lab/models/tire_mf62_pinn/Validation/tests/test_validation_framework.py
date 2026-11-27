#!/usr/bin/env python3
"""
Unit Tests for Mirena Dynamics Tire Model Validation Framework
Tests statistical metrics, feature extractions, model evaluator physics gates,
in-plot metrics box annotations, and missing-channel error handling.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

VAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if VAL_DIR not in sys.path:
    sys.path.append(VAL_DIR)

from core.metrics import compute_metrics, decompose_error, bootstrap_confidence_interval
from core.features import extract_lateral_features, extract_aligning_features, extract_vertical_features
from core.evaluator import TireModelEvaluator
from core.style import add_metrics_box, add_characterization_badge
import matplotlib.pyplot as plt


class TestValidationFramework(unittest.TestCase):
    def test_compute_metrics_identity(self):
        y_true = np.array([100.0, 200.0, 300.0, 400.0])
        y_pred = np.array([100.0, 200.0, 300.0, 400.0])
        m = compute_metrics(y_true, y_pred)
        self.assertAlmostEqual(m["rmse"], 0.0)
        self.assertAlmostEqual(m["mae"], 0.0)
        self.assertAlmostEqual(m["bias"], 0.0)
        self.assertAlmostEqual(m["r2"], 1.0)
        self.assertEqual(m["n"], 4)

    def test_compute_metrics_empty(self):
        m = compute_metrics([], [])
        self.assertEqual(m["rmse"], 0.0)
        self.assertEqual(m["n"], 0)

    def test_error_decomposition(self):
        # Mathematical identity: MSE = Bias^2 + Variance
        y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        y_pred = y_true + 5.0 + np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        decomp = decompose_error(y_true, y_pred)
        self.assertAlmostEqual(decomp["mse"], decomp["bias_sq"] + decomp["variance"], places=6)
        self.assertAlmostEqual(decomp["pct_bias"] + decomp["pct_variance"], 100.0, places=6)

    def test_bootstrap_ci(self):
        rng = np.random.RandomState(42)
        y_true = rng.uniform(500, 2000, size=100)
        y_pred = y_true + rng.normal(0, 50, size=100)
        ci = bootstrap_confidence_interval(y_true, y_pred, n_boot=50)
        self.assertTrue(ci["rmse_ci_lo"] <= ci["rmse_mean"] <= ci["rmse_ci_hi"])
        self.assertTrue(ci["r2_ci_lo"] <= ci["r2_mean"] <= ci["r2_ci_hi"])

    def test_feature_extraction(self):
        a = np.linspace(-0.20, 0.20, 101)
        fy = 2000.0 * np.sin(10.0 * a) # synthetic curve
        feat = extract_lateral_features(a, fy)
        self.assertGreater(feat["stiffness_n_rad"], 10000.0)
        self.assertAlmostEqual(feat["fy_max_n"], 2000.0, delta=50.0)
        self.assertAlmostEqual(feat["fy_zero_n"], 0.0, delta=1.0)

    def test_pinn_physics_gates(self):
        evaluator = TireModelEvaluator("R25B")
        
        # 1. Zero-slip passivity: at alpha = 0, gamma = 0, delta Fy must be near 0
        dfy_zero, dmz_zero = evaluator.eval_pinn([0.0], [800.0], [0.0], [83.0])
        self.assertLess(abs(dfy_zero[0]), 1e-3)
        self.assertLess(abs(dmz_zero[0]), 1e-4)

        # 2. Airborne suppression: at Fz = 0, delta Fy must be 0
        dfy_air, dmz_air = evaluator.eval_pinn([0.10], [0.0], [0.0], [83.0])
        self.assertAlmostEqual(dfy_air[0], 0.0)
        self.assertAlmostEqual(dmz_air[0], 0.0)

        # 3. Directional consistency & clamps
        dfy_pos, _ = evaluator.eval_pinn([0.08], [800.0], [0.0], [83.0])
        dfy_neg, _ = evaluator.eval_pinn([-0.08], [800.0], [0.0], [83.0])
        self.assertLess(abs(dfy_pos[0]), 200.0)
        self.assertLess(abs(dfy_neg[0]), 200.0)

        # 4. Hard clamp boundary: delta Fy must never exceed +/- 200 N
        dfy_huge, dmz_huge = evaluator.eval_pinn([0.15], [1100.0], [0.05], [83.0])
        self.assertLessEqual(abs(dfy_huge[0]), 200.0)
        self.assertLessEqual(abs(dmz_huge[0]), 10.0)

    def test_evaluator_missing_channels(self):
        # Ensure a dataframe lacking optional channels (f_x, m_x) evaluates safely
        df_sparse = pd.DataFrame({
            "alpha": [0.0, 0.05, 0.10],
            "f_z": [500.0, 500.0, 500.0],
            "gamma": [0.0, 0.0, 0.0],
            "p_kpa": [83.0, 83.0, 83.0],
            "f_y": [0.0, 500.0, 900.0],
            "m_z": [0.0, 10.0, 15.0]
        })
        evaluator = TireModelEvaluator("R25B")
        res = evaluator.eval_dataset(df_sparse)
        self.assertIn("fy_m3", res)
        self.assertEqual(len(res["fy_m3"]), 3)

    def test_metrics_box_annotation(self):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        box_data = {
            "Baseline": {"rmse": 180.0, "mae": 130.0, "bias": -10.0, "r2": 0.98, "n": 52294},
            "Optimized": {"rmse": 120.0, "mae": 90.0, "bias": -30.0, "r2": 0.99, "n": 52294}
        }
        add_metrics_box(ax, box_data, loc="upper left")
        add_characterization_badge(ax, "TEST BADGE")
        plt.close(fig)

    def test_sample_count_reconciliation(self):
        r25b_val_path = "data/ttc_tire/Hoosier_16x7.5-10/Processed/R25B_val.csv"
        r20_val_path = "data/ttc_tire/Hoosier_16x7.5-10/Processed/R20_val.csv"
        df25 = pd.read_csv(r25b_val_path)
        df20 = pd.read_csv(r20_val_path)
        self.assertEqual(len(df25), 52294, "R25B validation retained rows must be exactly 52,294.")
        self.assertEqual(len(df20), 52154, "R20 validation retained rows must be exactly 52,154.")


if __name__ == "__main__":
    unittest.main()
