#!/usr/bin/env python3
"""
Unit Tests for Tire Coordinate Systems, Signs, and Transformations
Mirena Dynamics - Senior Tire Dynamics Quality Assurance
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

# Repository path resolution
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(TEST_DIR, "../../.."))
PROCESSED_DIR = os.path.join(REPO_ROOT, "data/ttc_tire/Hoosier_16x7.5-10/Processed")


class TestTireSignsAndTransforms(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load sample processed datasets
        cls.r25b_tr = pd.read_csv(os.path.join(PROCESSED_DIR, "R25B_train.csv"))
        cls.r20_tr = pd.read_csv(os.path.join(PROCESSED_DIR, "R20_train.csv"))

    def test_normal_load_positive(self):
        """Verify normal load f_z is strictly positive (compressive) in target convention."""
        self.assertTrue(np.all(self.r25b_tr["f_z"] > 0.0), "f_z must be positive in R25B_train")
        self.assertTrue(np.all(self.r20_tr["f_z"] > 0.0), "f_z must be positive in R20_train")
        self.assertGreaterEqual(self.r25b_tr["f_z"].min(), 100.0, "Minimum f_z should be >= 100 N")

    def test_positive_alpha_produces_positive_fy(self):
        """Verify positive slip angle alpha produces positive lateral force fy in linear range."""
        for df, name in [(self.r25b_tr, "R25B"), (self.r20_tr, "R20")]:
            mask_pos_small = (df["sa_deg"] > 1.0) & (df["sa_deg"] < 4.0) & (df["ia_deg"].abs() < 0.2)
            mean_fy = df.loc[mask_pos_small, "f_y"].mean()
            self.assertGreater(mean_fy, 400.0, f"Positive alpha must yield positive Fy for {name}, got {mean_fy:.1f} N")

            mask_neg_small = (df["sa_deg"] < -1.0) & (df["sa_deg"] > -4.0) & (df["ia_deg"].abs() < 0.2)
            mean_fy_neg = df.loc[mask_neg_small, "f_y"].mean()
            self.assertLess(mean_fy_neg, -400.0, f"Negative alpha must yield negative Fy for {name}, got {mean_fy_neg:.1f} N")

    def test_aligning_torque_is_restoring(self):
        """Verify Mz is restoring (negative for positive alpha in ISO convention with +Z upward)."""
        for df, name in [(self.r25b_tr, "R25B"), (self.r20_tr, "R20")]:
            mask_pos_small = (df["sa_deg"] > 1.0) & (df["sa_deg"] < 4.0) & (df["ia_deg"].abs() < 0.2)
            mean_mz = df.loc[mask_pos_small, "m_z"].mean()
            self.assertLess(mean_mz, -15.0, f"Aligning torque Mz must be restoring (< 0) for alpha > 0 in {name}, got {mean_mz:.2f} Nm")

            mask_neg_small = (df["sa_deg"] < -1.0) & (df["sa_deg"] > -4.0) & (df["ia_deg"].abs() < 0.2)
            mean_mz_neg = df.loc[mask_neg_small, "m_z"].mean()
            self.assertGreater(mean_mz_neg, 15.0, f"Aligning torque Mz must be restoring (> 0) for alpha < 0 in {name}, got {mean_mz_neg:.2f} Nm")

    def test_camber_thrust_sign(self):
        """Verify positive inclination (tilt to left, gamma > 0) adds positive lateral force (camber thrust)."""
        for df, name in [(self.r25b_tr, "R25B"), (self.r20_tr, "R20")]:
            # In Calspan, IA was 0, 2, 4 deg (tilt to right, so gamma is -0, -2, -4 deg).
            # Tilting to right (gamma < 0) must create force to right (fy < 0).
            mask_camber_4 = (df["ia_deg"].round(1) == 4.0) & (df["sa_deg"].abs() < 0.3) & (df["f_z"].between(900, 1100))
            mask_camber_0 = (df["ia_deg"].round(1) == 0.0) & (df["sa_deg"].abs() < 0.3) & (df["f_z"].between(900, 1100))
            
            fy_c4 = df.loc[mask_camber_4, "f_y"].mean()
            fy_c0 = df.loc[mask_camber_0, "f_y"].mean()
            gamma_c4 = df.loc[mask_camber_4, "gamma"].mean()
            gamma_c0 = df.loc[mask_camber_0, "gamma"].mean()

            delta_fy = fy_c4 - fy_c0
            delta_gamma = gamma_c4 - gamma_c0

            camber_stiffness = delta_fy / delta_gamma
            self.assertGreater(camber_stiffness, 2000.0, f"Camber thrust stiffness must be positive for {name}, got {camber_stiffness:.1f} N/rad")

    def test_overturning_moment_sign(self):
        """Verify overturning moment Mx sign is consistent with wheel tilt."""
        for df, name in [(self.r25b_tr, "R25B"), (self.r20_tr, "R20")]:
            # When tilted to right (gamma < 0), upward normal force at right of centerline creates negative torque about +X.
            mask_camber_4 = (df["ia_deg"].round(1) == 4.0) & (df["sa_deg"].abs() < 0.3) & (df["f_z"].between(900, 1100))
            mean_mx = df.loc[mask_camber_4, "m_x"].mean()
            self.assertLess(mean_mx, -20.0, f"Overturning moment Mx must be negative for rightward tilt in {name}, got {mean_mx:.2f} Nm")

    def test_provenance_retention(self):
        """Verify provenance channels are present and monotonic."""
        for df, name in [(self.r25b_tr, "R25B"), (self.r20_tr, "R20")]:
            for col in ["source_file", "compound", "round", "run", "raw_row_idx", "sweep_id", "t"]:
                self.assertIn(col, df.columns, f"Missing provenance column {col} in {name}")
            self.assertTrue(np.all(np.diff(df["t"]) >= 0.0), f"Timestamps must be non-decreasing in {name}")


if __name__ == "__main__":
    unittest.main()
