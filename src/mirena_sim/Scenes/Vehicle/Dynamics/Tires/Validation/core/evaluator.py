#!/usr/bin/env python3
"""
Tire Model Validation Framework — Model Evaluator Core
Loads and evaluates all three model paradigms:
  - Model 1: Baseline MF 6.2
  - Model 2: Optimized MF 6.2
  - Model 3: Hybrid MF 6.2 + Physics-Informed Residual PINN
Supports exact continuous Fz, load/slip/airborne physics gates, and multi-compound support.
"""

import os
import sys
import json
import numpy as np

# Ensure Training module is accessible
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TRAINING_DIR = os.path.join(BASE_DIR, "Training")
if TRAINING_DIR not in sys.path:
    sys.path.append(TRAINING_DIR)

from calibrate_mf62 import Pacejka62Fitter

MF62_DIR = os.path.join(BASE_DIR, "MF62")
NEURAL_DIR = os.path.join(BASE_DIR, "NeuralCorrection")


class TireModelEvaluator:
    """
    Evaluates Baseline MF 6.2, Optimized MF 6.2, and Hybrid MF 6.2 + PINN
    with strict preservation of continuous measured conditions and physics gating.
    """
    def __init__(self, compound: str = "R25B"):
        self.compound = compound.upper()
        self.c_lower = self.compound.lower()

        # 1. Load Model 1 (Baseline)
        base_path = os.path.join(MF62_DIR, f"baseline_hoosier_16x7.5_10_{self.c_lower}_mf62.json")
        if not os.path.exists(base_path):
            base_path = os.path.join(MF62_DIR, "baseline_hoosier_16x7.5_10_r25b_mf62.json")
        with open(base_path) as fp:
            self.params_m1 = json.load(fp)

        # 2. Load Model 2 (Optimized)
        opt_path = os.path.join(MF62_DIR, f"hoosier_16x7.5_10_{self.c_lower}_mf62.json")
        if not os.path.exists(opt_path):
            opt_path = os.path.join(MF62_DIR, "hoosier_16x7.5_10_mf62.json")
        with open(opt_path) as fp:
            self.params_m2 = json.load(fp)

        # 3. Load Model 3 (PINN weights)
        weights_path = os.path.join(NEURAL_DIR, f"neural_residual_weights_{self.c_lower}.json")
        if not os.path.exists(weights_path):
            weights_path = os.path.join(NEURAL_DIR, "neural_residual_weights.json")
        with open(weights_path) as fp:
            self.nn_data = json.load(fp)

        self.r0_m1 = self.params_m1.get("UNLOADED_RADIUS", 0.200)
        self.r0_m2 = self.params_m2.get("UNLOADED_RADIUS", 0.200)
        self.fnomin = self.params_m2.get("FNOMIN", 800.0)
        self.pinom = self.params_m2.get("PINOM", 83.0)

    def eval_pinn(self, alpha, fz, gamma, p):
        """
        Vectorized evaluation of the 16x16 Tanh PINN with physics gates.
        """
        alpha, fz, gamma, p = np.broadcast_arrays(
            np.asarray(alpha, dtype=float),
            np.asarray(fz, dtype=float),
            np.asarray(gamma, dtype=float),
            np.asarray(p, dtype=float)
        )

        orig_shape = alpha.shape
        alpha_flat = alpha.reshape(-1)
        fz_flat = fz.reshape(-1)
        gamma_flat = gamma.reshape(-1)
        p_flat = p.reshape(-1)

        norm = self.nn_data["normalization"]
        weights = self.nn_data["weights"]
        gates = self.nn_data.get("physics_gates", {})

        x_mean = np.array(norm["x_mean"])
        x_std = np.array(norm["x_std"])
        y_mean = np.array(norm["y_mean"])
        y_std = np.array(norm["y_std"])

        X_in = np.column_stack([alpha_flat, fz_flat, gamma_flat, p_flat])
        X_norm = (X_in - x_mean) / x_std

        w1, b1 = np.array(weights["w1"]), np.array(weights["b1"])
        w2, b2 = np.array(weights["w2"]), np.array(weights["b2"])
        w3, b3 = np.array(weights["w3"]), np.array(weights["b3"])

        h1 = np.tanh(X_norm @ w1.T + b1)
        h2 = np.tanh(h1 @ w2.T + b2)
        out_raw = h2 @ w3.T + b3

        raw_fy = out_raw[:, 0] * y_std[0] + y_mean[0]
        raw_mz = out_raw[:, 1] * y_std[1] + y_mean[1]

        # Physics gates (schema v2)
        slip_scale = gates.get("slip_scale_rad", 0.006)
        load_trans = gates.get("load_trans_n", 100.0)
        alpha_limit = gates.get("alpha_limit_rad", 0.22)
        alpha_slope = gates.get("alpha_slope_rad", 0.02)
        fz_limit = gates.get("fz_limit_n", 1300.0)
        fz_slope = gates.get("fz_slope_n", 100.0)
        clip_fy = gates.get("clip_fy_n", 200.0)
        clip_mz = gates.get("clip_mz_nm", 10.0)

        g_slip = 1.0 - np.exp(-(alpha_flat ** 2 + gamma_flat ** 2) / (2.0 * slip_scale ** 2))
        t = np.clip(fz_flat / load_trans, 0.0, 1.0)
        g_load = t * t * (3.0 - 2.0 * t)
        a_abs = np.sqrt(alpha_flat ** 2 + 1e-6)
        g_alpha = 1.0 / (1.0 + np.exp((a_abs - alpha_limit) / alpha_slope))
        g_fz = 1.0 / (1.0 + np.exp((fz_flat - fz_limit) / fz_slope))
        scale = g_slip * g_load * g_alpha * g_fz

        delta_fy = clip_fy * np.tanh(raw_fy * scale / clip_fy)
        delta_mz = clip_mz * np.tanh(raw_mz * scale / clip_mz)

        return delta_fy.reshape(orig_shape), delta_mz.reshape(orig_shape)

    def eval_pure_lateral(self, alpha, fz, gamma=0.0, p=83.0, fx=0.0):
        """
        Evaluates pure lateral response for all three models.
        """
        alpha, fz, gamma, p, fx = np.broadcast_arrays(
            np.asarray(alpha, dtype=float),
            np.asarray(fz, dtype=float),
            np.asarray(gamma, dtype=float),
            np.asarray(p, dtype=float),
            np.asarray(fx, dtype=float)
        )

        # Model 1 (Baseline)
        fy_m1, svy_m1 = Pacejka62Fitter.compute_fy0(self.params_m1, alpha, fz, gamma, p, fnomin=self.fnomin, pinom=self.pinom)
        mz_m1 = Pacejka62Fitter.compute_mz(self.params_m1, alpha, fz, gamma, p, fx=fx, fy_in=fy_m1, svy_in=svy_m1, fnomin=self.fnomin, pinom=self.pinom, r0=self.r0_m1)
        mx_m1 = Pacejka62Fitter.compute_mx(self.params_m1, fz, gamma, fy_m1, p, fnomin=self.fnomin, pinom=self.pinom, r0=self.r0_m1)

        # Model 2 (Optimized)
        fy_m2, svy_m2 = Pacejka62Fitter.compute_fy0(self.params_m2, alpha, fz, gamma, p, fnomin=self.fnomin, pinom=self.pinom)
        mz_m2 = Pacejka62Fitter.compute_mz(self.params_m2, alpha, fz, gamma, p, fx=fx, fy_in=fy_m2, svy_in=svy_m2, fnomin=self.fnomin, pinom=self.pinom, r0=self.r0_m2)
        mx_m2 = Pacejka62Fitter.compute_mx(self.params_m2, fz, gamma, fy_m2, p, fnomin=self.fnomin, pinom=self.pinom, r0=self.r0_m2)

        # Model 3 (Hybrid)
        delta_fy, delta_mz = self.eval_pinn(alpha, fz, gamma, p)
        fy_m3 = fy_m2 + delta_fy
        mz_m3 = mz_m2 + delta_mz
        mx_m3 = mx_m2

        return {
            "fy_m1": fy_m1, "mz_m1": mz_m1, "mx_m1": mx_m1, "svy_m1": svy_m1,
            "fy_m2": fy_m2, "mz_m2": mz_m2, "mx_m2": mx_m2, "svy_m2": svy_m2,
            "fy_m3": fy_m3, "mz_m3": mz_m3, "mx_m3": mx_m3,
            "delta_fy": delta_fy, "delta_mz": delta_mz
        }

    def eval_combined(self, kappa, alpha, fz, gamma=0.0, p=83.0):
        """
        Evaluates combined slip forces and moments for Model 2 (MF 6.2) and Model 3 (Hybrid).
        Strictly matches Godot TireModel.gd implementation down to machine precision.
        """
        kappa, alpha, fz, gamma, p = np.broadcast_arrays(
            np.asarray(kappa, dtype=float),
            np.asarray(alpha, dtype=float),
            np.asarray(fz, dtype=float),
            np.asarray(gamma, dtype=float),
            np.asarray(p, dtype=float)
        )

        orig_shape = kappa.shape
        k_flat = kappa.reshape(-1)
        a_flat = alpha.reshape(-1)
        fz_flat = fz.reshape(-1)
        g_flat = gamma.reshape(-1)
        p_flat = p.reshape(-1)

        comb = Pacejka62Fitter.compute_combined_forces(
            self.params_m2, k_flat, a_flat, fz_flat, g_flat, p_flat,
            fnomin=self.fnomin, pinom=self.pinom
        )
        fx_m2 = comb["Fx"]
        fy_m2 = comb["Fy"]
        mz_m2 = Pacejka62Fitter.compute_mz(
            self.params_m2, a_flat, fz_flat, g_flat, p_flat, kappa=k_flat,
            comb=comb, fnomin=self.fnomin, pinom=self.pinom, r0=self.r0_m2
        )
        mx_m2 = Pacejka62Fitter.compute_mx(
            self.params_m2, fz_flat, g_flat, fy_m2, p_flat,
            fnomin=self.fnomin, pinom=self.pinom, r0=self.r0_m2
        )

        delta_fy, delta_mz = self.eval_pinn(a_flat, fz_flat, g_flat, p_flat)
        gyk = comb["Gyk"]
        fy_m3 = fy_m2 + gyk * delta_fy
        mz_m3 = mz_m2 + gyk * delta_mz
        fx_m3 = fx_m2
        mx_m3 = mx_m2

        return {
            "fx_m2": fx_m2.reshape(orig_shape),
            "fy_m2": fy_m2.reshape(orig_shape),
            "mz_m2": mz_m2.reshape(orig_shape),
            "mx_m2": mx_m2.reshape(orig_shape),
            "fx_m3": fx_m3.reshape(orig_shape),
            "fy_m3": fy_m3.reshape(orig_shape),
            "mz_m3": mz_m3.reshape(orig_shape),
            "mx_m3": mx_m3.reshape(orig_shape),
            "delta_fy": delta_fy.reshape(orig_shape),
            "delta_mz": delta_mz.reshape(orig_shape)
        }

    def eval_dataset(self, df):
        """
        Evaluates models across all rows of an experimental dataset dataframe.
        Preserves actual measured f_z, alpha, gamma, p_kpa.
        """
        alpha = df["alpha"].values
        fz = df["f_z"].values
        gamma = df["gamma"].values
        p = df["p_kpa"].values
        fx = df["f_x"].values if "f_x" in df.columns else np.zeros_like(fz)
        fy_exp = df["f_y"].values if "f_y" in df.columns else np.zeros_like(fz)
        mz_exp = df["m_z"].values if "m_z" in df.columns else np.zeros_like(fz)
        mx_exp = df["m_x"].values if "m_x" in df.columns else np.zeros_like(fz)

        preds = self.eval_pure_lateral(alpha, fz, gamma, p, fx)
        res = {
            "alpha": alpha, "f_z": fz, "gamma": gamma, "p_kpa": p, "f_x": fx,
            "fy_exp": fy_exp, "mz_exp": mz_exp, "mx_exp": mx_exp
        }
        res.update(preds)
        return res
