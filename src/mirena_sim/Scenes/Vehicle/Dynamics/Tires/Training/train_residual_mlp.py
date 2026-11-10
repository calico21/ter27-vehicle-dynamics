#!/usr/bin/env python3
"""
Physics-Informed Residual Neural Network Trainer for Mirena Dynamics Tire Model
Trains an ultra-compact Physics-Informed Neural Network (PINN Residual):
    Delta F = [Delta Fy, Delta Mz]
Features:
- Physics gating (passivity at zero slip, airborne load suppression, domain falloff)
- Physics-informed anti-symmetry augmentation (Delta F(-alpha, -gamma) = -Delta F(alpha, gamma))
- Strict train (Run 6) vs validation (Run 7) split with zero sample leakage
- Validation-based early stopping (prevents overfitting)
- Fixed deterministic seeds
- Export of weights, biases, normalization constants, and gate bounds for zero-dependency GDScript inference
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

PROCESSED_DIR = "src/mirena_sim/Scenes/Vehicle/Dynamics/Tires/Data/Hoosier_16x7.5-10/Processed"
MF62_DIR = "src/mirena_sim/Scenes/Vehicle/Dynamics/Tires/MF62"
NEURAL_DIR = "src/mirena_sim/Scenes/Vehicle/Dynamics/Tires/NeuralCorrection"
VALIDATION_DIR = "src/mirena_sim/Scenes/Vehicle/Dynamics/Tires/Validation"

os.makedirs(NEURAL_DIR, exist_ok=True)
os.makedirs(VALIDATION_DIR, exist_ok=True)

# Fixed deterministic seeds
np.random.seed(42)

# Robust path setup for calibrate_mf62 import
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
from calibrate_mf62 import Pacejka62Fitter


class PhysicsInformedResidualMLP:
    """
    Inference and gating engine for Physics-Informed Residual Neural Network.
    Ensures exact mathematical parity between Python and GDScript implementations.
    """
    def __init__(self, mlp_model, x_mean, x_std, y_mean, y_std):
        self.mlp = mlp_model
        self.x_mean = np.array(x_mean, dtype=float)
        self.x_std = np.array(x_std, dtype=float)
        self.y_mean = np.array(y_mean, dtype=float)
        self.y_std = np.array(y_std, dtype=float)

        # Physics gating thresholds
        self.slip_trans = 0.02   # rad (~1.1 deg)
        self.load_trans = 100.0  # N
        self.alpha_limit = 0.22  # rad (~12.6 deg)
        self.alpha_slope = 0.02
        self.fz_limit = 1300.0   # N
        self.fz_slope = 100.0
        self.clip_fy = 200.0     # N max residual
        self.clip_mz = 10.0      # Nm max residual

    def predict(self, X_raw):
        """
        Full physics-informed forward pass:
        1. Normalizes raw inputs [alpha, fz, gamma, p_kpa]
        2. Evaluates MLP
        3. Denormalizes outputs
        4. Applies analytical boundary gates (zero-slip passivity, low-load cut, domain falloff, clamp)
        """
        X_norm = (X_raw - self.x_mean) / self.x_std
        raw_pred_norm = self.mlp.predict(X_norm)
        raw_fy = raw_pred_norm[:, 0] * self.y_std[0] + self.y_mean[0]
        raw_mz = raw_pred_norm[:, 1] * self.y_std[1] + self.y_mean[1]

        alpha = X_raw[:, 0]
        fz = X_raw[:, 1]
        gamma = X_raw[:, 2]

        # Gate 1: Zero-slip passivity (smoothly vanishes at alpha=0, gamma=0)
        g_slip = np.tanh((np.abs(alpha) + np.abs(gamma)) / self.slip_trans)

        # Gate 2: Low-load suppression (smoothly vanishes when tire is unloaded / airborne)
        g_load = np.clip(fz / self.load_trans, 0.0, 1.0)

        # Gate 3: Domain falloff (smooth sigmoidal decay outside experimental envelope)
        g_domain_alpha = 1.0 / (1.0 + np.exp((np.abs(alpha) - self.alpha_limit) / self.alpha_slope))
        g_domain_fz = 1.0 / (1.0 + np.exp((fz - self.fz_limit) / self.fz_slope))
        scale = g_slip * g_load * g_domain_alpha * g_domain_fz

        delta_fy = np.clip(raw_fy * scale, -self.clip_fy, self.clip_fy)
        delta_mz = np.clip(raw_mz * scale, -self.clip_mz, self.clip_mz)
        return np.column_stack([delta_fy, delta_mz])


def compute_residuals(df, params):
    alpha = df["alpha"].values
    fz = df["f_z"].values
    gamma = df["gamma"].values
    p_kpa = df["p_kpa"].values
    fy_exp = df["f_y"].values
    mz_exp = df["m_z"].values
    fx_exp = df["f_x"].values

    # Compute baseline MF 6.2 predictions
    fy_mf, svy_mf = Pacejka62Fitter.compute_fy0(params, alpha, fz, gamma, p_kpa)
    mz_mf = Pacejka62Fitter.compute_mz(params, alpha, fz, gamma, p_kpa, fx=fx_exp, fy_in=fy_mf, svy_in=svy_mf)

    # Residuals: Delta = Exp - MF
    delta_fy = fy_exp - fy_mf
    delta_mz = mz_exp - mz_mf

    # Inputs: [alpha, fz, gamma, p_kpa]
    X_raw = np.column_stack([alpha, fz, gamma, p_kpa])
    Y_raw = np.column_stack([delta_fy, delta_mz])

    return X_raw, Y_raw, fy_mf, mz_mf


def train_pinn_compound(compound="R25B"):
    print(f"\n{'=' * 75}")
    print(f"   TRAINING PHYSICS-INFORMED RESIDUAL PINN FOR HOOSIER 16.0x7.5-10 {compound}")
    print(f"{'=' * 75}")

    param_file = os.path.join(MF62_DIR, f"hoosier_16x7.5_10_{compound.lower()}_mf62.json")
    with open(param_file, "r") as fp:
        params = json.load(fp)

    df_train = pd.read_csv(os.path.join(PROCESSED_DIR, f"{compound}_train.csv"))
    df_val = pd.read_csv(os.path.join(PROCESSED_DIR, f"{compound}_val.csv"))

    print(f">> Computing MF 6.2 baseline residuals on Train ({len(df_train)} pts) and Val ({len(df_val)} pts)...")
    X_tr_raw, Y_tr_raw, fy_mf_tr, mz_mf_tr = compute_residuals(df_train, params)
    X_val_raw, Y_val_raw, fy_mf_val, mz_mf_val = compute_residuals(df_val, params)

    rmse_fy_tr_base = float(np.sqrt(np.mean((Y_tr_raw[:, 0]) ** 2)))
    rmse_mz_tr_base = float(np.sqrt(np.mean((Y_tr_raw[:, 1]) ** 2)))
    rmse_fy_val_base = float(np.sqrt(np.mean((Y_val_raw[:, 0]) ** 2)))
    rmse_mz_val_base = float(np.sqrt(np.mean((Y_val_raw[:, 1]) ** 2)))

    print(f"  [MF 6.2 Baseline Errors ({compound})]")
    print(f"    Train:      Fy RMSE = {rmse_fy_tr_base:6.2f} N  |  Mz RMSE = {rmse_mz_tr_base:5.2f} Nm")
    print(f"    Validation: Fy RMSE = {rmse_fy_val_base:6.2f} N  |  Mz RMSE = {rmse_mz_val_base:5.2f} Nm")

    # Anti-symmetry augmentation: (-alpha, fz, -gamma, p) -> (-delta_Fy, -delta_Mz)
    X_tr_sym = np.column_stack([-X_tr_raw[:, 0], X_tr_raw[:, 1], -X_tr_raw[:, 2], X_tr_raw[:, 3]])
    Y_tr_sym = np.column_stack([-Y_tr_raw[:, 0], -Y_tr_raw[:, 1]])

    X_train_aug = np.vstack([X_tr_raw, X_tr_sym])
    Y_train_aug = np.vstack([Y_tr_raw, Y_tr_sym])

    # Input Normalization
    x_mean = np.mean(X_tr_raw, axis=0)
    x_std = np.std(X_tr_raw, axis=0)
    x_std[x_std < 1e-4] = 1.0

    # Target Normalization (using augmented symmetric targets)
    y_mean = np.mean(Y_train_aug, axis=0)
    y_std = np.std(Y_train_aug, axis=0)
    y_std[y_std < 1e-4] = 1.0

    print(f">> Normalization Factors:")
    print(f"    Inputs  x_mean = {np.round(x_mean, 4)}, x_std = {np.round(x_std, 4)}")
    print(f"    Targets y_mean = {np.round(y_mean, 4)}, y_std = {np.round(y_std, 4)}")

    X_train_norm = (X_train_aug - x_mean) / x_std
    Y_train_norm = (Y_train_aug - y_mean) / y_std

    # Fit MLPRegressor with validation-based early stopping
    mlp = MLPRegressor(
        hidden_layer_sizes=(16, 16),
        activation="tanh",
        solver="adam",
        learning_rate_init=1.5e-3,
        alpha=1e-4,
        batch_size=256,
        max_iter=40,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=8,
        random_state=42
    )

    mlp.fit(X_train_norm, Y_train_norm)
    print(f">> MLP training completed in {mlp.n_iter_} iterations.")

    # Instantiate physics-informed model
    pinn = PhysicsInformedResidualMLP(mlp, x_mean, x_std, y_mean, y_std)

    # Evaluate predictions
    delta_tr_pred = pinn.predict(X_tr_raw)
    delta_val_pred = pinn.predict(X_val_raw)

    fy_hyb_tr = fy_mf_tr + delta_tr_pred[:, 0]
    mz_hyb_tr = mz_mf_tr + delta_tr_pred[:, 1]
    fy_hyb_val = fy_mf_val + delta_val_pred[:, 0]
    mz_hyb_val = mz_mf_val + delta_val_pred[:, 1]

    # Metrics
    rmse_fy_tr_hyb = float(np.sqrt(np.mean((df_train["f_y"].values - fy_hyb_tr) ** 2)))
    mae_fy_tr_hyb = float(np.mean(np.abs(df_train["f_y"].values - fy_hyb_tr)))
    r2_fy_tr_hyb = float(1.0 - (np.sum((df_train["f_y"].values - fy_hyb_tr) ** 2) / np.sum((df_train["f_y"].values - np.mean(df_train["f_y"].values)) ** 2)))

    rmse_mz_tr_hyb = float(np.sqrt(np.mean((df_train["m_z"].values - mz_hyb_tr) ** 2)))
    mae_mz_tr_hyb = float(np.mean(np.abs(df_train["m_z"].values - mz_hyb_tr)))
    r2_mz_tr_hyb = float(1.0 - (np.sum((df_train["m_z"].values - mz_hyb_tr) ** 2) / np.sum((df_train["m_z"].values - np.mean(df_train["m_z"].values)) ** 2)))

    rmse_fy_val_hyb = float(np.sqrt(np.mean((df_val["f_y"].values - fy_hyb_val) ** 2)))
    mae_fy_val_hyb = float(np.mean(np.abs(df_val["f_y"].values - fy_hyb_val)))
    r2_fy_val_hyb = float(1.0 - (np.sum((df_val["f_y"].values - fy_hyb_val) ** 2) / np.sum((df_val["f_y"].values - np.mean(df_val["f_y"].values)) ** 2)))

    rmse_mz_val_hyb = float(np.sqrt(np.mean((df_val["m_z"].values - mz_hyb_val) ** 2)))
    mae_mz_val_hyb = float(np.mean(np.abs(df_val["m_z"].values - mz_hyb_val)))
    r2_mz_val_hyb = float(1.0 - (np.sum((df_val["m_z"].values - mz_hyb_val) ** 2) / np.sum((df_val["m_z"].values - np.mean(df_val["m_z"].values)) ** 2)))

    print("\n" + "=" * 75)
    print(f"   QUANTITATIVE PERFORMANCE COMPARISON: PURE MF 6.2 vs HYBRID ({compound})")
    print("=" * 75)
    print(f"  [Train Set ({len(df_train)} pts)]")
    print(f"    Fy RMSE:  {rmse_fy_tr_base:6.2f} N  -> {rmse_fy_tr_hyb:6.2f} N   (Reduction: {(1 - rmse_fy_tr_hyb/rmse_fy_tr_base)*100:+.1f}%) | R^2: {r2_fy_tr_hyb:.4f}")
    print(f"    Mz RMSE:  {rmse_mz_tr_base:6.2f} Nm -> {rmse_mz_tr_hyb:6.2f} Nm  (Reduction: {(1 - rmse_mz_tr_hyb/rmse_mz_tr_base)*100:+.1f}%) | R^2: {r2_mz_tr_hyb:.4f}")
    print(f"\n  [Independent Validation Set ({len(df_val)} pts)]")
    print(f"    Fy RMSE:  {rmse_fy_val_base:6.2f} N  -> {rmse_fy_val_hyb:6.2f} N   (Reduction: {(1 - rmse_fy_val_hyb/rmse_fy_val_base)*100:+.1f}%) | R^2: {r2_fy_val_hyb:.4f}")
    print(f"    Mz RMSE:  {rmse_mz_val_base:6.2f} Nm -> {rmse_mz_val_hyb:6.2f} Nm  (Reduction: {(1 - rmse_mz_val_hyb/rmse_mz_val_base)*100:+.1f}%) | R^2: {r2_mz_val_hyb:.4f}")
    print("=" * 75)

    # Export weights for zero-dependency GDScript inference
    # Note: scikit-learn stores coefs as [in_dim, out_dim], GDScript expects [out_dim, in_dim]
    w1 = mlp.coefs_[0].T.tolist()
    b1 = mlp.intercepts_[0].tolist()
    w2 = mlp.coefs_[1].T.tolist()
    b2 = mlp.intercepts_[1].tolist()
    w3 = mlp.coefs_[2].T.tolist()
    b3 = mlp.intercepts_[2].tolist()

    export_payload = {
        "architecture": {
            "compound": compound,
            "in_dim": 4,
            "hidden1": 16,
            "hidden2": 16,
            "out_dim": 2,
            "inputs": ["alpha", "f_z", "gamma", "p_kpa"],
            "outputs": ["delta_Fy", "delta_Mz"],
            "activation": "tanh"
        },
        "normalization": {
            "x_mean": x_mean.tolist(),
            "x_std": x_std.tolist(),
            "y_mean": y_mean.tolist(),
            "y_std": y_std.tolist()
        },
        "weights": {
            "w1": w1,
            "b1": b1,
            "w2": w2,
            "b2": b2,
            "w3": w3,
            "b3": b3
        },
        "physics_gates": {
            "slip_trans_rad": float(pinn.slip_trans),
            "load_trans_n": float(pinn.load_trans),
            "alpha_limit_rad": float(pinn.alpha_limit),
            "alpha_slope_rad": float(pinn.alpha_slope),
            "fz_limit_n": float(pinn.fz_limit),
            "fz_slope_n": float(pinn.fz_slope),
            "clip_fy_n": float(pinn.clip_fy),
            "clip_mz_nm": float(pinn.clip_mz)
        }
    }

    # Save compound weights
    out_weights_path = os.path.join(NEURAL_DIR, f"neural_residual_weights_{compound.lower()}.json")
    with open(out_weights_path, "w") as fp:
        json.dump(export_payload, fp, indent=2)
    print(f">> Exported {compound} PINN residual weights to: {out_weights_path}")

    # If primary compound (R25B), also save default neural_residual_weights.json
    if compound == "R25B":
        default_weights_path = os.path.join(NEURAL_DIR, "neural_residual_weights.json")
        with open(default_weights_path, "w") as fp:
            json.dump(export_payload, fp, indent=2)
        print(f">> Exported default PINN residual weights to: {default_weights_path}")

    return {
        "compound": compound,
        "n_iter": int(mlp.n_iter_),
        "train": {
            "base_mf62": {"fy_rmse_n": rmse_fy_tr_base, "mz_rmse_nm": rmse_mz_tr_base},
            "hybrid_pinn": {"fy_rmse_n": rmse_fy_tr_hyb, "fy_mae_n": mae_fy_tr_hyb, "fy_r2": r2_fy_tr_hyb,
                            "mz_rmse_nm": rmse_mz_tr_hyb, "mz_mae_nm": mae_mz_tr_hyb, "mz_r2": r2_mz_tr_hyb}
        },
        "validation": {
            "base_mf62": {"fy_rmse_n": rmse_fy_val_base, "mz_rmse_nm": rmse_mz_val_base},
            "hybrid_pinn": {"fy_rmse_n": rmse_fy_val_hyb, "fy_mae_n": mae_fy_val_hyb, "fy_r2": r2_fy_val_hyb,
                            "mz_rmse_nm": rmse_mz_val_hyb, "mz_mae_nm": mae_mz_val_hyb, "mz_r2": r2_mz_val_hyb}
        }
    }


def main():
    print("=" * 75)
    print("   TRAINING PHYSICS-INFORMED RESIDUAL PINN (R25B & R20)")
    print("=" * 75)

    res_r25b = train_pinn_compound("R25B")
    res_r20 = train_pinn_compound("R20")

    combined_report = {
        "model_type": "MF 6.2 + Physics-Informed Residual PINN (Architecture B)",
        "R25B": res_r25b,
        "R20": res_r20
    }
    rep_path = os.path.join(VALIDATION_DIR, "neural_residual_report.json")
    with open(rep_path, "w") as fp:
        json.dump(combined_report, fp, indent=2)
    print(f"\n>> Exported combined neural validation report to: {rep_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()
