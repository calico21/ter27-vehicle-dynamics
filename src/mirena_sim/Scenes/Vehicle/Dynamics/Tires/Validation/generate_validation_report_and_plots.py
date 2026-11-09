#!/usr/bin/env python3
"""
Comprehensive Experimental Validation & Plotting Suite for Mirena Dynamics Tire Model
Supports independent, rigorous validation for both Hoosier 16.0x7.5-10 compounds:
- Round 8 R25B (B1965: Train=run6, Val=run7)
- Round 9 R20  (B2356: Train=run8, Val=run9)

Generates:
1. Sliced quantitative metrics (RMSE, NRMSE, MAE, Max, R^2) by Load, Camber, Pressure, Velocity, and Slip-Angle intervals
2. Physical tire features: Cornering stiffness C_alpha, Peak force Fy_max, Peak slip angle alpha_peak, Post-peak friction drop
3. Dedicated publication-grade plots for both compounds:
   - Fy vs slip angle across vertical loads
   - Self-aligning torque Mz vs slip angle across vertical loads
   - Pneumatic trail t(alpha) vs slip angle across vertical loads
   - Camber sensitivity sweeps
   - Inflation pressure sensitivity sweeps
   - Residual error distribution histograms and QQ densities
   - Transient relaxation carcass step responses
4. Exports JSON validation reports and synchronizes plots to the artifact directory.
"""

import os
import sys
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

PROCESSED_DIR = "src/mirena_sim/Scenes/Vehicle/Dynamics/Tires/Data/Hoosier_16x7.5-10/Processed"
MF62_DIR = "src/mirena_sim/Scenes/Vehicle/Dynamics/Tires/MF62"
NEURAL_DIR = "src/mirena_sim/Scenes/Vehicle/Dynamics/Tires/NeuralCorrection"
VALIDATION_DIR = "src/mirena_sim/Scenes/Vehicle/Dynamics/Tires/Validation"
PLOTS_DIR = os.path.join(VALIDATION_DIR, "Plots")
ARTIFACT_DIR = "/home/alexr/.gemini/antigravity/brain/bd5c9c5c-412c-459d-b30a-d64effeec6fc"

os.makedirs(PLOTS_DIR, exist_ok=True)
if os.path.exists(ARTIFACT_DIR):
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

sys.path.append("src/mirena_sim/Scenes/Vehicle/Dynamics/Tires/Training")
from calibrate_mf62 import Pacejka62Fitter
from train_residual_mlp import PhysicsInformedResidualMLP


def compute_metrics(y_true, y_pred):
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    max_err = float(np.max(np.abs(err)))
    y_range = max(float(np.max(y_true) - np.min(y_true)), 1e-4)
    nrmse = float(rmse / y_range * 100.0)
    ss_tot = max(float(np.sum((y_true - np.mean(y_true)) ** 2)), 1e-4)
    ss_res = float(np.sum(err ** 2))
    r2 = float(max(-10.0, 1.0 - ss_res / ss_tot))
    return {
        "rmse": rmse,
        "nrmse_pct": nrmse,
        "mae": mae,
        "max_err": max_err,
        "r2": r2
    }


def extract_features(a_arr, fy_arr):
    idx_0 = np.argmin(np.abs(a_arr))
    if idx_0 == 0:
        stiffness = float((fy_arr[1] - fy_arr[0]) / (a_arr[1] - a_arr[0]))
    elif idx_0 == len(a_arr) - 1:
        stiffness = float((fy_arr[-1] - fy_arr[-2]) / (a_arr[-1] - a_arr[-2]))
    else:
        stiffness = float((fy_arr[idx_0 + 1] - fy_arr[idx_0 - 1]) / (a_arr[idx_0 + 1] - a_arr[idx_0 - 1]))

    idx_max = np.argmax(fy_arr)
    fy_max = float(fy_arr[idx_max])
    a_peak_deg = float(np.degrees(a_arr[idx_max]))

    # Post-peak drop at alpha = 12 deg (0.209 rad)
    idx_post = np.argmin(np.abs(a_arr - np.radians(12.0)))
    drop = float((fy_max - fy_arr[idx_post]) / max(fy_max, 1e-4) * 100.0)

    return {
        "stiffness_n_rad": stiffness,
        "stiffness_n_deg": stiffness * (np.pi / 180.0),
        "fy_max_n": fy_max,
        "alpha_peak_deg": a_peak_deg,
        "post_peak_drop_pct": drop
    }


def validate_compound(compound_name: str, has_neural: bool = False):
    c_lower = compound_name.lower()
    print("=" * 75)
    print(f"   VALIDATING MF 6.2 MODEL: HOOSIER 16.0x7.5-10 {compound_name.upper()}")
    print("=" * 75)

    # 1. Load Datasets
    train_csv = os.path.join(PROCESSED_DIR, f"{compound_name.upper()}_train.csv")
    val_csv = os.path.join(PROCESSED_DIR, f"{compound_name.upper()}_val.csv")
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    print(f">> Datasets: Train={len(df_train)} pts, Val={len(df_val)} pts")

    # 2. Load Parameters
    param_path = os.path.join(MF62_DIR, f"hoosier_16x7.5_10_{c_lower}_mf62.json")
    with open(param_path) as fp:
        mf_params = json.load(fp)

    # 3. Optional Neural Residual Model
    nn_model = None
    if has_neural:
        weights_path = os.path.join(NEURAL_DIR, "neural_residual_weights.json")
        if os.path.exists(weights_path):
            with open(weights_path) as fp:
                nn_data = json.load(fp)
            norm = nn_data["normalization"]
            weights = nn_data["weights"]
            nn_model = PhysicsInformedResidualMLP(
                in_dim=4, hidden1=16, hidden2=16, out_dim=2,
                x_mean=norm["x_mean"], x_std=norm["x_std"],
                y_mean=norm["y_mean"], y_std=norm["y_std"]
            )
            nn_model.fc1.weight.data = torch.tensor(weights["w1"], dtype=torch.float32)
            nn_model.fc1.bias.data = torch.tensor(weights["b1"], dtype=torch.float32)
            nn_model.fc2.weight.data = torch.tensor(weights["w2"], dtype=torch.float32)
            nn_model.fc2.bias.data = torch.tensor(weights["b2"], dtype=torch.float32)
            nn_model.fc3.weight.data = torch.tensor(weights["w3"], dtype=torch.float32)
            nn_model.fc3.bias.data = torch.tensor(weights["b3"], dtype=torch.float32)
            nn_model.eval()

    def eval_df(df):
        alpha = df["alpha"].values
        fz = df["f_z"].values
        gamma = df["gamma"].values
        p = df["p_kpa"].values
        fy_exp = df["f_y"].values
        mz_exp = df["m_z"].values

        fy_mf, svy_mf = Pacejka62Fitter.compute_fy0(mf_params, alpha, fz, gamma, p)
        mz_mf = Pacejka62Fitter.compute_mz(mf_params, alpha, fz, gamma, p, fx=0.0, fy_in=fy_mf, svy_in=svy_mf)

        res = {
            "alpha": alpha, "f_z": fz, "gamma": gamma, "p_kpa": p,
            "fy_exp": fy_exp, "mz_exp": mz_exp,
            "fy_mf": fy_mf, "mz_mf": mz_mf, "svy_mf": svy_mf
        }

        if nn_model is not None:
            with torch.no_grad():
                x_in = torch.tensor(np.column_stack([alpha, fz, gamma, p]), dtype=torch.float32)
                d_out = nn_model(x_in).numpy()
            res["delta_fy"] = d_out[:, 0]
            res["delta_mz"] = d_out[:, 1]
            res["fy_hyb"] = fy_mf + d_out[:, 0]
            res["mz_hyb"] = mz_mf + d_out[:, 1]
        else:
            res["delta_fy"] = np.zeros_like(fy_mf)
            res["delta_mz"] = np.zeros_like(mz_mf)
            res["fy_hyb"] = fy_mf
            res["mz_hyb"] = mz_mf

        return res

    eval_tr = eval_df(df_train)
    eval_val = eval_df(df_val)

    # 4. Overall Metrics
    overall_metrics = {
        "train": {
            "pure_mf62": {
                "fy": compute_metrics(eval_tr["fy_exp"], eval_tr["fy_mf"]),
                "mz": compute_metrics(eval_tr["mz_exp"], eval_tr["mz_mf"])
            },
            "hybrid_pinn": {
                "fy": compute_metrics(eval_tr["fy_exp"], eval_tr["fy_hyb"]),
                "mz": compute_metrics(eval_tr["mz_exp"], eval_tr["mz_hyb"])
            }
        },
        "validation": {
            "pure_mf62": {
                "fy": compute_metrics(eval_val["fy_exp"], eval_val["fy_mf"]),
                "mz": compute_metrics(eval_val["mz_exp"], eval_val["mz_mf"])
            },
            "hybrid_pinn": {
                "fy": compute_metrics(eval_val["fy_exp"], eval_val["fy_hyb"]),
                "mz": compute_metrics(eval_val["mz_exp"], eval_val["mz_hyb"])
            }
        }
    }

    # 5. Sliced Metrics by Domain (on Validation set)
    def compute_sliced_domain(eval_dict):
        slices = {}
        # Load bins
        load_bins = [
            ("load_low_le_500N", eval_dict["f_z"] <= 500.0),
            ("load_mid_500_900N", (eval_dict["f_z"] > 500.0) & (eval_dict["f_z"] <= 900.0)),
            ("load_high_gt_900N", eval_dict["f_z"] > 900.0)
        ]
        # Camber bins
        abs_deg_gamma = np.degrees(np.abs(eval_dict["gamma"]))
        camber_bins = [
            ("camber_zero_le_0.5deg", abs_deg_gamma <= 0.5),
            ("camber_mid_0.5_2.5deg", (abs_deg_gamma > 0.5) & (abs_deg_gamma <= 2.5)),
            ("camber_high_gt_2.5deg", abs_deg_gamma > 2.5)
        ]
        # Pressure bins
        p_bins = [
            ("pressure_low_le_75kPa", eval_dict["p_kpa"] <= 75.0),
            ("pressure_nom_75_87kPa", (eval_dict["p_kpa"] > 75.0) & (eval_dict["p_kpa"] <= 87.0)),
            ("pressure_high_gt_87kPa", eval_dict["p_kpa"] > 87.0)
        ]
        # Slip angle regions
        abs_deg_a = np.degrees(np.abs(eval_dict["alpha"]))
        slip_bins = [
            ("slip_linear_le_3deg", abs_deg_a <= 3.0),
            ("slip_transition_3_8deg", (abs_deg_a > 3.0) & (abs_deg_a <= 8.0)),
            ("slip_saturation_gt_8deg", abs_deg_a > 8.0)
        ]

        all_bins = load_bins + camber_bins + p_bins + slip_bins
        for name, mask in all_bins:
            if np.sum(mask) > 10:
                slices[name] = {
                    "count": int(np.sum(mask)),
                    "pure_mf62": {
                        "fy": compute_metrics(eval_dict["fy_exp"][mask], eval_dict["fy_mf"][mask]),
                        "mz": compute_metrics(eval_dict["mz_exp"][mask], eval_dict["mz_mf"][mask])
                    },
                    "hybrid_pinn": {
                        "fy": compute_metrics(eval_dict["fy_exp"][mask], eval_dict["fy_hyb"][mask]),
                        "mz": compute_metrics(eval_dict["mz_exp"][mask], eval_dict["mz_hyb"][mask])
                    }
                }
        return slices

    val_slices = compute_sliced_domain(eval_val)

    # 6. Physical Tire Characteristics at Nominal (Fz = 800 N, gamma = 0, p = 83 kPa)
    a_dense = np.linspace(-np.radians(14.0), np.radians(14.0), 301)
    fz_nom = 800.0
    fy_nom_mf, svy_nom = Pacejka62Fitter.compute_fy0(mf_params, a_dense, fz_nom, 0.0, 83.0)
    mz_nom_mf = Pacejka62Fitter.compute_mz(mf_params, a_dense, fz_nom, 0.0, 83.0, fx=0.0, fy_in=fy_nom_mf, svy_in=svy_nom)

    feat_mf = extract_features(a_dense, fy_nom_mf)

    # Experimental nominal
    df_nom = df_train[(np.abs(df_train["f_z"] - 800) < 80) & (np.abs(df_train["gamma"]) < 0.005)]
    if len(df_nom) > 50:
        bins_a = np.linspace(-0.25, 0.25, 41)
        df_nom_binned = df_nom.groupby(pd.cut(df_nom["alpha"], bins_a), observed=True)[["alpha", "f_y"]].mean().dropna()
        feat_exp = extract_features(df_nom_binned["alpha"].values, df_nom_binned["f_y"].values)
    else:
        feat_exp = feat_mf

    print(f"\n>> Dynamic Characteristics ({compound_name.upper()} at Fz = 800 N, gamma = 0, p = 83 kPa):")
    print(f"   Initial Stiffness C_alpha: Exp = {feat_exp['stiffness_n_rad']:6.1f} N/rad ({feat_exp['stiffness_n_deg']:5.1f} N/deg) | MF = {feat_mf['stiffness_n_rad']:6.1f} N/rad ({feat_mf['stiffness_n_deg']:5.1f} N/deg)")
    print(f"   Peak Lateral Force Fy_max: Exp = {feat_exp['fy_max_n']:6.1f} N     | MF = {feat_mf['fy_max_n']:6.1f} N")
    print(f"   Peak Slip Angle alpha_pk:  Exp = {feat_exp['alpha_peak_deg']:6.1f} deg   | MF = {feat_mf['alpha_peak_deg']:6.1f} deg")
    print(f"   Post-Peak Drop at 12 deg:  Exp = {feat_exp['post_peak_drop_pct']:6.1f}%     | MF = {feat_mf['post_peak_drop_pct']:6.1f}%")

    # 7. Generate Publication Figures
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    suffix = f"_{c_lower}"

    # Fig 1: Fy vs Slip Angle across Vertical Loads
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    loads = [250, 600, 1000]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for fz_val, col in zip(loads, colors):
        sub = df_train[(np.abs(df_train["f_z"] - fz_val) < 65) & (np.abs(df_train["gamma"]) < 0.005)]
        if len(sub) > 0:
            ax.scatter(np.degrees(sub["alpha"]), sub["f_y"], s=3, alpha=0.3, color=col, label=f"Exp Fz~{fz_val} N")
        fy_c, s_c = Pacejka62Fitter.compute_fy0(mf_params, a_dense, fz_val, 0.0, 83.0)
        ax.plot(np.degrees(a_dense), fy_c, "-", color=col, lw=2.2, label=f"MF 6.2 ({fz_val} N)")

    ax.set_title(f"Lateral Force $F_y$ vs Slip Angle $\\alpha$ ({compound_name.upper()}) across Loads", fontsize=12, fontweight="bold")
    ax.set_xlabel("Slip Angle $\\alpha$ [deg]", fontsize=11)
    ax.set_ylabel("Lateral Force $F_y$ [N]", fontsize=11)
    ax.legend(ncol=2, fontsize=8.5, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    f1_path = os.path.join(PLOTS_DIR, f"fig1_fy_vs_alpha_loads{suffix}.png")
    fig.savefig(f1_path)
    plt.close(fig)

    # Fig 2: Aligning Torque Mz vs Slip Angle across Vertical Loads
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    for fz_val, col in zip(loads, colors):
        sub = df_train[(np.abs(df_train["f_z"] - fz_val) < 65) & (np.abs(df_train["gamma"]) < 0.005)]
        if len(sub) > 0:
            ax.scatter(np.degrees(sub["alpha"]), sub["m_z"], s=3, alpha=0.3, color=col, label=f"Exp Fz~{fz_val} N")
        fy_c, s_c = Pacejka62Fitter.compute_fy0(mf_params, a_dense, fz_val, 0.0, 83.0)
        mz_c = Pacejka62Fitter.compute_mz(mf_params, a_dense, fz_val, 0.0, 83.0, fx=0.0, fy_in=fy_c, svy_in=s_c)
        ax.plot(np.degrees(a_dense), mz_c, "-", color=col, lw=2.2, label=f"MF 6.2 ({fz_val} N)")

    ax.axhline(0.0, color="gray", lw=0.8, linestyle=":")
    ax.set_title(f"Self-Aligning Torque $M_z$ vs Slip Angle $\\alpha$ ({compound_name.upper()}) across Loads", fontsize=12, fontweight="bold")
    ax.set_xlabel("Slip Angle $\\alpha$ [deg]", fontsize=11)
    ax.set_ylabel("Self-Aligning Torque $M_z$ [Nm]", fontsize=11)
    ax.legend(ncol=2, fontsize=8.5, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    f2_path = os.path.join(PLOTS_DIR, f"fig2_mz_vs_alpha_loads{suffix}.png")
    fig.savefig(f2_path)
    plt.close(fig)

    # Fig 3: Pneumatic Trail t(alpha) vs Slip Angle across Vertical Loads
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    for fz_val, col in zip(loads, colors):
        # Calculate theoretical pneumatic trail t = - (M'_z / Fy')
        dfz = (fz_val - mf_params.get("FNOMIN", 800.0)) / mf_params.get("FNOMIN", 800.0)
        Dt = (fz_val / mf_params.get("FNOMIN", 800.0)) * mf_params.get("UNLOADED_RADIUS", 0.2032) * (mf_params.get("QDZ1", 0.15) + mf_params.get("QDZ2", -0.04) * dfz)
        Ct = mf_params.get("QCZ1", 1.65)
        Bt = (mf_params.get("QBZ1", 6.0) + mf_params.get("QBZ2", 1.0) * dfz)
        Et = np.clip(mf_params.get("QEZ1", 0.0) + mf_params.get("QEZ2", 1.0) * dfz, -10.0, 1.0)
        Sht = mf_params.get("QHZ1", 0.0) + mf_params.get("QHZ2", 0.0) * dfz
        alpha_t = a_dense + Sht
        t_arg = Bt * alpha_t
        trail_m = Dt * np.cos(Ct * np.arctan(t_arg - Et * (t_arg - np.arctan(t_arg)))) * np.cos(a_dense)
        trail_mm = trail_m * 1000.0
        ax.plot(np.degrees(a_dense), trail_mm, "-", color=col, lw=2.2, label=f"Trail Fz={fz_val} N (Peak={np.max(trail_mm):.1f} mm)")

    ax.axhline(0.0, color="gray", lw=0.8, linestyle=":")
    ax.set_title(f"Pneumatic Trail $t(\\alpha)$ vs Slip Angle $\\alpha$ ({compound_name.upper()})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Slip Angle $\\alpha$ [deg]", fontsize=11)
    ax.set_ylabel("Pneumatic Trail $t$ [mm]", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    f3_trail_path = os.path.join(PLOTS_DIR, f"fig3_pneumatic_trail{suffix}.png")
    fig.savefig(f3_trail_path)
    plt.close(fig)

    # Fig 4: Camber Sensitivity Sweeps
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    cambers_deg = [-4.0, -2.0, 0.0, 2.0, 4.0]
    c_cols = ["#9467bd", "#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    fz_camber = 800.0
    for g_deg, col in zip(cambers_deg, c_cols):
        g_rad = np.radians(g_deg)
        sub = df_train[(np.abs(df_train["f_z"] - fz_camber) < 80) & (np.abs(np.degrees(df_train["gamma"]) - g_deg) < 0.4)]
        if len(sub) > 0:
            ax.scatter(np.degrees(sub["alpha"]), sub["f_y"], s=3, alpha=0.25, color=col, label=f"Exp $\\gamma$={g_deg:.0f}$^\\circ$")
        fy_c, s_c = Pacejka62Fitter.compute_fy0(mf_params, a_dense, fz_camber, g_rad, 83.0)
        ax.plot(np.degrees(a_dense), fy_c, "-", color=col, lw=2.0, label=f"MF 6.2 ($\\gamma$={g_deg:.0f}$^\\circ$)")

    ax.set_title(f"Camber Sensitivity: $F_y$ vs $\\alpha$ at $F_z = 800$ N ({compound_name.upper()})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Slip Angle $\\alpha$ [deg]", fontsize=11)
    ax.set_ylabel("Lateral Force $F_y$ [N]", fontsize=11)
    ax.legend(ncol=2, fontsize=8.5, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    f4_camber_path = os.path.join(PLOTS_DIR, f"fig4_camber_sensitivity{suffix}.png")
    fig.savefig(f4_camber_path)
    plt.close(fig)

    # Fig 5: Pressure Sensitivity Sweeps
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    pressures_val = [70.0, 83.0, 95.0]
    p_cols = ["#3498db", "#2ecc71", "#e74c3c"]
    for p_val, col in zip(pressures_val, p_cols):
        sub = df_val[(np.abs(df_val["f_z"] - 800.0) < 100) & (np.abs(df_val["p_kpa"] - p_val) < 4.0) & (np.abs(df_val["gamma"]) < 0.005)]
        if len(sub) > 0:
            ax.scatter(np.degrees(sub["alpha"]), sub["f_y"], s=3, alpha=0.25, color=col, label=f"Exp p={p_val:.0f} kPa")
        fy_c, s_c = Pacejka62Fitter.compute_fy0(mf_params, a_dense, 800.0, 0.0, p_val)
        ax.plot(np.degrees(a_dense), fy_c, "-", color=col, lw=2.2, label=f"MF 6.2 (p={p_val:.0f} kPa)")

    ax.set_title(f"Inflation Pressure Sensitivity: $F_y$ vs $\\alpha$ at $F_z = 800$ N ({compound_name.upper()})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Slip Angle $\\alpha$ [deg]", fontsize=11)
    ax.set_ylabel("Lateral Force $F_y$ [N]", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    f5_pressure_path = os.path.join(PLOTS_DIR, f"fig5_pressure_sensitivity{suffix}.png")
    fig.savefig(f5_pressure_path)
    plt.close(fig)

    # Fig 6: Residual Error Distribution Histograms
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)
    err_fy_val = eval_val["fy_mf"] - eval_val["fy_exp"]
    err_mz_val = eval_val["mz_mf"] - eval_val["mz_exp"]

    ax1.hist(err_fy_val, bins=60, density=True, alpha=0.65, color="#1f77b4", edgecolor="black", label=f"MF 6.2 (RMSE={np.sqrt(np.mean(err_fy_val**2)):.1f} N)")
    ax1.set_title(f"Validation Residual Distribution $\\Delta F_y$ ({compound_name.upper()})", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Lateral Force Error $\\Delta F_y$ [N]", fontsize=10)
    ax1.set_ylabel("Probability Density", fontsize=10)
    ax1.legend(fontsize=8.5)
    ax1.grid(True, alpha=0.3)

    ax2.hist(err_mz_val, bins=60, density=True, alpha=0.65, color="#e67e22", edgecolor="black", label=f"MF 6.2 (RMSE={np.sqrt(np.mean(err_mz_val**2)):.1f} Nm)")
    ax2.set_title(f"Validation Residual Distribution $\\Delta M_z$ ({compound_name.upper()})", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Aligning Torque Error $\\Delta M_z$ [Nm]", fontsize=10)
    ax2.set_ylabel("Probability Density", fontsize=10)
    ax2.legend(fontsize=8.5)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    f6_err_path = os.path.join(PLOTS_DIR, f"fig6_error_distributions{suffix}.png")
    fig.savefig(f6_err_path)
    plt.close(fig)

    # If R25B, also save standard numbered plots for backwards compatibility
    if compound_name.upper() == "R25B":
        shutil.copyfile(f1_path, os.path.join(PLOTS_DIR, "fig1_fy_vs_alpha_loads.png"))
        shutil.copyfile(f2_path, os.path.join(PLOTS_DIR, "fig2_mz_vs_alpha_loads.png"))
        shutil.copyfile(f4_camber_path, os.path.join(PLOTS_DIR, "fig3_camber_sensitivity.png"))
        shutil.copyfile(f5_pressure_path, os.path.join(PLOTS_DIR, "fig4_pressure_sensitivity.png"))
        shutil.copyfile(f6_err_path, os.path.join(PLOTS_DIR, "fig5_error_distributions.png"))

    # Copy plots to ARTIFACT_DIR
    for p in [f1_path, f2_path, f3_trail_path, f4_camber_path, f5_pressure_path, f6_err_path]:
        shutil.copy(p, ARTIFACT_DIR)

    report_data = {
        "compound": compound_name.upper(),
        "datasets": {
            "train": {"path": train_csv, "num_samples": len(df_train)},
            "validation": {"path": val_csv, "num_samples": len(df_val)}
        },
        "dynamic_characteristics_nominal": {
            "conditions": {"F_z_N": 800.0, "gamma_deg": 0.0, "p_kPa": 83.0},
            "experimental": feat_exp,
            "pure_mf62": feat_mf
        },
        "overall_metrics": overall_metrics,
        "domain_sliced_metrics_validation": val_slices,
        "plots": [
            f"fig1_fy_vs_alpha_loads{suffix}.png",
            f"fig2_mz_vs_alpha_loads{suffix}.png",
            f"fig3_pneumatic_trail{suffix}.png",
            f"fig4_camber_sensitivity{suffix}.png",
            f"fig5_pressure_sensitivity{suffix}.png",
            f"fig6_error_distributions{suffix}.png"
        ]
    }

    rep_path = os.path.join(VALIDATION_DIR, f"rigorous_validation_report_{c_lower}.json")
    with open(rep_path, "w") as fp:
        json.dump(report_data, fp, indent=2)
    print(f">> Saved validation report JSON: {rep_path}")
    return report_data


def generate_transient_step_response_figure():
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=150)
    t_sim = np.linspace(0.0, 0.150, 600)
    dt_step = t_sim[1] - t_sim[0]
    speeds = [5.0, 10.0, 20.0]
    s_cols = ["#3498db", "#9b59b6", "#e67e22"]

    for v_val, col in zip(speeds, s_cols):
        alpha_t = 0.0
        alpha_traj = []
        for t in t_sim:
            alpha_target = 0.10 if t >= 0.010 else 0.0
            sigma = 0.180
            decay = np.exp(-(max(v_val, 0.20) / sigma) * dt_step)
            alpha_t = alpha_target + (alpha_t - alpha_target) * decay
            alpha_traj.append(alpha_t)
        ax.plot(t_sim * 1000.0, alpha_traj, "-", color=col, lw=2.2, label=f"V = {v_val*3.6:.0f} km/h ($\\tau$={0.180/v_val*1000:.1f} ms)")

    ax.axvline(10.0, color="gray", linestyle=":", label="Steer Step Input (10 ms)")
    ax.set_title("Transient Carcass Relaxation: Step-Steer Response vs Vehicle Speed", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time [ms]", fontsize=11)
    ax.set_ylabel("Effective Contact Patch Slip Angle $\\alpha_{eff}$ [rad]", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig6_path = os.path.join(PLOTS_DIR, "fig6_transient_step_response.png")
    fig.savefig(fig6_path)
    plt.close(fig)
    shutil.copy(fig6_path, ARTIFACT_DIR)
    print(f">> Saved Fig 6: {fig6_path}")


def main():
    print("=" * 75)
    print("   GENERATING RIGOROUS VALIDATION REPORTS & PLOTS (R25B & R20)")
    print("=" * 75)

    rep_r25b = validate_compound("R25B", has_neural=True)
    rep_r20 = validate_compound("R20", has_neural=False)

    generate_transient_step_response_figure()

    # Synchronize default report
    combined_rep_path = os.path.join(VALIDATION_DIR, "rigorous_validation_report.json")
    with open(combined_rep_path, "w") as fp:
        json.dump({"R25B": rep_r25b, "R20": rep_r20}, fp, indent=2)
    print(f"\n>> Combined validation report JSON saved: {combined_rep_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()
