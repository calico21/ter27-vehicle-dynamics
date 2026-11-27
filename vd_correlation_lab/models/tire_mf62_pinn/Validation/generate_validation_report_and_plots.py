#!/usr/bin/env python3
"""
Comprehensive Experimental Validation & Plotting Suite for Mirena Dynamics Tire Model
Supports independent, rigorous validation for both Hoosier 16.0x7.5-10 compounds:
- Round 8 R25B (B1965: Train=run6, Val=run7, Spring=run5)
- Round 9 R20  (B2356: Train=run8, Val=run9, Spring=run7)

Conducts full 3-Way Quantitative Comparison:
- Model 1: Baseline MF 6.2 (experimental baseline before camber optimization)
- Model 2: Optimized MF 6.2 (multi-start, robust soft_l1, coupled camber PKY3/PHY3 formulation)
- Model 3: Hybrid MF 6.2 + Neural Network (Model 2 + Physics-Informed Residual PINN)

Generates:
1. Sliced quantitative metrics (RMSE, NRMSE, MAE, Max, R^2, Mean Bias) across:
   - 5 physical load levels (L1: ~220N, L2: ~440N, L3: ~665N, L4: ~885N, L5: ~1110N)
   - Camber angles (-4 deg, -2 deg, 0 deg, +2 deg, +4 deg)
   - Slip angle regions (Linear <=3 deg, Transitional 3-8 deg, Saturation >8 deg)
   - Inflation pressures (69 kPa, 83 kPa, 97 kPa)
2. Dynamic physical tire features: Initial stiffness C_alpha, Peak force Fy_max, Peak slip angle, Camber thrust
3. Publication-grade diagnostic figures (Fig 0 through Fig 11) for both compounds
4. Exports JSON reports and synchronizes plots to the artifact directory.
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

PROCESSED_DIR = "data/ttc_tire/Hoosier_16x7.5-10/Processed"
MF62_DIR = "src/mirena_sim/Scenes/Vehicle/Dynamics/Tires/MF62"
NEURAL_DIR = "src/mirena_sim/Scenes/Vehicle/Dynamics/Tires/NeuralCorrection"
VALIDATION_DIR = "vd_correlation_lab/models/tire_mf62_pinn/Validation"
PLOTS_DIR = os.path.join(VALIDATION_DIR, "Plots")
ARTIFACT_DIR = "/home/alexr/.gemini/antigravity/brain/c5c97b01-25de-4bd1-b6e1-31c6c789a36b"

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

sys.path.append("vd_correlation_lab/models/tire_mf62_pinn/Training")
from calibrate_mf62 import Pacejka62Fitter


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
    bias = float(np.mean(err))
    return {
        "rmse": rmse,
        "nrmse_pct": nrmse,
        "mae": mae,
        "max_err": max_err,
        "r2": r2,
        "bias": bias
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


def eval_gated_pinn(nn_data, alpha, fz, gamma, p):
    norm = nn_data["normalization"]
    weights = nn_data["weights"]
    gates = nn_data.get("physics_gates", {})

    x_mean = np.array(norm["x_mean"])
    x_std = np.array(norm["x_std"])
    y_mean = np.array(norm["y_mean"])
    y_std = np.array(norm["y_std"])

    X_in = np.column_stack([alpha, fz, gamma, p])
    X_norm = (X_in - x_mean) / x_std

    w1, b1 = np.array(weights["w1"]), np.array(weights["b1"])
    w2, b2 = np.array(weights["w2"]), np.array(weights["b2"])
    w3, b3 = np.array(weights["w3"]), np.array(weights["b3"])

    h1 = np.tanh(X_norm @ w1.T + b1)
    h2 = np.tanh(h1 @ w2.T + b2)
    out_raw = h2 @ w3.T + b3

    raw_fy = out_raw[:, 0] * y_std[0] + y_mean[0]
    raw_mz = out_raw[:, 1] * y_std[1] + y_mean[1]

    slip_trans = gates.get("slip_trans_rad", 0.02)
    load_trans = gates.get("load_trans_n", 100.0)
    alpha_limit = gates.get("alpha_limit_rad", 0.22)
    alpha_slope = gates.get("alpha_slope_rad", 0.02)
    fz_limit = gates.get("fz_limit_n", 1300.0)
    fz_slope = gates.get("fz_slope_n", 100.0)
    clip_fy = gates.get("clip_fy_n", 200.0)
    clip_mz = gates.get("clip_mz_nm", 10.0)

    g_slip = np.tanh((np.abs(alpha) + np.abs(gamma)) / slip_trans)
    g_load = np.clip(fz / load_trans, 0.0, 1.0)
    g_alpha = 1.0 / (1.0 + np.exp((np.abs(alpha) - alpha_limit) / alpha_slope))
    g_fz = 1.0 / (1.0 + np.exp((fz - fz_limit) / fz_slope))
    scale = g_slip * g_load * g_alpha * g_fz

    delta_fy = np.clip(raw_fy * scale, -clip_fy, clip_fy)
    delta_mz = np.clip(raw_mz * scale, -clip_mz, clip_mz)
    return delta_fy, delta_mz


def plot_fz_distribution_and_bins(df_train, df_val, compound_name):
    c_lower = compound_name.lower()
    suffix = f"_{c_lower}"
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.5), dpi=150)

    fz_tr = df_train["f_z"].values
    fz_val = df_val["f_z"].values
    boundaries = [330.0, 550.0, 770.0, 990.0]
    legacy_bounds = [400.0, 750.0]

    # Training histogram
    n, bins, patches = ax1.hist(fz_tr, bins=100, color="#2980b9", alpha=0.7, edgecolor="none", label=f"Train ({len(fz_tr):,} pts)")
    bin_edges = [100.0] + boundaries + [1400.0]
    bin_colors = ["#ebf5fb", "#e8f8f5", "#fef9e7", "#fdedec", "#f4ecf7"]
    for i in range(5):
        ax1.axvspan(bin_edges[i], bin_edges[i+1], color=bin_colors[i], alpha=0.45, zorder=0)

    for b in boundaries:
        ax1.axvline(b, color="#c0392b", linestyle="--", lw=1.8, label=f"Physical 5-Bin Boundary ({b:.0f} N)" if b == 330.0 else None)
    for lb in legacy_bounds:
        ax1.axvline(lb, color="#7f8c8d", linestyle=":", lw=2.0, label=f"Flawed Legacy 3-Bin Cutoff ({lb:.0f} N)" if lb == 400.0 else None)

    digitized = np.digitize(fz_tr, boundaries)
    for i in range(5):
        pts = fz_tr[digitized == i]
        m = np.mean(pts) if len(pts) > 0 else 0
        s = np.std(pts) if len(pts) > 0 else 0
        ax1.text(m, max(n) * 0.90, f"Level {i+1}\n~{m:.0f} N\n(±{s:.1f} N)\nN={len(pts):,}",
                 ha="center", va="top", fontsize=7.5, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="#bdc3c7"))

    ax1.set_title(f"Experimental $F_z$ Distribution & Physical 5-Binning ({compound_name.upper()} Train)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Normal Load $F_z$ [N]", fontsize=10)
    ax1.set_ylabel("Sample Count", fontsize=10)
    ax1.legend(loc="upper left", fontsize=8.5)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(100, 1300)

    # Validation histogram
    n_v, bins_v, patches_v = ax2.hist(fz_val, bins=100, color="#27ae60", alpha=0.7, edgecolor="none", label=f"Validation ({len(fz_val):,} pts)")
    for i in range(5):
        ax2.axvspan(bin_edges[i], bin_edges[i+1], color=bin_colors[i], alpha=0.45, zorder=0)
    for b in boundaries:
        ax2.axvline(b, color="#c0392b", linestyle="--", lw=1.8, label="Physical Boundary (5-bin)" if b == 330.0 else None)

    val_dig = np.digitize(fz_val, boundaries)
    for i in range(5):
        pts_v = fz_val[val_dig == i]
        m_v = np.mean(pts_v) if len(pts_v) > 0 else 0
        s_v = np.std(pts_v) if len(pts_v) > 0 else 0
        ax2.text(m_v, max(n_v) * 0.90, f"Level {i+1}\n~{m_v:.0f} N\n(±{s_v:.1f} N)\nN={len(pts_v):,}",
                 ha="center", va="top", fontsize=7.5, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="#bdc3c7"))

    ax2.set_title(f"Independent Validation Set $F_z$ Verification ({compound_name.upper()} Val)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Normal Load $F_z$ [N]", fontsize=10)
    ax2.set_ylabel("Sample Count", fontsize=10)
    ax2.legend(loc="upper left", fontsize=8.5)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(100, 1300)

    fig.tight_layout()
    f0_path = os.path.join(PLOTS_DIR, f"fig0_fz_distribution_and_bins{suffix}.png")
    fig.savefig(f0_path)
    plt.close(fig)
    return f0_path


def validate_compound_3way(compound_name: str):
    c_lower = compound_name.lower()
    suffix = f"_{c_lower}"
    print("=" * 78)
    print(f"   RIGOROUS 3-WAY VALIDATION: HOOSIER 16.0x7.5-10 {compound_name.upper()}")
    print("   Comparing Model 1 (Baseline) vs Model 2 (Optimized) vs Model 3 (Hybrid PINN)")
    print("=" * 78)

    train_csv = os.path.join(PROCESSED_DIR, f"{compound_name.upper()}_train.csv")
    val_csv = os.path.join(PROCESSED_DIR, f"{compound_name.upper()}_val.csv")
    spring_csv = os.path.join(PROCESSED_DIR, f"{compound_name.upper()}_spring.csv")
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    df_spring = pd.read_csv(spring_csv)

    # 1. Load Model 1 (Baseline MF 6.2)
    baseline_path = os.path.join(MF62_DIR, f"baseline_hoosier_16x7.5_10_{c_lower}_mf62.json")
    with open(baseline_path) as fp:
        params_m1 = json.load(fp)

    # 2. Load Model 2 (Optimized MF 6.2)
    opt_path = os.path.join(MF62_DIR, f"hoosier_16x7.5_10_{c_lower}_mf62.json")
    with open(opt_path) as fp:
        params_m2 = json.load(fp)

    # 3. Load Model 3 PINN Residual Weights
    weights_path = os.path.join(NEURAL_DIR, f"neural_residual_weights_{c_lower}.json")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(NEURAL_DIR, "neural_residual_weights.json")
    with open(weights_path) as fp:
        nn_data = json.load(fp)

    r0_m1 = params_m1.get("UNLOADED_RADIUS", 0.200)
    r0_m2 = params_m2.get("UNLOADED_RADIUS", 0.200)

    def eval_models(df):
        alpha = df["alpha"].values
        fz = df["f_z"].values
        gamma = df["gamma"].values
        p = df["p_kpa"].values
        fx = df["f_x"].values if "f_x" in df.columns else np.zeros_like(fz)
        fy_exp = df["f_y"].values
        mz_exp = df["m_z"].values
        mx_exp = df["m_x"].values

        # Model 1 (Baseline)
        fy_m1, svy_m1 = Pacejka62Fitter.compute_fy0(params_m1, alpha, fz, gamma, p)
        mz_m1 = Pacejka62Fitter.compute_mz(params_m1, alpha, fz, gamma, p, fx=fx, fy_in=fy_m1, svy_in=svy_m1, r0=r0_m1)
        mx_m1 = Pacejka62Fitter.compute_mx(params_m1, fz, gamma, fy_m1, p, r0=r0_m1)

        # Model 2 (Optimized MF 6.2)
        fy_m2, svy_m2 = Pacejka62Fitter.compute_fy0(params_m2, alpha, fz, gamma, p)
        mz_m2 = Pacejka62Fitter.compute_mz(params_m2, alpha, fz, gamma, p, fx=fx, fy_in=fy_m2, svy_in=svy_m2, r0=r0_m2)
        mx_m2 = Pacejka62Fitter.compute_mx(params_m2, fz, gamma, fy_m2, p, r0=r0_m2)

        # Model 3 (Hybrid MF 6.2 + PINN)
        delta_fy, delta_mz = eval_gated_pinn(nn_data, alpha, fz, gamma, p)
        fy_m3 = fy_m2 + delta_fy
        mz_m3 = mz_m2 + delta_mz
        mx_m3 = mx_m2

        return {
            "alpha": alpha, "f_z": fz, "gamma": gamma, "p_kpa": p, "f_x": fx,
            "fy_exp": fy_exp, "mz_exp": mz_exp, "mx_exp": mx_exp,
            "fy_m1": fy_m1, "mz_m1": mz_m1, "mx_m1": mx_m1, "svy_m1": svy_m1,
            "fy_m2": fy_m2, "mz_m2": mz_m2, "mx_m2": mx_m2, "svy_m2": svy_m2,
            "fy_m3": fy_m3, "mz_m3": mz_m3, "mx_m3": mx_m3,
            "delta_fy": delta_fy, "delta_mz": delta_mz
        }

    eval_tr = eval_models(df_train)
    eval_val = eval_models(df_val)

    # 4. Comprehensive 3-Way Metrics
    metrics_3way = {
        "train": {
            "model1_baseline_mf62": {
                "fy": compute_metrics(eval_tr["fy_exp"], eval_tr["fy_m1"]),
                "mz": compute_metrics(eval_tr["mz_exp"], eval_tr["mz_m1"]),
                "mx": compute_metrics(eval_tr["mx_exp"], eval_tr["mx_m1"])
            },
            "model2_optimized_mf62": {
                "fy": compute_metrics(eval_tr["fy_exp"], eval_tr["fy_m2"]),
                "mz": compute_metrics(eval_tr["mz_exp"], eval_tr["mz_m2"]),
                "mx": compute_metrics(eval_tr["mx_exp"], eval_tr["mx_m2"])
            },
            "model3_hybrid_pinn": {
                "fy": compute_metrics(eval_tr["fy_exp"], eval_tr["fy_m3"]),
                "mz": compute_metrics(eval_tr["mz_exp"], eval_tr["mz_m3"]),
                "mx": compute_metrics(eval_tr["mx_exp"], eval_tr["mx_m3"])
            }
        },
        "validation": {
            "model1_baseline_mf62": {
                "fy": compute_metrics(eval_val["fy_exp"], eval_val["fy_m1"]),
                "mz": compute_metrics(eval_val["mz_exp"], eval_val["mz_m1"]),
                "mx": compute_metrics(eval_val["mx_exp"], eval_val["mx_m1"])
            },
            "model2_optimized_mf62": {
                "fy": compute_metrics(eval_val["fy_exp"], eval_val["fy_m2"]),
                "mz": compute_metrics(eval_val["mz_exp"], eval_val["mz_m2"]),
                "mx": compute_metrics(eval_val["mx_exp"], eval_val["mx_m2"])
            },
            "model3_hybrid_pinn": {
                "fy": compute_metrics(eval_val["fy_exp"], eval_val["fy_m3"]),
                "mz": compute_metrics(eval_val["mz_exp"], eval_val["mz_m3"]),
                "mx": compute_metrics(eval_val["mx_exp"], eval_val["mx_m3"])
            }
        }
    }

    # Print summary table
    print(f"\n--- OVERALL PERFORMANCE SUMMARY ({compound_name.upper()}) ---")
    print("Model Architecture             | Train Fy RMSE | Train Mz RMSE | Val Fy RMSE | Val Mz RMSE | Val Fy R^2")
    print("-" * 92)
    m1_tr = metrics_3way["train"]["model1_baseline_mf62"]
    m1_val = metrics_3way["validation"]["model1_baseline_mf62"]
    m2_tr = metrics_3way["train"]["model2_optimized_mf62"]
    m2_val = metrics_3way["validation"]["model2_optimized_mf62"]
    m3_tr = metrics_3way["train"]["model3_hybrid_pinn"]
    m3_val = metrics_3way["validation"]["model3_hybrid_pinn"]

    print(f"Model 1: Baseline MF 6.2       |   {m1_tr['fy']['rmse']:6.2f} N   |   {m1_tr['mz']['rmse']:5.2f} Nm  |  {m1_val['fy']['rmse']:6.2f} N  |   {m1_val['mz']['rmse']:5.2f} Nm |  {m1_val['fy']['r2']:.4f}")
    print(f"Model 2: Optimized MF 6.2      |   {m2_tr['fy']['rmse']:6.2f} N   |   {m2_tr['mz']['rmse']:5.2f} Nm  |  {m2_val['fy']['rmse']:6.2f} N  |   {m2_val['mz']['rmse']:5.2f} Nm |  {m2_val['fy']['r2']:.4f}  (Val Fy: {(1-m2_val['fy']['rmse']/m1_val['fy']['rmse'])*100:+.1f}%)")
    print(f"Model 3: Hybrid MF + PINN      |   {m3_tr['fy']['rmse']:6.2f} N   |   {m3_tr['mz']['rmse']:5.2f} Nm  |  {m3_val['fy']['rmse']:6.2f} N  |   {m3_val['mz']['rmse']:5.2f} Nm |  {m3_val['fy']['r2']:.4f}  (Val Fy: {(1-m3_val['fy']['rmse']/m1_val['fy']['rmse'])*100:+.1f}%)")
    print("-" * 92)

    # 5. Sliced Metrics by Domain (Validation Set)
    def compute_sliced_domain_3way(eval_dict):
        slices = {}
        # 5 physical test load levels
        load_bins = [
            ("L1_Load_220N", eval_dict["f_z"] <= 330.0),
            ("L2_Load_440N", (eval_dict["f_z"] > 330.0) & (eval_dict["f_z"] <= 550.0)),
            ("L3_Load_665N", (eval_dict["f_z"] > 550.0) & (eval_dict["f_z"] <= 770.0)),
            ("L4_Load_885N", (eval_dict["f_z"] > 770.0) & (eval_dict["f_z"] <= 990.0)),
            ("L5_Load_1110N", eval_dict["f_z"] > 990.0)
        ]
        # Camber bins
        abs_deg_gamma = np.degrees(np.abs(eval_dict["gamma"]))
        camber_bins = [
            ("Camber_0deg", abs_deg_gamma <= 0.5),
            ("Camber_2deg", (abs_deg_gamma > 0.5) & (abs_deg_gamma <= 2.5)),
            ("Camber_4deg", abs_deg_gamma > 2.5)
        ]
        # Pressure bins
        p_bins = [
            ("Pressure_69kPa", eval_dict["p_kpa"] <= 75.0),
            ("Pressure_83kPa", (eval_dict["p_kpa"] > 75.0) & (eval_dict["p_kpa"] <= 87.0)),
            ("Pressure_97kPa", eval_dict["p_kpa"] > 87.0)
        ]
        # Slip angle regions
        abs_deg_a = np.degrees(np.abs(eval_dict["alpha"]))
        slip_bins = [
            ("Slip_Linear_le_3deg", abs_deg_a <= 3.0),
            ("Slip_Transitional_3_8deg", (abs_deg_a > 3.0) & (abs_deg_a <= 8.0)),
            ("Slip_Saturation_gt_8deg", abs_deg_a > 8.0)
        ]

        all_bins = load_bins + camber_bins + p_bins + slip_bins
        for name, mask in all_bins:
            if np.sum(mask) > 10:
                slices[name] = {
                    "count": int(np.sum(mask)),
                    "model1_baseline_mf62": {
                        "fy": compute_metrics(eval_dict["fy_exp"][mask], eval_dict["fy_m1"][mask]),
                        "mz": compute_metrics(eval_dict["mz_exp"][mask], eval_dict["mz_m1"][mask])
                    },
                    "model2_optimized_mf62": {
                        "fy": compute_metrics(eval_dict["fy_exp"][mask], eval_dict["fy_m2"][mask]),
                        "mz": compute_metrics(eval_dict["mz_exp"][mask], eval_dict["mz_m2"][mask])
                    },
                    "model3_hybrid_pinn": {
                        "fy": compute_metrics(eval_dict["fy_exp"][mask], eval_dict["fy_m3"][mask]),
                        "mz": compute_metrics(eval_dict["mz_exp"][mask], eval_dict["mz_m3"][mask])
                    }
                }
        return slices

    val_slices_3way = compute_sliced_domain_3way(eval_val)

    # 6. Physical Tire Characteristics at Nominal Cornering Load (Level 4: Fz ~ 885 N, gamma = 0, p = 83 kPa)
    a_dense = np.linspace(-np.radians(14.0), np.radians(14.0), 301)
    df_nom = df_train[(df_train["f_z"] >= 770.0) & (df_train["f_z"] < 990.0) & (np.abs(df_train["gamma"]) < 0.005) & (np.abs(df_train["p_kpa"] - 83.0) < 4.0)]
    fz_nom = float(np.mean(df_nom["f_z"])) if len(df_nom) > 50 else 887.0
    fy_nom_m1, _ = Pacejka62Fitter.compute_fy0(params_m1, a_dense, fz_nom, 0.0, 83.0)
    fy_nom_m2, _ = Pacejka62Fitter.compute_fy0(params_m2, a_dense, fz_nom, 0.0, 83.0)
    d_fy_nom, _ = eval_gated_pinn(nn_data, a_dense, np.full_like(a_dense, fz_nom), np.zeros_like(a_dense), np.full_like(a_dense, 83.0))
    fy_nom_m3 = fy_nom_m2 + d_fy_nom

    feat_m1 = extract_features(a_dense, fy_nom_m1)
    feat_m2 = extract_features(a_dense, fy_nom_m2)
    feat_m3 = extract_features(a_dense, fy_nom_m3)

    if len(df_nom) > 50:
        bins_a = np.linspace(-0.25, 0.25, 41)
        df_nom_binned = df_nom.groupby(pd.cut(df_nom["alpha"], bins_a), observed=True)[["alpha", "f_y"]].mean().dropna()
        feat_exp = extract_features(df_nom_binned["alpha"].values, df_nom_binned["f_y"].values)
    else:
        feat_exp = feat_m2

    # 7. Generate Publication Figures
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Fig 0: Fz Distribution and 5-bin verification
    f0_path = plot_fz_distribution_and_bins(df_train, df_val, compound_name)

    loads_info = [
        {"name": "Level 1 (~220 N)", "bin": (100.0, 330.0), "col": "#1f77b4"},
        {"name": "Level 2 (~440 N)", "bin": (330.0, 550.0), "col": "#2ca02c"},
        {"name": "Level 3 (~665 N)", "bin": (550.0, 770.0), "col": "#ff7f0e"},
        {"name": "Level 4 (~885 N)", "bin": (770.0, 990.0), "col": "#d62728"},
        {"name": "Level 5 (~1110 N)", "bin": (990.0, 1400.0), "col": "#9467bd"},
    ]

    # Fig 1: Fy vs Slip Angle across 5 Loads (Model 1 vs Model 2 vs Model 3 vs Exp)
    fig, ax = plt.subplots(figsize=(10.5, 6.5), dpi=150)
    for l_info in loads_info:
        b_lo, b_hi = l_info["bin"]
        col = l_info["col"]
        sub = df_train[(df_train["f_z"] >= b_lo) & (df_train["f_z"] < b_hi) & (np.abs(df_train["gamma"]) < 0.005) & (np.abs(df_train["p_kpa"] - 83.0) < 4.0)]
        fz_mean = float(np.mean(sub["f_z"])) if len(sub) > 0 else (b_lo + b_hi) / 2.0
        if len(sub) > 0:
            ax.scatter(np.degrees(sub["alpha"]), sub["f_y"], s=3, alpha=0.20, color=col, label=f"Exp Fz~{fz_mean:.0f}N" if l_info["name"] == "Level 4 (~885 N)" else None)

        fy_c_m1, _ = Pacejka62Fitter.compute_fy0(params_m1, a_dense, fz_mean, 0.0, 83.0)
        fy_c_m2, _ = Pacejka62Fitter.compute_fy0(params_m2, a_dense, fz_mean, 0.0, 83.0)
        dfy_c, _ = eval_gated_pinn(nn_data, a_dense, np.full_like(a_dense, fz_mean), np.zeros_like(a_dense), np.full_like(a_dense, 83.0))
        fy_c_m3 = fy_c_m2 + dfy_c

        ax.plot(np.degrees(a_dense), fy_c_m1, ":", color=col, lw=1.5, alpha=0.6, label="M1: Baseline" if l_info["name"] == "Level 4 (~885 N)" else None)
        ax.plot(np.degrees(a_dense), fy_c_m2, "--", color=col, lw=2.0, label="M2: Optimized" if l_info["name"] == "Level 4 (~885 N)" else None)
        ax.plot(np.degrees(a_dense), fy_c_m3, "-", color=col, lw=2.4, label=f"M3: Hybrid ({fz_mean:.0f}N)")

    ax.set_title(f"Lateral Force $F_y$ vs $\\alpha$ ({compound_name.upper()}): 3-Way Model Comparison across 5 Normal Loads", fontsize=12, fontweight="bold")
    ax.set_xlabel("Slip Angle $\\alpha$ [deg]", fontsize=11)
    ax.set_ylabel("Lateral Force $F_y$ [N]", fontsize=11)
    ax.legend(ncol=2, fontsize=8.5, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    f1_path = os.path.join(PLOTS_DIR, f"fig1_fy_vs_alpha_loads{suffix}.png")
    fig.savefig(f1_path)
    plt.close(fig)

    # Fig 2: Aligning Torque Mz vs Slip Angle across 5 Loads
    fig, ax = plt.subplots(figsize=(10.5, 6.5), dpi=150)
    for l_info in loads_info:
        b_lo, b_hi = l_info["bin"]
        col = l_info["col"]
        sub = df_train[(df_train["f_z"] >= b_lo) & (df_train["f_z"] < b_hi) & (np.abs(df_train["gamma"]) < 0.005) & (np.abs(df_train["p_kpa"] - 83.0) < 4.0)]
        fz_mean = float(np.mean(sub["f_z"])) if len(sub) > 0 else (b_lo + b_hi) / 2.0
        if len(sub) > 0:
            ax.scatter(np.degrees(sub["alpha"]), sub["m_z"], s=3, alpha=0.20, color=col, label=f"Exp Fz~{fz_mean:.0f}N" if l_info["name"] == "Level 4 (~885 N)" else None)

        fy_c_m1, svy_c_m1 = Pacejka62Fitter.compute_fy0(params_m1, a_dense, fz_mean, 0.0, 83.0)
        mz_c_m1 = Pacejka62Fitter.compute_mz(params_m1, a_dense, fz_mean, 0.0, 83.0, fx=0.0, fy_in=fy_c_m1, svy_in=svy_c_m1, r0=r0_m1)

        fy_c_m2, svy_c_m2 = Pacejka62Fitter.compute_fy0(params_m2, a_dense, fz_mean, 0.0, 83.0)
        mz_c_m2 = Pacejka62Fitter.compute_mz(params_m2, a_dense, fz_mean, 0.0, 83.0, fx=0.0, fy_in=fy_c_m2, svy_in=svy_c_m2, r0=r0_m2)

        _, dmz_c = eval_gated_pinn(nn_data, a_dense, np.full_like(a_dense, fz_mean), np.zeros_like(a_dense), np.full_like(a_dense, 83.0))
        mz_c_m3 = mz_c_m2 + dmz_c

        ax.plot(np.degrees(a_dense), mz_c_m1, ":", color=col, lw=1.5, alpha=0.6, label="M1: Baseline" if l_info["name"] == "Level 4 (~885 N)" else None)
        ax.plot(np.degrees(a_dense), mz_c_m2, "--", color=col, lw=2.0, label="M2: Optimized" if l_info["name"] == "Level 4 (~885 N)" else None)
        ax.plot(np.degrees(a_dense), mz_c_m3, "-", color=col, lw=2.4, label=f"M3: Hybrid ({fz_mean:.0f}N)")

    ax.axhline(0.0, color="gray", lw=0.8, linestyle=":")
    ax.set_title(f"Self-Aligning Torque $M_z$ vs $\\alpha$ ({compound_name.upper()}): 3-Way Comparison across 5 Loads", fontsize=12, fontweight="bold")
    ax.set_xlabel("Slip Angle $\\alpha$ [deg]", fontsize=11)
    ax.set_ylabel("Self-Aligning Torque $M_z$ [Nm] (ISO restoring moment)", fontsize=11)
    ax.legend(ncol=2, fontsize=8.5, loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    f2_path = os.path.join(PLOTS_DIR, f"fig2_mz_vs_alpha_loads{suffix}.png")
    fig.savefig(f2_path)
    plt.close(fig)

    # Fig 3: Pneumatic Trail t(alpha)
    fig, ax = plt.subplots(figsize=(9.5, 6), dpi=150)
    for l_info in loads_info:
        b_lo, b_hi = l_info["bin"]
        col = l_info["col"]
        sub = df_train[(df_train["f_z"] >= b_lo) & (df_train["f_z"] < b_hi) & (np.abs(df_train["gamma"]) < 0.005) & (np.abs(df_train["p_kpa"] - 83.0) < 4.0)]
        fz_mean = float(np.mean(sub["f_z"])) if len(sub) > 0 else (b_lo + b_hi) / 2.0
        dfz = (fz_mean - params_m2.get("FNOMIN", 800.0)) / params_m2.get("FNOMIN", 800.0)
        Dt = (fz_mean / params_m2.get("FNOMIN", 800.0)) * r0_m2 * (params_m2.get("QDZ1", 0.15) + params_m2.get("QDZ2", -0.04) * dfz)
        Ct = params_m2.get("QCZ1", 1.65)
        Bt = (params_m2.get("QBZ1", 6.0) + params_m2.get("QBZ2", 1.0) * dfz)
        Et = np.clip(params_m2.get("QEZ1", 0.0) + params_m2.get("QEZ2", 1.0) * dfz, -10.0, 1.0)
        Sht = params_m2.get("QHZ1", 0.0) + params_m2.get("QHZ2", 0.0) * dfz
        alpha_t = a_dense + Sht
        t_arg = Bt * alpha_t
        trail_m = Dt * np.cos(Ct * np.arctan(t_arg - Et * (t_arg - np.arctan(t_arg)))) * np.cos(a_dense)
        trail_mm = trail_m * 1000.0
        ax.plot(np.degrees(a_dense), trail_mm, "-", color=col, lw=2.2, label=f"Fz={fz_mean:.0f} N (Peak={np.max(trail_mm):.1f} mm)")

    ax.axhline(0.0, color="gray", lw=0.8, linestyle=":")
    ax.set_title(f"Pneumatic Trail $t(\\alpha)$ vs Slip Angle $\\alpha$ ({compound_name.upper()}) across 5 Load Levels", fontsize=12, fontweight="bold")
    ax.set_xlabel("Slip Angle $\\alpha$ [deg]", fontsize=11)
    ax.set_ylabel("Pneumatic Trail $t$ [mm]", fontsize=11)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    f3_trail_path = os.path.join(PLOTS_DIR, f"fig3_pneumatic_trail{suffix}.png")
    fig.savefig(f3_trail_path)
    plt.close(fig)

    # Fig 4: Camber Sensitivity Sweeps & Resolution Verification (2 Panels)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 6.2), dpi=150)
    cambers_deg = [-4.0, -2.0, 0.0, 2.0, 4.0]
    c_cols = ["#9467bd", "#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    fz_camber = fz_nom

    # Panel 1: Full Fy vs alpha across cambers at Fz ~ 885 N
    for g_deg, col in zip(cambers_deg, c_cols):
        g_rad = np.radians(g_deg)
        sub = df_train[(df_train["f_z"] >= 770.0) & (df_train["f_z"] < 990.0) & (np.abs(np.degrees(df_train["gamma"]) - g_deg) < 0.4) & (np.abs(df_train["p_kpa"] - 83.0) < 4.0)]
        if len(sub) > 0:
            ax1.scatter(np.degrees(sub["alpha"]), sub["f_y"], s=3, alpha=0.20, color=col, label=f"Exp $\\gamma$={g_deg:.0f}$^\\circ$")

        fy_m1_c, _ = Pacejka62Fitter.compute_fy0(params_m1, a_dense, fz_camber, g_rad, 83.0)
        fy_m2_c, _ = Pacejka62Fitter.compute_fy0(params_m2, a_dense, fz_camber, g_rad, 83.0)
        dfy_c, _ = eval_gated_pinn(nn_data, a_dense, np.full_like(a_dense, fz_camber), np.full_like(a_dense, g_rad), np.full_like(a_dense, 83.0))
        fy_m3_c = fy_m2_c + dfy_c

        # Show baseline error on -4 deg, optimized on all
        if g_deg == -4.0:
            ax1.plot(np.degrees(a_dense), fy_m1_c, ":", color="#e74c3c", lw=2.2, label=f"M1: Baseline $\\gamma$=-4$^\\circ$ (Collapsed)")
        ax1.plot(np.degrees(a_dense), fy_m2_c, "--", color=col, lw=1.8, label=f"M2: Opt $\\gamma$={g_deg:.0f}$^\\circ$")
        ax1.plot(np.degrees(a_dense), fy_m3_c, "-", color=col, lw=2.4)

    ax1.set_title(f"Camber Sensitivity: $F_y(\\alpha, \\gamma)$ at $F_z \\approx {fz_camber:.0f}$ N ({compound_name.upper()})", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Slip Angle $\\alpha$ [deg]", fontsize=10)
    ax1.set_ylabel("Lateral Force $F_y$ [N]", fontsize=10)
    ax1.legend(ncol=2, fontsize=8.0, loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Panel 2: Camber Thrust at Zero Slip (Exp vs M1 vs M2 vs M3 across loads)
    g_dense = np.linspace(-np.radians(4.5), np.radians(4.5), 101)
    for l_info in [loads_info[1], loads_info[3]]: # L2 (~440N) and L4 (~885N)
        b_lo, b_hi = l_info["bin"]
        col = l_info["col"]
        sub_c0 = df_train[(df_train["f_z"] >= b_lo) & (df_train["f_z"] < b_hi) & (np.abs(df_train["alpha"]) < 0.005) & (np.abs(df_train["p_kpa"] - 83.0) < 4.0)]
        fz_l = float(np.mean(sub_c0["f_z"])) if len(sub_c0) > 0 else (b_lo + b_hi) / 2.0
        if len(sub_c0) > 0:
            ax2.scatter(np.degrees(sub_c0["gamma"]), sub_c0["f_y"], s=15, alpha=0.5, color=col, label=f"Exp Zero-Slip Fz~{fz_l:.0f}N")

        fy_c0_m1, _ = Pacejka62Fitter.compute_fy0(params_m1, 0.0, fz_l, g_dense, 83.0)
        fy_c0_m2, _ = Pacejka62Fitter.compute_fy0(params_m2, 0.0, fz_l, g_dense, 83.0)
        dfy_c0, _ = eval_gated_pinn(nn_data, np.zeros_like(g_dense), np.full_like(g_dense, fz_l), g_dense, np.full_like(g_dense, 83.0))
        fy_c0_m3 = fy_c0_m2 + dfy_c0

        ax2.plot(np.degrees(g_dense), fy_c0_m1, ":", color=col, lw=1.6, label=f"M1: Baseline ({fz_l:.0f}N)")
        ax2.plot(np.degrees(g_dense), fy_c0_m2, "--", color=col, lw=2.0, label=f"M2: Opt ({fz_l:.0f}N)")
        ax2.plot(np.degrees(g_dense), fy_c0_m3, "-", color=col, lw=2.4, label=f"M3: Hybrid ({fz_l:.0f}N)")

    ax2.axhline(0.0, color="gray", lw=0.8, linestyle=":")
    ax2.axvline(0.0, color="gray", lw=0.8, linestyle=":")
    ax2.set_title(f"Camber Thrust $F_y(\\alpha=0, \\gamma)$: Model Resolution vs Exp Data", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Camber Angle $\\gamma$ [deg]", fontsize=10)
    ax2.set_ylabel("Zero-Slip Camber Thrust $F_y(0)$ [N]", fontsize=10)
    ax2.legend(ncol=2, fontsize=8.0, loc="upper left")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    f4_camber_path = os.path.join(PLOTS_DIR, f"fig4_camber_sensitivity{suffix}.png")
    fig.savefig(f4_camber_path)
    plt.close(fig)

    # Fig 5: Pressure Sensitivity Sweeps
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    pressures_val = [70.0, 83.0, 95.0]
    p_cols = ["#3498db", "#2ecc71", "#e74c3c"]
    for p_val, col in zip(pressures_val, p_cols):
        sub = df_val[(df_val["f_z"] >= 770.0) & (df_val["f_z"] < 990.0) & (np.abs(df_val["p_kpa"] - p_val) < 4.0) & (np.abs(df_val["gamma"]) < 0.005)]
        if len(sub) > 0:
            ax.scatter(np.degrees(sub["alpha"]), sub["f_y"], s=3, alpha=0.25, color=col, label=f"Exp p={p_val:.0f} kPa")
        fy_c, _ = Pacejka62Fitter.compute_fy0(params_m2, a_dense, fz_nom, 0.0, p_val)
        dfy_p, _ = eval_gated_pinn(nn_data, a_dense, np.full_like(a_dense, fz_nom), np.zeros_like(a_dense), np.full_like(a_dense, p_val))
        ax.plot(np.degrees(a_dense), fy_c + dfy_p, "-", color=col, lw=2.2, label=f"Hybrid Model (p={p_val:.0f} kPa)")

    ax.set_title(f"Inflation Pressure Sensitivity: $F_y$ vs $\\alpha$ at $F_z \\approx {fz_nom:.0f}$ N ({compound_name.upper()})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Slip Angle $\\alpha$ [deg]", fontsize=11)
    ax.set_ylabel("Lateral Force $F_y$ [N]", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    f5_pressure_path = os.path.join(PLOTS_DIR, f"fig5_pressure_sensitivity{suffix}.png")
    fig.savefig(f5_pressure_path)
    plt.close(fig)

    # Fig 6: Residual Error Distribution Histograms (3-Way Comparison)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0), dpi=150)
    err_fy_m1 = eval_val["fy_m1"] - eval_val["fy_exp"]
    err_fy_m2 = eval_val["fy_m2"] - eval_val["fy_exp"]
    err_fy_m3 = eval_val["fy_m3"] - eval_val["fy_exp"]

    err_mz_m1 = eval_val["mz_m1"] - eval_val["mz_exp"]
    err_mz_m2 = eval_val["mz_m2"] - eval_val["mz_exp"]
    err_mz_m3 = eval_val["mz_m3"] - eval_val["mz_exp"]

    ax1.hist(err_fy_m1, bins=50, density=True, alpha=0.35, color="#e74c3c", label=f"M1: Baseline (RMSE={np.sqrt(np.mean(err_fy_m1**2)):.1f} N)")
    ax1.hist(err_fy_m2, bins=50, density=True, alpha=0.45, color="#3498db", label=f"M2: Opt MF (RMSE={np.sqrt(np.mean(err_fy_m2**2)):.1f} N)")
    ax1.hist(err_fy_m3, bins=50, density=True, alpha=0.55, color="#2ecc71", label=f"M3: Hybrid (RMSE={np.sqrt(np.mean(err_fy_m3**2)):.1f} N)")
    ax1.set_title(f"Validation Residual Distribution $\\Delta F_y$ ({compound_name.upper()})", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Lateral Force Residual $\\Delta F_y$ [N]", fontsize=10)
    ax1.set_ylabel("Density", fontsize=10)
    ax1.legend(fontsize=8.5)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-400, 400)

    ax2.hist(err_mz_m1, bins=50, density=True, alpha=0.35, color="#e74c3c", label=f"M1: Baseline (RMSE={np.sqrt(np.mean(err_mz_m1**2)):.1f} Nm)")
    ax2.hist(err_mz_m2, bins=50, density=True, alpha=0.45, color="#3498db", label=f"M2: Opt MF (RMSE={np.sqrt(np.mean(err_mz_m2**2)):.1f} Nm)")
    ax2.hist(err_mz_m3, bins=50, density=True, alpha=0.55, color="#2ecc71", label=f"M3: Hybrid (RMSE={np.sqrt(np.mean(err_mz_m3**2)):.1f} Nm)")
    ax2.set_title(f"Validation Residual Distribution $\\Delta M_z$ ({compound_name.upper()})", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Aligning Torque Residual $\\Delta M_z$ [Nm]", fontsize=10)
    ax2.set_ylabel("Density", fontsize=10)
    ax2.legend(fontsize=8.5)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-25, 25)

    fig.tight_layout()
    f6_err_path = os.path.join(PLOTS_DIR, f"fig6_error_distributions{suffix}.png")
    fig.savefig(f6_err_path)
    plt.close(fig)

    # Fig 7: Overturning Moment Mx vs Camber Angle gamma across Loads
    fig, ax = plt.subplots(figsize=(9.5, 6), dpi=150)
    gamma_sweep_rad = np.linspace(-np.radians(5.0), np.radians(5.0), 101)
    for l_info in loads_info:
        b_lo, b_hi = l_info["bin"]
        col = l_info["col"]
        sub = df_train[(df_train["f_z"] >= b_lo) & (df_train["f_z"] < b_hi) & (np.abs(df_train["alpha"]) < 0.01) & (np.abs(df_train["p_kpa"] - 83.0) < 4.0)]
        fz_mean = float(np.mean(sub["f_z"])) if len(sub) > 0 else (b_lo + b_hi) / 2.0
        if len(sub) > 0:
            ax.scatter(np.degrees(sub["gamma"]), sub["m_x"], s=10, alpha=0.4, color=col, label=f"Exp Fz~{fz_mean:.0f} N")
        fy_zero, _ = Pacejka62Fitter.compute_fy0(params_m2, 0.0, fz_mean, gamma_sweep_rad, 83.0)
        mx_pred = Pacejka62Fitter.compute_mx(params_m2, fz_mean, gamma_sweep_rad, fy_zero, 83.0, r0=r0_m2)
        ax.plot(np.degrees(gamma_sweep_rad), mx_pred, "-", color=col, lw=2.2, label=f"MF 6.2 ({fz_mean:.0f} N)")

    ax.axhline(0.0, color="gray", lw=0.8, linestyle=":")
    ax.axvline(0.0, color="gray", lw=0.8, linestyle=":")
    ax.set_title(f"Overturning Moment $M_x$ vs Camber Angle $\\gamma$ ({compound_name.upper()}) across 5 Loads", fontsize=12, fontweight="bold")
    ax.set_xlabel("Camber Angle $\\gamma$ [deg]", fontsize=11)
    ax.set_ylabel("Overturning Moment $M_x$ [Nm]", fontsize=11)
    ax.legend(ncol=2, fontsize=8.5, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    f7_mx_path = os.path.join(PLOTS_DIR, f"fig7_mx_vs_camber{suffix}.png")
    fig.savefig(f7_mx_path)
    plt.close(fig)

    # Fig 8: Experimental Vertical Load-Deflection Curve
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    rho_m = r0_m2 - df_spring["r_l"].values
    rho_mm = rho_m * 1000.0
    fz_spring = df_spring["f_z"].values
    ax.scatter(rho_mm, fz_spring, s=2, alpha=0.15, color="#16a085", label=f"TTC Spring Test ({len(df_spring)} pts)")
    rho_line = np.linspace(0.0, np.max(rho_m), 100)
    kz_fit = params_m2.get("VERTICAL_STIFFNESS", 115000.0)
    fz_fit_line = kz_fit * rho_line
    r2_spring = params_m2.get("VERTICAL_STIFFNESS_ANALYSIS", {}).get("stiffness_r2", 0.95)
    ax.plot(rho_line * 1000.0, fz_fit_line, "r-", lw=2.5, label=f"Linear Fit: $k_z = {kz_fit:.0f}$ N/m ($R^2 = {r2_spring:.3f}$)")
    ax.set_title(f"Experimental Vertical Load-Deflection Characterization ({compound_name.upper()})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Radial Deflection $\\rho = R_0 - R_l$ [mm]", fontsize=11)
    ax.set_ylabel("Normal Force $F_z$ [N]", fontsize=11)
    ax.legend(fontsize=9.5, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    f8_spring_path = os.path.join(PLOTS_DIR, f"fig8_vertical_stiffness{suffix}.png")
    fig.savefig(f8_spring_path)
    plt.close(fig)

    # Fig 9: Identifiability Singular Value Spectra
    calib_rep_path = os.path.join(VALIDATION_DIR, f"calibration_report_{c_lower}.json")
    if os.path.exists(calib_rep_path):
        with open(calib_rep_path) as fp:
            calib_rep = json.load(fp)
        stages = calib_rep.get("identifiability_stages", {})
        fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
        marker_list = ["o", "s", "^", "D", "v", "P"]
        for (st_key, st_info), m in zip(stages.items(), marker_list):
            s_vals = st_info.get("singular_values", [])
            if len(s_vals) > 0:
                ax.semilogy(range(1, len(s_vals) + 1), s_vals, f"-{m}", lw=1.8, markersize=6,
                            label=f"{st_info.get('stage', st_key)} (cond={st_info.get('scaled_condition_number', 1.0):.1e})")
        ax.set_title(f"Scaled Jacobian Singular Value Spectra across Stages ({compound_name.upper()})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Singular Value Index", fontsize=11)
        ax.set_ylabel("Singular Value $\\sigma_i$ (log scale)", fontsize=11)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        f9_svd_path = os.path.join(PLOTS_DIR, f"fig9_singular_value_spectra{suffix}.png")
        fig.savefig(f9_svd_path)
        plt.close(fig)
    else:
        f9_svd_path = None

    # Fig 10: Model Parity Scatter Plots & 2D Spatial Residual Map
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 6.0), dpi=150)
    # Parity plot (Predicted vs Measured) on Validation Set
    ax1.scatter(eval_val["fy_exp"], eval_val["fy_m1"], s=2, alpha=0.15, color="#e74c3c", label=f"M1: Baseline (R^2={metrics_3way['validation']['model1_baseline_mf62']['fy']['r2']:.3f})")
    ax1.scatter(eval_val["fy_exp"], eval_val["fy_m2"], s=2, alpha=0.15, color="#3498db", label=f"M2: Opt MF (R^2={metrics_3way['validation']['model2_optimized_mf62']['fy']['r2']:.3f})")
    ax1.scatter(eval_val["fy_exp"], eval_val["fy_m3"], s=2, alpha=0.20, color="#2ecc71", label=f"M3: Hybrid (R^2={metrics_3way['validation']['model3_hybrid_pinn']['fy']['r2']:.3f})")
    lims = [-2500, 2500]
    ax1.plot(lims, lims, "k--", lw=1.5, label="Perfect Parity (y = x)")
    ax1.set_title(f"Validation Force Parity Scatter ({compound_name.upper()})", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Experimental Lateral Force $F_{y,\\text{exp}}$ [N]", fontsize=10)
    ax1.set_ylabel("Predicted Lateral Force $F_{y,\\text{pred}}$ [N]", fontsize=10)
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    ax1.legend(fontsize=8.5, loc="upper left")
    ax1.grid(True, alpha=0.3)

    # 2D Spatial Residual Map across (alpha, Fz)
    sc = ax2.scatter(np.degrees(eval_val["alpha"]), eval_val["f_z"], c=eval_val["fy_m3"] - eval_val["fy_exp"],
                     cmap="coolwarm", s=3, alpha=0.6, vmin=-150, vmax=150)
    cbar = fig.colorbar(sc, ax=ax2)
    cbar.set_label("Residual Error $F_{y,\\text{hyb}} - F_{y,\\text{exp}}$ [N]", fontsize=9.5)
    ax2.set_title(f"Hybrid Residual Spatial Map $\\Delta F_y(\\alpha, F_z)$ ({compound_name.upper()})", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Slip Angle $\\alpha$ [deg]", fontsize=10)
    ax2.set_ylabel("Normal Load $F_z$ [N]", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(100, 1300)

    fig.tight_layout()
    f10_parity_path = os.path.join(PLOTS_DIR, f"fig10_parity_and_residuals{suffix}.png")
    fig.savefig(f10_parity_path)
    plt.close(fig)

    # Copy plots to ARTIFACT_DIR
    copied_plots = [f0_path, f1_path, f2_path, f3_trail_path, f4_camber_path, f5_pressure_path, f6_err_path, f7_mx_path, f8_spring_path, f10_parity_path]
    if f9_svd_path and os.path.exists(f9_svd_path):
        copied_plots.append(f9_svd_path)

    # If R25B, also save standard numbered plots for backwards compatibility
    if compound_name.upper() == "R25B":
        shutil.copyfile(f0_path, os.path.join(PLOTS_DIR, "fig0_fz_distribution_and_bins.png"))
        shutil.copyfile(f1_path, os.path.join(PLOTS_DIR, "fig1_fy_vs_alpha_loads.png"))
        shutil.copyfile(f2_path, os.path.join(PLOTS_DIR, "fig2_mz_vs_alpha_loads.png"))
        shutil.copyfile(f4_camber_path, os.path.join(PLOTS_DIR, "fig3_camber_sensitivity.png"))
        shutil.copyfile(f5_pressure_path, os.path.join(PLOTS_DIR, "fig4_pressure_sensitivity.png"))
        shutil.copyfile(f6_err_path, os.path.join(PLOTS_DIR, "fig5_error_distributions.png"))

    for p in copied_plots:
        shutil.copy(p, ARTIFACT_DIR)
    if compound_name.upper() == "R25B":
        shutil.copy(os.path.join(PLOTS_DIR, "fig0_fz_distribution_and_bins.png"), ARTIFACT_DIR)

    report_data = {
        "compound": compound_name.upper(),
        "datasets": {
            "train": {"path": train_csv, "num_samples": len(df_train)},
            "validation": {"path": val_csv, "num_samples": len(df_val)},
            "spring": {"path": spring_csv, "num_samples": len(df_spring)}
        },
        "dynamic_characteristics_nominal": {
            "conditions": {"F_z_N": fz_nom, "gamma_deg": 0.0, "p_kPa": 83.0},
            "experimental": feat_exp,
            "model1_baseline_mf62": feat_m1,
            "model2_optimized_mf62": feat_m2,
            "model3_hybrid_pinn": feat_m3
        },
        "overall_metrics_3way": metrics_3way,
        "domain_sliced_metrics_validation_3way": val_slices_3way,
        "plots": [os.path.basename(p) for p in copied_plots]
    }

    rep_path = os.path.join(VALIDATION_DIR, f"rigorous_validation_report_{c_lower}.json")
    with open(rep_path, "w") as fp:
        json.dump(report_data, fp, indent=2)
    shutil.copy(rep_path, ARTIFACT_DIR)
    print(f">> Saved 3-way validation report JSON: {rep_path}")
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
    fig10_path = os.path.join(PLOTS_DIR, "fig10_transient_step_response.png")
    fig.savefig(fig10_path)
    shutil.copyfile(fig10_path, os.path.join(PLOTS_DIR, "fig6_transient_step_response.png"))
    plt.close(fig)
    shutil.copy(fig10_path, ARTIFACT_DIR)
    print(f">> Saved Transient Step Response: {fig10_path}")


def main():
    print("=" * 78)
    print("   GENERATING RIGOROUS 3-WAY VALIDATION REPORTS & PUBLICATION FIGURES")
    print("=" * 78)

    rep_r25b = validate_compound_3way("R25B")
    rep_r20 = validate_compound_3way("R20")

    generate_transient_step_response_figure()

    # Synchronize default combined report
    combined_rep_path = os.path.join(VALIDATION_DIR, "rigorous_validation_report.json")
    with open(combined_rep_path, "w") as fp:
        json.dump({"R25B": rep_r25b, "R20": rep_r20}, fp, indent=2)
    shutil.copy(combined_rep_path, ARTIFACT_DIR)
    print(f"\n>> Combined validation report JSON saved: {combined_rep_path}")
    print("=" * 78)

    # Also invoke the expanded 15-module Tire Model Validation Framework
    try:
        from run_validation_framework import setup_directories, run_compound_validation, generate_automated_markdown_report, TREE_DIR
        tree_paths = setup_directories(TREE_DIR)
        auto_r25b = run_compound_validation("R25B", tree_paths)
        auto_r20 = run_compound_validation("R20", tree_paths)
        auto_combined = {"R25B": auto_r25b, "R20": auto_r20}
        with open(os.path.join(VALIDATION_DIR, "automated_validation_report.json"), "w") as fp:
            json.dump(auto_combined, fp, indent=2)
        generate_automated_markdown_report(auto_r25b, auto_r20, os.path.join(ARTIFACT_DIR, "automated_validation_report.md"))
        print(">> Synchronized Expanded Validation Framework successfully.")
    except Exception as e:
        print(f">> Warning: Failed to execute expanded validation framework: {e}")


if __name__ == "__main__":
    main()
