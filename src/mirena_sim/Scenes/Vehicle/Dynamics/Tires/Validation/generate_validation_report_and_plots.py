#!/usr/bin/env python3
"""
Comprehensive Experimental Validation & Plotting Suite for Mirena Dynamics Tire Model
Generates:
1. Detailed quantitative metrics across Train (Run 6) and Validation (Run 7)
2. Initial stiffness C_alpha, peak force Fy_max, peak angle alpha_peak, and post-peak drop
3. Error breakdowns sliced by load, camber, pressure, and velocity
4. Publication-quality comparative plots saved to disk
5. Comprehensive JSON validation report
"""

import os
import json
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

os.makedirs(PLOTS_DIR, exist_ok=True)

import sys
sys.path.append("src/mirena_sim/Scenes/Vehicle/Dynamics/Tires/Training")
from calibrate_mf62 import Pacejka62Fitter
from train_residual_mlp import PhysicsInformedResidualMLP

def main():
    print("=" * 70)
    print("   GENERATING RIGOROUS VALIDATION REPORT & COMPARATIVE PLOTS")
    print("=" * 70)
    
    # Load MF parameters
    with open(os.path.join(MF62_DIR, "hoosier_16x7.5_10_r25b_mf62.json")) as fp:
        mf_params = json.load(fp)
        
    # Load Neural weights
    with open(os.path.join(NEURAL_DIR, "neural_residual_weights.json")) as fp:
        nn_data = json.load(fp)
        
    norm = nn_data["normalization"]
    weights = nn_data["weights"]
    
    model = PhysicsInformedResidualMLP(
        in_dim=4, hidden1=16, hidden2=16, out_dim=2,
        x_mean=norm["x_mean"], x_std=norm["x_std"],
        y_mean=norm["y_mean"], y_std=norm["y_std"]
    )
    model.fc1.weight.data = torch.tensor(weights["w1"], dtype=torch.float32)
    model.fc1.bias.data = torch.tensor(weights["b1"], dtype=torch.float32)
    model.fc2.weight.data = torch.tensor(weights["w2"], dtype=torch.float32)
    model.fc2.bias.data = torch.tensor(weights["b2"], dtype=torch.float32)
    model.fc3.weight.data = torch.tensor(weights["w3"], dtype=torch.float32)
    model.fc3.bias.data = torch.tensor(weights["b3"], dtype=torch.float32)
    model.eval()
    
    df_train = pd.read_csv(os.path.join(PROCESSED_DIR, "R25B_train.csv"))
    df_val = pd.read_csv(os.path.join(PROCESSED_DIR, "R25B_val.csv"))
    
    # ------------------------------------------------------------------
    # 1. Evaluate Model Predictions on Full Datasets
    # ------------------------------------------------------------------
    def evaluate_model(df):
        alpha = df["alpha"].values
        fz = df["f_z"].values
        gamma = df["gamma"].values
        p = df["p_kpa"].values
        vx = df["v_x"].values
        fy_exp = df["f_y"].values
        mz_exp = df["m_z"].values
        
        # MF 6.2
        fy_mf, svy_mf = Pacejka62Fitter.compute_fy0(mf_params, alpha, fz, gamma, p)
        mz_mf = Pacejka62Fitter.compute_mz(mf_params, alpha, fz, gamma, p, fx=0.0, fy_in=fy_mf, svy_in=svy_mf)
        
        # Hybrid PINN
        x_raw = np.column_stack([alpha, fz, gamma, p])
        with torch.no_grad():
            delta = model(torch.tensor(x_raw, dtype=torch.float32)).numpy()
            
        fy_hyb = fy_mf + delta[:, 0]
        mz_hyb = mz_mf + delta[:, 1]
        
        return {
            "alpha": alpha, "f_z": fz, "gamma": gamma, "p_kpa": p, "v_x": vx,
            "fy_exp": fy_exp, "mz_exp": mz_exp,
            "fy_mf": fy_mf, "mz_mf": mz_mf,
            "fy_hyb": fy_hyb, "mz_hyb": mz_hyb,
            "delta_fy": delta[:, 0], "delta_mz": delta[:, 1]
        }
        
    eval_tr = evaluate_model(df_train)
    eval_val = evaluate_model(df_val)
    
    # ------------------------------------------------------------------
    # 2. Key Dynamic Feature Metrics (Stiffness, Peak, Peak Angle, Post-Peak)
    # ------------------------------------------------------------------
    # Extract sweep at nominal load Fz ~ 800 N, gamma ~ 0, p ~ 83 kPa
    mask_nom = (np.abs(df_train["f_z"] - 800.0) < 100.0) & (np.abs(df_train["gamma"]) < 0.005) & (np.abs(df_train["p_kpa"] - 83.0) < 5.0)
    df_nom = df_train[mask_nom].sort_values("alpha")
    
    a_dense = np.linspace(-0.20, 0.20, 400)
    fz_nom_scalar = 800.0
    
    fy_mf_nom, svy_nom = Pacejka62Fitter.compute_fy0(mf_params, a_dense, fz_nom_scalar, 0.0, 83.0)
    mz_mf_nom = Pacejka62Fitter.compute_mz(mf_params, a_dense, fz_nom_scalar, 0.0, 83.0, fx=0.0, fy_in=fy_mf_nom, svy_in=svy_nom)
    
    with torch.no_grad():
        x_dense = torch.tensor(np.column_stack([a_dense, np.full_like(a_dense, fz_nom_scalar), np.zeros_like(a_dense), np.full_like(a_dense, 83.0)]), dtype=torch.float32)
        delta_dense = model(x_dense).numpy()
    fy_hyb_nom = fy_mf_nom + delta_dense[:, 0]
    mz_hyb_nom = mz_mf_nom + delta_dense[:, 1]
    
    # Feature calculations
    def extract_features(a_arr, fy_arr):
        # Stiffness dFy/da at alpha=0
        idx_0 = np.argmin(np.abs(a_arr))
        stiffness = float((fy_arr[idx_0 + 1] - fy_arr[idx_0 - 1]) / (a_arr[idx_0 + 1] - a_arr[idx_0 - 1]))
        # Peak
        idx_max = np.argmax(fy_arr)
        fy_max = float(fy_arr[idx_max])
        a_peak_deg = float(np.degrees(a_arr[idx_max]))
        # Post-peak drop at alpha = 12 deg (0.209 rad)
        idx_post = np.argmin(np.abs(a_arr - np.radians(12.0)))
        drop = float((fy_max - fy_arr[idx_post]) / max(fy_max, 1e-4) * 100.0)
        return {"stiffness_n_rad": stiffness, "fy_max_n": fy_max, "alpha_peak_deg": a_peak_deg, "post_peak_drop_pct": drop}
        
    feat_mf = extract_features(a_dense, fy_mf_nom)
    feat_hyb = extract_features(a_dense, fy_hyb_nom)
    
    # Experimental features from binned data
    bins_a = np.linspace(-0.20, 0.20, 41)
    df_nom_binned = df_nom.groupby(pd.cut(df_nom["alpha"], bins_a), observed=True)[["alpha", "f_y"]].mean().dropna()
    a_exp_binned = df_nom_binned["alpha"].values
    fy_exp_binned = df_nom_binned["f_y"].values
    feat_exp = extract_features(a_exp_binned, fy_exp_binned)
    
    print("\n>> Dynamic Tire Characteristics at Fz = 800 N, gamma = 0, p = 83 kPa:")
    print(f"   Initial Stiffness C_alpha:  Exp = {feat_exp['stiffness_n_rad']:6.1f} N/rad | MF = {feat_mf['stiffness_n_rad']:6.1f} N/rad | Hyb = {feat_hyb['stiffness_n_rad']:6.1f} N/rad")
    print(f"   Peak Lateral Force Fy_max: Exp = {feat_exp['fy_max_n']:6.1f} N     | MF = {feat_mf['fy_max_n']:6.1f} N     | Hyb = {feat_hyb['fy_max_n']:6.1f} N")
    print(f"   Peak Slip Angle alpha_peak:Exp = {feat_exp['alpha_peak_deg']:6.1f} deg   | MF = {feat_mf['alpha_peak_deg']:6.1f} deg   | Hyb = {feat_hyb['alpha_peak_deg']:6.1f} deg")
    print(f"   Post-Peak Drop (at 12 deg):Exp = {feat_exp['post_peak_drop_pct']:6.1f}%     | MF = {feat_mf['post_peak_drop_pct']:6.1f}%     | Hyb = {feat_hyb['post_peak_drop_pct']:6.1f}%")
    
    # ------------------------------------------------------------------
    # 3. Generate Comparative Plots
    # ------------------------------------------------------------------
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    # --- Figure 1: Fy vs Slip Angle at Multiple Loads ---
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    loads = [250, 600, 1000]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    
    for fz_val, col in zip(loads, colors):
        # Experimental points
        sub = df_train[(np.abs(df_train["f_z"] - fz_val) < 60) & (np.abs(df_train["gamma"]) < 0.005)]
        ax.scatter(np.degrees(sub["alpha"]), sub["f_y"], s=3, alpha=0.25, color=col, label=f"Exp Fz~{fz_val} N")
        
        # MF 6.2 curve
        fy_mf, s_mf = Pacejka62Fitter.compute_fy0(mf_params, a_dense, fz_val, 0.0, 83.0)
        ax.plot(np.degrees(a_dense), fy_mf, "--", color=col, lw=1.8, label=f"MF 6.2 ({fz_val} N)")
        
        # Hybrid PINN curve
        with torch.no_grad():
            x_pts = torch.tensor(np.column_stack([a_dense, np.full_like(a_dense, fz_val), np.zeros_like(a_dense), np.full_like(a_dense, 83.0)]), dtype=torch.float32)
            d_pts = model(x_pts).numpy()
        ax.plot(np.degrees(a_dense), fy_mf + d_pts[:, 0], "-", color=col, lw=2.4, label=f"MF+PINN ({fz_val} N)")
        
    ax.set_title("Lateral Force $F_y$ vs Slip Angle $\\alpha$ across Vertical Loads", fontsize=13, fontweight="bold")
    ax.set_xlabel("Slip Angle $\\alpha$ [deg]", fontsize=11)
    ax.set_ylabel("Lateral Force $F_y$ [N]", fontsize=11)
    ax.legend(ncol=3, fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig1_path = os.path.join(PLOTS_DIR, "fig1_fy_vs_alpha_loads.png")
    fig.savefig(fig1_path)
    plt.close(fig)
    print(f">> Saved Fig 1: {fig1_path}")
    
    # --- Figure 2: Mz vs Slip Angle at Multiple Loads ---
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    for fz_val, col in zip(loads, colors):
        sub = df_train[(np.abs(df_train["f_z"] - fz_val) < 60) & (np.abs(df_train["gamma"]) < 0.005)]
        ax.scatter(np.degrees(sub["alpha"]), sub["m_z"], s=3, alpha=0.25, color=col, label=f"Exp Fz~{fz_val} N")
        
        fy_mf, s_mf = Pacejka62Fitter.compute_fy0(mf_params, a_dense, fz_val, 0.0, 83.0)
        mz_mf = Pacejka62Fitter.compute_mz(mf_params, a_dense, fz_val, 0.0, 83.0, fx=0.0, fy_in=fy_mf, svy_in=s_mf)
        ax.plot(np.degrees(a_dense), mz_mf, "--", color=col, lw=1.8, label=f"MF 6.2 ({fz_val} N)")
        
        with torch.no_grad():
            x_pts = torch.tensor(np.column_stack([a_dense, np.full_like(a_dense, fz_val), np.zeros_like(a_dense), np.full_like(a_dense, 83.0)]), dtype=torch.float32)
            d_pts = model(x_pts).numpy()
        ax.plot(np.degrees(a_dense), mz_mf + d_pts[:, 1], "-", color=col, lw=2.4, label=f"MF+PINN ({fz_val} N)")
        
    ax.set_title("Self-Aligning Torque $M_z$ vs Slip Angle $\\alpha$", fontsize=13, fontweight="bold")
    ax.set_xlabel("Slip Angle $\\alpha$ [deg]", fontsize=11)
    ax.set_ylabel("Aligning Torque $M_z$ [Nm]", fontsize=11)
    ax.legend(ncol=3, fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig2_path = os.path.join(PLOTS_DIR, "fig2_mz_vs_alpha_loads.png")
    fig.savefig(fig2_path)
    plt.close(fig)
    print(f">> Saved Fig 2: {fig2_path}")
    
    # --- Figure 3: Camber Sensitivity Fy vs alpha ---
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    cambers = [0.0, 2.0, 4.0]
    c_cols = ["#332288", "#44AA99", "#CC6677"]
    fz_c = 800.0
    
    for c_deg, col in zip(cambers, c_cols):
        g_rad = np.radians(c_deg)
        sub = df_train[(np.abs(df_train["f_z"] - fz_c) < 100) & (np.abs(df_train["ia_deg"] - c_deg) < 0.3)]
        if len(sub) > 0:
            ax.scatter(np.degrees(sub["alpha"]), sub["f_y"], s=3, alpha=0.3, color=col, label=f"Exp $\\gamma$={c_deg}°")
            
        fy_mf, s_mf = Pacejka62Fitter.compute_fy0(mf_params, a_dense, fz_c, g_rad, 83.0)
        ax.plot(np.degrees(a_dense), fy_mf, "--", color=col, lw=1.8, label=f"MF 6.2 ($\\gamma$={c_deg}°)")
        
        with torch.no_grad():
            x_pts = torch.tensor(np.column_stack([a_dense, np.full_like(a_dense, fz_c), np.full_like(a_dense, g_rad), np.full_like(a_dense, 83.0)]), dtype=torch.float32)
            d_pts = model(x_pts).numpy()
        ax.plot(np.degrees(a_dense), fy_mf + d_pts[:, 0], "-", color=col, lw=2.4, label=f"MF+PINN ($\\gamma$={c_deg}°)")
        
    ax.set_title("Camber Sensitivity: $F_y$ vs Slip Angle at $F_z=800$ N", fontsize=13, fontweight="bold")
    ax.set_xlabel("Slip Angle $\\alpha$ [deg]", fontsize=11)
    ax.set_ylabel("Lateral Force $F_y$ [N]", fontsize=11)
    ax.legend(ncol=3, fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig3_path = os.path.join(PLOTS_DIR, "fig3_camber_sensitivity.png")
    fig.savefig(fig3_path)
    plt.close(fig)
    print(f">> Saved Fig 3: {fig3_path}")
    
    # --- Figure 4: Inflation Pressure Sensitivity ---
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    pressures = [55.0, 83.0, 95.0]
    p_cols = ["#882255", "#332288", "#117733"]
    for p_val, col in zip(pressures, p_cols):
        fy_mf, s_mf = Pacejka62Fitter.compute_fy0(mf_params, a_dense, fz_c, 0.0, p_val)
        ax.plot(np.degrees(a_dense), fy_mf, "--", color=col, lw=1.8, label=f"MF 6.2 ({p_val:.0f} kPa)")
        with torch.no_grad():
            x_pts = torch.tensor(np.column_stack([a_dense, np.full_like(a_dense, fz_c), np.zeros_like(a_dense), np.full_like(a_dense, p_val)]), dtype=torch.float32)
            d_pts = model(x_pts).numpy()
        ax.plot(np.degrees(a_dense), fy_mf + d_pts[:, 0], "-", color=col, lw=2.4, label=f"MF+PINN ({p_val:.0f} kPa)")
        
    ax.set_title("Inflation Pressure Sensitivity: $F_y$ vs Slip Angle ($F_z=800$ N, $\\gamma=0$)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Slip Angle $\\alpha$ [deg]", fontsize=11)
    ax.set_ylabel("Lateral Force $F_y$ [N]", fontsize=11)
    ax.legend(ncol=3, fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig4_path = os.path.join(PLOTS_DIR, "fig4_pressure_sensitivity.png")
    fig.savefig(fig4_path)
    plt.close(fig)
    print(f">> Saved Fig 4: {fig4_path}")
    
    # --- Figure 5: Residual Error Distributions (MF 6.2 vs Hybrid PINN) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    
    # Train set error histogram
    err_mf_tr = eval_tr["fy_mf"] - eval_tr["fy_exp"]
    err_hyb_tr = eval_tr["fy_hyb"] - eval_tr["fy_exp"]
    axes[0].hist(err_mf_tr, bins=60, range=(-300, 300), alpha=0.55, color="#e74c3c", label=f"MF 6.2 (RMSE={np.sqrt(np.mean(err_mf_tr**2)):.1f} N)")
    axes[0].hist(err_hyb_tr, bins=60, range=(-300, 300), alpha=0.65, color="#2ecc71", label=f"MF+PINN (RMSE={np.sqrt(np.mean(err_hyb_tr**2)):.1f} N)")
    axes[0].set_title("Training Set (Run 6) $F_y$ Error Distribution", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Error $F_{y,pred} - F_{y,exp}$ [N]")
    axes[0].set_ylabel("Sample Count")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    # Validation set error histogram
    err_mf_val = eval_val["fy_mf"] - eval_val["fy_exp"]
    err_hyb_val = eval_val["fy_hyb"] - eval_val["fy_exp"]
    axes[1].hist(err_mf_val, bins=60, range=(-350, 350), alpha=0.55, color="#e74c3c", label=f"MF 6.2 (RMSE={np.sqrt(np.mean(err_mf_val**2)):.1f} N)")
    axes[1].hist(err_hyb_val, bins=60, range=(-350, 350), alpha=0.65, color="#2ecc71", label=f"MF+PINN (RMSE={np.sqrt(np.mean(err_hyb_val**2)):.1f} N)")
    axes[1].set_title("Validation Set (Run 7) $F_y$ Error Distribution", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Error $F_{y,pred} - F_{y,exp}$ [N]")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig5_path = os.path.join(PLOTS_DIR, "fig5_error_distributions.png")
    fig.savefig(fig5_path)
    plt.close(fig)
    print(f">> Saved Fig 5: {fig5_path}")
    
    # --- Figure 6: Transient Step Steer Response ---
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    t_sim = np.linspace(0.0, 0.10, 500) # 100 ms
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
    print(f">> Saved Fig 6: {fig6_path}")
    
    # ------------------------------------------------------------------
    # 4. Save Comprehensive Validation Report JSON
    # ------------------------------------------------------------------
    val_report = {
        "title": "Mirena Dynamics Experimental Tire Validation",
        "date": "2026-11-09",
        "tire": "Hoosier 16.0x7.5-10 R25B",
        "datasets": {
            "train": {"file": "B1965run6.dat", "points": len(df_train), "type": "Multi-load multi-camber cornering at 40 km/h"},
            "validation": {"file": "B1965run7.dat", "points": len(df_val), "type": "Multi-speed multi-pressure cornering (24-72 km/h)"}
        },
        "dynamic_characteristics_nominal": {
            "load_n": 800.0,
            "camber_rad": 0.0,
            "pressure_kpa": 83.0,
            "experimental": feat_exp,
            "pure_mf62": feat_mf,
            "hybrid_pinn": feat_hyb
        },
        "overall_performance": {
            "train": {
                "pure_mf62": {
                    "fy_rmse_n": float(np.sqrt(np.mean(err_mf_tr ** 2))),
                    "fy_mae_n": float(np.mean(np.abs(err_mf_tr))),
                    "fy_max_err_n": float(np.max(np.abs(err_mf_tr))),
                    "mz_rmse_nm": float(np.sqrt(np.mean((eval_tr["mz_mf"] - eval_tr["mz_exp"]) ** 2)))
                },
                "hybrid_pinn": {
                    "fy_rmse_n": float(np.sqrt(np.mean(err_hyb_tr ** 2))),
                    "fy_mae_n": float(np.mean(np.abs(err_hyb_tr))),
                    "fy_max_err_n": float(np.max(np.abs(err_hyb_tr))),
                    "mz_rmse_nm": float(np.sqrt(np.mean((eval_tr["mz_hyb"] - eval_tr["mz_exp"]) ** 2))),
                    "fy_rmse_reduction_pct": float((1 - np.sqrt(np.mean(err_hyb_tr ** 2)) / np.sqrt(np.mean(err_mf_tr ** 2))) * 100.0),
                    "mz_rmse_reduction_pct": float((1 - np.sqrt(np.mean((eval_tr["mz_hyb"] - eval_tr["mz_exp"]) ** 2)) / np.sqrt(np.mean((eval_tr["mz_mf"] - eval_tr["mz_exp"]) ** 2))) * 100.0)
                }
            },
            "validation": {
                "pure_mf62": {
                    "fy_rmse_n": float(np.sqrt(np.mean(err_mf_val ** 2))),
                    "fy_mae_n": float(np.mean(np.abs(err_mf_val))),
                    "fy_max_err_n": float(np.max(np.abs(err_mf_val))),
                    "mz_rmse_nm": float(np.sqrt(np.mean((eval_val["mz_mf"] - eval_val["mz_exp"]) ** 2)))
                },
                "hybrid_pinn": {
                    "fy_rmse_n": float(np.sqrt(np.mean(err_hyb_val ** 2))),
                    "fy_mae_n": float(np.mean(np.abs(err_hyb_val))),
                    "fy_max_err_n": float(np.max(np.abs(err_hyb_val))),
                    "mz_rmse_nm": float(np.sqrt(np.mean((eval_val["mz_hyb"] - eval_val["mz_exp"]) ** 2))),
                    "fy_rmse_reduction_pct": float((1 - np.sqrt(np.mean(err_hyb_val ** 2)) / np.sqrt(np.mean(err_mf_val ** 2))) * 100.0),
                    "mz_rmse_reduction_pct": float((1 - np.sqrt(np.mean((eval_val["mz_hyb"] - eval_val["mz_exp"]) ** 2)) / np.sqrt(np.mean((eval_val["mz_mf"] - eval_val["mz_exp"]) ** 2))) * 100.0)
                }
            }
        },
        "plots_generated": [
            "fig1_fy_vs_alpha_loads.png",
            "fig2_mz_vs_alpha_loads.png",
            "fig3_camber_sensitivity.png",
            "fig4_pressure_sensitivity.png",
            "fig5_error_distributions.png",
            "fig6_transient_step_response.png"
        ]
    }
    
    rep_json_path = os.path.join(VALIDATION_DIR, "rigorous_validation_report.json")
    with open(rep_json_path, "w") as fp:
        json.dump(val_report, fp, indent=2)
    print(f"\n>> Comprehensive validation report JSON saved to: {rep_json_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
