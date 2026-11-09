#!/usr/bin/env python3
"""
Physics-Informed Residual Neural Network Trainer for Mirena Dynamics Tire Model
Trains an ultra-compact Physics-Informed Neural Network (PINN Residual) in PyTorch:
    Delta F = [Delta Fy, Delta Mz]
Features:
- Physics gating (passivity at zero slip, airborne load suppression, domain falloff) embedded directly in forward graph
- Physics-informed anti-symmetry loss penalty (Delta F(-alpha, -gamma) = -Delta F(alpha, gamma))
- Strict train (Run 6) vs validation (Run 7) split with zero sample leakage
- Validation-based early stopping and best checkpoint selection (prevents overfitting)
- Fixed deterministic seeds (torch, numpy)
- Export of weights, biases, normalization constants, and gate bounds for zero-dependency GDScript inference
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

PROCESSED_DIR = "src/mirena_sim/Scenes/Vehicle/Dynamics/Tires/Data/Hoosier_16x7.5-10/Processed"
MF62_DIR = "src/mirena_sim/Scenes/Vehicle/Dynamics/Tires/MF62"
NEURAL_DIR = "src/mirena_sim/Scenes/Vehicle/Dynamics/Tires/NeuralCorrection"
VALIDATION_DIR = "src/mirena_sim/Scenes/Vehicle/Dynamics/Tires/Validation"

os.makedirs(NEURAL_DIR, exist_ok=True)
os.makedirs(VALIDATION_DIR, exist_ok=True)

# Set deterministic seeds
torch.manual_seed(42)
np.random.seed(42)

# Import calibrated MF 6.2 evaluator
from calibrate_mf62 import Pacejka62Fitter

class PhysicsInformedResidualMLP(nn.Module):
    def __init__(self, in_dim=4, hidden1=16, hidden2=16, out_dim=2, x_mean=None, x_std=None, y_mean=None, y_std=None):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden1)
        self.act1 = nn.Tanh()
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.act2 = nn.Tanh()
        self.fc3 = nn.Linear(hidden2, out_dim)
        
        # Fixed normalization buffers
        self.register_buffer("x_mean", torch.tensor(x_mean, dtype=torch.float32) if x_mean is not None else torch.zeros(in_dim))
        self.register_buffer("x_std", torch.tensor(x_std, dtype=torch.float32) if x_std is not None else torch.ones(in_dim))
        self.register_buffer("y_mean", torch.tensor(y_mean, dtype=torch.float32) if y_mean is not None else torch.zeros(out_dim))
        self.register_buffer("y_std", torch.tensor(y_std, dtype=torch.float32) if y_std is not None else torch.ones(out_dim))
        
        # Physics gating thresholds
        self.slip_trans = 0.02 # rad (~1.1 deg)
        self.load_trans = 100.0 # N
        self.alpha_limit = 0.22 # rad (~12.6 deg)
        self.alpha_slope = 0.02
        self.fz_limit = 1300.0 # N
        self.fz_slope = 100.0
        self.clip_fy = 200.0 # N max residual
        self.clip_mz = 10.0  # Nm max residual

    def forward_unconstrained(self, x_norm):
        h = self.act1(self.fc1(x_norm))
        h = self.act2(self.fc2(h))
        return self.fc3(h)

    def forward(self, x_raw):
        """
        Full physics-informed forward pass:
        1. Normalizes raw inputs [alpha, fz, gamma, p_kpa]
        2. Evaluates MLP
        3. Denormalizes outputs
        4. Applies analytical boundary gates (zero-slip passivity, low-load cut, domain falloff, clamp)
        """
        x_norm = (x_raw - self.x_mean) / self.x_std
        out_norm = self.forward_unconstrained(x_norm)
        
        delta_raw_fy = out_norm[:, 0] * self.y_std[0] + self.y_mean[0]
        delta_raw_mz = out_norm[:, 1] * self.y_std[1] + self.y_mean[1]
        
        alpha = x_raw[:, 0]
        fz = x_raw[:, 1]
        gamma = x_raw[:, 2]
        
        # Gate 1: Zero-slip passivity (smoothly vanishes at alpha=0, gamma=0)
        g_slip = torch.tanh((torch.abs(alpha) + torch.abs(gamma)) / self.slip_trans)
        
        # Gate 2: Low-load suppression (smoothly vanishes when tire is unloaded / airborne)
        g_load = torch.clamp(fz / self.load_trans, 0.0, 1.0)
        
        # Gate 3: Domain falloff (smooth sigmoidal decay outside experimental envelope)
        g_domain_alpha = 1.0 / (1.0 + torch.exp((torch.abs(alpha) - self.alpha_limit) / self.alpha_slope))
        g_domain_fz = 1.0 / (1.0 + torch.exp((fz - self.fz_limit) / self.fz_slope))
        g_domain = g_domain_alpha * g_domain_fz
        
        scale = g_slip * g_load * g_domain
        
        delta_fy = torch.clamp(delta_raw_fy * scale, -self.clip_fy, self.clip_fy)
        delta_mz = torch.clamp(delta_raw_mz * scale, -self.clip_mz, self.clip_mz)
        
        return torch.stack([delta_fy, delta_mz], dim=1)


def compute_residuals(df, params):
    alpha = df["alpha"].values
    fz = df["f_z"].values
    gamma = df["gamma"].values
    p_kpa = df["p_kpa"].values
    fy_exp = df["f_y"].values
    mz_exp = df["m_z"].values
    
    # Compute baseline MF 6.2 predictions
    fy_mf, svy_mf = Pacejka62Fitter.compute_fy0(params, alpha, fz, gamma, p_kpa)
    mz_mf = Pacejka62Fitter.compute_mz(params, alpha, fz, gamma, p_kpa, fx=0.0, fy_in=fy_mf, svy_in=svy_mf)
    
    # Residuals: Delta = Exp - MF
    delta_fy = fy_exp - fy_mf
    delta_mz = mz_exp - mz_mf
    
    # Inputs: [alpha, fz, gamma, p_kpa]
    X_raw = np.column_stack([alpha, fz, gamma, p_kpa])
    Y_raw = np.column_stack([delta_fy, delta_mz])
    
    return X_raw, Y_raw, fy_mf, mz_mf


def main():
    print("=" * 70)
    print("   TRAINING PHYSICS-INFORMED RESIDUAL PINN (Delta F = F_exp - F_MF62)")
    print("=" * 70)
    
    # Load calibrated R25B parameters
    param_file = os.path.join(MF62_DIR, "hoosier_16x7.5_10_r25b_mf62.json")
    with open(param_file, "r") as fp:
        params = json.load(fp)
    
    # Load training (Run 6) and validation (Run 7) datasets
    df_train = pd.read_csv(os.path.join(PROCESSED_DIR, "R25B_train.csv"))
    df_val = pd.read_csv(os.path.join(PROCESSED_DIR, "R25B_val.csv"))
    
    print(f">> Computing MF 6.2 baseline residuals on Train ({len(df_train)} pts) and Val ({len(df_val)} pts)...")
    X_tr_raw, Y_tr_raw, fy_mf_tr, mz_mf_tr = compute_residuals(df_train, params)
    X_val_raw, Y_val_raw, fy_mf_val, mz_mf_val = compute_residuals(df_val, params)
    
    # Baseline MF 6.2 errors before neural correction
    rmse_fy_tr_base = np.sqrt(np.mean((Y_tr_raw[:, 0]) ** 2))
    rmse_mz_tr_base = np.sqrt(np.mean((Y_tr_raw[:, 1]) ** 2))
    rmse_fy_val_base = np.sqrt(np.mean((Y_val_raw[:, 0]) ** 2))
    rmse_mz_val_base = np.sqrt(np.mean((Y_val_raw[:, 1]) ** 2))
    
    print(f"  [MF 6.2 Baseline Errors]")
    print(f"    Train:      Fy RMSE = {rmse_fy_tr_base:6.2f} N  |  Mz RMSE = {rmse_mz_tr_base:5.2f} Nm")
    print(f"    Validation: Fy RMSE = {rmse_fy_val_base:6.2f} N  |  Mz RMSE = {rmse_mz_val_base:5.2f} Nm")
    
    # Input Normalization (mean and std across training set)
    x_mean = np.mean(X_tr_raw, axis=0)
    x_std = np.std(X_tr_raw, axis=0)
    x_std[x_std < 1e-4] = 1.0
    
    # Target Normalization
    y_mean = np.mean(Y_tr_raw, axis=0)
    y_std = np.std(Y_tr_raw, axis=0)
    y_std[y_std < 1e-4] = 1.0
    
    print(f"\n>> Normalization Factors:")
    print(f"    Inputs  x_mean = {np.round(x_mean, 4)}, x_std = {np.round(x_std, 4)}")
    print(f"    Targets y_mean = {np.round(y_mean, 4)}, y_std = {np.round(y_std, 4)}")
    
    tensor_x_tr = torch.tensor(X_tr_raw, dtype=torch.float32)
    tensor_y_tr = torch.tensor(Y_tr_raw, dtype=torch.float32)
    tensor_x_val = torch.tensor(X_val_raw, dtype=torch.float32)
    tensor_y_val = torch.tensor(Y_val_raw, dtype=torch.float32)
    
    train_dataset = TensorDataset(tensor_x_tr, tensor_y_tr)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    model = PhysicsInformedResidualMLP(
        in_dim=4, hidden1=16, hidden2=16, out_dim=2,
        x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std
    )
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    
    print("\n>> Training Physics-Informed Residual PINN with early stopping on validation loss...")
    epochs = 35
    best_val_loss = float("inf")
    best_state_dict = None
    best_epoch = 0
    patience = 8
    patience_counter = 0
    
    lambda_sym = 0.05
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        
        for bx_raw, by_raw in train_loader:
            optimizer.zero_grad()
            
            # Forward pass through physics gates
            pred = model(bx_raw)
            data_loss = criterion(pred, by_raw)
            
            # Anti-symmetry constraint: Delta F(-alpha, fz, -gamma, p) == -Delta F(alpha, fz, gamma, p)
            bx_mirrored = bx_raw.clone()
            bx_mirrored[:, 0] = -bx_mirrored[:, 0] # -alpha
            bx_mirrored[:, 2] = -bx_mirrored[:, 2] # -gamma
            pred_mirrored = model(bx_mirrored)
            
            sym_loss = torch.mean((pred[:, 0] + pred_mirrored[:, 0]) ** 2) + torch.mean((pred[:, 1] + pred_mirrored[:, 1]) ** 2)
            
            loss = data_loss + lambda_sym * sym_loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(bx_raw)
            
        scheduler.step()
        train_loss = total_loss / len(tensor_x_tr)
        
        # Evaluate on unseen Validation Set
        model.eval()
        with torch.no_grad():
            val_pred = model(tensor_x_val)
            val_loss = criterion(val_pred, tensor_y_val).item()
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
            improved_mark = " [*Best*]"
        else:
            patience_counter += 1
            improved_mark = ""
            
        if epoch % 5 == 0 or epoch == epochs or improved_mark != "":
            print(f"  Epoch {epoch:2d}/{epochs} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}{improved_mark}")
            
        if patience_counter >= patience:
            print(f">> Early stopping triggered at epoch {epoch} (no validation improvement for {patience} epochs).")
            break
            
    print(f">> Restoring best model weights from Epoch {best_epoch} (Val Loss: {best_val_loss:.5f})...")
    model.load_state_dict(best_state_dict)
    
    # Quantitative Final Evaluation
    model.eval()
    with torch.no_grad():
        delta_tr_pred = model(tensor_x_tr).numpy()
        delta_val_pred = model(tensor_x_val).numpy()
        
    fy_hyb_tr = fy_mf_tr + delta_tr_pred[:, 0]
    mz_hyb_tr = mz_mf_tr + delta_tr_pred[:, 1]
    fy_hyb_val = fy_mf_val + delta_val_pred[:, 0]
    mz_hyb_val = mz_mf_val + delta_val_pred[:, 1]
    
    # Train metrics
    rmse_fy_tr_hyb = float(np.sqrt(np.mean((df_train["f_y"].values - fy_hyb_tr) ** 2)))
    mae_fy_tr_hyb = float(np.mean(np.abs(df_train["f_y"].values - fy_hyb_tr)))
    r2_fy_tr_hyb = float(1.0 - (np.sum((df_train["f_y"].values - fy_hyb_tr) ** 2) / np.sum((df_train["f_y"].values - np.mean(df_train["f_y"].values)) ** 2)))
    
    rmse_mz_tr_hyb = float(np.sqrt(np.mean((df_train["m_z"].values - mz_hyb_tr) ** 2)))
    mae_mz_tr_hyb = float(np.mean(np.abs(df_train["m_z"].values - mz_hyb_tr)))
    r2_mz_tr_hyb = float(1.0 - (np.sum((df_train["m_z"].values - mz_hyb_tr) ** 2) / np.sum((df_train["m_z"].values - np.mean(df_train["m_z"].values)) ** 2)))
    
    # Validation metrics
    rmse_fy_val_hyb = float(np.sqrt(np.mean((df_val["f_y"].values - fy_hyb_val) ** 2)))
    mae_fy_val_hyb = float(np.mean(np.abs(df_val["f_y"].values - fy_hyb_val)))
    r2_fy_val_hyb = float(1.0 - (np.sum((df_val["f_y"].values - fy_hyb_val) ** 2) / np.sum((df_val["f_y"].values - np.mean(df_val["f_y"].values)) ** 2)))
    
    rmse_mz_val_hyb = float(np.sqrt(np.mean((df_val["m_z"].values - mz_hyb_val) ** 2)))
    mae_mz_val_hyb = float(np.mean(np.abs(df_val["m_z"].values - mz_hyb_val)))
    r2_mz_val_hyb = float(1.0 - (np.sum((df_val["m_z"].values - mz_hyb_val) ** 2) / np.sum((df_val["m_z"].values - np.mean(df_val["m_z"].values)) ** 2)))
    
    print("\n" + "=" * 70)
    print("   QUANTITATIVE PERFORMANCE COMPARISON: PURE MF 6.2 vs HYBRID MF+PINN")
    print("=" * 70)
    print(f"  [Train Set (B1965run6)]")
    print(f"    Fy RMSE:  {rmse_fy_tr_base:6.2f} N  -> {rmse_fy_tr_hyb:6.2f} N   (Reduction: {(1 - rmse_fy_tr_hyb/rmse_fy_tr_base)*100:+.1f}%) | R^2: {r2_fy_tr_hyb:.4f}")
    print(f"    Mz RMSE:  {rmse_mz_tr_base:6.2f} Nm -> {rmse_mz_tr_hyb:6.2f} Nm  (Reduction: {(1 - rmse_mz_tr_hyb/rmse_mz_tr_base)*100:+.1f}%) | R^2: {r2_mz_tr_hyb:.4f}")
    print(f"\n  [Validation Set (B1965run7)] (Unseen speed & pressure sweeps)")
    print(f"    Fy RMSE:  {rmse_fy_val_base:6.2f} N  -> {rmse_fy_val_hyb:6.2f} N   (Reduction: {(1 - rmse_fy_val_hyb/rmse_fy_val_base)*100:+.1f}%) | R^2: {r2_fy_val_hyb:.4f}")
    print(f"    Mz RMSE:  {rmse_mz_val_base:6.2f} Nm -> {rmse_mz_val_hyb:6.2f} Nm  (Reduction: {(1 - rmse_mz_val_hyb/rmse_mz_val_base)*100:+.1f}%) | R^2: {r2_mz_val_hyb:.4f}")
    print("=" * 70)
    
    # Export weights for zero-dependency GDScript inference
    w1 = model.fc1.weight.detach().cpu().numpy().tolist()
    b1 = model.fc1.bias.detach().cpu().numpy().tolist()
    w2 = model.fc2.weight.detach().cpu().numpy().tolist()
    b2 = model.fc2.bias.detach().cpu().numpy().tolist()
    w3 = model.fc3.weight.detach().cpu().numpy().tolist()
    b3 = model.fc3.bias.detach().cpu().numpy().tolist()
    
    export_payload = {
        "architecture": {
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
            "slip_trans_rad": float(model.slip_trans),
            "load_trans_n": float(model.load_trans),
            "alpha_limit_rad": float(model.alpha_limit),
            "alpha_slope_rad": float(model.alpha_slope),
            "fz_limit_n": float(model.fz_limit),
            "fz_slope_n": float(model.fz_slope),
            "clip_fy_n": float(model.clip_fy),
            "clip_mz_nm": float(model.clip_mz)
        }
    }
    
    out_weights_path = os.path.join(NEURAL_DIR, "neural_residual_weights.json")
    with open(out_weights_path, "w") as fp:
        json.dump(export_payload, fp, indent=2)
    print(f">> Exported PINN residual weights to: {out_weights_path}")
    
    # Save validation report
    neural_report = {
        "model_type": "MF 6.2 + Physics-Informed Residual PINN",
        "training_checkpoint": {
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss
        },
        "metrics": {
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
    }
    rep_path = os.path.join(VALIDATION_DIR, "neural_residual_report.json")
    with open(rep_path, "w") as fp:
        json.dump(neural_report, fp, indent=2)
    print(f">> Exported neural validation report to: {rep_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
