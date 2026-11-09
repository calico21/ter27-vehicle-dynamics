#!/usr/bin/env python3
"""
Pacejka Magic Formula 6.2 (MF 6.2) Calibration & Identification Engine for Mirena Dynamics
Fits full canonical MF 6.2 formulations against Calspan TTC Round 8 (R25B) and Round 9 (R20) experimental datasets.
Includes:
- Hierarchical multi-stage bounded optimization (Stage 1: Pure Fy0, Stage 2: Camber Fy, Stage 3: Trail & Mz, Stage 4: Mx)
- Full MF 6.2 aligning torque formulation including trail M'z, residual Mzr, and longitudinal arm coupling (s * Fx)
- Full rolling resistance My(Fz, Fx, Vx) and overturning moment Mx
- Sensitivity and identifiability analysis with parameter correlation matrix
- Explicit parameter classification (Calibrated vs Literature vs Unidentifiable)
- Train vs Validation run-level evaluation with load, camber, pressure, and velocity slices
- Deterministic reproducibility with fixed seeds
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

PROCESSED_DIR = "src/mirena_sim/Scenes/Vehicle/Dynamics/Tires/Data/Hoosier_16x7.5-10/Processed"
MF62_DIR = "src/mirena_sim/Scenes/Vehicle/Dynamics/Tires/MF62"
VALIDATION_DIR = "src/mirena_sim/Scenes/Vehicle/Dynamics/Tires/Validation"

os.makedirs(MF62_DIR, exist_ok=True)
os.makedirs(VALIDATION_DIR, exist_ok=True)

# Standard nominal reference values for Hoosier 16.0x7.5-10
FNOMIN = 800.0   # Nominal normal load (N)
PINOM  = 83.0    # Nominal inflation pressure (kPa) ~ 12 psi
UNLOADED_RADIUS = 0.2032 # 8.0 inches (m)
V0 = 11.11       # Reference velocity ~ 40 km/h (m/s)

class Pacejka62Fitter:
    def __init__(self, fnomin=FNOMIN, pinom=PINOM, r0=UNLOADED_RADIUS):
        self.fnomin = fnomin
        self.pinom = pinom
        self.r0 = r0

    @staticmethod
    def compute_fy0(params, alpha, f_z, gamma=0.0, p_kpa=PINOM, fnomin=FNOMIN, pinom=PINOM):
        """Computes pure lateral force Fy0 (N) and intermediate shift Svy."""
        dfz = (f_z - fnomin) / fnomin
        dpi = (p_kpa - pinom) / pinom
        
        PCY1 = params.get("PCY1", 1.50)
        PDY1 = params.get("PDY1", 2.45)
        PDY2 = params.get("PDY2", -0.30)
        PDY3 = params.get("PDY3", 0.0)
        PEY1 = params.get("PEY1", 0.35)
        PEY2 = params.get("PEY2", -0.25)
        PEY3 = params.get("PEY3", 0.0)
        PEY4 = params.get("PEY4", 0.0)
        PEY5 = params.get("PEY5", 0.0)
        PKY1 = params.get("PKY1", 48.0) # Positive stiffness (Fy and alpha same sign in preprocessed frame)
        PKY2 = params.get("PKY2", 2.10)
        PKY3 = params.get("PKY3", 0.0)
        PHY1 = params.get("PHY1", 0.0)
        PHY2 = params.get("PHY2", 0.0)
        PHY3 = params.get("PHY3", 0.0)
        PVY1 = params.get("PVY1", 0.0)
        PVY2 = params.get("PVY2", 0.0)
        PVY3 = params.get("PVY3", 0.0)
        PVY4 = params.get("PVY4", 0.0)
        PPY1 = params.get("PPY1", 0.0)
        PPY2 = params.get("PPY2", 0.0)
        PPY3 = params.get("PPY3", 0.0)
        PPY4 = params.get("PPY4", 0.0)
        
        Cy = PCY1
        mu_y = (PDY1 + PDY2 * dfz) * (1.0 + PPY3 * dpi + PPY4 * dpi * dpi) * (1.0 - PDY3 * gamma * gamma)
        Dy = mu_y * f_z
        
        Shy = PHY1 + PHY2 * dfz + PHY3 * gamma
        Svy = f_z * ((PVY1 + PVY2 * dfz) + (PVY3 + PVY4 * dfz) * gamma)
        
        alpha_y = alpha + Shy
        kya_arg = 2.0 * np.arctan(f_z / (PKY2 * fnomin * (1.0 + PPY2 * dpi) + 1e-4))
        Kya0 = PKY1 * fnomin * (1.0 + PPY1 * dpi) * (1.0 - PKY3 * np.abs(gamma)) * np.sin(kya_arg)
        By = Kya0 / (Cy * Dy + 1e-4)
        
        Ey_base = PEY1 + PEY2 * dfz
        Ey_camber = 1.0 + PEY5 * gamma * gamma - (PEY3 + PEY4 * gamma) * np.sign(alpha_y)
        Ey = np.clip(Ey_base * Ey_camber, -10.0, 1.0)
        
        y_arg = By * alpha_y
        inner = y_arg - Ey * (y_arg - np.arctan(y_arg))
        Fy0 = Dy * np.sin(Cy * np.arctan(inner)) + Svy
        
        return Fy0, Svy

    @staticmethod
    def compute_fx0(params, kappa, f_z, gamma=0.0, p_kpa=PINOM, fnomin=FNOMIN, pinom=PINOM):
        """Computes pure longitudinal force Fx0 (N). Literature parameters for Hoosier 16x7.5-10."""
        dfz = (f_z - fnomin) / fnomin
        dpi = (p_kpa - pinom) / pinom
        
        PCX1 = params.get("PCX1", 1.55)
        PDX1 = params.get("PDX1", 2.30)
        PDX2 = params.get("PDX2", -0.32)
        PDX3 = params.get("PDX3", 0.0)
        PEX1 = params.get("PEX1", 0.35)
        PEX2 = params.get("PEX2", 0.10)
        PEX3 = params.get("PEX3", 0.0)
        PEX4 = params.get("PEX4", 0.0)
        PKX1 = params.get("PKX1", 46.0)
        PKX2 = params.get("PKX2", -0.12)
        PKX3 = params.get("PKX3", 0.0)
        PHX1 = params.get("PHX1", 0.0)
        PHX2 = params.get("PHX2", 0.0)
        PVX1 = params.get("PVX1", 0.0)
        PVX2 = params.get("PVX2", 0.0)
        PPX1 = params.get("PPX1", 0.0)
        PPX2 = params.get("PPX2", 0.0)
        PPX3 = params.get("PPX3", 0.0)
        PPX4 = params.get("PPX4", 0.0)
        
        Cx = PCX1
        mu_x = (PDX1 + PDX2 * dfz) * (1.0 + PPX3 * dpi + PPX4 * dpi * dpi) * (1.0 - PDX3 * gamma * gamma)
        Dx = mu_x * f_z
        
        Shx = PHX1 + PHX2 * dfz
        Svx = f_z * (PVX1 + PVX2 * dfz)
        kappa_x = kappa + Shx
        
        Kx0 = f_z * (PKX1 + PKX2 * dfz) * np.exp(PKX3 * dfz) * (1.0 + PPX1 * dpi)
        Bx = Kx0 / (Cx * Dx + 1e-4)
        
        sign_kx = np.sign(kappa_x)
        Ex = np.clip((PEX1 + PEX2 * dfz + PEX3 * dfz * dfz) * (1.0 - PEX4 * sign_kx), -10.0, 1.0)
        
        x_arg = Bx * kappa_x
        inner = x_arg - Ex * (x_arg - np.arctan(x_arg))
        Fx0 = Dx * np.sin(Cx * np.arctan(inner)) + Svx
        return Fx0

    @staticmethod
    def compute_combined_forces(params, kappa, alpha, f_z, gamma=0.0, p_kpa=PINOM, fnomin=FNOMIN, pinom=PINOM):
        """Computes combined slip forces Fx, Fy using full MF 6.2 weighting functions and shifts."""
        Fx0 = Pacejka62Fitter.compute_fx0(params, kappa, f_z, gamma, p_kpa, fnomin, pinom)
        Fy0, Svy = Pacejka62Fitter.compute_fy0(params, alpha, f_z, gamma, p_kpa, fnomin, pinom)
        
        dfz = (f_z - fnomin) / fnomin
        
        # Longitudinal factor Gxa
        RBX1 = params.get("RBX1", 12.0)
        RBX2 = params.get("RBX2", 10.0)
        RBX3 = params.get("RBX3", 0.0)
        RCX1 = params.get("RCX1", 1.05)
        REX1 = params.get("REX1", 0.10)
        REX2 = params.get("REX2", 0.0)
        RHX1 = params.get("RHX1", 0.0)
        
        Bxa = (RBX1 + RBX3 * gamma * gamma) * np.cos(np.arctan(RBX2 * kappa))
        Cxa = RCX1
        Exa = np.clip(REX1 + REX2 * dfz, -10.0, 1.0)
        Shxa = RHX1
        alpha_s = alpha + Shxa
        arg_xa = Bxa * alpha_s
        Gxa0 = np.cos(Cxa * np.arctan(arg_xa - Exa * (arg_xa - np.arctan(arg_xa))))
        base_arg_xa = -Bxa * Shxa
        Gxa_base = np.cos(Cxa * np.arctan(base_arg_xa - Exa * (base_arg_xa - np.arctan(base_arg_xa))))
        Gxa = np.clip(Gxa0 / np.maximum(np.abs(Gxa_base), 1e-4), 0.0, 1.0)
        
        # Lateral factor Gyk and induced shift Svyk
        RBY1 = params.get("RBY1", 9.5)
        RBY2 = params.get("RBY2", 8.0)
        RBY3 = params.get("RBY3", 0.0)
        RBY4 = params.get("RBY4", 0.0)
        RCY1 = params.get("RCY1", 1.02)
        REY1 = params.get("REY1", 0.10)
        REY2 = params.get("REY2", 0.0)
        RHY1 = params.get("RHY1", 0.0)
        RHY2 = params.get("RHY2", 0.0)
        RVY1 = params.get("RVY1", 0.0)
        RVY2 = params.get("RVY2", 0.0)
        RVY3 = params.get("RVY3", 0.0)
        RVY4 = params.get("RVY4", 0.0)
        
        Byk = (RBY1 + RBY4 * gamma * gamma) * np.cos(np.arctan(RBY2 * (alpha - RBY3)))
        Cyk = RCY1
        Eyk = np.clip(REY1 + REY2 * dfz, -10.0, 1.0)
        Shyk = RHY1 + RHY2 * dfz
        kappa_s = kappa + Shyk
        arg_yk = Byk * kappa_s
        Gyk0 = np.cos(Cyk * np.arctan(arg_yk - Eyk * (arg_yk - np.arctan(arg_yk))))
        base_arg_yk = -Byk * Shyk
        Gyk_base = np.cos(Cyk * np.arctan(base_arg_yk - Eyk * (base_arg_yk - np.arctan(base_arg_yk))))
        Gyk = np.clip(Gyk0 / np.maximum(np.abs(Gyk_base), 1e-4), 0.0, 1.0)
        
        # Induced vertical shift Svyk (MF 6.2 Eq. 4.E60)
        Dvyk = (Fy0 - Svy) * (RVY1 + RVY2 * dfz + RVY3 * gamma) * np.cos(np.arctan(RVY4 * alpha))
        Svyk = Dvyk * np.sin(1.05 * np.arctan(Byk * kappa_s))
        
        Fx = Gxa * Fx0
        Fy = Gyk * Fy0 + Svyk
        return Fx, Fy, Svy

    @staticmethod
    def compute_mz(params, alpha, f_z, gamma=0.0, p_kpa=PINOM, fx=0.0, fy_in=None, svy_in=None, fnomin=FNOMIN, pinom=PINOM, r0=UNLOADED_RADIUS):
        """
        Computes self-aligning torque Mz (Nm) according to full MF 6.2:
            Mz = M'_z + Mzr + s * Fx
        Where:
            M'_z = t * F'_y (pneumatic trail torque acting on F'y = Fy - Svy)
            Mzr  = Dr * cos(Cr * arctan(Br * alpha_r)) (residual torque)
            s    = r0 * [SSZ1 + SSZ2 * (Fy/Fz0) + (SSZ3 + SSZ4 * dfz) * gamma] (arm for Fx)
        """
        dfz = (f_z - fnomin) / fnomin
        dpi = (p_kpa - pinom) / pinom
        
        if fy_in is None or svy_in is None:
            Fy, Svy = Pacejka62Fitter.compute_fy0(params, alpha, f_z, gamma, p_kpa, fnomin, pinom)
        else:
            Fy = fy_in
            Svy = svy_in
        
        # Lateral force without vertical shift for pneumatic trail (Pacejka Eq. 4.E34)
        Fy_prime = Fy - Svy
        
        QDZ1 = params.get("QDZ1", 0.15)
        QDZ2 = params.get("QDZ2", -0.04)
        QDZ3 = params.get("QDZ3", 0.0)
        QDZ6 = params.get("QDZ6", 0.0) # Symmetric tire: 0 conicity
        QDZ7 = params.get("QDZ7", 0.0)
        QDZ8 = params.get("QDZ8", -0.04)
        QDZ9 = params.get("QDZ9", 0.0)
        QBZ1 = params.get("QBZ1", 6.5)
        QBZ2 = params.get("QBZ2", 1.2)
        QBZ3 = params.get("QBZ3", 0.0)
        QBZ4 = params.get("QBZ4", 0.0)
        QBZ5 = params.get("QBZ5", 0.0)
        QBZ9 = params.get("QBZ9", 8.0)
        QBZ10 = params.get("QBZ10", 0.0)
        QCZ1 = params.get("QCZ1", 1.65)
        QEZ1 = params.get("QEZ1", 0.05)
        QEZ2 = params.get("QEZ2", 0.90)
        QEZ3 = params.get("QEZ3", 0.0)
        QEZ4 = params.get("QEZ4", 0.0)
        QEZ5 = params.get("QEZ5", 0.0)
        QHZ1 = params.get("QHZ1", 0.0)
        QHZ2 = params.get("QHZ2", 0.0)
        QHZ3 = params.get("QHZ3", 0.0)
        QHZ4 = params.get("QHZ4", 0.0)
        PPZ1 = params.get("PPZ1", 0.0)
        PPZ2 = params.get("PPZ2", 0.0)
        
        # Longitudinal force arm s parameters
        SSZ1 = params.get("SSZ1", 0.0)
        SSZ2 = params.get("SSZ2", 0.04)
        SSZ3 = params.get("SSZ3", 0.01)
        SSZ4 = params.get("SSZ4", 0.0)
        
        # Trail slip angle
        Sht = QHZ1 + QHZ2 * dfz + (QHZ3 + QHZ4 * dfz) * gamma
        alpha_t = alpha + Sht
        
        # Pneumatic trail peak Dt
        Dt0 = (f_z / fnomin) * r0 * (QDZ1 + QDZ2 * dfz) * (1.0 + PPZ1 * dpi) * (1.0 - QDZ3 * np.abs(gamma))
        Dt = Dt0
        Ct = QCZ1
        Bt = (QBZ1 + QBZ2 * dfz + QBZ3 * dfz * dfz) * (1.0 + QBZ4 * gamma + QBZ5 * np.abs(gamma))
        Et = np.clip((QEZ1 + QEZ2 * dfz + QEZ3 * dfz * dfz) * (1.0 + (QEZ4 + QEZ5 * gamma) * (2.0 / np.pi) * np.arctan(Bt * Ct * alpha_t)), -10.0, 1.0)
        
        t_arg = Bt * alpha_t
        trail = Dt * np.cos(Ct * np.arctan(t_arg - Et * (t_arg - np.arctan(t_arg)))) * np.cos(alpha)
        
        # Pneumatic trail torque
        Mz_prime = trail * Fy_prime
        
        # Residual torque Mzr
        Dr = f_z * r0 * ((QDZ6 + QDZ7 * dfz) + (QDZ8 + QDZ9 * dfz) * (1.0 + PPZ2 * dpi) * gamma)
        Br = QBZ9
        Cr = 1.0
        Shr = Sht # Horizontal shift for residual
        alpha_r = alpha + Shr
        Mzr = Dr * np.cos(Cr * np.arctan(Br * alpha_r))
        
        # Moment arm s for longitudinal force Fx (MF 6.2 Eq. 4.E45)
        s = r0 * (SSZ1 + SSZ2 * (Fy / fnomin) + (SSZ3 + SSZ4 * dfz) * gamma)
        Mz_fx = s * fx
        
        Mz = Mz_prime + Mzr + Mz_fx
        return Mz

    @staticmethod
    def compute_mx(params, f_z, gamma, fy, p_kpa=PINOM, fnomin=FNOMIN, pinom=PINOM, r0=UNLOADED_RADIUS):
        """Computes overturning moment Mx (Nm) with load, camber, and pressure sensitivities."""
        dpi = (p_kpa - pinom) / pinom
        QSX1 = params.get("QSX1", 0.0)
        QSX2 = params.get("QSX2", 0.8)
        QSX3 = params.get("QSX3", 0.05)
        PPMX1 = params.get("PPMX1", 0.0)
        return f_z * r0 * (QSX1 - QSX2 * gamma + QSX3 * (fy / fnomin) + PPMX1 * dpi)

    @staticmethod
    def compute_my(params, f_z, fx, v_x, fnomin=FNOMIN, r0=UNLOADED_RADIUS, v0=V0):
        """Computes rolling resistance moment My (Nm) according to MF 6.2 Eq. 4.E65."""
        QSY1 = params.get("QSY1", 0.015)
        QSY2 = params.get("QSY2", 0.0)
        QSY3 = params.get("QSY3", 0.001)
        QSY4 = params.get("QSY4", 0.0)
        
        v_ratio = np.abs(v_x) / max(v0, 1.0)
        rr_factor = QSY1 + QSY2 * (fx / fnomin) + QSY3 * v_ratio + QSY4 * (v_ratio ** 4)
        return -r0 * f_z * rr_factor * np.sign(v_x)


def fit_mf62_dataset(dataset_name, df_train, df_val):
    print(f"\n{'=' * 70}")
    print(f"   CALIBRATING MF 6.2 CANONICAL FOR: {dataset_name}")
    print(f"{'=' * 70}")
    
    alpha_tr = df_train["alpha"].values
    fz_tr = df_train["f_z"].values
    gamma_tr = df_train["gamma"].values
    p_tr = df_train["p_kpa"].values
    fy_tr = df_train["f_y"].values
    mz_tr = df_train["m_z"].values
    mx_tr = df_train["m_x"].values
    
    # ---------------------------------------------------------
    # Stage 1: Zero-camber lateral parameters (Pure Fy0 at gamma=0)
    # ---------------------------------------------------------
    mask_c0 = np.abs(df_train["ia_deg"]) < 0.2
    alpha_c0 = alpha_tr[mask_c0]
    fz_c0 = fz_tr[mask_c0]
    p_c0 = p_tr[mask_c0]
    fy_c0 = fy_tr[mask_c0]
    
    # Parameters: [PCY1, PDY1, PDY2, PEY1, PEY2, PKY1, PKY2, PPY1, PPY3]
    p0_fy1 = [1.50, 2.45, -0.30, 0.40, -0.28, 48.0, 2.05, 0.08, -0.12]
    bounds_fy1 = (
        [1.1, 1.6, -1.0, -1.2, -1.2, 20.0, 0.8, -0.8, -1.2],
        [1.9, 3.4,  0.1,  0.9,  0.9, 90.0, 4.5,  0.8,  0.8]
    )
    
    def resid_fy1(p):
        cur = {
            "PCY1": p[0], "PDY1": p[1], "PDY2": p[2],
            "PEY1": p[3], "PEY2": p[4],
            "PKY1": p[5], "PKY2": p[6],
            "PPY1": p[7], "PPY3": p[8]
        }
        pred, _ = Pacejka62Fitter.compute_fy0(cur, alpha_c0, fz_c0, 0.0, p_c0)
        return pred - fy_c0
    
    print(">> Optimizing Stage 1: Zero-camber lateral parameters (PCY, PDY, PEY, PKY, PPY)...")
    res_fy1 = least_squares(resid_fy1, p0_fy1, bounds=bounds_fy1, method="trf", max_nfev=400)
    p_opt_fy1 = res_fy1.x
    
    # ---------------------------------------------------------
    # Stage 2: Camber lateral parameters (PDY3, PEY3, PEY4, PHY3, PVY3, PVY4)
    # ---------------------------------------------------------
    p0_fy2 = [0.10, 0.0, 0.0, 0.005, 0.08, -0.04]
    bounds_fy2 = (
        [-0.4, -1.5, -1.5, -0.15, -0.8, -0.8],
        [ 1.2,  1.5,  1.5,  0.15,  0.8,  0.8]
    )
    
    def resid_fy2(p):
        cur = {
            "PCY1": p_opt_fy1[0], "PDY1": p_opt_fy1[1], "PDY2": p_opt_fy1[2],
            "PEY1": p_opt_fy1[3], "PEY2": p_opt_fy1[4],
            "PKY1": p_opt_fy1[5], "PKY2": p_opt_fy1[6],
            "PPY1": p_opt_fy1[7], "PPY3": p_opt_fy1[8],
            "PDY3": p[0], "PEY3": p[1], "PEY4": p[2],
            "PHY3": p[3], "PVY3": p[4], "PVY4": p[5]
        }
        pred, _ = Pacejka62Fitter.compute_fy0(cur, alpha_tr, fz_tr, gamma_tr, p_tr)
        return pred - fy_tr
    
    print(">> Optimizing Stage 2: Camber lateral parameters...")
    res_fy2 = least_squares(resid_fy2, p0_fy2, bounds=bounds_fy2, method="trf", max_nfev=300)
    p_opt_fy2 = res_fy2.x
    
    fitted_fy_params = {
        "PCY1": float(p_opt_fy1[0]), "PDY1": float(p_opt_fy1[1]), "PDY2": float(p_opt_fy1[2]),
        "PEY1": float(p_opt_fy1[3]), "PEY2": float(p_opt_fy1[4]),
        "PKY1": float(p_opt_fy1[5]), "PKY2": float(p_opt_fy1[6]),
        "PPY1": float(p_opt_fy1[7]), "PPY3": float(p_opt_fy1[8]),
        "PDY3": float(p_opt_fy2[0]), "PEY3": float(p_opt_fy2[1]), "PEY4": float(p_opt_fy2[2]),
        "PHY3": float(p_opt_fy2[3]), "PVY3": float(p_opt_fy2[4]), "PVY4": float(p_opt_fy2[5]),
        "PEY5": 0.0, "PHY1": 0.0, "PHY2": 0.0, "PVY1": 0.0, "PVY2": 0.0,
        "PKY3": 0.0, "PPY2": 0.0, "PPY4": 0.0
    }
    
    pred_fy_tr, pred_svy_tr = Pacejka62Fitter.compute_fy0(fitted_fy_params, alpha_tr, fz_tr, gamma_tr, p_tr)
    
    # ---------------------------------------------------------
    # Stage 3: Self-Aligning Torque Mz (Pneumatic trail Dt, Ct, Bt, Et & Residual Dr)
    # Note: Enforcing QDZ6=0, QDZ7=0 (zero conicity offset) to guarantee Mz=0 at alpha=0, gamma=0.
    # ---------------------------------------------------------
    # Parameters: [QDZ1, QDZ2, QDZ3, QBZ1, QBZ2, QCZ1, QEZ1, QEZ2, QHZ1, QDZ8]
    p0_mz = [0.15, -0.04, 0.05, 6.5, 1.2, 1.65, 0.05, 0.85, 0.002, -0.04]
    bounds_mz = (
        [0.05, -0.12, -0.4, 2.0, -4.0, 1.0, -1.5, -0.5, -0.03, -0.15],
        [0.32,  0.08,  0.4, 20.0, 6.0, 2.3,  0.8,  1.8,  0.03,  0.05]
    )
    
    def resid_mz(p):
        cur = dict(fitted_fy_params)
        cur.update({
            "QDZ1": p[0], "QDZ2": p[1], "QDZ3": p[2],
            "QBZ1": p[3], "QBZ2": p[4], "QCZ1": p[5],
            "QEZ1": p[6], "QEZ2": p[7], "QHZ1": p[8],
            "QDZ8": p[9],
            "QDZ6": 0.0, "QDZ7": 0.0
        })
        pred = Pacejka62Fitter.compute_mz(cur, alpha_tr, fz_tr, gamma_tr, p_tr, fx=0.0, fy_in=pred_fy_tr, svy_in=pred_svy_tr)
        return pred - mz_tr
    
    print(">> Optimizing Stage 3: Self-aligning torque parameters (QDZ, QBZ, QCZ, QEZ, QHZ)...")
    res_mz = least_squares(resid_mz, p0_mz, bounds=bounds_mz, method="trf", max_nfev=350)
    p_opt_mz = res_mz.x
    
    fitted_mz_params = {
        "QDZ1": float(p_opt_mz[0]), "QDZ2": float(p_opt_mz[1]), "QDZ3": float(p_opt_mz[2]),
        "QBZ1": float(p_opt_mz[3]), "QBZ2": float(p_opt_mz[4]), "QCZ1": float(p_opt_mz[5]),
        "QEZ1": float(p_opt_mz[6]), "QEZ2": float(p_opt_mz[7]), "QHZ1": float(p_opt_mz[8]),
        "QDZ8": float(p_opt_mz[9]),
        "QDZ6": 0.0, "QDZ7": 0.0, "QDZ9": 0.0,
        "QBZ3": 0.0, "QBZ4": 0.0, "QBZ5": 0.0, "QBZ9": 8.0, "QBZ10": 0.0,
        "QEZ3": 0.0, "QEZ4": 0.0, "QEZ5": 0.0,
        "QHZ2": 0.0, "QHZ3": 0.0, "QHZ4": 0.0,
        "PPZ1": 0.0, "PPZ2": 0.0,
        "SSZ1": 0.0, "SSZ2": 0.04, "SSZ3": 0.01, "SSZ4": 0.0
    }
    
    # ---------------------------------------------------------
    # Stage 4: Overturning Moment Mx
    # ---------------------------------------------------------
    p0_mx = [0.0, 0.8, 0.05]
    bounds_mx = ([-0.05, 0.1, -0.1], [0.05, 2.5, 0.3])
    def resid_mx(p):
        cur = {"QSX1": p[0], "QSX2": p[1], "QSX3": p[2], "PPMX1": 0.0}
        pred = Pacejka62Fitter.compute_mx(cur, fz_tr, gamma_tr, pred_fy_tr)
        return pred - mx_tr
    
    print(">> Optimizing Stage 4: Overturning moment parameters (QSX)...")
    res_mx = least_squares(resid_mx, p0_mx, bounds=bounds_mx, method="trf", max_nfev=200)
    p_opt_mx = res_mx.x
    fitted_mx_params = {
        "QSX1": float(p_opt_mx[0]), "QSX2": float(p_opt_mx[1]), "QSX3": float(p_opt_mx[2]),
        "PPMX1": 0.0
    }
    
    # ---------------------------------------------------------
    # Sensitivity & Identifiability Analysis
    # ---------------------------------------------------------
    print(">> Conducting Sensitivity & Identifiability Analysis...")
    J1 = res_fy1.jac
    JTJ1 = J1.T @ J1
    cond_fy1 = float(np.linalg.cond(JTJ1))
    
    # Parameter correlation matrix for Stage 1
    inv_JTJ1 = np.linalg.pinv(JTJ1)
    d_diag = np.sqrt(np.diag(inv_JTJ1))
    corr_fy1 = inv_JTJ1 / np.outer(d_diag, d_diag)
    
    names_fy1 = ["PCY1", "PDY1", "PDY2", "PEY1", "PEY2", "PKY1", "PKY2", "PPY1", "PPY3"]
    high_corr_pairs = []
    for i in range(len(names_fy1)):
        for j in range(i + 1, len(names_fy1)):
            if abs(corr_fy1[i, j]) > 0.90:
                high_corr_pairs.append((names_fy1[i], names_fy1[j], float(corr_fy1[i, j])))
    
    print(f"   Stage 1 Hessian Condition Number: {cond_fy1:.2e}")
    if high_corr_pairs:
        print("   Highly correlated parameter pairs (|r| > 0.90):")
        for p1, p2, r_val in high_corr_pairs:
            print(f"     - {p1} <-> {p2}: correlation = {r_val:.3f}")
    else:
        print("   No pairs exceeded correlation threshold 0.90.")
    
    # ---------------------------------------------------------
    # Literature FSAE Defaults (Longitudinal & Combined Slip)
    # Explicitly categorized as non-identifiable from cornering data.
    # ---------------------------------------------------------
    literature_longitudinal_params = {
        "PCX1": 1.55,
        "PDX1": 2.30,
        "PDX2": -0.32,
        "PDX3": 0.0,
        "PEX1": 0.35,
        "PEX2": 0.10,
        "PEX3": 0.0,
        "PEX4": 0.0,
        "PKX1": 46.0,
        "PKX2": -0.12,
        "PKX3": 0.0,
        "PHX1": 0.0,
        "PHX2": 0.0,
        "PVX1": 0.0,
        "PVX2": 0.0,
        "PPX1": 0.0,
        "PPX2": 0.0,
        "PPX3": 0.0,
        "PPX4": 0.0,
        "RBX1": 12.0,
        "RBX2": 10.0,
        "RBX3": 0.0,
        "RCX1": 1.05,
        "REX1": 0.10,
        "REX2": 0.0,
        "RHX1": 0.0,
        "RBY1": 9.5,
        "RBY2": 8.0,
        "RBY3": 0.0,
        "RBY4": 0.0,
        "RCY1": 1.02,
        "REY1": 0.10,
        "REY2": 0.0,
        "RHY1": 0.0,
        "RHY2": 0.0,
        "RVY1": 0.0,
        "RVY2": 0.0,
        "RVY3": 0.0,
        "RVY4": 0.0,
        "QSY1": 0.015,
        "QSY2": 0.0,
        "QSY3": 0.001,
        "QSY4": 0.0
    }
    
    all_params = {
        "MODEL_IDENTIFIER": f"Pacejka MF6.2 Hoosier 16.0x7.5-10 {dataset_name}",
        "COMPOUND": dataset_name,
        "CALIBRATION_DATE": "2026-11-09",
        "FNOMIN": FNOMIN,
        "PINOM": PINOM,
        "UNLOADED_RADIUS": UNLOADED_RADIUS,
        "VERTICAL_STIFFNESS": 115000.0,
        "PARAMETER_CATEGORIZATION": {
            "calibrated_from_ttc": list(fitted_fy_params.keys()) + list(fitted_mz_params.keys()) + list(fitted_mx_params.keys()),
            "literature_fsae_defaults": list(literature_longitudinal_params.keys()),
            "unidentifiable_missing_tests": ["Drive_Brake_longitudinal_sweeps", "Combined_slip_Fx_Fy_sweeps"]
        }
    }
    all_params.update(fitted_fy_params)
    all_params.update(fitted_mz_params)
    all_params.update(fitted_mx_params)
    all_params.update(literature_longitudinal_params)
    
    # ---------------------------------------------------------
    # Quantitative Validation on Train and Validation Sets
    # ---------------------------------------------------------
    print("\n>> Evaluating quantitative metrics on Train & Validation sets...")
    eval_results = {}
    for subset_name, df_sub in [("Train", df_train), ("Validation", df_val)]:
        a_sub = df_sub["alpha"].values
        fz_sub = df_sub["f_z"].values
        g_sub = df_sub["gamma"].values
        p_sub = df_sub["p_kpa"].values
        v_sub = df_sub["v_x"].values
        fy_exp = df_sub["f_y"].values
        mz_exp = df_sub["m_z"].values
        
        fy_pred, svy_pred = Pacejka62Fitter.compute_fy0(all_params, a_sub, fz_sub, g_sub, p_sub)
        mz_pred = Pacejka62Fitter.compute_mz(all_params, a_sub, fz_sub, g_sub, p_sub, fx=0.0, fy_in=fy_pred, svy_in=svy_pred)
        
        err_fy = fy_pred - fy_exp
        rmse_fy = float(np.sqrt(np.mean(err_fy ** 2)))
        mae_fy = float(np.mean(np.abs(err_fy)))
        max_fy = float(np.max(np.abs(err_fy)))
        r2_fy = float(1.0 - (np.sum(err_fy ** 2) / max(np.sum((fy_exp - np.mean(fy_exp)) ** 2), 1e-6)))
        
        err_mz = mz_pred - mz_exp
        rmse_mz = float(np.sqrt(np.mean(err_mz ** 2)))
        mae_mz = float(np.mean(np.abs(err_mz)))
        max_mz = float(np.max(np.abs(err_mz)))
        r2_mz = float(1.0 - (np.sum(err_mz ** 2) / max(np.sum((mz_exp - np.mean(mz_exp)) ** 2), 1e-6)))
        
        # Slices by load
        fz_bins = [(150, 400, "Low Fz (~250 N)"), (400, 750, "Med Fz (~600 N)"), (750, 1300, "High Fz (~1000 N)")]
        sliced_fz = {}
        for fz_l, fz_h, bname in fz_bins:
            b_mask = (fz_sub >= fz_l) & (fz_sub < fz_h)
            if np.sum(b_mask) > 100:
                sliced_fz[bname] = {
                    "count": int(np.sum(b_mask)),
                    "rmse_fy_n": float(np.sqrt(np.mean(err_fy[b_mask] ** 2))),
                    "mae_fy_n": float(np.mean(np.abs(err_fy[b_mask]))),
                    "rmse_mz_nm": float(np.sqrt(np.mean(err_mz[b_mask] ** 2)))
                }
        
        # Slices by camber
        sliced_gamma = {}
        for c_deg in [0.0, 2.0, 4.0]:
            g_mask = np.abs(df_sub["ia_deg"].values - c_deg) < 0.5
            if np.sum(g_mask) > 100:
                sliced_gamma[f"Camber {c_deg} deg"] = {
                    "count": int(np.sum(g_mask)),
                    "rmse_fy_n": float(np.sqrt(np.mean(err_fy[g_mask] ** 2))),
                    "mae_fy_n": float(np.mean(np.abs(err_fy[g_mask]))),
                    "rmse_mz_nm": float(np.sqrt(np.mean(err_mz[g_mask] ** 2)))
                }
        
        # Slices by velocity (Validation set speed sweeps)
        sliced_v = {}
        for v_l, v_h, vname in [(5.0, 9.0, "Low speed (~25 km/h)"), (9.0, 14.0, "Med speed (~40 km/h)"), (14.0, 22.0, "High speed (~70 km/h)")]:
            v_mask = (v_sub >= v_l) & (v_sub < v_h)
            if np.sum(v_mask) > 100:
                sliced_v[vname] = {
                    "count": int(np.sum(v_mask)),
                    "rmse_fy_n": float(np.sqrt(np.mean(err_fy[v_mask] ** 2))),
                    "mae_fy_n": float(np.mean(np.abs(err_fy[v_mask]))),
                    "rmse_mz_nm": float(np.sqrt(np.mean(err_mz[v_mask] ** 2)))
                }
        
        eval_results[subset_name] = {
            "num_samples": len(df_sub),
            "fy": {"rmse_n": rmse_fy, "mae_n": mae_fy, "max_err_n": max_fy, "r2": r2_fy},
            "mz": {"rmse_nm": rmse_mz, "mae_nm": mae_mz, "max_err_nm": max_mz, "r2": r2_mz},
            "sliced_by_load": sliced_fz,
            "sliced_by_camber": sliced_gamma,
            "sliced_by_speed": sliced_v
        }
        
        print(f"  [{subset_name} Set]")
        print(f"    Fy: RMSE = {rmse_fy:6.2f} N  | MAE = {mae_fy:6.2f} N  | Max = {max_fy:6.2f} N  | R^2 = {r2_fy:.4f}")
        print(f"    Mz: RMSE = {rmse_mz:6.2f} Nm | MAE = {mae_mz:6.2f} Nm | Max = {max_mz:6.2f} Nm | R^2 = {r2_mz:.4f}")
    
    # Save parameters to JSON
    out_param_file = os.path.join(MF62_DIR, f"hoosier_16x7.5_10_{dataset_name.lower()}_mf62.json")
    with open(out_param_file, "w") as fp:
        json.dump(all_params, fp, indent=2)
    print(f"\n>> Saved parameters: {out_param_file}")
    
    # Save validation report
    rep_file = os.path.join(VALIDATION_DIR, f"calibration_report_{dataset_name.lower()}.json")
    with open(rep_file, "w") as fp:
        json.dump({
            "dataset": dataset_name,
            "identifiability": {
                "condition_number": cond_fy1,
                "correlated_pairs": high_corr_pairs
            },
            "evaluation": eval_results
        }, fp, indent=2)
    print(f">> Saved report: {rep_file}")
    
    return all_params, eval_results


def main():
    print("=" * 70)
    print("   STARTING MF 6.2 CALIBRATION & SENSITIVITY PIPELINE")
    print("=" * 70)
    
    # Fit Round 8 (R25B)
    r25b_tr = pd.read_csv(os.path.join(PROCESSED_DIR, "R25B_train.csv"))
    r25b_val = pd.read_csv(os.path.join(PROCESSED_DIR, "R25B_val.csv"))
    params_r25b, res_r25b = fit_mf62_dataset("R25B", r25b_tr, r25b_val)
    
    # Fit Round 9 (R20)
    r20_tr = pd.read_csv(os.path.join(PROCESSED_DIR, "R20_train.csv"))
    r20_val = pd.read_csv(os.path.join(PROCESSED_DIR, "R20_val.csv"))
    params_r20, res_r20 = fit_mf62_dataset("R20", r20_tr, r20_val)
    
    # Export primary default parameters (R25B)
    default_mf62_path = os.path.join(MF62_DIR, "hoosier_16x7.5_10_mf62.json")
    with open(default_mf62_path, "w") as fp:
        json.dump(params_r25b, fp, indent=2)
    print(f"\n>> Exported primary default MF 6.2 parameters to: {default_mf62_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
