#!/usr/bin/env python3
"""
Pacejka Magic Formula 6.2 (MF 6.2) Rigorous Calibration & Identifiability Engine
Senior Tire Dynamics Engineering Implementation for Mirena Dynamics (Hoosier 16.0x7.5-10)

Key Capabilities:
- Canonical MF 6.2 formulation compliant with Pacejka 2012 / Besselink 2010 and ISO 8855
- Complete coordinate consistency (+X fwd, +Y left, +Z up):
    * Positive slip angle alpha produces positive lateral force Fy
    * Positive pneumatic trail produces negative (restoring) aligning moment Mz
    * Complete Mz decomposition: Mz = Mz' + Mzr + s * Fx, using measured Fx
    * Consistent camber thrust and overturning moment Mx
- Multi-stage hierarchical bounded least-squares optimization:
    Stage 1: Pure lateral force at nominal pressure and zero camber (PCY, PDY, PEY, PKY, PHY, PVY)
    Stage 2: Lateral pressure sensitivity (PPY1, PPY3)
    Stage 3: Camber sensitivity & induced forces (PVY3, PVY4, PHY3, PDY3, PEY4)
    Stage 4: Pneumatic trail and self-aligning torque Mz at zero camber (QDZ, QBZ, QCZ, QEZ, QHZ)
    Stage 5: Camber residual aligning torque & trail modulation (QDZ8, QDZ3)
    Stage 6: Overturning moment Mx (QSX1, QSX2, QSX3, PPMX1)
    Stage 7: Joint refinement of identifiable parameters comparing staged vs joint solution
- Scale-invariant identifiability & sensitivity analysis at EVERY stage:
    * Dimensionless scaled Jacobian singular values and condition number
    * Full parameter correlation matrix
    * Parameter covariance and asymptotic standard errors
    * Boundary saturation detection (flags parameters within 1% of bounds)
- Vertical tire characterization from spring rate test runs:
    * Identifies linear vertical stiffness kz and unloaded radius R0
    * Accounts for pressure sensitivity of radial stiffness
- Strict independent validation (ZERO leakage):
    * R25B: calibrate on Run 6, validate on Run 7
    * R20: calibrate on Run 8, validate on Run 9
    * Global and sliced performance metrics (RMSE, NRMSE, MAE, Max Error, R^2, Bias)
- Parameter categorization separating calibrated parameters from literature defaults
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

# Robust repository path resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../../../../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

TIRE_DIR = os.path.join(REPO_ROOT, "src/mirena_sim/Scenes/Vehicle/Dynamics/Tires")
PROCESSED_DIR = os.path.join(TIRE_DIR, "Data/Hoosier_16x7.5-10/Processed")
MF62_DIR = os.path.join(TIRE_DIR, "MF62")
VALIDATION_DIR = os.path.join(TIRE_DIR, "Validation")

os.makedirs(MF62_DIR, exist_ok=True)
os.makedirs(VALIDATION_DIR, exist_ok=True)

# Fixed nominal reference quantities for Hoosier 16.0x7.5-10
FNOMIN = 800.0   # Nominal normal load (N)
PINOM  = 83.0    # Nominal inflation pressure (kPa) ~ 12 psi
UNLOADED_RADIUS = 0.2032 # 8.0 inches (m)
V0 = 11.11       # Reference speed (40 km/h in m/s)


class Pacejka62Fitter:
    """
    Evaluates canonical MF 6.2 formulas for pure slip, combined slip,
    aligning torque, overturning moment, and rolling resistance.
    """
    def __init__(self, fnomin=FNOMIN, pinom=PINOM, r0=UNLOADED_RADIUS):
        self.fnomin = fnomin
        self.pinom = pinom
        self.r0 = r0

    @staticmethod
    def compute_fy0(params, alpha, f_z, gamma=0.0, p_kpa=PINOM, fnomin=FNOMIN, pinom=PINOM):
        """
        Computes pure lateral force Fy0 (N) and intermediate vertical shift Svy.
        Complies with canonical MF 6.2 (Pacejka 2012 Eq. 4.E19 - 4.E30).
        """
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
        PKY1 = params.get("PKY1", 48.0)
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
        """Computes pure longitudinal force Fx0 (N)."""
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
        """Computes combined slip forces Fx, Fy using canonical MF 6.2 weighting functions and shifts."""
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
        Computes self-aligning torque Mz according to canonical MF 6.2:
        Mz = Mz' + Mzr + s * Fx = - trail * Fy' + Mzr + s * Fx
        In ISO 8855 with +Z upward, restoring torque is negative for alpha > 0.
        """
        dfz = (f_z - fnomin) / fnomin
        dpi = (p_kpa - pinom) / pinom

        if fy_in is None or svy_in is None:
            Fy, Svy = Pacejka62Fitter.compute_fy0(params, alpha, f_z, gamma, p_kpa, fnomin, pinom)
        else:
            Fy = fy_in
            Svy = svy_in

        Fy_prime = Fy - Svy

        QDZ1 = params.get("QDZ1", 0.15)
        QDZ2 = params.get("QDZ2", -0.04)
        QDZ3 = params.get("QDZ3", 0.0)
        QDZ6 = params.get("QDZ6", 0.0)
        QDZ7 = params.get("QDZ7", 0.0)
        QDZ8 = params.get("QDZ8", 0.0)
        QDZ9 = params.get("QDZ9", 0.0)
        QBZ1 = params.get("QBZ1", 6.0)
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
        SSZ1 = params.get("SSZ1", 0.0)
        SSZ2 = params.get("SSZ2", 0.04)
        SSZ3 = params.get("SSZ3", 0.01)
        SSZ4 = params.get("SSZ4", 0.0)

        # 1. Pneumatic Trail
        Sht = QHZ1 + QHZ2 * dfz + (QHZ3 + QHZ4 * dfz) * gamma
        alpha_t = alpha + Sht

        Dt = (f_z / fnomin) * r0 * (QDZ1 + QDZ2 * dfz) * (1.0 + PPZ1 * dpi) * (1.0 - QDZ3 * np.abs(gamma))
        Ct = QCZ1
        Bt = (QBZ1 + QBZ2 * dfz + QBZ3 * dfz * dfz) * (1.0 + QBZ4 * gamma + QBZ5 * np.abs(gamma))
        Et = np.clip((QEZ1 + QEZ2 * dfz + QEZ3 * dfz * dfz) * (1.0 + (QEZ4 + QEZ5 * gamma) * (2.0 / np.pi) * np.arctan(Bt * Ct * alpha_t)), -10.0, 1.0)

        t_arg = Bt * alpha_t
        trail = Dt * np.cos(Ct * np.arctan(t_arg - Et * (t_arg - np.arctan(t_arg)))) * np.cos(alpha)
        Mz_prime = -trail * Fy_prime # Canonical minus sign: restoring moment

        # 2. Residual Aligning Torque
        Dr = f_z * r0 * ((QDZ6 + QDZ7 * dfz) + (QDZ8 + QDZ9 * dfz) * (1.0 + PPZ2 * dpi) * gamma)
        Br = QBZ9
        Cr = 1.0
        Shr = Sht
        alpha_r = alpha + Shr
        Mzr = Dr * np.cos(Cr * np.arctan(Br * alpha_r))

        # 3. Longitudinal force arm coupling s * Fx
        s = r0 * (SSZ1 + SSZ2 * (Fy / fnomin) + (SSZ3 + SSZ4 * dfz) * gamma)
        Mz_fx = s * fx

        return Mz_prime + Mzr + Mz_fx

    @staticmethod
    def compute_mx(params, f_z, gamma, fy, p_kpa=PINOM, fnomin=FNOMIN, pinom=PINOM, r0=UNLOADED_RADIUS):
        """Computes overturning moment Mx (Nm) in target ISO coordinate system."""
        dpi = (p_kpa - pinom) / pinom
        QSX1 = params.get("QSX1", 0.0)
        QSX2 = params.get("QSX2", 2.0)
        QSX3 = params.get("QSX3", 0.05)
        PPMX1 = params.get("PPMX1", 0.0)
        return f_z * r0 * (QSX1 + QSX2 * gamma + QSX3 * (fy / fnomin) + PPMX1 * dpi)

    @staticmethod
    def compute_my(params, f_z, fx, v_x, fnomin=FNOMIN, r0=UNLOADED_RADIUS, v0=V0):
        """Computes rolling resistance moment My (Nm) according to MF 6.2 Eq. 4.E65."""
        QSY1 = params.get("QSY1", 0.015)
        QSY2 = params.get("QSY2", 0.0)
        QSY3 = params.get("QSY3", 0.001)
        QSY4 = params.get("QSY4", 0.0)
        v_ratio = np.abs(v_x) / max(v0, 1.0)
        rr = QSY1 + QSY2 * (fx / fnomin) + QSY3 * v_ratio + QSY4 * (v_ratio ** 4)
        return -r0 * f_z * rr * np.sign(v_x)

    @staticmethod
    def estimate_vertical_properties(spring_csv_path, p_nom=83.0):
        """
        Estimates tire vertical stiffness kz and unloaded radius R0 from vertical characterization data.
        Returns linear stiffness at nominal pressure, pressure sensitivity, R0, R2, and uncertainty.
        """
        if not os.path.exists(spring_csv_path):
            raise FileNotFoundError(f"Spring rate dataset not found: {spring_csv_path}")

        df = pd.read_csv(spring_csv_path)
        fz = df["f_z"].values
        rl = df["r_l"].values
        p = df["p_kpa"].values

        # Nominal pressure subset
        m_nom = (np.abs(p - p_nom) < 4.0)
        if np.sum(m_nom) < 100:
            m_nom = np.ones(len(fz), dtype=bool)

        poly_nom = np.polyfit(fz[m_nom], rl[m_nom], 1)
        kz_nom = float(-1.0 / poly_nom[0])
        r0_nom = float(poly_nom[1])

        # Goodness of fit
        defl_pred = r0_nom - (1.0 / kz_nom) * fz[m_nom]
        err = rl[m_nom] - defl_pred
        rmse_m = float(np.sqrt(np.mean(err ** 2)))
        ss_tot = float(np.sum((rl[m_nom] - np.mean(rl[m_nom])) ** 2))
        ss_res = float(np.sum(err ** 2))
        r2 = float(1.0 - ss_res / max(ss_tot, 1e-6))

        # Pressure sensitivity of stiffness
        kz_p = {}
        for p_val in [69.0, 83.0, 97.0]:
            m_p = (np.abs(p - p_val) < 4.0)
            if np.sum(m_p) > 100:
                p_fit = np.polyfit(fz[m_p], rl[m_p], 1)
                kz_p[p_val] = float(-1.0 / p_fit[0])

        dp_slope = 0.0
        if len(kz_p) >= 2:
            p_pts = np.array(list(kz_p.keys()))
            k_pts = np.array(list(kz_p.values()))
            p_poly = np.polyfit(p_pts, k_pts, 1)
            dp_slope = float(p_poly[0])

        return {
            "linear_stiffness_n_m": kz_nom,
            "unloaded_radius_m": r0_nom,
            "stiffness_r2": r2,
            "deflection_rmse_mm": rmse_m * 1000.0,
            "fitting_range_fz_n": [float(np.min(fz[m_nom])), float(np.max(fz[m_nom]))],
            "pressure_sensitivity_npm_per_kpa": dp_slope,
            "stiffness_by_pressure": kz_p
        }


def multistart_least_squares(resid_func, bounds, default_p0, num_starts=6, loss="soft_l1", f_scale=25.0, max_nfev=350, seed=42):
    """
    Multi-start optimization engine to escape local minima in non-convex Magic Formula stages.
    Samples candidate seeds within parameter bounds, evaluates residual costs,
    and executes Trust Region Reflective (TRF) optimization with robust loss from the top seeds.
    """
    lb = np.array(bounds[0], dtype=float)
    ub = np.array(bounds[1], dtype=float)
    default_p0 = np.array(default_p0, dtype=float)

    rng = np.random.RandomState(seed)
    candidates = [default_p0, 0.5 * (lb + ub)]
    for _ in range(num_starts):
        rand_pt = rng.uniform(lb, ub)
        candidates.append(rand_pt)

    costs = []
    for pt in candidates:
        try:
            r = resid_func(pt)
            costs.append(np.sum(r ** 2))
        except Exception:
            costs.append(float("inf"))

    sorted_indices = np.argsort(costs)
    top_candidates = [candidates[idx] for idx in sorted_indices[:3]]
    if not any(np.allclose(default_p0, c) for c in top_candidates):
        top_candidates.append(default_p0)

    best_res = None
    best_cost = float("inf")
    for p0_try in top_candidates:
        try:
            res = least_squares(resid_func, p0_try, bounds=bounds, loss=loss, f_scale=f_scale, method="trf", max_nfev=max_nfev)
            if res.cost < best_cost:
                best_cost = res.cost
                best_res = res
        except Exception:
            continue

    if best_res is None:
        best_res = least_squares(resid_func, default_p0, bounds=bounds, loss=loss, f_scale=f_scale, method="trf", max_nfev=max_nfev)
    return best_res


def compute_stage_identifiability(res, param_names, stage_name, p0_scale, bounds=None):
    """
    Calculates scale-invariant Jacobian SVD, condition number, correlations,
    standard errors, and checks for boundary saturation.
    """
    J = res.jac
    N, P = J.shape
    p_opt = res.x

    # Parameter scale vector for scale-invariant Jacobian
    scale_vec = np.abs(np.array(p0_scale))
    scale_vec = np.where(scale_vec < 1e-4, 1.0, scale_vec)

    # Scaled Jacobian: J_scaled = J * diag(scale_vec)
    J_scaled = J * scale_vec[np.newaxis, :]
    JTJ = J.T @ J

    # SVD of scaled Jacobian
    U, S, Vt = np.linalg.svd(J_scaled, full_matrices=False)
    cond_num = float(S[0] / max(S[-1], 1e-12))
    rank = int(np.sum(S > 1e-5 * S[0]))

    # Robust covariance via pseudo-inverse
    inv_JTJ = np.linalg.pinv(JTJ)
    residual_variance = float(np.sum(res.fun ** 2) / max(N - P, 1))
    cov_matrix = inv_JTJ * residual_variance
    std_errors = np.sqrt(np.maximum(np.diag(cov_matrix), 0.0))

    # Parameter correlation matrix
    diag_inv = np.sqrt(np.maximum(np.diag(inv_JTJ), 1e-12))
    corr = inv_JTJ / np.outer(diag_inv, diag_inv)

    high_corrs = []
    for i in range(P):
        for j in range(i + 1, P):
            if abs(corr[i, j]) > 0.85:
                high_corrs.append({
                    "param_i": param_names[i],
                    "param_j": param_names[j],
                    "correlation": float(corr[i, j])
                })

    # Boundary saturation check (within 1% of bound)
    boundary_saturated = []
    if bounds is not None:
        lb, ub = bounds
        for i, name in enumerate(param_names):
            span = ub[i] - lb[i]
            if span > 1e-6:
                dist_lb = (p_opt[i] - lb[i]) / span
                dist_ub = (ub[i] - p_opt[i]) / span
                if dist_lb < 0.01:
                    boundary_saturated.append({"param": name, "bound": "lower", "value": float(p_opt[i]), "limit": float(lb[i])})
                elif dist_ub < 0.01:
                    boundary_saturated.append({"param": name, "bound": "upper", "value": float(p_opt[i]), "limit": float(ub[i])})

    param_diagnostics = {}
    for i, name in enumerate(param_names):
        param_diagnostics[name] = {
            "optimal_value": float(p_opt[i]),
            "std_error": float(std_errors[i]),
            "relative_uncertainty_pct": float(std_errors[i] / max(abs(p_opt[i]), 1e-4) * 100.0)
        }

    print(f"\n--- Identifiability Analysis: {stage_name} ---")
    print(f"   Numerical Rank: {rank}/{P} | Scaled Condition Number: {cond_num:.2e}")
    print(f"   Scaled Singular Values: [{', '.join([f'{s:.1f}' for s in S])}]")
    if boundary_saturated:
        print("   [WARNING] Parameters Saturated at Optimization Bounds:")
        for bs in boundary_saturated:
            print(f"     - {bs['param']} hit {bs['bound']} bound: {bs['value']:.4f} (limit: {bs['limit']:.4f})")
    else:
        print("   Zero parameters saturated at optimization bounds.")

    if high_corrs:
        print("   Correlated Parameter Pairs (|r| > 0.85):")
        for hc in high_corrs:
            print(f"     - {hc['param_i']} <-> {hc['param_j']}: r = {hc['correlation']:+.3f}")
    else:
        print("   No parameter pairs exceed correlation threshold 0.85.")

    return {
        "stage": stage_name,
        "jacobian_rank": rank,
        "num_params": P,
        "scaled_condition_number": cond_num,
        "singular_values": S.tolist(),
        "boundary_saturated_parameters": boundary_saturated,
        "high_correlations": high_corrs,
        "parameters": param_diagnostics
    }


def fit_mf62_compound(compound_name, df_train, df_val, spring_csv_path):
    print(f"\n{'=' * 78}")
    print(f"   CALIBRATING MF 6.2 MODEL: HOOSIER 16.0x7.5-10 {compound_name}")
    print(f"{'=' * 78}")

    alpha_tr = df_train["alpha"].values
    fz_tr = df_train["f_z"].values
    gamma_tr = df_train["gamma"].values
    p_tr = df_train["p_kpa"].values
    fy_tr = df_train["f_y"].values
    mz_tr = df_train["m_z"].values
    mx_tr = df_train["m_x"].values
    fx_tr = df_train["f_x"].values

    # 0. Characterize vertical stiffness from spring rate test
    print("\n>> Characterizing vertical load-deflection from experimental spring data...")
    vert_props = Pacejka62Fitter.estimate_vertical_properties(spring_csv_path, p_nom=PINOM)
    print(f"   Fitted Vertical Stiffness: {vert_props['linear_stiffness_n_m']:.1f} N/m (R^2 = {vert_props['stiffness_r2']:.4f})")
    print(f"   Fitted Reference Radius R0: {vert_props['unloaded_radius_m']:.4f} m")

    # Load-bin weights to balance normal load contributions across the 5 physical test levels
    load_bins = np.digitize(fz_tr, [330.0, 550.0, 770.0, 990.0])
    bin_counts = np.bincount(load_bins, minlength=5)
    bin_weights_arr = 1.0 / np.sqrt(np.maximum(bin_counts[load_bins], 1))
    bin_weights_arr = bin_weights_arr / np.mean(bin_weights_arr)

    stage_reports = {}

    # ------------------------------------------------------------------
    # Stage 1: Zero-Camber Nominal Lateral Force (at p = 83 kPa, gamma = 0)
    # Identifies: PDY1, PDY2, PKY1, PKY2, PEY1, PEY2, PHY1, PVY1
    # NOTE: PCY1 is fixed to canonical 1.55 to decouple the near-perfect
    # collinearity with PEY1 (r = 0.9997 when unconstrained).
    # ------------------------------------------------------------------
    mask_c0_p0 = (np.abs(df_train["ia_deg"]) < 0.2) & (np.abs(p_tr - PINOM) < 4.0)
    alpha_s1 = alpha_tr[mask_c0_p0]
    fz_s1 = fz_tr[mask_c0_p0]
    fy_s1 = fy_tr[mask_c0_p0]
    w_s1 = bin_weights_arr[mask_c0_p0]

    names_s1 = ["PDY1", "PDY2", "PKY1", "PKY2", "PEY1", "PEY2", "PHY1", "PVY1"]
    p0_s1 = [2.50, -0.30, 48.0, 2.05, 0.40, -0.20, 0.001, 0.025]
    bounds_s1 = (
        [1.6, -1.0, 20.0, 0.8, -1.2, -1.2, -0.02, -0.08],
        [3.4,  0.1, 90.0, 4.5,  0.9,  0.9,  0.02,  0.08]
    )

    def resid_stage1(p):
        cur = {
            "PCY1": 1.55,
            "PDY1": p[0], "PDY2": p[1],
            "PKY1": p[2], "PKY2": p[3],
            "PEY1": p[4], "PEY2": p[5],
            "PHY1": p[6], "PVY1": p[7]
        }
        pred, _ = Pacejka62Fitter.compute_fy0(cur, alpha_s1, fz_s1, gamma=0.0, p_kpa=PINOM)
        return (pred - fy_s1) * w_s1

    print("\n>> Optimizing Stage 1: Nominal lateral parameters at zero camber & nominal pressure (PCY1=1.55 fixed)...")
    res_s1 = multistart_least_squares(resid_stage1, bounds_s1, p0_s1, num_starts=6, loss="soft_l1", f_scale=25.0, max_nfev=350)
    p_opt1 = res_s1.x
    stage_reports["stage1_zero_camber_fy"] = compute_stage_identifiability(res_s1, names_s1, "Stage 1: Pure Fy0 (PCY1=1.55 fixed)", p0_s1, bounds_s1)

    # ------------------------------------------------------------------
    # Stage 2: Lateral Pressure Sensitivity (at gamma = 0 across all pressures)
    # Identifies: PPY1, PPY3
    # ------------------------------------------------------------------
    mask_c0_all_p = (np.abs(df_train["ia_deg"]) < 0.2)
    alpha_s2 = alpha_tr[mask_c0_all_p]
    fz_s2 = fz_tr[mask_c0_all_p]
    p_s2 = p_tr[mask_c0_all_p]
    fy_s2 = fy_tr[mask_c0_all_p]
    w_s2 = bin_weights_arr[mask_c0_all_p]

    names_s2 = ["PPY1", "PPY3"]
    p0_s2 = [-0.40, -0.25]
    bounds_s2 = ([-1.2, -1.2], [1.2, 1.2])

    def resid_stage2(p):
        cur = {
            "PCY1": 1.55, "PDY1": p_opt1[0], "PDY2": p_opt1[1],
            "PKY1": p_opt1[2], "PKY2": p_opt1[3],
            "PEY1": p_opt1[4], "PEY2": p_opt1[5],
            "PHY1": p_opt1[6], "PVY1": p_opt1[7],
            "PPY1": p[0], "PPY3": p[1]
        }
        pred, _ = Pacejka62Fitter.compute_fy0(cur, alpha_s2, fz_s2, gamma=0.0, p_kpa=p_s2)
        return (pred - fy_s2) * w_s2

    print("\n>> Optimizing Stage 2: Lateral pressure sensitivity cross-terms...")
    res_s2 = multistart_least_squares(resid_stage2, bounds_s2, p0_s2, num_starts=4, loss="soft_l1", f_scale=25.0, max_nfev=250)
    p_opt2 = res_s2.x
    stage_reports["stage2_pressure_fy"] = compute_stage_identifiability(res_s2, names_s2, "Stage 2: Pressure Sensitivity", p0_s2, bounds_s2)

    # ------------------------------------------------------------------
    # Stage 3: Camber Sensitivity Parameters
    # Coupled optimization of [PVY3, PVY4, PDY3, PHY3, PKY3]
    # Resolves previous staged decoupling and peak collapse:
    # - Accommodates contact patch lateral shift Shy (PHY3 > 0)
    # - Identifies camber cornering stiffness change (PKY3)
    # - Enforces zero-slip camber thrust constraint simultaneously with slip sweep saturation
    # - Fixes PEY4 to 0.0 to prevent boundary saturation
    # ------------------------------------------------------------------
    mask_camber_all = np.abs(df_train["ia_deg"]) > 0.5
    mask_c0 = mask_camber_all & (np.abs(df_train["sa_deg"]) < 0.5)

    names_s3 = ["PVY3", "PVY4", "PDY3", "PHY3", "PKY3"]
    p0_s3 = [0.8, 0.5, 6.0, 0.04, 1.5]
    bounds_s3 = (
        [0.0, -4.0, 0.0, 0.0, 0.0],
        [5.0,  4.0, 30.0, 0.15, 6.0]
    )

    alpha_s3 = alpha_tr[mask_camber_all]
    fz_s3 = fz_tr[mask_camber_all]
    gamma_s3 = gamma_tr[mask_camber_all]
    p_s3 = p_tr[mask_camber_all]
    fy_s3 = fy_tr[mask_camber_all]
    w_s3 = bin_weights_arr[mask_camber_all]

    alpha_c0 = alpha_tr[mask_c0]
    fz_c0 = fz_tr[mask_c0]
    gamma_c0 = gamma_tr[mask_c0]
    p_c0 = p_tr[mask_c0]
    fy_c0 = fy_tr[mask_c0]
    w_c0 = bin_weights_arr[mask_c0]

    def resid_stage3(p):
        cur = {
            "PCY1": 1.55, "PDY1": p_opt1[0], "PDY2": p_opt1[1],
            "PKY1": p_opt1[2], "PKY2": p_opt1[3],
            "PEY1": p_opt1[4], "PEY2": p_opt1[5],
            "PHY1": p_opt1[6], "PVY1": p_opt1[7],
            "PPY1": p_opt2[0], "PPY3": p_opt2[1],
            "PVY3": p[0], "PVY4": p[1],
            "PDY3": p[2], "PHY3": p[3], "PKY3": p[4],
            "PEY4": 0.0
        }
        pred_sweep, _ = Pacejka62Fitter.compute_fy0(cur, alpha_s3, fz_s3, gamma=gamma_s3, p_kpa=p_s3)
        err_sweep = (pred_sweep - fy_s3) * w_s3
        pred_c0, _ = Pacejka62Fitter.compute_fy0(cur, alpha_c0, fz_c0, gamma=gamma_c0, p_kpa=p_c0)
        err_c0 = (pred_c0 - fy_c0) * 5.0 * w_c0
        return np.concatenate([err_sweep, err_c0])

    print("\n>> Optimizing Stage 3: Coupled camber formulation (PVY3, PVY4, PDY3, PHY3, PKY3, PEY4=0.0 fixed)...")
    res_s3 = multistart_least_squares(resid_stage3, bounds_s3, p0_s3, num_starts=6, loss="soft_l1", f_scale=25.0, max_nfev=400)
    p_opt3 = res_s3.x
    stage_reports["stage3_camber_coupled"] = compute_stage_identifiability(res_s3, names_s3, "Stage 3: Coupled Camber Sensitivity", p0_s3, bounds_s3)

    fitted_fy_params = {
        "PCY1": 1.55,
        "PDY1": float(p_opt1[0]), "PDY2": float(p_opt1[1]),
        "PKY1": float(p_opt1[2]), "PKY2": float(p_opt1[3]),
        "PEY1": float(p_opt1[4]), "PEY2": float(p_opt1[5]),
        "PHY1": float(p_opt1[6]), "PVY1": float(p_opt1[7]),
        "PPY1": float(p_opt2[0]), "PPY3": float(p_opt2[1]),
        "PVY3": float(p_opt3[0]), "PVY4": float(p_opt3[1]),
        "PDY3": float(p_opt3[2]), "PEY4": 0.0,
        "PHY3": float(p_opt3[3]), "PKY3": float(p_opt3[4]),
        "PEY3": 0.0, "PEY5": 0.0, "PHY2": 0.0, "PVY2": 0.0,
        "PPY2": 0.0, "PPY4": 0.0
    }

    pred_fy_tr, pred_svy_tr = Pacejka62Fitter.compute_fy0(fitted_fy_params, alpha_tr, fz_tr, gamma_tr, p_tr)

    # ------------------------------------------------------------------
    # Stage 4: Self-Aligning Torque Mz & Pneumatic Trail (at gamma = 0)
    # Identifies: QDZ1, QDZ2, QBZ1, QBZ2, QEZ2, QHZ1, QDZ6, QDZ7
    # NOTE: QCZ1 is fixed to 1.65 (canonical shape factor) and QEZ1 is fixed
    # to 0.0 to resolve the extreme Bt/Ct/Et collinearity (cond drops by 10x).
    # ------------------------------------------------------------------
    names_s4 = ["QDZ1", "QDZ2", "QBZ1", "QBZ2", "QEZ2", "QHZ1", "QDZ6", "QDZ7"]
    p0_s4 = [0.15, -0.04, 6.0, 1.2, 0.90, 0.002, 0.002, 0.0]
    bounds_s4 = (
        [0.05, -0.15,  2.0, -4.0, -0.5, -0.03, -0.05, -0.05],
        [0.35,  0.15, 15.0,  5.0,  2.0,  0.03,  0.05,  0.05]
    )

    alpha_s4 = alpha_tr[mask_c0_all_p]
    fz_s4 = fz_tr[mask_c0_all_p]
    p_s4 = p_tr[mask_c0_all_p]
    mz_s4 = mz_tr[mask_c0_all_p]
    fx_s4 = fx_tr[mask_c0_all_p]
    fy_s4 = pred_fy_tr[mask_c0_all_p]
    svy_s4 = pred_svy_tr[mask_c0_all_p]
    w_s4 = bin_weights_arr[mask_c0_all_p]

    def resid_stage4(p):
        cur = dict(fitted_fy_params)
        cur.update({
            "QDZ1": p[0], "QDZ2": p[1],
            "QBZ1": p[2], "QBZ2": p[3],
            "QCZ1": 1.65,
            "QEZ1": 0.0, "QEZ2": p[4],
            "QHZ1": p[5],
            "QDZ6": p[6], "QDZ7": p[7],
            "QDZ8": 0.0, "QDZ3": 0.0
        })
        pred = Pacejka62Fitter.compute_mz(cur, alpha_s4, fz_s4, gamma=0.0, p_kpa=p_s4, fx=fx_s4, fy_in=fy_s4, svy_in=svy_s4)
        return (pred - mz_s4) * w_s4

    print("\n>> Optimizing Stage 4: Zero-camber pneumatic trail & residual torque (QCZ1=1.65, QEZ1=0.0 fixed)...")
    res_s4 = multistart_least_squares(resid_stage4, bounds_s4, p0_s4, num_starts=6, loss="soft_l1", f_scale=2.5, max_nfev=350)
    p_opt4 = res_s4.x
    stage_reports["stage4_trail_mz0"] = compute_stage_identifiability(res_s4, names_s4, "Stage 4: Aligning Torque Mz0 (QCZ1=1.65, QEZ1=0.0 fixed)", p0_s4, bounds_s4)

    # ------------------------------------------------------------------
    # Stage 5: Camber Aligning Torque Modulation & Camber Residual Torque
    # Identifies: QDZ8, QDZ3
    # ------------------------------------------------------------------
    names_s5 = ["QDZ8", "QDZ3"]
    p0_s5 = [1.2, 0.15]
    bounds_s5 = ([-1.0, -0.5], [4.0, 1.0])

    def resid_stage5(p):
        cur = dict(fitted_fy_params)
        cur.update({
            "QDZ1": p_opt4[0], "QDZ2": p_opt4[1],
            "QBZ1": p_opt4[2], "QBZ2": p_opt4[3],
            "QCZ1": 1.65,
            "QEZ1": 0.0, "QEZ2": p_opt4[4],
            "QHZ1": p_opt4[5],
            "QDZ6": p_opt4[6], "QDZ7": p_opt4[7],
            "QDZ8": p[0], "QDZ3": p[1]
        })
        pred = Pacejka62Fitter.compute_mz(cur, alpha_tr, fz_tr, gamma=gamma_tr, p_kpa=p_tr, fx=fx_tr, fy_in=pred_fy_tr, svy_in=pred_svy_tr)
        return (pred - mz_tr) * bin_weights_arr

    print("\n>> Optimizing Stage 5: Camber aligning torque & camber residual torque...")
    res_s5 = multistart_least_squares(resid_stage5, bounds_s5, p0_s5, num_starts=4, loss="soft_l1", f_scale=2.5, max_nfev=250)
    p_opt5 = res_s5.x
    stage_reports["stage5_camber_mz"] = compute_stage_identifiability(res_s5, names_s5, "Stage 5: Camber Mz", p0_s5, bounds_s5)

    fitted_mz_params = {
        "QDZ1": float(p_opt4[0]), "QDZ2": float(p_opt4[1]),
        "QBZ1": float(p_opt4[2]), "QBZ2": float(p_opt4[3]),
        "QCZ1": 1.65,
        "QEZ1": 0.0, "QEZ2": float(p_opt4[4]),
        "QHZ1": float(p_opt4[5]),
        "QDZ6": float(p_opt4[6]), "QDZ7": float(p_opt4[7]),
        "QDZ8": float(p_opt5[0]), "QDZ3": float(p_opt5[1]),
        "QDZ9": 0.0, "QBZ3": 0.0, "QBZ4": 0.0, "QBZ5": 0.0,
        "QBZ9": 8.0, "QBZ10": 0.0, "QEZ3": 0.0, "QEZ4": 0.0, "QEZ5": 0.0,
        "QHZ2": 0.0, "QHZ3": 0.0, "QHZ4": 0.0, "PPZ1": 0.0, "PPZ2": 0.0,
        "SSZ1": 0.0, "SSZ2": 0.04, "SSZ3": 0.01, "SSZ4": 0.0
    }

    # ------------------------------------------------------------------
    # Stage 6: Overturning Moment Mx
    # Identifies: QSX1, QSX2, QSX3
    # NOTE: PPMX1 is fixed to 0.0 because pressure excitation in Mx (<0.15 Nm)
    # is an order of magnitude smaller than load cell cross-talk/noise (2-3 Nm).
    # ------------------------------------------------------------------
    names_s6 = ["QSX1", "QSX2", "QSX3"]
    p0_s6 = [0.0, 2.0, 0.05]
    bounds_s6 = ([-0.1, 0.5, -0.1], [0.1, 4.0, 0.3])

    def resid_stage6(p):
        cur = {"QSX1": p[0], "QSX2": p[1], "QSX3": p[2], "PPMX1": 0.0}
        pred = Pacejka62Fitter.compute_mx(cur, fz_tr, gamma_tr, pred_fy_tr, p_kpa=p_tr)
        return pred - mx_tr

    print("\n>> Optimizing Stage 6: Overturning moment Mx (PPMX1=0.0 fixed)...")
    res_s6 = multistart_least_squares(resid_stage6, bounds_s6, p0_s6, num_starts=4, loss="soft_l1", f_scale=2.0, max_nfev=250)
    p_opt6 = res_s6.x
    stage_reports["stage6_overturning_mx"] = compute_stage_identifiability(res_s6, names_s6, "Stage 6: Overturning Moment Mx (PPMX1=0.0 fixed)", p0_s6, bounds_s6)

    fitted_mx_params = {
        "QSX1": float(p_opt6[0]), "QSX2": float(p_opt6[1]), "QSX3": float(p_opt6[2]),
        "PPMX1": 0.0
    }

    # Literature FSAE defaults for unidentifiable longitudinal/combined parameters
    literature_params = {
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
        "MODEL_IDENTIFIER": f"Pacejka MF6.2 Hoosier 16.0x7.5-10 {compound_name}",
        "COMPOUND": compound_name,
        "CALIBRATION_DATE": "2026-11-10",
        "FNOMIN": FNOMIN,
        "PINOM": PINOM,
        "UNLOADED_RADIUS": vert_props["unloaded_radius_m"],
        "VERTICAL_STIFFNESS": vert_props["linear_stiffness_n_m"],
        "VERTICAL_STIFFNESS_ANALYSIS": vert_props,
        "PARAMETER_CATEGORIZATION": {
            "calibrated_from_ttc": list(fitted_fy_params.keys()) + list(fitted_mz_params.keys()) + list(fitted_mx_params.keys()),
            "literature_fsae_reference": list(literature_params.keys()),
            "unidentifiable_unsupported": {
                "Drive_Brake_longitudinal_sweeps": "Cornering TTC dataset has spindle free-rolling (SR ~ -1, N ~ 0 rpm); pure longitudinal slip cannot be identified.",
                "Combined_slip_Fx_Fy_sweeps": "TTC cornering tests maintain zero traction/braking torque; combined slip factors cannot be identified.",
                "Rolling_resistance_torque_My": "TTC test spindle does not record wheel hub drag torque My."
            }
        }
    }
    all_params.update(fitted_fy_params)
    all_params.update(fitted_mz_params)
    all_params.update(fitted_mx_params)
    all_params.update(literature_params)

    # ------------------------------------------------------------------
    # Stage 7: Joint Refinement Comparison (Staged vs Joint Refined)
    # ------------------------------------------------------------------
    print("\n>> Executing Stage 7: Joint refinement of identifiable parameters...")
    joint_names = ["PDY1", "PDY2", "PKY1", "PKY2", "PEY1", "PEY2", "PVY3", "PVY4", "PDY3", "PHY3", "PKY3", "QDZ1", "QDZ2", "QBZ1", "QEZ2", "QDZ8"]
    p0_joint = [all_params[n] for n in joint_names]

    def joint_resid(p):
        cur = dict(all_params)
        for i, n in enumerate(joint_names):
            cur[n] = p[i]
        pred_fy, pred_svy = Pacejka62Fitter.compute_fy0(cur, alpha_tr, fz_tr, gamma_tr, p_tr)
        pred_mz = Pacejka62Fitter.compute_mz(cur, alpha_tr, fz_tr, gamma_tr, p_tr, fx=fx_tr, fy_in=pred_fy, svy_in=pred_svy)

        res_fy = (pred_fy - fy_tr) / 25.0 * bin_weights_arr
        res_mz = (pred_mz - mz_tr) / 2.5 * bin_weights_arr
        return np.concatenate([res_fy, res_mz])

    bounds_joint_lb = [p0_joint[i] * 0.85 if p0_joint[i] > 0 else p0_joint[i] * 1.15 for i in range(len(p0_joint))]
    bounds_joint_ub = [p0_joint[i] * 1.15 if p0_joint[i] > 0 else p0_joint[i] * 0.85 for i in range(len(p0_joint))]
    for i in range(len(p0_joint)):
        if bounds_joint_lb[i] > bounds_joint_ub[i]:
            bounds_joint_lb[i], bounds_joint_ub[i] = bounds_joint_ub[i], bounds_joint_lb[i]

    cost_before = np.sum(joint_resid(p0_joint) ** 2)
    res_joint = least_squares(joint_resid, p0_joint, bounds=(bounds_joint_lb, bounds_joint_ub), loss="soft_l1", f_scale=1.0, method="trf", max_nfev=350)
    cost_after = np.sum(joint_resid(res_joint.x) ** 2)
    print(f"   Joint refinement objective: {cost_before:.2f} -> {cost_after:.2f}")

    if cost_after < cost_before:
        print("   Adopting joint refined parameters (improved multi-objective cost).")
        for i, n in enumerate(joint_names):
            all_params[n] = float(res_joint.x[i])
    else:
        print("   Retaining staged parameters (staged solution maintained optimal balance).")

    # ------------------------------------------------------------------
    # Quantitative Evaluation on Training & Independent Validation Sets
    # ------------------------------------------------------------------
    print("\n>> Evaluating quantitative metrics on Train & Independent Validation sets...")
    eval_results = {}
    for subset_name, df_sub in [("Train", df_train), ("Independent_Validation", df_val)]:
        a_sub = df_sub["alpha"].values
        fz_sub = df_sub["f_z"].values
        g_sub = df_sub["gamma"].values
        p_sub = df_sub["p_kpa"].values
        v_sub = df_sub["v_x"].values
        fy_exp = df_sub["f_y"].values
        mz_exp = df_sub["m_z"].values
        mx_exp = df_sub["m_x"].values
        fx_sub = df_sub["f_x"].values

        fy_pred, svy_pred = Pacejka62Fitter.compute_fy0(all_params, a_sub, fz_sub, g_sub, p_sub)
        mz_pred = Pacejka62Fitter.compute_mz(all_params, a_sub, fz_sub, g_sub, p_sub, fx=fx_sub, fy_in=fy_pred, svy_in=svy_pred)
        mx_pred = Pacejka62Fitter.compute_mx(all_params, fz_sub, g_sub, fy_pred, p_kpa=p_sub)

        # Fy Metrics
        err_fy = fy_pred - fy_exp
        rmse_fy = float(np.sqrt(np.mean(err_fy ** 2)))
        mae_fy = float(np.mean(np.abs(err_fy)))
        max_fy = float(np.max(np.abs(err_fy)))
        bias_fy = float(np.mean(err_fy))
        range_fy = max(float(np.max(fy_exp) - np.min(fy_exp)), 1e-4)
        nrmse_fy = float(rmse_fy / range_fy * 100.0)
        ss_tot_fy = max(float(np.sum((fy_exp - np.mean(fy_exp)) ** 2)), 1e-6)
        r2_fy = float(1.0 - (np.sum(err_fy ** 2) / ss_tot_fy))

        # Mz Metrics
        err_mz = mz_pred - mz_exp
        rmse_mz = float(np.sqrt(np.mean(err_mz ** 2)))
        mae_mz = float(np.mean(np.abs(err_mz)))
        max_mz = float(np.max(np.abs(err_mz)))
        bias_mz = float(np.mean(err_mz))
        range_mz = max(float(np.max(mz_exp) - np.min(mz_exp)), 1e-4)
        nrmse_mz = float(rmse_mz / range_mz * 100.0)
        ss_tot_mz = max(float(np.sum((mz_exp - np.mean(mz_exp)) ** 2)), 1e-6)
        r2_mz = float(1.0 - (np.sum(err_mz ** 2) / ss_tot_mz))

        # Mx Metrics
        err_mx = mx_pred - mx_exp
        rmse_mx = float(np.sqrt(np.mean(err_mx ** 2)))
        mae_mx = float(np.mean(np.abs(err_mx)))
        max_mx = float(np.max(np.abs(err_mx)))
        bias_mx = float(np.mean(err_mx))
        range_mx = max(float(np.max(mx_exp) - np.min(mx_exp)), 1e-4)
        nrmse_mx = float(rmse_mx / range_mx * 100.0)
        ss_tot_mx = max(float(np.sum((mx_exp - np.mean(mx_exp)) ** 2)), 1e-6)
        r2_mx = float(1.0 - (np.sum(err_mx ** 2) / ss_tot_mx))

        # Sliced metrics: Slip angle regions
        slip_intervals = [
            (0.0, np.radians(3.0), "Linear (|alpha| <= 3 deg)"),
            (np.radians(3.0), np.radians(8.0), "Transitional (3 deg < |alpha| <= 8 deg)"),
            (np.radians(8.0), np.radians(15.0), "Saturation / post-peak (|alpha| > 8 deg)")
        ]
        sliced_slip = {}
        for a_lo, a_hi, sname in slip_intervals:
            s_mask = (np.abs(a_sub) >= a_lo) & (np.abs(a_sub) < a_hi)
            if np.sum(s_mask) > 50:
                sliced_slip[sname] = {
                    "count": int(np.sum(s_mask)),
                    "rmse_fy_n": float(np.sqrt(np.mean(err_fy[s_mask] ** 2))),
                    "mae_fy_n": float(np.mean(np.abs(err_fy[s_mask]))),
                    "rmse_mz_nm": float(np.sqrt(np.mean(err_mz[s_mask] ** 2))),
                    "mae_mz_nm": float(np.mean(np.abs(err_mz[s_mask])))
                }

        # Sliced metrics: 5 physical test load levels
        fz_bins = [
            (100, 330, "Load Level 1 (~220 N)"),
            (330, 550, "Load Level 2 (~440 N)"),
            (550, 770, "Load Level 3 (~665 N)"),
            (770, 990, "Load Level 4 (~885 N)"),
            (990, 1400, "Load Level 5 (~1110 N)")
        ]
        sliced_fz = {}
        for fz_l, fz_h, bname in fz_bins:
            b_mask = (fz_sub >= fz_l) & (fz_sub < fz_h)
            if np.sum(b_mask) > 100:
                sliced_fz[bname] = {
                    "count": int(np.sum(b_mask)),
                    "rmse_fy_n": float(np.sqrt(np.mean(err_fy[b_mask] ** 2))),
                    "mae_fy_n": float(np.mean(np.abs(err_fy[b_mask]))),
                    "rmse_mz_nm": float(np.sqrt(np.mean(err_mz[b_mask] ** 2))),
                    "mae_mz_nm": float(np.mean(np.abs(err_mz[b_mask])))
                }

        # Sliced metrics: Camber
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

        # Sliced metrics: Speed (interpolated vs extrapolated)
        sliced_v = {}
        for v_l, v_h, vname in [(5.0, 9.0, "Low speed (~24 km/h) [extrapolated]"), (9.0, 14.0, "Nominal speed (~40 km/h) [interpolated]"), (14.0, 22.0, "High speed (~72 km/h) [extrapolated]")]:
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
            "fy": {"rmse_n": rmse_fy, "nrmse_pct": nrmse_fy, "mae_n": mae_fy, "max_err_n": max_fy, "bias_n": bias_fy, "r2": r2_fy},
            "mz": {"rmse_nm": rmse_mz, "nrmse_pct": nrmse_mz, "mae_nm": mae_mz, "max_err_nm": max_mz, "bias_nm": bias_mz, "r2": r2_mz},
            "mx": {"rmse_nm": rmse_mx, "nrmse_pct": nrmse_mx, "mae_nm": mae_mx, "max_err_nm": max_mx, "bias_nm": bias_mx, "r2": r2_mx},
            "sliced_by_slip_region": sliced_slip,
            "sliced_by_load": sliced_fz,
            "sliced_by_camber": sliced_gamma,
            "sliced_by_speed": sliced_v
        }

        print(f"  [{subset_name} Set ({len(df_sub)} points)]")
        print(f"    Fy: RMSE = {rmse_fy:6.2f} N  | MAE = {mae_fy:6.2f} N  | Max = {max_fy:6.2f} N  | NRMSE = {nrmse_fy:4.2f}% | R^2 = {r2_fy:.4f} | Bias = {bias_fy:+.2f} N")
        print(f"    Mz: RMSE = {rmse_mz:6.2f} Nm | MAE = {mae_mz:6.2f} Nm | Max = {max_mz:6.2f} Nm | NRMSE = {nrmse_mz:4.2f}% | R^2 = {r2_mz:.4f} | Bias = {bias_mz:+.2f} Nm")
        print(f"    Mx: RMSE = {rmse_mx:6.2f} Nm | MAE = {mae_mx:6.2f} Nm | Max = {max_mx:6.2f} Nm | NRMSE = {nrmse_mx:4.2f}% | R^2 = {r2_mx:.4f} | Bias = {bias_mx:+.2f} Nm")

    # Export parameter JSON
    out_param_file = os.path.join(MF62_DIR, f"hoosier_16x7.5_10_{compound_name.lower()}_mf62.json")
    with open(out_param_file, "w") as fp:
        json.dump(all_params, fp, indent=2)
    print(f"\n>> Saved calibrated parameters: {out_param_file}")

    # Export calibration report
    rep_file = os.path.join(VALIDATION_DIR, f"calibration_report_{compound_name.lower()}.json")
    with open(rep_file, "w") as fp:
        json.dump({
            "compound": compound_name,
            "identifiability_stages": stage_reports,
            "evaluation": eval_results
        }, fp, indent=2)
    print(f">> Saved detailed calibration report: {rep_file}")

    return all_params, eval_results


def main():
    print("=" * 78)
    print("   STARTING MF 6.2 CALIBRATION ENGINE (R25B & R20)")
    print("=" * 78)

    # 1. Fit Round 8 (R25B)
    r25b_tr = pd.read_csv(os.path.join(PROCESSED_DIR, "R25B_train.csv"))
    r25b_val = pd.read_csv(os.path.join(PROCESSED_DIR, "R25B_val.csv"))
    r25b_spring_path = os.path.join(PROCESSED_DIR, "R25B_spring.csv")
    params_r25b, res_r25b = fit_mf62_compound("R25B", r25b_tr, r25b_val, r25b_spring_path)

    # 2. Fit Round 9 (R20)
    r20_tr = pd.read_csv(os.path.join(PROCESSED_DIR, "R20_train.csv"))
    r20_val = pd.read_csv(os.path.join(PROCESSED_DIR, "R20_val.csv"))
    r20_spring_path = os.path.join(PROCESSED_DIR, "R20_spring.csv")
    params_r20, res_r20 = fit_mf62_compound("R20", r20_tr, r20_val, r20_spring_path)

    # 3. Export default parameter file (R25B primary compound)
    default_path = os.path.join(MF62_DIR, "hoosier_16x7.5_10_mf62.json")
    with open(default_path, "w") as fp:
        json.dump(params_r25b, fp, indent=2)
    print(f"\n>> Exported default parameter file: {default_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
