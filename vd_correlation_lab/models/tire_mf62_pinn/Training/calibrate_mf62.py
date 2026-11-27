#!/usr/bin/env python3
"""
Pacejka MF 6.2 Calibration & Identifiability Engine — Mirena Dynamics (Hoosier 16.0x7.5-10)
Parameter-file schema v2.

Frame: ISO 8855 (+X fwd, +Y left, +Z up). alpha, gamma in rad. Data are "left wheel as tested";
the right wheel is obtained at runtime by flipping conicity/ply-steer terms (see Pacejka62.gd).

Key improvements vs v1:
- R0 is a model constant (0.2032 m) shared by fitter and runtime. The spring-test intercept is stored
  only as a diagnostic (R0_SPRING_FIT_M).
- Mz follows canonical Pacejka 2012 / Besselink 2010 structure:
    Mz = -t(alpha_t,eq) * (Fy - SVyk) + Mzr(alpha_r,eq) + s * Fx
    alpha_t = alpha + SHt ; alpha_r = alpha + SHf ; SHf = SHy + SVy/Kya
    Dt = (Fz/Fz0) R0 (QDZ1 + QDZ2 dfz)(1 + PPZ1 dpi)(1 + QDZ3|gamma| + QDZ4 gamma^2)
    Br = QBZ9 + QBZ10 By Cy ; Dr carries cos(alpha)
- Combined slip: Gxa/Gyk denominators use +B*SH; SVyk = mu_y Fz (RVY1+RVY2 dfz+RVY3 gamma) cos(atan(RVY4 alpha))
  * sin(RVY5 atan(RVY6 kappa)).
- Mz uses the MODEL Fx (not measured Fx): measured Fx contains cornering drag the runtime does not
  reproduce, which biased Mz parameters and the NN residual.
- Stage 7 chooses staged vs joint by blocked (leave-condition-out) CV inside TRAIN only; validation is untouched.
- Standard errors are inflated by the lag-1 residual autocorrelation (effective-N correction).
  Optimizer status/nfev recorded; x_scale="jac".
- PVY3 lower bound released; QDZ3/QDZ8 bounds widened (QDZ3 sign: +QDZ3|gamma|).
- Exports CALIBRATION_ID (hash of parameters) and canonical .tir export for external simulation suites.
"""

import os
import sys
import json
import hashlib
import datetime
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

SCHEMA_VERSION = 2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TIRE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PROCESSED_DIR = os.path.join(TIRE_DIR, "Data/Hoosier_16x7.5-10/Processed")
DRIVE_BRAKE_PROCESSED_DIR = os.path.join(TIRE_DIR, "Data/Hoosier_18\u00d76.0-10/Processed")
MF62_DIR = os.path.join(TIRE_DIR, "MF62")
VALIDATION_DIR = os.path.join(TIRE_DIR, "Validation")

FNOMIN = 800.0          # N
PINOM = 83.0            # kPa (~12 psi)
R0_MODEL = 0.2032       # m, model constant (8.0 in)
UNLOADED_RADIUS = R0_MODEL  # backward-compatible alias
V0 = 11.11              # m/s
JOINT_CV_FOLDS = 4

# Single source of truth. Pacejka62.gd REQUIRED_KEYS must equal these keys.
ALL_DEFAULTS = {
    # Fy
    "PCY1": 1.55, "PDY1": 2.45, "PDY2": -0.30, "PDY3": 0.0, "PEY1": 0.35, "PEY2": -0.25, "PEY3": 0.0,
    "PEY4": 0.0, "PEY5": 0.0, "PKY1": 48.0, "PKY2": 2.10, "PKY3": 0.0, "PHY1": 0.0, "PHY2": 0.0,
    "PHY3": 0.0, "PVY1": 0.0, "PVY2": 0.0, "PVY3": 0.0, "PVY4": 0.0, "PPY1": 0.0, "PPY2": 0.0,
    "PPY3": 0.0, "PPY4": 0.0,
    # Fx (literature, not identifiable from cornering TTC)
    "PCX1": 1.55, "PDX1": 2.30, "PDX2": -0.32, "PDX3": 0.0, "PEX1": 0.35, "PEX2": 0.10, "PEX3": 0.0,
    "PEX4": 0.0, "PKX1": 46.0, "PKX2": -0.12, "PKX3": 0.0, "PHX1": 0.0, "PHX2": 0.0, "PVX1": 0.0,
    "PVX2": 0.0, "PPX1": 0.0, "PPX2": 0.0, "PPX3": 0.0, "PPX4": 0.0,
    # Combined slip (literature)
    "RBX1": 12.0, "RBX2": 10.0, "RBX3": 0.0, "RCX1": 1.05, "REX1": 0.10, "REX2": 0.0, "RHX1": 0.0,
    "RBY1": 9.5, "RBY2": 8.0, "RBY3": 0.0, "RBY4": 0.0, "RCY1": 1.02, "REY1": 0.10, "REY2": 0.0,
    "RHY1": 0.0, "RHY2": 0.0, "RVY1": 0.0, "RVY2": 0.0, "RVY3": 0.0, "RVY4": 0.0, "RVY5": 1.9, "RVY6": 10.0,
    # Mz
    "QDZ1": 0.15, "QDZ2": -0.04, "QDZ3": 0.0, "QDZ4": 0.0, "QDZ6": 0.0, "QDZ7": 0.0, "QDZ8": 0.0,
    "QDZ9": 0.0, "QBZ1": 6.0, "QBZ2": 1.2, "QBZ3": 0.0, "QBZ4": 0.0, "QBZ5": 0.0, "QBZ9": 8.0,
    "QBZ10": 0.0, "QCZ1": 1.65, "QEZ1": 0.0, "QEZ2": 0.90, "QEZ3": 0.0, "QEZ4": 0.0, "QEZ5": 0.0,
    "QHZ1": 0.0, "QHZ2": 0.0, "QHZ3": 0.0, "QHZ4": 0.0, "PPZ1": 0.0, "PPZ2": 0.0,
    "SSZ1": 0.0, "SSZ2": 0.04, "SSZ3": 0.01, "SSZ4": 0.0,
    # Mx
    "QSX1": 0.0, "QSX2": 2.0, "QSX3": 0.05, "PPMX1": 0.0,
    # My (literature)
    "QSY1": 0.015, "QSY2": 0.0, "QSY3": 0.001, "QSY4": 0.0,
}

LITERATURE_KEYS = [k for k in ALL_DEFAULTS if k[:3] == "QSY"]


def _p(params, key):
    return params.get(key, ALL_DEFAULTS[key])


def _shape_cos(b, c, e, x):
    bx = b * x
    return np.cos(c * np.arctan(bx - e * (bx - np.arctan(bx))))


def _sgn(x):
    return np.where(x >= 0.0, 1.0, -1.0)


class Pacejka62Fitter:
    """Vectorised MF 6.2 evaluation (numpy)."""

    @staticmethod
    def compute_fy0_full(params, alpha, f_z, gamma=0.0, p_kpa=PINOM, fnomin=FNOMIN, pinom=PINOM):
        g = lambda k: _p(params, k)
        dfz = (f_z - fnomin) / fnomin
        dpi = (p_kpa - pinom) / pinom
        Cy = g("PCY1")
        camber_f = np.maximum(1.0 - g("PDY3") * gamma * gamma, 0.2)
        mu_y = (g("PDY1") + g("PDY2") * dfz) * (1.0 + g("PPY3") * dpi + g("PPY4") * dpi * dpi) * camber_f
        Dy = mu_y * f_z
        Shy = g("PHY1") + g("PHY2") * dfz + g("PHY3") * gamma
        Svy = f_z * ((g("PVY1") + g("PVY2") * dfz) + (g("PVY3") + g("PVY4") * dfz) * gamma)
        alpha_y = alpha + Shy
        kya_arg = 2.0 * np.arctan(f_z / (g("PKY2") * fnomin * (1.0 + g("PPY2") * dpi) + 1e-4))
        Kya = g("PKY1") * fnomin * (1.0 + g("PPY1") * dpi) * (1.0 - g("PKY3") * np.abs(gamma)) * np.sin(kya_arg)
        By = Kya / (Cy * Dy + 1e-4)
        Ey = np.clip((g("PEY1") + g("PEY2") * dfz) *
                     (1.0 + g("PEY5") * gamma * gamma - (g("PEY3") + g("PEY4") * gamma) * np.sign(alpha_y)), -10.0, 1.0)
        y = By * alpha_y
        Fy0 = Dy * np.sin(Cy * np.arctan(y - Ey * (y - np.arctan(y)))) + Svy
        return {"Fy0": Fy0, "Svy": Svy, "Shy": Shy, "Kya": Kya, "By": By, "Cy": Cy, "mu_y": mu_y}

    @staticmethod
    def compute_fy0(params, alpha, f_z, gamma=0.0, p_kpa=PINOM, fnomin=FNOMIN, pinom=PINOM):
        d = Pacejka62Fitter.compute_fy0_full(params, alpha, f_z, gamma, p_kpa, fnomin, pinom)
        return d["Fy0"], d["Svy"]

    @staticmethod
    def compute_fx0_full(params, kappa, f_z, gamma=0.0, p_kpa=PINOM, fnomin=FNOMIN, pinom=PINOM):
        g = lambda k: _p(params, k)
        dfz = (f_z - fnomin) / fnomin
        dpi = (p_kpa - pinom) / pinom
        Cx = g("PCX1")
        mu_x = (g("PDX1") + g("PDX2") * dfz) * (1.0 + g("PPX3") * dpi + g("PPX4") * dpi * dpi) * np.maximum(1.0 - g("PDX3") * gamma * gamma, 0.2)
        Dx = mu_x * f_z
        kx = kappa + g("PHX1") + g("PHX2") * dfz
        Svx = f_z * (g("PVX1") + g("PVX2") * dfz)
        Kxk = f_z * (g("PKX1") + g("PKX2") * dfz) * np.exp(g("PKX3") * dfz) * (1.0 + g("PPX1") * dpi + g("PPX2") * dpi * dpi)
        Bx = Kxk / (Cx * Dx + 1e-4)
        Ex = np.clip((g("PEX1") + g("PEX2") * dfz + g("PEX3") * dfz * dfz) * (1.0 - g("PEX4") * np.sign(kx)), -10.0, 1.0)
        x = Bx * kx
        Fx0 = Dx * np.sin(Cx * np.arctan(x - Ex * (x - np.arctan(x)))) + Svx
        return Fx0, Kxk

    @staticmethod
    def compute_fx0(params, kappa, f_z, gamma=0.0, p_kpa=PINOM, fnomin=FNOMIN, pinom=PINOM):
        return Pacejka62Fitter.compute_fx0_full(params, kappa, f_z, gamma, p_kpa, fnomin, pinom)[0]

    @staticmethod
    def compute_combined_forces(params, kappa, alpha, f_z, gamma=0.0, p_kpa=PINOM, fnomin=FNOMIN, pinom=PINOM):
        """Returns dict: Fx, Fy, Fy0, Svy, Svyk, Shf, Kya, Kxk, By, Cy, Gyk."""
        g = lambda k: _p(params, k)
        fy = Pacejka62Fitter.compute_fy0_full(params, alpha, f_z, gamma, p_kpa, fnomin, pinom)
        Fx0, Kxk = Pacejka62Fitter.compute_fx0_full(params, kappa, f_z, gamma, p_kpa, fnomin, pinom)
        dfz = (f_z - fnomin) / fnomin

        Bxa = (g("RBX1") + g("RBX3") * gamma * gamma) * np.cos(np.arctan(g("RBX2") * kappa))
        Exa = np.clip(g("REX1") + g("REX2") * dfz, -10.0, 1.0)
        Shxa = g("RHX1")
        Gxa = np.maximum(_shape_cos(Bxa, g("RCX1"), Exa, alpha + Shxa) /
                         np.maximum(_shape_cos(Bxa, g("RCX1"), Exa, Shxa), 1e-4), 0.0)

        Byk = (g("RBY1") + g("RBY4") * gamma * gamma) * np.cos(np.arctan(g("RBY2") * (alpha - g("RBY3"))))
        Eyk = np.clip(g("REY1") + g("REY2") * dfz, -10.0, 1.0)
        Shyk = g("RHY1") + g("RHY2") * dfz
        kappa_s = kappa + Shyk
        Gyk = np.maximum(_shape_cos(Byk, g("RCY1"), Eyk, kappa_s) /
                         np.maximum(_shape_cos(Byk, g("RCY1"), Eyk, Shyk), 1e-4), 0.0)

        Dvyk = fy["mu_y"] * f_z * (g("RVY1") + g("RVY2") * dfz + g("RVY3") * gamma) * np.cos(np.arctan(g("RVY4") * alpha))
        Svyk = Dvyk * np.sin(g("RVY5") * np.arctan(g("RVY6") * kappa))

        Shf = fy["Shy"] + fy["Svy"] / (fy["Kya"] + 1e-3)
        return {"Fx": Gxa * Fx0, "Fy": Gyk * fy["Fy0"] + Svyk, "Fy0": fy["Fy0"], "Svy": fy["Svy"], "Svyk": Svyk,
                "Shf": Shf, "Kya": fy["Kya"], "Kxk": Kxk, "By": fy["By"], "Cy": fy["Cy"], "Gyk": Gyk}

    @staticmethod
    def compute_mz(params, alpha, f_z, gamma=0.0, p_kpa=PINOM, kappa=0.0, fx=None, comb=None,
                   fnomin=FNOMIN, pinom=PINOM, r0=R0_MODEL, is_right_side=False, **kwargs):
        """Mz = -t*(Fy - SVyk) + Mzr + s*Fx. `comb` = output of compute_combined_forces (computed if None).
        fx=None uses the model Fx (what the runtime has)."""
        g = lambda k: _p(params, k)
        if comb is None:
            comb = Pacejka62Fitter.compute_combined_forces(params, kappa, alpha, f_z, gamma, p_kpa, fnomin, pinom)
        Fy = comb["Fy"]
        Fx = comb["Fx"] if fx is None else fx
        dfz = (f_z - fnomin) / fnomin
        dpi = (p_kpa - pinom) / pinom

        Sht = g("QHZ1") + g("QHZ2") * dfz + (g("QHZ3") + g("QHZ4") * dfz) * gamma
        alpha_t = alpha + Sht
        alpha_r = alpha + comb["Shf"]
        kx = comb["Kxk"] / (comb["Kya"] + 1e-3) * kappa
        alpha_t_eq = _sgn(alpha_t) * np.sqrt(alpha_t ** 2 + kx ** 2)
        alpha_r_eq = _sgn(alpha_r) * np.sqrt(alpha_r ** 2 + kx ** 2)

        Dt = (f_z / fnomin) * r0 * (g("QDZ1") + g("QDZ2") * dfz) * (1.0 + g("PPZ1") * dpi) * \
             (1.0 + g("QDZ3") * np.abs(gamma) + g("QDZ4") * gamma * gamma)
        Ct = g("QCZ1")
        Bt = (g("QBZ1") + g("QBZ2") * dfz + g("QBZ3") * dfz * dfz) * (1.0 + g("QBZ4") * gamma + g("QBZ5") * np.abs(gamma))
        t_arg = Bt * alpha_t_eq
        Et = np.clip((g("QEZ1") + g("QEZ2") * dfz + g("QEZ3") * dfz * dfz) *
                     (1.0 + (g("QEZ4") + g("QEZ5") * gamma) * (2.0 / np.pi) * np.arctan(Bt * Ct * alpha_t_eq)), -10.0, 1.0)
        trail = Dt * np.cos(Ct * np.arctan(t_arg - Et * (t_arg - np.arctan(t_arg)))) * np.cos(alpha)
        Mz_prime = -trail * (Fy - comb["Svyk"])

        side = -1.0 if is_right_side else 1.0
        Dr = f_z * r0 * (side * (g("QDZ6") + g("QDZ7") * dfz) +
                         (g("QDZ8") + g("QDZ9") * dfz) * (1.0 + g("PPZ2") * dpi) * gamma) * np.cos(alpha)
        Br = g("QBZ9") + g("QBZ10") * comb["By"] * comb["Cy"]
        Mzr = Dr * np.cos(np.arctan(Br * alpha_r_eq))

        s = r0 * (g("SSZ1") + g("SSZ2") * (Fy / fnomin) + (g("SSZ3") + g("SSZ4") * dfz) * gamma)
        return Mz_prime + Mzr + s * Fx

    @staticmethod
    def compute_mx(params, f_z, gamma, fy, p_kpa=PINOM, fnomin=FNOMIN, pinom=PINOM, r0=R0_MODEL, is_right_side=False):
        dpi = (p_kpa - pinom) / pinom
        qsx1 = -_p(params, "QSX1") if is_right_side else _p(params, "QSX1")
        return f_z * r0 * (qsx1 + _p(params, "QSX2") * gamma +
                           _p(params, "QSX3") * (fy / fnomin) + _p(params, "PPMX1") * dpi)

    @staticmethod
    def compute_my(params, f_z, fx, v_x, fnomin=FNOMIN, r0=R0_MODEL, v0=V0, v_eps=0.5):
        v_ratio = np.abs(v_x) / max(v0, 1.0)
        rr = _p(params, "QSY1") + _p(params, "QSY2") * (fx / fnomin) + _p(params, "QSY3") * v_ratio + \
             _p(params, "QSY4") * v_ratio ** 4
        return -r0 * f_z * rr * np.tanh(v_x / v_eps)

    @staticmethod
    def estimate_vertical_properties(spring_csv_path, p_nom=PINOM):
        """Linear vertical stiffness kz. The intercept is a linear extrapolation to Fz=0 (diagnostic only)."""
        if not os.path.exists(spring_csv_path):
            raise FileNotFoundError(f"Spring rate dataset not found: {spring_csv_path}")
        df = pd.read_csv(spring_csv_path)
        fz, rl, p = df["f_z"].values, df["r_l"].values, df["p_kpa"].values

        m_nom = np.abs(p - p_nom) < 4.0
        if np.sum(m_nom) < 100:
            m_nom = np.ones(len(fz), dtype=bool)
        poly = np.polyfit(fz[m_nom], rl[m_nom], 1)
        kz_nom = float(-1.0 / poly[0])
        r_icpt = float(poly[1])
        err = rl[m_nom] - np.polyval(poly, fz[m_nom])
        ss_tot = float(np.sum((rl[m_nom] - np.mean(rl[m_nom])) ** 2))
        r2 = float(1.0 - np.sum(err ** 2) / max(ss_tot, 1e-6))

        kz_p = {}
        for p_val in [69.0, 83.0, 97.0]:
            m_p = np.abs(p - p_val) < 4.0
            if np.sum(m_p) > 100:
                kz_p[p_val] = float(-1.0 / np.polyfit(fz[m_p], rl[m_p], 1)[0])
        dp_slope = 0.0
        if len(kz_p) >= 2:
            dp_slope = float(np.polyfit(list(kz_p.keys()), list(kz_p.values()), 1)[0])
        return {
            "linear_stiffness_n_m": kz_nom,
            "intercept_radius_m_extrapolated": r_icpt,
            "stiffness_r2": r2,
            "deflection_rmse_mm": float(np.sqrt(np.mean(err ** 2)) * 1000.0),
            "fitting_range_fz_n": [float(np.min(fz[m_nom])), float(np.max(fz[m_nom]))],
            "pressure_sensitivity_npm_per_kpa": dp_slope,
            "stiffness_by_pressure": kz_p,
        }


def evaluate_pure_slip(params, alpha, fz, gamma, p_kpa, is_right_side=False):
    """Pure-slip (kappa=0) evaluation exactly as the runtime does it: model Fx, SVyk included in Mz.
    Returns (Fy, Mz, comb)."""
    alpha = np.asarray(alpha, dtype=float)
    comb = Pacejka62Fitter.compute_combined_forces(params, np.zeros_like(alpha), alpha, fz, gamma, p_kpa)
    mz = Pacejka62Fitter.compute_mz(params, alpha, fz, gamma, p_kpa, kappa=0.0, fx=None, comb=comb, is_right_side=is_right_side)
    return comb["Fy"], mz, comb


def calibration_id(params):
    core = {k: v for k, v in params.items() if k not in ("CALIBRATION_ID", "CALIBRATION_DATE")}
    return hashlib.sha256(json.dumps(core, sort_keys=True, default=float).encode()).hexdigest()[:16]


def export_tir(params, out_path):
    """Exports parameters to a canonical TYDEX / standard Pacejka .tir format for external CAD/Sim suites."""
    comp = str(params.get("COMPOUND", "HOOSIER_16X7.5_10")).upper()
    lines = [
        "[MDI_HEADER]",
        "FILE_TYPE                = 'tir'",
        "FILE_VERSION             = 3.0",
        "FILE_FORMAT              = 'ASCII'",
        f"! Hoosier 16.0x7.5-10 {comp} — Mirena Dynamics MF 6.2 Calibration",
        f"! Calibration ID: {params.get('CALIBRATION_ID', 'N/A')}",
        f"! Date: {params.get('CALIBRATION_DATE', datetime.date.today().isoformat())}",
        "$----------------------------------------------------------------units",
        "[UNITS]",
        "LENGTH                   = 'meter'",
        "FORCE                    = 'newton'",
        "ANGLE                    = 'radians'",
        "MASS                     = 'kg'",
        "TIME                     = 'second'",
        "$----------------------------------------------------------------model",
        "[MODEL]",
        "PROPERTY_FILE_FORMAT     = 'PAC2002'",
        "USE_MODE                 = 14",
        "VXLOW                    = 1.0",
        "LONGVL                   = 11.11",
        "TYRESIDE                 = 'LEFT'",
        "$------------------------------------------------------------dimension",
        "[DIMENSION]",
        f"UNLOADED_RADIUS          = {params.get('UNLOADED_RADIUS', R0_MODEL):.4f}",
        "WIDTH                    = 0.1905",
        "RIM_RADIUS               = 0.1270",
        "$-------------------------------------------------------------vertical",
        "[VERTICAL]",
        f"VERTICAL_STIFFNESS       = {params.get('VERTICAL_STIFFNESS', 115000.0):.1f}",
        f"FNOMIN                   = {params.get('FNOMIN', FNOMIN):.1f}",
        f"NOMPRES                  = {params.get('PINOM', PINOM) * 1000.0:.1f}",
        "$--------------------------------------------------scaling_coefficients",
        "[SCALING_COEFFICIENTS]",
        "LFZO                     = 1.0",
        "LCX                      = 1.0",
        "LMUX                     = 1.0",
        "LEX                      = 1.0",
        "LKX                      = 1.0",
        "LHX                      = 1.0",
        "LVX                      = 1.0",
        "LCY                      = 1.0",
        "LMUY                     = 1.0",
        "LEY                      = 1.0",
        "LKY                      = 1.0",
        "LHY                      = 1.0",
        "LVY                      = 1.0",
        "LTR                      = 1.0",
        "LRES                     = 1.0",
        "LXAL                     = 1.0",
        "LYKA                     = 1.0",
        "LVYKA                    = 1.0",
        "LS                       = 1.0",
        "$---------------------------------------------------------longitudinal",
        "[LONGITUDINAL_COEFFICIENTS]",
    ]
    for k in sorted([k for k in params if k[:2] in ("PC", "PD", "PE", "PK", "PH", "PV", "PP") and k[2] == "X"]):
        lines.append(f"{k:<25}= {params[k]:.6g}")
    lines.append("$--------------------------------------------------------------lateral")
    lines.append("[LATERAL_COEFFICIENTS]")
    for k in sorted([k for k in params if k[:2] in ("PC", "PD", "PE", "PK", "PH", "PV", "PP") and k[2] == "Y"]):
        lines.append(f"{k:<25}= {params[k]:.6g}")
    lines.append("$-------------------------------------------------------combined_slip")
    for k in sorted([k for k in params if k[:2] in ("RB", "RC", "RE", "RH", "RV")]):
        lines.append(f"{k:<25}= {params[k]:.6g}")
    lines.append("$------------------------------------------------------------aligning")
    lines.append("[ALIGNING_COEFFICIENTS]")
    for k in sorted([k for k in params if k[:2] in ("QD", "QB", "QC", "QE", "QH", "SS") and k[2] == "Z"] + ["PPZ1", "PPZ2"]):
        lines.append(f"{k:<25}= {params[k]:.6g}")
    lines.append("$---------------------------------------------------------overturning")
    lines.append("[OVERTURNING_COEFFICIENTS]")
    for k in sorted([k for k in params if k[:2] == "QS" and k[2] == "X"] + ["PPMX1"]):
        lines.append(f"{k:<25}= {params[k]:.6g}")
    lines.append("$---------------------------------------------------rolling_resistance")
    lines.append("[ROLLING_COEFFICIENTS]")
    for k in sorted([k for k in params if k[:2] == "QS" and k[2] == "Y"]):
        lines.append(f"{k:<25}= {params[k]:.6g}")

    with open(out_path, "w") as fp:
        fp.write("\n".join(lines) + "\n")
    print(f">> Exported canonical .tir: {out_path}")


def _sub(d, m):
    return {k: (v[m] if isinstance(v, np.ndarray) else v) for k, v in d.items()}


def multistart_least_squares(resid_func, bounds, default_p0, num_starts=6, loss="soft_l1", f_scale=25.0,
                             max_nfev=350, seed=42):
    lb = np.array(bounds[0], dtype=float)
    ub = np.array(bounds[1], dtype=float)
    default_p0 = np.array(default_p0, dtype=float)
    rng = np.random.RandomState(seed)
    candidates = [default_p0, 0.5 * (lb + ub)] + [rng.uniform(lb, ub) for _ in range(num_starts)]

    costs = []
    for pt in candidates:
        try:
            costs.append(float(np.sum(resid_func(pt) ** 2)))
        except Exception:
            costs.append(float("inf"))
    top = [candidates[i] for i in np.argsort(costs)[:3]]
    if not any(np.allclose(default_p0, c) for c in top):
        top.append(default_p0)

    best, best_cost = None, float("inf")
    for p0_try in top:
        try:
            res = least_squares(resid_func, p0_try, bounds=bounds, loss=loss, f_scale=f_scale, method="trf",
                                x_scale="jac", max_nfev=max_nfev)
            if res.cost < best_cost:
                best, best_cost = res, res.cost
        except Exception:
            continue
    if best is None:
        best = least_squares(resid_func, default_p0, bounds=bounds, loss=loss, f_scale=f_scale, method="trf",
                             x_scale="jac", max_nfev=max_nfev)
    return best


def compute_stage_identifiability(res, param_names, stage_name, p0_scale, bounds=None):
    """Scaled-Jacobian SVD, correlations, SEs (inflated by lag-1 residual autocorrelation), bound saturation."""
    J = res.jac
    N, P = J.shape
    p_opt = res.x

    scale_vec = np.abs(np.array(p0_scale, dtype=float))
    scale_vec = np.where(scale_vec < 1e-4, 1.0, scale_vec)
    S = np.linalg.svd(J * scale_vec[np.newaxis, :], full_matrices=False)[1]
    cond_num = float(S[0] / max(S[-1], 1e-12))
    rank = int(np.sum(S > 1e-5 * S[0]))

    inv_JTJ = np.linalg.pinv(J.T @ J)
    r = res.fun - np.mean(res.fun)
    rho = float(np.clip(np.dot(r[:-1], r[1:]) / max(np.dot(r, r), 1e-12), 0.0, 0.999))
    inflation = (1.0 + rho) / (1.0 - rho)
    cov = inv_JTJ * float(np.sum(res.fun ** 2) / max(N - P, 1)) * inflation
    std_errors = np.sqrt(np.maximum(np.diag(cov), 0.0))

    d = np.sqrt(np.maximum(np.diag(inv_JTJ), 1e-12))
    corr = inv_JTJ / np.outer(d, d)
    high_corrs = [{"param_i": param_names[i], "param_j": param_names[j], "correlation": float(corr[i, j])}
                  for i in range(P) for j in range(i + 1, P) if abs(corr[i, j]) > 0.85]

    saturated = []
    if bounds is not None:
        lb, ub = bounds
        for i, name in enumerate(param_names):
            span = ub[i] - lb[i]
            if span > 1e-6:
                if (p_opt[i] - lb[i]) / span < 0.01:
                    saturated.append({"param": name, "bound": "lower", "value": float(p_opt[i]), "limit": float(lb[i])})
                elif (ub[i] - p_opt[i]) / span < 0.01:
                    saturated.append({"param": name, "bound": "upper", "value": float(p_opt[i]), "limit": float(ub[i])})

    diag = {n: {"optimal_value": float(p_opt[i]), "std_error": float(std_errors[i]),
                "relative_uncertainty_pct": float(std_errors[i] / max(abs(p_opt[i]), 1e-4) * 100.0)}
            for i, n in enumerate(param_names)}

    print(f"\n--- Identifiability: {stage_name} ---")
    print(f"   Rank {rank}/{P} | scaled cond {cond_num:.2e} | status {res.status} nfev {res.nfev} | "
          f"lag-1 rho {rho:.3f} (SE x{np.sqrt(inflation):.2f})")
    if res.status == 0:
        print("   [WARNING] max_nfev reached: not converged.")
    for bs in saturated:
        print(f"   [WARNING] {bs['param']} at {bs['bound']} bound: {bs['value']:.4g} (limit {bs['limit']:.4g})")
    for hc in high_corrs:
        print(f"   corr |r|>0.85: {hc['param_i']} <-> {hc['param_j']}: {hc['correlation']:+.3f}")

    return {"stage": stage_name, "jacobian_rank": rank, "num_params": P, "scaled_condition_number": cond_num,
            "singular_values": S.tolist(), "optimizer_status": int(res.status), "nfev": int(res.nfev),
            "residual_lag1_autocorr": rho, "se_inflation_factor": float(np.sqrt(inflation)),
            "boundary_saturated_parameters": saturated, "high_correlations": high_corrs, "parameters": diag}


def _err_block(err, exp):
    rng = max(float(np.max(exp) - np.min(exp)), 1e-4)
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_tot = max(float(np.sum((exp - np.mean(exp)) ** 2)), 1e-6)
    return {"rmse": rmse, "nrmse_pct": rmse / rng * 100.0, "mae": float(np.mean(np.abs(err))),
            "max_err": float(np.max(np.abs(err))), "bias": float(np.mean(err)),
            "r2": float(1.0 - np.sum(err ** 2) / ss_tot)}


def _slice(err_fy, err_mz, mask, min_count):
    if np.sum(mask) <= min_count:
        return None
    return {"count": int(np.sum(mask)), "rmse_fy_n": float(np.sqrt(np.mean(err_fy[mask] ** 2))),
            "mae_fy_n": float(np.mean(np.abs(err_fy[mask]))), "rmse_mz_nm": float(np.sqrt(np.mean(err_mz[mask] ** 2))),
            "mae_mz_nm": float(np.mean(np.abs(err_mz[mask])))}


def fit_longitudinal_and_combined():
    """
    Fits pure longitudinal (Fx0) and identifiable combined-slip (Gxa, Gyk) MF 6.2 parameters
    using Calspan TTC Round 9 Drive/Brake R20 data (Hoosier 18.0x6.0-10, 6-in & 7-in rims)
    with strict sweep-grouped train/validation splitting.
    """
    pure_csv = os.path.join(DRIVE_BRAKE_PROCESSED_DIR, "R20_18x6_pure_longitudinal.csv")
    comb_csv = os.path.join(DRIVE_BRAKE_PROCESSED_DIR, "R20_18x6_combined_slip.csv")
    if not (os.path.exists(pure_csv) and os.path.exists(comb_csv)):
        from preprocess_ttc import run_drive_brake_pipeline
        run_drive_brake_pipeline()

    print(f"\n{'=' * 78}\n   CALIBRATING LONGITUDINAL & COMBINED-SLIP MF 6.2 (TTC ROUND 9 R20 DRIVE/BRAKE)\n{'=' * 78}")
    df_pure = pd.read_csv(pure_csv, low_memory=False)
    df_comb = pd.read_csv(comb_csv, low_memory=False)
    df_ptr = df_pure[df_pure["split"] == "train"].copy()
    df_pva = df_pure[df_pure["split"] == "val"].copy()
    df_ctr = df_comb[df_comb["split"] == "train"].copy()
    df_cva = df_comb[df_comb["split"] == "val"].copy()

    def _lb_weights(fz_arr):
        bins = np.digitize(fz_arr, [400.0, 770.0, 990.0])
        w = 1.0 / np.sqrt(np.maximum(np.bincount(bins, minlength=4)[bins], 1))
        return w / np.mean(w)

    k_tr, fz_tr, g_tr, p_tr, fx_tr = (df_ptr[c].values for c in ("kappa", "f_z", "gamma", "p_kpa", "f_x"))
    w_tr = _lb_weights(fz_tr)

    # Neutral pure-longitudinal prior (fixed unidentifiable shifts/asymmetries at 0.0)
    base_fx = dict(ALL_DEFAULTS)
    for z in ["PDX3", "PEX2", "PEX3", "PEX4", "PKX3", "PHX1", "PHX2", "PVX1", "PVX2", "PPX1", "PPX2", "PPX3", "PPX4"]:
        base_fx[z] = 0.0
    base_fx["PCX1"] = 1.55

    # Stage L1: Zero-camber pure longitudinal force + pressure scaling (PCX1=1.55 fixed)
    m_l1 = df_ptr["ia_deg"].abs().values < 0.2
    names_l1 = ["PDX1", "PDX2", "PKX1", "PKX2", "PEX1", "PPX1", "PPX3", "PPX4"]
    p0_l1 = [2.15, -0.05, 40.0, -7.5, 0.10, -0.68, -0.28, 0.60]
    b_l1 = ([1.4, -0.8, 20.0, -25.0, -2.0, -1.5, -1.2, -1.5],
            [3.2,  0.4, 80.0,  25.0,  0.95, 0.5,  0.5,  1.5])

    def r_l1(p):
        cur = {**base_fx, **dict(zip(names_l1, p))}
        pred = Pacejka62Fitter.compute_fx0(cur, k_tr[m_l1], fz_tr[m_l1], gamma=0.0, p_kpa=p_tr[m_l1])
        return (pred - fx_tr[m_l1]) * w_tr[m_l1]

    print("\n>> Stage 8 (Longitudinal L1): pure Fx0 + pressure (PCX1=1.55 fixed)")
    res_l1 = multistart_least_squares(r_l1, b_l1, p0_l1, 6, f_scale=40.0, max_nfev=350, seed=42)
    diag_l1 = compute_stage_identifiability(res_l1, names_l1, "Stage 8: Pure Fx0 & Pressure", p0_l1, b_l1)

    # Stage L2: Camber friction reduction PDX3
    m_l2 = df_ptr["ia_deg"].abs().values > 0.5
    base_l1 = {**base_fx, **dict(zip(names_l1, [float(v) for v in res_l1.x]))}
    names_l2 = ["PDX3"]
    p0_l2, b_l2 = [6.5], ([-2.0], [25.0])

    def r_l2(p):
        cur = {**base_l1, "PDX3": p[0]}
        pred = Pacejka62Fitter.compute_fx0(cur, k_tr[m_l2], fz_tr[m_l2], gamma=g_tr[m_l2], p_kpa=p_tr[m_l2])
        return (pred - fx_tr[m_l2]) * w_tr[m_l2]

    print("\n>> Stage 9 (Longitudinal L2): camber friction PDX3")
    res_l2 = multistart_least_squares(r_l2, b_l2, p0_l2, 4, f_scale=40.0, max_nfev=200, seed=43)
    diag_l2 = compute_stage_identifiability(res_l2, names_l2, "Stage 9: Camber Fx (PDX3)", p0_l2, b_l2)

    staged_fx = {**base_l1, "PDX3": float(res_l2.x[0])}

    # 4-fold sweep-grouped CV inside TRAIN: staged vs joint refinement
    joint_names = names_l1 + names_l2
    p0_j = np.array([staged_fx[n] for n in joint_names], dtype=float)
    lb_j = [b_l1[0][i] for i in range(len(names_l1))] + [b_l2[0][0]]
    ub_j = [b_l1[1][i] for i in range(len(names_l1))] + [b_l2[1][0]]
    sweeps_tr = sorted(df_ptr["global_sweep_id"].unique())
    perm_sw = np.random.RandomState(42).permutation(sweeps_tr)
    sw_to_fold = {sw: i % 4 for i, sw in enumerate(perm_sw)}
    fold_tr = df_ptr["global_sweep_id"].map(sw_to_fold).values

    def _cv_resid(p, mask):
        cur = {**base_fx, **dict(zip(joint_names, p))}
        pred = Pacejka62Fitter.compute_fx0(cur, k_tr[mask], fz_tr[mask], g_tr[mask], p_tr[mask])
        return (pred - fx_tr[mask]) * w_tr[mask] / 40.0

    def _robust_cost(r):
        return float(np.sum(2.0 * (np.sqrt(1.0 + r * r) - 1.0)))

    cv_staged, cv_joint = [], []
    for f_i in range(4):
        ho = fold_tr == f_i
        rk = least_squares(lambda p: _cv_resid(p, ~ho), p0_j, bounds=(lb_j, ub_j),
                           loss="soft_l1", f_scale=1.0, method="trf", x_scale="jac", max_nfev=200)
        cv_staged.append(_robust_cost(_cv_resid(p0_j, ho)) / np.sum(ho))
        cv_joint.append(_robust_cost(_cv_resid(rk.x, ho)) / np.sum(ho))

    adopt_joint = float(np.mean(cv_joint)) < 0.995 * float(np.mean(cv_staged))
    if adopt_joint:
        rj = multistart_least_squares(lambda p: _cv_resid(p, np.ones(len(df_ptr), dtype=bool)) * 40.0,
                                      (lb_j, ub_j), p0_j, 4, f_scale=40.0, max_nfev=350, seed=44)
        final_fx = {**base_fx, **dict(zip(joint_names, [float(v) for v in rj.x]))}
    else:
        final_fx = dict(staged_fx)
    print(f"   Longitudinal 4-fold Sweep-CV cost/sample: staged={np.mean(cv_staged):.5f} vs joint={np.mean(cv_joint):.5f} "
          f"-> {'ADOPT joint' if adopt_joint else 'KEEP staged'}")

    # Combined-Slip Stages C1 (Gxa) & C2 (Gyk)
    k_ct, a_ct, fz_ct, g_ct, p_ct, fx_ct = (df_ctr[c].values for c in ("kappa", "alpha", "f_z", "gamma", "p_kpa", "f_x"))
    w_ct = _lb_weights(fz_ct)
    fx0_ct = Pacejka62Fitter.compute_fx0(final_fx, k_ct, fz_ct, g_ct, p_ct)

    base_comb = {**final_fx}
    for z in ["RBX3", "REX2", "RHX1", "RBY3", "RBY4", "REY2", "RHY2", "RVY1", "RVY2", "RVY3", "RVY4"]:
        base_comb[z] = 0.0
    base_comb.update({"RCX1": 1.05, "REX1": 0.10, "RCY1": 1.02, "REY1": 0.10, "RVY5": 1.9, "RVY6": 10.0})

    names_c1, p0_c1, b_c1 = ["RBX1", "RBX2"], [12.0, 10.0], ([2.0, 0.0], [30.0, 30.0])

    def r_c1(p):
        bxa = p[0] * np.cos(np.arctan(p[1] * k_ct))
        gxa = np.maximum(_shape_cos(bxa, base_comb["RCX1"], base_comb["REX1"], a_ct), 0.0)
        return (gxa * fx0_ct - fx_ct) * w_ct

    print("\n>> Stage 10 (Combined C1): longitudinal weighting Gxa [RBX1, RBX2]")
    res_c1 = multistart_least_squares(r_c1, b_c1, p0_c1, 6, f_scale=40.0, max_nfev=250, seed=45)
    diag_c1 = compute_stage_identifiability(res_c1, names_c1, "Stage 10: Combined Gxa", p0_c1, b_c1)

    # Sweep-normalized Fy0 for model-free Gyk identification
    def _add_fy0_sw(df_in):
        refs = {
            sw: float((s.loc[s["kappa"].abs() < 0.015, "f_y"] / s.loc[s["kappa"].abs() < 0.015, "f_z"]).mean())
            for sw, s in df_in.groupby("global_sweep_id") if (s["kappa"].abs() < 0.015).sum() >= 3
        }
        return df_in["global_sweep_id"].map(refs).values * df_in["f_z"].values

    fy0_sw_ct = _add_fy0_sw(df_ctr)
    fy0_sw_va = _add_fy0_sw(df_cva)
    fy_ct = df_ctr["f_y"].values
    names_c2, p0_c2, b_c2 = ["RBY1", "RBY2", "RHY1"], [14.0, 12.0, 0.01], ([2.0, 0.0, -0.05], [25.0, 25.0, 0.05])

    def r_c2(p):
        byk = p[0] * np.cos(np.arctan(p[1] * a_ct))
        gyk = np.maximum(
            _shape_cos(byk, base_comb["RCY1"], base_comb["REY1"], k_ct + p[2]) /
            np.maximum(_shape_cos(byk, base_comb["RCY1"], base_comb["REY1"], p[2]), 1e-4),
            0.0,
        )
        return (gyk * fy0_sw_ct - fy_ct) * w_ct

    print("\n>> Stage 11 (Combined C2): lateral weighting Gyk [RBY1, RBY2, RHY1]")
    res_c2 = multistart_least_squares(r_c2, b_c2, p0_c2, 6, f_scale=30.0, max_nfev=250, seed=46)
    diag_c2 = compute_stage_identifiability(res_c2, names_c2, "Stage 11: Combined Gyk", p0_c2, b_c2)

    long_comb_keys = [k for k in ALL_DEFAULTS if k.startswith(("PCX", "PDX", "PEX", "PKX", "PHX", "PVX", "PPX", "RB", "RC", "RE", "RH", "RV"))]
    fitted_lc = {**base_comb, **dict(zip(names_c1, [float(v) for v in res_c1.x])), **dict(zip(names_c2, [float(v) for v in res_c2.x]))}
    long_comb_params = {k: float(fitted_lc[k]) for k in long_comb_keys}

    # Evaluation summary (Train & Val) vs Literature Baseline
    def _m_blk(y_t, y_p):
        b = _err_block(y_p - y_t, y_t)
        b["n"] = int(len(y_t))
        return b

    eval_summary = {}
    for s_nm, df_p, df_c, fy0_sw in [("Train", df_ptr, df_ctr, fy0_sw_ct), ("Validation", df_pva, df_cva, fy0_sw_va)]:
        p_fx_cal = Pacejka62Fitter.compute_fx0({**ALL_DEFAULTS, **long_comb_params}, df_p["kappa"].values, df_p["f_z"].values, df_p["gamma"].values, df_p["p_kpa"].values)
        p_fx_lit = Pacejka62Fitter.compute_fx0(ALL_DEFAULTS, df_p["kappa"].values, df_p["f_z"].values, df_p["gamma"].values, df_p["p_kpa"].values)
        c_cal = Pacejka62Fitter.compute_combined_forces({**ALL_DEFAULTS, **long_comb_params}, df_c["kappa"].values, df_c["alpha"].values, df_c["f_z"].values, df_c["gamma"].values, df_c["p_kpa"].values)
        c_lit = Pacejka62Fitter.compute_combined_forces({**ALL_DEFAULTS, **{k: v for k, v in long_comb_params.items() if k[:2] in ("PC", "PD", "PE", "PK", "PH", "PV", "PP")}}, df_c["kappa"].values, df_c["alpha"].values, df_c["f_z"].values, df_c["gamma"].values, df_c["p_kpa"].values)
        eval_summary[s_nm] = {
            "pure_fx_samples": int(len(df_p)),
            "pure_fx_calibrated": _m_blk(df_p["f_x"].values, p_fx_cal),
            "pure_fx_baseline_lit": _m_blk(df_p["f_x"].values, p_fx_lit),
            "combined_samples": int(len(df_c)),
            "combined_fx_calibrated": _m_blk(df_c["f_x"].values, c_cal["Fx"]),
            "combined_fx_baseline_lit": _m_blk(df_c["f_x"].values, c_lit["Fx"]),
            "combined_fy_gyk_sweep_ref_calibrated": _m_blk(df_c["f_y"].values, c_cal["Gyk"] * fy0_sw),
            "combined_fy_gyk_sweep_ref_baseline_lit": _m_blk(df_c["f_y"].values, c_lit["Gyk"] * fy0_sw + c_lit["Svyk"]),
        }
    print(f"  [Longitudinal Val] Fx RMSE {eval_summary['Validation']['pure_fx_calibrated']['rmse']:.2f} N "
          f"(R2 {eval_summary['Validation']['pure_fx_calibrated']['r2']:.4f} vs Lit {eval_summary['Validation']['pure_fx_baseline_lit']['rmse']:.1f} N) | "
          f"Comb Gyk Fy RMSE {eval_summary['Validation']['combined_fy_gyk_sweep_ref_calibrated']['rmse']:.2f} N "
          f"(vs Lit {eval_summary['Validation']['combined_fy_gyk_sweep_ref_baseline_lit']['rmse']:.1f} N)")

    stages_lc = {
        "stage8_longitudinal_fx0": diag_l1,
        "stage9_longitudinal_camber_pdx3": diag_l2,
        "stage10_combined_gxa": diag_c1,
        "stage11_combined_gyk": diag_c2,
    }
    return long_comb_params, stages_lc, eval_summary


def fit_mf62_compound(compound_name, df_train, df_val, spring_csv_path, long_comb_bundle=None):
    print(f"\n{'=' * 78}\n   CALIBRATING MF 6.2: HOOSIER 16.0x7.5-10 {compound_name} (schema v{SCHEMA_VERSION})\n{'=' * 78}")

    alpha_tr, fz_tr, gamma_tr, p_tr = (df_train[c].values for c in ("alpha", "f_z", "gamma", "p_kpa"))
    fy_tr, mz_tr, mx_tr = (df_train[c].values for c in ("f_y", "m_z", "m_x"))

    vert = Pacejka62Fitter.estimate_vertical_properties(spring_csv_path, p_nom=PINOM)
    print(f">> kz = {vert['linear_stiffness_n_m']:.1f} N/m (R2 {vert['stiffness_r2']:.4f}); "
          f"R0 model = {R0_MODEL} m (spring intercept {vert['intercept_radius_m_extrapolated']:.4f} m, diagnostic)")

    load_bins = np.digitize(fz_tr, [330.0, 550.0, 770.0, 990.0])
    w_all = 1.0 / np.sqrt(np.maximum(np.bincount(load_bins, minlength=5)[load_bins], 1))
    w_all = w_all / np.mean(w_all)

    reports = {}

    # ---- Stage 1: zero camber, nominal pressure. PCY1 fixed (collinear with PEY1 otherwise).
    m1 = (np.abs(df_train["ia_deg"]) < 0.2) & (np.abs(p_tr - PINOM) < 4.0)
    a1, f1, y1, w1 = alpha_tr[m1], fz_tr[m1], fy_tr[m1], w_all[m1]
    names1 = ["PDY1", "PDY2", "PKY1", "PKY2", "PEY1", "PEY2", "PHY1", "PVY1"]
    p01 = [2.50, -0.30, 48.0, 2.05, 0.40, -0.20, 0.001, 0.025]
    b1 = ([1.6, -1.0, 20.0, 0.8, -1.2, -1.2, -0.02, -0.08], [3.4, 0.1, 90.0, 4.5, 0.9, 0.9, 0.02, 0.08])

    def r1(p):
        cur = dict(zip(names1, p))
        cur["PCY1"] = 1.55
        pred, _ = Pacejka62Fitter.compute_fy0(cur, a1, f1, gamma=0.0, p_kpa=PINOM)
        return (pred - y1) * w1

    print("\n>> Stage 1: pure Fy0 (PCY1=1.55 fixed)")
    res1 = multistart_least_squares(r1, b1, p01, 6, max_nfev=350)
    po1 = res1.x
    reports["stage1_zero_camber_fy"] = compute_stage_identifiability(res1, names1, "Stage 1: Pure Fy0", p01, b1)

    # ---- Stage 2: pressure sensitivity
    m2 = np.abs(df_train["ia_deg"]) < 0.2
    a2, f2, pp2, y2, w2 = alpha_tr[m2], fz_tr[m2], p_tr[m2], fy_tr[m2], w_all[m2]
    names2, p02, b2 = ["PPY1", "PPY3"], [-0.40, -0.25], ([-1.2, -1.2], [1.2, 1.2])
    base1 = dict(zip(names1, po1))
    base1["PCY1"] = 1.55

    def r2(p):
        cur = dict(base1)
        cur.update({"PPY1": p[0], "PPY3": p[1]})
        pred, _ = Pacejka62Fitter.compute_fy0(cur, a2, f2, gamma=0.0, p_kpa=pp2)
        return (pred - y2) * w2

    print("\n>> Stage 2: pressure sensitivity")
    res2 = multistart_least_squares(r2, b2, p02, 4, max_nfev=250)
    po2 = res2.x
    reports["stage2_pressure_fy"] = compute_stage_identifiability(res2, names2, "Stage 2: Pressure", p02, b2)

    # ---- Stage 3: camber. PVY3 lower bound released (was pinned at 0 in v1). PEY4 fixed at 0.
    m3 = np.abs(df_train["ia_deg"]) > 0.5
    m3c = m3 & (np.abs(df_train["sa_deg"]) < 0.5)
    names3 = ["PVY3", "PVY4", "PDY3", "PHY3", "PKY3"]
    p03 = [0.3, 0.5, 6.0, 0.04, 1.5]
    b3 = ([-1.0, -4.0, 0.0, 0.0, 0.0], [5.0, 4.0, 30.0, 0.15, 6.0])
    base2 = dict(base1)
    base2.update({"PPY1": po2[0], "PPY3": po2[1], "PEY4": 0.0})

    def r3(p):
        cur = dict(base2)
        cur.update(dict(zip(names3, p)))
        ps, _ = Pacejka62Fitter.compute_fy0(cur, alpha_tr[m3], fz_tr[m3], gamma_tr[m3], p_tr[m3])
        pc, _ = Pacejka62Fitter.compute_fy0(cur, alpha_tr[m3c], fz_tr[m3c], gamma_tr[m3c], p_tr[m3c])
        return np.concatenate([(ps - fy_tr[m3]) * w_all[m3], (pc - fy_tr[m3c]) * 5.0 * w_all[m3c]])

    print("\n>> Stage 3: coupled camber (PVY3, PVY4, PDY3, PHY3, PKY3)")
    res3 = multistart_least_squares(r3, b3, p03, 6, max_nfev=400)
    po3 = res3.x
    reports["stage3_camber_coupled"] = compute_stage_identifiability(res3, names3, "Stage 3: Camber", p03, b3)

    fitted_fy = dict(base2)
    fitted_fy.update(dict(zip(names3, [float(v) for v in po3])))
    fitted_fy = {k: float(v) for k, v in fitted_fy.items()}
    comb_tr = Pacejka62Fitter.compute_combined_forces(fitted_fy, np.zeros_like(alpha_tr), alpha_tr, fz_tr, gamma_tr, p_tr)
    pred_fy_tr = comb_tr["Fy"]

    # ---- Stage 4: zero-camber Mz. QCZ1=1.65, QEZ1=0 fixed. Model Fx used.
    names4 = ["QDZ1", "QDZ2", "QBZ1", "QBZ2", "QEZ2", "QHZ1", "QDZ6", "QDZ7"]
    p04 = [0.15, -0.04, 6.0, 1.2, 0.90, 0.002, 0.002, 0.0]
    b4 = ([0.05, -0.15, 2.0, -4.0, -0.5, -0.03, -0.05, -0.05], [0.35, 0.15, 15.0, 5.0, 2.0, 0.03, 0.05, 0.05])
    a4, f4, g4, pp4, y4, w4 = alpha_tr[m2], fz_tr[m2], gamma_tr[m2], p_tr[m2], mz_tr[m2], w_all[m2]
    comb4 = _sub(comb_tr, m2)

    def r4(p):
        cur = dict(fitted_fy)
        cur.update(dict(zip(names4, p)))
        cur.update({"QCZ1": 1.65, "QEZ1": 0.0, "QDZ8": 0.0, "QDZ3": 0.0})
        return (Pacejka62Fitter.compute_mz(cur, a4, f4, g4, pp4, comb=comb4) - y4) * w4

    print("\n>> Stage 4: pneumatic trail & residual torque at zero camber")
    res4 = multistart_least_squares(r4, b4, p04, 6, f_scale=2.5, max_nfev=350)
    po4 = res4.x
    reports["stage4_trail_mz0"] = compute_stage_identifiability(res4, names4, "Stage 4: Mz0", p04, b4)

    # ---- Stage 5: camber Mz. QDZ3 now enters as (1 + QDZ3|gamma|): sign flipped vs v1.
    names5, p05, b5 = ["QDZ8", "QDZ3"], [1.2, 0.5], ([-3.0, -4.0], [10.0, 12.0])
    base4 = dict(fitted_fy)
    base4.update(dict(zip(names4, [float(v) for v in po4])))
    base4.update({"QCZ1": 1.65, "QEZ1": 0.0})

    def r5(p):
        cur = dict(base4)
        cur.update({"QDZ8": p[0], "QDZ3": p[1]})
        return (Pacejka62Fitter.compute_mz(cur, alpha_tr, fz_tr, gamma_tr, p_tr, comb=comb_tr) - mz_tr) * w_all

    print("\n>> Stage 5: camber aligning torque")
    res5 = multistart_least_squares(r5, b5, p05, 4, f_scale=2.5, max_nfev=250)
    po5 = res5.x
    reports["stage5_camber_mz"] = compute_stage_identifiability(res5, names5, "Stage 5: Camber Mz", p05, b5)

    # ---- Stage 6: Mx. PPMX1 fixed (excitation below load-cell crosstalk).
    names6, p06, b6 = ["QSX1", "QSX2", "QSX3"], [0.0, 2.0, 0.05], ([-0.1, 0.5, -0.1], [0.1, 4.0, 0.3])

    def r6(p):
        cur = dict(zip(names6, p))
        cur["PPMX1"] = 0.0
        return Pacejka62Fitter.compute_mx(cur, fz_tr, gamma_tr, pred_fy_tr, p_kpa=p_tr) - mx_tr

    print("\n>> Stage 6: overturning moment")
    res6 = multistart_least_squares(r6, b6, p06, 4, f_scale=2.0, max_nfev=250)
    po6 = res6.x
    reports["stage6_overturning_mx"] = compute_stage_identifiability(res6, names6, "Stage 6: Mx", p06, b6)

    # ---- Assemble
    all_params = {
        "MODEL_IDENTIFIER": f"Pacejka MF6.2 Hoosier 16.0x7.5-10 {compound_name}",
        "SCHEMA_VERSION": SCHEMA_VERSION,
        "COMPOUND": compound_name,
        "CALIBRATION_DATE": datetime.date.today().isoformat(),
        "FNOMIN": FNOMIN,
        "PINOM": PINOM,
        "UNLOADED_RADIUS": R0_MODEL,
        "R0_SPRING_FIT_M": vert["intercept_radius_m_extrapolated"],
        "VERTICAL_STIFFNESS": vert["linear_stiffness_n_m"],
        "FZ_VALID_MAX": float(np.percentile(fz_tr, 99.5)),
        "VERTICAL_STIFFNESS_ANALYSIS": vert,
    }
    all_params.update(ALL_DEFAULTS)
    if long_comb_bundle is None:
        long_comb_bundle = fit_longitudinal_and_combined()
    long_comb_params, long_comb_stages, long_eval_summary = long_comb_bundle
    all_params.update(long_comb_params)
    all_params.update(fitted_fy)
    all_params.update({"QDZ8": float(po5[0]), "QDZ3": float(po5[1]), "QCZ1": 1.65, "QEZ1": 0.0})
    all_params.update(dict(zip(names4, [float(v) for v in po4])))
    all_params.update(dict(zip(names6, [float(v) for v in po6])))
    all_params["PPMX1"] = 0.0
    fitted_lat_keys = list(names1) + ["PCY1"] + names2 + names3 + ["PEY4"] + names4 + names5 + names6
    fitted_long_comb_keys = ["PDX1", "PDX2", "PDX3", "PKX1", "PKX2", "PEX1", "PPX1", "PPX3", "PPX4",
                             "RBX1", "RBX2", "RBY1", "RBY2", "RHY1"]
    all_params["PARAMETER_CATEGORIZATION"] = {
        "calibrated_from_ttc": fitted_lat_keys + fitted_long_comb_keys,
        "calibrated_lateral_from_ttc_16x7_5": fitted_lat_keys,
        "calibrated_longitudinal_combined_from_ttc_r9_r20_18x6": fitted_long_comb_keys,
        "longitudinal_compound_source": (
            "TTC Round 9 R20 Drive/Brake (18.0x6.0-10, 6-in & 7-in rims) integrated for R20 same compound."
            if compound_name.upper() == "R20"
            else "R20 TTC Round 9 Drive/Brake surrogate priors (no R25B Drive/Brake TTC dataset exists)."
        ),
        "fixed_or_provisional_unidentifiable": {
            "PCX1": "1.55 (collinear with PEX1)",
            "PKX3_PEX2_PEX3_PEX4_PHX_PVX_PPX2": "0.0 (collinear / unidentifiable on stabilized Drive/Brake sweeps)",
            "RCX1_REX1_RCY1_REY1": "1.05, 0.10, 1.02, 0.10 (provisional across only SA in {-3, -6} deg)",
            "RBX3_REX2_RHX1_RBY3_RBY4_REY2_RHY2_RVY1_4": "0.0 (provisional; no +SA or |SA|>6 deg coverage)",
        },
        "literature_fsae_reference": LITERATURE_KEYS,
        "unidentifiable_unsupported": {
            "Rolling_resistance_My": "TTC spindle does not record hub drag torque.",
            "Combined_slip_positive_and_high_SA": "Round 9 Drive/Brake covers SA in {-3, -6} deg only.",
            "Surface_scaling": "No lambda_mu factors: flat-trac grip is used as-is (overestimates asphalt grip).",
        },
    }
    reports.update(long_comb_stages)

    # ---- Stage 7: staged vs joint, decided by blocked CV inside TRAIN (validation untouched).
    print("\n>> Stage 7: joint refinement, blocked CV on train")
    joint_names = ["PDY1", "PDY2", "PKY1", "PKY2", "PEY1", "PEY2", "PVY3", "PVY4", "PDY3", "PHY3", "PKY3",
                   "QDZ1", "QDZ2", "QBZ1", "QEZ2", "QDZ8"]
    p0j = np.array([all_params[n] for n in joint_names], dtype=float)
    delta = np.maximum(0.15 * np.abs(p0j), 1e-3)
    bj = (p0j - delta, p0j + delta)

    def factory(sl):
        def f(p):
            cur = dict(all_params)
            cur.update(zip(joint_names, p))
            fy, mz, _ = evaluate_pure_slip(cur, alpha_tr[sl], fz_tr[sl], gamma_tr[sl], p_tr[sl])
            return np.concatenate([(fy - fy_tr[sl]) / 25.0 * w_all[sl], (mz - mz_tr[sl]) / 2.5 * w_all[sl]])
        return f

    def robust(r):
        return float(np.sum(2.0 * (np.sqrt(1.0 + r * r) - 1.0)))

    n = len(alpha_tr)
    edges = np.linspace(0, n, JOINT_CV_FOLDS + 1).astype(int)
    cv_staged, cv_joint = [], []
    for k in range(JOINT_CV_FOLDS):
        held = np.zeros(n, dtype=bool)
        held[edges[k]:edges[k + 1]] = True
        rk = least_squares(factory(~held), p0j, bounds=bj, loss="soft_l1", f_scale=1.0, method="trf",
                           x_scale="jac", max_nfev=120)
        f_held = factory(held)
        cv_staged.append(robust(f_held(p0j)) / held.sum())
        cv_joint.append(robust(f_held(rk.x)) / held.sum())
    cv_s, cv_j = float(np.mean(cv_staged)), float(np.mean(cv_joint))
    adopt = cv_j < 0.99 * cv_s
    print(f"   Held-out robust cost/sample: staged {cv_s:.4f} | joint {cv_j:.4f} -> "
          f"{'ADOPT joint (refit on all train)' if adopt else 'KEEP staged'}")
    if adopt:
        rj = least_squares(factory(np.ones(n, dtype=bool)), p0j, bounds=bj, loss="soft_l1", f_scale=1.0,
                           method="trf", x_scale="jac", max_nfev=350)
        for nme, v in zip(joint_names, rj.x):
            all_params[nme] = float(v)
    reports["stage7_joint_cv"] = {"cv_cost_staged": cv_staged, "cv_cost_joint": cv_joint, "adopted_joint": bool(adopt),
                                  "note": "staged params were fit on all train rows, so the comparison is biased toward staged"}

    # ---- Evaluation
    print("\n>> Evaluating Train / Independent Validation")
    evals = {}
    for subset, df_sub in [("Train", df_train), ("Independent_Validation", df_val)]:
        a, fz, g, pk, vx = (df_sub[c].values for c in ("alpha", "f_z", "gamma", "p_kpa", "v_x"))
        fy_e, mz_e, mx_e = (df_sub[c].values for c in ("f_y", "m_z", "m_x"))
        fy_p, mz_p, _ = evaluate_pure_slip(all_params, a, fz, g, pk)
        mx_p = Pacejka62Fitter.compute_mx(all_params, fz, g, fy_p, p_kpa=pk)
        efy, emz = fy_p - fy_e, mz_p - mz_e
        sl = {}
        for lo, hi, nm in [(0.0, 3.0, "Linear (|alpha| <= 3 deg)"), (3.0, 8.0, "Transitional (3-8 deg)"),
                           (8.0, 15.0, "Saturation (>8 deg)")]:
            s = _slice(efy, emz, (np.abs(a) >= np.radians(lo)) & (np.abs(a) < np.radians(hi)), 50)
            if s:
                sl[nm] = s
        sf = {}
        for lo, hi, nm in [(100, 330, "Load 1 (~220 N)"), (330, 550, "Load 2 (~440 N)"), (550, 770, "Load 3 (~665 N)"),
                           (770, 990, "Load 4 (~885 N)"), (990, 1400, "Load 5 (~1110 N)")]:
            s = _slice(efy, emz, (fz >= lo) & (fz < hi), 100)
            if s:
                sf[nm] = s
        sg = {}
        for c in (0.0, 2.0, 4.0):
            s = _slice(efy, emz, np.abs(df_sub["ia_deg"].values - c) < 0.5, 100)
            if s:
                sg[f"Camber {c} deg"] = s
        sv = {}
        for lo, hi, nm in [(5.0, 9.0, "Low speed (~24 km/h) [extrapolated]"), (9.0, 14.0, "Nominal (~40 km/h)"),
                           (14.0, 22.0, "High speed (~72 km/h) [extrapolated]")]:
            s = _slice(efy, emz, (vx >= lo) & (vx < hi), 100)
            if s:
                sv[nm] = s
        evals[subset] = {"num_samples": len(df_sub), "fy": _err_block(efy, fy_e), "mz": _err_block(emz, mz_e),
                         "mx": _err_block(mx_p - mx_e, mx_e), "sliced_by_slip_region": sl, "sliced_by_load": sf,
                         "sliced_by_camber": sg, "sliced_by_speed": sv}
        e = evals[subset]
        print(f"  [{subset}] Fy RMSE {e['fy']['rmse']:.1f} N (bias {e['fy']['bias']:+.1f}, R2 {e['fy']['r2']:.4f}) | "
              f"Mz RMSE {e['mz']['rmse']:.2f} Nm (R2 {e['mz']['r2']:.4f}) | Mx RMSE {e['mx']['rmse']:.2f} Nm")

    all_params["CALIBRATION_ID"] = calibration_id(all_params)

    os.makedirs(MF62_DIR, exist_ok=True)
    os.makedirs(VALIDATION_DIR, exist_ok=True)
    out = os.path.join(MF62_DIR, f"hoosier_16x7.5_10_{compound_name.lower()}_mf62.json")
    with open(out, "w") as fp:
        json.dump(all_params, fp, indent=2)
    print(f"\n>> Saved {out}  [CALIBRATION_ID {all_params['CALIBRATION_ID']}]")
    
    # Export canonical .tir file
    tir_out = os.path.join(MF62_DIR, f"hoosier_16x7.5_10_{compound_name.lower()}.tir")
    export_tir(all_params, tir_out)
    
    with open(os.path.join(VALIDATION_DIR, f"calibration_report_{compound_name.lower()}.json"), "w") as fp:
        json.dump({"compound": compound_name, "calibration_id": all_params["CALIBRATION_ID"],
                   "identifiability_stages": reports, "evaluation": evals,
                   "longitudinal_evaluation": long_eval_summary}, fp, indent=2)
    return all_params, evals


def main():
    print("=" * 78 + "\n   MF 6.2 CALIBRATION ENGINE (R25B & R20), schema v%d\n" % SCHEMA_VERSION + "=" * 78)
    os.makedirs(MF62_DIR, exist_ok=True)
    os.makedirs(VALIDATION_DIR, exist_ok=True)
    long_comb_bundle = fit_longitudinal_and_combined()
    params = {}
    for comp in ("R25B", "R20"):
        tr = pd.read_csv(os.path.join(PROCESSED_DIR, f"{comp}_train.csv"))
        va = pd.read_csv(os.path.join(PROCESSED_DIR, f"{comp}_val.csv"))
        params[comp], _ = fit_mf62_compound(comp, tr, va, os.path.join(PROCESSED_DIR, f"{comp}_spring.csv"),
                                            long_comb_bundle=long_comb_bundle)
    default_path = os.path.join(MF62_DIR, "hoosier_16x7.5_10_mf62.json")
    with open(default_path, "w") as fp:
        json.dump(params["R25B"], fp, indent=2)
    default_tir = os.path.join(MF62_DIR, "hoosier_16x7.5_10.tir")
    export_tir(params["R25B"], default_tir)
    print(f"\n>> Default parameter file (R25B): {default_path}")


if __name__ == "__main__":
    main()
