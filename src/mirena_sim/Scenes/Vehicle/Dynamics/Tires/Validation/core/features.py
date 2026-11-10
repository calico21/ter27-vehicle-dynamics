#!/usr/bin/env python3
"""
Tire Model Validation Framework — Feature Extraction Core
Extracts physically meaningful parameters: cornering stiffness, peak forces,
pneumatic trail, camber thrust, zero crossings, post-peak decay, and sensitivities.
"""

import numpy as np


def extract_lateral_features(alpha_rad, fy_n):
    """
    Extracts key lateral response features from Fy(alpha) curve.
    
    Args:
        alpha_rad: sorted array of slip angles in radians
        fy_n: corresponding lateral force values in Newtons
        
    Returns:
        dict with stiffness_n_rad, stiffness_n_deg, fy_max_n, alpha_peak_deg,
        post_peak_drop_pct, post_peak_slope_n_deg, fy_zero_n, asymmetry_pct
    """
    alpha_rad = np.asarray(alpha_rad, dtype=float)
    fy_n = np.asarray(fy_n, dtype=float)
    sort_idx = np.argsort(alpha_rad)
    a = alpha_rad[sort_idx]
    fy = fy_n[sort_idx]

    # 1. Cornering stiffness C_alpha = dFy/da at a=0 using points within +/- 1.0 deg
    mask_lin = np.abs(np.degrees(a)) <= 1.0
    if np.sum(mask_lin) >= 3:
        p = np.polyfit(a[mask_lin], fy[mask_lin], 1)
        stiffness_rad = float(p[0])
    else:
        idx_0 = np.argmin(np.abs(a))
        if idx_0 == 0:
            stiffness_rad = float((fy[1] - fy[0]) / (a[1] - a[0]))
        elif idx_0 == len(a) - 1:
            stiffness_rad = float((fy[-1] - fy[-2]) / (a[-1] - a[-2]))
        else:
            stiffness_rad = float((fy[idx_0 + 1] - fy[idx_0 - 1]) / (a[idx_0 + 1] - a[idx_0 - 1]))

    stiffness_deg = float(stiffness_rad * (np.pi / 180.0))

    # 2. Peak lateral force and peak slip angle
    idx_max = np.argmax(fy)
    fy_max = float(fy[idx_max])
    alpha_peak_deg = float(np.degrees(a[idx_max]))

    # Minimum force (negative peak)
    idx_min = np.argmin(fy)
    fy_min = float(fy[idx_min])
    alpha_min_deg = float(np.degrees(a[idx_min]))

    # Asymmetry
    denom = max(abs(fy_max) + abs(fy_min), 1e-4) / 2.0
    asymmetry_pct = float(abs(abs(fy_max) - abs(fy_min)) / denom * 100.0)

    # 3. Post-peak decay at |alpha| = 12 deg (or highest available angle)
    target_post = np.radians(12.0)
    idx_post = np.argmin(np.abs(a - target_post))
    fy_post = float(fy[idx_post])
    post_drop_pct = float(max(0.0, (fy_max - fy_post) / max(fy_max, 1e-4) * 100.0))
    delta_a_deg = max(np.degrees(a[idx_post] - a[idx_max]), 0.1)
    post_slope = float((fy_post - fy_max) / delta_a_deg)

    # 4. Zero slip force Fy(0)
    idx_zero = np.argmin(np.abs(a))
    fy_zero = float(fy[idx_zero])

    return {
        "stiffness_n_rad": stiffness_rad,
        "stiffness_n_deg": stiffness_deg,
        "fy_max_n": fy_max,
        "alpha_peak_deg": alpha_peak_deg,
        "fy_min_n": fy_min,
        "alpha_min_deg": alpha_min_deg,
        "asymmetry_pct": asymmetry_pct,
        "post_peak_drop_pct": post_drop_pct,
        "post_peak_slope_n_deg": post_slope,
        "fy_zero_n": fy_zero
    }


def extract_aligning_features(alpha_rad, mz_nm):
    """
    Extracts key aligning torque features: stiffness, peak Mz, zero crossing.
    """
    alpha_rad = np.asarray(alpha_rad, dtype=float)
    mz_nm = np.asarray(mz_nm, dtype=float)
    sort_idx = np.argsort(alpha_rad)
    a = alpha_rad[sort_idx]
    mz = mz_nm[sort_idx]

    # Initial aligning stiffness
    mask_lin = np.abs(np.degrees(a)) <= 1.0
    if np.sum(mask_lin) >= 3:
        p = np.polyfit(a[mask_lin], mz[mask_lin], 1)
        stiff_rad = float(p[0])
    else:
        idx_0 = np.argmin(np.abs(a))
        if 0 < idx_0 < len(a) - 1:
            stiff_rad = float((mz[idx_0 + 1] - mz[idx_0 - 1]) / (a[idx_0 + 1] - a[idx_0 - 1]))
        else:
            stiff_rad = 0.0

    stiff_deg = float(stiff_rad * (np.pi / 180.0))

    # Peaks
    idx_max = np.argmax(mz)
    mz_max = float(mz[idx_max])
    alpha_max_deg = float(np.degrees(a[idx_max]))

    idx_min = np.argmin(mz)
    mz_min = float(mz[idx_min])
    alpha_min_deg = float(np.degrees(a[idx_min]))

    # Zero crossing angle alpha_0 (where Mz crosses 0 after peak)
    # Looking for sign change in positive alpha region after max
    zero_crossing_deg = None
    post_max_mask = (a >= a[idx_max]) & (a <= np.radians(14.0))
    if np.sum(post_max_mask) > 1:
        a_sub = a[post_max_mask]
        mz_sub = mz[post_max_mask]
        signs = np.sign(mz_sub)
        sign_changes = np.where(np.diff(signs))[0]
        if len(sign_changes) > 0:
            sc_i = sign_changes[0]
            # Linear interpolation
            a0 = a_sub[sc_i] - mz_sub[sc_i] * (a_sub[sc_i+1] - a_sub[sc_i]) / (mz_sub[sc_i+1] - mz_sub[sc_i])
            zero_crossing_deg = float(np.degrees(a0))

    return {
        "aligning_stiffness_nm_rad": stiff_rad,
        "aligning_stiffness_nm_deg": stiff_deg,
        "mz_max_nm": mz_max,
        "alpha_max_deg": alpha_max_deg,
        "mz_min_nm": mz_min,
        "alpha_min_deg": alpha_min_deg,
        "zero_crossing_deg": zero_crossing_deg
    }


def extract_camber_features(gamma_rad, fy_camber_thrust_n):
    """
    Extracts camber thrust stiffness dFy/dgamma at gamma = 0.
    """
    gamma_rad = np.asarray(gamma_rad, dtype=float)
    fy = np.asarray(fy_camber_thrust_n, dtype=float)
    if len(gamma_rad) >= 2:
        p = np.polyfit(gamma_rad, fy, 1)
        stiff_rad = float(p[0])
        stiff_deg = float(stiff_rad * (np.pi / 180.0))
    else:
        stiff_rad = 0.0
        stiff_deg = 0.0
    return {
        "camber_thrust_stiffness_n_rad": stiff_rad,
        "camber_thrust_stiffness_n_deg": stiff_deg
    }


def extract_vertical_features(deflection_m, fz_n):
    """
    Extracts linear vertical stiffness kz and fit quality from deflection test.
    """
    deflection_m = np.asarray(deflection_m, dtype=float)
    fz_n = np.asarray(fz_n, dtype=float)
    if len(deflection_m) < 10:
        return {"kz_n_m": 0.0, "kz_n_mm": 0.0, "r2": 0.0, "rmse_mm": 0.0}

    # Constrained through origin: Fz = kz * rho
    kz = float(np.sum(deflection_m * fz_n) / max(np.sum(deflection_m ** 2), 1e-9))
    fz_pred = kz * deflection_m
    err_f = fz_pred - fz_n
    ss_tot = float(np.sum((fz_n - np.mean(fz_n)) ** 2))
    ss_res = float(np.sum(err_f ** 2))
    r2 = float(max(0.0, 1.0 - ss_res / max(ss_tot, 1e-6)))
    err_rho_mm = np.abs((fz_n / max(kz, 1e-4) - deflection_m) * 1000.0)
    rmse_mm = float(np.sqrt(np.mean(err_rho_mm ** 2)))

    return {
        "kz_n_m": kz,
        "kz_n_mm": kz / 1000.0,
        "r2": r2,
        "rmse_mm": rmse_mm
    }
