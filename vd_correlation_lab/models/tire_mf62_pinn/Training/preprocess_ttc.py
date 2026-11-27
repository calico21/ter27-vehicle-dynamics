#!/usr/bin/env python3
"""
TTC Data Preprocessing & Conditioning Pipeline for Mirena Dynamics
Senior Tire Dynamics Engineering Implementation

Processes Calspan Tire Test Consortium (TTC) Round 8 (R25B) and Round 9 (R20) files:
- Cornering Calibration: R25B Run 6 (B1965run6.dat), R20 Run 8 (B2356run8.dat)
- Independent Validation: R25B Run 7 (B1965run7.dat), R20 Run 9 (B2356run9.dat)
- Vertical Characterization: R25B Run 5 (B1965run5.dat), R20 Run 7 (B2356run7.dat)

Key Features:
- Rigorous coordinate transformation from Calspan SAE to ISO 8855 / target simulation:
    Normal force:       f_z = -FZ  (compressive normal load positive, N)
    Lateral force:      f_y = -FY  (positive slip alpha produces positive lateral force f_y, N)
    Longitudinal force: f_x = FX   (forward drive force positive, N)
    Aligning moment:    m_z = -MZ  (restoring torque about upward +Z axis is negative for alpha > 0, Nm)
    Overturning moment: m_x = MX   (moment about forward +X axis, Nm)
    Slip angle:         alpha = radians(SA) (rad)
    Inclination angle:  gamma = -radians(IA) (top tilt left is positive, rad)
    Velocity:           v_x = V / 3.6 (m/s)
    Pressure:           p_kpa = P (gauge pressure, kPa)
    Loaded radius:      r_l = RL * 0.01 (m)
- Contiguous sweep segmentation based on timestamp continuity (gap dt > 1.0 s)
- Physics-based actuator turnaround transient detection (|dSA/dt| > 15 deg/s) per sweep
- Provenance retention for every row (source file, round, run, raw_row_idx, sweep_id)
- Separation of cornering datasets from vertical characterization (spring rate) datasets
- Strict data integrity validation with zero tolerance for corrupted channels
- Exports machine-readable dataset manifest and filtering statistics
"""

import os
import sys
import json
import argparse
import shutil
import numpy as np
import pandas as pd

# Repository-relative path resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../../../../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

TIRE_DATA_DIR = os.path.join(REPO_ROOT, "data/ttc_tire/Hoosier_16x7.5-10")
RAW_DIR = os.path.join(TIRE_DATA_DIR, "Raw")
PROCESSED_DIR = os.path.join(TIRE_DATA_DIR, "Processed")
METADATA_DIR = os.path.join(TIRE_DATA_DIR, "Metadata")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)

RUN_MAP = {
    "R25B_train": {
        "file": os.path.join(RAW_DIR, "Round8_R25B", "B1965run6.dat"),
        "compound": "R25B",
        "round": 8,
        "run": 6,
        "category": "cornering_calibration",
        "description": "Round 8 R25B Cornering calibration test at 40 kph (multi-load, multi-camber, multi-pressure)"
    },
    "R25B_val": {
        "file": os.path.join(RAW_DIR, "Round8_R25B", "B1965run7.dat"),
        "compound": "R25B",
        "round": 8,
        "run": 7,
        "category": "cornering_validation",
        "description": "Round 8 R25B Independent cornering validation test with speed sweeps (24-72 kph) & low pressure"
    },
    "R25B_spring": {
        "file": os.path.join(RAW_DIR, "Round8_R25B", "B1965run5.dat"),
        "compound": "R25B",
        "round": 8,
        "run": 5,
        "category": "vertical_characterization",
        "description": "Round 8 R25B Vertical load-deflection and radial stiffness characterization"
    },
    "R20_train": {
        "file": os.path.join(RAW_DIR, "Round9_R20", "B2356run8.dat"),
        "compound": "R20",
        "round": 9,
        "run": 8,
        "category": "cornering_calibration",
        "description": "Round 9 R20 Cornering calibration test at 40 kph (multi-load, multi-camber, multi-pressure)"
    },
    "R20_val": {
        "file": os.path.join(RAW_DIR, "Round9_R20", "B2356run9.dat"),
        "compound": "R20",
        "round": 9,
        "run": 9,
        "category": "cornering_validation",
        "description": "Round 9 R20 Independent cornering validation test with speed sweeps (24-72 kph) & low pressure"
    },
    "R20_spring": {
        "file": os.path.join(RAW_DIR, "Round9_R20", "B2356run7.dat"),
        "compound": "R20",
        "round": 9,
        "run": 7,
        "category": "vertical_characterization",
        "description": "Round 9 R20 Vertical load-deflection and radial stiffness characterization"
    }
}

REQUIRED_CHANNELS = ["ET", "V", "SA", "IA", "FZ", "FY", "FX", "MZ", "MX", "P", "RL"]
DRIVE_BRAKE_EXTRA_CHANNELS = ["N", "RE", "SR", "SL"]

DRIVE_BRAKE_DIR = os.path.join(REPO_ROOT, "data/ttc_tire/Hoosier_18\u00d76.0-10")
DRIVE_BRAKE_PROCESSED_DIR = os.path.join(DRIVE_BRAKE_DIR, "Processed")
DRIVE_BRAKE_METADATA_DIR = os.path.join(DRIVE_BRAKE_DIR, "Metadata")

DRIVE_BRAKE_RUN_MAP = {
    "R20_18x6_6in_warmup": {
        "file": os.path.join(DRIVE_BRAKE_DIR, "B2356run68.dat"),
        "compound": "R20", "round": 9, "run": 68, "rim_width_in": 6.0,
        "role": "warmup", "has_sl": False,
        "description": "Round 9 R20 18x6.0-10 (6-in rim) warmup and conditioning run"
    },
    "R20_18x6_6in_pressures": {
        "file": os.path.join(DRIVE_BRAKE_DIR, "B2356run69.dat"),
        "compound": "R20", "round": 9, "run": 69, "rim_width_in": 6.0,
        "role": "drive_brake_pressure_sweeps", "has_sl": True,
        "description": "Round 9 R20 18x6.0-10 (6-in rim) initial 12 psi, 10 psi, 14 psi Drive/Brake sweeps"
    },
    "R20_18x6_6in_8psi_speed": {
        "file": os.path.join(DRIVE_BRAKE_DIR, "B2356run70.dat"),
        "compound": "R20", "round": 9, "run": 70, "rim_width_in": 6.0,
        "role": "drive_brake_8psi_final12_speed", "has_sl": True,
        "description": "Round 9 R20 18x6.0-10 (6-in rim) 8 psi, final 12 psi, 24/72 kph Drive/Brake sweeps"
    },
    "R20_18x6_7in_warmup": {
        "file": os.path.join(DRIVE_BRAKE_DIR, "B2356run71.dat"),
        "compound": "R20", "round": 9, "run": 71, "rim_width_in": 7.0,
        "role": "warmup", "has_sl": False,
        "description": "Round 9 R20 18x6.0-10 (7-in rim) warmup and conditioning run"
    },
    "R20_18x6_7in_pressures": {
        "file": os.path.join(DRIVE_BRAKE_DIR, "B2356run72.dat"),
        "compound": "R20", "round": 9, "run": 72, "rim_width_in": 7.0,
        "role": "drive_brake_pressure_sweeps", "has_sl": True,
        "description": "Round 9 R20 18x6.0-10 (7-in rim) initial 12 psi, 10 psi, 14 psi Drive/Brake sweeps"
    },
    "R20_18x6_7in_8psi_speed": {
        "file": os.path.join(DRIVE_BRAKE_DIR, "B2356run73.dat"),
        "compound": "R20", "round": 9, "run": 73, "rim_width_in": 7.0,
        "role": "drive_brake_8psi_final12_speed", "has_sl": True,
        "description": "Round 9 R20 18x6.0-10 (7-in rim) 8 psi, final 12 psi, 24/72 kph Drive/Brake sweeps"
    },
}


def parse_ttc_dat(file_path, required_channels=None, allow_exact_duplicates=False, return_meta=False):
    """
    Parses a raw Calspan TTC .dat file with strict validation.
    Line 0: TIRF metadata header
    Line 1: Channel names (tab-separated)
    Line 2: Channel units (tab-separated)
    Line 3+: Data rows
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"TTC raw file not found: {file_path}")

    req = REQUIRED_CHANNELS if required_channels is None else list(required_channels)
    with open(file_path, "r", encoding="utf-8", errors="replace") as fp:
        line0 = fp.readline().strip()
        line1 = fp.readline().strip()
        line2 = fp.readline().strip()

    channels = line1.split("\t")
    missing_ch = [ch for ch in req if ch not in channels]
    if missing_ch:
        raise ValueError(f"File {os.path.basename(file_path)} is missing required channels: {missing_ch}")

    df = pd.read_csv(file_path, header=1, skiprows=[2], sep="\t")

    # Data integrity checks
    nan_count = df[req].isna().sum().sum()
    if nan_count > 0:
        raise ValueError(f"Corrupted data in {file_path}: {nan_count} NaN values detected in required channels.")

    if len(df) < 2:
        raise ValueError(f"File {file_path} contains insufficient data points ({len(df)}).")

    n_exact_dups = 0
    if allow_exact_duplicates:
        df["raw_row_idx"] = np.arange(len(df), dtype=int)
        data_cols = [c for c in df.columns if c != "raw_row_idx"]
        dup_mask = df.duplicated(subset=data_cols, keep="first")
        n_exact_dups = int(dup_mask.sum())
        if n_exact_dups > 0:
            df = df.loc[~dup_mask].reset_index(drop=True)

    et = df["ET"].values
    non_mono = np.sum(np.diff(et) <= 0)
    if non_mono > 0:
        raise ValueError(f"Non-monotonic timestamps detected in {file_path}: {non_mono} steps with dt <= 0.")

    if return_meta:
        return line0, line1, line2, df, {"exact_duplicates_removed": n_exact_dups}
    return line0, line2, df


def segment_sweeps(et_arr, min_gap_s=1.0):
    """
    Identifies contiguous test sweeps by detecting pauses/gaps (dt > min_gap_s).
    Returns an array of sweep IDs (1-indexed) for each sample.
    """
    dt = np.diff(et_arr)
    gap_indices = np.where(dt > min_gap_s)[0]
    starts = [0] + (gap_indices + 1).tolist()
    ends = gap_indices.tolist() + [len(et_arr) - 1]

    sweep_ids = np.zeros(len(et_arr), dtype=int)
    for i, (s, e) in enumerate(zip(starts, ends)):
        sweep_ids[s:e+1] = i + 1

    return sweep_ids, len(starts)


def detect_turnaround_transients(et_arr, sa_arr, sweep_ids, num_sweeps, max_dsa_dt=10.0, min_sa_turnaround=11.0):
    """
    Detects actuator turnaround transients within each individual sweep.
    At peak steering travel (|SA| > 11.0 deg), the hydraulic actuator rapidly decelerates and reverses,
    producing an impulse spike in dSA/dt (> 10-25 deg/s) and transient contact patch shear reversal.
    These points are flagged for exclusion while strictly preserving valid steady-state sweep data.
    """
    is_transient = np.zeros(len(et_arr), dtype=bool)

    for sw_id in range(1, num_sweeps + 1):
        mask = (sweep_ids == sw_id)
        indices = np.where(mask)[0]
        if len(indices) < 5:
            continue

        t_sub = et_arr[indices]
        sa_sub = sa_arr[indices]
        dt_sub = np.diff(t_sub)
        dt_sub = np.where(dt_sub <= 0, 1e-4, dt_sub)
        dsa = np.diff(sa_sub) / dt_sub

        high_dsa_idx = np.where((np.abs(dsa) > max_dsa_dt) & (np.abs(sa_sub[:-1]) >= min_sa_turnaround))[0]
        for r_idx in high_dsa_idx:
            is_transient[indices[r_idx]] = True
            if r_idx + 1 < len(indices):
                is_transient[indices[r_idx + 1]] = True

    return is_transient


def process_cornering_dataset(df, file_path, compound, round_num, run_num, min_fz=100.0, max_fz=1800.0, min_v_kph=15.0):
    """
    Processes and conditions a cornering dataset.
    Converts units and signs from Calspan SAE to ISO 8855 / target simulation convention.
    """
    n_raw = len(df)
    et = df["ET"].values
    sa = df["SA"].values
    v = df["V"].values
    fz_raw = -df["FZ"].values

    # 1. Sweep segmentation
    sweep_ids, num_sweeps = segment_sweeps(et, min_gap_s=1.0)

    # 2. Reversal transient detection (actuator spikes > 10 deg/s at |SA| >= 11 deg)
    turnaround_transient = detect_turnaround_transients(et, sa, sweep_ids, num_sweeps, max_dsa_dt=10.0, min_sa_turnaround=11.0)

    # 3. Filtering masks
    mask_v = (v >= min_v_kph)
    mask_fz = (fz_raw >= min_fz) & (fz_raw <= max_fz)
    mask_steady = ~turnaround_transient

    valid_mask = mask_v & mask_fz & mask_steady

    # Detailed filtering statistics
    rej_v = int(np.sum(~mask_v))
    rej_fz = int(np.sum(mask_v & ~mask_fz))
    rej_turnaround = int(np.sum(mask_v & mask_fz & ~mask_steady))
    retained_count = int(np.sum(valid_mask))

    sub = df[valid_mask].copy()
    n_pts = len(sub)

    # Unit conversions & coordinate transforms
    # ISO 8855 (+X fwd, +Y left, +Z up):
    #   f_z = -FZ (N, compressive positive)
    #   f_y = -FY (N, positive slip produces positive lateral force)
    #   f_x = FX  (N)
    #   m_z = -MZ (Nm, restoring moment about +Z is negative)
    #   m_x = MX  (Nm, overturning moment about +X)
    #   alpha = radians(SA) (rad)
    #   gamma = -radians(IA) (rad, wheel top tilt towards left is positive)
    #   v_x = V / 3.6 (m/s)
    #   p_kpa = P (kPa)
    #   r_l = RL * 0.01 (m)

    res_df = pd.DataFrame({
        "source_file": [os.path.basename(file_path)] * n_pts,
        "compound": [compound] * n_pts,
        "round": [round_num] * n_pts,
        "run": [run_num] * n_pts,
        "raw_row_idx": sub.index.values,
        "sweep_id": sweep_ids[valid_mask],
        "t": sub["ET"].values,
        "v_x": sub["V"].values / 3.6,
        "sa_deg": sub["SA"].values,
        "alpha": np.radians(sub["SA"].values),
        "ia_deg": sub["IA"].values,
        "gamma": -np.radians(sub["IA"].values),
        "p_kpa": sub["P"].values,
        "f_z": -sub["FZ"].values,
        "f_y": -sub["FY"].values,
        "f_x": sub["FX"].values,
        "m_z": -sub["MZ"].values,
        "m_x": sub["MX"].values,
        "r_l": sub["RL"].values * 0.01,
        "t_inside": sub["TSTI"].values,
        "t_center": sub["TSTC"].values,
        "t_outside": sub["TSTO"].values,
        "t_road": sub["RST"].values,
        "t_ambient": sub["AmbTmp"].values
    })

    stats = {
        "raw_samples": n_raw,
        "retained_samples": retained_count,
        "retained_pct": float(retained_count / n_raw * 100.0),
        "rejected_by_velocity": rej_v,
        "rejected_by_normal_load": rej_fz,
        "rejected_by_turnaround_transient": rej_turnaround,
        "num_sweeps": num_sweeps,
        "operating_ranges": {
            "fz_range_n": [float(res_df["f_z"].min()), float(res_df["f_z"].max())],
            "alpha_range_deg": [float(res_df["sa_deg"].min()), float(res_df["sa_deg"].max())],
            "gamma_range_deg": [float(np.degrees(res_df["gamma"].min())), float(np.degrees(res_df["gamma"].max()))],
            "pressure_range_kpa": [float(res_df["p_kpa"].min()), float(res_df["p_kpa"].max())],
            "velocity_range_mps": [float(res_df["v_x"].min()), float(res_df["v_x"].max())],
            "fx_range_n": [float(res_df["f_x"].min()), float(res_df["f_x"].max())],
            "fy_range_n": [float(res_df["f_y"].min()), float(res_df["f_y"].max())],
            "mz_range_nm": [float(res_df["m_z"].min()), float(res_df["m_z"].max())],
            "mx_range_nm": [float(res_df["m_x"].min()), float(res_df["m_x"].max())]
        }
    }

    return res_df, stats


def process_spring_dataset(df, file_path, compound, round_num, run_num, min_fz=50.0):
    """
    Processes a vertical load-deflection characterization dataset.
    Isolates near-zero camber and near-zero slip points for radial tire stiffness identification.
    """
    n_raw = len(df)
    fz = -df["FZ"].values
    rl = df["RL"].values * 0.01 # cm to m
    p = df["P"].values
    sa = df["SA"].values
    ia = df["IA"].values

    mask = (fz >= min_fz) & (np.abs(sa) <= 1.0) & (np.abs(ia) <= 1.0)
    sub = df[mask].copy()
    n_pts = len(sub)

    res_df = pd.DataFrame({
        "source_file": [os.path.basename(file_path)] * n_pts,
        "compound": [compound] * n_pts,
        "round": [round_num] * n_pts,
        "run": [run_num] * n_pts,
        "raw_row_idx": sub.index.values,
        "t": sub["ET"].values,
        "f_z": -sub["FZ"].values,
        "r_l": sub["RL"].values * 0.01,
        "p_kpa": sub["P"].values,
        "v_x": sub["V"].values / 3.6
    })

    stats = {
        "raw_samples": n_raw,
        "retained_samples": n_pts,
        "retained_pct": float(n_pts / n_raw * 100.0),
        "operating_ranges": {
            "fz_range_n": [float(res_df["f_z"].min()), float(res_df["f_z"].max())],
            "rl_range_m": [float(res_df["r_l"].min()), float(res_df["r_l"].max())],
            "pressure_range_kpa": [float(res_df["p_kpa"].min()), float(res_df["p_kpa"].max())]
        }
    }

    return res_df, stats


def _classify_drive_brake_block(run_num, sweep_id):
    """Maps (run, sweep_id) to experimental block and phase role for Round 9 Drive/Brake."""
    if run_num in (68, 71):
        if sweep_id <= 31:
            return "step_load_sweep_1", "warmup_run"
        if sweep_id == 32:
            return "warmup_sa_sweep_1", "warmup_run"
        if sweep_id <= 47:
            return "step_load_sweep_2", "warmup_run"
        if sweep_id == 48:
            return "warmup_sa_sweep_2", "warmup_run"
        return "warmup_sa_hold_checks", "warmup_run"
    if run_num in (69, 72):
        if sweep_id <= 2:
            return "run_warmup_12psi", "run_warmup"
        if sweep_id <= 38:
            return "initial_12psi_40kph", "break_in_initial_12psi"
        if sweep_id <= 74:
            return "10psi_40kph", "steady_state"
        return "14psi_40kph", "steady_state"
    if run_num in (70, 73):
        if sweep_id <= 2:
            return "run_warmup_12psi", "run_warmup"
        if sweep_id <= 38:
            return "8psi_40kph", "steady_state"
        if sweep_id <= 74:
            return "final_12psi_40kph", "steady_state"
        if sweep_id <= 86:
            return "speed_24kph_12psi", "speed_sweep"
        return "speed_72kph_12psi", "speed_sweep"
    raise ValueError(f"Unsupported Drive/Brake run number: {run_num}")


def _convert_ttc_to_iso(sub_df, file_path, compound, round_num, run_num, sweep_ids_arr, extra_cols=None):
    """Converts Calspan SAE channels to Mirena ISO 8855 convention (+X fwd, +Y left, +Z up)."""
    n_pts = len(sub_df)
    raw_idx = sub_df["raw_row_idx"].values if "raw_row_idx" in sub_df.columns else sub_df.index.values
    data = {
        "source_file": [os.path.basename(file_path)] * n_pts,
        "compound": [compound] * n_pts,
        "round": [round_num] * n_pts,
        "run": [run_num] * n_pts,
        "raw_row_idx": raw_idx,
        "sweep_id": sweep_ids_arr,
        "t": sub_df["ET"].values,
        "v_x": sub_df["V"].values / 3.6,
        "sa_deg": sub_df["SA"].values,
        "alpha": np.radians(sub_df["SA"].values),
        "ia_deg": sub_df["IA"].values,
        "gamma": -np.radians(sub_df["IA"].values),
        "p_kpa": sub_df["P"].values,
        "f_z": -sub_df["FZ"].values,
        "f_y": -sub_df["FY"].values,
        "f_x": sub_df["FX"].values,
        "m_z": -sub_df["MZ"].values,
        "m_x": sub_df["MX"].values,
        "r_l": sub_df["RL"].values * 0.01,
        "t_inside": sub_df["TSTI"].values,
        "t_center": sub_df["TSTC"].values,
        "t_outside": sub_df["TSTO"].values,
        "t_road": sub_df["RST"].values,
        "t_ambient": sub_df["AmbTmp"].values,
    }
    if extra_cols:
        data.update(extra_cols)
    return pd.DataFrame(data)


def process_drive_brake_run(df, file_path, info, min_fz=100.0, max_fz=1800.0, min_v_kph=15.0):
    """
    Audits and converts one Round 9 Drive/Brake TTC run (18.0x6.0-10) into ISO 8855
    while preserving sweep structure, half-sweep tags, and physics-based quality flags.
    """
    run_num = info["run"]
    rim = info["rim_width_in"]
    et = df["ET"].values
    sweep_ids, num_sweeps = segment_sweeps(et, min_gap_s=1.0)
    fz_iso = -df["FZ"].values
    v_kph = df["V"].values

    block_ids = np.empty(len(df), dtype=object)
    phase_roles = np.empty(len(df), dtype=object)
    half_sweeps = np.empty(len(df), dtype=object)
    quality_flags = np.full(len(df), "valid", dtype=object)
    global_sweep_ids = np.array([f"R{run_num}_S{s:03d}" for s in sweep_ids], dtype=object)

    sweep_summaries = []
    has_sl = "SL" in df.columns

    for sw_i in range(1, num_sweeps + 1):
        sw_mask = (sweep_ids == sw_i)
        idx = np.where(sw_mask)[0]
        block_id, phase_role = _classify_drive_brake_block(run_num, sw_i)
        block_ids[idx] = block_id
        phase_roles[idx] = phase_role

        if not has_sl:
            half_sweeps[idx] = "warmup_no_sl"
            quality_flags[idx] = "no_sl_channel"
        else:
            # Segment into H1 (ramp down to peak brake) and H2 (ramp up) at argmin(SR)
            sr_sub = df["SR"].values[idx]
            i_turn = int(np.argmin(sr_sub))
            h1_idx = idx[:i_turn + 1]
            h2_idx = idx[i_turn + 1:]
            half_sweeps[h1_idx] = "H1_ramp_down"
            half_sweeps[h2_idx] = "H2_ramp_up"

            for h_name, h_idx in [("H1", h1_idx), ("H2", h2_idx)]:
                if len(h_idx) == 0:
                    continue
                if len(h_idx) < 50:
                    quality_flags[h_idx] = "incomplete_half_sweep"
                    continue
                fz_h = fz_iso[h_idx]
                fx_h = df["FX"].values[h_idx]
                sl_h = df["SL"].values[h_idx]
                fz_med_h = float(np.median(fz_h))
                m_lin = np.abs(fx_h) < max(0.5 * fz_med_h, 120.0)
                if np.sum(m_lin) < 8 or np.std(fx_h[m_lin] / np.maximum(fz_h[m_lin], 50.0)) <= 0.05:
                    quality_flags[h_idx] = "sl_ref_anomaly"
                else:
                    sl0 = float(np.polyfit(fx_h[m_lin] / fz_h[m_lin], sl_h[m_lin], 1)[1])
                    if abs(sl0) > 0.008:
                        quality_flags[h_idx] = "sl_ref_anomaly"

        # Sample-level physics gates (only override "valid")
        m_low_v = (v_kph[idx] < min_v_kph) & (quality_flags[idx] == "valid")
        quality_flags[idx[m_low_v]] = "low_velocity"
        m_low_fz = ((fz_iso[idx] < min_fz) | (fz_iso[idx] > max_fz)) & (quality_flags[idx] == "valid")
        quality_flags[idx[m_low_fz]] = "low_fz_airborne"

        sub = df.iloc[idx]
        valid_n = int(np.sum(quality_flags[idx] == "valid"))
        sweep_summaries.append({
            "global_sweep_id": f"R{run_num}_S{sw_i:03d}",
            "run": run_num,
            "sweep_id": sw_i,
            "rim_width_in": rim,
            "block_id": block_id,
            "phase_role": phase_role,
            "n_samples": int(len(idx)),
            "n_valid_samples": valid_n,
            "t_start_s": float(sub["ET"].iloc[0]),
            "t_end_s": float(sub["ET"].iloc[-1]),
            "duration_s": float(sub["ET"].iloc[-1] - sub["ET"].iloc[0]),
            "v_kph_med": float(sub["V"].median()),
            "p_kpa_med": float(sub["P"].median()),
            "sa_deg_med": float(sub["SA"].median()),
            "ia_deg_med": float(sub["IA"].median()),
            "fz_n_min": float(fz_iso[idx].min()),
            "fz_n_med": float(np.median(fz_iso[idx])),
            "fz_n_max": float(fz_iso[idx].max()),
            "sl_min": float(sub["SL"].min()) if has_sl else np.nan,
            "sl_max": float(sub["SL"].max()) if has_sl else np.nan,
            "sr_min": float(sub["SR"].min()),
            "sr_max": float(sub["SR"].max()),
            "fx_n_min": float(sub["FX"].min()),
            "fx_n_max": float(sub["FX"].max()),
            "tstc_start_c": float(sub["TSTC"].iloc[0]),
            "tstc_end_c": float(sub["TSTC"].iloc[-1]),
            "quality_issues": ",".join(sorted(set(quality_flags[idx]) - {"valid"})) or "none",
        })

    if not has_sl:
        return None, pd.DataFrame(sweep_summaries)

    is_ss_valid = (
        (quality_flags == "valid")
        & np.isin(phase_roles, ["steady_state", "speed_sweep"])
    )
    extra = {
        "tire_spec": ["Hoosier_18x6.0-10"] * len(df),
        "rim_width_in": np.full(len(df), rim, dtype=float),
        "global_sweep_id": global_sweep_ids,
        "block_id": block_ids,
        "half_sweep": half_sweeps,
        "phase_role": phase_roles,
        "quality_flag": quality_flags,
        "is_steady_state_valid": is_ss_valid,
        "omega": df["N"].values * (2.0 * np.pi / 60.0),
        "r_e": df["RE"].values * 0.01,
        "kappa": df["SL"].values,
        "sr_sae": df["SR"].values,
    }
    res_df = _convert_ttc_to_iso(df, file_path, info["compound"], info["round"], run_num, sweep_ids, extra_cols=extra)
    return res_df, pd.DataFrame(sweep_summaries)


def run_drive_brake_pipeline():
    """Processes all 6 TTC Round 9 Hoosier 18.0x6.0-10 files and exports datasets + manifest."""
    print("=" * 78)
    print("   TTC ROUND 9 DRIVE/BRAKE PREPROCESSING (Hoosier 18.0x6.0-10 R20)")
    print("=" * 78)
    os.makedirs(DRIVE_BRAKE_PROCESSED_DIR, exist_ok=True)
    os.makedirs(DRIVE_BRAKE_METADATA_DIR, exist_ok=True)

    all_dfs = []
    all_sweeps = []
    manifest = {
        "tire_specification": "Hoosier 18.0x6.0-10 R20 (6-in & 7-in rims; NOT 16x7.5-10)",
        "round": 9,
        "nominal_conditions": {
            "fnomin_n": 800.0,
            "pinom_kpa": 83.0,
            "unloaded_radius_m": 0.2286,
            "reference_speed_mps": 11.11,
        },
        "coordinate_conventions": {
            "frame": "ISO 8855 / Mirena simulation (+X fwd, +Y left, +Z up)",
            "kappa": "SL (free-rolling effective radius referenced dimensionless slip ratio, +drive/-brake)",
            "sr_sae": "SR (Calspan SAE loaded-radius slip ratio, retained for audit transparency)",
            "f_x": "FX (N, +drive / -brake)",
            "f_y": "-FY (N, positive alpha produces positive f_y)",
            "f_z": "-FZ (N, compressive positive)",
            "m_z": "-MZ (Nm)",
            "m_x": "MX (Nm)",
            "alpha": "radians(SA) (rad)",
            "gamma": "-radians(IA) (rad)",
        },
        "runs": {},
    }

    for key, info in DRIVE_BRAKE_RUN_MAP.items():
        fpath = info["file"]
        req_ch = REQUIRED_CHANNELS + (DRIVE_BRAKE_EXTRA_CHANNELS if info["has_sl"] else ["N", "RE", "SR"])
        l0, l1, l2, df, meta = parse_ttc_dat(
            fpath, required_channels=req_ch, allow_exact_duplicates=True, return_meta=True
        )
        res_df, sw_df = process_drive_brake_run(df, fpath, info)
        all_sweeps.append(sw_df)
        if res_df is not None:
            all_dfs.append(res_df)

        n_ss = int(res_df["is_steady_state_valid"].sum()) if res_df is not None else 0
        n_brk_in = int((res_df["phase_role"] == "break_in_initial_12psi").sum()) if res_df is not None else 0
        n_anom = int((res_df["quality_flag"] != "valid").sum()) if res_df is not None else len(df)
        print(
            f"Run {info['run']} ({info['rim_width_in']:.0f}in, {info['role']}): "
            f"raw={len(df)+meta['exact_duplicates_removed']}, dups_dropped={meta['exact_duplicates_removed']}, "
            f"sweeps={len(sw_df)}, steady_valid={n_ss}, break_in={n_brk_in}, flagged_invalid={n_anom}"
        )
        manifest["runs"][f"run_{info['run']}"] = {
            "key": key,
            "file": os.path.basename(fpath),
            "header": l0,
            "channels": l1.split("\t"),
            "units": l2.split("\t"),
            "rim_width_in": info["rim_width_in"],
            "role": info["role"],
            "exact_duplicates_removed": meta["exact_duplicates_removed"],
            "rows_after_dedup": int(len(df)),
            "num_sweeps": int(len(sw_df)),
            "steady_state_valid_rows": n_ss,
            "break_in_rows": n_brk_in,
            "flagged_invalid_rows": n_anom,
        }

    df_all = pd.concat(all_dfs, ignore_index=True)
    df_sweeps = pd.concat(all_sweeps, ignore_index=True)
    df_ss = df_all[df_all["is_steady_state_valid"]].copy()
    df_pure = df_ss[df_ss["sa_deg"].abs() < 0.2].copy()
    df_comb = df_ss[df_ss["sa_deg"].abs() >= 2.5].copy()

    # Deterministic sweep-level train / validation split (NO row-wise splitting):
    # - 24 kph and 72 kph speed blocks -> 100% held out in validation (out-of-domain speed test)
    # - 40 kph steady-state blocks -> checkerboard holdout across (rim, pressure, camber, load):
    #   approx 25-33% of sweeps in validation, zero shared sweeps between train and validation.
    pure_val_sweeps = set()
    comb_val_sweeps = set()
    for (rim, blk, sa_r, ia_r), grp in df_ss.groupby(
        ["rim_width_in", "block_id", df_ss["sa_deg"].round(0), df_ss["ia_deg"].round(0)]
    ):
        sw_order = grp.groupby("global_sweep_id")["f_z"].median().sort_values().index.tolist()
        if blk in ("speed_24kph_12psi", "speed_72kph_12psi"):
            chosen = sw_order
        else:
            # Deterministic checkerboard pick (1 or 2 of the 4 load sweeps per condition)
            p_idx = {"8psi_40kph": 0, "10psi_40kph": 1, "final_12psi_40kph": 2, "14psi_40kph": 3}[blk]
            r_idx = 0 if rim == 6.0 else 1
            ia_idx = int(abs(ia_r) // 2)
            # Pick 1 sweep out of 4 per cell, rotating through all 4 loads across conditions (~25% val at 40kph)
            pick = (p_idx + r_idx + ia_idx) % len(sw_order)
            chosen = [sw_order[pick]]
        if abs(sa_r) < 0.5:
            pure_val_sweeps.update(chosen)
        else:
            comb_val_sweeps.update(chosen)

    df_all["split"] = "excluded"
    m_pure = df_all["is_steady_state_valid"] & (df_all["sa_deg"].abs() < 0.2)
    m_comb = df_all["is_steady_state_valid"] & (df_all["sa_deg"].abs() >= 2.5)
    df_all.loc[m_pure & ~df_all["global_sweep_id"].isin(pure_val_sweeps), "split"] = "train"
    df_all.loc[m_pure & df_all["global_sweep_id"].isin(pure_val_sweeps), "split"] = "val"
    df_all.loc[m_comb & ~df_all["global_sweep_id"].isin(comb_val_sweeps), "split"] = "train"
    df_all.loc[m_comb & df_all["global_sweep_id"].isin(comb_val_sweeps), "split"] = "val"

    # Also tag split on df_sweeps
    val_all = pure_val_sweeps | comb_val_sweeps
    def _sw_split(row):
        if row["phase_role"] not in ("steady_state", "speed_sweep"):
            return "excluded"
        return "val" if row["global_sweep_id"] in val_all else "train"
    df_sweeps["split"] = df_sweeps.apply(_sw_split, axis=1)

    df_pure = df_all[m_pure].copy()
    df_comb = df_all[m_comb].copy()

    outputs = {
        "R20_18x6_drive_brake_all.csv": df_all,
        "R20_18x6_pure_longitudinal.csv": df_pure,
        "R20_18x6_combined_slip.csv": df_comb,
        "R20_18x6_sweeps_summary.csv": df_sweeps,
    }
    for fname, out_df in outputs.items():
        p1 = os.path.join(DRIVE_BRAKE_PROCESSED_DIR, fname)
        out_df.to_csv(p1, index=False)

    manifest["splits"] = {
        "pure_train_rows": int((df_pure["split"] == "train").sum()),
        "pure_val_rows": int((df_pure["split"] == "val").sum()),
        "pure_train_sweeps": int(df_pure.loc[df_pure["split"] == "train", "global_sweep_id"].nunique()),
        "pure_val_sweeps": int(df_pure.loc[df_pure["split"] == "val", "global_sweep_id"].nunique()),
        "comb_train_rows": int((df_comb["split"] == "train").sum()),
        "comb_val_rows": int((df_comb["split"] == "val").sum()),
        "comb_train_sweeps": int(df_comb.loc[df_comb["split"] == "train", "global_sweep_id"].nunique()),
        "comb_val_sweeps": int(df_comb.loc[df_comb["split"] == "val", "global_sweep_id"].nunique()),
        "break_in_initial_12psi_rows": int((df_all["phase_role"] == "break_in_initial_12psi").sum()),
        "run_warmup_rows": int((df_all["phase_role"] == "run_warmup").sum()),
        "flagged_invalid_rows": int((df_all["quality_flag"] != "valid").sum()),
    }
    m_path = os.path.join(DRIVE_BRAKE_METADATA_DIR, "dataset_manifest.json")
    with open(m_path, "w") as fp:
        json.dump(manifest, fp, indent=2)

    print("\n>> Summary:", json.dumps(manifest["splits"], indent=2))
    print(f">> Exported Drive/Brake datasets to {DRIVE_BRAKE_PROCESSED_DIR}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="TTC Data Preprocessing Pipeline")
    parser.add_argument("--dataset", choices=["cornering", "drive_brake", "all"], default="cornering")
    args = parser.parse_args()

    if args.dataset in ("drive_brake", "all"):
        run_drive_brake_pipeline()
    if args.dataset == "drive_brake":
        return

    print("=" * 78)
    print("   TTC DATA PREPROCESSING & CONDITIONING PIPELINE FOR MIRENA DYNAMICS")
    print("=" * 78)

    manifest = {
        "tire_specification": "Hoosier 16.0x7.5-10, 8.0-inch rim",
        "nominal_conditions": {
            "fnomin_n": 800.0,
            "pinom_kpa": 83.0,
            "unloaded_radius_m": 0.2032,
            "reference_speed_mps": 11.11
        },
        "coordinate_conventions": {
            "frame": "ISO 8855 / target vehicle simulation",
            "x_axis": "+X forward",
            "y_axis": "+Y left",
            "z_axis": "+Z upward",
            "normal_force": "f_z = -FZ (compressive normal load positive, N)",
            "lateral_force": "f_y = -FY (positive slip produces positive lateral force, N)",
            "aligning_torque": "m_z = -MZ (restoring moment about upward Z is negative, Nm)",
            "camber_angle": "gamma = -radians(IA) (wheel tilt left is positive, rad)"
        },
        "datasets": {}
    }

    total_cornering_rows = 0
    total_spring_rows = 0

    for key, info in RUN_MAP.items():
        f_path = info["file"]
        print(f"\nProcessing {key} ({os.path.basename(f_path)})...")
        header0, units_line, df = parse_ttc_dat(f_path)
        print(f"  TIRF Header: {header0}")

        if "spring" in key:
            proc_df, stats = process_spring_dataset(df, f_path, info["compound"], info["round"], info["run"])
            out_file = os.path.join(PROCESSED_DIR, f"{key}.csv")
            proc_df.to_csv(out_file, index=False)
            total_spring_rows += stats["retained_samples"]

            print(f"  Category: Vertical Characterization")
            print(f"  Raw rows: {stats['raw_samples']} -> Retained: {stats['retained_samples']} ({stats['retained_pct']:.2f}%)")
            print(f"  Fz range: [{stats['operating_ranges']['fz_range_n'][0]:.1f}, {stats['operating_ranges']['fz_range_n'][1]:.1f}] N")
            print(f"  Loaded radius range: [{stats['operating_ranges']['rl_range_m'][0]*100:.2f}, {stats['operating_ranges']['rl_range_m'][1]*100:.2f}] cm")
            print(f"  Saved to: {out_file}")

            manifest["datasets"][key] = {
                "file": os.path.basename(out_file),
                "compound": info["compound"],
                "round": info["round"],
                "run": info["run"],
                "category": info["category"],
                "is_cornering_calibration": False,
                "statistics": stats,
                "description": info["description"]
            }
        else:
            proc_df, stats = process_cornering_dataset(df, f_path, info["compound"], info["round"], info["run"])
            out_file = os.path.join(PROCESSED_DIR, f"{key}.csv")
            proc_df.to_csv(out_file, index=False)
            total_cornering_rows += stats["retained_samples"]

            print(f"  Category: {info['category']}")
            print(f"  Raw rows: {stats['raw_samples']} -> Retained: {stats['retained_samples']} ({stats['retained_pct']:.2f}%)")
            print(f"  Turnaround transients removed: {stats['rejected_by_turnaround_transient']} across {stats['num_sweeps']} sweeps")
            print(f"  Fz range:     [{stats['operating_ranges']['fz_range_n'][0]:.1f}, {stats['operating_ranges']['fz_range_n'][1]:.1f}] N")
            print(f"  Alpha range:  [{stats['operating_ranges']['alpha_range_deg'][0]:.2f}, {stats['operating_ranges']['alpha_range_deg'][1]:.2f}] deg")
            print(f"  Gamma range:  [{stats['operating_ranges']['gamma_range_deg'][0]:.2f}, {stats['operating_ranges']['gamma_range_deg'][1]:.2f}] deg")
            print(f"  Pressure:     [{stats['operating_ranges']['pressure_range_kpa'][0]:.1f}, {stats['operating_ranges']['pressure_range_kpa'][1]:.1f}] kPa")
            print(f"  Velocity:     [{stats['operating_ranges']['velocity_range_mps'][0]:.1f}, {stats['operating_ranges']['velocity_range_mps'][1]:.1f}] m/s")
            print(f"  Saved to: {out_file}")

            manifest["datasets"][key] = {
                "file": os.path.basename(out_file),
                "compound": info["compound"],
                "round": info["round"],
                "run": info["run"],
                "category": info["category"],
                "is_cornering_calibration": (info["category"] == "cornering_calibration"),
                "statistics": stats,
                "description": info["description"]
            }

    manifest["summary"] = {
        "total_cornering_rows": total_cornering_rows,
        "total_vertical_spring_rows": total_spring_rows,
        "cornering_rows_breakdown": {
            "R25B_train": manifest["datasets"]["R25B_train"]["statistics"]["retained_samples"],
            "R25B_val": manifest["datasets"]["R25B_val"]["statistics"]["retained_samples"],
            "R20_train": manifest["datasets"]["R20_train"]["statistics"]["retained_samples"],
            "R20_val": manifest["datasets"]["R20_val"]["statistics"]["retained_samples"]
        }
    }

    manifest_file = os.path.join(METADATA_DIR, "dataset_manifest.json")
    with open(manifest_file, "w") as fp:
        json.dump(manifest, fp, indent=2)

    print("\n" + "=" * 78)
    print(f"   PREPROCESSING SUMMARY")
    print(f"   Total cornering rows (4 runs): {total_cornering_rows}")
    print(f"     - R25B Train: {manifest['datasets']['R25B_train']['statistics']['retained_samples']} | Val: {manifest['datasets']['R25B_val']['statistics']['retained_samples']}")
    print(f"     - R20  Train: {manifest['datasets']['R20_train']['statistics']['retained_samples']} | Val: {manifest['datasets']['R20_val']['statistics']['retained_samples']}")
    print(f"   Total vertical characterization rows: {total_spring_rows} (EXCLUDED from cornering count)")
    print(f"   Manifest exported to: {manifest_file}")
    print("=" * 78)


if __name__ == "__main__":
    main()
