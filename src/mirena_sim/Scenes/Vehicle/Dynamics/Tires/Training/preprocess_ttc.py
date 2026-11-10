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
import numpy as np
import pandas as pd

# Repository-relative path resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../../../../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

TIRE_DATA_DIR = os.path.join(REPO_ROOT, "src/mirena_sim/Scenes/Vehicle/Dynamics/Tires/Data/Hoosier_16x7.5-10")
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


def parse_ttc_dat(file_path):
    """
    Parses a raw Calspan TTC .dat file with strict validation.
    Line 0: TIRF metadata header
    Line 1: Channel names (tab-separated)
    Line 2: Channel units (tab-separated)
    Line 3+: Data rows
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"TTC raw file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8", errors="replace") as fp:
        line0 = fp.readline().strip()
        line1 = fp.readline().strip()
        line2 = fp.readline().strip()

    channels = line1.split("\t")
    missing_ch = [ch for ch in REQUIRED_CHANNELS if ch not in channels]
    if missing_ch:
        raise ValueError(f"File {os.path.basename(file_path)} is missing required channels: {missing_ch}")

    df = pd.read_csv(file_path, header=1, skiprows=[2], sep="\t")

    # Data integrity checks
    nan_count = df[REQUIRED_CHANNELS].isna().sum().sum()
    if nan_count > 0:
        raise ValueError(f"Corrupted data in {file_path}: {nan_count} NaN values detected in required channels.")

    et = df["ET"].values
    if len(et) < 2:
        raise ValueError(f"File {file_path} contains insufficient data points ({len(et)}).")

    non_mono = np.sum(np.diff(et) <= 0)
    if non_mono > 0:
        raise ValueError(f"Non-monotonic timestamps detected in {file_path}: {non_mono} steps with dt <= 0.")

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


def main():
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
