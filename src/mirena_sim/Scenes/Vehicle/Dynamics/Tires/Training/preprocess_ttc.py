#!/usr/bin/env python3
"""
TTC Data Preprocessing & Conditioning Pipeline for Mirena Dynamics
Reads Calspan Tire Test Consortium (TTC) Round 8 (R25B) and Round 9 (R20) files,
filters raw telemetry, converts units to SI, and partitions train/validation datasets.
"""

import os
import json
import glob
import numpy as np
import pandas as pd

RAW_DIR = "src/mirena_sim/Scenes/Vehicle/Dynamics/Tires/Data/Hoosier_16x7.5-10/Raw"
PROCESSED_DIR = "src/mirena_sim/Scenes/Vehicle/Dynamics/Tires/Data/Hoosier_16x7.5-10/Processed"
METADATA_DIR = "src/mirena_sim/Scenes/Vehicle/Dynamics/Tires/Data/Hoosier_16x7.5-10/Metadata"

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)

RUN_MAP = {
    "R25B_train": {
        "file": os.path.join(RAW_DIR, "Round8_R25B", "B1965run6.dat"),
        "compound": "R25B",
        "round": 8,
        "run": 6,
        "type": "cornering_train",
        "description": "Round 8 R25B Cornering test at 40 kph, multi-load, multi-camber, multi-pressure"
    },
    "R25B_val": {
        "file": os.path.join(RAW_DIR, "Round8_R25B", "B1965run7.dat"),
        "compound": "R25B",
        "round": 8,
        "run": 7,
        "type": "cornering_val",
        "description": "Round 8 R25B Cornering test with speed sweeps (24-72 kph) & pressure sweeps"
    },
    "R20_train": {
        "file": os.path.join(RAW_DIR, "Round9_R20", "B2356run8.dat"),
        "compound": "R20",
        "round": 9,
        "run": 8,
        "type": "cornering_train",
        "description": "Round 9 R20 Cornering test at 40 kph, multi-load, multi-camber, multi-pressure"
    },
    "R20_val": {
        "file": os.path.join(RAW_DIR, "Round9_R20", "B2356run9.dat"),
        "compound": "R20",
        "round": 9,
        "run": 9,
        "type": "cornering_val",
        "description": "Round 9 R20 Cornering test with speed sweeps (24-72 kph) & low pressure"
    },
    "R25B_spring": {
        "file": os.path.join(RAW_DIR, "Round8_R25B", "B1965run5.dat"),
        "compound": "R25B",
        "round": 8,
        "run": 5,
        "type": "spring_rate",
        "description": "Round 8 R25B Vertical deflection and spring rate characterization"
    },
    "R20_spring": {
        "file": os.path.join(RAW_DIR, "Round9_R20", "B2356run7.dat"),
        "compound": "R20",
        "round": 9,
        "run": 7,
        "type": "spring_rate",
        "description": "Round 9 R20 Vertical deflection and spring rate characterization"
    }
}

def parse_ttc_dat(file_path):
    with open(file_path, "r") as fp:
        line1 = fp.readline().strip()
    
    # Read table: line 1 is channel names, line 2 is units
    df = pd.read_csv(file_path, header=1, skiprows=[2], sep="\t")
    return line1, df

def process_cornering_data(df, file_path, compound, round_num, run_num, min_fz=150.0, min_v=15.0):
    # Calspan sign: FZ is negative in Calspan, so physical Fz = -FZ
    # In Calspan SAE convention:
    #   SA > 0 => FY < 0. For standard Pacejka formulation:
    #   Slip angle alpha in rad, camber gamma in rad.
    #   Pacejka convention: lateral force has same sign as slip angle in linear range (Fy ≈ C_alpha * alpha).
    #   Therefore, Fy_pacejka = -FY_calspan.
    #   Similarly, self-aligning torque Mz opposes slip angle: Mz_calspan > 0 for SA > 0.
    #   In Pacejka convention: Mz_pacejka = MZ_calspan (or -t * Fy).
    
    f_z_raw = -df["FZ"]
    mask = (f_z_raw >= min_fz) & (df["V"] >= min_v)
    sub = df[mask].copy()
    
    n_pts = len(sub)
    res = pd.DataFrame({
        "source_file": [os.path.basename(file_path)] * n_pts,
        "compound": [compound] * n_pts,
        "round": [round_num] * n_pts,
        "run": [run_num] * n_pts,
        "raw_row_idx": sub.index.values,
        "t": sub["ET"].values,
        "v_x": sub["V"].values / 3.6,
        "sa_deg": sub["SA"].values,
        "alpha": np.radians(sub["SA"].values),
        "ia_deg": sub["IA"].values,
        "gamma": np.radians(sub["IA"].values),
        "p_kpa": sub["P"].values,
        "f_z": f_z_raw[mask].values,
        "f_y": -sub["FY"].values,
        "f_x": sub["FX"].values,
        "m_z": sub["MZ"].values,
        "m_x": sub["MX"].values,
        "r_l": sub["RL"].values * 0.01
    })
    
    return res

def main():
    print("=" * 70)
    print("   PREPROCESSING TTC DATA FOR HOOSIER 16x7.5-10")
    print("=" * 70)
    
    manifest = {
        "tire_model": "Hoosier 16.0x7.5-10",
        "datasets": {}
    }
    
    for key, info in RUN_MAP.items():
        f_path = info["file"]
        if not os.path.exists(f_path):
            print(f"Warning: File not found: {f_path}")
            continue
        
        header, df = parse_ttc_dat(f_path)
        print(f"\nProcessing {key} ({os.path.basename(f_path)})...")
        print(f"  Raw points: {len(df)}")
        
        if "spring" in key:
            # Process spring rate data
            fz = -df["FZ"].values
            rl = df["RL"].values * 0.01
            p = df["P"].values
            mask = fz > 50.0
            spring_df = pd.DataFrame({
                "f_z": fz[mask],
                "r_l": rl[mask],
                "p_kpa": p[mask]
            })
            out_file = os.path.join(PROCESSED_DIR, f"{key}.csv")
            spring_df.to_csv(out_file, index=False)
            print(f"  Saved spring dataset: {out_file} ({len(spring_df)} rows)")
            
            manifest["datasets"][key] = {
                "file": os.path.basename(out_file),
                "compound": info["compound"],
                "type": info["type"],
                "rows": len(spring_df),
                "description": info["description"]
            }
        else:
            proc_df = process_cornering_data(df, f_path, info["compound"], info["round"], info["run"])
            out_file = os.path.join(PROCESSED_DIR, f"{key}.csv")
            proc_df.to_csv(out_file, index=False)
            
            fz_range = [float(proc_df["f_z"].min()), float(proc_df["f_z"].max())]
            alpha_range = [float(proc_df["sa_deg"].min()), float(proc_df["sa_deg"].max())]
            gamma_range = [float(proc_df["ia_deg"].min()), float(proc_df["ia_deg"].max())]
            p_range = [float(proc_df["p_kpa"].min()), float(proc_df["p_kpa"].max())]
            v_range = [float(proc_df["v_x"].min()), float(proc_df["v_x"].max())]
            
            print(f"  Cleaned points: {len(proc_df)}")
            print(f"  Fz range: [{fz_range[0]:.1f}, {fz_range[1]:.1f}] N")
            print(f"  Alpha range: [{alpha_range[0]:.2f}, {alpha_range[1]:.2f}] deg")
            print(f"  Gamma range: [{gamma_range[0]:.2f}, {gamma_range[1]:.2f}] deg")
            print(f"  Pressure range: [{p_range[0]:.1f}, {p_range[1]:.1f}] kPa")
            print(f"  Velocity range: [{v_range[0]:.1f}, {v_range[1]:.1f}] m/s")
            print(f"  Saved to: {out_file}")
            
            manifest["datasets"][key] = {
                "file": os.path.basename(out_file),
                "compound": info["compound"],
                "type": info["type"],
                "rows": len(proc_df),
                "fz_range_n": fz_range,
                "alpha_range_deg": alpha_range,
                "gamma_range_deg": gamma_range,
                "pressure_range_kpa": p_range,
                "velocity_range_mps": v_range,
                "description": info["description"]
            }
    
    manifest_file = os.path.join(METADATA_DIR, "dataset_manifest.json")
    with open(manifest_file, "w") as fp:
        json.dump(manifest, fp, indent=2)
    print(f"\nManifest saved to: {manifest_file}")
    print("=" * 70)

if __name__ == "__main__":
    main()
