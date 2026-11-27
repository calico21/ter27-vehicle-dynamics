#!/usr/bin/env python3
"""
Physics sanity gates and data quality certification for TeR26 telemetry.

Evaluates:
1. ISO 8855 sign consistency: sign(wz * ay) > 0 and sign(delta_wheel * wz) > 0.
2. Kinematic-inertial consistency residuals on valid segments:
   - res_ax = dvx/dt - (ax_cg + wz * vy)
   - res_ay = ay_cg - (wz * vx + dvy/dt)
   Requires RMS < 0.35 m/s^2 on valid_for_sysid segments.
3. Static zero-velocity bias verification.
4. Wheel-event and corruption statistics (% raw corrupted vs % valid after cleaning).
5. IR tire temperature RL <-> FR remap verification.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


RESIDUAL_RMS_LIMIT_MS2 = 0.35


def compute_kinematic_residuals(
    df: pd.DataFrame,
    dt: float = 0.005,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute longitudinal and lateral rigid-body kinematic residuals per run.

    res_ax = dvx/dt - (ax_cg + wz * vy)
    res_ay = ay_cg - (wz * vx + dvy/dt)
    """
    n = len(df)
    res_ax = np.zeros(n, dtype=np.float64)
    res_ay = np.zeros(n, dtype=np.float64)

    if "run_id" in df.columns:
        groups = df.groupby("run_id", sort=False).indices
    else:
        groups = {"single": np.arange(n)}

    vx_all = df["vx_ekf_ms"].to_numpy(dtype=np.float64)
    vy_all = df["vy_ekf_ms"].to_numpy(dtype=np.float64)
    ax_all = df["ax_cg_ms2"].to_numpy(dtype=np.float64)
    ay_all = df["ay_cg_ms2"].to_numpy(dtype=np.float64)
    wz_all = df["wz_rads"].to_numpy(dtype=np.float64)

    for idx in groups.values():
        idx = np.asarray(idx, dtype=np.int64)
        if len(idx) < 5:
            continue
        vx = vx_all[idx]
        vy = vy_all[idx]
        ax = ax_all[idx]
        ay = ay_all[idx]
        wz = wz_all[idx]

        dvx_dt = np.gradient(vx, dt)
        dvy_dt = np.gradient(vy, dt)

        res_ax[idx] = dvx_dt - (ax + wz * vy)
        res_ay[idx] = ay - (wz * vx + dvy_dt)

    return res_ax, res_ay


def build_valid_for_sysid_mask(
    df: pd.DataFrame,
    res_ax: np.ndarray,
    res_ay: np.ndarray,
    max_abs_res_ms2: float = 1.25,
    min_vx_ms: float = 3.0,
) -> np.ndarray:
    """Construct strict boolean mask for System Identification windows."""
    n = len(df)
    is_track = (
        (df["log_type"].to_numpy() == "track")
        if "log_type" in df.columns
        else np.ones(n, dtype=bool)
    )
    vx = df["vx_ekf_ms"].to_numpy(dtype=np.float64)
    finite_core = (
        np.isfinite(vx)
        & np.isfinite(df["vy_ekf_ms"].to_numpy(dtype=np.float64))
        & np.isfinite(df["ax_cg_ms2"].to_numpy(dtype=np.float64))
        & np.isfinite(df["ay_cg_ms2"].to_numpy(dtype=np.float64))
        & np.isfinite(df["wz_rads"].to_numpy(dtype=np.float64))
        & np.isfinite(df["delta_wheel_rad"].to_numpy(dtype=np.float64))
        & np.isfinite(res_ax)
        & np.isfinite(res_ay)
    )
    not_dropout = ~df["can_dropout"].to_numpy(dtype=bool) if "can_dropout" in df.columns else np.ones(n, dtype=bool)
    not_imu_spike = ~df["imu_spike"].to_numpy(dtype=bool) if "imu_spike" in df.columns else np.ones(n, dtype=bool)
    not_wheelspin = ~df["flag_wheelspin"].to_numpy(dtype=bool) if "flag_wheelspin" in df.columns else np.ones(n, dtype=bool)
    not_lockup = ~df["flag_brake_lockup"].to_numpy(dtype=bool) if "flag_brake_lockup" in df.columns else np.ones(n, dtype=bool)
    not_lift = ~df["flag_inner_wheel_lift"].to_numpy(dtype=bool) if "flag_inner_wheel_lift" in df.columns else np.ones(n, dtype=bool)

    # Trim 0.25 s at boundaries of each run to avoid edge-derivative artifacts
    not_edge = np.ones(n, dtype=bool)
    if "run_id" in df.columns:
        for idx in df.groupby("run_id", sort=False).indices.values():
            idx = np.asarray(idx, dtype=np.int64)
            pad = min(50, len(idx) // 4)
            if pad > 0:
                not_edge[idx[:pad]] = False
                not_edge[idx[-pad:]] = False

    res_ok = (np.abs(res_ax) < max_abs_res_ms2) & (np.abs(res_ay) < max_abs_res_ms2)
    mask = (
        is_track
        & finite_core
        & (vx >= min_vx_ms)
        & not_dropout
        & not_imu_spike
        & not_wheelspin
        & not_lockup
        & not_lift
        & not_edge
        & res_ok
    )
    return mask


def evaluate_sanity_gates(
    df: pd.DataFrame,
    dt: float = 0.005,
    rms_limit_ms2: float = RESIDUAL_RMS_LIMIT_MS2,
) -> Dict[str, Any]:
    """Run all physics sanity gates on the cleaned telemetry DataFrame."""
    res_ax, res_ay = compute_kinematic_residuals(df, dt=dt)
    if "valid_for_sysid" in df.columns:
        valid = df["valid_for_sysid"].to_numpy(dtype=bool)
    else:
        valid = build_valid_for_sysid_mask(df, res_ax, res_ay)

    n_total = int(len(df))
    is_track = (df["log_type"].to_numpy() == "track") if "log_type" in df.columns else np.ones(n_total, dtype=bool)
    is_active = df["is_active"].to_numpy(dtype=bool) if "is_active" in df.columns else (df["vx_ekf_ms"].to_numpy() >= 3.0)
    track_active = is_track & is_active

    n_track = int(np.sum(is_track))
    n_track_active = int(np.sum(track_active))
    n_valid = int(np.sum(valid))

    # 1. Kinematic residuals on valid_for_sysid
    if n_valid > 10:
        rms_ax = float(np.sqrt(np.mean(res_ax[valid] ** 2)))
        rms_ay = float(np.sqrt(np.mean(res_ay[valid] ** 2)))
        p95_ax = float(np.percentile(np.abs(res_ax[valid]), 95))
        p95_ay = float(np.percentile(np.abs(res_ay[valid]), 95))
        mean_ax = float(np.mean(res_ax[valid]))
        mean_ay = float(np.mean(res_ay[valid]))
    else:
        rms_ax = rms_ay = p95_ax = p95_ay = mean_ax = mean_ay = float("nan")

    # Also compute residuals on all track_active for transparency
    if n_track_active > 10:
        rms_ax_all_active = float(np.sqrt(np.mean(res_ax[track_active] ** 2)))
        rms_ay_all_active = float(np.sqrt(np.mean(res_ay[track_active] ** 2)))
    else:
        rms_ax_all_active = rms_ay_all_active = float("nan")

    # 2. ISO 8855 sign assertions on valid dynamic cornering
    wz = df["wz_rads"].to_numpy(dtype=np.float64)
    ay = df["ay_cg_ms2"].to_numpy(dtype=np.float64)
    delta = df["delta_wheel_rad"].to_numpy(dtype=np.float64)
    corner = valid & (np.abs(wz) > 0.10) & (np.abs(ay) > 1.0)
    if int(np.sum(corner)) > 20:
        corr_wz_ay = float(np.corrcoef(wz[corner], ay[corner])[0, 1])
        corr_delta_wz = float(np.corrcoef(delta[corner], wz[corner])[0, 1])
        sign_wz_ay_pos_frac = float(np.mean((wz[corner] * ay[corner]) > 0.0))
        sign_delta_wz_pos_frac = float(np.mean((delta[corner] * wz[corner]) > 0.0))
    else:
        corr_wz_ay = corr_delta_wz = sign_wz_ay_pos_frac = sign_delta_wz_pos_frac = float("nan")

    gate_iso_wz_ay = bool(np.isfinite(corr_wz_ay) and corr_wz_ay > 0.85 and sign_wz_ay_pos_frac > 0.90)
    gate_iso_delta_wz = bool(np.isfinite(corr_delta_wz) and corr_delta_wz > 0.80 and sign_delta_wz_pos_frac > 0.85)
    gate_res_ax = bool(np.isfinite(rms_ax) and rms_ax < rms_limit_ms2)
    gate_res_ay = bool(np.isfinite(rms_ay) and rms_ay < rms_limit_ms2)
    gate_no_nan_valid = bool(
        n_valid > 0
        and np.all(np.isfinite(df.loc[valid, ["vx_ekf_ms", "vy_ekf_ms", "ax_cg_ms2", "ay_cg_ms2", "wz_rads", "delta_wheel_rad"]].to_numpy()))
    )

    # 3. Static bias residual check
    rest = df["vx_wheel_ms"].to_numpy(dtype=np.float64) < 0.25 if "vx_wheel_ms" in df.columns else (df["vx_ekf_ms"].to_numpy() < 0.25)
    if int(np.sum(rest)) > 50:
        rest_ax_mean = float(np.mean(df["ax_cg_ms2"].to_numpy()[rest]))
        rest_ay_mean = float(np.mean(df["ay_cg_ms2"].to_numpy()[rest]))
        rest_wz_mean = float(np.mean(df["wz_rads"].to_numpy()[rest]))
    else:
        rest_ax_mean = rest_ay_mean = rest_wz_mean = 0.0
    gate_static_bias = bool(abs(rest_ax_mean) < 0.20 and abs(rest_ay_mean) < 0.20 and abs(rest_wz_mean) < 0.01)

    # 4. Per-run breakdown
    per_run: Dict[str, Any] = {}
    if "run_id" in df.columns:
        for rid, idx in df.groupby("run_id", sort=False).indices.items():
            idx = np.asarray(idx, dtype=np.int64)
            v_r = valid[idx]
            nv = int(np.sum(v_r))
            per_run[str(rid)] = {
                "n_samples": int(len(idx)),
                "log_type": str(df["log_type"].iloc[idx[0]]) if "log_type" in df.columns else "track",
                "n_valid_for_sysid": nv,
                "pct_valid_of_run": round(100.0 * nv / max(len(idx), 1), 2),
                "rms_res_ax_ms2": round(float(np.sqrt(np.mean(res_ax[idx][v_r] ** 2))), 4) if nv > 10 else None,
                "rms_res_ay_ms2": round(float(np.sqrt(np.mean(res_ay[idx][v_r] ** 2))), 4) if nv > 10 else None,
            }

    all_passed = bool(
        gate_iso_wz_ay
        and gate_iso_delta_wz
        and gate_res_ax
        and gate_res_ay
        and gate_no_nan_valid
        and gate_static_bias
    )

    return {
        "all_gates_passed": all_passed,
        "rms_limit_ms2": rms_limit_ms2,
        "gates": {
            "iso8855_wz_ay_positive": {
                "passed": gate_iso_wz_ay,
                "corr_wz_ay": round(corr_wz_ay, 5),
                "positive_sign_fraction": round(sign_wz_ay_pos_frac, 5),
            },
            "iso8855_delta_wz_positive": {
                "passed": gate_iso_delta_wz,
                "corr_delta_wz": round(corr_delta_wz, 5),
                "positive_sign_fraction": round(sign_delta_wz_pos_frac, 5),
            },
            "kinematic_residual_ax_rms": {
                "passed": gate_res_ax,
                "rms_ms2": round(rms_ax, 5),
                "p95_abs_ms2": round(p95_ax, 5),
                "mean_ms2": round(mean_ax, 5),
                "all_track_active_rms_ms2": round(rms_ax_all_active, 5),
            },
            "kinematic_residual_ay_rms": {
                "passed": gate_res_ay,
                "rms_ms2": round(rms_ay, 5),
                "p95_abs_ms2": round(p95_ay, 5),
                "mean_ms2": round(mean_ay, 5),
                "all_track_active_rms_ms2": round(rms_ay_all_active, 5),
            },
            "no_nan_on_valid_segments": {
                "passed": gate_no_nan_valid,
            },
            "static_zero_velocity_bias": {
                "passed": gate_static_bias,
                "rest_ax_mean_ms2": round(rest_ax_mean, 5),
                "rest_ay_mean_ms2": round(rest_ay_mean, 5),
                "rest_wz_mean_rads": round(rest_wz_mean, 6),
            },
        },
        "coverage": {
            "n_total_samples": n_total,
            "n_track_samples": n_track,
            "n_track_active_samples": n_track_active,
            "n_valid_for_sysid": n_valid,
            "pct_valid_of_total": round(100.0 * n_valid / max(n_total, 1), 2),
            "pct_valid_of_track": round(100.0 * n_valid / max(n_track, 1), 2),
            "pct_valid_of_track_active": round(100.0 * n_valid / max(n_track_active, 1), 2),
        },
        "per_run": per_run,
    }


def main() -> None:
    import argparse
    import sys

    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from vd_correlation_lab.config.paths import resolve_ter26_ground_truth

    parser = argparse.ArgumentParser(description="Run physics sanity gates on cleaned TeR26 telemetry.")
    parser.add_argument(
        "--parquet",
        type=Path,
        default=resolve_ter26_ground_truth("parquet"),
        help="Path to cleaned telemetry parquet file.",
    )
    args = parser.parse_args()
    df = pd.read_parquet(args.parquet)
    report = evaluate_sanity_gates(df)
    print(json.dumps(report, indent=2), flush=True)
    if not report["all_gates_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
