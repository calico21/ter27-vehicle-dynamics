#!/usr/bin/env python3
"""
Extended 6-Dimension A/B/C Correlation Suite for Tecnun eRacing TeR26 RWD.

Evaluates strictly on the 30% out-of-sample validation split (`split == 'val_30'`)
of `ter26_clean_ground_truth.parquet`:
  (A) Baseline            : Current repo TeR26 model (Godot headless `full_14dof`)
  (B) +3D Kinematics      : High-fidelity CasADi TeR26 + exact 3D double-wishbone LUTs (nominal params)
  (C) +3D Kin + SysID     : Variant (B) with 11 parameters identified strictly on `calib_70`

Outputs in `vd_correlation_lab/results/ter26_extended_correlation/`:
  - `ter26_abc_correlation_report.json`
  - `fig01_timeseries_oos.png` (300 DPI, zero emojis)
  - `fig02_bode_coherence.png` (300 DPI, zero emojis)
  - `fig03_understeer_gg.png` (300 DPI, zero emojis)
  - `fig04_regime_heatmap_residuals.png` (300 DPI, zero emojis)
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import coherence, csd, welch

CORR_DIR = Path(__file__).resolve().parent
LAB_DIR = CORR_DIR.parents[1]
REPO_ROOT = LAB_DIR.parent
ORACLE_DIR = LAB_DIR / "oracle"
OUT_DIR = LAB_DIR / "results" / "ter26_extended_correlation"
VAULT_DIR = REPO_ROOT / "outputs" / "validation_vault" / "01_ter26_sim_to_real"
SYSID_JSON = OUT_DIR / "ter26_sysid_params.json"
REPORT_JSON = OUT_DIR / "ter26_abc_correlation_report.json"
AUDIT_JSON = OUT_DIR / "telemetry_audit_report.json"

for p in (str(REPO_ROOT), str(LAB_DIR), str(ORACLE_DIR), str(OUT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from vd_correlation_lab.config.paths import resolve_ter26_ground_truth  # noqa: E402
from vd_correlation_lab.analysis.visualization.plot_style import (  # noqa: E402
    PALETTE,
    add_footer,
    add_kpi_box,
    add_suptitle,
    apply_motec_style,
    save_fig,
    style_ax,
)
PARQUET_PATH = resolve_ter26_ground_truth("parquet")

from vd_correlation_lab.analysis.benchmarks.godot_runner import run_godot_batch  # noqa: E402
from vd_correlation_lab.analysis.benchmarks.metrics_engine import compute_kus, compute_signal_metrics  # noqa: E402
from vd_correlation_lab.models.vehicle_14dof.ter26_high_fidelity_model import (  # noqa: E402
    TeR26HighFidelityCasADi,
    default_sysid_theta,
    theta_dict_to_vec,
)
from vd_correlation_lab.models.vehicle_14dof.ter27_casadi_model import slip_power_smooth, tire_temp_dot_smooth  # noqa: E402

G_ACC = 9.80665
WHEELBASE_M = 1.550


def _sanitize_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return None if (math.isnan(val) or math.isinf(val)) else val
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _sanitize_json(obj.tolist())
    return obj


def _extended_signal_metrics(real: np.ndarray, sim: np.ndarray, dt: float = 0.01) -> Dict[str, float]:
    m = compute_signal_metrics(real, sim, dt=dt)
    mask = np.isfinite(real) & np.isfinite(sim)
    if int(np.sum(mask)) >= 20:
        r_a = real[mask]
        s_a = sim[mask]
        pk_r = float(np.percentile(np.abs(r_a), 99.5))
        pk_s = float(np.percentile(np.abs(s_a), 99.5))
        m["peak_real"] = pk_r
        m["peak_sim"] = pk_s
        m["peak_err_pct"] = 100.0 * (pk_s - pk_r) / max(pk_r, 1e-6)
    else:
        m["peak_real"] = float("nan")
        m["peak_sim"] = float("nan")
        m["peak_err_pct"] = float("nan")
    return m


def _run_variant_a_godot(val_runs: List[Tuple[str, pd.DataFrame]], horizon_s: float) -> Dict[str, pd.DataFrame]:
    """Run current repo TeR26 Godot full_14dof model on the exact 100 Hz val_30 slices."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="ter26_varA_"))
    jobs: List[Dict[str, Any]] = []
    out_paths: Dict[str, Path] = {}

    for run_id, df_r in val_runs:
        in_csv = tmp_dir / f"{run_id}_in.csv"
        out_csv = tmp_dir / f"{run_id}_out_h{int(horizon_s)}.csv"
        # Columns expected by vd_correlation_lab/analysis/benchmarks/vd_lab_replay.gd
        df_in = pd.DataFrame({
            "t_s": df_r["t_s"].to_numpy(dtype=np.float64),
            "vx_ms": df_r["vx_ekf_ms"].to_numpy(dtype=np.float64),
            "ax_ms2": df_r["ax_cg_ms2"].to_numpy(dtype=np.float64),
            "ay_ms2": df_r["ay_cg_ms2"].to_numpy(dtype=np.float64),
            "wz_rads": df_r["wz_rads"].to_numpy(dtype=np.float64),
            "steer_deg": df_r["delta_wheel_deg"].to_numpy(dtype=np.float64),
            "throttle": df_r["throttle"].to_numpy(dtype=np.float64),
            "brake": df_r["brake"].to_numpy(dtype=np.float64),
            "trq_limit_nm": df_r["trq_limit_nm"].to_numpy(dtype=np.float64),
            "wheel_rpm_fl": np.nan_to_num(df_r["wheel_rpm_fl"].to_numpy(dtype=np.float64), nan=0.0),
            "wheel_rpm_fr": np.nan_to_num(df_r["wheel_rpm_fr"].to_numpy(dtype=np.float64), nan=0.0),
            "wheel_rpm_rl": np.nan_to_num(df_r["wheel_rpm_rl"].to_numpy(dtype=np.float64), nan=0.0),
            "wheel_rpm_rr": np.nan_to_num(df_r["wheel_rpm_rr"].to_numpy(dtype=np.float64), nan=0.0),
        })
        df_in.to_csv(in_csv, index=False)
        jobs.append({
            "kind": "replay",
            "vehicle_id": "ter26_rwd",
            "setup": "default",
            "fidelity": "full_14dof",
            "input_csv": str(in_csv),
            "out_csv": str(out_csv),
            "horizon_s": float(horizon_s),
            "ecu_mode": 0,
            "tv_map_id": 2,
        })
        out_paths[run_id] = out_csv

    run_godot_batch(REPO_ROOT, jobs, timeout_s=240, max_workers=3)
    res: Dict[str, pd.DataFrame] = {}
    for run_id, p in out_paths.items():
        res[run_id] = pd.read_csv(p)
    return res


def _run_casadi_variant(
    model: TeR26HighFidelityCasADi,
    val_runs: List[Tuple[str, pd.DataFrame]],
    theta: Dict[str, float],
    horizon_observer_tau_s: float,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Run CasADi TeR26HighFidelityCasADi across all val_30 track runs."""
    th_vec = theta_dict_to_vec(theta)
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for run_id, df_r in val_runs:
        t_s = df_r["t_s"].to_numpy(dtype=np.float64)
        vx = df_r["vx_ekf_ms"].to_numpy(dtype=np.float64)
        wz = df_r["wz_rads"].to_numpy(dtype=np.float64)
        delta_rad = df_r["delta_wheel_rad"].to_numpy(dtype=np.float64)
        thr = df_r["throttle"].to_numpy(dtype=np.float64)
        brk = df_r["brake"].to_numpy(dtype=np.float64)
        trq = df_r["trq_limit_nm"].to_numpy(dtype=np.float64)
        ax_cmd = model.map_pedals_to_ax_cmd_array(vx, thr, brk, trq)

        sim = model.simulate_trajectory(
            t_s=t_s,
            delta_rad=delta_rad,
            ax_cmd_drive=ax_cmd,
            vx_meas=vx,
            wz_meas=wz,
            theta=th_vec,
            horizon_observer_tau_s=horizon_observer_tau_s,
        )
        out[run_id] = sim
    return out


def _concat_variant_a(val_runs: List[Tuple[str, pd.DataFrame]], sim_dict: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
    vx_s, ax_s, ay_s, wz_s = [], [], [], []
    rpm_rl, rpm_rr, kap_rl, kap_rr, fz_min = [], [], [], [], []
    for run_id, df_r in val_runs:
        s = sim_dict[run_id]
        n = min(len(df_r), len(s))
        vx_s.append(s["vx_sim"].to_numpy(dtype=np.float64)[:n])
        ax_s.append(s["ax_sim"].to_numpy(dtype=np.float64)[:n])
        ay_s.append(s["ay_sim"].to_numpy(dtype=np.float64)[:n])
        wz_s.append(s["wz_sim"].to_numpy(dtype=np.float64)[:n])
        rpm_rl.append(s["wheel_rpm_rl_sim"].to_numpy(dtype=np.float64)[:n])
        rpm_rr.append(s["wheel_rpm_rr_sim"].to_numpy(dtype=np.float64)[:n])
        kap_rl.append(s["kappa_rl"].to_numpy(dtype=np.float64)[:n])
        kap_rr.append(s["kappa_rr"].to_numpy(dtype=np.float64)[:n])
        fzs = np.column_stack([
            s["fz_fl"].to_numpy(dtype=np.float64)[:n],
            s["fz_fr"].to_numpy(dtype=np.float64)[:n],
            s["fz_rl"].to_numpy(dtype=np.float64)[:n],
            s["fz_rr"].to_numpy(dtype=np.float64)[:n],
        ])
        fz_min.append(np.min(fzs, axis=1))
    return {
        "vx_sim": np.concatenate(vx_s),
        "ax_sim": np.concatenate(ax_s),
        "ay_sim": np.concatenate(ay_s),
        "wz_sim": np.concatenate(wz_s),
        "wheel_rpm_rl_sim": np.concatenate(rpm_rl),
        "wheel_rpm_rr_sim": np.concatenate(rpm_rr),
        "kappa_rl_sim": np.concatenate(kap_rl),
        "kappa_rr_sim": np.concatenate(kap_rr),
        "fz_min_sim": np.concatenate(fz_min),
    }


def _concat_casadi(val_runs: List[Tuple[str, pd.DataFrame]], sim_dict: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    keys_1d = ["vx_sim", "vy_sim", "ax_sim", "ay_sim", "wz_sim", "phi_sim"]
    out: Dict[str, np.ndarray] = {}
    for k in keys_1d:
        out[k] = np.concatenate([sim_dict[rid][k] for rid, _ in val_runs])
    out["fz_sim"] = np.vstack([sim_dict[rid]["fz_sim"] for rid, _ in val_runs])
    out["zw_sim"] = np.vstack([sim_dict[rid]["zw_sim"] for rid, _ in val_runs])
    out["fx_w_sim"] = np.vstack([sim_dict[rid]["fx_w_sim"] for rid, _ in val_runs])
    out["fy_w_sim"] = np.vstack([sim_dict[rid]["fy_w_sim"] for rid, _ in val_runs])
    out["kappa_sim"] = np.vstack([sim_dict[rid]["kappa_sim"] for rid, _ in val_runs])
    out["alphas_sim"] = np.vstack([sim_dict[rid]["alphas_sim"] for rid, _ in val_runs])
    out["wheel_rpm_sim"] = np.vstack([sim_dict[rid]["wheel_rpm_sim"] for rid, _ in val_runs])
    out["wheel_rpm_rl_sim"] = out["wheel_rpm_sim"][:, 2]
    out["wheel_rpm_rr_sim"] = out["wheel_rpm_sim"][:, 3]
    out["kappa_rl_sim"] = out["kappa_sim"][:, 2]
    out["kappa_rr_sim"] = out["kappa_sim"][:, 3]
    out["fz_min_sim"] = np.min(out["fz_sim"], axis=1)
    return out


def compute_dim1_regimes(
    real: Dict[str, np.ndarray],
    variants: Dict[str, Dict[str, np.ndarray]],
    mask_eval: np.ndarray,
) -> Dict[str, Any]:
    vx = real["vx_ms"]
    ax = real["ax_ms2"]
    ay = real["ay_ms2"]
    wz = real["wz_rads"]
    dd = np.abs(real["delta_dot_rads"])

    regimes = {
        "straight_braking": mask_eval & (vx > 3.0) & (np.abs(ay) < 2.5) & (ax < -1.5),
        "turn_in_trail_braking": mask_eval & (vx > 3.0) & (np.abs(ay) >= 2.5) & (ax < -1.0),
        "steady_state_cornering": mask_eval & (vx > 3.0) & (np.abs(ay) >= 2.5) & (np.abs(ax) <= 1.0) & (dd < 0.35),
        "rwd_exit_acceleration": mask_eval & (vx > 3.0) & (np.abs(ay) >= 2.5) & (ax > 1.0),
        "fast_transients_slalom": mask_eval & (vx > 3.0) & (dd >= 0.45),
        "low_speed_lt_45kmh": mask_eval & (vx >= 3.0) & (vx < 12.5),
        "high_speed_ge_45kmh": mask_eval & (vx >= 12.5),
    }

    out: Dict[str, Any] = {}
    for rname, rmask in regimes.items():
        n_pts = int(np.sum(rmask))
        entry: Dict[str, Any] = {"n": n_pts}
        for vname, vdata in variants.items():
            if n_pts < 20:
                entry[vname] = {"rmse_ax": None, "rmse_ay": None, "rmse_wz": None, "rmse_vx": None}
            else:
                entry[vname] = {
                    "rmse_ax": float(np.sqrt(np.mean((vdata["ax_sim"][rmask] - ax[rmask]) ** 2))),
                    "rmse_ay": float(np.sqrt(np.mean((vdata["ay_sim"][rmask] - ay[rmask]) ** 2))),
                    "rmse_wz": float(np.sqrt(np.mean((vdata["wz_sim"][rmask] - wz[rmask]) ** 2))),
                    "rmse_vx": float(np.sqrt(np.mean((vdata["vx_sim"][rmask] - vx[rmask]) ** 2))),
                }
        out[rname] = entry
    return out


def compute_dim2_frequency_response(
    real: Dict[str, np.ndarray],
    variants: Dict[str, Dict[str, np.ndarray]],
    mask_eval: np.ndarray,
    fs_hz: float = 100.0,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Empirical transfer functions (wz/delta, ay/delta) in 0.1 - 5.0 Hz."""
    u = real["delta_rad"][mask_eval]
    nperseg = min(512, max(128, len(u) // 8))
    f_psd, s_uu = welch(u, fs=fs_hz, nperseg=nperseg)
    band = (f_psd >= 0.15) & (f_psd <= 5.0)
    f_band = f_psd[band]

    def _tf(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        _, s_uy = csd(u, y, fs=fs_hz, nperseg=nperseg)
        _, coh = coherence(u, y, fs=fs_hz, nperseg=nperseg)
        h = s_uy / np.maximum(s_uu, 1e-12)
        gain_db = 20.0 * np.log10(np.maximum(np.abs(h), 1e-8))
        phase_deg = np.degrees(np.angle(h))
        return gain_db[band], phase_deg[band], coh[band]

    g_wz_r, p_wz_r, c_wz_r = _tf(real["wz_rads"][mask_eval])
    g_ay_r, p_ay_r, c_ay_r = _tf(real["ay_ms2"][mask_eval])

    curves: Dict[str, Any] = {
        "freq_hz": f_band,
        "real": {
            "wz_gain_db": g_wz_r, "wz_phase_deg": p_wz_r, "wz_coh": c_wz_r,
            "ay_gain_db": g_ay_r, "ay_phase_deg": p_ay_r, "ay_coh": c_ay_r,
        },
    }
    summary: Dict[str, Any] = {
        "freq_band_hz": [float(f_band[0]), float(f_band[-1])],
        "real_mean_coherence_wz": float(np.mean(c_wz_r)),
        "real_mean_coherence_ay": float(np.mean(c_ay_r)),
    }

    low_f = f_band <= 1.5
    high_f = f_band > 1.5
    for vname, vdata in variants.items():
        g_wz, p_wz, c_wz = _tf(vdata["wz_sim"][mask_eval])
        g_ay, p_ay, c_ay = _tf(vdata["ay_sim"][mask_eval])
        curves[vname] = {
            "wz_gain_db": g_wz, "wz_phase_deg": p_wz, "wz_coh": c_wz,
            "ay_gain_db": g_ay, "ay_phase_deg": p_ay, "ay_coh": c_ay,
        }
        # Circular phase difference in [-180, 180]
        dp_wz = (p_wz - p_wz_r + 180.0) % 360.0 - 180.0
        dp_ay = (p_ay - p_ay_r + 180.0) % 360.0 - 180.0
        # Weight by real coherence for meaningful transfer function error
        w_wz = np.maximum(c_wz_r, 0.05)
        w_ay = np.maximum(c_ay_r, 0.05)
        summary[vname] = {
            "wz_gain_rmse_db": float(np.sqrt(np.average((g_wz - g_wz_r) ** 2, weights=w_wz))),
            "wz_phase_rmse_deg": float(np.sqrt(np.average(dp_wz ** 2, weights=w_wz))),
            "wz_mean_coh": float(np.mean(c_wz)),
            "ay_gain_rmse_db": float(np.sqrt(np.average((g_ay - g_ay_r) ** 2, weights=w_ay))),
            "ay_phase_rmse_deg": float(np.sqrt(np.average(dp_ay ** 2, weights=w_ay))),
            "ay_mean_coh": float(np.mean(c_ay)),
            "wz_gain_rmse_low_db": float(np.sqrt(np.mean((g_wz[low_f] - g_wz_r[low_f]) ** 2))) if np.any(low_f) else None,
            "wz_gain_rmse_high_db": float(np.sqrt(np.mean((g_wz[high_f] - g_wz_r[high_f]) ** 2))) if np.any(high_f) else None,
        }
    return summary, curves


def _gg_polar_boundary(ay_g: np.ndarray, ax_g: np.ndarray, n_bins: int = 36) -> Tuple[np.ndarray, np.ndarray, float]:
    ang = np.arctan2(ax_g, ay_g)
    rad = np.hypot(ay_g, ax_g)
    edges = np.linspace(-math.pi, math.pi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    r_env = np.zeros(n_bins, dtype=np.float64)
    for i in range(n_bins):
        m = (ang >= edges[i]) & (ang < edges[i + 1])
        if int(np.sum(m)) >= 5:
            r_env[i] = float(np.percentile(rad[m], 98.0))
        else:
            r_env[i] = float(np.max(rad[m])) if np.any(m) else 0.0
    dtheta = 2.0 * math.pi / n_bins
    area = float(0.5 * np.sum(r_env ** 2) * dtheta)
    return centers, r_env, area


def compute_dim3_understeer_gg(
    real: Dict[str, np.ndarray],
    variants: Dict[str, Dict[str, np.ndarray]],
    mask_eval: np.ndarray,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    vx_r = real["vx_ms"][mask_eval]
    ay_r = real["ay_ms2"][mask_eval]
    ax_r = real["ax_ms2"][mask_eval]
    wz_r = real["wz_rads"][mask_eval]
    del_deg = real["delta_deg"][mask_eval]

    kus_real, _ = compute_kus(vx_r, ay_r, wz_r, del_deg, wheelbase_m=WHEELBASE_M)
    th_gg, r_gg_real, area_real = _gg_polar_boundary(ay_r / G_ACC, ax_r / G_ACC)

    summary: Dict[str, Any] = {
        "kus_real_deg_g": float(kus_real),
        "gg_area_real_g2": float(area_real),
    }
    gg_curves: Dict[str, Any] = {"theta": th_gg, "real_r": r_gg_real}

    for vname, vdata in variants.items():
        vx_s = vdata["vx_sim"][mask_eval]
        ay_s = vdata["ay_sim"][mask_eval]
        ax_s = vdata["ax_sim"][mask_eval]
        wz_s = vdata["wz_sim"][mask_eval]
        kus_s, _ = compute_kus(vx_s, ay_s, wz_s, del_deg, wheelbase_m=WHEELBASE_M)
        _, r_gg_s, area_s = _gg_polar_boundary(ay_s / G_ACC, ax_s / G_ACC)
        gg_curves[f"{vname}_r"] = r_gg_s

        # Phase portrait wz vs ay: normalized 2D Euclidean RMSE
        wz_ay_dist = np.sqrt(((wz_s - wz_r) / 0.5) ** 2 + ((ay_s - ay_r) / 5.0) ** 2)
        summary[vname] = {
            "kus_sim_deg_g": float(kus_s),
            "kus_err_deg_g": float(kus_s - kus_real),
            "gg_area_sim_g2": float(area_s),
            "gg_area_err_pct": float(100.0 * (area_s - area_real) / max(area_real, 1e-6)),
            "gg_boundary_rmse_g": float(np.sqrt(np.mean((r_gg_s - r_gg_real) ** 2))),
            "wz_ay_phase_norm_rmse": float(np.sqrt(np.mean(wz_ay_dist ** 2))),
        }
    return summary, gg_curves


def compute_dim4_wheels_susp(
    real: Dict[str, np.ndarray],
    variants: Dict[str, Dict[str, np.ndarray]],
    model_kin: TeR26HighFidelityCasADi,
    mask_eval: np.ndarray,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "can_sensor_audit": {
            "front_wheel_rpm_present": False,
            "rear_wheel_rpm_present": True,
            "damper_potentiometers_present": False,
            "note": "October CAN logs contain 0 frames for Front_RPM, Front_Susp, Rear_Susp; FL/FR wheel speeds and damper travel are evaluated via exact 3D model predictions while RL/RR wheel speeds, rear slip kappa_RL/RR, and inner-wheel lift are directly correlated.",
        }
    }
    rl_r = real["wheel_rpm_rl"][mask_eval]
    rr_r = real["wheel_rpm_rr"][mask_eval]
    krl_r = real["kappa_rl"][mask_eval]
    krr_r = real["kappa_rr"][mask_eval]
    lift_flag = real["flag_inner_wheel_lift"][mask_eval].astype(bool)

    for vname, vdata in variants.items():
        m_rl = compute_signal_metrics(rl_r, vdata["wheel_rpm_rl_sim"][mask_eval])
        m_rr = compute_signal_metrics(rr_r, vdata["wheel_rpm_rr_sim"][mask_eval])
        # Correlate kappa on valid slip range [-0.25, 0.25]
        km = np.isfinite(krl_r) & np.isfinite(krr_r) & (np.abs(krl_r) < 0.25) & (np.abs(krr_r) < 0.25)
        krl_m = compute_signal_metrics(krl_r[km], vdata["kappa_rl_sim"][mask_eval][km]) if int(np.sum(km)) > 30 else {}
        krr_m = compute_signal_metrics(krr_r[km], vdata["kappa_rr_sim"][mask_eval][km]) if int(np.sum(km)) > 30 else {}

        fz_min = vdata["fz_min_sim"][mask_eval]
        pred_unload = fz_min < 200.0
        tp = int(np.sum(pred_unload & lift_flag))
        fp = int(np.sum(pred_unload & (~lift_flag)))
        fn = int(np.sum((~pred_unload) & lift_flag))
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2.0 * prec * rec / max(prec + rec, 1e-9)

        ventry: Dict[str, Any] = {
            "wheel_rpm_rl": {"r": m_rl["r"], "r2": m_rl["r2"], "rmse": m_rl["rmse"], "mae": m_rl["mae"]},
            "wheel_rpm_rr": {"r": m_rr["r"], "r2": m_rr["r2"], "rmse": m_rr["rmse"], "mae": m_rr["mae"]},
            "kappa_rl": {"r": krl_m.get("r"), "rmse": krl_m.get("rmse")},
            "kappa_rr": {"r": krr_m.get("r"), "rmse": krr_m.get("rmse")},
            "inner_wheel_unloading": {
                "flagged_samples": int(np.sum(lift_flag)),
                "predicted_lt_200N_samples": int(np.sum(pred_unload)),
                "mean_fz_min_flagged_N": float(np.mean(fz_min[lift_flag])) if np.any(lift_flag) else None,
                "mean_fz_min_unflagged_N": float(np.mean(fz_min[~lift_flag])) if np.any(~lift_flag) else None,
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
            },
        }
        if "zw_sim" in vdata:
            zw_mm = vdata["zw_sim"][mask_eval] * 1e3
            ventry["predicted_wheel_bump_mm"] = {
                "fl_p05_p50_p95": [float(np.percentile(zw_mm[:, 0], q)) for q in (5, 50, 95)],
                "fr_p05_p50_p95": [float(np.percentile(zw_mm[:, 1], q)) for q in (5, 50, 95)],
                "rl_p05_p50_p95": [float(np.percentile(zw_mm[:, 2], q)) for q in (5, 50, 95)],
                "rr_p05_p50_p95": [float(np.percentile(zw_mm[:, 3], q)) for q in (5, 50, 95)],
                "max_abs_bump_mm": float(np.max(np.abs(zw_mm))),
            }
        out[vname] = ventry
    return out


def compute_dim5_thermal(
    real: Dict[str, np.ndarray],
    var_c: Dict[str, np.ndarray],
    mask_eval: np.ndarray,
    dt: float = 0.01,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Compute per-corner cumulative slip energy integral(F . v_s dt) and lumped Hoosier R20
    thermal state T_tire_sim(t), plus correlation of grip residual with thermal state.
    """
    n = len(real["vx_ms"])
    vx = var_c["vx_sim"]
    fx_w = var_c["fx_w_sim"]
    fy_w = var_c["fy_w_sim"]
    kap = var_c["kappa_sim"]
    alp = var_c["alphas_sim"]

    q_slip = np.zeros((n, 4), dtype=np.float64)
    e_cum_kj = np.zeros((n, 4), dtype=np.float64)
    t_sim = np.full((n, 4), 42.0, dtype=np.float64)  # Start warm on track
    t_track = 28.0

    for i in range(4):
        v_wy = -vx * np.tan(np.clip(alp[:, i], -0.45, 0.45))
        q_i = np.abs(fx_w[:, i] * vx * kap[:, i]) + np.abs(fy_w[:, i] * v_wy)
        q_slip[:, i] = q_i
        e_cum_kj[:, i] = np.cumsum(q_i) * dt * 1e-3
        t_cur = 42.0
        for k in range(n):
            t_sim[k, i] = t_cur
            t_dot = float(tire_temp_dot_smooth(t_cur, t_track, float(q_i[k]), float(vx[k])))
            t_cur = float(np.clip(t_cur + dt * t_dot, 25.0, 120.0))

    t_mean = np.mean(t_sim, axis=1)
    # Grip residual on high-lateral samples (|ay| > 4 m/s^2)
    hi_lat = mask_eval & (np.abs(real["ay_ms2"]) > 4.0)
    e_ay_abs = np.abs(real["ay_ms2"][hi_lat]) - np.abs(var_c["ay_sim"][hi_lat])
    r_temp = float(np.corrcoef(e_ay_abs, t_mean[hi_lat])[0, 1]) if int(np.sum(hi_lat)) > 30 else 0.0
    r2_temp_pct = 100.0 * (r_temp ** 2)

    # Also check measured motor temperature correlation if available
    mtemp = 0.5 * (real["motor_temp_l_c"] + real["motor_temp_r_c"])
    m_ok = hi_lat & np.isfinite(mtemp)
    if int(np.sum(m_ok)) > 30 and np.std(mtemp[m_ok]) > 1e-3:
        r_mtemp = float(np.corrcoef(np.abs(real["ay_ms2"][m_ok]) - np.abs(var_c["ay_sim"][m_ok]), mtemp[m_ok])[0, 1])
    else:
        r_mtemp = 0.0

    out = {
        "ir_sensor_can_audit": {
            "rl_fr_remap_verified_in_telemetry_pipeline": True,
            "raw_ir_frames_in_october_logs": 0,
            "note": "IR tire temp CAN IDs (0x3F0/3F4/3F8/3FC) have 0 frames in October logs; thermal correlation evaluates physics-integrated cumulative slip energy and lumped Hoosier R20 thermal state T_tire_sim per corner.",
        },
        "cumulative_slip_energy_kj_val30": {
            "FL": float(e_cum_kj[-1, 0]),
            "FR": float(e_cum_kj[-1, 1]),
            "RL": float(e_cum_kj[-1, 2]),
            "RR": float(e_cum_kj[-1, 3]),
            "total": float(np.sum(e_cum_kj[-1, :])),
        },
        "simulated_tire_temp_c_summary": {
            "FL_mean_max": [float(np.mean(t_sim[mask_eval, 0])), float(np.max(t_sim[mask_eval, 0]))],
            "FR_mean_max": [float(np.mean(t_sim[mask_eval, 1])), float(np.max(t_sim[mask_eval, 1]))],
            "RL_mean_max": [float(np.mean(t_sim[mask_eval, 2])), float(np.max(t_sim[mask_eval, 2]))],
            "RR_mean_max": [float(np.mean(t_sim[mask_eval, 3])), float(np.max(t_sim[mask_eval, 3]))],
        },
        "grip_residual_vs_sim_tire_temp_r": r_temp,
        "grip_residual_variance_explained_by_tire_temp_pct": r2_temp_pct,
        "grip_residual_vs_motor_temp_r": r_mtemp,
    }
    return out, t_mean


def _block_bootstrap_rmse_ci(
    real: np.ndarray,
    sim: np.ndarray,
    block_len: int = 200,
    n_boot: int = 200,
    seed: int = 42,
) -> Dict[str, float]:
    """95% block-bootstrap CI for RMSE (2.0 s blocks @ 100 Hz)."""
    rng = np.random.default_rng(seed)
    err2 = (sim - real) ** 2
    n = len(err2)
    n_blocks = max(1, n // block_len)
    blocks = err2[: n_blocks * block_len].reshape(n_blocks, block_len)
    block_means = np.mean(blocks, axis=1)
    idx = rng.integers(0, n_blocks, size=(n_boot, n_blocks))
    boot_rmse = np.sqrt(np.mean(block_means[idx], axis=1))
    return {
        "rmse": float(np.sqrt(np.mean(err2))),
        "ci95_low": float(np.percentile(boot_rmse, 2.5)),
        "ci95_high": float(np.percentile(boot_rmse, 97.5)),
    }


def compute_dim6_stats_and_error_budget(
    real: Dict[str, np.ndarray],
    variants: Dict[str, Dict[str, np.ndarray]],
    t_tire_sim_mean: np.ndarray,
    mask_eval: np.ndarray,
) -> Dict[str, Any]:
    ci_table: Dict[str, Any] = {}
    for vname, vdata in variants.items():
        ci_table[vname] = {
            "vx_ms": _block_bootstrap_rmse_ci(real["vx_ms"][mask_eval], vdata["vx_sim"][mask_eval]),
            "ax_ms2": _block_bootstrap_rmse_ci(real["ax_ms2"][mask_eval], vdata["ax_sim"][mask_eval]),
            "ay_ms2": _block_bootstrap_rmse_ci(real["ay_ms2"][mask_eval], vdata["ay_sim"][mask_eval]),
            "wz_rads": _block_bootstrap_rmse_ci(real["wz_rads"][mask_eval], vdata["wz_sim"][mask_eval]),
        }

    # Residual attribution for Variant C against [vx, |ay|, |delta_dot|, T_tire_sim, |z_w_max|]
    vc = variants["C_3d_kin_sysid"]
    e_ay = real["ay_ms2"][mask_eval] - vc["ay_sim"][mask_eval]
    e_wz = real["wz_rads"][mask_eval] - vc["wz_sim"][mask_eval]

    feat_names = ["vx_ms", "abs_ay_ms2", "abs_delta_dot_rads", "T_tire_sim_c", "max_abs_zw_mm"]
    zw_max_mm = np.max(np.abs(vc["zw_sim"][mask_eval]), axis=1) * 1e3
    X_raw = np.column_stack([
        real["vx_ms"][mask_eval],
        np.abs(real["ay_ms2"][mask_eval]),
        np.abs(real["delta_dot_rads"][mask_eval]),
        t_tire_sim_mean[mask_eval],
        zw_max_mm,
    ])
    X_std = (X_raw - np.mean(X_raw, axis=0)) / np.maximum(np.std(X_raw, axis=0), 1e-9)

    def _attrib(err: np.ndarray) -> Dict[str, Any]:
        # Regress |err| (error magnitude) and signed err
        y_abs = np.abs(err)
        y_abs_c = y_abs - np.mean(y_abs)
        beta, _, _, _ = np.linalg.lstsq(X_std, y_abs_c, rcond=None)
        y_hat = X_std @ beta
        r2_tot = float(1.0 - np.sum((y_abs_c - y_hat) ** 2) / max(np.sum(y_abs_c ** 2), 1e-9))

        # Individual correlations and Pratt relative importance (beta_j * r_j)
        per_feat: Dict[str, Any] = {}
        pratt_sum = 0.0
        pratt_raw = {}
        for j, fname in enumerate(feat_names):
            r_abs = float(np.corrcoef(y_abs, X_raw[:, j])[0, 1])
            r_sgn = float(np.corrcoef(err, X_raw[:, j])[0, 1])
            pratt = max(float(beta[j] * r_abs / max(np.std(y_abs), 1e-9)), 0.0)
            pratt_raw[fname] = pratt
            pratt_sum += pratt
            per_feat[fname] = {
                "pearson_r_abs_err": r_abs,
                "pearson_r_signed_err": r_sgn,
                "std_beta": float(beta[j]),
            }
        for fname in feat_names:
            per_feat[fname]["explained_share_pct"] = float(100.0 * pratt_raw[fname] / max(pratt_sum, 1e-9)) * max(r2_tot, 0.0)
        return {
            "total_r2_abs_err": r2_tot,
            "unexplained_noise_pct": float(100.0 * (1.0 - max(r2_tot, 0.0))),
            "features": per_feat,
        }

    return {
        "bootstrap_95ci_oos_rmse": ci_table,
        "residual_attribution_variant_C": {
            "ay_residual": _attrib(e_ay),
            "wz_residual": _attrib(e_wz),
        },
    }


def _save_to_out_and_vault(fig: plt.Figure, filename: str) -> None:
    import shutil
    vault_path = save_fig(fig, VAULT_DIR / filename, dpi=300, close=True, tight=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(vault_path, OUT_DIR / filename)


def generate_plots(
    real: Dict[str, np.ndarray],
    variants: Dict[str, Dict[str, np.ndarray]],
    mask_eval: np.ndarray,
    dim1: Dict[str, Any],
    dim2_curves: Dict[str, Any],
    dim3_curves: Dict[str, Any],
    dim6: Dict[str, Any],
) -> None:
    apply_motec_style(dpi=300)

    c_real = PALETTE["white"]
    c_a = PALETTE["crimson"]
    c_b = PALETTE["cyan"]
    c_c = PALETTE["emerald"]

    # --- FIG 01: OOS Time-Series Overlay (30 s high-excitation segment) ---
    idx_eval = np.where(mask_eval)[0]
    win_n = min(3000, len(idx_eval))
    best_i0 = 0
    best_sc = -1.0
    for i0 in range(0, max(1, len(idx_eval) - win_n), 500):
        sl = idx_eval[i0 : i0 + win_n]
        sc = float(np.std(real["ay_ms2"][sl])) + 0.5 * float(np.std(real["ax_ms2"][sl]))
        if sc > best_sc:
            best_sc = sc
            best_i0 = i0
    sl = idx_eval[best_i0 : best_i0 + win_n]
    t_plot = np.arange(len(sl)) * 0.01

    fig, axes = plt.subplots(4, 1, figsize=(13.5, 9.8), sharex=True)
    specs = [
        ("vx_ms", "vx_sim", "Speed vx [m/s]"),
        ("ax_ms2", "ax_sim", "Long. Accel ax [m/s^2]"),
        ("ay_ms2", "ay_sim", "Lat. Accel ay [m/s^2]"),
        ("wz_rads", "wz_sim", "Yaw Rate wz [rad/s]"),
    ]
    ci_c = dim6["bootstrap_95ci_oos_rmse"]["C_3d_kin_sysid"]
    ci_a = dim6["bootstrap_95ci_oos_rmse"]["A_baseline"]
    for idx_s, (ax_p, (kr, ks, ylabel)) in enumerate(zip(axes, specs)):
        ax_p.plot(t_plot, real[kr][sl], color=c_real, lw=1.6, label="Real Telemetry (OOS val_30)", zorder=4)
        ax_p.plot(t_plot, variants["A_baseline"][ks][sl], color=c_a, lw=1.1, alpha=0.80, ls="--", label="(A) Baseline Godot 14-DOF")
        ax_p.plot(t_plot, variants["B_3d_kin"][ks][sl], color=c_b, lw=1.1, alpha=0.85, ls="-.", label="(B) +3D Kinematics")
        ax_p.plot(t_plot, variants["C_3d_kin_sysid"][ks][sl], color=c_c, lw=1.3, alpha=0.95, label="(C) +3D Kin + CasADi SysID", zorder=3)
        style_ax(ax_p, ylabel=ylabel, legend=(idx_s == 0), legend_loc="upper right", legend_ncol=4)
        add_kpi_box(
            ax_p,
            [
                f"RMSE (A): {ci_a[kr]['rmse']:.3f}",
                f"RMSE (C): {ci_c[kr]['rmse']:.3f}",
            ],
            loc="lower right",
            fontsize=7.6,
            border_color=c_c,
        )
    axes[-1].set_xlabel("High-Excitation Out-of-Sample Window Time [s]", color=PALETTE["text"])
    add_suptitle(
        fig,
        "TeR26 RWD Sim-to-Real Time-Series Overlay (Strict 30% OOS val_30)",
        "100 Hz Track Telemetry vs (A) Godot 14-DOF Baseline, (B) +3D Kinematics, (C) +3D Kin + CasADi SysID",
    )
    add_footer(fig)
    fig.subplots_adjust(top=0.90, bottom=0.07, hspace=0.14)
    _save_to_out_and_vault(fig, "fig01_timeseries_oos.png")

    # --- FIG 02: Bode & Coherence (0.15 - 5.0 Hz) ---
    f = dim2_curves["freq_hz"]
    fig, axes = plt.subplots(3, 2, figsize=(13.5, 9.0), sharex=True)
    for col, (sig, title) in enumerate([("wz", "Yaw Rate / Steer (wz / delta)"), ("ay", "Lateral Accel / Steer (ay / delta)")]):
        axes[0, col].semilogx(f, dim2_curves["real"][f"{sig}_gain_db"], color=c_real, lw=1.8, label="Real Telemetry")
        axes[0, col].semilogx(f, dim2_curves["A_baseline"][f"{sig}_gain_db"], color=c_a, lw=1.2, ls="--", label="(A) Baseline")
        axes[0, col].semilogx(f, dim2_curves["B_3d_kin"][f"{sig}_gain_db"], color=c_b, lw=1.2, ls="-.", label="(B) +3D Kin")
        axes[0, col].semilogx(f, dim2_curves["C_3d_kin_sysid"][f"{sig}_gain_db"], color=c_c, lw=1.4, label="(C) +3D Kin + SysID")
        style_ax(axes[0, col], title=title, ylabel="Gain [dB]", legend=(col == 0), legend_loc="lower left", grid_minor=True)

        axes[1, col].semilogx(f, dim2_curves["real"][f"{sig}_phase_deg"], color=c_real, lw=1.8)
        axes[1, col].semilogx(f, dim2_curves["A_baseline"][f"{sig}_phase_deg"], color=c_a, lw=1.2, ls="--")
        axes[1, col].semilogx(f, dim2_curves["B_3d_kin"][f"{sig}_phase_deg"], color=c_b, lw=1.2, ls="-.")
        axes[1, col].semilogx(f, dim2_curves["C_3d_kin_sysid"][f"{sig}_phase_deg"], color=c_c, lw=1.4)
        style_ax(axes[1, col], ylabel="Phase [deg]", grid_minor=True)

        axes[2, col].semilogx(f, dim2_curves["real"][f"{sig}_coh"], color=c_real, lw=1.8)
        axes[2, col].semilogx(f, dim2_curves["A_baseline"][f"{sig}_coh"], color=c_a, lw=1.2, ls="--")
        axes[2, col].semilogx(f, dim2_curves["B_3d_kin"][f"{sig}_coh"], color=c_b, lw=1.2, ls="-.")
        axes[2, col].semilogx(f, dim2_curves["C_3d_kin_sysid"][f"{sig}_coh"], color=c_c, lw=1.4)
        axes[2, col].set_ylim(0.0, 1.05)
        style_ax(axes[2, col], xlabel="Frequency [Hz]", ylabel="Coherence gamma^2", grid_minor=True)
    add_suptitle(
        fig,
        "TeR26 Empirical Frequency Response & Coherence (OOS val_30, 0.15 - 5.0 Hz)",
        "Cross-Spectral Transfer Functions (wz/delta and ay/delta) and Input-Output Coherence",
    )
    add_footer(fig)
    fig.subplots_adjust(top=0.89, bottom=0.07, hspace=0.16, wspace=0.18)
    _save_to_out_and_vault(fig, "fig02_bode_coherence.png")

    # --- FIG 03: Understeer Curve, wz-ay Phase Portrait & G-G Envelope ---
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.2))
    vx_e = real["vx_ms"][mask_eval]
    ay_e = real["ay_ms2"][mask_eval]
    del_e = real["delta_deg"][mask_eval]
    m_us = (vx_e > 4.0) & (np.abs(ay_e) > 1.5)
    ay_bins = np.linspace(-1.3, 1.3, 27)
    ay_mid = 0.5 * (ay_bins[:-1] + ay_bins[1:])

    def _bin_us(ay_arr: np.ndarray, wz_arr: np.ndarray, vx_arr: np.ndarray) -> np.ndarray:
        us = del_e[m_us] - np.degrees(WHEELBASE_M * wz_arr[m_us] / np.maximum(vx_arr[m_us], 2.0))
        ay_g = ay_arr[m_us] / G_ACC
        out_b = np.full(len(ay_mid), np.nan)
        for bi in range(len(ay_mid)):
            mb = (ay_g >= ay_bins[bi]) & (ay_g < ay_bins[bi + 1])
            if int(np.sum(mb)) >= 15:
                out_b[bi] = float(np.median(us[mb]))
        return out_b

    axes[0].plot(ay_mid, _bin_us(ay_e, real["wz_rads"][mask_eval], vx_e), color=c_real, lw=2.0, marker="o", ms=3.2, label="Real")
    axes[0].plot(ay_mid, _bin_us(variants["A_baseline"]["ay_sim"][mask_eval], variants["A_baseline"]["wz_sim"][mask_eval], variants["A_baseline"]["vx_sim"][mask_eval]), color=c_a, lw=1.4, ls="--", label="(A) Baseline")
    axes[0].plot(ay_mid, _bin_us(variants["B_3d_kin"]["ay_sim"][mask_eval], variants["B_3d_kin"]["wz_sim"][mask_eval], variants["B_3d_kin"]["vx_sim"][mask_eval]), color=c_b, lw=1.4, ls="-.", label="(B) +3D Kin")
    axes[0].plot(ay_mid, _bin_us(variants["C_3d_kin_sysid"]["ay_sim"][mask_eval], variants["C_3d_kin_sysid"]["wz_sim"][mask_eval], variants["C_3d_kin_sysid"]["vx_sim"][mask_eval]), color=c_c, lw=1.8, label="(C) +3D Kin + SysID")
    style_ax(axes[0], title="Understeer Characteristic", xlabel="Lateral Acceleration ay [g]", ylabel="delta_wheel - L*wz/vx [deg]", legend=True, legend_loc="upper left")

    step_sc = max(1, int(np.sum(mask_eval)) // 2500)
    axes[1].scatter(ay_e[::step_sc] / G_ACC, real["wz_rads"][mask_eval][::step_sc], s=6, alpha=0.35, color=c_real, label="Real")
    axes[1].scatter(variants["C_3d_kin_sysid"]["ay_sim"][mask_eval][::step_sc] / G_ACC, variants["C_3d_kin_sysid"]["wz_sim"][mask_eval][::step_sc], s=6, alpha=0.35, color=c_c, label="(C) +3D Kin + SysID")
    style_ax(axes[1], title="Yaw Rate vs Lateral Accel Phase Portrait", xlabel="ay [g]", ylabel="wz [rad/s]", legend=True, legend_loc="upper left")

    th_c = np.append(dim3_curves["theta"], dim3_curves["theta"][0])
    for key, col, ls, lbl in [
        ("real_r", c_real, "-", "Real (98th pct)"),
        ("A_baseline_r", c_a, "--", "(A) Baseline"),
        ("B_3d_kin_r", c_b, "-.", "(B) +3D Kin"),
        ("C_3d_kin_sysid_r", c_c, "-", "(C) +3D Kin + SysID"),
    ]:
        r_c = np.append(dim3_curves[key], dim3_curves[key][0])
        axes[2].plot(r_c * np.cos(th_c), r_c * np.sin(th_c), color=col, ls=ls, lw=1.8 if key in ("real_r", "C_3d_kin_sysid_r") else 1.3, label=lbl)
    axes[2].axhline(0, color=PALETTE["spine"], lw=0.8)
    axes[2].axvline(0, color=PALETTE["spine"], lw=0.8)
    axes[2].set_aspect("equal", "box")
    style_ax(axes[2], title="4-Quadrant G-G Envelope Boundary", xlabel="Lateral ay [g]", ylabel="Longitudinal ax [g]", legend=True, legend_loc="lower right")

    add_suptitle(
        fig,
        "TeR26 Nonlinear Handling Correlation: Understeer, Phase Portrait & G-G Envelope",
        "Strict Out-of-Sample val_30 Track Validation (ISO 8855)",
    )
    add_footer(fig)
    fig.subplots_adjust(top=0.84, bottom=0.12, wspace=0.24)
    _save_to_out_and_vault(fig, "fig03_understeer_gg.png")

    # --- FIG 04: Regime Heatmap + Bootstrap 95% CI + Residual Attribution ---
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.4))
    reg_keys = list(dim1.keys())
    reg_labels = [k.replace("_", " ") for k in reg_keys]
    mat = np.zeros((len(reg_keys), 3), dtype=np.float64)
    for i, rk in enumerate(reg_keys):
        for j, vk in enumerate(["A_baseline", "B_3d_kin", "C_3d_kin_sysid"]):
            v = dim1[rk][vk]["rmse_ay"]
            mat[i, j] = np.nan if v is None else float(v)
    im = axes[0].imshow(mat, cmap="inferno_r", aspect="auto")
    axes[0].set_xticks([0, 1, 2])
    axes[0].set_xticklabels(["(A) Base", "(B) +3DKin", "(C) +SysID"], color=PALETTE["text"])
    axes[0].set_yticks(range(len(reg_keys)))
    axes[0].set_yticklabels(reg_labels, fontsize=8, color=PALETTE["text"])
    med_val = float(np.nanmedian(mat)) if np.any(np.isfinite(mat)) else 2.5
    for i in range(len(reg_keys)):
        for j in range(3):
            if np.isfinite(mat[i, j]):
                txt_col = "#0D1117" if mat[i, j] < med_val else PALETTE["white"]
                axes[0].text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8.2, fontweight="bold", color=txt_col)
    axes[0].set_title("OOS ay RMSE [m/s^2] by Regime", color=PALETTE["text"], fontsize=10.5, fontweight="bold", loc="left", pad=7.0)
    axes[0].grid(False)
    cb = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    cb.ax.yaxis.set_tick_params(color=PALETTE["muted"], labelcolor=PALETTE["muted"])

    sigs = [("vx_ms", "vx [m/s]"), ("ax_ms2", "ax [m/s^2]"), ("ay_ms2", "ay [m/s^2]"), ("wz_rads", "wz*10 [rad/s]")]
    x_pos = np.arange(len(sigs))
    w_bar = 0.25
    ci = dim6["bootstrap_95ci_oos_rmse"]
    for idx_v, (vk, col, lbl) in enumerate([("A_baseline", c_a, "(A) Baseline"), ("B_3d_kin", c_b, "(B) +3D Kin"), ("C_3d_kin_sysid", c_c, "(C) +SysID")]):
        vals, lo_e, hi_e = [], [], []
        for sk, _ in sigs:
            mult = 10.0 if sk == "wz_rads" else 1.0
            d = ci[vk][sk]
            vals.append(d["rmse"] * mult)
            lo_e.append(max(0.0, (d["rmse"] - d["ci95_low"]) * mult))
            hi_e.append(max(0.0, (d["ci95_high"] - d["rmse"]) * mult))
        axes[1].bar(
            x_pos + (idx_v - 1) * w_bar,
            vals,
            width=w_bar,
            color=col,
            alpha=0.88,
            label=lbl,
            yerr=[lo_e, hi_e],
            capsize=3,
            error_kw={"ecolor": PALETTE["white"], "lw": 1.0},
        )
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([lbl for _, lbl in sigs], fontsize=8.2)
    style_ax(axes[1], title="Out-of-Sample RMSE with 95% CIs", ylabel="OOS RMSE (Block-Bootstrap 95% CI)", legend=True, legend_loc="upper left")

    attr_ay = dim6["residual_attribution_variant_C"]["ay_residual"]["features"]
    attr_wz = dim6["residual_attribution_variant_C"]["wz_residual"]["features"]
    fnames = list(attr_ay.keys())
    y_pos = np.arange(len(fnames))
    ay_sh = [attr_ay[fn]["explained_share_pct"] for fn in fnames]
    wz_sh = [attr_wz[fn]["explained_share_pct"] for fn in fnames]
    axes[2].barh(y_pos - 0.18, ay_sh, height=0.35, color=PALETTE["cyan"], label="|e_ay| share [%]")
    axes[2].barh(y_pos + 0.18, wz_sh, height=0.35, color=PALETTE["amber"], label="|e_wz| share [%]")
    axes[2].set_yticks(y_pos)
    axes[2].set_yticklabels(fnames, fontsize=8.2)
    style_ax(axes[2], title="Variant (C) Residual Error Budget", xlabel="Variance Share Explained [%]", legend=True, legend_loc="lower right")

    add_suptitle(
        fig,
        "TeR26 Statistical Diagnostics: Dynamic Regimes, Bootstrap 95% CIs & Error Budget",
        "A/B/C Ablation Across 7 Operating Regimes and Pratt Variance Attribution",
    )
    add_footer(fig)
    fig.subplots_adjust(top=0.84, bottom=0.12, wspace=0.28)
    _save_to_out_and_vault(fig, "fig04_regime_heatmap_residuals.png")


def generate_can_telemetry_audit_plot(audit: Dict[str, Any], df_all: pd.DataFrame) -> None:
    """Generate fig05_can_telemetry_audit.png in unified MoTeC dark-engineering style."""
    apply_motec_style(dpi=300)
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.2))

    # Panel A: Per-run CAN sample coverage & valid_for_sysid breakdown
    per_run = audit["sanity_gates"]["per_run"]
    run_ids = list(per_run.keys())
    short_labels = [rid.replace("2026100", "Oct0").split(".")[0] for rid in run_ids]
    tot_k = np.array([per_run[r]["n_samples"] / 1000.0 for r in run_ids])
    val_k = np.array([per_run[r]["n_valid_for_sysid"] / 1000.0 for r in run_ids])
    oth_k = np.maximum(tot_k - val_k, 0.0)
    x = np.arange(len(run_ids))
    axes[0, 0].bar(x, val_k, color=PALETTE["emerald"], width=0.58, label="Valid for SysID")
    axes[0, 0].bar(x, oth_k, bottom=val_k, color=PALETTE["slate"], alpha=0.55, width=0.58, label="Idle / Paddock / Wheel Events")
    for i in range(len(run_ids)):
        pct = per_run[run_ids[i]]["pct_valid_of_run"]
        axes[0, 0].text(i, tot_k[i] + 2.5, f"{pct:.1f}%", ha="center", va="bottom", fontsize=7.8, color=PALETTE["text"], fontfamily="DejaVu Sans Mono")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(short_labels, rotation=15, ha="right", fontsize=7.8)
    axes[0, 0].set_ylim(0, float(np.max(tot_k)) * 1.18)
    style_ax(axes[0, 0], title="CAN Log Coverage & Valid-for-SysID Breakdown (200 Hz)", ylabel="Samples [x1000]", legend=True, legend_loc="upper right")
    dq = audit["data_quality_summary"]
    add_kpi_box(
        axes[0, 0],
        [
            f"Total 200Hz : {dq['total_samples_200hz']:,}",
            f"Valid Track : {dq['pct_valid_of_track_active']:.2f}%",
            f"Corrupted   : {dq['pct_raw_corrupted_all']:.3f}%",
            f"calib/val   : 70% / 30%",
        ],
        loc="upper left",
        fontsize=7.6,
        border_color=PALETTE["emerald"],
    )

    # Panel B: EKF + RTS Kinematic Residual Distributions vs 0.35 m/s^2 Gate
    m_val = df_all["valid_for_sysid"].to_numpy(dtype=bool)
    res_ax = df_all.loc[m_val, "res_ax_ms2"].to_numpy(dtype=np.float64)
    res_ay = df_all.loc[m_val, "res_ay_ms2"].to_numpy(dtype=np.float64)
    res_ax = res_ax[np.isfinite(res_ax)]
    res_ay = res_ay[np.isfinite(res_ay)]
    bins = np.linspace(-1.4, 1.4, 71)
    axes[0, 1].hist(res_ax, bins=bins, density=True, alpha=0.58, color=PALETTE["cyan"], label="res_ax (EKF+RTS)")
    axes[0, 1].hist(res_ay, bins=bins, density=True, alpha=0.52, color=PALETTE["amber"], label="res_ay (EKF+RTS)")
    axes[0, 1].axvline(-0.35, color=PALETTE["crimson"], ls="--", lw=1.3, label="RMS Gate (+/-0.35 m/s^2)")
    axes[0, 1].axvline(0.35, color=PALETTE["crimson"], ls="--", lw=1.3)
    style_ax(axes[0, 1], title="EKF+RTS Kinematic Consistency Residuals (Valid SysID)", xlabel="Residual [m/s^2]", ylabel="Probability Density", legend=True, legend_loc="upper right")
    g_ax = audit["sanity_gates"]["gates"]["kinematic_residual_ax_rms"]
    g_ay = audit["sanity_gates"]["gates"]["kinematic_residual_ay_rms"]
    add_kpi_box(
        axes[0, 1],
        [
            f"RMS res_ax : {g_ax['rms_ms2']:.4f} m/s^2 (<0.35)",
            f"RMS res_ay : {g_ay['rms_ms2']:.4f} m/s^2 (<0.35)",
            f"p95 |res_ax|: {g_ax['p95_abs_ms2']:.3f} m/s^2",
            f"p95 |res_ay|: {g_ay['p95_abs_ms2']:.3f} m/s^2",
        ],
        loc="upper left",
        fontsize=7.6,
        border_color=PALETTE["cyan"],
    )

    # Panel C: Per-Run Kinematic Residual RMS vs 0.35 m/s^2 Threshold
    track_runs = [r for r in run_ids if per_run[r]["log_type"] == "track"]
    tr_labels = [r.replace("2026100", "Oct0").split(".")[0] for r in track_runs]
    rms_ax_r = [per_run[r]["rms_res_ax_ms2"] for r in track_runs]
    rms_ay_r = [per_run[r]["rms_res_ay_ms2"] for r in track_runs]
    xt = np.arange(len(track_runs))
    axes[1, 0].bar(xt - 0.18, rms_ax_r, width=0.34, color=PALETTE["cyan"], label="RMS res_ax [m/s^2]")
    axes[1, 0].bar(xt + 0.18, rms_ay_r, width=0.34, color=PALETTE["amber"], label="RMS res_ay [m/s^2]")
    axes[1, 0].axhline(0.35, color=PALETTE["crimson"], ls="--", lw=1.5, label="Pooled Gate Limit (0.35 m/s^2)")
    for i in range(len(track_runs)):
        axes[1, 0].text(i - 0.18, rms_ax_r[i] + 0.008, f"{rms_ax_r[i]:.3f}", ha="center", va="bottom", fontsize=7.6, color=PALETTE["cyan"], fontfamily="DejaVu Sans Mono")
        axes[1, 0].text(i + 0.18, rms_ay_r[i] + 0.008, f"{rms_ay_r[i]:.3f}", ha="center", va="bottom", fontsize=7.6, color=PALETTE["amber"], fontfamily="DejaVu Sans Mono")
    axes[1, 0].set_xticks(xt)
    axes[1, 0].set_xticklabels(tr_labels, fontsize=8.2)
    axes[1, 0].set_ylim(0, 0.44)
    style_ax(axes[1, 0], title="Per-Track-Run EKF+RTS Residual RMS", ylabel="Residual RMS [m/s^2]", legend=True, legend_loc="upper right")

    # Panel D: CAN Latency Alignment & ISO 8855 Sanity Gates Scorecard
    gcal = audit["global_calibration"]
    gates = audit["sanity_gates"]["gates"]
    lat_labels = ["CAN Accel\nShift", "CAN Steer\nShift", "Steer-to-wz\nTotal Lag"]
    lat_vals = [gcal["dt_can_accel_ms"], gcal["dt_can_steer_ms"], gcal["steer_vs_wz_total_lag_ms"]]
    lat_cols = [PALETTE["cyan"], PALETTE["violet"], PALETTE["amber"]]
    yl = np.arange(len(lat_labels))
    axes[1, 1].barh(yl, lat_vals, height=0.48, color=lat_cols, alpha=0.90)
    axes[1, 1].axvline(0.0, color=PALETTE["white"], lw=0.9)
    for i, val in enumerate(lat_vals):
        off = 5.0 if val >= 0 else -5.0
        ha = "left" if val >= 0 else "right"
        axes[1, 1].text(val + off, i, f"{val:+.1f} ms", va="center", ha=ha, fontsize=8.2, color=PALETTE["text"], fontfamily="DejaVu Sans Mono")
    axes[1, 1].set_yticks(yl)
    axes[1, 1].set_yticklabels(lat_labels, fontsize=8.4)
    axes[1, 1].set_xlim(-125.0, 185.0)
    style_ax(axes[1, 1], title="CAN Latency Alignment & Certified Calibration Gates", xlabel="Latency Offset [ms]")
    add_kpi_box(
        axes[1, 1],
        [
            f"All 6 Gates : {'PASS' if audit['sanity_gates']['all_gates_passed'] else 'FAIL'}",
            f"corr(wz,ay) : {gates['iso8855_wz_ay_positive']['corr_wz_ay']:.4f} (ISO 8855)",
            f"corr(d,wz)  : {gates['iso8855_delta_wz_positive']['corr_delta_wz']:.4f}",
            f"IMU Yaw Off : {gcal['imu_yaw_offset_deg']:.2f} deg",
            f"i_steer cal : {gcal['i_steer_calibrated']:.4f} (R2={gcal['i_steer_kinematic_r2']:.4f})",
            f"IR RL<->FR  : Verified (r=1.000)",
        ],
        loc="lower right",
        fontsize=7.5,
        border_color=PALETTE["emerald"],
    )

    add_suptitle(
        fig,
        "TeR26 CAN Telemetry Audit, Latency Alignment & EKF+RTS Sanity Gates",
        "5 October CAN Logs (356,891 samples @ 200 Hz) | ISO 8855 Certified Ground Truth",
    )
    add_footer(fig)
    fig.subplots_adjust(top=0.89, bottom=0.07, hspace=0.28, wspace=0.22)
    _save_to_out_and_vault(fig, "fig05_can_telemetry_audit.png")


def main() -> None:
    t0 = time.perf_counter()
    print("[ExtendedCorr] Loading certified parquet & SysID params...", flush=True)
    df = pd.read_parquet(PARQUET_PATH)
    sysid_data = json.loads(SYSID_JSON.read_text(encoding="utf-8"))
    theta_sysid = sysid_data["identified_params"]

    # Extract strictly 30% OOS validation split (val_30) on track runs, downsampled cleanly 200 -> 100 Hz (stride=2)
    val_df = df[(df["log_type"] == "track") & (df["split"] == "val_30")].copy()
    val_runs: List[Tuple[str, pd.DataFrame]] = []
    for run_id, grp in val_df.groupby("run_id", sort=True):
        grp_100 = grp.iloc[::2].reset_index(drop=True)
        val_runs.append((str(run_id), grp_100))
    total_oos = sum(len(g) for _, g in val_runs)
    print(f"[ExtendedCorr] OOS val_30 split @ 100 Hz: {len(val_runs)} track runs, {total_oos} total samples.", flush=True)

    # Concatenate real ground truth arrays
    real = {
        "vx_ms": np.concatenate([g["vx_ekf_ms"].to_numpy(dtype=np.float64) for _, g in val_runs]),
        "ax_ms2": np.concatenate([g["ax_cg_ms2"].to_numpy(dtype=np.float64) for _, g in val_runs]),
        "ay_ms2": np.concatenate([g["ay_cg_ms2"].to_numpy(dtype=np.float64) for _, g in val_runs]),
        "wz_rads": np.concatenate([g["wz_rads"].to_numpy(dtype=np.float64) for _, g in val_runs]),
        "delta_deg": np.concatenate([g["delta_wheel_deg"].to_numpy(dtype=np.float64) for _, g in val_runs]),
        "delta_rad": np.concatenate([g["delta_wheel_rad"].to_numpy(dtype=np.float64) for _, g in val_runs]),
        "delta_dot_rads": np.concatenate([g["delta_dot_rads"].to_numpy(dtype=np.float64) for _, g in val_runs]),
        "wheel_rpm_rl": np.concatenate([g["wheel_rpm_rl"].to_numpy(dtype=np.float64) for _, g in val_runs]),
        "wheel_rpm_rr": np.concatenate([g["wheel_rpm_rr"].to_numpy(dtype=np.float64) for _, g in val_runs]),
        "kappa_rl": np.concatenate([g["kappa_rl"].to_numpy(dtype=np.float64) for _, g in val_runs]),
        "kappa_rr": np.concatenate([g["kappa_rr"].to_numpy(dtype=np.float64) for _, g in val_runs]),
        "motor_temp_l_c": np.concatenate([g["motor_temp_l_c"].to_numpy(dtype=np.float64) for _, g in val_runs]),
        "motor_temp_r_c": np.concatenate([g["motor_temp_r_c"].to_numpy(dtype=np.float64) for _, g in val_runs]),
        "is_active": np.concatenate([g["is_active"].to_numpy(dtype=bool) for _, g in val_runs]),
        "valid_for_sysid": np.concatenate([g["valid_for_sysid"].to_numpy(dtype=bool) for _, g in val_runs]),
        "flag_inner_wheel_lift": np.concatenate([g["flag_inner_wheel_lift"].to_numpy(dtype=bool) for _, g in val_runs]),
    }

    mask_active = real["is_active"]
    mask_valid = real["valid_for_sysid"]
    print(f"[ExtendedCorr] Active OOS samples: {int(np.sum(mask_active))} | valid_for_sysid OOS samples: {int(np.sum(mask_valid))}", flush=True)

    # 1. Run Variant (A) Baseline: Current Repo TeR26 Godot full_14dof (both H10s & Open-Loop)
    print("[ExtendedCorr] Running Variant (A) Baseline (Godot full_14dof H10s + Open-Loop)...", flush=True)
    sim_a_h10_runs = _run_variant_a_godot(val_runs, horizon_s=10.0)
    sim_a_open_runs = _run_variant_a_godot(val_runs, horizon_s=0.0)
    var_a_h10 = _concat_variant_a(val_runs, sim_a_h10_runs)
    var_a_open = _concat_variant_a(val_runs, sim_a_open_runs)

    # 2. Run Variant (B) +3D Kinematics (nominal params) & Variant (C) +3D Kin + SysID
    print("[ExtendedCorr] Running Variant (B) +3D Kinematics & Variant (C) +3D Kin + CasADi SysID...", flush=True)
    model_3d = TeR26HighFidelityCasADi(use_3d_kinematics=True, use_thermal_coupling=True)
    theta_nom = default_sysid_theta(model_3d.params)

    var_b_h10 = _concat_casadi(val_runs, _run_casadi_variant(model_3d, val_runs, theta_nom, horizon_observer_tau_s=2.5))
    var_b_open = _concat_casadi(val_runs, _run_casadi_variant(model_3d, val_runs, theta_nom, horizon_observer_tau_s=0.0))

    var_c_h10 = _concat_casadi(val_runs, _run_casadi_variant(model_3d, val_runs, theta_sysid, horizon_observer_tau_s=2.5))
    var_c_open = _concat_casadi(val_runs, _run_casadi_variant(model_3d, val_runs, theta_sysid, horizon_observer_tau_s=0.0))

    # Primary comparison dictionary (H10s observer mode matching repo benchmark + Open-loop reported alongside)
    variants_h10 = {"A_baseline": var_a_h10, "B_3d_kin": var_b_h10, "C_3d_kin_sysid": var_c_h10}
    variants_open = {"A_baseline": var_a_open, "B_3d_kin": var_b_open, "C_3d_kin_sysid": var_c_open}

    # Global core metrics (both H10s & Open-Loop, on active OOS & valid_for_sysid OOS)
    def _eval_global(vdict: Dict[str, Dict[str, np.ndarray]], m: np.ndarray) -> Dict[str, Any]:
        out_g: Dict[str, Any] = {}
        for vname, vd in vdict.items():
            out_g[vname] = {
                "vx_ms": _extended_signal_metrics(real["vx_ms"][m], vd["vx_sim"][m]),
                "ax_ms2": _extended_signal_metrics(real["ax_ms2"][m], vd["ax_sim"][m]),
                "ay_ms2": _extended_signal_metrics(real["ay_ms2"][m], vd["ay_sim"][m]),
                "wz_rads": _extended_signal_metrics(real["wz_rads"][m], vd["wz_sim"][m]),
            }
        return out_g

    global_h10_active = _eval_global(variants_h10, mask_active)
    global_h10_valid = _eval_global(variants_h10, mask_valid)
    global_open_active = _eval_global(variants_open, mask_active)
    global_open_valid = _eval_global(variants_open, mask_valid)

    # Compute all 6 Extended Dimensions on H10s active OOS (matching repo benchmark continuity)
    print("[ExtendedCorr] Computing Extended Dimensions 1 - 6...", flush=True)
    dim1 = compute_dim1_regimes(real, variants_h10, mask_active)
    dim2_summary, dim2_curves = compute_dim2_frequency_response(real, variants_h10, mask_active, fs_hz=100.0)
    dim3_summary, dim3_curves = compute_dim3_understeer_gg(real, variants_h10, mask_active)
    dim4 = compute_dim4_wheels_susp(real, variants_h10, model_3d, mask_active)
    dim5, t_tire_sim_mean = compute_dim5_thermal(real, var_c_h10, mask_active, dt=0.01)
    dim6 = compute_dim6_stats_and_error_budget(real, variants_h10, t_tire_sim_mean, mask_active)

    print("[ExtendedCorr] Generating 5 MoTeC 300 DPI zero-emoji diagnostic plots...", flush=True)
    generate_plots(real, variants_h10, mask_active, dim1, dim2_curves, dim3_curves, dim6)
    audit_data = json.loads(AUDIT_JSON.read_text(encoding="utf-8")) if AUDIT_JSON.exists() else {}
    if audit_data:
        generate_can_telemetry_audit_plot(audit_data, df)

    # Load repo baseline full_14dof metrics.json for historical continuity reference
    repo_base_path = LAB_DIR / "vehicles" / "ter26_rwd" / "models" / "full_14dof" / "results" / "metrics.json"
    repo_base_ref = {}
    if repo_base_path.exists():
        rb = json.loads(repo_base_path.read_text(encoding="utf-8"))
        repo_base_ref = {
            "pooled_horizon_full_log": {k: rb["pooled_horizon"][k] for k in ("vx_ms", "ax_ms2", "ay_ms2", "wz_rads")},
            "pooled_open_full_log": {k: rb["pooled_open"][k] for k in ("vx_ms", "ax_ms2", "ay_ms2", "wz_rads")},
        }

    report = {
        "metadata": {
            "vehicle": "TeR26 RWD",
            "evaluation_split": "val_30 (strict 30% out-of-sample temporal split, 0 samples used in SysID)",
            "sample_rate_hz": 100.0,
            "n_runs": len(val_runs),
            "run_ids": [rid for rid, _ in val_runs],
            "total_oos_samples": int(total_oos),
            "active_oos_samples": int(np.sum(mask_active)),
            "valid_for_sysid_oos_samples": int(np.sum(mask_valid)),
            "wall_time_s": float(time.perf_counter() - t0),
        },
        "repo_baseline_reference_full_log": repo_base_ref,
        "sysid_identified_params": sysid_data["parameter_summary"],
        "global_metrics_h10_active_oos": global_h10_active,
        "global_metrics_h10_valid_oos": global_h10_valid,
        "global_metrics_open_active_oos": global_open_active,
        "global_metrics_open_valid_oos": global_open_valid,
        "dim1_dynamic_regimes": dim1,
        "dim2_frequency_response": dim2_summary,
        "dim3_understeer_and_gg": dim3_summary,
        "dim4_wheels_slip_suspension": dim4,
        "dim5_thermal_correlation": dim5,
        "dim6_statistical_diagnostics_and_error_budget": dim6,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    VAULT_DIR.mkdir(parents=True, exist_ok=True)
    clean_report = _sanitize_json(report)
    REPORT_JSON.write_text(json.dumps(clean_report, indent=2), encoding="utf-8")

    vault_metrics = {
        "vault": "01_ter26_sim_to_real",
        "metadata": clean_report["metadata"],
        "telemetry_audit": {
            "global_calibration": audit_data.get("global_calibration", {}),
            "data_quality_summary": audit_data.get("data_quality_summary", {}),
            "sanity_gates": audit_data.get("sanity_gates", {}),
            "ir_tire_temp_remap": audit_data.get("ir_tire_temp_remap", {}),
        },
        "global_metrics_h10_active_oos": clean_report["global_metrics_h10_active_oos"],
        "global_metrics_h10_valid_oos": clean_report["global_metrics_h10_valid_oos"],
        "global_metrics_open_active_oos": clean_report["global_metrics_open_active_oos"],
        "dim1_dynamic_regimes": clean_report["dim1_dynamic_regimes"],
        "dim2_frequency_response": clean_report["dim2_frequency_response"],
        "dim3_understeer_and_gg": clean_report["dim3_understeer_and_gg"],
        "dim4_wheels_slip_suspension": clean_report["dim4_wheels_slip_suspension"],
        "dim5_thermal_correlation": clean_report["dim5_thermal_correlation"],
        "dim6_statistical_diagnostics_and_error_budget": clean_report["dim6_statistical_diagnostics_and_error_budget"],
        "sysid_identified_params": clean_report["sysid_identified_params"],
    }
    (VAULT_DIR / "metrics.json").write_text(json.dumps(_sanitize_json(vault_metrics), indent=2), encoding="utf-8")
    print(f"[ExtendedCorr] Saved full report to {REPORT_JSON} and vault metrics to {VAULT_DIR / 'metrics.json'} ({time.perf_counter() - t0:.1f} s)", flush=True)

    # Print concise A/B/C summary table
    print("\n=== OOS val_30 (Active, H10s) A/B/C SUMMARY ===", flush=True)
    for sig in ("vx_ms", "ax_ms2", "ay_ms2", "wz_rads"):
        ma = global_h10_active["A_baseline"][sig]
        mb = global_h10_active["B_3d_kin"][sig]
        mc = global_h10_active["C_3d_kin_sysid"][sig]
        print(
            f"  {sig:8s} | A: RMSE={ma['rmse']:.3f} R2={ma['r2']:.3f} r={ma['r']:.3f} "
            f"| B: RMSE={mb['rmse']:.3f} R2={mb['r2']:.3f} r={mb['r']:.3f} "
            f"| C: RMSE={mc['rmse']:.3f} R2={mc['r2']:.3f} r={mc['r']:.3f}",
            flush=True,
        )

    print("\n=== OOS val_30 (Active, Open-Loop) A/B/C SUMMARY ===", flush=True)
    for sig in ("vx_ms", "ax_ms2", "ay_ms2", "wz_rads"):
        ma = global_open_active["A_baseline"][sig]
        mb = global_open_active["B_3d_kin"][sig]
        mc = global_open_active["C_3d_kin_sysid"][sig]
        print(
            f"  {sig:8s} | A: RMSE={ma['rmse']:.3f} R2={ma['r2']:.3f} r={ma['r']:.3f} "
            f"| B: RMSE={mb['rmse']:.3f} R2={mb['r2']:.3f} r={mb['r']:.3f} "
            f"| C: RMSE={mc['rmse']:.3f} R2={mc['r2']:.3f} r={mc['r']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
