#!/usr/bin/env python3
"""
CasADi + IPOPT Multiple-Shooting System Identification for TeR26 High-Fidelity Model.

Strictly uses ONLY the 70% calibration split (`split == 'calib_70'` AND
`valid_for_sysid == True`) from `ter26_clean_ground_truth.parquet`.
Zero samples from `val_30` are ever touched during parameter identification.

Identifies 11 physical parameters:
  1. mu_x_scale   : Longitudinal peak grip multiplier
  2. mu_y_scale   : Lateral peak grip multiplier
  3. kya_scale    : Cornering stiffness scale (MF 6.2 K_y_alpha)
  4. camber_scale : Camber thrust & sensitivity scale
  5. h_cg         : CG height [m]
  6. i_zz         : Yaw moment of inertia [kg*m^2]
  7. cl_a         : Aerodynamic downforce CL*A [m^2]
  8. cd_a         : Aerodynamic drag CD*A [m^2]
  9. c_damp_scale : Effective suspension/yaw damping scale
 10. k_steer_comp : Steering compliance stiffness [Nm/rad]
 11. sigma_y      : Tire lateral relaxation length [m]

Output:
  - vd_correlation_lab/results/ter26_extended_correlation/ter26_sysid_params.json
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import casadi as ca
import numpy as np
import pandas as pd

ORACLE_DIR = Path(__file__).resolve().parent
LAB_DIR = ORACLE_DIR.parents[1]
REPO_ROOT = LAB_DIR.parent
OUT_DIR = LAB_DIR / "results" / "ter26_extended_correlation"
OUT_JSON = OUT_DIR / "ter26_sysid_params.json"

for _p in (str(REPO_ROOT), str(ORACLE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from vd_correlation_lab.config.paths import resolve_ter26_ground_truth  # noqa: E402
PARQUET_PATH = resolve_ter26_ground_truth("parquet")

from vd_correlation_lab.models.vehicle_14dof.ter26_high_fidelity_model import (  # noqa: E402
    SYSID_PARAM_NAMES,
    TeR26HighFidelityCasADi,
    default_sysid_theta,
    theta_dict_to_vec,
    theta_vec_to_dict,
)


# Physical parameter bounds (lb, ub) and Tikhonov prior weights (small, physical)
PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    "mu_x_scale": (0.65, 1.55),
    "mu_y_scale": (0.65, 1.55),
    "kya_scale": (0.55, 1.85),
    "camber_scale": (0.40, 1.80),
    "h_cg": (0.210, 0.360),
    "i_zz": (45.0, 125.0),
    "cl_a": (1.00, 6.50),
    "cd_a": (0.60, 2.60),
    "c_damp_scale": (0.35, 2.50),
    "k_steer_comp": (2500.0, 30000.0),
    "sigma_y": (0.06, 0.60),
}

# Prior std devs for weak Tikhonov regularization (keeps unexcited directions well-conditioned)
PARAM_PRIOR_STD: Dict[str, float] = {
    "mu_x_scale": 0.45,
    "mu_y_scale": 0.45,
    "kya_scale": 0.50,
    "camber_scale": 0.40,
    "h_cg": 0.045,
    "i_zz": 22.0,
    "cl_a": 1.80,
    "cd_a": 0.65,
    "c_damp_scale": 0.55,
    "k_steer_comp": 6000.0,
    "sigma_y": 0.16,
}


def extract_calibration_windows(
    df: pd.DataFrame,
    model: TeR26HighFidelityCasADi,
    downsample_stride: int = 8,
    win_len_samples_200hz: int = 560,
    max_windows: int = 18,
) -> List[Dict[str, Any]]:
    """
    Select diverse high-excitation contiguous windows strictly from
    `split == 'calib_70'` & `valid_for_sysid == True`.
    200 Hz / stride 8 = 25 Hz (dt = 0.04 s), 560 samples = 2.8 s -> 70 nodes/window.
    """
    cal = df[(df["log_type"] == "track") & (df["split"] == "calib_70")].copy()
    candidates: List[Dict[str, Any]] = []

    for run_id, grp in cal.groupby("run_id", sort=True):
        grp = grp.reset_index(drop=True)
        valid = grp["valid_for_sysid"].to_numpy(dtype=bool)
        t_arr = grp["t_s"].to_numpy(dtype=np.float64)
        vx_arr = grp["vx_ekf_ms"].to_numpy(dtype=np.float64)
        vy_arr = grp["vy_ekf_ms"].to_numpy(dtype=np.float64)
        wz_arr = grp["wz_rads"].to_numpy(dtype=np.float64)
        ax_arr = grp["ax_cg_ms2"].to_numpy(dtype=np.float64)
        ay_arr = grp["ay_cg_ms2"].to_numpy(dtype=np.float64)
        del_arr = grp["delta_wheel_rad"].to_numpy(dtype=np.float64)
        thr_arr = grp["throttle"].to_numpy(dtype=np.float64)
        brk_arr = grp["brake"].to_numpy(dtype=np.float64)
        trq_arr = grp["trq_limit_nm"].to_numpy(dtype=np.float64)

        ax_cmd_all = model.map_pedals_to_ax_cmd_array(vx_arr, thr_arr, brk_arr, trq_arr)

        # Find contiguous runs of valid_for_sysid == True
        n = len(grp)
        i = 0
        while i < n:
            if not valid[i]:
                i += 1
                continue
            j = i
            while j < n and valid[j] and (j == i or (t_arr[j] - t_arr[j - 1]) < 0.02):
                j += 1
            seg_len = j - i
            if seg_len >= win_len_samples_200hz:
                # Slice non-overlapping windows from this valid segment
                pos = i
                while pos + win_len_samples_200hz <= j:
                    sl = slice(pos, pos + win_len_samples_200hz, downsample_stride)
                    vx_w = vx_arr[sl]
                    ay_w = ay_arr[sl]
                    ax_w = ax_arr[sl]
                    wz_w = wz_arr[sl]
                    # Score excitation: combine lateral, longitudinal, yaw-rate dynamics, and speed
                    score = (
                        float(np.std(ay_w)) / 3.5
                        + float(np.std(ax_w)) / 2.0
                        + float(np.std(wz_w)) / 0.45
                        + 0.4 * float(np.percentile(np.abs(ay_w), 95)) / 10.0
                        + 0.3 * float(np.mean(vx_w)) / 12.0
                    )
                    if float(np.min(vx_w)) >= 3.2 and score > 1.2:
                        candidates.append({
                            "run_id": str(run_id),
                            "t_start": float(t_arr[pos]),
                            "t_end": float(t_arr[pos + win_len_samples_200hz - 1]),
                            "score": score,
                            "t_s": t_arr[sl].copy(),
                            "vx": vx_w.copy(),
                            "vy": vy_arr[sl].copy(),
                            "wz": wz_w.copy(),
                            "ax": ax_w.copy(),
                            "ay": ay_w.copy(),
                            "delta": del_arr[sl].copy(),
                            "ax_cmd": ax_cmd_all[sl].copy(),
                        })
                    pos += win_len_samples_200hz
            i = j

    # Balance across runs: pick top windows per run up to max_windows
    by_run: Dict[str, List[Dict[str, Any]]] = {}
    for c in candidates:
        by_run.setdefault(c["run_id"], []).append(c)
    for rid in by_run:
        by_run[rid].sort(key=lambda d: d["score"], reverse=True)

    selected: List[Dict[str, Any]] = []
    per_run_cap = max(1, max_windows // max(len(by_run), 1))
    for rid in sorted(by_run.keys()):
        selected.extend(by_run[rid][:per_run_cap])
    # Fill remaining slots with highest overall scores not yet selected
    used = {(w["run_id"], w["t_start"]) for w in selected}
    rest = [c for c in sorted(candidates, key=lambda d: d["score"], reverse=True) if (c["run_id"], c["t_start"]) not in used]
    while len(selected) < max_windows and rest:
        selected.append(rest.pop(0))

    return selected


def run_multiple_shooting_sysid() -> Dict[str, Any]:
    t0_all = time.perf_counter()
    print("[SysID] Loading certified ground truth parquet...", flush=True)
    df = pd.read_parquet(PARQUET_PATH)

    model = TeR26HighFidelityCasADi(use_3d_kinematics=True, use_thermal_coupling=True)
    nom_dict = default_sysid_theta(model.params)
    th_nom = theta_dict_to_vec(nom_dict)

    windows = extract_calibration_windows(df, model, downsample_stride=10, win_len_samples_200hz=480, max_windows=12)
    total_nodes = sum(len(w["t_s"]) for w in windows)
    print(f"[SysID] Selected {len(windows)} diverse calib_70 windows ({total_nodes} nodes @ 20 Hz, 0 val_30 samples).", flush=True)
    for idx, w in enumerate(windows[:6]):
        print(
            f"  win[{idx:02d}] {w['run_id']} t=[{w['t_start']:.1f},{w['t_end']:.1f}]s "
            f"score={w['score']:.2f} max|ay|={np.max(np.abs(w['ay'])):.2f} max|ax|={np.max(np.abs(w['ax'])):.2f}",
            flush=True,
        )

    # Normalize parameter decision vector u_th in [-1, 1] around nominal for IPOPT conditioning
    n_th = len(SYSID_PARAM_NAMES)
    lb_th = np.array([PARAM_BOUNDS[k][0] for k in SYSID_PARAM_NAMES], dtype=np.float64)
    ub_th = np.array([PARAM_BOUNDS[k][1] for k in SYSID_PARAM_NAMES], dtype=np.float64)
    scale_th = 0.5 * (ub_th - lb_th)
    mid_th = 0.5 * (ub_th + lb_th)
    prior_std = np.array([PARAM_PRIOR_STD[k] for k in SYSID_PARAM_NAMES], dtype=np.float64)

    opti = ca.Opti()
    z_th = opti.variable(n_th)
    theta_sym = mid_th + scale_th * z_th
    opti.subject_to(opti.bounded(-1.0, z_th, 1.0))
    opti.set_initial(z_th, (th_nom - mid_th) / scale_th)

    # Measurement noise scales for weighted NLS
    s_ax = 1.00   # m/s^2
    s_ay = 1.10   # m/s^2
    s_wz = 0.10   # rad/s
    s_vx = 0.60   # m/s
    s_vy = 0.45   # m/s (EKF estimated vy, lighter weight)

    t4_const = ca.DM([55.0, 55.0, 55.0, 55.0])
    cost = 0.0
    n_obs = 0

    for w in windows:
        m = len(w["t_s"])
        dt_w = float(np.median(np.diff(w["t_s"])))
        X_w = opti.variable(5, m)  # [vx, vy, wz, alpha_f, alpha_r]

        # Feasible warm start from forward RK4 rollout at nominal theta (inf_pr = 0 at iter 0)
        x_init = np.zeros((5, m), dtype=np.float64)
        vx0 = float(np.clip(w["vx"][0], 1.5, 35.0))
        vy0 = float(np.clip(w["vy"][0], -4.0, 4.0))
        wz0 = float(np.clip(w["wz"][0], -3.0, 3.0))
        af0 = float(np.clip(w["delta"][0] - math.atan2(vy0 + model.params.a_cog * wz0, vx0), -0.35, 0.35))
        ar0 = float(np.clip(-math.atan2(vy0 - model.params.b_cog * wz0, vx0), -0.35, 0.35))
        x_init[:, 0] = [vx0, vy0, wz0, af0, ar0]
        for k in range(m - 1):
            u_k_num = np.array([w["delta"][k], w["ax_cmd"][k], w["ax"][k], w["ay"][k]], dtype=np.float64)
            xn, _ = model.fn_rk4(x_init[:, k], u_k_num, [55.0, 55.0, 55.0, 55.0], th_nom, dt_w)
            x_init[:, k + 1] = np.clip(
                np.asarray(xn, dtype=np.float64).ravel(),
                [1.0, -5.5, -3.2, -0.40, -0.40],
                [34.0, 5.5, 3.2, 0.40, 0.40],
            )
        opti.set_initial(X_w, x_init)

        # State bounds for solver safety
        opti.subject_to(opti.bounded(0.8, X_w[0, :], 35.0))
        opti.subject_to(opti.bounded(-6.0, X_w[1, :], 6.0))
        opti.subject_to(opti.bounded(-3.5, X_w[2, :], 3.5))
        opti.subject_to(opti.bounded(-0.42, X_w[3, :], 0.42))
        opti.subject_to(opti.bounded(-0.42, X_w[4, :], 0.42))

        for k in range(m):
            u_k = ca.DM([
                float(w["delta"][k]),
                float(w["ax_cmd"][k]),
                float(w["ax"][k]),
                float(w["ay"][k]),
            ])
            if k < m - 1:
                x_next_k, y_k = model.fn_rk4(X_w[:, k], u_k, t4_const, theta_sym, dt_w)
                opti.subject_to(X_w[:, k + 1] == x_next_k)
            else:
                _, y_k = model.fn_ode(X_w[:, k], u_k, t4_const, theta_sym)

            ax_k = y_k[0]
            ay_k = y_k[1]
            cost = cost + (
                0.70 * ((ax_k - float(w["ax"][k])) / s_ax) ** 2
                + 1.35 * ((ay_k - float(w["ay"][k])) / s_ay) ** 2
                + 1.50 * ((X_w[2, k] - float(w["wz"][k])) / s_wz) ** 2
                + 0.80 * ((X_w[0, k] - float(w["vx"][k])) / s_vx) ** 2
                + 0.20 * ((X_w[1, k] - float(w["vy"][k])) / s_vy) ** 2
            )
            n_obs += 1

    # Normalize data loss by number of observations + weak physical Tikhonov prior
    reg_weight = 0.015
    reg_term = reg_weight * ca.sumsqr((theta_sym - th_nom) / prior_std)
    total_obj = (cost / float(max(n_obs, 1))) + reg_term
    opti.minimize(total_obj)

    opts = {
        "print_time": False,
        "ipopt.print_level": 5,
        "ipopt.max_iter": 250,
        "ipopt.tol": 1e-4,
        "ipopt.acceptable_tol": 1e-3,
        "ipopt.acceptable_iter": 10,
        "ipopt.hessian_approximation": "limited-memory",
        "ipopt.limited_memory_max_history": 20,
        "ipopt.mu_strategy": "adaptive",
        "ipopt.linear_solver": "mumps",
    }
    opti.solver("ipopt", opts)

    print("[SysID] Solving CasADi + IPOPT Multiple-Shooting NLP...", flush=True)
    t0_solve = time.perf_counter()
    try:
        sol = opti.solve()
        status = sol.stats().get("return_status", "Solve_Succeeded")
        iters = int(sol.stats().get("iter_count", -1))
        th_opt = np.asarray(sol.value(theta_sym), dtype=np.float64).ravel()
        obj_val = float(sol.value(total_obj))
        data_val = float(sol.value(cost / float(max(n_obs, 1))))
    except RuntimeError as exc:
        print(f"[SysID] IPOPT stopped with exception ({exc}); extracting best iterate.", flush=True)
        status = opti.stats().get("return_status", "Recovered_Debug")
        iters = int(opti.stats().get("iter_count", -1))
        th_opt = np.asarray(opti.debug.value(theta_sym), dtype=np.float64).ravel()
        obj_val = float(opti.debug.value(total_obj))
        data_val = float(opti.debug.value(cost / float(max(n_obs, 1))))
    solve_s = time.perf_counter() - t0_solve

    opt_dict = theta_vec_to_dict(th_opt)
    print(f"[SysID] Finished in {solve_s:.2f} s | status={status} | iters={iters} | data_cost={data_val:.4f} | total_obj={obj_val:.4f}", flush=True)
    print("\n--- Identified TeR26 Physical Parameters (70% Calibration Split) ---", flush=True)
    param_table = {}
    for k in SYSID_PARAM_NAMES:
        v0 = float(nom_dict[k])
        v1 = float(opt_dict[k])
        pct = 100.0 * (v1 - v0) / max(abs(v0), 1e-9)
        lo, hi = PARAM_BOUNDS[k]
        at_bound = (abs(v1 - lo) < 1e-4 * (hi - lo)) or (abs(v1 - hi) < 1e-4 * (hi - lo))
        param_table[k] = {
            "nominal": v0,
            "identified": v1,
            "delta_pct": pct,
            "bounds": [lo, hi],
            "at_bound": bool(at_bound),
        }
        print(f"  {k:15s}: nom={v0:9.4f} -> id={v1:9.4f} ({pct:+6.2f}%)  bounds=[{lo}, {hi}] {'[BOUND]' if at_bound else ''}", flush=True)

    payload = {
        "metadata": {
            "vehicle": "TeR26 RWD",
            "method": "CasADi 3.7.2 + IPOPT Multiple-Shooting RK4 NLS",
            "data_source": str(PARQUET_PATH.name),
            "split_used": "calib_70 (valid_for_sysid == True ONLY; 0 val_30 samples)",
            "n_windows": len(windows),
            "total_nodes": total_nodes,
            "sample_rate_hz": 20.0,
            "solve_time_s": solve_s,
            "total_wall_s": time.perf_counter() - t0_all,
            "ipopt_status": status,
            "ipopt_iterations": iters,
            "final_data_cost": data_val,
            "final_total_obj": obj_val,
        },
        "nominal_params": nom_dict,
        "identified_params": opt_dict,
        "parameter_summary": param_table,
        "calibration_windows": [
            {
                "run_id": w["run_id"],
                "t_start": w["t_start"],
                "t_end": w["t_end"],
                "score": w["score"],
                "n_nodes": len(w["t_s"]),
            }
            for w in windows
        ],
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[SysID] Saved identified parameters to {OUT_JSON}", flush=True)
    return payload


if __name__ == "__main__":
    run_multiple_shooting_sysid()
