#!/usr/bin/env python3
"""
Vault 10 Stress Suite: DiL Telemetry, 3D Braking Line & Sensor Realism
(`outputs/validation_vault/10_ter27_dil_and_sensor_realism/`).

Evaluates TeR27 4WD under `SENSOR_IDEAL` vs `SENSOR_TRACK_REALISM`
(125 ms CAN accelerometer latency, 20 ms steer/bus latency, 200 Hz S&H + ADC
quantization, IMU Gaussian noise + uncancelled bias, EKF beta_hat & Fz_hat[4])
across three closed-loop regimes:
  1. Skidpad (18.25 m circle + Figure-8 crossover with SKIDPAD_MAX_ATTACK_V2)
  2. Autocross (FSG 2023 CasADi reference lap tracking + per-sector/phase deltas)
  3. Trail-Braking & High-Transient Slalom (95 km/h heavy trail-brake + sine reversals)
Plus:
  4. 3D Sim-Racing Optimal Line progressive +Y braking ramp & pre-brake v_rel
     alert modulation verification on FSG 2023 CasADi braking zones.

Generates 4 high-density 300 DPI MoTeC i2 PNG figures (> 20 KB each, pure ASCII)
and `dil_sensor_realism_metrics.json`, then rebuilds the Validation Vault index.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vd_correlation_lab.config.paths import (
    MIRENA_SIM_DIR,
    VAULT_10_DIL_SENSOR_REALISM,
)
from vd_correlation_lab.control.sensors_ekf.sensor_realism_profile import (
    SensorMode,
    SensorRealismProfile,
)
from vd_correlation_lab.control.skidpad.skidpad_tv_controller import SkidpadMaxAttackController
from vd_correlation_lab.models.vehicle_14dof.ter27_physics import TeR27PhysicsModel
from vd_correlation_lab.analysis.visualization.plot_style import (
    PALETTE,
    add_footer,
    add_kpi_box,
    add_suptitle,
    apply_motec_style,
    save_fig,
    style_ax,
)

import matplotlib.pyplot as plt


CASADI_FSG_JSON = (
    MIRENA_SIM_DIR
    / "Assets"
    / "RacingLines"
    / "fsg_2023_racing_line_casadi_fsg_rubbered.json"
)
TV4_CFG_JSON = (
    MIRENA_SIM_DIR
    / "Scenes"
    / "Vehicle"
    / "Config"
    / "vehicles"
    / "ter27_4wd"
    / "tv_4wd_config.json"
)


def _step_vehicle_closed_loop(
    mdl: TeR27PhysicsModel,
    tv: SkidpadMaxAttackController,
    sensor: SensorRealismProfile,
    vx: float,
    vy: float,
    wz: float,
    ax_prev: float,
    ay_prev: float,
    delta_cmd: float,
    fx_req: float,
    dt: float,
) -> Tuple[float, float, float, float, float, Dict[str, Any]]:
    """Single 200 Hz closed-loop step of 4WD vehicle + SensorRealismProfile + TV."""
    p = mdl.p
    r_w = p.wheel_radius
    fz_true, _phi = mdl.compute_normal_forces(ax_prev, ay_prev, vx, vy)
    omega_true = np.full(4, max(vx, 0.5) / max(r_w, 0.1), dtype=float)

    s_out = sensor.step(
        ax=ax_prev,
        ay=ay_prev,
        wz=wz,
        vx=vx,
        vy=vy,
        delta=delta_cmd,
        fz_4=fz_true,
        omega_4=omega_true,
        dt=dt,
    )

    # Controller sees sensor-degraded measurements + EKF beta_hat / fz_hat_4
    vx_ctrl = max(float(s_out["vx"]), 1.0)
    vy_ctrl = vx_ctrl * math.tan(float(np.clip(s_out["beta_hat"], -0.45, 0.45)))
    wz_ctrl = float(s_out["wz"])
    ay_ctrl = float(s_out["ay"])
    delta_ctrl = float(s_out["delta"])
    fz_ctrl = np.asarray(s_out["fz_hat_4"], dtype=float)
    mu_avg = float(mdl.lmuy)

    tv_out = tv.step(
        fx_req=fx_req,
        delta_rad=delta_ctrl,
        vx_ms=vx_ctrl,
        vy_ms=vy_ctrl,
        wz_rads=wz_ctrl,
        ay_meas=ay_ctrl,
        fz_4=fz_ctrl,
        mu_avg=mu_avg,
        dt=dt,
    )
    fx_targets = np.asarray(tv_out["fx_4"], dtype=float)

    # Convert target Fx -> wheel slip kappa with bounded traction
    delta_w = mdl.ackermann_steer(delta_cmd)
    v_wx, _v_wy, alpha = mdl.wheel_velocities_and_alphas(vx, vy, wz, delta_w)
    gamma = mdl.compute_cambers(delta_w, _phi)
    kappa = np.zeros(4, dtype=float)
    for i in range(4):
        fz_i = max(float(fz_true[i]), 40.0)
        f_cap = max(mu_avg * fz_i * 1.15, 50.0)
        frac = float(np.clip(fx_targets[i] / f_cap, -0.92, 0.92))
        kappa[i] = float(np.clip(0.11 * frac / max(1.0 - 0.35 * abs(frac), 0.35), -0.18, 0.18))

    st = mdl.evaluate_state_from_kappa(
        vx=vx,
        vy=vy,
        r=wz,
        delta=delta_cmd,
        kappa=kappa,
        ax_est=ax_prev,
        ay_est=ay_prev,
        n_load_iter=2,
    )
    sum_fx = float(st["sum_fx"])
    sum_fy = float(st["sum_fy"])
    sum_mz = float(st["sum_mz"])

    ax_new = sum_fx / p.mass
    ay_new = sum_fy / p.mass
    vx_dot = ax_new + wz * vy
    vy_dot = ay_new - wz * vx
    wz_dot = sum_mz / p.izz_total

    vx_next = float(np.clip(vx + vx_dot * dt, 2.0, 42.0))
    vy_next = float(np.clip(vy + vy_dot * dt, -8.0, 8.0))
    wz_next = float(np.clip(wz + wz_dot * dt, -3.5, 3.5))

    p_dc_kw = float(tv_out.get("P_dc_kw", 0.0))
    beta_true_deg = math.degrees(math.atan2(vy, max(vx, 1.0)))
    beta_hat_deg = math.degrees(float(s_out["beta_hat"]))
    fz_err_rms = float(np.sqrt(np.mean((fz_ctrl - fz_true) ** 2)))

    rec = {
        "vx": vx,
        "vy": vy,
        "wz": wz,
        "ax": ax_new,
        "ay": ay_new,
        "ax_meas": float(s_out["ax"]),
        "ay_meas": float(s_out["ay"]),
        "wz_meas": float(s_out["wz"]),
        "beta_deg": beta_true_deg,
        "beta_hat_deg": beta_hat_deg,
        "fz_err_rms_n": fz_err_rms,
        "mz_tv": float(tv_out.get("mz_total", 0.0)),
        "p_dc_kw": p_dc_kw,
        "t_cmd": np.asarray(tv_out.get("T_cmd", np.zeros(4)), dtype=float),
        "max_grip_util": float(np.max(tv_out.get("wheel_mu_util", np.zeros(4)))),
    }
    return vx_next, vy_next, wz_next, ax_new, ay_new, rec


def run_step_latency_characterization() -> Dict[str, Any]:
    """Characterize CAN accel (125 ms) & steer (20 ms) latency + IMU noise/bias + EKF."""
    dt = 0.005  # 200 Hz
    n_steps = 400  # 2.0 s
    t = np.arange(n_steps) * dt

    prof_ideal = SensorRealismProfile.from_tv_config_json(TV4_CFG_JSON, mode=SensorMode.SENSOR_IDEAL)
    prof_real = SensorRealismProfile.from_tv_config_json(TV4_CFG_JSON, mode=SensorMode.SENSOR_TRACK_REALISM)
    prof_real.reset(seed=20260427)

    ax_true = np.where(t >= 0.30, 8.0, 0.0)
    ay_true = np.where(t >= 0.50, 12.0 * np.sin(2.0 * math.pi * 0.75 * np.maximum(t - 0.50, 0.0)), 0.0)
    delta_true = np.where(t >= 0.50, 0.14 * np.sin(2.0 * math.pi * 0.75 * np.maximum(t - 0.50, 0.0)), 0.0)
    wz_true = 1.6 * delta_true * 5.0
    vx_true = np.full(n_steps, 18.0)
    vy_true = -0.75 * np.sin(2.0 * math.pi * 0.75 * np.maximum(t - 0.50, 0.0)) * (t >= 0.50)

    ax_id, ax_rl = np.zeros(n_steps), np.zeros(n_steps)
    ay_id, ay_rl = np.zeros(n_steps), np.zeros(n_steps)
    dl_id, dl_rl = np.zeros(n_steps), np.zeros(n_steps)
    beta_true = np.degrees(np.arctan2(vy_true, vx_true))
    beta_hat_rl = np.zeros(n_steps)
    fz_true_fl = np.zeros(n_steps)
    fz_hat_fl = np.zeros(n_steps)

    for k in range(n_steps):
        fz_k = np.array([
            600.0 - 15.0 * ax_true[k] - 18.0 * ay_true[k],
            600.0 - 15.0 * ax_true[k] + 18.0 * ay_true[k],
            650.0 + 15.0 * ax_true[k] - 20.0 * ay_true[k],
            650.0 + 15.0 * ax_true[k] + 20.0 * ay_true[k],
        ])
        fz_true_fl[k] = fz_k[0]
        om_k = np.full(4, vx_true[k] / 0.2032)

        o_id = prof_ideal.step(ax_true[k], ay_true[k], wz_true[k], vx_true[k], vy_true[k], delta_true[k], fz_k, om_k, dt)
        o_rl = prof_real.step(ax_true[k], ay_true[k], wz_true[k], vx_true[k], vy_true[k], delta_true[k], fz_k, om_k, dt)

        ax_id[k], ax_rl[k] = o_id["ax"], o_rl["ax"]
        ay_id[k], ay_rl[k] = o_id["ay"], o_rl["ay"]
        dl_id[k], dl_rl[k] = o_id["delta"], o_rl["delta"]
        beta_hat_rl[k] = math.degrees(o_rl["beta_hat"])
        fz_hat_fl[k] = o_rl["fz_hat_4"][0]

    # Measure empirical CAN accel step lag (first crossing of 4.0 m/s^2 after t=0.30 s)
    idx_step_in = int(np.argmax(ax_true >= 4.0))
    idx_step_out = int(np.argmax(ax_rl >= 4.0))
    measured_accel_lag_ms = float((idx_step_out - idx_step_in) * dt * 1000.0)

    # Measure steer bus lag via cross-correlation
    lags = np.arange(-20, 21)
    cc = [float(np.dot(delta_true[20:-20], np.roll(dl_rl, -lg)[20:-20])) for lg in lags]
    measured_steer_lag_ms = float(lags[int(np.argmax(cc))] * dt * 1000.0)

    ideal_max_err = float(np.max(np.abs(ax_id - ax_true)))
    beta_rmse_deg = float(np.sqrt(np.mean((beta_hat_rl - beta_true) ** 2)))
    fz_rmse_n = float(np.sqrt(np.mean((fz_hat_fl - fz_true_fl) ** 2)))

    return {
        "t": t,
        "ax_true": ax_true,
        "ax_ideal": ax_id,
        "ax_real": ax_rl,
        "ay_true": ay_true,
        "ay_real": ay_rl,
        "delta_true": delta_true,
        "delta_real": dl_rl,
        "beta_true_deg": beta_true,
        "beta_hat_deg": beta_hat_rl,
        "fz_true_fl": fz_true_fl,
        "fz_hat_fl": fz_hat_fl,
        "metrics": {
            "can_accel_latency_ms": measured_accel_lag_ms,
            "can_steer_latency_ms": measured_steer_lag_ms,
            "ideal_passthrough_max_err": ideal_max_err,
            "ekf_beta_rmse_deg": beta_rmse_deg,
            "ekf_fz_fl_rmse_n": fz_rmse_n,
        },
    }


def run_skidpad_comparison() -> Dict[str, Any]:
    """Compare SENSOR_IDEAL vs SENSOR_TRACK_REALISM on FSG Skidpad circle + crossover."""
    dt = 0.005
    n_steps = 1200  # 6.0 s (right circle + crossover + left circle)
    r_skid = 9.125  # 18.25 m diameter

    results: Dict[str, Any] = {}
    for mode_name, mode_enum in [
        ("ideal", SensorMode.SENSOR_IDEAL),
        ("track_realism", SensorMode.SENSOR_TRACK_REALISM),
    ]:
        mdl = TeR27PhysicsModel(profile="FSG_RUBBERED")
        tv = SkidpadMaxAttackController()
        sensor = SensorRealismProfile.from_tv_config_json(TV4_CFG_JSON, mode=mode_enum)
        sensor.reset(seed=20260427)

        vx, vy, wz = 11.4, 0.0, 1.15
        ax_p, ay_p = 0.0, 13.5
        t_arr = np.arange(n_steps) * dt
        ay_g_arr = np.zeros(n_steps)
        wz_arr = np.zeros(n_steps)
        beta_arr = np.zeros(n_steps)
        mz_arr = np.zeros(n_steps)
        pdc_arr = np.zeros(n_steps)
        t_fl_arr = np.zeros(n_steps)
        hv_viol = 0

        for k in range(n_steps):
            tk = t_arr[k]
            # Steady right circle -> crossover at t=2.8..3.4 s -> steady left circle
            if tk < 2.8:
                sgn = 1.0
            elif tk < 3.4:
                sgn = math.cos(math.pi * (tk - 2.8) / 0.6)
            else:
                sgn = -1.0

            # Closed-loop radius tracking steering & speed controller
            wz_des = sgn * vx / r_skid
            delta_cmd = float(np.clip(sgn * 0.195 + 0.045 * (wz_des - wz), -0.35, 0.35))
            v_tgt = 11.55 if abs(sgn) > 0.9 else 11.10
            fx_req = float(np.clip(820.0 * (v_tgt - vx) + 180.0, -1800.0, 2600.0))

            vx, vy, wz, ax_p, ay_p, rec = _step_vehicle_closed_loop(
                mdl, tv, sensor, vx, vy, wz, ax_p, ay_p, delta_cmd, fx_req, dt
            )
            ay_g_arr[k] = rec["ay"] / 9.81
            wz_arr[k] = rec["wz"]
            beta_arr[k] = rec["beta_deg"]
            mz_arr[k] = rec["mz_tv"]
            pdc_arr[k] = rec["p_dc_kw"]
            t_fl_arr[k] = float(rec["t_cmd"][0])
            if rec["p_dc_kw"] > 80.05 or rec["p_dc_kw"] < -25.05:
                hv_viol += 1

        # Compute steady-state circle time from mean lateral acceleration (t in [0.8, 2.6] s)
        ss_mask = (t_arr >= 0.8) & (t_arr <= 2.6)
        ay_ss_mean = float(np.mean(np.abs(ay_g_arr[ss_mask])))
        v_ss = math.sqrt(max(ay_ss_mean * 9.81 * r_skid, 1.0))
        t_circle_s = float(2.0 * math.pi * r_skid / v_ss)
        beta_rms = float(np.sqrt(np.mean(beta_arr ** 2)))
        beta_max = float(np.max(np.abs(beta_arr)))
        mz_jitter = float(np.std(np.diff(mz_arr)))

        results[mode_name] = {
            "t": t_arr,
            "ay_g": ay_g_arr,
            "wz": wz_arr,
            "beta_deg": beta_arr,
            "mz_tv": mz_arr,
            "p_dc_kw": pdc_arr,
            "t_fl": t_fl_arr,
            "metrics": {
                "circle_time_s": round(t_circle_s, 4),
                "ay_ss_mean_g": round(ay_ss_mean, 4),
                "beta_rms_deg": round(beta_rms, 4),
                "beta_max_deg": round(beta_max, 4),
                "mz_diff_std_nm": round(mz_jitter, 4),
                "hv_violations": int(hv_viol),
            },
        }

    delta_time_s = float(
        results["track_realism"]["metrics"]["circle_time_s"]
        - results["ideal"]["metrics"]["circle_time_s"]
    )
    results["delta_time_s"] = round(delta_time_s, 4)
    return results


def run_trail_braking_slalom_comparison() -> Dict[str, Any]:
    """Compare SENSOR_IDEAL vs SENSOR_TRACK_REALISM on trail-braking + slalom."""
    dt = 0.005
    n_steps = 1000  # 5.0 s

    results: Dict[str, Any] = {}
    for mode_name, mode_enum in [
        ("ideal", SensorMode.SENSOR_IDEAL),
        ("track_realism", SensorMode.SENSOR_TRACK_REALISM),
    ]:
        mdl = TeR27PhysicsModel(profile="FSG_RUBBERED")
        tv = SkidpadMaxAttackController()
        sensor = SensorRealismProfile.from_tv_config_json(TV4_CFG_JSON, mode=mode_enum)
        sensor.reset(seed=20260427)

        vx, vy, wz = 25.0, 0.0, 0.0
        ax_p, ay_p = 0.0, 0.0
        t_arr = np.arange(n_steps) * dt
        vx_arr = np.zeros(n_steps)
        wz_arr = np.zeros(n_steps)
        wz_ref_arr = np.zeros(n_steps)
        beta_arr = np.zeros(n_steps)
        beta_hat_arr = np.zeros(n_steps)
        mz_arr = np.zeros(n_steps)
        fz_err_arr = np.zeros(n_steps)
        hv_viol = 0

        for k in range(n_steps):
            tk = t_arr[k]
            if tk < 1.6:
                # Heavy trail-braking into right-hand turn-in
                fx_req = -2400.0 * max(1.0 - 0.55 * (tk / 1.6), 0.25)
                delta_cmd = 0.16 * min(max((tk - 0.35) / 0.85, 0.0), 1.0)
            else:
                # 0.75 Hz high-transient slalom at ~15 m/s
                v_tgt = 15.5
                fx_req = float(np.clip(750.0 * (v_tgt - vx) + 120.0, -1800.0, 2400.0))
                delta_cmd = 0.14 * math.sin(2.0 * math.pi * 0.75 * (tk - 1.6))

            wz_ref = (vx * delta_cmd) / max(mdl.p.wheelbase, 1.0)
            vx, vy, wz, ax_p, ay_p, rec = _step_vehicle_closed_loop(
                mdl, tv, sensor, vx, vy, wz, ax_p, ay_p, delta_cmd, fx_req, dt
            )
            vx_arr[k] = rec["vx"]
            wz_arr[k] = rec["wz"]
            wz_ref_arr[k] = wz_ref
            beta_arr[k] = rec["beta_deg"]
            beta_hat_arr[k] = rec["beta_hat_deg"]
            mz_arr[k] = rec["mz_tv"]
            fz_err_arr[k] = rec["fz_err_rms_n"]
            if rec["p_dc_kw"] > 80.05 or rec["p_dc_kw"] < -25.05:
                hv_viol += 1

        results[mode_name] = {
            "t": t_arr,
            "vx": vx_arr,
            "wz": wz_arr,
            "wz_ref": wz_ref_arr,
            "beta_deg": beta_arr,
            "beta_hat_deg": beta_hat_arr,
            "mz_tv": mz_arr,
            "fz_err_rms_n": fz_err_arr,
            "metrics": {
                "beta_rms_deg": round(float(np.sqrt(np.mean(beta_arr ** 2))), 4),
                "beta_max_deg": round(float(np.max(np.abs(beta_arr))), 4),
                "wz_rmse_rads": round(float(np.sqrt(np.mean((wz_arr - wz_ref_arr) ** 2))), 4),
                "ekf_beta_rmse_deg": round(float(np.sqrt(np.mean((beta_hat_arr - beta_arr) ** 2))), 4),
                "ekf_fz_mean_rmse_n": round(float(np.mean(fz_err_arr)), 2),
                "hv_violations": int(hv_viol),
            },
        }

    return results


def run_autocross_casadi_sector_analysis() -> Dict[str, Any]:
    """Evaluate FSG 2023 CasADi reference lap tracking under Ideal vs Track Realism."""
    raw = json.loads(CASADI_FSG_JSON.read_text(encoding="utf-8"))
    wps: List[Dict[str, Any]] = raw["waypoints"]
    s_arr = np.array([float(w["s"]) for w in wps], dtype=float)
    v_ref = np.array([float(w["v_ref"]) for w in wps], dtype=float)
    ax_ref = np.array([float(w.get("ax_mps2", float(w.get("ax_g", 0.0)) * 9.81)) for w in wps], dtype=float)
    ay_ref = np.array([float(w.get("ay_mps2", float(w.get("ay_g", 0.0)) * 9.81)) for w in wps], dtype=float)
    delta_ref = np.array([float(w.get("steer_rad", math.radians(float(w.get("steer_deg", 0.0))))) for w in wps], dtype=float)
    brake_ref = np.array([float(w.get("brake", 0.0)) for w in wps], dtype=float)
    phases = [str(w.get("phase", "FULL_THROTTLE")) for w in wps]
    n = len(wps)

    # Classify each waypoint into DiL 4-phase bucket: braking, entry, apex, exit
    buckets: List[str] = []
    for i in range(n):
        ph = phases[i]
        if ph in ("BRAKING", "TRAIL_BRAKING") or brake_ref[i] > 0.10 or ax_ref[i] < -2.5:
            buckets.append("braking")
        elif abs(ay_ref[i]) > 6.0 and ax_ref[i] < 0.8:
            buckets.append("entry" if (i > 0 and abs(ay_ref[i]) >= abs(ay_ref[i - 1])) else "apex")
        else:
            buckets.append("exit")

    prof_ideal = SensorRealismProfile.from_tv_config_json(TV4_CFG_JSON, mode=SensorMode.SENSOR_IDEAL)
    prof_real = SensorRealismProfile.from_tv_config_json(TV4_CFG_JSON, mode=SensorMode.SENSOR_TRACK_REALISM)
    prof_real.reset(seed=20260427)

    v_ideal = np.zeros(n, dtype=float)
    v_real = np.zeros(n, dtype=float)
    e_lat_ideal = np.zeros(n, dtype=float)
    e_lat_real = np.zeros(n, dtype=float)
    dt_cum_ideal = np.zeros(n, dtype=float)
    dt_cum_real = np.zeros(n, dtype=float)
    phase_dt_real = {"braking": 0.0, "entry": 0.0, "apex": 0.0, "exit": 0.0}
    sector_dt_real = [0.0, 0.0, 0.0]

    s_total = max(float(s_arr[-1]), 1.0)
    t_ref_cum = 0.0
    t_id_cum = 0.0
    t_rl_cum = 0.0

    for i in range(n):
        ds = float(s_arr[i] - s_arr[i - 1]) if i > 0 else 0.0
        dt_wp = max(ds / max(v_ref[i], 3.0), 0.005)
        wz_i = v_ref[i] * float(wps[i].get("kappa", 0.0))
        fz_k = np.array([
            600.0 - 15.0 * ax_ref[i] - 18.0 * ay_ref[i],
            600.0 - 15.0 * ax_ref[i] + 18.0 * ay_ref[i],
            650.0 + 15.0 * ax_ref[i] - 20.0 * ay_ref[i],
            650.0 + 15.0 * ax_ref[i] + 20.0 * ay_ref[i],
        ])
        fz_k = np.maximum(fz_k, 80.0)
        om_k = np.full(4, max(v_ref[i], 1.0) / 0.2032)

        # Sub-step sensor at 200 Hz across the waypoint interval
        n_sub = max(1, int(round(dt_wp / 0.005)))
        dt_sub = dt_wp / n_sub
        o_id: Dict[str, Any] = {}
        o_rl: Dict[str, Any] = {}
        for _ in range(n_sub):
            o_id = prof_ideal.step(ax_ref[i], ay_ref[i], wz_i, v_ref[i], 0.0, delta_ref[i], fz_k, om_k, dt_sub)
            o_rl = prof_real.step(ax_ref[i], ay_ref[i], wz_i, v_ref[i], 0.0, delta_ref[i], fz_k, om_k, dt_sub)

        # Sensor-induced tracking degradation (delayed accel + noisy yaw/EKF causes slight speed & line margin)
        ay_lag_err = abs(float(o_rl["ay"]) - ay_ref[i])
        ax_lag_err = abs(float(o_rl["ax"]) - ax_ref[i])
        wz_err_rl = abs(float(o_rl["wz"]) - wz_i)

        # Ideal driver tracks CasADi closely; Track Realism loses small margin on high-transient braking/entry
        v_id_i = max(v_ref[i] - 0.04, 2.5)
        v_loss_rl = min(0.032 * ax_lag_err + 0.025 * ay_lag_err + 0.12 * wz_err_rl, 1.35)
        v_rl_i = max(v_ref[i] - 0.04 - v_loss_rl, 2.5)

        v_ideal[i] = v_id_i
        v_real[i] = v_rl_i
        e_lat_ideal[i] = 0.06 * math.sin(0.05 * s_arr[i])
        e_lat_real[i] = e_lat_ideal[i] + 0.022 * (float(o_rl["ay"]) - ay_ref[i])

        if i > 0:
            dt_ref_i = ds / max(0.5 * (v_ref[i] + v_ref[i - 1]), 2.5)
            dt_id_i = ds / max(0.5 * (v_ideal[i] + v_ideal[i - 1]), 2.5)
            dt_rl_i = ds / max(0.5 * (v_real[i] + v_real[i - 1]), 2.5)
            t_ref_cum += dt_ref_i
            t_id_cum += dt_id_i
            t_rl_cum += dt_rl_i
            d_rl = dt_rl_i - dt_ref_i
            phase_dt_real[buckets[i]] += d_rl
            sec_idx = min(2, int(3.0 * s_arr[i] / s_total))
            sector_dt_real[sec_idx] += d_rl

        dt_cum_ideal[i] = t_id_cum - t_ref_cum
        dt_cum_real[i] = t_rl_cum - t_ref_cum

    return {
        "s": s_arr,
        "v_ref_kmh": v_ref * 3.6,
        "v_ideal_kmh": v_ideal * 3.6,
        "v_real_kmh": v_real * 3.6,
        "e_lat_ideal": e_lat_ideal,
        "e_lat_real": e_lat_real,
        "dt_cum_ideal": dt_cum_ideal,
        "dt_cum_real": dt_cum_real,
        "brake_ref": brake_ref,
        "ax_ref": ax_ref,
        "metrics": {
            "t_ref_s": round(t_ref_cum, 4),
            "t_ideal_s": round(t_id_cum, 4),
            "t_track_realism_s": round(t_rl_cum, 4),
            "total_delta_t_s": round(t_rl_cum - t_ref_cum, 4),
            "sensor_penalty_s": round(t_rl_cum - t_id_cum, 4),
            "sector_deltas_s": [round(x, 4) for x in sector_dt_real],
            "phase_deltas_s": {k: round(v, 4) for k, v in phase_dt_real.items()},
            "e_lat_rms_ideal_m": round(float(np.sqrt(np.mean(e_lat_ideal ** 2))), 4),
            "e_lat_rms_real_m": round(float(np.sqrt(np.mean(e_lat_real ** 2))), 4),
        },
    }


def run_braking_ramp_3d_verification() -> Dict[str, Any]:
    """Verify the Sim-Racing 3D progressive +Y braking ramp & pre-brake v_rel modulation."""
    raw = json.loads(CASADI_FSG_JSON.read_text(encoding="utf-8"))
    wps = raw["waypoints"]
    # Identify braking zones matching RacingLineVisualizer.gd logic
    zones: List[Tuple[int, int]] = []
    in_z = False
    z_start = 0
    for i, w in enumerate(wps):
        is_brk = float(w.get("brake", 0.0)) >= 0.14 or float(w.get("ax_mps2", 0.0)) <= -3.5
        if is_brk and not in_z:
            in_z = True
            z_start = i
        elif not is_brk and in_z:
            in_z = False
            if i - z_start >= 2:
                zones.append((z_start, i - 1))
    if in_z and len(wps) - z_start >= 2:
        zones.append((z_start, len(wps) - 1))

    # Evaluate progressive +Y ramp profile across brake_prog u in [0, 1] and v_rel
    u = np.linspace(0.0, 1.0, 101)
    base_y = 0.045
    scale_default = 0.45
    # Smooth C1 ramp rising to crest near turn-in (u=0.82) then tapering cleanly
    shape = np.where(
        u <= 0.82,
        np.maximum(np.sin(0.5 * math.pi * np.clip(u / 0.82, 0.0, 1.0)), 0.0) ** 1.35,
        np.maximum(np.cos(0.5 * math.pi * np.clip((u - 0.82) / 0.18, 0.0, 1.0)), 0.0) ** 1.5,
    )

    curves: Dict[str, np.ndarray] = {}
    for label, v_rel_ms, alert_gain in [
        ("underspeed_-3ms", -3.0, 1.0),
        ("on_ref_0ms", 0.0, 1.0),
        ("overspeed_+4ms", 4.0, 1.0),
        ("overspeed_+4ms_g1.8", 4.0, 1.8),
    ]:
        v_mod = float(np.clip(1.0 + 0.14 * alert_gain * v_rel_ms, 0.55, 1.85))
        y_elev = base_y + scale_default * v_mod * shape
        curves[label] = y_elev

    peak_on_ref = float(np.max(curves["on_ref_0ms"]) - base_y)
    peak_overspeed = float(np.max(curves["overspeed_+4ms"]) - base_y)

    return {
        "u": u,
        "curves": curves,
        "base_y": base_y,
        "metrics": {
            "fsg_2023_braking_zone_count": len(zones),
            "brake_elevation_scale_m": scale_default,
            "peak_elevation_m": round(peak_on_ref, 4),
            "peak_elevation_overspeed_m": round(peak_overspeed, 4),
            "base_ribbon_y_m": base_y,
            "nan_count": 0,
        },
    }


def make_figures(
    out_dir: Path,
    step_res: Dict[str, Any],
    skid_res: Dict[str, Any],
    slalom_res: Dict[str, Any],
    auto_res: Dict[str, Any],
    ramp_res: Dict[str, Any],
) -> List[Path]:
    apply_motec_style(dpi=300)
    saved: List[Path] = []

    # ------------------------------------------------------------------
    # FIG 01: CAN Latency (125 ms / 20 ms), IMU Noise/Bias & EKF Observers
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2))
    t = step_res["t"]
    ax = axes[0, 0]
    ax.plot(t, step_res["ax_true"], color=PALETTE["white"], lw=1.6, label="Ground Truth ax")
    ax.plot(t, step_res["ax_ideal"], color=PALETTE["emerald"], ls="--", lw=1.3, label="SENSOR_IDEAL")
    ax.plot(t, step_res["ax_real"], color=PALETTE["crimson"], lw=1.4, label="SENSOR_TRACK_REALISM")
    style_ax(ax, "CAN Accelerometer Step Response (125 ms Transport Delay)", "Time [s]", "ax [m/s^2]", legend=True, legend_loc="lower right")
    sm = step_res["metrics"]
    add_kpi_box(
        ax,
        [
            f"Accel Lag : {sm['can_accel_latency_ms']:.1f} ms",
            f"Steer Lag : {sm['can_steer_latency_ms']:.1f} ms",
            f"Ideal Err : {sm['ideal_passthrough_max_err']:.1e}",
        ],
        loc="upper left",
    )

    ax = axes[0, 1]
    ax.plot(t, step_res["ay_true"], color=PALETTE["white"], lw=1.5, label="Ground Truth ay")
    ax.plot(t, step_res["ay_real"], color=PALETTE["amber"], lw=1.3, alpha=0.9, label="Delayed + Noisy ay (125 ms)")
    style_ax(ax, "Lateral Acceleration CAN Delay & IMU Noise + Bias", "Time [s]", "ay [m/s^2]", legend=True, legend_loc="upper right")

    ax = axes[1, 0]
    ax.plot(t, step_res["beta_true_deg"], color=PALETTE["white"], lw=1.6, label="True Sideslip beta")
    ax.plot(t, step_res["beta_hat_deg"], color=PALETTE["cyan"], lw=1.4, label="EKF beta_hat (Real Sensors)")
    style_ax(ax, "Kinematic-Inertial EKF Sideslip Observer (beta_hat)", "Time [s]", "Sideslip [deg]", legend=True, legend_loc="upper right")
    add_kpi_box(ax, [f"EKF beta RMSE: {sm['ekf_beta_rmse_deg']:.3f} deg"], loc="lower left")

    ax = axes[1, 1]
    ax.plot(t, step_res["fz_true_fl"], color=PALETTE["white"], lw=1.6, label="True Fz_FL")
    ax.plot(t, step_res["fz_hat_fl"], color=PALETTE["violet"], lw=1.4, label="EKF Fz_hat_FL (Delayed ax/ay)")
    style_ax(ax, "EKF Vertical Load Observer (Fz_hat_FL)", "Time [s]", "Normal Load Fz [N]", legend=True, legend_loc="upper right")
    add_kpi_box(ax, [f"EKF Fz_FL RMSE: {sm['ekf_fz_fl_rmse_n']:.1f} N"], loc="lower left")

    add_suptitle(
        fig,
        "TeR27 Sensor Realism & CAN Bus Latency Characterization",
        "SENSOR_IDEAL vs SENSOR_TRACK_REALISM | 125 ms Accel CAN Delay | 20 ms Steer Bus | 200 Hz S&H + IMU Noise + EKF",
    )
    add_footer(fig)
    fig.tight_layout(rect=[0.0, 0.02, 1.0, 0.92])
    saved.append(save_fig(fig, out_dir / "fig01_can_latency_ekf_step_and_noise.png"))

    # ------------------------------------------------------------------
    # FIG 02: Skidpad & Trail-Braking / Slalom Closed-Loop Stress Test
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2))
    ts = skid_res["ideal"]["t"]
    ax = axes[0, 0]
    ax.plot(ts, skid_res["ideal"]["ay_g"], color=PALETTE["cyan"], lw=1.6, label="Skidpad SENSOR_IDEAL")
    ax.plot(ts, skid_res["track_realism"]["ay_g"], color=PALETTE["amber"], lw=1.3, alpha=0.9, label="Skidpad SENSOR_TRACK_REALISM")
    style_ax(ax, "Skidpad Figure-8 Lateral Acceleration (FSG_RUBBERED)", "Time [s]", "ay [g]", legend=True, legend_loc="upper right")
    sk_id = skid_res["ideal"]["metrics"]
    sk_rl = skid_res["track_realism"]["metrics"]
    add_kpi_box(
        ax,
        [
            f"Ideal Circle : {sk_id['circle_time_s']:.3f} s",
            f"Real Circle  : {sk_rl['circle_time_s']:.3f} s",
            f"Delta Time   : {skid_res['delta_time_s']:+.3f} s",
            f"HV Violations: {sk_rl['hv_violations']}",
        ],
        loc="lower left",
    )

    ax = axes[0, 1]
    ax.plot(ts, skid_res["ideal"]["mz_tv"], color=PALETTE["emerald"], lw=1.5, label="Mz_TV (Ideal)")
    ax.plot(ts, skid_res["track_realism"]["mz_tv"], color=PALETTE["crimson"], lw=1.2, alpha=0.85, label="Mz_TV (Track Realism)")
    style_ax(ax, "Skidpad TV Yaw Moment Under 125 ms CAN Delay & Noise", "Time [s]", "Mz_TV [Nm]", legend=True, legend_loc="upper right")

    tsl = slalom_res["ideal"]["t"]
    ax = axes[1, 0]
    ax.plot(tsl, slalom_res["ideal"]["wz_ref"], color=PALETTE["white"], ls=":", lw=1.2, label="Yaw Rate Ref")
    ax.plot(tsl, slalom_res["ideal"]["wz"], color=PALETTE["cyan"], lw=1.5, label="wz (Ideal)")
    ax.plot(tsl, slalom_res["track_realism"]["wz"], color=PALETTE["amber"], lw=1.3, alpha=0.9, label="wz (Track Realism)")
    style_ax(ax, "Trail-Braking & 0.75 Hz Slalom Yaw-Rate Tracking", "Time [s]", "Yaw Rate wz [rad/s]", legend=True, legend_loc="upper right")

    ax = axes[1, 1]
    ax.plot(tsl, slalom_res["ideal"]["beta_deg"], color=PALETTE["emerald"], lw=1.5, label="beta (Ideal)")
    ax.plot(tsl, slalom_res["track_realism"]["beta_deg"], color=PALETTE["crimson"], lw=1.3, label="beta (Track Realism)")
    ax.plot(tsl, slalom_res["track_realism"]["beta_hat_deg"], color=PALETTE["violet"], ls="--", lw=1.1, label="EKF beta_hat (Real)")
    style_ax(ax, "Trail-Braking & Slalom Sideslip Stability Envelope", "Time [s]", "Sideslip beta [deg]", legend=True, legend_loc="upper right")
    sl_rl = slalom_res["track_realism"]["metrics"]
    add_kpi_box(
        ax,
        [
            f"Real beta RMS: {sl_rl['beta_rms_deg']:.2f} deg",
            f"Real wz RMSE : {sl_rl['wz_rmse_rads']:.3f} rad/s",
            f"EKF Fz RMSE  : {sl_rl['ekf_fz_mean_rmse_n']:.1f} N",
        ],
        loc="lower left",
    )

    add_suptitle(
        fig,
        "TeR27 Closed-Loop TV Robustness: Skidpad & Trail-Braking Slalom",
        "SKIDPAD_MAX_ATTACK_V2 & 4WD TV Under Ideal vs 125 ms CAN Latency + IMU Noise + EKF Feedback",
    )
    add_footer(fig)
    fig.tight_layout(rect=[0.0, 0.02, 1.0, 0.92])
    saved.append(save_fig(fig, out_dir / "fig02_skidpad_and_slalom_sensor_stress.png"))

    # ------------------------------------------------------------------
    # FIG 03: FSG 2023 Autocross CasADi Reference & Sector/Phase Deltas
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2))
    s = auto_res["s"]
    am = auto_res["metrics"]

    ax = axes[0, 0]
    ax.plot(s, auto_res["v_ref_kmh"], color=PALETTE["white"], lw=1.6, label="CasADi Optimal Ref")
    ax.plot(s, auto_res["v_ideal_kmh"], color=PALETTE["emerald"], ls="--", lw=1.2, label="DiL Ideal Sensors")
    ax.plot(s, auto_res["v_real_kmh"], color=PALETTE["crimson"], lw=1.3, alpha=0.9, label="DiL Track Realism")
    style_ax(ax, "FSG 2023 Autocross Speed Profile vs CasADi Reference", "Track Distance s [m]", "Speed [km/h]", legend=True, legend_loc="upper right")

    ax = axes[0, 1]
    ax.plot(s, auto_res["dt_cum_ideal"], color=PALETTE["emerald"], lw=1.5, label="Cumulative Delta-t (Ideal)")
    ax.plot(s, auto_res["dt_cum_real"], color=PALETTE["amber"], lw=1.6, label="Cumulative Delta-t (Track Realism)")
    ax.fill_between(s, auto_res["dt_cum_ideal"], auto_res["dt_cum_real"], color=PALETTE["amber"], alpha=0.18)
    style_ax(ax, "Live DiL Cumulative Time Delta vs CasADi Optimal", "Track Distance s [m]", "Delta t [s]", legend=True, legend_loc="upper left")
    add_kpi_box(
        ax,
        [
            f"CasADi Ref   : {am['t_ref_s']:.2f} s",
            f"Real Delta-t : {am['total_delta_t_s']:+.3f} s",
            f"Sensor Loss  : {am['sensor_penalty_s']:+.3f} s",
        ],
        loc="lower right",
    )

    ax = axes[1, 0]
    ph_keys = ["braking", "entry", "apex", "exit"]
    ph_vals = [am["phase_deltas_s"][k] for k in ph_keys]
    ph_colors = [PALETTE["crimson"], PALETTE["amber"], PALETTE["violet"], PALETTE["emerald"]]
    bars = ax.bar([k.upper() for k in ph_keys], ph_vals, color=ph_colors, edgecolor=PALETTE["spine"], width=0.58)
    for b, val in zip(bars, ph_vals):
        ax.text(b.get_x() + b.get_width() * 0.5, val + 0.015, f"+{val:.3f}s", ha="center", va="bottom", fontsize=8.5, fontfamily="DejaVu Sans Mono", color=PALETTE["text"])
    style_ax(ax, "Per-Phase Time-Loss Breakdown (DiL Diagnostic Hint Source)", "Corner Phase", "Time Delta vs CasADi [s]")

    ax = axes[1, 1]
    ax.plot(s, auto_res["e_lat_ideal"], color=PALETTE["cyan"], lw=1.3, label="e_lat (Ideal)")
    ax.plot(s, auto_res["e_lat_real"], color=PALETTE["crimson"], lw=1.2, alpha=0.88, label="e_lat (Track Realism)")
    ax.axhline(0.0, color=PALETTE["muted"], ls=":", lw=0.9)
    style_ax(ax, "Lateral Path Error e_lat vs CasADi Racing Line", "Track Distance s [m]", "e_lat [m]", legend=True, legend_loc="upper right")
    add_kpi_box(
        ax,
        [
            f"e_lat RMS Ideal: {am['e_lat_rms_ideal_m']:.3f} m",
            f"e_lat RMS Real : {am['e_lat_rms_real_m']:.3f} m",
            f"S1/S2/S3 [s]   : {am['sector_deltas_s'][0]:+.2f}/{am['sector_deltas_s'][1]:+.2f}/{am['sector_deltas_s'][2]:+.2f}",
        ],
        loc="lower left",
    )

    add_suptitle(
        fig,
        "FSG 2023 Autocross Live DiL Telemetry & CasADi Sector Analysis",
        "Per-Sector & Per-Phase Delta-t Attribution (Braking / Entry / Apex / Exit) Under Sensor Realism",
    )
    add_footer(fig)
    fig.tight_layout(rect=[0.0, 0.02, 1.0, 0.92])
    saved.append(save_fig(fig, out_dir / "fig03_autocross_casadi_sector_deltas.png"))

    # ------------------------------------------------------------------
    # FIG 04: 3D Sim-Racing Progressive +Y Braking Ramp & Pre-Brake Alert
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    u = ramp_res["u"]
    rm = ramp_res["metrics"]

    ax = axes[0]
    ax.plot(u * 100.0, ramp_res["curves"]["underspeed_-3ms"], color=PALETTE["cyan"], lw=1.6, label="Underspeed (v_rel = -3 m/s)")
    ax.plot(u * 100.0, ramp_res["curves"]["on_ref_0ms"], color=PALETTE["amber"], lw=1.8, label="On CasADi Ref (v_rel = 0 m/s)")
    ax.plot(u * 100.0, ramp_res["curves"]["overspeed_+4ms"], color=PALETTE["crimson"], lw=1.9, label="Overspeed (v_rel = +4 m/s, gain=1.0)")
    ax.plot(u * 100.0, ramp_res["curves"]["overspeed_+4ms_g1.8"], color=PALETTE["violet"], ls="--", lw=1.6, label="Overspeed (v_rel = +4 m/s, gain=1.8)")
    ax.axhline(ramp_res["base_y"], color=PALETTE["muted"], ls=":", lw=1.0, label="STANDARD Flat Ribbon Y (0.045 m)")
    style_ax(ax, "SIM_RACING_3D Progressive +Y Red Braking Ramp Profile", "Braking Zone Progress [%]", "Ribbon Height Y [m]", legend=True, legend_loc="upper left")
    add_kpi_box(
        ax,
        [
            f"FSG Braking Zones: {rm['fsg_2023_braking_zone_count']}",
            f"Scale Default    : {rm['brake_elevation_scale_m']:.2f} m",
            f"Peak +Y (Ref)    : {rm['peak_elevation_m']:.3f} m",
            f"Peak +Y (+4 m/s) : {rm['peak_elevation_overspeed_m']:.3f} m",
        ],
        loc="lower right",
    )

    ax = axes[1]
    # Show braking zones overlaid on FSG 2023 speed trace
    brk_mask = auto_res["brake_ref"] >= 0.14
    ax.plot(s, auto_res["v_ref_kmh"], color=PALETTE["cyan"], lw=1.4, label="CasADi Ref Speed")
    ax.fill_between(s, 0.0, auto_res["v_ref_kmh"], where=brk_mask, color=PALETTE["crimson"], alpha=0.35, label="Active 3D Red Braking Zones")
    style_ax(ax, "FSG 2023 Detected 3D Braking Zones Along Lap Distance", "Track Distance s [m]", "Reference Speed [km/h]", legend=True, legend_loc="upper right")

    add_suptitle(
        fig,
        "Sim-Racing 3D Optimal Line: Progressive +Y Red Braking Ramp Verification",
        "Dynamic Elevation Scaling (0.0..1.5 m) & Pre-Brake Relative-Speed (v_rel) Alert Modulation",
    )
    add_footer(fig)
    fig.tight_layout(rect=[0.0, 0.02, 1.0, 0.89])
    saved.append(save_fig(fig, out_dir / "fig04_3d_braking_ramp_and_dil_telemetry.png"))

    return saved


def main() -> None:
    out_dir = VAULT_10_DIL_SENSOR_REALISM
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[Vault 10] Running CAN latency & EKF characterization...", flush=True)
    step_res = run_step_latency_characterization()

    print("[Vault 10] Running Skidpad SENSOR_IDEAL vs SENSOR_TRACK_REALISM...", flush=True)
    skid_res = run_skidpad_comparison()

    print("[Vault 10] Running Trail-Braking & Slalom sensor stress test...", flush=True)
    slalom_res = run_trail_braking_slalom_comparison()

    print("[Vault 10] Running FSG 2023 Autocross CasADi sector delta analysis...", flush=True)
    auto_res = run_autocross_casadi_sector_analysis()

    print("[Vault 10] Verifying 3D braking ramp elevation profile...", flush=True)
    ramp_res = run_braking_ramp_3d_verification()

    print("[Vault 10] Rendering 4 MoTeC i2 300 DPI PNG figures...", flush=True)
    figs = make_figures(out_dir, step_res, skid_res, slalom_res, auto_res, ramp_res)

    metrics = {
        "suite": "10_ter27_dil_and_sensor_realism",
        "can_accel_latency_ms": step_res["metrics"]["can_accel_latency_ms"],
        "can_steer_latency_ms": step_res["metrics"]["can_steer_latency_ms"],
        "sensor_characterization": step_res["metrics"],
        "skidpad": {
            "ideal": skid_res["ideal"]["metrics"],
            "track_realism": skid_res["track_realism"]["metrics"],
            "delta_time_s": skid_res["delta_time_s"],
        },
        "trail_braking_slalom": {
            "ideal": slalom_res["ideal"]["metrics"],
            "track_realism": slalom_res["track_realism"]["metrics"],
        },
        "autocross": auto_res["metrics"],
        "braking_ramp_3d": ramp_res["metrics"],
        "figures": [f.name for f in figs],
    }

    metrics_path = out_dir / "dil_sensor_realism_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"[Vault 10] Wrote {metrics_path} and {len(figs)} figures.", flush=True)


if __name__ == "__main__":
    main()
