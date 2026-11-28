#!/usr/bin/env python3
"""
Extreme Asymmetric Stress-Testing & Q_w Refinement Suite for Stochastic Tube-MPC
(TeR27 4WD on compiled 14-DOF CasADi model libter_diff_14dof.so).

Stress-tests OnlineGripEstimator + StochasticTubeMPCController under
SENSOR_TRACK_REALISM (125 ms CAN accel latency, 20 ms steer/wheel CAN latency,
15 ms jitter, 1.0x realistic IMU noise) on two extreme asymmetric failure modes:
  - Scenario A (Patched Asphalt / Split-mu):
      Outer wheels on rubbered asphalt (lambda_mu = 0.57) while inner wheels
      hit a dirty/wet patch (lambda_mu = 0.40) mid-Autocross cornering.
  - Scenario B1 (Sudden Motor Loss / 3WD Limp-Home - Heavy Braking / Entry):
      Instantaneous cut to 0 Nm of FL motor mid-heavy braking into corner entry.
  - Scenario B2 (Sudden Motor Loss / 3WD Limp-Home - Mid-Slalom Transition):
      Instantaneous cut to 0 Nm of FL motor mid-slalom lateral transition.

Compares three controllers apples-to-apples:
  1. deterministic_4wd_tv        (no stochastic tube backoff)
  2. baseline_tube_mpc           (symmetric Q_w tube-MPC)
  3. refined_asymmetric_tube_mpc (refined Q_w with sigma_Mz_asym^2 + 3WD limp-home)

Exports full trajectories, KPIs, and minimum sector-time penalty sweep to:
  outputs/differentiable_deep_validation/tube_mpc_asymmetric_stress_report.json
"""

from __future__ import annotations

from collections import deque
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vd_correlation_lab.control.sensors_ekf.online_mu_and_relaxation import (
    LAMBDA_MU_COLD_TEST_OCT,
    LAMBDA_MU_FSG_RUBBERED,
)
from vd_correlation_lab.control.sensors_ekf.sensor_realism_profile import (
    SensorMode,
    SensorRealismProfile,
)
from vd_correlation_lab.control.stochastic_tube.chance_constraints import (
    ChanceConstraintConfig,
)
from vd_correlation_lab.control.stochastic_tube.covariance_propagator import (
    ProcessNoiseConfig,
)
from vd_correlation_lab.control.stochastic_tube.tube_mpc_controller import (
    StochasticTubeMPCController,
    TubeMPCConfig,
)
from vd_correlation_lab.models.differentiable_casadi import (
    DIST_IDX,
    OUTPUT_IDX,
    PARAM_IDX,
    STATE_IDX,
    DifferentiableVehicle14Dof,
    load_compiled_14dof_library,
)

# Physical constants for TeR27 4WD
TRACK_M: float = 1.230
WHEELBASE_M: float = 1.540
FRONT_WEIGHT_DIST: float = 0.45
L_F_M: float = WHEELBASE_M * (1.0 - FRONT_WEIGHT_DIST)  # 0.847 m
L_R_M: float = WHEELBASE_M * FRONT_WEIGHT_DIST          # 0.693 m
LAMBDA_MU_WET_PATCH: float = 0.40


def _wrap_angle(ang: float) -> float:
    return (ang + math.pi) % (2.0 * math.pi) - math.pi


def _build_polyline_path(
    s_grid: np.ndarray,
    kappa_fn: Callable[[float], float],
) -> Dict[str, np.ndarray]:
    """Integrate curvature kappa(s) from (0, 0, psi=0) into a dense centerline."""
    n = len(s_grid)
    x_arr = np.zeros(n, dtype=np.float64)
    y_arr = np.zeros(n, dtype=np.float64)
    psi_arr = np.zeros(n, dtype=np.float64)
    kap_arr = np.zeros(n, dtype=np.float64)
    kap_arr[0] = float(kappa_fn(float(s_grid[0])))
    for i in range(1, n):
        ds = float(s_grid[i] - s_grid[i - 1])
        s_mid = 0.5 * float(s_grid[i] + s_grid[i - 1])
        k_mid = float(kappa_fn(s_mid))
        psi_mid = psi_arr[i - 1] + 0.5 * k_mid * ds
        x_arr[i] = x_arr[i - 1] + ds * math.cos(psi_mid)
        y_arr[i] = y_arr[i - 1] + ds * math.sin(psi_mid)
        psi_arr[i] = _wrap_angle(psi_arr[i - 1] + k_mid * ds)
        kap_arr[i] = float(kappa_fn(float(s_grid[i])))
    return {"s": s_grid, "x": x_arr, "y": y_arr, "psi": psi_arr, "kappa": kap_arr}


def _project_onto_path(
    x_pos: float,
    y_pos: float,
    psi: float,
    path: Dict[str, np.ndarray],
    idx_hint: int,
) -> Tuple[float, float, float, float, float, int]:
    """
    Project (x_pos, y_pos, psi) onto dense centerline near idx_hint.
    Returns (e_y, e_psi, psi_ref, kappa_ref, s_now, new_idx).
    """
    px = path["x"]
    py = path["y"]
    n = len(px)
    lo = max(0, idx_hint - 40)
    hi = min(n, idx_hint + 180)
    dx = x_pos - px[lo:hi]
    dy = y_pos - py[lo:hi]
    d2 = dx * dx + dy * dy
    best_local = int(np.argmin(d2))
    idx = lo + best_local

    psi_ref = float(path["psi"][idx])
    kap_ref = float(path["kappa"][idx])
    s_now = float(path["s"][idx])
    dx_b = x_pos - float(px[idx])
    dy_b = y_pos - float(py[idx])
    # Left normal = (-sin(psi_ref), cos(psi_ref))
    e_y = -math.sin(psi_ref) * dx_b + math.cos(psi_ref) * dy_b
    e_psi = _wrap_angle(psi - psi_ref)
    return e_y, e_psi, psi_ref, kap_ref, s_now, idx


def _build_scenario_path(scenario_key: str) -> Dict[str, np.ndarray]:
    """Build exact centerline geometry for Scenario A, B1, or B2."""
    s_grid = np.linspace(0.0, 260.0, 2601, dtype=np.float64)  # 0.1 m resolution

    if scenario_key == "scenario_a_split_mu_autocross_corner":
        # Steady Autocross corner R = 10.0 m (CCW, kappa = +1/10.0)
        r_corner = 10.0
        return _build_polyline_path(s_grid, lambda s: 1.0 / r_corner)

    if scenario_key == "scenario_b1_3wd_limp_home_braking_entry":
        # 22 m straight braking zone -> 10 m clothoid turn-in -> R = 11.5 m corner
        r_corner = 11.5

        def kappa_b1(s: float) -> float:
            if s <= 22.0:
                return 0.0
            if s <= 32.0:
                u = (s - 22.0) / 10.0
                return (1.0 / r_corner) * (u * u * (3.0 - 2.0 * u))
            return 1.0 / r_corner

        return _build_polyline_path(s_grid, kappa_b1)

    if scenario_key == "scenario_b2_3wd_limp_home_slalom_transition":
        # High-speed Autocross slalom: peak curvature 0.090 m^-1 (R_min = 11.1 m), period 36 m
        amp_k = 0.090
        period_m = 36.0

        def kappa_b2(s: float) -> float:
            ramp = min(max(s / 6.0, 0.0), 1.0)
            return ramp * amp_k * math.sin(2.0 * math.pi * s / period_m)

        return _build_polyline_path(s_grid, kappa_b2)

    raise ValueError(f"Unknown scenario_key: {scenario_key}")


def _interpolate_time_at_distance(t_arr: List[float], s_arr: List[float], s_ref: float) -> float:
    """Linearly interpolate exact time [s] to reach arc-length s_ref [m]."""
    if not t_arr or not s_arr:
        return 0.0
    if s_ref <= s_arr[0]:
        return float(t_arr[0])
    for i in range(1, len(s_arr)):
        if s_arr[i] >= s_ref:
            ds = max(s_arr[i] - s_arr[i - 1], 1e-9)
            frac = (s_ref - s_arr[i - 1]) / ds
            return float(t_arr[i - 1] + frac * (t_arr[i] - t_arr[i - 1]))
    return float(t_arr[-1])


def run_asymmetric_scenario(
    model: DifferentiableVehicle14Dof,
    lib: Any,
    *,
    scenario_key: str,
    controller_mode: str,
    asym_speed_backoff_gain: float = 0.20,
    seed: int = 20260515,
    record_stride: int = 4,
    track_half_width_m: float = 1.500,
    vehicle_half_width_m: float = 0.720,
) -> Dict[str, Any]:
    """
    Run one 200 Hz closed-loop 14-DOF CasADi asymmetric stress scenario.

    controller_mode in {
        "deterministic_4wd_tv",
        "baseline_tube_mpc",
        "refined_asymmetric_tube_mpc",
    }
    """
    dt = 0.005
    corridor_margin_m = track_half_width_m - vehicle_half_width_m  # 0.780 m
    path = _build_scenario_path(scenario_key)

    # Scenario-specific physical setup
    if scenario_key == "scenario_a_split_mu_autocross_corner":
        duration_s = 7.0
        v_init_ms = 12.0
        v_target_nom_ms = 13.4
        r_init = 10.0
        wz_init = v_init_ms / r_init
        delta_init = math.atan(WHEELBASE_M / r_init)
        fault_time_s = 2.0
        patch_end_s = 5.2
    elif scenario_key == "scenario_b1_3wd_limp_home_braking_entry":
        duration_s = 6.5
        v_init_ms = 19.5
        v_target_nom_ms = 20.5
        wz_init = 0.0
        delta_init = 0.0
        fault_time_s = 0.80
        patch_end_s = None
    elif scenario_key == "scenario_b2_3wd_limp_home_slalom_transition":
        duration_s = 7.0
        v_init_ms = 12.8
        v_target_nom_ms = 14.4
        wz_init = 0.0
        delta_init = 0.0
        fault_time_s = 1.65
        patch_end_s = None
    else:
        raise ValueError(f"Unknown scenario_key: {scenario_key}")

    n_steps = int(round(duration_s / dt))

    # Initialize 14-DOF state
    x = model.default_state(vx=v_init_ms, vy=0.0, wz=wz_init)
    theta_true = model.default_theta().copy()
    theta_nom = model.default_theta().copy()
    w_base = model.default_disturbance(t_tire_c=75.0, t_motor_c=55.0).copy()
    w_nom = w_base.copy()

    # Preserve FSG_RUBBERED = 0.57 on dry rubbered surface
    theta_true[PARAM_IDX["lambda_mux"]] = LAMBDA_MU_FSG_RUBBERED
    theta_true[PARAM_IDX["lambda_muy"]] = LAMBDA_MU_FSG_RUBBERED
    theta_nom[PARAM_IDX["lambda_mux"]] = LAMBDA_MU_FSG_RUBBERED
    theta_nom[PARAM_IDX["lambda_muy"]] = LAMBDA_MU_FSG_RUBBERED

    # Realistic 125 ms CAN accel latency + 20 ms steer/wheel latency + 1.0x IMU noise
    sensor = SensorRealismProfile(
        mode=SensorMode.SENSOR_TRACK_REALISM,
        seed=seed,
        config={
            "sample_rate_hz": 200.0,
            "can_accel_latency_ms": 125.0,
            "can_steer_latency_ms": 20.0,
            "imu_noise_scale": 1.0,
        },
    )

    noise_cfg = ProcessNoiseConfig(
        can_accel_latency_s=0.125,
        can_steer_latency_s=0.020,
        can_jitter_s=0.015,
        imu_noise_scale=1.0,
    )
    chance_cfg = ChanceConstraintConfig(
        eps_track=1e-4,
        eps_friction=1e-4,
        eps_power=1e-4,
        min_cone_clearance_m=0.150,
        track_half_width_m=track_half_width_m,
        vehicle_half_width_m=vehicle_half_width_m,
    )

    stoch_en = controller_mode != "deterministic_4wd_tv"
    asym_en = controller_mode == "refined_asymmetric_tube_mpc"

    ctrl_cfg = TubeMPCConfig(
        stochastic_enabled=stoch_en,
        use_exact_jacobians=True,
        jacobian_stride=4,
        preview_horizon_steps=6,
        asymmetric_qw_enabled=asym_en,
        asym_speed_backoff_gain=asym_speed_backoff_gain,
    )
    ctrl = StochasticTubeMPCController(
        cfg=ctrl_cfg,
        chance_cfg=chance_cfg,
        noise_cfg=noise_cfg,
        compiled_lib=lib,
        lambda_mu_init=LAMBDA_MU_FSG_RUBBERED,
        p_init_mu=1.0e-3,
    )

    u_curr = model.default_input().copy()
    u_curr[0] = delta_init
    u_plant = u_curr.copy()

    # 20 ms (4 ticks) CAN delay ring buffer for inverter torque feedback t_meas_4
    torque_delay_ticks = 4
    torque_buf: deque = deque([np.zeros(4, dtype=np.float64) for _ in range(torque_delay_ticks + 1)])

    _, last_y = lib.step_dynamics(x, u_plant, theta_true, w_base)

    # Telemetry accumulators
    traj: Dict[str, List[float]] = {
        "t_s": [],
        "s_m": [],
        "e_y_m": [],
        "cone_clearance_m": [],
        "b_cone_m": [],
        "b_friction": [],
        "friction_ellipse_scale": [],
        "v_ref_scale": [],
        "vx_ms": [],
        "ay_ms2": [],
        "wz_rads": [],
        "beta_deg": [],
        "lambda_mu_hat": [],
        "sigma_mz_asym_nm": [],
        "p_dc_kw": [],
        "dMz_dbeta": [],
        "T_FL_plant_nm": [],
        "T_FR_plant_nm": [],
        "T_RL_plant_nm": [],
        "T_RR_plant_nm": [],
        "d_mz_ext_nm": [],
    }
    full_t: List[float] = []
    full_s: List[float] = []

    min_clearance_m = 1e9
    max_abs_ey_m = 0.0
    cones_hit = 0
    cone_violation_ticks = 0
    hv_violations = 0
    bspd_violations = 0
    max_p_dc_kw = -1e9
    min_p_dc_kw = 1e9
    max_dmz_dbeta = -1e9
    max_abs_beta_deg = 0.0
    max_abs_wz_rads = 0.0
    max_sigma_mz_asym = 0.0
    max_abs_dmz_ext = 0.0
    vx_sum = 0.0
    ay_abs_sum = 0.0
    b_cone_max = 0.0
    b_fric_max = 0.0
    limp_home_detected_time_s: Optional[float] = None
    path_idx = 0

    for k in range(n_steps):
        t_s = k * dt
        x_pos = float(x[STATE_IDX["X"]])
        y_pos = float(x[STATE_IDX["Y"]])
        psi = float(x[STATE_IDX["psi"]])
        vx_t = float(x[STATE_IDX["vx"]])
        vy_t = float(x[STATE_IDX["vy"]])
        wz_t = float(x[STATE_IDX["wz"]])
        om_4_t = [
            float(x[STATE_IDX["omega_FL"]]),
            float(x[STATE_IDX["omega_FR"]]),
            float(x[STATE_IDX["omega_RL"]]),
            float(x[STATE_IDX["omega_RR"]]),
        ]
        fz_4_t = [
            float(last_y[OUTPUT_IDX["Fz_FL"]]),
            float(last_y[OUTPUT_IDX["Fz_FR"]]),
            float(last_y[OUTPUT_IDX["Fz_RL"]]),
            float(last_y[OUTPUT_IDX["Fz_RR"]]),
        ]
        ax_t = float(last_y[OUTPUT_IDX["ax"]])
        ay_t = float(last_y[OUTPUT_IDX["ay"]])

        e_y, e_psi, psi_ref, kap_ref, s_now, path_idx = _project_onto_path(
            x_pos, y_pos, psi, path, path_idx
        )
        full_t.append(t_s)
        full_s.append(s_now)

        clearance_m = corridor_margin_m - abs(e_y)
        min_clearance_m = min(min_clearance_m, clearance_m)
        max_abs_ey_m = max(max_abs_ey_m, abs(e_y))
        if clearance_m < 0.0:
            cones_hit += 1
        if clearance_m < 0.150:
            cone_violation_ticks += 1

        # Lookahead curvature for braking entry (Scenario B1) so controller anticipates the corner
        if scenario_key == "scenario_b1_3wd_limp_home_braking_entry":
            s_look = min(s_now + max(0.85 * vx_t, 8.0), float(path["s"][-1]))
            idx_look = min(int(round(s_look / 0.1)), len(path["kappa"]) - 1)
            kap_for_speed = max(abs(kap_ref), abs(float(path["kappa"][idx_look])))
        else:
            kap_for_speed = kap_ref

        # Sensor realism measurement (125 ms CAN accel + 20 ms steer/gyro/wheel + IMU noise)
        meas = sensor.step(
            ax=ax_t,
            ay=ay_t,
            wz=wz_t,
            vx=vx_t,
            vy=vy_t,
            delta=float(u_curr[0]),
            fz_4=fz_4_t,
            omega_4=om_4_t,
            dt=dt,
        )

        # Delayed inverter torque feedback (20 ms CAN)
        t_meas_4 = torque_buf.popleft()

        # Nominal forces for OnlineGripEstimator (evaluated at COLD_TEST_OCT = 0.47 reference)
        fx_tot_true = sum(float(last_y[OUTPUT_IDX[f"Fx_{c}"]]) for c in ("FL", "FR", "RL", "RR"))
        fy_tot_true = sum(float(last_y[OUTPUT_IDX[f"Fy_{c}"]]) for c in ("FL", "FR", "RL", "RR"))
        nom_ratio = LAMBDA_MU_COLD_TEST_OCT / LAMBDA_MU_FSG_RUBBERED
        fx_nom_est = fx_tot_true * nom_ratio
        fy_nom_est = fy_tot_true * nom_ratio

        # Keep nominal model synchronized with online grip belief
        theta_nom[PARAM_IDX["lambda_mux"]] = ctrl.grip_est.lambda_mu_hat
        theta_nom[PARAM_IDX["lambda_muy"]] = ctrl.grip_est.lambda_mu_hat

        # For B1 braking entry, compute v_target_eff from lookahead curvature
        v_target_step = v_target_nom_ms
        if scenario_key == "scenario_b1_3wd_limp_home_braking_entry" and kap_for_speed > 1e-4:
            # Anticipatory corner entry speed cap so heavy braking occurs before turn-in
            mu_est = 2.68 * ctrl.grip_est.lambda_mu_hat
            v_look = math.sqrt(max((mu_est * 9.81) / kap_for_speed, 1.0))
            v_target_step = min(v_target_nom_ms, v_look)

        ctrl_out = ctrl.step(
            x_true=x,
            u_prev=u_curr,
            theta_nom=theta_nom,
            w_nom=w_nom,
            meas=meas,
            e_y=e_y,
            e_psi=e_psi,
            psi_ref=psi_ref,
            kappa_ref=kap_ref,
            v_target_nom=v_target_step,
            fx_nom_est_n=fx_nom_est,
            fy_nom_est_n=fy_nom_est,
            dt=dt,
            t_meas_4=t_meas_4,
        )
        u_curr = ctrl_out["u_cmd"]

        if limp_home_detected_time_s is None and any(ctrl_out["motor_fault_mask"]):
            limp_home_detected_time_s = t_s

        # Apply extreme asymmetric plant fault / split-mu disturbance
        u_plant = u_curr.copy()
        w_plant = w_base.copy()

        if scenario_key == "scenario_a_split_mu_autocross_corner":
            in_patch = (t_s >= fault_time_s) and (patch_end_s is not None and t_s <= patch_end_s)
            if in_patch:
                # Outer wheels (FR, RR) remain on FSG_RUBBERED (0.57);
                # Inner wheels (FL, RL) hit dirty/wet patch (0.40)
                frac_loss = (LAMBDA_MU_WET_PATCH - LAMBDA_MU_FSG_RUBBERED) / LAMBDA_MU_FSG_RUBBERED
                fx_fl = float(last_y[OUTPUT_IDX["Fx_FL"]])
                fx_rl = float(last_y[OUTPUT_IDX["Fx_RL"]])
                fy_fl = float(last_y[OUTPUT_IDX["Fy_FL"]])
                fy_rl = float(last_y[OUTPUT_IDX["Fy_RL"]])
                dfx_in = frac_loss * (fx_fl + fx_rl)
                dfy_in = frac_loss * (fy_fl + fy_rl)
                # Exact rigid-body yaw moment from inner-wheel (left, y=+track/2) force deficit
                dmz_x = -0.5 * TRACK_M * dfx_in
                dmz_y = L_F_M * (frac_loss * fy_fl) - L_R_M * (frac_loss * fy_rl)
                # Additional asymmetric scrub/breakaway destabilizing yaw pulse on patch entry
                dmz_scrub = 185.0 * math.sin(min((t_s - fault_time_s) * 2.5, math.pi))
                dmz_tot = dmz_x + dmz_y + dmz_scrub

                w_plant[DIST_IDX["d_fx_ext"]] = dfx_in
                w_plant[DIST_IDX["d_fy_ext"]] = dfy_in
                w_plant[DIST_IDX["d_mz_ext"]] = dmz_tot
                # Inner wheels also lose torque traction ceiling on the 0.40 patch
                grip_ratio = LAMBDA_MU_WET_PATCH / LAMBDA_MU_FSG_RUBBERED
                u_plant[1] *= grip_ratio
                u_plant[3] *= grip_ratio
        else:
            # Scenario B1 & B2: Instantaneous FL motor loss -> 0 Nm
            if t_s >= fault_time_s:
                u_plant[1] = 0.0

        # Push actual plant wheel motor torques into 20 ms CAN feedback buffer
        torque_buf.append(u_plant[1:5].copy())

        x_next, y_next = lib.step_dynamics(x, u_plant, theta_true, w_plant)
        x = x_next
        last_y = y_next

        # Evaluate actual plant HV DC power (with motor fault if active)
        p_plant_kw = float(ctrl.loss_model.evaluate_4(u_plant[1:5], np.asarray(om_4_t, dtype=np.float64))["p_dc_kw"])
        p_cmd_kw = float(ctrl_out["p_dc_kw"])
        p_check_kw = max(p_cmd_kw, p_plant_kw) if p_cmd_kw >= 0 else min(p_cmd_kw, p_plant_kw)
        max_p_dc_kw = max(max_p_dc_kw, p_cmd_kw, p_plant_kw)
        min_p_dc_kw = min(min_p_dc_kw, p_cmd_kw, p_plant_kw)
        if ctrl_out["hv_violation"] or p_cmd_kw > 80.001 or p_cmd_kw < -25.001 or p_plant_kw > 80.001 or p_plant_kw < -25.001:
            hv_violations += 1
        if ctrl_out["bspd_violation"]:
            bspd_violations += 1

        dmz_db = float(ctrl_out["dMz_dbeta"])
        max_dmz_dbeta = max(max_dmz_dbeta, dmz_db)
        beta_deg = math.degrees(float(last_y[OUTPUT_IDX["beta"]]))
        max_abs_beta_deg = max(max_abs_beta_deg, abs(beta_deg))
        max_abs_wz_rads = max(max_abs_wz_rads, abs(wz_t))
        max_sigma_mz_asym = max(max_sigma_mz_asym, float(ctrl_out["sigma_mz_asym_nm"]))
        max_abs_dmz_ext = max(max_abs_dmz_ext, abs(float(w_plant[DIST_IDX["d_mz_ext"]])))

        bo = ctrl_out["backoff"]
        vx_sum += vx_t
        ay_abs_sum += abs(ay_t)
        b_cone_max = max(b_cone_max, bo.b_cone_m)
        b_fric_max = max(b_fric_max, bo.b_friction)

        if k % record_stride == 0:
            traj["t_s"].append(round(t_s, 4))
            traj["s_m"].append(round(s_now, 4))
            traj["e_y_m"].append(round(e_y, 5))
            traj["cone_clearance_m"].append(round(clearance_m, 5))
            traj["b_cone_m"].append(round(bo.b_cone_m, 5))
            traj["b_friction"].append(round(bo.b_friction, 5))
            traj["friction_ellipse_scale"].append(round(bo.friction_ellipse_scale, 5))
            traj["v_ref_scale"].append(round(bo.v_ref_scale, 5))
            traj["vx_ms"].append(round(vx_t, 4))
            traj["ay_ms2"].append(round(ay_t, 4))
            traj["wz_rads"].append(round(wz_t, 4))
            traj["beta_deg"].append(round(beta_deg, 4))
            traj["lambda_mu_hat"].append(round(float(ctrl_out["lambda_mu_hat"]), 5))
            traj["sigma_mz_asym_nm"].append(round(float(ctrl_out["sigma_mz_asym_nm"]), 3))
            traj["p_dc_kw"].append(round(p_check_kw, 4))
            traj["dMz_dbeta"].append(round(dmz_db, 2))
            traj["T_FL_plant_nm"].append(round(float(u_plant[1]), 3))
            traj["T_FR_plant_nm"].append(round(float(u_plant[2]), 3))
            traj["T_RL_plant_nm"].append(round(float(u_plant[3]), 3))
            traj["T_RR_plant_nm"].append(round(float(u_plant[4]), 3))
            traj["d_mz_ext_nm"].append(round(float(w_plant[DIST_IDX["d_mz_ext"]]), 3))

    summary = {
        "scenario": scenario_key,
        "controller": controller_mode,
        "duration_s": duration_s,
        "fault_time_s": fault_time_s,
        "distance_covered_m": round(float(full_s[-1]), 4),
        "cones_hit": int(cones_hit),
        "cone_clearance_lt_150mm_ticks": int(cone_violation_ticks),
        "min_cone_clearance_m": round(float(min_clearance_m), 4),
        "max_abs_ey_m": round(float(max_abs_ey_m), 4),
        "mean_vx_ms": round(vx_sum / max(n_steps, 1), 4),
        "mean_abs_ay_ms2": round(ay_abs_sum / max(n_steps, 1), 4),
        "max_abs_beta_deg": round(float(max_abs_beta_deg), 4),
        "max_abs_wz_rads": round(float(max_abs_wz_rads), 4),
        "max_sigma_mz_asym_nm": round(float(max_sigma_mz_asym), 3),
        "max_abs_d_mz_ext_nm": round(float(max_abs_dmz_ext), 3),
        "max_b_cone_m": round(float(b_cone_max), 4),
        "max_b_friction": round(float(b_fric_max), 4),
        "hv_violations": int(hv_violations),
        "bspd_violations": int(bspd_violations),
        "max_p_dc_kw": round(float(max_p_dc_kw), 4),
        "min_p_dc_kw": round(float(min_p_dc_kw), 4),
        "max_dMz_dbeta": round(float(max_dmz_dbeta), 2),
        "limp_home_detected_time_s": (
            round(limp_home_detected_time_s, 4) if limp_home_detected_time_s is not None else None
        ),
        "directional_stability_pass": bool(max_dmz_dbeta < 0.0),
        "cone_safety_pass": bool(cones_hit == 0 and min_clearance_m >= 0.150),
        "hv_bspd_pass": bool(hv_violations == 0 and bspd_violations == 0),
        "all_gates_pass": bool(
            cones_hit == 0
            and min_clearance_m >= 0.150
            and hv_violations == 0
            and bspd_violations == 0
            and max_dmz_dbeta < 0.0
        ),
    }
    return {
        "summary": summary,
        "trajectory": traj,
        "_full_t": full_t,
        "_full_s": full_s,
    }


def run_asymmetric_stress_suite(
    out_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run all 3 asymmetric stress scenarios across 3 controllers + time-penalty sweep."""
    t0 = time.perf_counter()
    print("[ASYM-TUBE] Loading DifferentiableVehicle14Dof (ter27_4wd) & libter_diff_14dof.so...", flush=True)
    model = DifferentiableVehicle14Dof(
        vehicle_id="ter27_4wd",
        suspension_mode="3d_double_wishbone_lut",
        use_pinn=True,
        grip_profile="FSG_RUBBERED",
        dt=0.005,
        rk4_substeps=2,
    )
    lib = load_compiled_14dof_library()

    scenario_specs = [
        (
            "scenario_a_split_mu_autocross_corner",
            "Scenario A: Split-mu Patched Asphalt (Outer 0.57 / Inner 0.40) Mid-Autocross Corner",
        ),
        (
            "scenario_b1_3wd_limp_home_braking_entry",
            "Scenario B1: Sudden FL Motor Loss (3WD Limp-Home) Mid-Heavy Braking & Corner Entry",
        ),
        (
            "scenario_b2_3wd_limp_home_slalom_transition",
            "Scenario B2: Sudden FL Motor Loss (3WD Limp-Home) Mid-Slalom Lateral Transition",
        ),
    ]
    controllers = (
        "deterministic_4wd_tv",
        "baseline_tube_mpc",
        "refined_asymmetric_tube_mpc",
    )

    scenarios_out: Dict[str, Any] = {}
    all_refined_pass = True

    for sc_key, sc_title in scenario_specs:
        print(f"\n[ASYM-TUBE] === {sc_title} ===", flush=True)
        runs: Dict[str, Dict[str, Any]] = {}
        for ctrl_mode in controllers:
            res = run_asymmetric_scenario(
                model,
                lib,
                scenario_key=sc_key,
                controller_mode=ctrl_mode,
                asym_speed_backoff_gain=0.12,
            )
            runs[ctrl_mode] = res
            sm = res["summary"]
            print(
                f"  {ctrl_mode:28s}: cones={sm['cones_hit']:3d} "
                f"min_clr={sm['min_cone_clearance_m']:+.3f}m "
                f"mean_vx={sm['mean_vx_ms']:.2f}m/s "
                f"max_sigMz={sm['max_sigma_mz_asym_nm']:5.1f}Nm "
                f"hv={sm['hv_violations']} bspd={sm['bspd_violations']} "
                f"dMz/db={sm['max_dMz_dbeta']:.1f} pass={sm['all_gates_pass']}",
                flush=True,
            )

        # Common sector reference distance (95% of minimum distance covered across controllers)
        s_ref = round(0.95 * min(runs[c]["summary"]["distance_covered_m"] for c in controllers), 2)
        # Full-maneuver sector distance across non-crashed tube controllers (Baseline vs Refined)
        s_ref_tube = round(
            0.95
            * min(
                runs["baseline_tube_mpc"]["summary"]["distance_covered_m"],
                runs["refined_asymmetric_tube_mpc"]["summary"]["distance_covered_m"],
            ),
            2,
        )
        sector_times: Dict[str, float] = {}
        for c in controllers:
            t_sec = _interpolate_time_at_distance(runs[c]["_full_t"], runs[c]["_full_s"], s_ref)
            sector_times[c] = round(t_sec, 4)
            runs[c]["summary"]["sector_ref_m"] = s_ref
            runs[c]["summary"]["sector_time_s"] = round(t_sec, 4)

        t_base_full = round(
            _interpolate_time_at_distance(
                runs["baseline_tube_mpc"]["_full_t"],
                runs["baseline_tube_mpc"]["_full_s"],
                s_ref_tube,
            ),
            4,
        )
        t_ref_full = round(
            _interpolate_time_at_distance(
                runs["refined_asymmetric_tube_mpc"]["_full_t"],
                runs["refined_asymmetric_tube_mpc"]["_full_s"],
                s_ref_tube,
            ),
            4,
        )

        # Quantify exact minimum sector-time penalty sweep for refined_asymmetric_tube_mpc
        sweep_gains = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
        sweep_rows: List[Dict[str, Any]] = []
        min_safe_row: Optional[Dict[str, Any]] = None
        for g in sweep_gains:
            sw = run_asymmetric_scenario(
                model,
                lib,
                scenario_key=sc_key,
                controller_mode="refined_asymmetric_tube_mpc",
                asym_speed_backoff_gain=g,
                record_stride=20,
            )
            sw_s = sw["summary"]
            t_sw = _interpolate_time_at_distance(sw["_full_t"], sw["_full_s"], s_ref)
            dt_vs_det = t_sw - sector_times["deterministic_4wd_tv"]
            dt_vs_base = t_sw - sector_times["baseline_tube_mpc"]
            row = {
                "asym_speed_backoff_gain": g,
                "min_cone_clearance_m": sw_s["min_cone_clearance_m"],
                "cones_hit": sw_s["cones_hit"],
                "all_gates_pass": sw_s["all_gates_pass"],
                "sector_time_s": round(t_sw, 4),
                "delta_time_vs_det_s": round(dt_vs_det, 4),
                "delta_time_vs_baseline_tube_s": round(dt_vs_base, 4),
                "pct_penalty_vs_det": round(100.0 * dt_vs_det / max(sector_times["deterministic_4wd_tv"], 1e-6), 3),
            }
            sweep_rows.append(row)
            if sw_s["all_gates_pass"] and (
                min_safe_row is None or row["sector_time_s"] < min_safe_row["sector_time_s"]
            ):
                min_safe_row = row

        # Strip internal full arrays before JSON serialization
        serialized_runs = {
            c: {"summary": runs[c]["summary"], "trajectory": runs[c]["trajectory"]}
            for c in controllers
        }
        ref_pass = bool(runs["refined_asymmetric_tube_mpc"]["summary"]["all_gates_pass"])
        all_refined_pass = all_refined_pass and ref_pass

        t_det = sector_times["deterministic_4wd_tv"]
        t_base = sector_times["baseline_tube_mpc"]
        t_ref = sector_times["refined_asymmetric_tube_mpc"]
        scenarios_out[sc_key] = {
            "title": sc_title,
            "sector_ref_m": s_ref,
            "sector_ref_tube_full_m": s_ref_tube,
            "controllers": serialized_runs,
            "comparison": {
                "sector_time_det_s": t_det,
                "sector_time_baseline_tube_s": t_base,
                "sector_time_refined_tube_s": t_ref,
                "delta_time_refined_vs_det_s": round(t_ref - t_det, 4),
                "delta_time_refined_vs_baseline_s": round(t_ref - t_base, 4),
                "pct_penalty_refined_vs_det": round(100.0 * (t_ref - t_det) / max(t_det, 1e-6), 3),
                "sector_time_baseline_tube_full_s": t_base_full,
                "sector_time_refined_tube_full_s": t_ref_full,
                "delta_time_refined_vs_baseline_full_s": round(t_ref_full - t_base_full, 4),
                "pct_penalty_refined_vs_baseline_full": round(
                    100.0 * (t_ref_full - t_base_full) / max(t_base_full, 1e-6), 3
                ),
                "clearance_gain_refined_vs_det_m": round(
                    runs["refined_asymmetric_tube_mpc"]["summary"]["min_cone_clearance_m"]
                    - runs["deterministic_4wd_tv"]["summary"]["min_cone_clearance_m"],
                    4,
                ),
                "clearance_gain_refined_vs_baseline_m": round(
                    runs["refined_asymmetric_tube_mpc"]["summary"]["min_cone_clearance_m"]
                    - runs["baseline_tube_mpc"]["summary"]["min_cone_clearance_m"],
                    4,
                ),
                "min_safe_backoff_sweep_result": min_safe_row,
            },
            "time_penalty_sweep": sweep_rows,
        }

    elapsed_s = time.perf_counter() - t0
    report = {
        "schema_version": "tube_mpc_asymmetric_stress_v1",
        "vehicle_id": "ter27_4wd",
        "model_backend": "differentiable_casadi_14dof (libter_diff_14dof.so)",
        "sensor_profile": {
            "mode": "SENSOR_TRACK_REALISM",
            "can_accel_latency_ms": 125.0,
            "can_steer_latency_ms": 20.0,
            "can_jitter_ms": 15.0,
            "imu_noise_scale": 1.0,
        },
        "grip_profiles_preserved": {
            "FSG_RUBBERED": LAMBDA_MU_FSG_RUBBERED,
            "COLD_TEST_OCT": LAMBDA_MU_COLD_TEST_OCT,
            "WET_PATCH_INNER": LAMBDA_MU_WET_PATCH,
        },
        "elapsed_eval_s": round(elapsed_s, 3),
        "acceptance_gates": {
            "all_scenarios_refined_zero_cones_and_min_clear_ge_150mm": bool(
                all(
                    s["controllers"]["refined_asymmetric_tube_mpc"]["summary"]["cone_safety_pass"]
                    for s in scenarios_out.values()
                )
            ),
            "all_scenarios_refined_zero_hv_and_bspd_violations": bool(
                all(
                    s["controllers"]["refined_asymmetric_tube_mpc"]["summary"]["hv_bspd_pass"]
                    for s in scenarios_out.values()
                )
            ),
            "all_scenarios_refined_strict_dmz_dbeta_lt_0": bool(
                all(
                    s["controllers"]["refined_asymmetric_tube_mpc"]["summary"]["directional_stability_pass"]
                    for s in scenarios_out.values()
                )
            ),
            "overall_pass": bool(all_refined_pass),
        },
        "scenarios": scenarios_out,
    }

    target_path = (
        out_path
        if out_path is not None
        else (_REPO_ROOT / "outputs" / "differentiable_deep_validation" / "tube_mpc_asymmetric_stress_report.json")
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")
    print(
        f"\n[ASYM-TUBE] Report written to {target_path} "
        f"(overall_pass={report['acceptance_gates']['overall_pass']}, elapsed={elapsed_s:.2f}s)",
        flush=True,
    )
    return report


if __name__ == "__main__":
    run_asymmetric_stress_suite()
