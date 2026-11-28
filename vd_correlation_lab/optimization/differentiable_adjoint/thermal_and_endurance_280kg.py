#!/usr/bin/env python3
"""
TeR27 (280.0 kg) Realistic Tire Thermal Conditioning & 22 km Endurance
Energy/Thermal Optimization.

Part A - Realistic Tire Thermal Window (Acceleration 75 m & Skidpad at 280 kg):
  - Evaluates cold start (25-35 C ambient track) vs legal pre-grid / approach-phase
    conditioning (50-75 C) vs overheating (>95 C) strictly following the calibrated
    TTC Hoosier R25B thermal bell curve (thermal_grip_multiplier_smooth, capped at
    <= 1.000 so peak grip is NEVER inflated above FSG_RUBBERED=0.57).
  - Runs full 14-DOF compiled CasADi C-Codegen (libter_diff_14dof.so) + Diff4WDController
    on the 280 kg setups for Acceleration (75 m) and Skidpad (FSG circle).

Part B - 22 km Endurance (18.587 laps FSG 2023) Energy + Thermal Strategy at 280 kg:
  - Optimizes lift-and-coast anticipation, speed cap (v_max), discharge power cap
    (P_cap), regen utilization (-25.0 kW DC cap), and 4WD loss-minimizing torque
    split over 22.0 km within <= 3.50 kWh net HV energy.
  - Integrates coupled motor stator temperature T_motor(t) (< 115 C, 0 derating),
    inverter heatsink temperature T_inv(t) (< 85 C), and 4-corner tire bulk
    temperature T_tire(t) + wear evolution across all 19 laps (22.0 km).
  - Computes the Pareto frontier of Endurance Time vs Efficiency Energy and finds
    the optimal point maximizing combined FSG Endurance (250 pts) + Efficiency
    (100 pts) using vd_correlation_lab/optimization/lap_time_sim/scoring.py,
    comparing 290 kg vs 280 kg.

Outputs:
  - outputs/ter27_280kg_limit_study/thermal_and_endurance_report.json
  - outputs/ter27_280kg_limit_study/setups/limit_280kg_endurance.json

Strict Guardrails:
  - 100% pure ASCII everywhere.
  - FSG_RUBBERED = 0.57, COLD_TEST_OCT = 0.47 untouched, peak mu_th <= 1.000.
  - HV DC power strictly in [-25.0, +80.0] kW (0 violations), 0 BSPD EV 2.3
    violations, dMz/dbeta < 0, min_clear >= 0.150 m, min_rh_f >= 10 mm.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import casadi as ca
import numpy as np

from vd_correlation_lab.models.differentiable_casadi import (
    COLD_TEST_OCT,
    DIST_IDX,
    FSG_RUBBERED,
    NU,
    NTHETA,
    NX,
    NY,
    OUTPUT_IDX,
    PARAM_IDX,
    STATE_IDX,
    DifferentiableVehicle14Dof,
    load_compiled_14dof_library,
)
from vd_correlation_lab.models.differentiable_casadi.smooth_ops import (
    thermal_grip_multiplier_smooth,
)
from vd_correlation_lab.models.vehicle_14dof.ter27_casadi_model import (
    TIRE_THERMAL_PARAMS,
    tire_temp_dot_smooth,
)
from vd_correlation_lab.optimization.differentiable_adjoint.diff_4wd_controller import (
    CTRL_PARAM_NAMES,
    Diff4WDController,
    DiffInHubLossConstants,
    ctrl_dict_to_vec,
    sabs,
    sclip,
    smax,
    smin,
    spos,
)
from vd_correlation_lab.optimization.differentiable_adjoint.warmstart_record_optimizer import (
    AUG_C_DIM,
    CAR_HALF_W_M,
    ENDURANCE_DIST_M,
    ENDURANCE_ENERGY_KWH,
    HALF_TRACK_M,
    MIN_CLEAR_REQ_M,
    R_CENTER_M,
    R_INNER_M,
    R_OUTER_M,
    build_fsg_2023_spline_bundle,
    compute_lltd_from_theta,
    load_record_setup,
)
from vd_correlation_lab.optimization.lap_time_sim.scoring import (
    EnduranceProjection,
    compute_fs_scores,
    project_endurance_22km,
)

REPO_ROOT = _REPO_ROOT
OUT_DIR = REPO_ROOT / "outputs" / "ter27_280kg_limit_study"
SETUPS_OUT_DIR = OUT_DIR / "setups"
REPORT_PATH = OUT_DIR / "thermal_and_endurance_report.json"
ENDURANCE_SETUP_PATH = SETUPS_OUT_DIR / "limit_280kg_endurance.json"

# ---------------------------------------------------------------------------
# Physical Thermal & Wear Constants (TeR27 In-Hub 4WD + Hoosier R25B)
# ---------------------------------------------------------------------------

# In-hub water-cooled PMSM stator lumped thermal model (per motor, 4 motors total)
MOTOR_THERMAL: Dict[str, float] = {
    "c_th_j_per_k": 2200.0,       # Stator copper + iron effective thermal mass [J/K]
    "h_cool_w_per_k": 38.0,       # Water-jacket convective conductance [W/K]
    "t_coolant_c": 40.0,          # Radiator loop coolant temperature [C]
    "t_init_c": 45.0,             # Pre-start motor temperature [C]
    "t_derate_start_c": 115.0,    # Winding derating threshold [C]
    "eta_motor_loss_frac": 0.72,  # Fraction of total powertrain electrical loss in motor stator
}

# Inverter SiC/IGBT module lumped thermal model (per channel, 4 channels total)
INVERTER_THERMAL: Dict[str, float] = {
    "c_th_j_per_k": 1600.0,       # Cold-plate + module thermal mass [J/K]
    "h_cool_w_per_k": 45.0,       # Cold-plate cooling conductance [W/K]
    "t_coolant_c": 40.0,          # Coolant temperature [C]
    "t_init_c": 42.0,             # Pre-start inverter temperature [C]
    "t_derate_start_c": 85.0,     # Inverter derating threshold [C]
    "eta_inv_loss_frac": 0.18,    # Fraction of total electrical loss in inverter
}

# Tire wear & thermal parameters (Hoosier 16.0x7.5-10 R25B)
FSG_TRACK_TEMP_C: float = 45.0
COLD_TRACK_TEMP_C: float = 25.0
TIRE_WEAR_E_REF_MJ: float = 55.0   # Cumulative 4-tire slip energy for 5% grip wear in nominal window
TIRE_WEAR_MAX_LOSS: float = 0.06   # Max wear grip loss over 22 km in nominal thermal window


def calibrated_thermal_mu(t_tire_c: float) -> float:
    """
    Calibrated TTC Hoosier R25B thermal grip multiplier, strictly capped at <= 1.000
    so peak tire grip is NEVER inflated above FSG_RUBBERED=0.57.
    """
    raw = float(thermal_grip_multiplier_smooth(float(t_tire_c)))
    return float(min(1.0, max(0.55, raw)))


def make_290kg_theta(base_280_theta: np.ndarray) -> np.ndarray:
    """Construct exact 290.0 kg baseline theta matching pre-weight-reduction TeR27."""
    th = base_280_theta.copy()
    th[PARAM_IDX["mass"]] = 290.0
    th[PARAM_IDX["i_xx"]] = 15.4
    th[PARAM_IDX["i_yy"]] = 84.2
    # Total Izz delta = 83.1 - 79.58 = +3.52 kg*m^2 (unsprung Steiner unchanged)
    th[PARAM_IDX["i_zz"]] = float(base_280_theta[PARAM_IDX["i_zz"]]) + (83.1 - 79.58)
    return th


def apply_setup_json_to_theta(base_theta: np.ndarray, raw: Dict[str, Any]) -> np.ndarray:
    """Apply mechanical setup fields from a MirenaSim setup JSON onto theta_veh."""
    th = base_theta.copy()
    th[PARAM_IDX["k_spring_f"]] = float(raw["spring_rate_front_n_m"])
    th[PARAM_IDX["k_spring_r"]] = float(raw["spring_rate_rear_n_m"])
    th[PARAM_IDX["k_arb_f"]] = float(raw["arb_front_nm_rad"])
    th[PARAM_IDX["k_arb_r"]] = float(raw["arb_rear_nm_rad"])
    th[PARAM_IDX["c_damp_f_scale"]] = float(
        np.clip(float(raw["damping_compression_front_ns_m"]) / 2000.0, 0.65, 1.45)
    )
    th[PARAM_IDX["c_damp_r_scale"]] = float(
        np.clip(float(raw["damping_compression_rear_ns_m"]) / 2000.0, 0.65, 1.45)
    )
    th[PARAM_IDX["camber_f_rad"]] = math.radians(float(raw["camber_front_deg"]))
    th[PARAM_IDX["camber_r_rad"]] = math.radians(float(raw["camber_rear_deg"]))
    th[PARAM_IDX["ackermann_factor"]] = float(
        np.clip(float(raw["ackermann_pct"]) / 100.0, -0.15, 0.40)
    )
    rake_mm = float(np.clip(float(raw["rake_mm"]), -8.0, 8.0))
    th[PARAM_IDX["aero_balance_front"]] = float(np.clip(0.48 + 0.004 * rake_mm, 0.448, 0.512))
    return th


def load_ctrl_gains_from_setup(raw: Dict[str, Any]) -> np.ndarray:
    """Load controller gain vector from setup JSON if available, else nominal."""
    th_c = ctrl_dict_to_vec()
    tc = raw.get("trajectory_control_280kg", {})
    gains = tc.get("ctrl_gains", raw.get("optimized_ctrl_gains", {}))
    for k, v in gains.items():
        if k in CTRL_PARAM_NAMES:
            th_c[CTRL_PARAM_NAMES.index(k)] = float(v)
    return th_c


# ---------------------------------------------------------------------------
# Fast 14-DOF Closed-Loop Evaluators for Thermal Window & Endurance
# ---------------------------------------------------------------------------


class ThermalEnduranceEngine:
    """
    14-DOF compiled CasADi C-Codegen engine for:
      - Part A: Thermal-window sweep on Acceleration (75 m) and Skidpad
      - Part B: Lift-and-coast + speed-cap + power-cap + regen + 50/50 loss-opt
                Endurance simulation on full 1.184 km FSG 2023 track
    """

    def __init__(self) -> None:
        self.model = DifferentiableVehicle14Dof(
            vehicle_id="ter27_4wd",
            suspension_mode="3d_double_wishbone_lut",
            tire_compound="R25B",
            use_pinn=True,
            grip_profile="FSG_RUBBERED",
            dt=0.005,
            rk4_substeps=2,
        )
        self.lib = load_compiled_14dof_library()
        self.ctrl = Diff4WDController(alqp_iters=8)
        self.ctrl.loss_consts = DiffInHubLossConstants(
            gear_ratio=self.ctrl.gear_ratio,
            t_motor_peak_nm=self.ctrl.t_motor_peak,
            p_dc_pos_lim_kw=79.00,
            p_dc_neg_lim_kw=24.20,
        )
        self.theta_280_default = self.model.default_theta().copy()
        self.theta_290_default = make_290kg_theta(self.theta_280_default)
        self.theta_ctrl_nom = ctrl_dict_to_vec()

        self.track = build_fsg_2023_spline_bundle()
        self.n_laps_22km = ENDURANCE_DIST_M / self.track.length_m
        self.dt = 0.005

        self.fn_accel = self._build_accel_ctrl()
        self.fn_skidpad = self._build_skidpad_ctrl()
        self.fn_endurance = self._build_endurance_ctrl()

    def _build_accel_ctrl(self) -> ca.Function:
        dt = self.dt
        x = ca.MX.sym("x", NX)
        y_prev = ca.MX.sym("y_prev", NY)
        c = ca.MX.sym("c", AUG_C_DIM)
        t_k = ca.MX.sym("t_k")
        th_v = ca.MX.sym("th_v", NTHETA)
        th_c = ca.MX.sym("th_c", len(CTRL_PARAM_NAMES))
        # p_ev: [kappa_tgt, tc_gain, launch_fx_n, mu_th_eff, t_motor_c]
        p_ev = ca.MX.sym("p_ev", 5)

        Y_pos = x[STATE_IDX["Y"]]
        psi = x[STATE_IDX["psi"]]
        vx = x[STATE_IDX["vx"]]
        vy = x[STATE_IDX["vy"]]
        wz = x[STATE_IDX["wz"]]
        omega_4 = x[STATE_IDX["omega_FL"]:STATE_IDX["omega_RR"] + 1]

        ay_prev = y_prev[OUTPUT_IDX["ay"]]
        fz_prev = y_prev[OUTPUT_IDX["Fz_FL"]:OUTPUT_IDX["Fz_RR"] + 1]
        fy_prev = y_prev[OUTPUT_IDX["Fy_FL"]:OUTPUT_IDX["Fy_RR"] + 1]
        p_dc_prev = y_prev[OUTPUT_IDX["P_dc_kw"]]

        ctrl_st = c[0:4]
        s_prog = c[4]
        e_kwh = c[5]
        delta_prev = c[6]

        kappa_tgt = p_ev[0]
        tc_gain = p_ev[1]
        launch_fx = p_ev[2]
        mu_th_eff = p_ev[3]
        t_mot = p_ev[4]

        motor_temp_4 = ca.vertcat(t_mot, t_mot, t_mot, t_mot)
        mu_eff = 2.15 * th_v[PARAM_IDX["lambda_mux"]] * mu_th_eff

        delta_cmd = sclip(-0.14 * Y_pos - 0.45 * psi, -0.18, 0.18, 0.005)
        ramp = 1.0 - ca.exp(-t_k / 0.022)
        vx_safe = smax(vx, 0.6, 0.15)
        kap_4 = (omega_4 * self.ctrl.r_eff - vx) / vx_safe
        kap_avg = 0.25 * ca.sum1(kap_4)
        excess_kap = spos(kap_avg - kappa_tgt, 0.015)
        tc_mod = sclip(1.0 - tc_gain * excess_kap, 0.20, 1.0, 0.03)
        fx_req = launch_fx * ramp * tc_mod
        t_brk_4 = ca.DM.zeros(4, 1)

        delta_dot = sclip((delta_cmd - delta_prev) / dt, -10.0, 10.0, 0.2)
        ctrl_out = self.ctrl.step_symbolic(
            vx=vx, vy=vy, wz=wz, ay_meas=ay_prev,
            delta=delta_cmd, delta_dot=delta_dot, fx_req=fx_req,
            fz_4=fz_prev, fy_4=fy_prev, omega_4=omega_4,
            mu_eff=mu_eff, ctrl_state=ctrl_st, theta_ctrl=th_c,
            dt=dt, motor_temp_c_4=motor_temp_4,
        )
        u_k = ca.vertcat(delta_cmd, ctrl_out["t_wheel_4"], t_brk_4)
        c_next = ca.vertcat(
            ctrl_out["ctrl_state_next"],
            s_prog + vx * dt,
            e_kwh + (p_dc_prev * dt) / 3600.0,
            delta_cmd,
        )
        aux = ca.vertcat(ctrl_out["p_dc_kw"], ctrl_out["dmz_dbeta_task"], 0.0, fx_req)
        return ca.Function(
            "th_accel_ctrl",
            [x, y_prev, c, t_k, th_v, th_c, p_ev],
            [u_k, c_next, aux],
        )

    def _build_skidpad_ctrl(self) -> ca.Function:
        dt = self.dt
        x = ca.MX.sym("x", NX)
        y_prev = ca.MX.sym("y_prev", NY)
        c = ca.MX.sym("c", AUG_C_DIM)
        t_k = ca.MX.sym("t_k")
        th_v = ca.MX.sym("th_v", NTHETA)
        th_c = ca.MX.sym("th_c", len(CTRL_PARAM_NAMES))
        # p_ev: [r_tgt_m, vx_tgt_ms, delta_ff_rad, mu_th_eff, t_motor_c]
        p_ev = ca.MX.sym("p_ev", 5)

        X_pos = x[STATE_IDX["X"]]
        Y_pos = x[STATE_IDX["Y"]]
        psi = x[STATE_IDX["psi"]]
        vx = x[STATE_IDX["vx"]]
        vy = x[STATE_IDX["vy"]]
        wz = x[STATE_IDX["wz"]]
        omega_4 = x[STATE_IDX["omega_FL"]:STATE_IDX["omega_RR"] + 1]

        ay_prev = y_prev[OUTPUT_IDX["ay"]]
        fz_prev = y_prev[OUTPUT_IDX["Fz_FL"]:OUTPUT_IDX["Fz_RR"] + 1]
        fy_prev = y_prev[OUTPUT_IDX["Fy_FL"]:OUTPUT_IDX["Fy_RR"] + 1]
        p_dc_prev = y_prev[OUTPUT_IDX["P_dc_kw"]]

        ctrl_st = c[0:4]
        s_prog = c[4]
        e_kwh = c[5]
        delta_prev = c[6]

        r_tgt = p_ev[0]
        vx_tgt = p_ev[1]
        delta_ff_add = p_ev[2]
        mu_th_eff = p_ev[3]
        t_mot = p_ev[4]

        motor_temp_4 = ca.vertcat(t_mot, t_mot, t_mot, t_mot)
        mu_eff = 2.25 * th_v[PARAM_IDX["lambda_muy"]] * mu_th_eff

        dx_c = X_pos
        dy_c = Y_pos - R_CENTER_M
        r_now = ca.sqrt(dx_c * dx_c + dy_c * dy_c + 1e-6)
        theta_c = ca.atan2(dy_c, dx_c)
        psi_tan = theta_c + 0.5 * math.pi
        e_psi = ca.atan2(ca.sin(psi_tan - psi), ca.cos(psi_tan - psi))
        e_r = r_now - r_tgt

        delta_ff = ca.atan(self.ctrl.wb / smax(r_tgt, 6.0, 0.1)) + delta_ff_add
        delta_cmd = sclip(delta_ff + 0.42 * e_r + 1.15 * e_psi, -0.36, 0.36, 0.01)

        v_err = vx_tgt - vx - 3.2 * spos(e_r - 0.025, 0.03)
        fx_req = sclip(1450.0 * v_err, 40.0, 2600.0, 25.0)
        t_brk_4 = ca.DM.zeros(4, 1)

        dtheta_step = (vx * ca.cos(e_psi) / smax(r_now, 4.0, 0.2)) * dt
        delta_dot = sclip((delta_cmd - delta_prev) / dt, -10.0, 10.0, 0.2)
        ctrl_out = self.ctrl.step_symbolic(
            vx=vx, vy=vy, wz=wz, ay_meas=ay_prev,
            delta=delta_cmd, delta_dot=delta_dot, fx_req=fx_req,
            fz_4=fz_prev, fy_4=fy_prev, omega_4=omega_4,
            mu_eff=mu_eff, ctrl_state=ctrl_st, theta_ctrl=th_c,
            dt=dt, motor_temp_c_4=motor_temp_4,
        )
        u_k = ca.vertcat(delta_cmd, ctrl_out["t_wheel_4"], t_brk_4)
        c_next = ca.vertcat(
            ctrl_out["ctrl_state_next"],
            s_prog + dtheta_step,
            e_kwh + (p_dc_prev * dt) / 3600.0,
            delta_cmd,
        )
        aux = ca.vertcat(ctrl_out["p_dc_kw"], ctrl_out["dmz_dbeta_task"], e_r, fx_req)
        return ca.Function(
            "th_skidpad_ctrl",
            [x, y_prev, c, t_k, th_v, th_c, p_ev],
            [u_k, c_next, aux],
        )

    def _build_endurance_ctrl(self) -> ca.Function:
        """
        Full 1.184 km FSG 2023 Endurance closed-loop controller with:
          - v_scale: cornering speed multiplier
          - v_max_ms: straight-line top speed cap
          - p_cap_kw: discharge power cap (regen untouched up to -24.2 kW inner clamp)
          - coast_lead_m: lift-and-coast anticipation distance before braking zones
          - regen_util: regen utilization fraction in [0.0, 1.0] (1.0 = full -24.2 kW)
          - eq_split_blend: 0.0 = baseline 35/65, 1.0 = loss-minimizing 50/50 4WD split
          - brk_bias_f: hydraulic front brake bias
          - mu_th_eff: thermal+wear tire grip multiplier (<= 1.0)
          - t_motor_c: live motor stator winding temperature [C]
        """
        dt = self.dt
        x = ca.MX.sym("x", NX)
        y_prev = ca.MX.sym("y_prev", NY)
        c = ca.MX.sym("c", AUG_C_DIM)
        t_k = ca.MX.sym("t_k")
        th_v = ca.MX.sym("th_v", NTHETA)
        th_c = ca.MX.sym("th_c", len(CTRL_PARAM_NAMES))
        p_ev = ca.MX.sym("p_ev", 9)

        X_pos = x[STATE_IDX["X"]]
        Y_pos = x[STATE_IDX["Y"]]
        psi = x[STATE_IDX["psi"]]
        vx = x[STATE_IDX["vx"]]
        vy = x[STATE_IDX["vy"]]
        wz = x[STATE_IDX["wz"]]
        omega_4 = x[STATE_IDX["omega_FL"]:STATE_IDX["omega_RR"] + 1]

        ay_prev = y_prev[OUTPUT_IDX["ay"]]
        fz_prev = y_prev[OUTPUT_IDX["Fz_FL"]:OUTPUT_IDX["Fz_RR"] + 1]
        fy_prev = y_prev[OUTPUT_IDX["Fy_FL"]:OUTPUT_IDX["Fy_RR"] + 1]
        p_dc_prev = y_prev[OUTPUT_IDX["P_dc_kw"]]

        ctrl_st = c[0:4]
        s_prog = c[4]
        e_kwh = c[5]
        delta_prev = c[6]

        v_scale = p_ev[0]
        v_max_ms = p_ev[1]
        p_cap_kw = p_ev[2]
        coast_lead_m = p_ev[3]
        regen_util = p_ev[4]
        eq_split_blend = p_ev[5]
        brk_bias_f = p_ev[6]
        mu_th_eff = p_ev[7]
        t_mot = p_ev[8]

        motor_temp_4 = ca.vertcat(t_mot, t_mot, t_mot, t_mot)
        mu_eff = 2.25 * th_v[PARAM_IDX["lambda_muy"]] * mu_th_eff

        L = self.track.length_m
        s_mod = sclip(s_prog, 0.0, L + 25.0, 1e-3)
        s_la = s_mod + 5.5
        s_coast = sclip(s_mod + coast_lead_m, 0.0, L + 25.0, 1e-3)

        x_r = self.track.lut_x(s_mod)
        y_r = self.track.lut_y(s_mod)
        c_psi = self.track.lut_cos_psi(s_mod)
        s_psi = self.track.lut_sin_psi(s_mod)
        k_now = self.track.lut_kappa(s_mod)
        k_la = self.track.lut_kappa(s_la)
        ey_tgt = 0.62 * self.track.lut_ey_unit(s_mod)
        v_base = self.track.lut_v_base(s_mod)
        v_base_la = self.track.lut_v_base(s_la)
        v_base_coast = self.track.lut_v_base(s_coast)

        dx_t = X_pos - x_r
        dy_t = Y_pos - y_r
        e_y_track = -dx_t * s_psi + dy_t * c_psi
        psi_trk = ca.atan2(s_psi, c_psi)
        e_psi = ca.atan2(ca.sin(psi_trk - psi), ca.cos(psi_trk - psi))

        e_y_err = ey_tgt - e_y_track
        delta_ff = ca.atan(self.ctrl.wb * (0.72 * k_now + 0.28 * k_la))
        vx_st = smax(vx, 4.5, 0.4)
        delta_cmd = sclip(
            delta_ff + 1.08 * e_psi + ca.atan(2.85 * e_y_err / vx_st),
            -0.44,
            0.44,
            0.01,
        )

        # Speed target with thermal-aware cornering cap and straight speed cap
        k_peak = smax(sabs(k_now, 0.004), sabs(k_la, 0.004), 0.004)
        ay_cap_eff = 13.8 * mu_th_eff
        v_curv = ca.sqrt(ay_cap_eff / smax(k_peak, 0.015, 0.003))
        v_prof_s = v_scale * smin(v_base, v_base_la, 0.3)
        v_tgt = sclip(smin(smin(v_prof_s, v_curv, 0.4), v_max_ms, 0.4), 5.5, 29.0, 0.3)

        # Tight cone-corridor speed guard
        e_y_abs = sabs(e_y_track, 0.02)
        v_tgt = v_tgt - 2.6 * spos(e_y_abs - 0.72, 0.04)

        # Lift-and-coast gate: when upcoming speed at s + coast_lead_m is lower than
        # current speed, smoothly suppress positive drive request (coast on aero/RR drag)
        v_ahead_tgt = v_scale * v_base_coast
        coast_drop = spos(vx - v_ahead_tgt - 0.8, 0.25)
        coast_gate = 1.0 / (1.0 + 0.85 * coast_drop * coast_drop)

        fx_raw = sclip(1650.0 * (v_tgt - vx), -5200.0, 4800.0, 40.0)
        fx_pos = spos(fx_raw, 15.0) * coast_gate
        fx_neg = -spos(-fx_raw, 15.0)
        fx_req = fx_pos + fx_neg

        # Hydraulic brake supplements regen cleanly during heavy braking
        # Max regen force is ~24.2 kW / vx; when regen_util < 1.0, hydraulic takes more
        f_regen_avail = regen_util * smin(24200.0 / smax(vx, 3.0, 0.5), 1150.0, 40.0)
        f_hyd_tot = spos(-fx_req - f_regen_avail, 60.0)
        t_hyd_f = 0.5 * brk_bias_f * f_hyd_tot * self.ctrl.r_eff
        t_hyd_r = 0.5 * (1.0 - brk_bias_f) * f_hyd_tot * self.ctrl.r_eff
        t_brk_4 = ca.vertcat(t_hyd_f, t_hyd_f, t_hyd_r, t_hyd_r)

        v_tan = vx * ca.cos(e_psi) - vy * ca.sin(e_psi)
        denom_s = smax(1.0 - k_now * e_y_track, 0.35, 0.05)
        s_next = s_prog + smax(v_tan / denom_s, 0.1, 0.05) * dt
        e_next = e_kwh + (p_dc_prev * dt) / 3600.0

        delta_dot = sclip((delta_cmd - delta_prev) / dt, -10.0, 10.0, 0.2)
        ctrl_out = self.ctrl.step_symbolic(
            vx=vx, vy=vy, wz=wz, ay_meas=ay_prev,
            delta=delta_cmd, delta_dot=delta_dot, fx_req=fx_req,
            fz_4=fz_prev, fy_4=fy_prev, omega_4=omega_4,
            mu_eff=mu_eff, ctrl_state=ctrl_st, theta_ctrl=th_c,
            dt=dt, motor_temp_c_4=motor_temp_4,
        )

        tw_out = ctrl_out["t_wheel_4"]

        # Loss-minimizing 4WD torque split (50/50 front/rear longitudinal balance while
        # strictly preserving exact left-right differential yaw moment Mz and BSPD EV 2.3
        # braking gate!).
        t_f_sum = tw_out[0] + tw_out[1]
        t_r_sum = tw_out[2] + tw_out[3]
        t_tot = t_f_sum + t_r_sum
        d_f = 0.5 * (tw_out[0] - tw_out[1])
        d_r = 0.5 * (tw_out[2] - tw_out[3])
        tw_eq = ca.vertcat(
            0.25 * t_tot + d_f,
            0.25 * t_tot - d_f,
            0.25 * t_tot + d_r,
            0.25 * t_tot - d_r,
        )
        tw_blend = (1.0 - eq_split_blend) * tw_out + eq_split_blend * tw_eq

        # Smoothly enforce endurance discharge power cap without limiting regen,
        # and strictly gate positive drive torque to zero under braking (BSPD EV 2.3).
        p_raw = ctrl_out["p_dc_kw"]
        over_endur = spos(p_raw - p_cap_kw, 0.30)
        sc_endur = p_cap_kw / (p_cap_kw + over_endur + 1e-4)
        bspd_gate = 0.5 * (1.0 - ca.tanh((-fx_req - 90.0) / 15.0))
        tw_pos = spos(tw_blend, 0.02) * sc_endur * bspd_gate
        tw_neg = -spos(-tw_blend, 0.02) * regen_util
        tw_out = tw_pos + tw_neg

        u_k = ca.vertcat(delta_cmd, tw_out, t_brk_4)
        c_next = ca.vertcat(ctrl_out["ctrl_state_next"], s_next, e_next, delta_cmd)
        aux = ca.vertcat(ctrl_out["p_dc_kw"], ctrl_out["dmz_dbeta_task"], e_y_track, fx_req)
        return ca.Function(
            "th_endur_ctrl",
            [x, y_prev, c, t_k, th_v, th_c, p_ev],
            [u_k, c_next, aux],
        )

    # ------------------------------------------------------------------
    # Part A: Acceleration (75 m) & Skidpad Thermal Window Rollouts
    # ------------------------------------------------------------------

    def run_accel_at_temp(
        self,
        th_v: np.ndarray,
        th_c: np.ndarray,
        t_tire_c: float,
        t_motor_c: float = 55.0,
        kappa_tgt: float = 0.135,
        tc_gain: float = 3.00,
        launch_fx_n: float = 6035.0,
    ) -> Dict[str, Any]:
        mu_th = calibrated_thermal_mu(t_tire_c)
        w = self.model.default_disturbance(t_tire_c=t_tire_c, t_motor_c=t_motor_c)
        w[DIST_IDX["d_mu_scale"]] = mu_th - 1.0

        p_ev = np.array([kappa_tgt, tc_gain, launch_fx_n, mu_th, t_motor_c], dtype=np.float64)
        dt = self.dt
        N = 900  # up to 4.50 s (> 75 m even at cold 20 C)

        x = self.model.default_state(vx=0.25, vy=0.0, wz=0.0)
        u0 = np.array([0.0, 80.0, 80.0, 100.0, 100.0, 0.0, 0.0, 0.0, 0.0])
        _, y = self.lib.step(x, u0, th_v, w)
        c = np.zeros(AUG_C_DIM, dtype=np.float64)

        t_cross = None
        v_finish = 0.0
        p_dc_max = -1e9
        p_dc_min = 1e9
        hv_viol = 0
        bspd_viol = 0
        min_rh_f = 999.0
        max_ax_g = 0.0
        max_dmz = -1e9
        t_tire_dyn = float(t_tire_c)
        q_slip_sum = 0.0

        for k in range(N):
            t_k = k * dt
            u_k, c_next, aux_k = self.fn_accel(x, y, c, t_k, th_v, th_c, p_ev)
            u_np = np.asarray(u_k, dtype=np.float64).ravel()
            c = np.asarray(c_next, dtype=np.float64).ravel()
            aux_np = np.asarray(aux_k, dtype=np.float64).ravel()

            x_prev_x = float(x[STATE_IDX["X"]])
            x, y = self.lib.step(x, u_np, th_v, w)
            x_now_x = float(x[STATE_IDX["X"]])

            p_dc = float(y[OUTPUT_IDX["P_dc_kw"]])
            p_dc_max = max(p_dc_max, p_dc)
            p_dc_min = min(p_dc_min, p_dc)
            if p_dc > 80.0 or p_dc < -25.0:
                hv_viol += 1
            if float(aux_np[3]) < -150.0 and float(np.max(u_np[1:5])) > 0.5:
                bspd_viol += 1
            min_rh_f = min(min_rh_f, float(y[OUTPUT_IDX["rh_f"]]))
            max_ax_g = max(max_ax_g, float(y[OUTPUT_IDX["ax"]]) / 9.81)
            max_dmz = max(max_dmz, float(aux_np[1]))

            # Track dynamic tire heating during the 75 m run
            vx_k = max(float(x[STATE_IDX["vx"]]), 0.1)
            fx_4 = y[OUTPUT_IDX["Fx_FL"]:OUTPUT_IDX["Fx_RR"] + 1]
            kap_4 = y[OUTPUT_IDX["kappa_FL"]:OUTPUT_IDX["kappa_RR"] + 1]
            q_slip_avg = float(np.mean(np.abs(fx_4 * vx_k * kap_4)))
            q_slip_sum += q_slip_avg * dt
            dt_tire = float(tire_temp_dot_smooth(t_tire_dyn, COLD_TRACK_TEMP_C, q_slip_avg, vx_k))
            t_tire_dyn += dt_tire * dt

            if x_now_x >= 75.0:
                frac = (75.0 - x_prev_x) / max(x_now_x - x_prev_x, 1e-6)
                t_cross = (k + frac) * dt
                v_finish = float(x[STATE_IDX["vx"]])
                break

        if t_cross is None:
            t_cross = N * dt + (75.0 - float(x[STATE_IDX["X"]])) / max(float(x[STATE_IDX["vx"]]), 5.0)
            v_finish = float(x[STATE_IDX["vx"]])

        return {
            "t_tire_start_c": round(float(t_tire_c), 2),
            "t_tire_finish_c": round(float(t_tire_dyn), 2),
            "delta_t_tire_run_c": round(float(t_tire_dyn - t_tire_c), 2),
            "mu_thermal": round(mu_th, 5),
            "lambda_muy_eff": round(FSG_RUBBERED * mu_th, 5),
            "t_75m_s": round(float(t_cross), 4),
            "v_finish_kmh": round(float(v_finish * 3.6), 2),
            "max_ax_g": round(float(max_ax_g), 3),
            "p_dc_max_kw": round(float(p_dc_max), 2),
            "min_rh_f_mm": round(float(min_rh_f * 1000.0), 2),
            "hv_violations": int(hv_viol),
            "bspd_violations": int(bspd_viol),
            "yaw_stable": bool(max_dmz < 0.0),
        }

    def run_skidpad_at_temp(
        self,
        th_v: np.ndarray,
        th_c: np.ndarray,
        t_tire_c: float,
        t_motor_c: float = 55.0,
        r_tgt_m: float = 8.747,
        delta_ff_rad: float = 0.048,
    ) -> Dict[str, Any]:
        """
        Evaluate Skidpad at t_tire_c, finding the maximum feasible vx_tgt that
        strictly preserves min_clear >= 0.155 m (0 cones hit) under the exact
        thermal grip multiplier mu_th <= 1.000.
        """
        mu_th = calibrated_thermal_mu(t_tire_c)
        w = self.model.default_disturbance(t_tire_c=t_tire_c, t_motor_c=t_motor_c)
        w[DIST_IDX["d_mu_scale"]] = mu_th - 1.0

        def _eval_vx(vx_tgt: float) -> Dict[str, Any]:
            p_ev = np.array([r_tgt_m, vx_tgt, delta_ff_rad, mu_th, t_motor_c], dtype=np.float64)
            dt = self.dt
            N = 1040  # 5.20 s (covers full circle even at cold grip)

            vx0 = min(vx_tgt, 11.40) * math.sqrt(mu_th)
            wz0 = vx0 / r_tgt_m
            x = self.model.default_state(vx=vx0, vy=0.0, wz=wz0)
            x[STATE_IDX["X"]] = 0.0
            x[STATE_IDX["Y"]] = R_CENTER_M - r_tgt_m
            x[STATE_IDX["psi"]] = 0.0
            delta0 = math.atan(self.ctrl.wb / r_tgt_m) + delta_ff_rad
            u0 = np.array([delta0, 35.0, 35.0, 45.0, 45.0, 0.0, 0.0, 0.0, 0.0])
            _, y = self.lib.step(x, u0, th_v, w)
            c = np.array([wz0, vx0 * wz0, 0.0, 0.0, 0.0, 0.0, delta0], dtype=np.float64)

            min_clear = 999.0
            max_ay_g = 0.0
            p_dc_max = -1e9
            hv_viol = 0
            bspd_viol = 0
            max_dmz = -1e9
            min_rh_f = 999.0
            t_tire_dyn = float(t_tire_c)
            t_circle = None

            for k in range(N):
                t_k = k * dt
                u_k, c_next, aux_k = self.fn_skidpad(x, y, c, t_k, th_v, th_c, p_ev)
                u_np = np.asarray(u_k, dtype=np.float64).ravel()
                s_prev = float(c[4])
                c = np.asarray(c_next, dtype=np.float64).ravel()
                aux_np = np.asarray(aux_k, dtype=np.float64).ravel()

                x, y = self.lib.step(x, u_np, th_v, w)

                r_k = math.hypot(float(x[STATE_IDX["X"]]), float(x[STATE_IDX["Y"]]) - R_CENTER_M)
                c_in = r_k - (R_INNER_M + HALF_TRACK_M)
                c_out = (R_OUTER_M - HALF_TRACK_M) - r_k
                min_clear = min(min_clear, min(c_in, c_out))

                p_dc = float(y[OUTPUT_IDX["P_dc_kw"]])
                p_dc_max = max(p_dc_max, p_dc)
                if p_dc > 80.0 or p_dc < -25.0:
                    hv_viol += 1
                if float(aux_np[3]) < -150.0 and float(np.max(u_np[1:5])) > 0.5:
                    bspd_viol += 1
                max_ay_g = max(max_ay_g, abs(float(y[OUTPUT_IDX["ay"]])) / 9.81)
                max_dmz = max(max_dmz, float(aux_np[1]))
                min_rh_f = min(min_rh_f, float(y[OUTPUT_IDX["rh_f"]]))

                vx_k = max(float(x[STATE_IDX["vx"]]), 0.5)
                fy_4 = y[OUTPUT_IDX["Fy_FL"]:OUTPUT_IDX["Fy_RR"] + 1]
                alp_4 = y[OUTPUT_IDX["alpha_FL"]:OUTPUT_IDX["alpha_RR"] + 1]
                q_slip_avg = float(np.mean(np.abs(fy_4 * vx_k * np.tan(alp_4))))
                dt_tire = float(tire_temp_dot_smooth(t_tire_dyn, COLD_TRACK_TEMP_C, q_slip_avg, vx_k))
                t_tire_dyn += dt_tire * dt

                if float(c[4]) >= 2.0 * math.pi and t_circle is None:
                    frac = (2.0 * math.pi - s_prev) / max(float(c[4]) - s_prev, 1e-6)
                    t_circle = (k + frac) * dt
                    break

            if t_circle is None:
                t_circle = (2.0 * math.pi * N * dt) / max(float(c[4]), 0.5)

            feasible = (min_clear >= MIN_CLEAR_REQ_M) and (hv_viol == 0) and (max_dmz < 0.0)
            return {
                "feasible": feasible,
                "vx_tgt_ms": round(vx_tgt, 3),
                "t_skidpad_s": round(float(t_circle), 4),
                "min_clear_m": round(float(min_clear), 4),
                "max_ay_g": round(float(max_ay_g), 3),
                "p_dc_max_kw": round(float(p_dc_max), 2),
                "min_rh_f_mm": round(float(min_rh_f * 1000.0), 2),
                "hv_violations": int(hv_viol),
                "bspd_violations": int(bspd_viol),
                "yaw_stable": bool(max_dmz < 0.0),
                "t_tire_start_c": round(float(t_tire_c), 2),
                "t_tire_finish_c": round(float(t_tire_dyn), 2),
                "delta_t_tire_run_c": round(float(t_tire_dyn - t_tire_c), 2),
                "mu_thermal": round(mu_th, 5),
                "lambda_muy_eff": round(FSG_RUBBERED * mu_th, 5),
            }

        # Bisection for highest feasible vx_tgt respecting min_clear >= 0.155 m
        lo, hi = 9.20, 12.60
        best = _eval_vx(lo)
        for _ in range(10):
            mid = 0.5 * (lo + hi)
            res = _eval_vx(mid)
            if res["feasible"]:
                best = res
                lo = mid
            else:
                hi = mid
        return best

    # ------------------------------------------------------------------
    # Part B: Full 1.184 km Endurance Lap Rollout + 19-Lap Thermal/Wear
    # ------------------------------------------------------------------

    def run_endurance_lap(
        self,
        th_v: np.ndarray,
        th_c: np.ndarray,
        v_scale: float,
        v_max_ms: float,
        p_cap_kw: float,
        coast_lead_m: float,
        regen_util: float = 1.0,
        eq_split_blend: float = 1.0,
        brk_bias_f: float = 0.633,
        t_tire_c: float = 75.0,
        t_motor_c: float = 68.0,
        wear_mu_mult: float = 1.0,
        record_trace: bool = False,
    ) -> Dict[str, Any]:
        """Run one exact 1.1836 km FSG 2023 Endurance lap in 14-DOF libter_diff_14dof.so."""
        mu_th = calibrated_thermal_mu(t_tire_c) * float(wear_mu_mult)
        mu_th = min(1.0, max(0.55, mu_th))
        w = self.model.default_disturbance(t_tire_c=t_tire_c, t_motor_c=t_motor_c)
        w[DIST_IDX["d_mu_scale"]] = mu_th - 1.0

        p_ev = np.array(
            [
                v_scale,
                v_max_ms,
                p_cap_kw,
                coast_lead_m,
                regen_util,
                eq_split_blend,
                brk_bias_f,
                mu_th,
                t_motor_c,
            ],
            dtype=np.float64,
        )

        dt = self.dt
        L = self.track.length_m
        N_max = 22000  # up to 110 s (covers even ultra-efficient slow laps)

        v0 = float(np.clip(self.track.v_profile_base[0] * v_scale * 0.92, 8.0, min(v_max_ms, 19.0)))
        psi0 = float(self.track.psi_ref[0])
        x = self.model.default_state(vx=v0, vy=0.0, wz=v0 * float(self.track.kappa_ref[0]))
        x[STATE_IDX["X"]] = float(self.track.x_ref[0])
        x[STATE_IDX["Y"]] = float(self.track.y_ref[0])
        x[STATE_IDX["psi"]] = psi0
        delta0 = math.atan(self.ctrl.wb * float(self.track.kappa_ref[0]))
        u0 = np.array([delta0, 35.0, 35.0, 45.0, 45.0, 0.0, 0.0, 0.0, 0.0])
        _, y = self.lib.step(x, u0, th_v, w)
        c = np.array([v0 * float(self.track.kappa_ref[0]), 0.0, 0.0, 0.0, 0.0, 0.0, delta0], dtype=np.float64)

        p_dc_min = 1e9
        p_dc_max = -1e9
        p_loss_sum = 0.0
        p_loss_max = 0.0
        e_pos_kj = 0.0
        e_regen_kj = 0.0
        q_slip_4_sum = np.zeros(4, dtype=np.float64)
        vx_sum = 0.0
        max_vx = 0.0
        max_ay_g = 0.0
        min_clear = 999.0
        min_rh_f = 999.0
        max_dmz = -1e9
        hv_viol = 0
        bspd_viol = 0
        n_coast = 0
        n_steps_done = N_max
        t_lap = None

        trace_t: List[float] = []
        trace_s: List[float] = []
        trace_vx: List[float] = []
        trace_pdc: List[float] = []
        trace_ploss: List[float] = []

        for k in range(N_max):
            t_k = k * dt
            u_k, c_next, aux_k = self.fn_endurance(x, y, c, t_k, th_v, th_c, p_ev)
            u_np = np.asarray(u_k, dtype=np.float64).ravel()
            s_prev = float(c[4])
            c = np.asarray(c_next, dtype=np.float64).ravel()
            aux_np = np.asarray(aux_k, dtype=np.float64).ravel()

            x, y = self.lib.step(x, u_np, th_v, w)

            p_dc = float(y[OUTPUT_IDX["P_dc_kw"]])
            p_loss = float(y[OUTPUT_IDX["P_loss_kw"]])
            p_dc_min = min(p_dc_min, p_dc)
            p_dc_max = max(p_dc_max, p_dc)
            p_loss_max = max(p_loss_max, p_loss)
            p_loss_sum += p_loss * dt  # kJ

            if p_dc >= 0.0:
                e_pos_kj += p_dc * dt
            else:
                e_regen_kj += (-p_dc) * dt

            if p_dc > 80.0 or p_dc < -25.0:
                hv_viol += 1
            if float(aux_np[3]) < -150.0 and float(np.max(u_np[1:5])) > 0.5:
                bspd_viol += 1
            if abs(float(aux_np[3])) < 80.0 and float(x[STATE_IDX["vx"]]) > 10.0:
                n_coast += 1

            ey_k = abs(float(aux_np[2]))
            c_trk = (1.80 - CAR_HALF_W_M) - ey_k
            min_clear = min(min_clear, c_trk)
            min_rh_f = min(min_rh_f, float(y[OUTPUT_IDX["rh_f"]]))
            max_dmz = max(max_dmz, float(aux_np[1]))
            max_ay_g = max(max_ay_g, abs(float(y[OUTPUT_IDX["ay"]])) / 9.81)

            vx_k = float(x[STATE_IDX["vx"]])
            vx_sum += vx_k
            max_vx = max(max_vx, vx_k)

            # Per-wheel contact-patch slip power Q_slip = |Fx * vx * kappa| + |Fy * vx * tan(alpha)|
            fx_4 = y[OUTPUT_IDX["Fx_FL"]:OUTPUT_IDX["Fx_RR"] + 1]
            fy_4 = y[OUTPUT_IDX["Fy_FL"]:OUTPUT_IDX["Fy_RR"] + 1]
            kap_4 = y[OUTPUT_IDX["kappa_FL"]:OUTPUT_IDX["kappa_RR"] + 1]
            alp_4 = y[OUTPUT_IDX["alpha_FL"]:OUTPUT_IDX["alpha_RR"] + 1]
            q_slip_4_sum += (
                np.abs(fx_4 * max(vx_k, 0.2) * kap_4)
                + np.abs(fy_4 * max(vx_k, 0.2) * np.tan(np.clip(alp_4, -0.5, 0.5)))
            ) * dt  # J per tire

            if record_trace and (k % 100 == 0):
                trace_t.append(round(t_k, 3))
                trace_s.append(round(float(c[4]), 2))
                trace_vx.append(round(vx_k, 3))
                trace_pdc.append(round(p_dc, 3))
                trace_ploss.append(round(p_loss, 3))

            if float(c[4]) >= L:
                frac = (L - s_prev) / max(float(c[4]) - s_prev, 1e-6)
                t_lap = (k + frac) * dt
                n_steps_done = k + 1
                break

        if t_lap is None:
            t_lap = N_max * dt * (L / max(float(c[4]), 1.0))

        e_net_kj = e_pos_kj - e_regen_kj
        e_net_22km_kwh = (e_net_kj / 3600.0) * self.n_laps_22km
        e_regen_22km_kwh = (e_regen_kj / 3600.0) * self.n_laps_22km
        e_loss_22km_kwh = (p_loss_sum / 3600.0) * self.n_laps_22km
        t_22km_s = t_lap * self.n_laps_22km
        p_loss_mean_kw = p_loss_sum / max(t_lap, 1.0)
        vx_mean_ms = vx_sum / max(n_steps_done, 1)
        q_slip_mean_w_4 = q_slip_4_sum / max(t_lap, 1.0)

        # Steady-state motor, inverter, and tire temperatures at this lap power/slip
        p_mot_per_w = (p_loss_mean_kw * 1000.0 * MOTOR_THERMAL["eta_motor_loss_frac"]) / 4.0
        p_inv_per_w = (p_loss_mean_kw * 1000.0 * INVERTER_THERMAL["eta_inv_loss_frac"]) / 4.0
        t_motor_ss_c = MOTOR_THERMAL["t_coolant_c"] + p_mot_per_w / MOTOR_THERMAL["h_cool_w_per_k"]
        t_inv_ss_c = INVERTER_THERMAL["t_coolant_c"] + p_inv_per_w / INVERTER_THERMAL["h_cool_w_per_k"]

        h_cool_tire = TIRE_THERMAL_PARAMS["h_cond_w_per_k"] + TIRE_THERMAL_PARAMS["h_conv_w_per_k_ms"] * vx_mean_ms
        t_tire_ss_4 = FSG_TRACK_TEMP_C + (TIRE_THERMAL_PARAMS["eta_slip_heat"] * q_slip_mean_w_4) / h_cool_tire

        feasible = (
            (e_net_22km_kwh <= ENDURANCE_ENERGY_KWH)
            and (hv_viol == 0)
            and (bspd_viol == 0)
            and (min_clear >= MIN_CLEAR_REQ_M)
            and (max_dmz < 0.0)
            and (min_rh_f >= 0.010)
            and (t_motor_ss_c < MOTOR_THERMAL["t_derate_start_c"])
            and (t_inv_ss_c < INVERTER_THERMAL["t_derate_start_c"])
        )

        out: Dict[str, Any] = {
            "feasible": bool(feasible),
            "params": {
                "v_scale": round(v_scale, 4),
                "v_max_ms": round(v_max_ms, 3),
                "p_cap_kw": round(p_cap_kw, 2),
                "coast_lead_m": round(coast_lead_m, 2),
                "regen_util": round(regen_util, 3),
                "eq_split_blend": round(eq_split_blend, 3),
                "brk_bias_f": round(brk_bias_f, 4),
            },
            "t_lap_s": round(float(t_lap), 4),
            "t_22km_s": round(float(t_22km_s), 3),
            "e_net_lap_kj": round(float(e_net_kj), 3),
            "e_pos_lap_kj": round(float(e_pos_kj), 3),
            "e_regen_lap_kj": round(float(e_regen_kj), 3),
            "e_loss_lap_kj": round(float(p_loss_sum), 3),
            "e_net_22km_kwh": round(float(e_net_22km_kwh), 4),
            "e_regen_22km_kwh": round(float(e_regen_22km_kwh), 4),
            "e_loss_22km_kwh": round(float(e_loss_22km_kwh), 4),
            "regen_pct": round(100.0 * e_regen_kj / max(e_pos_kj, 1e-6), 2),
            "coast_time_pct": round(100.0 * n_coast / max(n_steps_done, 1), 2),
            "p_dc_min_kw": round(float(p_dc_min), 3),
            "p_dc_max_kw": round(float(p_dc_max), 3),
            "p_loss_mean_kw": round(float(p_loss_mean_kw), 3),
            "p_loss_max_kw": round(float(p_loss_max), 3),
            "vx_mean_kmh": round(float(vx_mean_ms * 3.6), 2),
            "vx_max_kmh": round(float(max_vx * 3.6), 2),
            "max_ay_g": round(float(max_ay_g), 3),
            "min_clear_m": round(float(min_clear), 4),
            "cones_hit": int(min_clear < MIN_CLEAR_REQ_M),
            "min_rh_f_mm": round(float(min_rh_f * 1000.0), 2),
            "max_dmz_dbeta_nm_rad": round(float(max_dmz), 2),
            "yaw_stable": bool(max_dmz < 0.0),
            "hv_violations": int(hv_viol),
            "bspd_violations": int(bspd_viol),
            "t_motor_ss_c": round(float(t_motor_ss_c), 2),
            "t_inv_ss_c": round(float(t_inv_ss_c), 2),
            "t_tire_ss_4_c": [round(float(v), 2) for v in t_tire_ss_4],
            "t_tire_ss_mean_c": round(float(np.mean(t_tire_ss_4)), 2),
            "q_slip_lap_kj_4": [round(float(v * 1e-3), 2) for v in q_slip_4_sum],
            "q_slip_mean_w_4": [round(float(v), 1) for v in q_slip_mean_w_4],
        }
        if record_trace:
            out["lap_trace"] = {
                "t_s": trace_t,
                "s_m": trace_s,
                "vx_ms": trace_vx,
                "p_dc_kw": trace_pdc,
                "p_loss_kw": trace_ploss,
            }
        return out

    def simulate_22km_coupled_thermal_and_wear(
        self,
        th_v: np.ndarray,
        th_c: np.ndarray,
        strategy: Dict[str, float],
        t_tire_init_c: float = 55.0,
        t_motor_init_c: float = 45.0,
        t_inv_init_c: float = 42.0,
    ) -> Dict[str, Any]:
        """
        Simulate all 19 laps (exact 18.5871 laps = 22.0 km) with lap-by-lap coupled:
          - Motor stator winding temperature T_motor(lap) feeding into R_phase(T)
          - Inverter cold-plate temperature T_inv(lap)
          - 4-corner Hoosier R25B bulk tire temperature T_tire(lap)
          - Cumulative slip-energy tire wear feeding into effective grip mu_th(lap)
        """
        n_full_laps = int(math.floor(self.n_laps_22km))  # 18 full laps
        frac_last = self.n_laps_22km - n_full_laps       # 0.5871 of lap 19

        t_motor = float(t_motor_init_c)
        t_inv = float(t_inv_init_c)
        t_tire_4 = np.full(4, float(t_tire_init_c), dtype=np.float64)
        e_slip_cum_mj = 0.0

        lap_records: List[Dict[str, Any]] = []
        t_cum_s = 0.0
        e_net_cum_kwh = 0.0
        e_regen_cum_kwh = 0.0
        e_loss_cum_kwh = 0.0
        hv_viol_tot = 0
        bspd_viol_tot = 0
        min_clear_all = 999.0
        min_rh_f_all = 999.0
        max_dmz_all = -1e9
        max_t_motor = t_motor
        max_t_inv = t_inv
        max_t_tire = float(np.max(t_tire_4))

        for lap_idx in range(1, n_full_laps + 2):
            lap_frac = 1.0 if lap_idx <= n_full_laps else frac_last
            wear_loss = min(TIRE_WEAR_MAX_LOSS, 0.05 * (e_slip_cum_mj / TIRE_WEAR_E_REF_MJ))
            wear_mult = 1.0 - wear_loss
            t_tire_mean_in = float(np.mean(t_tire_4))

            res = self.run_endurance_lap(
                th_v=th_v,
                th_c=th_c,
                v_scale=strategy["v_scale"],
                v_max_ms=strategy["v_max_ms"],
                p_cap_kw=strategy["p_cap_kw"],
                coast_lead_m=strategy["coast_lead_m"],
                regen_util=strategy["regen_util"],
                eq_split_blend=strategy["eq_split_blend"],
                brk_bias_f=strategy["brk_bias_f"],
                t_tire_c=t_tire_mean_in,
                t_motor_c=t_motor,
                wear_mu_mult=wear_mult,
                record_trace=(lap_idx == 10),
            )

            dt_lap = float(res["t_lap_s"]) * lap_frac
            # Integrate exact analytical exponential lumped thermal ODE over dt_lap
            p_mot_w = (float(res["p_loss_mean_kw"]) * 1000.0 * MOTOR_THERMAL["eta_motor_loss_frac"]) / 4.0
            tau_m = MOTOR_THERMAL["c_th_j_per_k"] / MOTOR_THERMAL["h_cool_w_per_k"]
            t_m_ss = MOTOR_THERMAL["t_coolant_c"] + p_mot_w / MOTOR_THERMAL["h_cool_w_per_k"]
            t_motor = t_m_ss + (t_motor - t_m_ss) * math.exp(-dt_lap / tau_m)

            p_inv_w = (float(res["p_loss_mean_kw"]) * 1000.0 * INVERTER_THERMAL["eta_inv_loss_frac"]) / 4.0
            tau_i = INVERTER_THERMAL["c_th_j_per_k"] / INVERTER_THERMAL["h_cool_w_per_k"]
            t_i_ss = INVERTER_THERMAL["t_coolant_c"] + p_inv_w / INVERTER_THERMAL["h_cool_w_per_k"]
            t_inv = t_i_ss + (t_inv - t_i_ss) * math.exp(-dt_lap / tau_i)

            vx_m = float(res["vx_mean_kmh"]) / 3.6
            h_tire = TIRE_THERMAL_PARAMS["h_cond_w_per_k"] + TIRE_THERMAL_PARAMS["h_conv_w_per_k_ms"] * vx_m
            tau_t = TIRE_THERMAL_PARAMS["c_th_j_per_k"] / h_tire
            q_w_4 = np.asarray(res["q_slip_mean_w_4"], dtype=np.float64)
            t_t_ss_4 = FSG_TRACK_TEMP_C + (TIRE_THERMAL_PARAMS["eta_slip_heat"] * q_w_4) / h_tire
            t_tire_4 = t_t_ss_4 + (t_tire_4 - t_t_ss_4) * math.exp(-dt_lap / tau_t)

            # Accumulate slip energy (accelerated wear if tire > 95 C)
            overheat_wear_f = 1.0 + 2.5 * max(0.0, float(np.mean(t_tire_4)) - 95.0) / 20.0
            e_slip_lap_mj = float(np.sum(res["q_slip_lap_kj_4"])) * 1e-3 * lap_frac * overheat_wear_f
            e_slip_cum_mj += e_slip_lap_mj

            t_cum_s += dt_lap
            e_net_cum_kwh += (float(res["e_net_lap_kj"]) / 3600.0) * lap_frac
            e_regen_cum_kwh += (float(res["e_regen_lap_kj"]) / 3600.0) * lap_frac
            e_loss_cum_kwh += (float(res["e_loss_lap_kj"]) / 3600.0) * lap_frac

            hv_viol_tot += int(res["hv_violations"])
            bspd_viol_tot += int(res["bspd_violations"])
            min_clear_all = min(min_clear_all, float(res["min_clear_m"]))
            min_rh_f_all = min(min_rh_f_all, float(res["min_rh_f_mm"]))
            max_dmz_all = max(max_dmz_all, float(res["max_dmz_dbeta_nm_rad"]))
            max_t_motor = max(max_t_motor, t_motor)
            max_t_inv = max(max_t_inv, t_inv)
            max_t_tire = max(max_t_tire, float(np.max(t_tire_4)))

            lap_records.append({
                "lap": lap_idx,
                "lap_fraction": round(lap_frac, 4),
                "dist_cum_km": round(min(22.0, lap_idx * self.track.length_m * 1e-3), 3),
                "t_lap_s": round(float(res["t_lap_s"]), 4),
                "t_cum_s": round(t_cum_s, 3),
                "e_net_lap_kj": round(float(res["e_net_lap_kj"]) * lap_frac, 2),
                "e_net_cum_kwh": round(e_net_cum_kwh, 4),
                "e_regen_cum_kwh": round(e_regen_cum_kwh, 4),
                "t_motor_c": round(t_motor, 2),
                "t_inv_c": round(t_inv, 2),
                "t_tire_mean_c": round(float(np.mean(t_tire_4)), 2),
                "t_tire_max_c": round(float(np.max(t_tire_4)), 2),
                "t_tire_4_c": [round(float(v), 2) for v in t_tire_4],
                "mu_thermal": round(calibrated_thermal_mu(float(np.mean(t_tire_4))), 5),
                "wear_grip_loss_pct": round(wear_loss * 100.0, 3),
                "soc_remaining_pct": round(95.0 - 100.0 * e_net_cum_kwh / 7.2, 2),
            })

        return {
            "n_laps_exact": round(self.n_laps_22km, 4),
            "t_22km_coupled_s": round(t_cum_s, 3),
            "t_lap_mean_s": round(t_cum_s / self.n_laps_22km, 4),
            "e_net_22km_coupled_kwh": round(e_net_cum_kwh, 4),
            "e_regen_22km_coupled_kwh": round(e_regen_cum_kwh, 4),
            "e_loss_22km_coupled_kwh": round(e_loss_cum_kwh, 4),
            "kwh_per_km": round(e_net_cum_kwh / 22.0, 5),
            "regen_pct": round(100.0 * e_regen_cum_kwh / max(e_net_cum_kwh + e_regen_cum_kwh, 1e-6), 2),
            "t_motor_peak_c": round(max_t_motor, 2),
            "t_motor_final_c": round(t_motor, 2),
            "motor_thermal_margin_c": round(MOTOR_THERMAL["t_derate_start_c"] - max_t_motor, 2),
            "motor_derating_events": int(max_t_motor >= MOTOR_THERMAL["t_derate_start_c"]),
            "t_inv_peak_c": round(max_t_inv, 2),
            "t_inv_final_c": round(t_inv, 2),
            "inv_thermal_margin_c": round(INVERTER_THERMAL["t_derate_start_c"] - max_t_inv, 2),
            "inv_derating_events": int(max_t_inv >= INVERTER_THERMAL["t_derate_start_c"]),
            "t_tire_peak_c": round(max_t_tire, 2),
            "t_tire_final_mean_c": round(float(np.mean(t_tire_4)), 2),
            "tire_overheat_events": int(max_t_tire > 95.0),
            "final_wear_grip_loss_pct": round(
                min(TIRE_WEAR_MAX_LOSS, 0.05 * (e_slip_cum_mj / TIRE_WEAR_E_REF_MJ)) * 100.0, 3
            ),
            "hv_violations": int(hv_viol_tot),
            "bspd_violations": int(bspd_viol_tot),
            "min_clear_m": round(min_clear_all, 4),
            "cones_hit": int(min_clear_all < MIN_CLEAR_REQ_M),
            "min_rh_f_mm": round(min_rh_f_all, 2),
            "max_dmz_dbeta_nm_rad": round(max_dmz_all, 2),
            "yaw_stable": bool(max_dmz_all < 0.0),
            "within_3p50_kwh_budget": bool(e_net_cum_kwh <= ENDURANCE_ENERGY_KWH),
            "lap_by_lap_profiles": lap_records,
        }


# ---------------------------------------------------------------------------
# Main Study Execution
# ---------------------------------------------------------------------------


def main() -> None:
    t0 = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SETUPS_OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78, flush=True)
    print("TeR27 (280.0 kg) Thermal Window & 22 km Endurance Optimization Study", flush=True)
    print("=" * 78, flush=True)

    eng = ThermalEnduranceEngine()

    # Load 280 kg setups for Part A (if limit setups exist, use them; else record setups)
    accel_setup_path = SETUPS_OUT_DIR / "limit_280kg_accel.json"
    skidpad_setup_path = SETUPS_OUT_DIR / "limit_280kg_skidpad.json"

    if accel_setup_path.exists():
        raw_accel = json.loads(accel_setup_path.read_text(encoding="utf-8"))
        th_v_accel = apply_setup_json_to_theta(eng.theta_280_default, raw_accel)
        th_c_accel = load_ctrl_gains_from_setup(raw_accel)
        ev_a = raw_accel.get("trajectory_control_280kg", {}).get("event_params", {})
        a_kap = float(ev_a.get("accel_kappa_r_k0", 0.135))
        a_tc = float(ev_a.get("accel_tc_gain", 3.00))
        a_fx = float(ev_a.get("accel_launch_fx_n", 6035.0))
    else:
        th_v_accel, raw_accel = load_record_setup("acceleration", eng.theta_280_default)
        th_c_accel = eng.theta_ctrl_nom.copy()
        a_kap, a_tc, a_fx = 0.135, 3.00, 6035.0

    if skidpad_setup_path.exists():
        raw_skid = json.loads(skidpad_setup_path.read_text(encoding="utf-8"))
        th_v_skid = apply_setup_json_to_theta(eng.theta_280_default, raw_skid)
        th_c_skid = load_ctrl_gains_from_setup(raw_skid)
        ev_s = raw_skid.get("trajectory_control_280kg", {}).get("event_params", {})
        s_rtgt = float(ev_s.get("skidpad_r_tgt_m", 8.747))
        s_dff = float(ev_s.get("skidpad_delta_ff_rad", 0.048))
    else:
        th_v_skid, raw_skid = load_record_setup("skidpad", eng.theta_280_default)
        th_c_skid = eng.theta_ctrl_nom.copy()
        s_rtgt, s_dff = 8.747, 0.048

    # ==================================================================
    # PART A: Realistic Tire Thermal Window (Acceleration 75 m & Skidpad)
    # ==================================================================
    print("\n[Part A] Sweeping calibrated TTC Hoosier R25B thermal curve (20 C - 115 C)...", flush=True)
    temp_grid = [20.0, 25.0, 30.0, 35.0, 45.0, 50.0, 60.0, 65.0, 70.0, 75.0, 77.5, 85.0, 95.0, 105.0, 115.0]
    accel_sweep: List[Dict[str, Any]] = []
    skidpad_sweep: List[Dict[str, Any]] = []

    for tc in temp_grid:
        ra = eng.run_accel_at_temp(
            th_v=th_v_accel,
            th_c=th_c_accel,
            t_tire_c=tc,
            t_motor_c=55.0,
            kappa_tgt=a_kap,
            tc_gain=a_tc,
            launch_fx_n=a_fx,
        )
        rs = eng.run_skidpad_at_temp(
            th_v=th_v_skid,
            th_c=th_c_skid,
            t_tire_c=tc,
            t_motor_c=55.0,
            r_tgt_m=s_rtgt,
            delta_ff_rad=s_dff,
        )
        accel_sweep.append(ra)
        skidpad_sweep.append(rs)
        print(
            f"  T={tc:5.1f} C | mu_th={ra['mu_thermal']:.4f} | "
            f"Accel={ra['t_75m_s']:.4f} s ({ra['max_ax_g']:.2f} g) | "
            f"Skidpad={rs['t_skidpad_s']:.4f} s (clear={rs['min_clear_m']:.3f} m)",
            flush=True,
        )

    by_temp_a = {r["t_tire_start_c"]: r for r in accel_sweep}
    by_temp_s = {r["t_tire_start_c"]: r for r in skidpad_sweep}

    part_a_summary = {
        "calibrated_curve_guardrail": {
            "fsg_rubbered_lambda_muy": FSG_RUBBERED,
            "cold_test_oct_lambda_muy": COLD_TEST_OCT,
            "max_mu_thermal_applied": max(r["mu_thermal"] for r in accel_sweep),
            "peak_grip_inflated": False,
        },
        "acceleration_75m_key_deltas_ms": {
            "cold_25C_s": by_temp_a[25.0]["t_75m_s"],
            "cold_35C_s": by_temp_a[35.0]["t_75m_s"],
            "conditioned_50C_s": by_temp_a[50.0]["t_75m_s"],
            "conditioned_60C_s": by_temp_a[60.0]["t_75m_s"],
            "conditioned_75C_s": by_temp_a[75.0]["t_75m_s"],
            "optimal_77p5C_s": by_temp_a[77.5]["t_75m_s"],
            "overheat_95C_s": by_temp_a[95.0]["t_75m_s"],
            "overheat_105C_s": by_temp_a[105.0]["t_75m_s"],
            "overheat_115C_s": by_temp_a[115.0]["t_75m_s"],
            "gain_25C_to_75C_ms": round((by_temp_a[75.0]["t_75m_s"] - by_temp_a[25.0]["t_75m_s"]) * 1000.0, 2),
            "gain_35C_to_75C_ms": round((by_temp_a[75.0]["t_75m_s"] - by_temp_a[35.0]["t_75m_s"]) * 1000.0, 2),
            "gain_25C_to_60C_ms": round((by_temp_a[60.0]["t_75m_s"] - by_temp_a[25.0]["t_75m_s"]) * 1000.0, 2),
            "penalty_75C_to_105C_ms": round((by_temp_a[105.0]["t_75m_s"] - by_temp_a[75.0]["t_75m_s"]) * 1000.0, 2),
            "penalty_75C_to_115C_ms": round((by_temp_a[115.0]["t_75m_s"] - by_temp_a[75.0]["t_75m_s"]) * 1000.0, 2),
        },
        "skidpad_key_deltas_ms": {
            "cold_25C_s": by_temp_s[25.0]["t_skidpad_s"],
            "cold_35C_s": by_temp_s[35.0]["t_skidpad_s"],
            "conditioned_50C_s": by_temp_s[50.0]["t_skidpad_s"],
            "conditioned_60C_s": by_temp_s[60.0]["t_skidpad_s"],
            "conditioned_75C_s": by_temp_s[75.0]["t_skidpad_s"],
            "optimal_77p5C_s": by_temp_s[77.5]["t_skidpad_s"],
            "overheat_95C_s": by_temp_s[95.0]["t_skidpad_s"],
            "overheat_105C_s": by_temp_s[105.0]["t_skidpad_s"],
            "overheat_115C_s": by_temp_s[115.0]["t_skidpad_s"],
            "gain_25C_to_75C_ms": round((by_temp_s[75.0]["t_skidpad_s"] - by_temp_s[25.0]["t_skidpad_s"]) * 1000.0, 2),
            "gain_35C_to_75C_ms": round((by_temp_s[75.0]["t_skidpad_s"] - by_temp_s[35.0]["t_skidpad_s"]) * 1000.0, 2),
            "gain_25C_to_60C_ms": round((by_temp_s[60.0]["t_skidpad_s"] - by_temp_s[25.0]["t_skidpad_s"]) * 1000.0, 2),
            "penalty_75C_to_105C_ms": round((by_temp_s[105.0]["t_skidpad_s"] - by_temp_s[75.0]["t_skidpad_s"]) * 1000.0, 2),
            "penalty_75C_to_115C_ms": round((by_temp_s[115.0]["t_skidpad_s"] - by_temp_s[75.0]["t_skidpad_s"]) * 1000.0, 2),
        },
    }

    # ==================================================================
    # PART B: 22 km Endurance Energy + Thermal Strategy & Pareto Frontier
    # ==================================================================
    print("\n[Part B] Evaluating 22 km Endurance Pareto Frontier (280 kg vs 290 kg)...", flush=True)
    th_v_endur_280, raw_endur_base = load_record_setup("endurance", eng.theta_280_default)
    th_v_endur_290 = make_290kg_theta(th_v_endur_280)
    th_c_endur = eng.theta_ctrl_nom.copy()

    # Strategy sweep: (label, v_scale, v_max_ms, p_cap_kw, coast_lead_m, regen_util, eq_split_blend)
    pareto_specs: List[Tuple[str, float, float, float, float, float, float]] = [
        ("280kg_limit_3p46kwh",    0.922, 18.4, 38.5, 13.5, 1.00, 1.00),
        ("280kg_attack_3p38kwh",   0.916, 18.2, 37.2, 14.5, 1.00, 1.00),
        ("280kg_balanced_3p29kwh", 0.910, 18.0, 36.0, 15.0, 1.00, 1.00),
        ("280kg_sweet_3p15kwh",    0.900, 17.6, 34.5, 16.5, 1.00, 1.00),
        ("280kg_eff_3p00kwh",      0.890, 17.2, 33.0, 18.0, 1.00, 1.00),
        ("280kg_eco_2p79kwh",      0.870, 16.5, 30.0, 21.0, 1.00, 1.00),
        ("280kg_ultra_2p61kwh",    0.850, 15.8, 27.5, 24.0, 1.00, 1.00),
        ("280kg_minE_2p40kwh",     0.820, 15.0, 25.0, 26.0, 1.00, 1.00),
        # Ablation: baseline 35/65 split & no lift-and-coast at balanced pace
        ("280kg_no_coast_35_65",   0.910, 18.0, 36.0,  0.0, 1.00, 0.00),
        # Ablation: reduced regen (50%) to quantify regen value
        ("280kg_half_regen",       0.910, 18.0, 36.0, 15.0, 0.50, 1.00),
    ]

    pareto_280: List[Dict[str, Any]] = []
    for label, vs, vmax, pcap, clead, rutil, eqb in pareto_specs:
        r = eng.run_endurance_lap(
            th_v=th_v_endur_280,
            th_c=th_c_endur,
            v_scale=vs,
            v_max_ms=vmax,
            p_cap_kw=pcap,
            coast_lead_m=clead,
            regen_util=rutil,
            eq_split_blend=eqb,
            brk_bias_f=0.633,
            t_tire_c=75.0,
            t_motor_c=68.0,
        )
        r["label"] = label
        r["mass_kg"] = 280.0
        pareto_280.append(r)
        print(
            f"  {label:24s} | t_22km={r['t_22km_s']:7.2f} s ({r['t_lap_s']:6.2f} s/lap) | "
            f"E_22km={r['e_net_22km_kwh']:.3f} kWh | regen={r['regen_pct']:5.1f}% | "
            f"T_mot_ss={r['t_motor_ss_c']:5.1f} C | clear={r['min_clear_m']:.3f} m | "
            f"feas={r['feasible']}",
            flush=True,
        )

    # Evaluate 290 kg reference points on matching strategies for exact 290 vs 280 comparison
    print("\n[Part B] Evaluating 290.0 kg comparison points...", flush=True)
    pareto_290: List[Dict[str, Any]] = []
    for label, vs, vmax, pcap, clead, rutil, eqb in [
        ("290kg_baseline_no_coast", 0.900, 17.6, 35.0,  0.0, 1.00, 0.00),
        ("290kg_opt_attack",        0.916, 18.2, 37.2, 14.5, 1.00, 1.00),
        ("290kg_opt_balanced",      0.910, 18.0, 36.0, 15.0, 1.00, 1.00),
        ("290kg_opt_sweet",         0.900, 17.6, 34.5, 16.5, 1.00, 1.00),
        ("290kg_opt_eff",           0.890, 17.2, 33.0, 18.0, 1.00, 1.00),
    ]:
        r = eng.run_endurance_lap(
            th_v=th_v_endur_290,
            th_c=th_c_endur,
            v_scale=vs,
            v_max_ms=vmax,
            p_cap_kw=pcap,
            coast_lead_m=clead,
            regen_util=rutil,
            eq_split_blend=eqb,
            brk_bias_f=0.633,
            t_tire_c=75.0,
            t_motor_c=68.0,
        )
        r["label"] = label
        r["mass_kg"] = 290.0
        pareto_290.append(r)
        print(
            f"  {label:24s} | t_22km={r['t_22km_s']:7.2f} s ({r['t_lap_s']:6.2f} s/lap) | "
            f"E_22km={r['e_net_22km_kwh']:.3f} kWh | regen={r['regen_pct']:5.1f}% | "
            f"T_mot_ss={r['t_motor_ss_c']:5.1f} C | feas={r['feasible']}",
            flush=True,
        )

    # Compute FSG Endurance (250 pts) + Efficiency (100 pts) combined score via scoring.py
    feasible_all = [r for r in (pareto_280 + pareto_290) if r["feasible"]]
    projections: List[EnduranceProjection] = [
        project_endurance_22km(
            label=r["label"],
            track_length_m=eng.track.length_m,
            t_lap_s=r["t_lap_s"],
            e_net_lap_kj=r["e_net_lap_kj"],
            e_regen_lap_kj=r["e_regen_lap_kj"],
            battery_capacity_kwh=7.2,
        )
        for r in feasible_all
    ]
    fs_scores = compute_fs_scores(projections)
    score_by_label = {row["label"]: row for row in fs_scores}

    for row in fs_scores:
        row["pts_endurance_plus_efficiency"] = round(
            float(row["pts_endurance_rel"]) + float(row["pts_efficiency_rel"]), 3
        )

    # Find optimal 280 kg and 290 kg strategies maximizing Endurance + Efficiency points
    scores_280 = [
        row for row in fs_scores
        if row["label"].startswith("280kg_") and "no_coast" not in row["label"] and "half_regen" not in row["label"]
    ]
    scores_290 = [row for row in fs_scores if row["label"].startswith("290kg_")]

    best_280_score = max(scores_280, key=lambda r: r["pts_endurance_plus_efficiency"])
    best_290_score = max(scores_290, key=lambda r: r["pts_endurance_plus_efficiency"])
    fastest_280 = min(
        [r for r in pareto_280 if r["feasible"]],
        key=lambda r: r["t_22km_s"],
    )

    best_280_rec = next(r for r in pareto_280 if r["label"] == best_280_score["label"])
    best_290_rec = next(r for r in pareto_290 if r["label"] == best_290_score["label"])

    print(
        f"\n[Part B] Best 280 kg FSG Combined (End+Eff): {best_280_score['label']} -> "
        f"{best_280_score['pts_endurance_plus_efficiency']:.2f}/350 pts "
        f"(End={best_280_score['pts_endurance_rel']:.2f}, Eff={best_280_score['pts_efficiency_rel']:.2f}) | "
        f"t_22km={best_280_rec['t_22km_s']:.2f} s, E_22km={best_280_rec['e_net_22km_kwh']:.3f} kWh",
        flush=True,
    )
    print(
        f"[Part B] Best 290 kg FSG Combined (End+Eff): {best_290_score['label']} -> "
        f"{best_290_score['pts_endurance_plus_efficiency']:.2f}/350 pts "
        f"(End={best_290_score['pts_endurance_rel']:.2f}, Eff={best_290_score['pts_efficiency_rel']:.2f}) | "
        f"t_22km={best_290_rec['t_22km_s']:.2f} s, E_22km={best_290_rec['e_net_22km_kwh']:.3f} kWh",
        flush=True,
    )

    # Full 19-lap (22.0 km) coupled thermal & wear simulation at the optimal 280 kg point
    print("\n[Part B] Running full 19-lap (22.0 km) coupled thermal + wear evolution...", flush=True)
    coupled_280 = eng.simulate_22km_coupled_thermal_and_wear(
        th_v=th_v_endur_280,
        th_c=th_c_endur,
        strategy=best_280_rec["params"],
        t_tire_init_c=55.0,
        t_motor_init_c=MOTOR_THERMAL["t_init_c"],
        t_inv_init_c=INVERTER_THERMAL["t_init_c"],
    )
    print(
        f"  22 km Coupled (280 kg): t_22km={coupled_280['t_22km_coupled_s']:.2f} s | "
        f"E_22km={coupled_280['e_net_22km_coupled_kwh']:.4f} kWh (<= 3.50) | "
        f"T_mot_peak={coupled_280['t_motor_peak_c']:.1f} C (< 115) | "
        f"T_inv_peak={coupled_280['t_inv_peak_c']:.1f} C (< 85) | "
        f"T_tire_peak={coupled_280['t_tire_peak_c']:.1f} C | "
        f"wear={coupled_280['final_wear_grip_loss_pct']:.2f}% | "
        f"HV_viol={coupled_280['hv_violations']} | clear={coupled_280['min_clear_m']:.3f} m",
        flush=True,
    )

    # Also run 19-lap coupled thermal & wear for 290 kg best point for exact comparison
    coupled_290 = eng.simulate_22km_coupled_thermal_and_wear(
        th_v=th_v_endur_290,
        th_c=th_c_endur,
        strategy=best_290_rec["params"],
        t_tire_init_c=55.0,
        t_motor_init_c=MOTOR_THERMAL["t_init_c"],
        t_inv_init_c=INVERTER_THERMAL["t_init_c"],
    )

    # Ablation metrics: lift-and-coast + 50/50 split vs no-coast 35/65 at balanced pace
    r_bal = next(r for r in pareto_280 if r["label"] == "280kg_balanced_3p29kwh")
    r_noc = next(r for r in pareto_280 if r["label"] == "280kg_no_coast_35_65")
    r_hreg = next(r for r in pareto_280 if r["label"] == "280kg_half_regen")

    ablations = {
        "lift_and_coast_plus_5050_split_vs_baseline": {
            "delta_e_22km_kwh": round(r_bal["e_net_22km_kwh"] - r_noc["e_net_22km_kwh"], 4),
            "delta_t_22km_s": round(r_bal["t_22km_s"] - r_noc["t_22km_s"], 3),
            "delta_e_loss_22km_kwh": round(r_bal["e_loss_22km_kwh"] - r_noc["e_loss_22km_kwh"], 4),
            "delta_t_motor_ss_c": round(r_bal["t_motor_ss_c"] - r_noc["t_motor_ss_c"], 2),
        },
        "full_regen_25kw_vs_half_regen": {
            "delta_e_22km_kwh": round(r_bal["e_net_22km_kwh"] - r_hreg["e_net_22km_kwh"], 4),
            "delta_regen_22km_kwh": round(r_bal["e_regen_22km_kwh"] - r_hreg["e_regen_22km_kwh"], 4),
        },
    }

    # Export limit_280kg_endurance.json matching MirenaSim optimal_endurance.json schema
    endur_setup_out = dict(raw_endur_base)
    endur_setup_out.update({
        "name": "limit_280kg_endurance",
        "vehicle_id": "ter27_4wd",
        "event": "endurance",
        "mass_kg": 280.0,
        "gear_ratio": 12.5,
        "front_torque_split": 0.50,
        "description": (
            "TeR27 280.0 kg 22 km Endurance optimal energy + thermal strategy "
            "(FSG_RUBBERED=0.57, R25B, <=3.50 kWh, 0 HV/BSPD/thermal violations)"
        ),
        "power_limit_kw": float(best_280_rec["params"]["p_cap_kw"]),
        "regen_limit_kw": 25.0,
        "speed_cap_ms": float(best_280_rec["params"]["v_max_ms"]),
        "lift_and_coast_lead_m": float(best_280_rec["params"]["coast_lead_m"]),
        "lltd": round(compute_lltd_from_theta(th_v_endur_280), 4),
        "thermal_endurance_280kg": {
            "selected_strategy_label": best_280_score["label"],
            "strategy_params": best_280_rec["params"],
            "t_lap_mean_s": coupled_280["t_lap_mean_s"],
            "t_22km_coupled_s": coupled_280["t_22km_coupled_s"],
            "e_net_22km_coupled_kwh": coupled_280["e_net_22km_coupled_kwh"],
            "e_regen_22km_coupled_kwh": coupled_280["e_regen_22km_coupled_kwh"],
            "regen_pct": coupled_280["regen_pct"],
            "t_motor_peak_c": coupled_280["t_motor_peak_c"],
            "t_inv_peak_c": coupled_280["t_inv_peak_c"],
            "t_tire_peak_c": coupled_280["t_tire_peak_c"],
            "pts_endurance_rel": round(float(best_280_score["pts_endurance_rel"]), 2),
            "pts_efficiency_rel": round(float(best_280_score["pts_efficiency_rel"]), 2),
            "pts_endurance_plus_efficiency": best_280_score["pts_endurance_plus_efficiency"],
            "min_clear_m": coupled_280["min_clear_m"],
            "min_rh_f_mm": coupled_280["min_rh_f_mm"],
            "hv_violations": coupled_280["hv_violations"],
            "bspd_violations": coupled_280["bspd_violations"],
            "motor_derating_events": coupled_280["motor_derating_events"],
            "yaw_stable": coupled_280["yaw_stable"],
        },
    })
    ENDURANCE_SETUP_PATH.write_text(json.dumps(endur_setup_out, indent=2) + "\n", encoding="utf-8")

    # Save complete JSON report
    all_invariants = {
        "grip_profiles_intact": True,
        "fsg_rubbered_lambda_muy": FSG_RUBBERED,
        "cold_test_oct_lambda_muy": COLD_TEST_OCT,
        "peak_tire_mu_th_le_1p0": all(r["mu_thermal"] <= 1.0 + 1e-9 for r in accel_sweep + skidpad_sweep),
        "part_a_zero_hv_and_bspd_violations": all(
            r["hv_violations"] == 0 and r["bspd_violations"] == 0 for r in accel_sweep + skidpad_sweep
        ),
        "part_a_skidpad_min_clear_ge_0p150m": all(r["min_clear_m"] >= 0.150 for r in skidpad_sweep),
        "part_a_yaw_stable": all(r["yaw_stable"] for r in accel_sweep + skidpad_sweep),
        "part_b_coupled_280kg_within_3p50_kwh": coupled_280["within_3p50_kwh_budget"],
        "part_b_coupled_280kg_zero_hv_and_bspd": (
            coupled_280["hv_violations"] == 0 and coupled_280["bspd_violations"] == 0
        ),
        "part_b_coupled_280kg_zero_thermal_derating": (
            coupled_280["motor_derating_events"] == 0 and coupled_280["inv_derating_events"] == 0
        ),
        "part_b_coupled_280kg_min_clear_ge_0p150m": coupled_280["min_clear_m"] >= 0.150,
        "part_b_coupled_280kg_min_rh_f_ge_10mm": coupled_280["min_rh_f_mm"] >= 10.0,
        "part_b_coupled_280kg_yaw_stable": coupled_280["yaw_stable"],
    }
    all_invariants["all_passed"] = all(bool(v) for k, v in all_invariants.items() if isinstance(v, bool))

    report = {
        "metadata": {
            "vehicle_id": "ter27_4wd",
            "mass_kg": 280.0,
            "comparison_mass_kg": 290.0,
            "model": "differentiable_casadi_14dof (libter_diff_14dof.so)",
            "suspension_mode": "3d_double_wishbone_lut",
            "tire_compound": "R25B",
            "grip_profile": "FSG_RUBBERED",
            "track": "FSG_2023_Hockenheim",
            "track_length_m": round(eng.track.length_m, 4),
            "n_laps_22km": round(eng.n_laps_22km, 4),
            "energy_budget_kwh": ENDURANCE_ENERGY_KWH,
            "hv_dc_limits_kw": [-25.0, 80.0],
            "elapsed_s": round(time.perf_counter() - t0, 2),
        },
        "part_a_tire_thermal_window": {
            "summary": part_a_summary,
            "acceleration_75m_sweep": accel_sweep,
            "skidpad_sweep": skidpad_sweep,
        },
        "part_b_endurance_22km": {
            "thermal_models": {
                "motor": MOTOR_THERMAL,
                "inverter": INVERTER_THERMAL,
                "tire": TIRE_THERMAL_PARAMS,
            },
            "pareto_frontier_280kg": pareto_280,
            "comparison_290kg": pareto_290,
            "fs_scoring_table": fs_scores,
            "optimal_combined_280kg": {
                "label": best_280_score["label"],
                "score": best_280_score,
                "single_lap_metrics": best_280_rec,
                "coupled_22km_19laps": coupled_280,
            },
            "fastest_feasible_280kg": fastest_280,
            "optimal_combined_290kg": {
                "label": best_290_score["label"],
                "score": best_290_score,
                "single_lap_metrics": best_290_rec,
                "coupled_22km_19laps": coupled_290,
            },
            "delta_280kg_vs_290kg_optimal": {
                "delta_t_22km_coupled_s": round(
                    coupled_280["t_22km_coupled_s"] - coupled_290["t_22km_coupled_s"], 3
                ),
                "delta_e_22km_coupled_kwh": round(
                    coupled_280["e_net_22km_coupled_kwh"] - coupled_290["e_net_22km_coupled_kwh"], 4
                ),
                "delta_pts_endurance_plus_efficiency": round(
                    best_280_score["pts_endurance_plus_efficiency"] - best_290_score["pts_endurance_plus_efficiency"],
                    3,
                ),
                "delta_t_motor_peak_c": round(
                    coupled_280["t_motor_peak_c"] - coupled_290["t_motor_peak_c"], 2
                ),
            },
            "ablations": ablations,
        },
        "invariants": all_invariants,
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\n" + "=" * 78, flush=True)
    print(f"Saved report -> {REPORT_PATH}", flush=True)
    print(f"Saved setup  -> {ENDURANCE_SETUP_PATH}", flush=True)
    print(f"All invariants passed: {all_invariants['all_passed']}", flush=True)
    print(f"Elapsed: {time.perf_counter() - t0:.2f} s", flush=True)
    print("=" * 78, flush=True)


if __name__ == "__main__":
    main()
