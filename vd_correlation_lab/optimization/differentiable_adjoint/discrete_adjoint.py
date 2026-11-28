#!/usr/bin/env python3
"""
Discrete Reverse-Mode Adjoint Sensitivity Engine for TeR27 (4WD).

Computes exact reverse-mode adjoint gradients dJ/dtheta across full closed-loop
14-DOF + Differentiable 4WD Controller horizons for:
  1. Skidpad (FSG circle, R_inner=7.625 m, R_outer=10.625 m, min_clear >= 0.150 m)
  2. Acceleration (75 m full-power launch under 80.0 kW HV DC ceiling)
  3. Autocross (multi-corner transient sector: braking hairpin + S-chicane + exit)

Uses the compiled C library outputs/casadi_cache/libter_diff_14dof.so
(step_dynamics, step_jacobians, output_jacobians) coupled to the C2-smooth
CasADi 4WD controller (Diff4WDController) via exact reverse-mode vector-Jacobian
products (VJP). Verifies adjoint gradients against central finite differences
to < 1e-6.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import casadi as ca
import numpy as np

from vd_correlation_lab.models.differentiable_casadi import (
    COLD_TEST_OCT,
    FSG_RUBBERED,
    NX,
    NU,
    NTHETA,
    NY,
    OUTPUT_IDX,
    PARAM_IDX,
    PARAM_NAMES,
    STATE_IDX,
    DifferentiableVehicle14Dof,
    load_compiled_14dof_library,
)
from vd_correlation_lab.optimization.differentiable_adjoint.diff_4wd_controller import (
    CTRL_PARAM_NAMES,
    Diff4WDController,
    ctrl_dict_to_vec,
    sabs,
    sclip,
    smax,
    spos,
    ssigmoid,
)

# Official FSG Skidpad geometry
R_INNER_M: float = 7.625
R_OUTER_M: float = 10.625
R_CENTER_M: float = 9.125
HALF_TRACK_M: float = 0.615
MIN_CONE_CLEAR_M: float = 0.150

# Augmented controller/progress state dim:
# [0:wz_lpf, 1:ay_smith, 2:mz_int, 3:r_err_prev, 4:s_prog, 5:psi_ref, 6:delta_prev]
AUG_C_DIM: int = 7

# Physical characteristic scale per parameter for normalized sensitivity ranking
# (change in lap time per typical physical engineering delta)
PARAM_CHAR_DELTA: Dict[str, float] = {
    "mass": 5.0,
    "i_xx": 1.0,
    "i_yy": 5.0,
    "i_zz": 5.0,
    "h_cg": 0.010,
    "front_weight_dist": 0.02,
    "k_spring_f": 5000.0,
    "k_spring_r": 5000.0,
    "k_arb_f": 1000.0,
    "k_arb_r": 1000.0,
    "c_damp_f_scale": 0.10,
    "c_damp_r_scale": 0.10,
    "h_rc_f": 0.010,
    "h_rc_r": 0.010,
    "anti_dive_f": 0.05,
    "anti_squat_r": 0.05,
    "camber_f_rad": math.radians(0.5),
    "camber_r_rad": math.radians(0.5),
    "ackermann_factor": 0.10,
    "cl_a": 0.30,
    "cd_a": 0.10,
    "aero_balance_front": 0.02,
    "lambda_mux": 0.02,
    "lambda_muy": 0.02,
    "kya_scale": 0.05,
    "sigma_y": 0.03,
    "sigma_x": 0.03,
    "k_steer_comp": 1000.0,
    # Control parameters
    "k_us_deg_g": 0.10,
    "k_ff": 0.10,
    "yaw_agility_gain": 0.10,
    "ki_yaw": 40.0,
    "kd_yaw": 10.0,
    "k_beta_scale": 0.10,
    "k_beta_dot_scale": 0.10,
    "beta_soft_damp_gain": 200.0,
    "skidpad_mz_ff_gain": 5.0,
    "fz_shed_gain": 0.10,
    "alqp_w_mz": 0.50,
    "alqp_w_fx": 0.25,
    "alqp_w_reg": 0.05,
    "aero_pitch_brake_gain": 0.05,
    "aero_pitch_accel_gain": 0.05,
    "wz_lead_t_s": 0.005,
    "ay_smith_alpha": 0.10,
}

PARAM_CATEGORIES: Dict[str, str] = {
    "mass": "mass_inertia",
    "i_xx": "mass_inertia",
    "i_yy": "mass_inertia",
    "i_zz": "mass_inertia",
    "h_cg": "mass_inertia",
    "front_weight_dist": "mass_inertia",
    "k_spring_f": "springs_arb",
    "k_spring_r": "springs_arb",
    "k_arb_f": "springs_arb",
    "k_arb_r": "springs_arb",
    "c_damp_f_scale": "asymmetric_damping",
    "c_damp_r_scale": "asymmetric_damping",
    "h_rc_f": "suspension_geometry",
    "h_rc_r": "suspension_geometry",
    "anti_dive_f": "suspension_geometry",
    "anti_squat_r": "suspension_geometry",
    "camber_f_rad": "suspension_geometry",
    "camber_r_rad": "suspension_geometry",
    "ackermann_factor": "ackermann_steering",
    "k_steer_comp": "ackermann_steering",
    "cl_a": "aero_ride_height",
    "cd_a": "aero_ride_height",
    "aero_balance_front": "aero_ride_height",
    "lambda_mux": "tire_grip_fixed",
    "lambda_muy": "tire_grip_fixed",
    "kya_scale": "tire_carcass",
    "sigma_y": "tire_carcass",
    "sigma_x": "tire_carcass",
    "k_us_deg_g": "control_gains",
    "k_ff": "control_gains",
    "yaw_agility_gain": "control_gains",
    "ki_yaw": "control_gains",
    "kd_yaw": "control_gains",
    "k_beta_scale": "control_gains",
    "k_beta_dot_scale": "control_gains",
    "beta_soft_damp_gain": "control_gains",
    "skidpad_mz_ff_gain": "control_gains",
    "fz_shed_gain": "alqp_weights",
    "alqp_w_mz": "alqp_weights",
    "alqp_w_fx": "alqp_weights",
    "alqp_w_reg": "alqp_weights",
    "aero_pitch_brake_gain": "aero_pitch_control",
    "aero_pitch_accel_gain": "aero_pitch_control",
    "wz_lead_t_s": "latency_predictor",
    "ay_smith_alpha": "latency_predictor",
}


@dataclass
class EventSpec:
    name: str
    n_steps: int
    dt: float
    x0: np.ndarray
    c0: np.ndarray
    y0: np.ndarray
    w_vec: np.ndarray


class DiscreteAdjointEngine:
    """
    Full-horizon discrete reverse-mode adjoint engine for TeR27 4WD across
    Skidpad, Acceleration (75 m), and Autocross.
    """

    def __init__(
        self,
        grip_profile: str = "FSG_RUBBERED",
        tire_compound: str = "R25B",
        suspension_mode: str = "3d_double_wishbone_lut",
        dt: float = 0.005,
    ) -> None:
        self.grip_profile = grip_profile
        self.tire_compound = tire_compound
        self.suspension_mode = suspension_mode
        self.dt = float(dt)

        self.model = DifferentiableVehicle14Dof(
            vehicle_id="ter27_4wd",
            suspension_mode=suspension_mode,
            tire_compound=tire_compound,
            use_pinn=True,
            grip_profile=grip_profile,
            dt=self.dt,
            rk4_substeps=2,
        )
        self.lib = load_compiled_14dof_library()
        self.ctrl = Diff4WDController(alqp_iters=8)

        self.theta_veh_nom = self.model.default_theta().copy()
        self.theta_ctrl_nom = ctrl_dict_to_vec()
        t_tire = 75.0 if grip_profile == "FSG_RUBBERED" else 35.0
        self.w_nom = self.model.default_disturbance(t_tire_c=t_tire, t_motor_c=60.0)

        # Build symbolic closed-loop step, stage-cost, and terminal-cost graphs
        self._event_graphs: Dict[str, Dict[str, ca.Function]] = {}
        for ev in ("skidpad", "acceleration", "autocross"):
            self._event_graphs[ev] = self._build_event_graphs(ev)

    def _make_event_spec(
        self,
        event: str,
        n_steps: Optional[int] = None,
    ) -> EventSpec:
        dt = self.dt
        w = self.w_nom.copy()
        if event == "skidpad":
            n = int(n_steps) if n_steps is not None else 480  # 2.40 s
            r_tgt = 8.85
            vx0 = 11.4
            wz0 = vx0 / r_tgt
            x0 = self.model.default_state(vx=vx0, vy=0.0, wz=wz0)
            x0[STATE_IDX["X"]] = 0.0
            x0[STATE_IDX["Y"]] = R_CENTER_M - r_tgt  # 0.275 m -> distance r_tgt from (0, 9.125)
            x0[STATE_IDX["psi"]] = 0.0
            delta0 = math.atan(self.ctrl.wb / r_tgt)
            u0 = np.array([delta0, 35.0, 35.0, 45.0, 45.0, 0.0, 0.0, 0.0, 0.0])
            _, y0 = self.lib.step(x0, u0, self.theta_veh_nom, w)
            c0 = np.array([wz0, vx0 * wz0, 0.0, 0.0, 0.0, 0.0, delta0], dtype=float)
            return EventSpec("skidpad", n, dt, x0, c0, y0, w)

        if event == "acceleration":
            n = int(n_steps) if n_steps is not None else 760  # 3.80 s (~75 m)
            vx0 = 1.5
            x0 = self.model.default_state(vx=vx0, vy=0.0, wz=0.0)
            u0 = np.array([0.0, 40.0, 40.0, 50.0, 50.0, 0.0, 0.0, 0.0, 0.0])
            _, y0 = self.lib.step(x0, u0, self.theta_veh_nom, w)
            c0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
            return EventSpec("acceleration", n, dt, x0, c0, y0, w)

        if event == "autocross":
            n = int(n_steps) if n_steps is not None else 700  # 3.50 s (~60 m sector)
            vx0 = 17.5
            x0 = self.model.default_state(vx=vx0, vy=0.0, wz=0.0)
            u0 = np.array([0.0, 60.0, 60.0, 70.0, 70.0, 0.0, 0.0, 0.0, 0.0])
            _, y0 = self.lib.step(x0, u0, self.theta_veh_nom, w)
            c0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
            return EventSpec("autocross", n, dt, x0, c0, y0, w)

        raise ValueError(f"Unknown event: {event}")

    def _build_event_graphs(self, event: str) -> Dict[str, ca.Function]:
        """Build CasADi SX functions + exact reverse-mode VJPs for an event."""
        dt = self.dt
        x = ca.SX.sym("x", NX)
        y_prev = ca.SX.sym("y_prev", NY)
        c = ca.SX.sym("c", AUG_C_DIM)
        t_k = ca.SX.sym("t_k")
        th_v = ca.SX.sym("th_v", NTHETA)
        th_c = ca.SX.sym("th_c", len(CTRL_PARAM_NAMES))

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

        ctrl_st = c[0:4]
        s_prog = c[4]
        psi_ref = c[5]
        delta_prev = c[6]

        # Effective grip estimate from vehicle lambda_muy (proportional to physical tire)
        mu_eff = 2.25 * th_v[PARAM_IDX["lambda_muy"]]

        # Event-specific smooth closed-loop driver & progress update
        if event == "skidpad":
            cx = 0.0
            cy = R_CENTER_M
            r_tgt = 8.85
            dx_c = X_pos - cx
            dy_c = Y_pos - cy
            r_now = ca.sqrt(dx_c * dx_c + dy_c * dy_c + 1e-6)
            # Tangent angle for CCW (left) circle around (cx, cy)
            theta_c = ca.atan2(dy_c, dx_c)
            psi_tan = theta_c + 0.5 * math.pi
            # Smooth wrapped heading error
            e_psi = ca.atan2(ca.sin(psi_tan - psi), ca.cos(psi_tan - psi))
            # Positive e_r means car is outside target radius -> steer more left (+)
            e_r = r_now - r_tgt
            delta_ff = ca.atan(self.ctrl.wb / r_tgt)
            delta_cmd = sclip(delta_ff + 0.35 * e_r + 1.10 * e_psi, -0.36, 0.36, 0.01)

            # Smooth grip-aware speed target on circle
            vx_tgt = 12.6
            # Back off smoothly if drifting wide of target circle
            v_err = vx_tgt - vx - 3.5 * spos(e_r - 0.05, 0.05)
            fx_req = sclip(1250.0 * v_err, 40.0, 2600.0, 25.0)
            t_brk_4 = ca.DM.zeros(4, 1)

            # Exact angular progress increment along circle
            dtheta_step = (vx * ca.cos(e_psi) / smax(r_now, 4.0, 0.2)) * dt
            s_next = s_prog + dtheta_step
            psi_ref_next = psi_tan

        elif event == "acceleration":
            # Straight-line stabilization
            delta_cmd = sclip(-0.14 * Y_pos - 0.45 * psi, -0.18, 0.18, 0.005)
            # Smooth launch ramp to full 4WD power
            ramp = 1.0 - ca.exp(-t_k / 0.08)
            fx_req = 4400.0 * ramp
            t_brk_4 = ca.DM.zeros(4, 1)
            s_next = s_prog + vx * dt
            psi_ref_next = 0.0

        else:  # autocross
            # Smooth C^inf multi-corner Autocross curvature profile kappa(s):
            # Hairpin left (+0.095 1/m @ s=22m) -> right sweeper (-0.072 1/m @ s=44m) -> left exit (+0.055 1/m @ s=64m)
            k1 = 0.095 * ca.exp(-((s_prog - 22.0) / 8.5) ** 2)
            k2 = -0.072 * ca.exp(-((s_prog - 44.0) / 8.0) ** 2)
            k3 = 0.055 * ca.exp(-((s_prog - 64.0) / 8.0) ** 2)
            kappa_s = k1 + k2 + k3

            # Lookahead curvature for smooth braking anticipation (8 m ahead)
            s_la = s_prog + 8.0
            k1_la = 0.095 * ca.exp(-((s_la - 22.0) / 8.5) ** 2)
            k2_la = -0.072 * ca.exp(-((s_la - 44.0) / 8.0) ** 2)
            k3_la = 0.055 * ca.exp(-((s_la - 64.0) / 8.0) ** 2)
            kappa_la = k1_la + k2_la + k3_la

            e_psi = ca.atan2(ca.sin(psi_ref - psi), ca.cos(psi_ref - psi))
            delta_ff = ca.atan(self.ctrl.wb * kappa_s)
            delta_cmd = sclip(delta_ff + 0.65 * e_psi, -0.36, 0.36, 0.01)

            # Smooth cornering speed target based on max(|kappa|, |kappa_la|)
            k_eff = smax(sabs(kappa_s, 0.004), sabs(kappa_la, 0.004), 0.004)
            a_lat_tgt = 15.5 * (th_v[PARAM_IDX["lambda_muy"]] / FSG_RUBBERED)
            v_tgt = sclip(ca.sqrt(a_lat_tgt / smax(k_eff, 0.022, 0.004)), 10.5, 25.0, 0.4)

            fx_req = sclip(1400.0 * (v_tgt - vx), -3200.0, 3800.0, 40.0)
            # Hydraulic brake supplements regen smoothly during heavy braking (fx_req < -1000 N)
            f_hyd_tot = spos(-fx_req - 900.0, 60.0)
            t_hyd_f = 0.32 * f_hyd_tot * self.ctrl.r_eff
            t_hyd_r = 0.18 * f_hyd_tot * self.ctrl.r_eff
            t_brk_4 = ca.vertcat(t_hyd_f, t_hyd_f, t_hyd_r, t_hyd_r)

            s_next = s_prog + vx * ca.cos(e_psi) * dt
            psi_ref_next = psi_ref + kappa_s * vx * dt

        delta_dot = sclip((delta_cmd - delta_prev) / dt, -10.0, 10.0, 0.2)

        ctrl_out = self.ctrl.step_symbolic(
            vx=vx,
            vy=vy,
            wz=wz,
            ay_meas=ay_prev,
            delta=delta_cmd,
            delta_dot=delta_dot,
            fx_req=fx_req,
            fz_4=fz_prev,
            fy_4=fy_prev,
            omega_4=omega_4,
            mu_eff=mu_eff,
            ctrl_state=ctrl_st,
            theta_ctrl=th_c,
            dt=dt,
        )

        u_k = ca.vertcat(delta_cmd, ctrl_out["t_wheel_4"], t_brk_4)
        c_next = ca.vertcat(ctrl_out["ctrl_state_next"], s_next, psi_ref_next, delta_cmd)
        aux = ca.vertcat(
            ctrl_out["p_dc_kw"],
            ctrl_out["dmz_dbeta_task"],
            ctrl_out["aero_pitch_delta_sf"],
            fx_req,
        )

        fn_ctrl = ca.Function(
            f"cl_ctrl_{event}",
            [x, y_prev, c, t_k, th_v, th_c],
            [u_k, c_next, aux],
        )

        # Reverse-mode VJP of fn_ctrl wrt (x, y_prev, c, th_v, th_c)
        adj_u = ca.SX.sym("adj_u", NU)
        adj_c = ca.SX.sym("adj_c", AUG_C_DIM)
        adj_aux = ca.SX.sym("adj_aux", 4)
        out_stack = ca.vertcat(u_k, c_next, aux)
        adj_stack = ca.vertcat(adj_u, adj_c, adj_aux)
        in_stack = ca.vertcat(x, y_prev, c, th_v, th_c)
        vjp_in = ca.jtimes(out_stack, in_stack, adj_stack, True)

        i0 = 0
        i1 = NX
        i2 = i1 + NY
        i3 = i2 + AUG_C_DIM
        i4 = i3 + NTHETA
        i5 = i4 + len(CTRL_PARAM_NAMES)

        fn_ctrl_vjp = ca.Function(
            f"cl_ctrl_vjp_{event}",
            [x, y_prev, c, t_k, th_v, th_c, adj_u, adj_c, adj_aux],
            [vjp_in[i0:i1], vjp_in[i1:i2], vjp_in[i2:i3], vjp_in[i3:i4], vjp_in[i4:i5]],
        )

        # Stage cost l_k(x, u, y, aux) and exact gradients
        u_sym = ca.SX.sym("u_sym", NU)
        y_sym = ca.SX.sym("y_sym", NY)
        aux_sym = ca.SX.sym("aux_sym", 4)
        p_dc_y = y_sym[OUTPUT_IDX["P_dc_kw"]]
        beta_y = y_sym[OUTPUT_IDX["beta"]]
        rh_f_y = y_sym[OUTPUT_IDX["rh_f"]]

        # Common safety barrier penalties (strictly smooth)
        pen_hv = (
            0.05 * spos(p_dc_y - 79.2, 0.2) ** 2
            + 0.05 * spos(-p_dc_y - 24.2, 0.2) ** 2
        )
        pen_beta = 2.0 * spos(sabs(beta_y, 0.005) - math.radians(5.5), 0.005) ** 2
        pen_stall = 50.0 * spos(0.0185 - rh_f_y, 0.001) ** 2

        if event == "skidpad":
            r_now_s = ca.sqrt(x[STATE_IDX["X"]] ** 2 + (x[STATE_IDX["Y"]] - R_CENTER_M) ** 2 + 1e-6)
            clear_in = r_now_s - (R_INNER_M + HALF_TRACK_M)
            clear_out = (R_OUTER_M - HALF_TRACK_M) - r_now_s
            pen_cone = (
                20.0 * spos(0.18 - clear_in, 0.015) ** 2
                + 20.0 * spos(0.18 - clear_out, 0.015) ** 2
                + 0.004 * (r_now_s - 8.70) ** 2
            )
            stage_l = (pen_hv + pen_beta + pen_stall + pen_cone) * dt
        elif event == "acceleration":
            pen_lat = 0.08 * (x[STATE_IDX["Y"]] ** 2) + 0.15 * (x[STATE_IDX["psi"]] ** 2)
            stage_l = (pen_hv + pen_beta + pen_stall + pen_lat) * dt
        else:  # autocross
            pen_trk = 0.002 * (y_sym[OUTPUT_IDX["ay"]] ** 2 * 1e-4)
            stage_l = (pen_hv + pen_beta + pen_stall + pen_trk) * dt

        fn_stage = ca.Function(
            f"stage_{event}",
            [x, u_sym, y_sym, aux_sym],
            [
                stage_l,
                ca.gradient(stage_l, x),
                ca.gradient(stage_l, u_sym),
                ca.gradient(stage_l, y_sym),
                ca.gradient(stage_l, aux_sym),
            ],
        )

        # Terminal cost Phi(x_N, c_N, T_sim) -> (J_term, t_event_s, dPhi/dx_N, dPhi/dc_N)
        T_sim = ca.SX.sym("T_sim")
        if event == "skidpad":
            # Full circle time = 2*pi * T_sim / delta_theta_total
            t_ev = (2.0 * math.pi * T_sim) / smax(c[4], 0.4, 0.05)
            phi_term = t_ev
        elif event == "acceleration":
            # Smooth 75 m finish time: T_sim + (75 - X_N) / vx_N
            t_ev = T_sim + (75.0 - x[STATE_IDX["X"]]) / smax(x[STATE_IDX["vx"]], 5.0, 0.5)
            phi_term = t_ev
        else:  # autocross
            # Smooth 75 m Autocross sector time
            t_ev = T_sim * (75.0 / smax(c[4], 8.0, 0.5))
            phi_term = t_ev

        fn_term = ca.Function(
            f"term_{event}",
            [x, c, T_sim],
            [phi_term, t_ev, ca.gradient(phi_term, x), ca.gradient(phi_term, c)],
        )

        return {
            "ctrl": fn_ctrl,
            "ctrl_vjp": fn_ctrl_vjp,
            "stage": fn_stage,
            "term": fn_term,
        }

    def rollout_and_adjoint(
        self,
        event: str,
        theta_veh: Optional[np.ndarray] = None,
        theta_ctrl: Optional[np.ndarray] = None,
        n_steps: Optional[int] = None,
        compute_grad: bool = True,
    ) -> Dict[str, Any]:
        """
        Run full-horizon closed-loop forward rollout and (optionally) exact
        discrete reverse-mode adjoint pass for `event`.
        """
        th_v = self.theta_veh_nom.copy() if theta_veh is None else np.asarray(theta_veh, dtype=np.float64)
        th_c = self.theta_ctrl_nom.copy() if theta_ctrl is None else np.asarray(theta_ctrl, dtype=np.float64)
        spec = self._make_event_spec(event, n_steps=n_steps)
        g = self._event_graphs[event]

        N = spec.n_steps
        dt = spec.dt
        T_sim = N * dt
        w = spec.w_vec

        # Trajectory buffers for reverse pass
        X_hist = np.zeros((N + 1, NX), dtype=np.float64)
        C_hist = np.zeros((N + 1, AUG_C_DIM), dtype=np.float64)
        Y_hist = np.zeros((N + 1, NY), dtype=np.float64)
        U_hist = np.zeros((N, NU), dtype=np.float64)
        AUX_hist = np.zeros((N, 4), dtype=np.float64)

        X_hist[0] = spec.x0
        C_hist[0] = spec.c0
        Y_hist[0] = spec.y0

        j_stage_tot = 0.0
        p_dc_min = 1e9
        p_dc_max = -1e9
        hv_violations = 0
        bspd_violations = 0
        min_clear_m = 999.0
        max_dmz_dbeta = -1e9
        min_rh_f_m = 999.0
        max_ay_g = 0.0
        max_vx_ms = 0.0

        # ---- Forward pass ----
        for k in range(N):
            t_k = k * dt
            u_k, c_next, aux_k = g["ctrl"](X_hist[k], Y_hist[k], C_hist[k], t_k, th_v, th_c)
            u_np = np.asarray(u_k, dtype=np.float64).ravel()
            c_np = np.asarray(c_next, dtype=np.float64).ravel()
            aux_np = np.asarray(aux_k, dtype=np.float64).ravel()

            x_next, y_k = self.lib.step(X_hist[k], u_np, th_v, w)

            U_hist[k] = u_np
            C_hist[k + 1] = c_np
            X_hist[k + 1] = x_next
            Y_hist[k + 1] = y_k
            AUX_hist[k] = aux_np

            l_k = float(g["stage"](X_hist[k], u_np, y_k, aux_np)[0])
            j_stage_tot += l_k

            # Telemetry & constraint checks
            p_dc_k = float(y_k[OUTPUT_IDX["P_dc_kw"]])
            p_dc_min = min(p_dc_min, p_dc_k)
            p_dc_max = max(p_dc_max, p_dc_k)
            if p_dc_k > 80.0 or p_dc_k < -25.0:
                hv_violations += 1

            # BSPD EV 2.3: when braking (fx_req < -150 N), forbid positive drive torque
            fx_req_k = float(aux_np[3])
            if fx_req_k < -150.0 and float(np.max(u_np[1:5])) > 0.5:
                bspd_violations += 1

            dmz_k = float(aux_np[1])
            max_dmz_dbeta = max(max_dmz_dbeta, dmz_k)
            min_rh_f_m = min(min_rh_f_m, float(y_k[OUTPUT_IDX["rh_f"]]))
            max_ay_g = max(max_ay_g, abs(float(y_k[OUTPUT_IDX["ay"]])) / 9.81)
            max_vx_ms = max(max_vx_ms, float(x_next[STATE_IDX["vx"]]))

            if event == "skidpad":
                r_k = math.hypot(float(x_next[STATE_IDX["X"]]), float(x_next[STATE_IDX["Y"]]) - R_CENTER_M)
                c_in = r_k - (R_INNER_M + HALF_TRACK_M)
                c_out = (R_OUTER_M - HALF_TRACK_M) - r_k
                min_clear_m = min(min_clear_m, min(c_in, c_out))

        phi_term, t_event_s, dphi_dxN, dphi_dcN = g["term"](X_hist[N], C_hist[N], T_sim)
        j_total = j_stage_tot + float(phi_term)

        out: Dict[str, Any] = {
            "event": event,
            "n_steps": N,
            "t_sim_s": T_sim,
            "cost": float(j_total),
            "event_time_s": float(t_event_s),
            "stage_penalty": float(j_stage_tot),
            "p_dc_min_kw": float(p_dc_min),
            "p_dc_max_kw": float(p_dc_max),
            "hv_violations": int(hv_violations),
            "bspd_violations": int(bspd_violations),
            "min_clear_m": float(min_clear_m) if event == "skidpad" else None,
            "cones_hit": int(min_clear_m < MIN_CONE_CLEAR_M) if event == "skidpad" else 0,
            "max_dmz_dbeta_nm_rad": float(max_dmz_dbeta),
            "yaw_stable": bool(max_dmz_dbeta < 0.0),
            "min_rh_f_mm": float(min_rh_f_m * 1000.0),
            "max_ay_g": float(max_ay_g),
            "max_vx_ms": float(max_vx_ms),
            "final_progress": float(C_hist[N, 4]),
            "final_x_m": float(X_hist[N, STATE_IDX["X"]]),
        }

        if not compute_grad:
            return out

        # ---- Exact Discrete Reverse-Mode Adjoint Pass (k = N-1 down to 0) ----
        lam_x = np.asarray(dphi_dxN, dtype=np.float64).ravel()
        lam_c = np.asarray(dphi_dcN, dtype=np.float64).ravel()
        lam_y = np.zeros(NY, dtype=np.float64)

        grad_th_v = np.zeros(NTHETA, dtype=np.float64)
        grad_th_c = np.zeros(len(CTRL_PARAM_NAMES), dtype=np.float64)

        for k in range(N - 1, -1, -1):
            t_k = k * dt
            x_k = X_hist[k]
            y_prev_k = Y_hist[k]
            c_k = C_hist[k]
            u_k = U_hist[k]
            y_k = Y_hist[k + 1]
            aux_k = AUX_hist[k]

            # Stage cost partial derivatives
            _, dl_dx, dl_du, dl_dy, dl_daux = g["stage"](x_k, u_k, y_k, aux_k)
            dl_dx_np = np.asarray(dl_dx, dtype=np.float64).ravel()
            dl_du_np = np.asarray(dl_du, dtype=np.float64).ravel()
            dl_dy_np = np.asarray(dl_dy, dtype=np.float64).ravel()
            dl_daux_np = np.asarray(dl_daux, dtype=np.float64).ravel()

            # Compiled 14-DOF vehicle Jacobians from libter_diff_14dof.so
            A_x, B_u, S_theta, _ = self.lib.jacobians(x_k, u_k, th_v, w)
            C_x, D_u, O_theta, _ = self.lib.output_jacobians(x_k, u_k, th_v, w)
            C_x_np = np.asarray(C_x, dtype=np.float64)
            D_u_np = np.asarray(D_u, dtype=np.float64)
            O_theta_np = np.asarray(O_theta, dtype=np.float64)

            g_y_tot = lam_y + dl_dy_np
            g_x_dyn = A_x.T @ lam_x + C_x_np.T @ g_y_tot
            g_u_tot = B_u.T @ lam_x + D_u_np.T @ g_y_tot + dl_du_np
            grad_th_v += S_theta.T @ lam_x + O_theta_np.T @ g_y_tot

            # Reverse-mode VJP through closed-loop driver + Diff4WDController
            gx_c, gyprev_c, gc_c, gthv_c, gthc_c = g["ctrl_vjp"](
                x_k, y_prev_k, c_k, t_k, th_v, th_c, g_u_tot, lam_c, dl_daux_np
            )
            lam_x = g_x_dyn + np.asarray(gx_c, dtype=np.float64).ravel() + dl_dx_np
            lam_y = np.asarray(gyprev_c, dtype=np.float64).ravel()
            lam_c = np.asarray(gc_c, dtype=np.float64).ravel()
            grad_th_v += np.asarray(gthv_c, dtype=np.float64).ravel()
            grad_th_c += np.asarray(gthc_c, dtype=np.float64).ravel()

        out["grad_theta_veh"] = grad_th_v
        out["grad_theta_ctrl"] = grad_th_c
        return out

    def verify_adjoint_vs_fd(
        self,
        event: str = "skidpad",
        n_steps: int = 80,
        fd_rel_eps: float = 1e-6,
    ) -> Dict[str, Any]:
        """
        Verify discrete reverse-mode adjoint gradient against central finite
        differences across representative mechanical and control parameters.
        """
        res = self.rollout_and_adjoint(event, n_steps=n_steps, compute_grad=True)
        g_v_ad = res["grad_theta_veh"]
        g_c_ad = res["grad_theta_ctrl"]

        check_veh = [
            "k_spring_f",
            "k_arb_r",
            "c_damp_f_scale",
            "c_damp_r_scale",
            "camber_f_rad",
            "ackermann_factor",
            "aero_balance_front",
            "h_rc_f",
        ]
        check_ctrl = [
            "k_us_deg_g",
            "k_ff",
            "yaw_agility_gain",
            "alqp_w_mz",
            "alqp_w_reg",
            "aero_pitch_brake_gain",
            "wz_lead_t_s",
        ]

        rows: List[Dict[str, Any]] = []
        max_abs_err = 0.0
        max_rel_err = 0.0

        for name in check_veh:
            idx = PARAM_IDX[name]
            step = fd_rel_eps * max(abs(self.theta_veh_nom[idx]), 1e-3)
            tp = self.theta_veh_nom.copy()
            tm = self.theta_veh_nom.copy()
            tp[idx] += step
            tm[idx] -= step
            cp = self.rollout_and_adjoint(event, theta_veh=tp, n_steps=n_steps, compute_grad=False)["cost"]
            cm = self.rollout_and_adjoint(event, theta_veh=tm, n_steps=n_steps, compute_grad=False)["cost"]
            g_fd = (cp - cm) / (2.0 * step)
            g_ad = float(g_v_ad[idx])
            a_err = abs(g_ad - g_fd)
            r_err = a_err / max(abs(g_fd), 1.0)
            max_abs_err = max(max_abs_err, a_err)
            max_rel_err = max(max_rel_err, r_err)
            rows.append({
                "param": name,
                "group": "vehicle",
                "g_adjoint": g_ad,
                "g_fd": float(g_fd),
                "abs_err": float(a_err),
                "rel_err": float(r_err),
            })

        for name in check_ctrl:
            idx = CTRL_PARAM_NAMES.index(name)
            step = fd_rel_eps * max(abs(self.theta_ctrl_nom[idx]), 1e-3)
            tp = self.theta_ctrl_nom.copy()
            tm = self.theta_ctrl_nom.copy()
            tp[idx] += step
            tm[idx] -= step
            cp = self.rollout_and_adjoint(event, theta_ctrl=tp, n_steps=n_steps, compute_grad=False)["cost"]
            cm = self.rollout_and_adjoint(event, theta_ctrl=tm, n_steps=n_steps, compute_grad=False)["cost"]
            g_fd = (cp - cm) / (2.0 * step)
            g_ad = float(g_c_ad[idx])
            a_err = abs(g_ad - g_fd)
            r_err = a_err / max(abs(g_fd), 1.0)
            max_abs_err = max(max_abs_err, a_err)
            max_rel_err = max(max_rel_err, r_err)
            rows.append({
                "param": name,
                "group": "control",
                "g_adjoint": g_ad,
                "g_fd": float(g_fd),
                "abs_err": float(a_err),
                "rel_err": float(r_err),
            })

        return {
            "event": event,
            "n_steps": n_steps,
            "max_abs_err": float(max_abs_err),
            "max_rel_err": float(max_rel_err),
            "passed_1e_6": bool(max_rel_err < 1e-6 and max_abs_err < 1e-6),
            "checks": rows,
        }

    def compute_sensitivity_rankings(
        self,
        theta_veh: Optional[np.ndarray] = None,
        theta_ctrl: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Compute exact reverse-mode adjoint sensitivities dT/dtheta across all 45
        TeR27 parameters (28 vehicle + 17 control) for Skidpad, Acceleration,
        and Autocross, returning ranked sensitivity tables.
        """
        per_event: Dict[str, Any] = {}
        all_names = list(PARAM_NAMES) + list(CTRL_PARAM_NAMES)
        combined_impact = np.zeros(len(all_names), dtype=np.float64)
        event_weights = {"skidpad": 1.0, "acceleration": 1.0, "autocross": 1.0}

        for ev in ("skidpad", "acceleration", "autocross"):
            t0 = time.perf_counter()
            res = self.rollout_and_adjoint(ev, theta_veh=theta_veh, theta_ctrl=theta_ctrl, compute_grad=True)
            elapsed_s = time.perf_counter() - t0
            g_all = np.concatenate([res["grad_theta_veh"], res["grad_theta_ctrl"]])

            rankings: List[Dict[str, Any]] = []
            for i, name in enumerate(all_names):
                delta_phys = PARAM_CHAR_DELTA.get(name, 1.0)
                dt_per_unit = float(g_all[i])
                dt_ms_per_delta = dt_per_unit * delta_phys * 1000.0
                combined_impact[i] += event_weights[ev] * abs(dt_ms_per_delta)
                rankings.append({
                    "param": name,
                    "category": PARAM_CATEGORIES.get(name, "other"),
                    "dT_dtheta_s_per_unit": dt_per_unit,
                    "char_delta": float(delta_phys),
                    "dT_ms_per_char_delta": float(dt_ms_per_delta),
                    "abs_impact_ms": float(abs(dt_ms_per_delta)),
                })
            rankings.sort(key=lambda r: r["abs_impact_ms"], reverse=True)
            for rank_i, row in enumerate(rankings, start=1):
                row["rank"] = rank_i

            per_event[ev] = {
                "event_time_s": float(res["event_time_s"]),
                "cost": float(res["cost"]),
                "p_dc_min_kw": float(res["p_dc_min_kw"]),
                "p_dc_max_kw": float(res["p_dc_max_kw"]),
                "hv_violations": int(res["hv_violations"]),
                "bspd_violations": int(res["bspd_violations"]),
                "min_clear_m": res["min_clear_m"],
                "cones_hit": int(res["cones_hit"]),
                "max_dmz_dbeta_nm_rad": float(res["max_dmz_dbeta_nm_rad"]),
                "yaw_stable": bool(res["yaw_stable"]),
                "min_rh_f_mm": float(res["min_rh_f_mm"]),
                "max_ay_g": float(res["max_ay_g"]),
                "max_vx_ms": float(res["max_vx_ms"]),
                "adjoint_wall_s": float(elapsed_s),
                "rankings": rankings,
            }

        combined_rankings: List[Dict[str, Any]] = []
        for i, name in enumerate(all_names):
            combined_rankings.append({
                "param": name,
                "category": PARAM_CATEGORIES.get(name, "other"),
                "combined_abs_impact_ms": float(combined_impact[i]),
                "skidpad_dT_ms": float(
                    next(r["dT_ms_per_char_delta"] for r in per_event["skidpad"]["rankings"] if r["param"] == name)
                ),
                "acceleration_dT_ms": float(
                    next(r["dT_ms_per_char_delta"] for r in per_event["acceleration"]["rankings"] if r["param"] == name)
                ),
                "autocross_dT_ms": float(
                    next(r["dT_ms_per_char_delta"] for r in per_event["autocross"]["rankings"] if r["param"] == name)
                ),
            })
        combined_rankings.sort(key=lambda r: r["combined_abs_impact_ms"], reverse=True)
        for rank_i, row in enumerate(combined_rankings, start=1):
            row["rank"] = rank_i

        return {
            "grip_profile": self.grip_profile,
            "cold_test_oct_lambda_muy": COLD_TEST_OCT,
            "fsg_rubbered_lambda_muy": FSG_RUBBERED,
            "events": per_event,
            "combined_ranking": combined_rankings,
        }


if __name__ == "__main__":
    eng = DiscreteAdjointEngine()
    print("=== Adjoint vs Central FD Verification (80 steps, all 3 events) ===", flush=True)
    for ev in ("skidpad", "acceleration", "autocross"):
        chk = eng.verify_adjoint_vs_fd(event=ev, n_steps=80)
        print(
            f"[{ev:12s}] max_abs_err={chk['max_abs_err']:.3e} | "
            f"max_rel_err={chk['max_rel_err']:.3e} | passed={chk['passed_1e_6']}",
            flush=True,
        )
