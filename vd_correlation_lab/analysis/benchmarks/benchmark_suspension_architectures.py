#!/usr/bin/env python3
"""
Controlled Physics Benchmark of the 4 Orthogonal Suspension Architectures
on the Differentiable CasADi 14-DOF Vehicle Dynamics Model (TeR27 4WD).

Compares all 4 orthogonal suspension modes under identical base mass (290.0 kg),
Hoosier 16.0x7.5-10 R25B MF6.2+PINN tires (FSG_RUBBERED = 0.57, COLD_TEST_OCT = 0.47
preserved intact), aero map (ClA = 4.68 m^2, CdA = 1.50 m^2, CoP = 48.0% front,
stall_rh_f = 18.0 mm), and identical Diff4WDController (AL-QP + aero-pitch +
[-25, +80] kW HV DC clamp):
  1. analytic_1d_lut          (1D camber/toe LUT, constant MR, digressive dampers, ARB)
  2. 3d_double_wishbone_lut   (3D progressive MR/RC, digressive dampers, ARB)
  3. decoupled_heave_roll     (3rd-element decoupled heave + roll, nominal rates)
  4. standard_pushrod         (corner coilover, linear asymmetric dampers, U-bar ARB)
Plus:
  5. decoupled_heave_roll_tuned (3rd-element heave stiffened 2.0x with compliant roll)

Evaluates:
  - Official FSG Skidpad (R_inner=7.625 m, R_outer=10.625 m, min_clear >= 0.150 m)
  - High-downforce / threshold-braking / slalom Autocross circuit (200 m, with
    realistic tarmac micro-undulations and apex kerb bumps)
  - Anti-dive (56.9%) / Anti-squat (41.2%) ablation vs 0% anti-geometry
  - Hardware mass penalty sweep (+0.0 to +4.0 kg) to determine the exact net
    lap-time break-even mass budget for decoupled_heave_roll.

Exports complete comparative tables, ride-height/CoP time-series, and mass
break-even curves to:
  outputs/differentiable_deep_validation/suspension_architecture_benchmark_report.json
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import casadi as ca
import numpy as np

from vd_correlation_lab.models.differentiable_casadi import (
    COLD_TEST_OCT,
    FSG_RUBBERED,
    NU,
    NX,
    NY,
    NTHETA,
    OUTPUT_IDX,
    PARAM_IDX,
    STATE_IDX,
    DifferentiableVehicle14Dof,
)
from vd_correlation_lab.models.registry import (
    MODEL_DIFFERENTIABLE_CASADI_14DOF,
    SUSPENSION_MODES,
    create_vehicle_model,
)
from vd_correlation_lab.models.vehicle_14dof.ter27_physics import (
    rake_to_aero_modifiers,
)
from vd_correlation_lab.optimization.differentiable_adjoint.diff_4wd_controller import (
    CTRL_PARAM_NAMES,
    Diff4WDController,
    ctrl_dict_to_vec,
    sabs,
    sclip,
    smax,
    spos,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPORT_PATH = (
    REPO_ROOT
    / "outputs"
    / "differentiable_deep_validation"
    / "suspension_architecture_benchmark_report.json"
)

# Official FSG Skidpad geometry
R_INNER_M: float = 7.625
R_OUTER_M: float = 10.625
R_CENTER_M: float = 9.125
HALF_TRACK_M: float = 0.615
MIN_CONE_CLEAR_M: float = 0.150

# Autocross circuit length for exact interpolated lap/sector timing
AUTOCROSS_TARGET_S_M: float = 200.0

# Optimal outside-wheel camber into the turn for Hoosier 16.0x7.5-10 R25B [deg]
OPT_OUTSIDE_CAMBER_DEG: float = 1.80


def _build_kinematics_evaluator(model: DifferentiableVehicle14Dof) -> ca.Function:
    """Build a fast CasADi SX evaluator for 4-wheel ground camber [rad] and MR."""
    x = ca.SX.sym("x", NX)
    u = ca.SX.sym("u", NU)
    theta = ca.SX.sym("theta", NTHETA)

    c = model.cfg
    L = c.wheelbase
    tf = c.track_front
    tr = c.track_rear
    fwd = sclip(theta[PARAM_IDX["front_weight_dist"]], 0.25, 0.75, 1e-4)
    a_cog = L * (1.0 - fwd)
    b_cog = L * fwd
    x_w = [a_cog, a_cog, -b_cog, -b_cog]
    y_w = [0.5 * tf, -0.5 * tf, 0.5 * tr, -0.5 * tr]

    z_cg_rel = x[STATE_IDX["Z"]] - theta[PARAM_IDX["h_cg"]]
    phi = x[STATE_IDX["phi"]]
    theta_p = x[STATE_IDX["theta_p"]]
    z_u = x[STATE_IDX["z_u_FL"]:STATE_IDX["z_u_RR"] + 1]

    cp = ca.cos(theta_p)
    sp = ca.sin(theta_p)
    sr = ca.sin(phi)

    dz_list = []
    for i in range(4):
        z_c_i = z_cg_rel + y_w[i] * cp * sr - x_w[i] * sp
        dz_list.append(z_u[i] - z_c_i)
    dz_4 = ca.vertcat(*dz_list)

    kin = model.susp.eval_kinematics_symbolic(
        dz_4=dz_4,
        delta_cmd=u[0],
        phi=phi,
        camber_f_rad=theta[PARAM_IDX["camber_f_rad"]],
        camber_r_rad=theta[PARAM_IDX["camber_r_rad"]],
        ackermann_factor=theta[PARAM_IDX["ackermann_factor"]],
    )
    return ca.Function("eval_kin", [x, u, theta], [kin["gammas"], kin["mrs"], dz_4])


def _build_4wd_controller_fn(
    ctrl: Diff4WDController,
    event: str,
    dt: float,
) -> ca.Function:
    """
    Build a compiled CasADi SX closed-loop driver + Diff4WDController function
    shared identically across all suspension architectures.
    """
    x = ca.SX.sym("x", NX)
    y_prev = ca.SX.sym("y_prev", NY)
    c_aug = ca.SX.sym("c_aug", 7)  # [wz_lpf, ay_smith, mz_int, r_err_prev, s_prog, psi_ref, delta_prev]
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

    ctrl_st = c_aug[0:4]
    s_prog = c_aug[4]
    psi_ref = c_aug[5]
    delta_prev = c_aug[6]

    mu_eff = 2.25 * th_v[PARAM_IDX["lambda_muy"]]

    if event == "skidpad":
        cx = 0.0
        cy = R_CENTER_M
        r_tgt = 8.82
        dx_c = X_pos - cx
        dy_c = Y_pos - cy
        r_now = ca.sqrt(dx_c * dx_c + dy_c * dy_c + 1e-6)
        theta_c = ca.atan2(dy_c, dx_c)
        psi_tan = theta_c + 0.5 * math.pi
        e_psi = ca.atan2(ca.sin(psi_tan - psi), ca.cos(psi_tan - psi))
        e_r = r_now - r_tgt

        delta_ff = ca.atan(ctrl.wb / r_tgt)
        delta_cmd = sclip(delta_ff + 0.42 * e_r + 1.25 * e_psi, -0.36, 0.36, 0.01)

        # Push to true tire/aero steady-state limit while protecting cone clearance
        vx_tgt = 13.6
        v_err = vx_tgt - vx - 4.5 * spos(e_r - 0.04, 0.03)
        fx_req = sclip(1400.0 * v_err, 20.0, 3000.0, 25.0)
        t_brk_4 = ca.DM.zeros(4, 1)

        dtheta_step = (vx * ca.cos(e_psi) / smax(r_now, 4.0, 0.2)) * dt
        s_next = s_prog + dtheta_step
        psi_ref_next = psi_tan

    else:  # autocross (200 m high-downforce straight + threshold braking hairpin + fast slalom + aero sweeper + exit)
        # Curvature profile kappa(s) [1/m]:
        #   s in [0, 48] m   : High-speed acceleration straight -> ~26.5 m/s (peak downforce)
        #   s ~ 68 m         : Heavy threshold braking into tight left hairpin (+0.105 1/m, R~9.5 m)
        #   s ~ 102 m        : Right slalom gate (-0.078 1/m)
        #   s ~ 124 m        : Left slalom gate (+0.078 1/m)
        #   s ~ 156 m        : High-speed right aero sweeper (-0.052 1/m, R~19.2 m)
        #   s in [175, 205] m: Full-power acceleration exit
        def _kappa_at(s_val: Any) -> Any:
            k_hairpin = 0.105 * ca.exp(-((s_val - 68.0) / 9.5) ** 2)
            k_slalom1 = -0.078 * ca.exp(-((s_val - 102.0) / 7.5) ** 2)
            k_slalom2 = 0.078 * ca.exp(-((s_val - 124.0) / 7.5) ** 2)
            k_sweeper = -0.052 * ca.exp(-((s_val - 156.0) / 11.5) ** 2)
            return k_hairpin + k_slalom1 + k_slalom2 + k_sweeper

        kappa_s = _kappa_at(s_prog)
        kappa_la1 = _kappa_at(s_prog + 10.0)
        kappa_la2 = _kappa_at(s_prog + 18.0)

        e_psi = ca.atan2(ca.sin(psi_ref - psi), ca.cos(psi_ref - psi))
        delta_ff = ca.atan(ctrl.wb * kappa_s)
        delta_cmd = sclip(delta_ff + 0.85 * e_psi, -0.36, 0.36, 0.01)

        # Aero-aware limit speed target: scales with instantaneous total normal load / actual weight
        # so architectures that preserve downforce & minimize stall genuinely corner/brake faster,
        # while added mass properly penalizes acceleration, braking, and lateral grip!
        fz_tot = ca.sum1(fz_prev)
        m_veh = smax(th_v[PARAM_IDX["mass"]], 150.0, 1.0)
        aero_grip_boost = sclip(fz_tot / (m_veh * 9.81), 0.90, 1.85, 0.02)
        k_eff = smax(
            sabs(kappa_s, 0.003),
            smax(sabs(kappa_la1, 0.003), 0.72 * sabs(kappa_la2, 0.003), 0.003),
            0.003,
        )
        # Tire load sensitivity: heavier car loses peak friction mu slightly (MF6.2 dFz effect)
        a_lat_base = 16.0 * (th_v[PARAM_IDX["lambda_muy"]] / FSG_RUBBERED) * (290.0 / m_veh) ** 0.35
        a_lat_lim = a_lat_base * (0.65 + 0.35 * aero_grip_boost)
        v_tgt = sclip(ca.sqrt(a_lat_lim / smax(k_eff, 0.015, 0.003)), 10.5, 29.0, 0.4)

        # Back off slightly if heading tracking error grows (true grip limit feedback)
        v_tgt_eff = v_tgt - 4.0 * spos(sabs(e_psi, 0.01) - 0.06, 0.02)
        fx_req = sclip(1650.0 * (v_tgt_eff - vx), -5200.0, 4400.0, 40.0)

        # Hydraulic brakes supplement regen during heavy threshold braking
        f_hyd_tot = spos(-fx_req - 950.0, 60.0)
        t_hyd_f = 0.32 * f_hyd_tot * ctrl.r_eff
        t_hyd_r = 0.18 * f_hyd_tot * ctrl.r_eff
        t_brk_4 = ca.vertcat(t_hyd_f, t_hyd_f, t_hyd_r, t_hyd_r)

        s_next = s_prog + vx * ca.cos(e_psi) * dt
        psi_ref_next = psi_ref + kappa_s * vx * dt

    delta_dot = sclip((delta_cmd - delta_prev) / dt, -12.0, 12.0, 0.2)

    ctrl_out = ctrl.step_symbolic(
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
    return ca.Function(
        f"bench_ctrl_{event}",
        [x, y_prev, c_aug, th_v, th_c],
        [u_k, c_next, aux],
    )


def _road_elevation_4(s_m: float, event: str, wheelbase: float) -> np.ndarray:
    """
    Deterministic realistic track surface profile z_road[4] (FL, FR, RL, RR) [m].
    - Skidpad: clean FSG asphalt with subtle 1.5 mm tarmac micro-undulation.
    - Autocross: realistic airfield tarmac undulations (3-4 mm) plus asymmetric
      apex kerb/patch excitations at hairpin (68 m), slalom (102/124 m), and
      high-speed sweeper (156 m) to excite digressive vs linear damping and
      heave/roll decoupling.
    """
    s_f = float(s_m)
    s_r = max(0.0, s_f - wheelbase)

    if event == "skidpad":
        # Subtle airfield expansion joints / micro-waviness (1.5 mm amplitude)
        zf_l = 0.0015 * math.sin(2.0 * math.pi * s_f / 14.0)
        zf_r = 0.0015 * math.sin(2.0 * math.pi * s_f / 14.0 + 0.6)
        zr_l = 0.0015 * math.sin(2.0 * math.pi * s_r / 14.0)
        zr_r = 0.0015 * math.sin(2.0 * math.pi * s_r / 14.0 + 0.6)
        return np.array([zf_l, zf_r, zr_l, zr_r], dtype=np.float64)

    def _surf_l(s: float) -> float:
        base = 0.0028 * math.sin(2.0 * math.pi * s / 16.0) + 0.0014 * math.sin(2.0 * math.pi * s / 4.2)
        kerb_hp = 0.0075 * math.exp(-((s - 68.0) / 1.8) ** 2)   # Left apex kerb at hairpin
        kerb_s2 = 0.0065 * math.exp(-((s - 124.0) / 1.5) ** 2)  # Left slalom kerb
        return base + kerb_hp + kerb_s2

    def _surf_r(s: float) -> float:
        base = 0.0028 * math.sin(2.0 * math.pi * s / 16.0 + 0.35) + 0.0014 * math.sin(2.0 * math.pi * s / 4.2 + 0.9)
        kerb_s1 = 0.0065 * math.exp(-((s - 102.0) / 1.5) ** 2)  # Right slalom kerb
        kerb_sw = 0.0060 * math.exp(-((s - 156.0) / 2.2) ** 2)  # Right sweeper apex patch
        return base + kerb_s1 + kerb_sw

    return np.array([_surf_l(s_f), _surf_r(s_f), _surf_l(s_r), _surf_r(s_r)], dtype=np.float64)


def _create_arch_model(
    arch_id: str,
    dt: float = 0.005,
) -> Tuple[DifferentiableVehicle14Dof, Dict[str, Any]]:
    """
    Instantiate a DifferentiableVehicle14Dof model for one of the 4 canonical
    suspension modes plus the tuned decoupled 3rd-element variant.
    """
    if arch_id == "decoupled_heave_roll_tuned":
        base_mode = "decoupled_heave_roll"
        heave_k_mult = 2.10
        heave_c_mult = 1.45
        desc = "Decoupled 3rd-element heave (2.1x heave stiffness, 1.45x heave damping) + compliant roll"
    else:
        base_mode = arch_id
        heave_k_mult = 1.0
        heave_c_mult = 1.0
        desc = f"Canonical {arch_id} suspension mode with nominal TeR27 parameters"

    model = create_vehicle_model(
        model_id=MODEL_DIFFERENTIABLE_CASADI_14DOF,
        vehicle_id="ter27_4wd",
        suspension_mode=base_mode,
        grip_profile="FSG_RUBBERED",
        tire_compound="R25B",
        use_pinn=True,
        dt=dt,
        rk4_substeps=2,
        use_thermal_coupling=False,
    )

    if heave_k_mult != 1.0 or heave_c_mult != 1.0:
        model.cfg.susp.k_heave_f *= heave_k_mult
        model.cfg.susp.k_heave_r *= heave_k_mult
        model.cfg.susp.c_heave_f *= heave_c_mult
        model.cfg.susp.c_heave_r *= heave_c_mult
        model._build_casadi_functions()

    meta = {
        "arch_id": arch_id,
        "suspension_mode": base_mode,
        "heave_stiffness_multiplier": heave_k_mult,
        "heave_damping_multiplier": heave_c_mult,
        "k_heave_f_nm": round(float(model.cfg.susp.k_heave_f), 1),
        "k_heave_r_nm": round(float(model.cfg.susp.k_heave_r), 1),
        "description": desc,
    }
    return model, meta


def run_event_simulation(
    model: DifferentiableVehicle14Dof,
    fn_kin: ca.Function,
    fn_ctrl: ca.Function,
    event: str,
    theta_veh: np.ndarray,
    theta_ctrl: np.ndarray,
    dt: float = 0.005,
    record_timeseries: bool = False,
    ts_stride: int = 10,
) -> Dict[str, Any]:
    """
    Execute a closed-loop 14-DOF simulation for Skidpad or Autocross and
    compute full ride-height, pitch, aero ClA_eff, CoP migration, LLTD,
    camber utilization, and lap-time metrics.
    """
    c = model.cfg
    cl_a_nom = float(theta_veh[PARAM_IDX["cl_a"]])
    aero_bal_nom = float(theta_veh[PARAM_IDX["aero_balance_front"]])
    stall_rh_f = float(c.stall_rh_f)
    w_base = model.default_disturbance(t_tire_c=75.0, t_motor_c=60.0)

    if event == "skidpad":
        n_steps = 1200  # 6.00 s (settle + full 360 deg timed circle)
        r_tgt = 8.82
        vx0 = 11.8
        wz0 = vx0 / r_tgt
        x = model.default_state(vx=vx0, vy=0.0, wz=wz0)
        x[STATE_IDX["X"]] = 0.0
        x[STATE_IDX["Y"]] = R_CENTER_M - r_tgt
        x[STATE_IDX["psi"]] = 0.0
        delta0 = math.atan(c.wheelbase / r_tgt)
        u0 = np.array([delta0, 35.0, 35.0, 45.0, 45.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        _, y = model.step(x, u0, theta_veh, w_base)
        c_aug = np.array([wz0, vx0 * wz0, 0.0, 0.0, 0.0, 0.0, delta0], dtype=np.float64)
        s_start_measure = 0.60  # rad (~34 deg settle before timing full circle)
    else:  # autocross
        n_steps = 2600  # up to 13.00 s (stops early once s >= AUTOCROSS_TARGET_S_M)
        vx0 = 14.0
        x = model.default_state(vx=vx0, vy=0.0, wz=0.0)
        u0 = np.array([0.0, 80.0, 80.0, 100.0, 100.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        _, y = model.step(x, u0, theta_veh, w_base)
        c_aug = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        s_start_measure = 0.0

    # Accumulators
    rh_f_list: List[float] = []
    rh_r_list: List[float] = []
    pitch_deg_list: List[float] = []
    roll_deg_list: List[float] = []
    cla_eff_model_list: List[float] = []
    cla_eff_map_list: List[float] = []
    f_down_list: List[float] = []
    cop_pct_list: List[float] = []
    lltd_pct_list: List[float] = []
    outside_camber_deg_list: List[float] = []
    outside_camber_err_list: List[float] = []
    vx_list: List[float] = []
    ay_g_list: List[float] = []
    ax_g_list: List[float] = []
    p_dc_list: List[float] = []
    dmz_dbeta_list: List[float] = []
    clear_list: List[float] = []

    hv_violations = 0
    bspd_violations = 0
    cones_hit = 0
    stall_steps = 0

    ts_rows: Dict[str, List[float]] = {
        "t_s": [],
        "s_m": [],
        "vx_ms": [],
        "ax_g": [],
        "ay_g": [],
        "rh_f_mm": [],
        "rh_r_mm": [],
        "pitch_deg": [],
        "roll_deg": [],
        "cop_front_pct": [],
        "cla_eff_m2": [],
        "lltd_pct": [],
        "gamma_fl_deg": [],
        "gamma_fr_deg": [],
        "gamma_rl_deg": [],
        "gamma_rr_deg": [],
        "p_dc_kw": [],
    }

    t_enter_lap: Optional[float] = None
    t_exit_lap: Optional[float] = None
    s_prev = float(c_aug[4])
    t_finish_ax: Optional[float] = None

    for k in range(n_steps):
        t_now = k * dt
        u_dm, c_next_dm, aux_dm = fn_ctrl(x, y, c_aug, theta_veh, theta_ctrl)
        u_k = np.asarray(u_dm, dtype=np.float64).ravel()
        c_next = np.asarray(c_next_dm, dtype=np.float64).ravel()
        aux_k = np.asarray(aux_dm, dtype=np.float64).ravel()

        s_curr = float(c_next[4])
        # Build road elevation disturbance for current arc-length
        s_road_m = s_curr * R_CENTER_M if event == "skidpad" else s_curr
        w_k = w_base.copy()
        w_k[4:8] = _road_elevation_4(s_road_m, event=event, wheelbase=c.wheelbase)

        x_next, y_next = model.step(x, u_k, theta_veh, w_k)

        # Evaluate kinematics (exact wheel ground cambers [rad])
        gammas_dm, _, _ = fn_kin(x_next, u_k, theta_veh)
        gammas_deg = np.degrees(np.asarray(gammas_dm, dtype=np.float64).ravel())

        vx_k = float(x_next[STATE_IDX["vx"]])
        vy_k = float(x_next[STATE_IDX["vy"]])
        ax_k = float(y_next[OUTPUT_IDX["ax"]])
        ay_k = float(y_next[OUTPUT_IDX["ay"]])
        rh_f_m = float(y_next[OUTPUT_IDX["rh_f"]])
        rh_r_m = float(y_next[OUTPUT_IDX["rh_r"]])
        rh_f_mm = rh_f_m * 1000.0
        rh_r_mm = rh_r_m * 1000.0
        pitch_deg = math.degrees(float(x_next[STATE_IDX["theta_p"]]))
        roll_deg = math.degrees(float(x_next[STATE_IDX["phi"]]))
        p_dc_kw = float(y_next[OUTPUT_IDX["P_dc_kw"]])
        dmz_dbeta = float(aux_k[1])

        # Exact aero stall ratio & effective ClA + CoP migration
        stall_ratio = float(
            np.clip((rh_f_m - 0.5 * stall_rh_f) / max(0.5 * stall_rh_f, 1e-3), 0.35, 1.0)
        )
        rake_mm = rh_r_mm - rh_f_mm
        cl_rake_scale, _, cop_rake_f = rake_to_aero_modifiers(rake_mm, base_cop_front=aero_bal_nom)
        # Front diffuser stall drops front aero load -> shifts CoP rearward
        cop_front_frac = float(np.clip(cop_rake_f - 0.12 * (1.0 - stall_ratio), 0.34, 0.58))
        cla_eff_model = cl_a_nom * stall_ratio
        cla_eff_map = cl_a_nom * stall_ratio * cl_rake_scale
        q_dyn = 0.5 * c.air_density * (vx_k * vx_k + vy_k * vy_k)
        f_down_n = q_dyn * cla_eff_model

        # Instantaneous LLTD from wheel normal loads
        fz_fl = float(y_next[OUTPUT_IDX["Fz_FL"]])
        fz_fr = float(y_next[OUTPUT_IDX["Fz_FR"]])
        fz_rl = float(y_next[OUTPUT_IDX["Fz_RL"]])
        fz_rr = float(y_next[OUTPUT_IDX["Fz_RR"]])
        dfz_f = abs(0.5 * (fz_fr - fz_fl))
        dfz_r = abs(0.5 * (fz_rr - fz_rl))
        lltd_pct = 100.0 * dfz_f / max(dfz_f + dfz_r, 1e-3)

        # Safety checks (HV DC [-25, +80] kW, BSPD EV 2.3)
        if p_dc_kw > 80.0 or p_dc_kw < -25.0:
            hv_violations += 1
        t_brk_sum = float(np.sum(u_k[5:9]))
        t_mot_pos = float(np.sum(np.maximum(u_k[1:5], 0.0)))
        if t_brk_sum > 200.0 and t_mot_pos > 150.0 and p_dc_kw > 5.0:
            bspd_violations += 1

        # Skidpad cone clearance
        if event == "skidpad":
            r_now = math.hypot(float(x_next[STATE_IDX["X"]]), float(x_next[STATE_IDX["Y"]]) - R_CENTER_M)
            c_in = r_now - (R_INNER_M + HALF_TRACK_M)
            c_out = (R_OUTER_M - HALF_TRACK_M) - r_now
            clear_m = min(c_in, c_out)
            if s_curr >= s_start_measure:
                clear_list.append(clear_m)
                if clear_m < 0.0:
                    cones_hit += 1

        # Measurement window
        in_window = (event == "autocross") or (s_curr >= s_start_measure)
        if in_window:
            rh_f_list.append(rh_f_mm)
            rh_r_list.append(rh_r_mm)
            pitch_deg_list.append(pitch_deg)
            roll_deg_list.append(roll_deg)
            cla_eff_model_list.append(cla_eff_model)
            cla_eff_map_list.append(cla_eff_map)
            f_down_list.append(f_down_n)
            cop_pct_list.append(cop_front_frac * 100.0)
            vx_list.append(vx_k)
            ay_g_list.append(ay_k / 9.81)
            ax_g_list.append(ax_k / 9.81)
            p_dc_list.append(p_dc_kw)
            dmz_dbeta_list.append(dmz_dbeta)
            if rh_f_mm < stall_rh_f * 1000.0:
                stall_steps += 1

            # Cornering-specific metrics (|ay| > 0.50 g)
            if abs(ay_k) > 0.50 * 9.81:
                lltd_pct_list.append(lltd_pct)
                # Outside tires in left turn (ay > 0) are FR(1), RR(3) where +gamma tilts into turn;
                # in right turn (ay < 0) are FL(0), RL(2) where -gamma tilts into turn.
                if ay_k > 0.0:
                    cam_out_f = float(gammas_deg[1])
                    cam_out_r = float(gammas_deg[3])
                else:
                    cam_out_f = float(-gammas_deg[0])
                    cam_out_r = float(-gammas_deg[2])
                cam_out_avg = 0.5 * (cam_out_f + cam_out_r)
                outside_camber_deg_list.append(cam_out_avg)
                outside_camber_err_list.append(abs(cam_out_avg - OPT_OUTSIDE_CAMBER_DEG))

        if record_timeseries and (k % ts_stride == 0):
            ts_rows["t_s"].append(round(t_now, 4))
            ts_rows["s_m"].append(round(s_road_m, 3))
            ts_rows["vx_ms"].append(round(vx_k, 3))
            ts_rows["ax_g"].append(round(ax_k / 9.81, 3))
            ts_rows["ay_g"].append(round(ay_k / 9.81, 3))
            ts_rows["rh_f_mm"].append(round(rh_f_mm, 3))
            ts_rows["rh_r_mm"].append(round(rh_r_mm, 3))
            ts_rows["pitch_deg"].append(round(pitch_deg, 4))
            ts_rows["roll_deg"].append(round(roll_deg, 4))
            ts_rows["cop_front_pct"].append(round(cop_front_frac * 100.0, 3))
            ts_rows["cla_eff_m2"].append(round(cla_eff_map, 4))
            ts_rows["lltd_pct"].append(round(lltd_pct if abs(ay_k) > 2.0 else 50.0, 2))
            ts_rows["gamma_fl_deg"].append(round(float(gammas_deg[0]), 3))
            ts_rows["gamma_fr_deg"].append(round(float(gammas_deg[1]), 3))
            ts_rows["gamma_rl_deg"].append(round(float(gammas_deg[2]), 3))
            ts_rows["gamma_rr_deg"].append(round(float(gammas_deg[3]), 3))
            ts_rows["p_dc_kw"].append(round(p_dc_kw, 2))

        # Exact interpolated event timing
        if event == "skidpad":
            if t_enter_lap is None and s_curr >= s_start_measure:
                frac = (s_start_measure - s_prev) / max(s_curr - s_prev, 1e-9)
                t_enter_lap = t_now + frac * dt
            if t_enter_lap is not None and t_exit_lap is None and s_curr >= (s_start_measure + 2.0 * math.pi):
                frac = (s_start_measure + 2.0 * math.pi - s_prev) / max(s_curr - s_prev, 1e-9)
                t_exit_lap = t_now + frac * dt
                x, y, c_aug = x_next, y_next, c_next
                break
        else:
            if t_finish_ax is None and s_curr >= AUTOCROSS_TARGET_S_M:
                frac = (AUTOCROSS_TARGET_S_M - s_prev) / max(s_curr - s_prev, 1e-9)
                t_finish_ax = t_now + frac * dt
                x, y, c_aug = x_next, y_next, c_next
                break

        s_prev = s_curr
        x, y, c_aug = x_next, y_next, c_next

    # Compute exact event time
    if event == "skidpad":
        if t_enter_lap is not None and t_exit_lap is not None:
            event_time_s = float(t_exit_lap - t_enter_lap)
        else:
            # Extrapolate exact 2*pi circle time from measured angular progress
            dtheta_tot = max(float(c_aug[4]) - s_start_measure, 1e-3)
            t_meas = max((len(rh_f_list)) * dt, 1e-3)
            event_time_s = float(t_meas * (2.0 * math.pi / dtheta_tot))
    else:
        if t_finish_ax is not None:
            event_time_s = float(t_finish_ax)
        else:
            s_tot = max(float(c_aug[4]), 1.0)
            event_time_s = float((n_steps * dt) * (AUTOCROSS_TARGET_S_M / s_tot))

    rh_f_arr = np.asarray(rh_f_list, dtype=np.float64)
    rh_r_arr = np.asarray(rh_r_list, dtype=np.float64)
    pitch_arr = np.asarray(pitch_deg_list, dtype=np.float64)
    roll_arr = np.asarray(roll_deg_list, dtype=np.float64)
    cla_mod_arr = np.asarray(cla_eff_model_list, dtype=np.float64)
    cla_map_arr = np.asarray(cla_eff_map_list, dtype=np.float64)
    f_down_arr = np.asarray(f_down_list, dtype=np.float64)
    cop_arr = np.asarray(cop_pct_list, dtype=np.float64)
    lltd_arr = np.asarray(lltd_pct_list, dtype=np.float64) if lltd_pct_list else np.array([50.0])
    cam_out_arr = np.asarray(outside_camber_deg_list, dtype=np.float64) if outside_camber_deg_list else np.array([0.0])
    cam_err_arr = np.asarray(outside_camber_err_list, dtype=np.float64) if outside_camber_err_list else np.array([0.0])
    vx_arr = np.asarray(vx_list, dtype=np.float64)
    ay_arr = np.asarray(ay_g_list, dtype=np.float64)
    ax_arr = np.asarray(ax_g_list, dtype=np.float64)
    p_dc_arr = np.asarray(p_dc_list, dtype=np.float64)
    dmz_arr = np.asarray(dmz_dbeta_list, dtype=np.float64)

    # Camber utilization score: 100% when outside tire holds OPT_OUTSIDE_CAMBER_DEG into turn
    camber_util_pct = float(
        100.0 * np.mean(np.clip(1.0 - cam_err_arr / 2.5, 0.0, 1.0))
    )

    metrics: Dict[str, Any] = {
        "event": event,
        "event_time_s": round(event_time_s, 5),
        "mean_vx_ms": round(float(np.mean(vx_arr)), 3),
        "max_vx_ms": round(float(np.max(vx_arr)), 3),
        "max_ay_g": round(float(np.max(np.abs(ay_arr))), 3),
        "max_ax_accel_g": round(float(np.max(ax_arr)), 3),
        "max_ax_brake_g": round(float(np.min(ax_arr)), 3),
        # Ride-height & aero platform control
        "mean_rh_f_mm": round(float(np.mean(rh_f_arr)), 3),
        "mean_rh_r_mm": round(float(np.mean(rh_r_arr)), 3),
        "std_rh_f_mm": round(float(np.std(rh_f_arr)), 3),
        "std_rh_r_mm": round(float(np.std(rh_r_arr)), 3),
        "min_rh_f_mm": round(float(np.min(rh_f_arr)), 3),
        "min_rh_r_mm": round(float(np.min(rh_r_arr)), 3),
        "ptp_rh_f_mm": round(float(np.ptp(rh_f_arr)), 3),
        "ptp_rh_r_mm": round(float(np.ptp(rh_r_arr)), 3),
        "stall_fraction_pct": round(100.0 * stall_steps / max(len(rh_f_arr), 1), 3),
        # Pitch & Roll
        "pitch_amp_deg": round(float(np.ptp(pitch_arr)), 4),
        "pitch_rms_deg": round(float(np.sqrt(np.mean(pitch_arr ** 2))), 4),
        "max_nose_dive_deg": round(float(np.max(pitch_arr)), 4),
        "max_squat_deg": round(float(np.min(pitch_arr)), 4),
        "roll_amp_deg": round(float(np.ptp(roll_arr)), 4),
        "roll_rms_deg": round(float(np.sqrt(np.mean(roll_arr ** 2))), 4),
        "max_abs_roll_deg": round(float(np.max(np.abs(roll_arr))), 4),
        # Aero downforce retention & CoP migration
        "mean_ClA_eff": round(float(np.mean(cla_map_arr)), 4),
        "mean_ClA_eff_model": round(float(np.mean(cla_mod_arr)), 4),
        "min_ClA_eff_model": round(float(np.min(cla_mod_arr)), 4),
        "ClA_retention_pct": round(100.0 * float(np.mean(cla_mod_arr)) / max(cl_a_nom, 1e-6), 3),
        "mean_downforce_N": round(float(np.mean(f_down_arr)), 1),
        "max_downforce_N": round(float(np.max(f_down_arr)), 1),
        "mean_CoP_pct": round(float(np.mean(cop_arr)), 3),
        "delta_CoP_pct": round(float(np.ptp(cop_arr)), 3),
        "std_CoP_pct": round(float(np.std(cop_arr)), 3),
        # LLTD & Camber utilization
        "mean_lltd_pct": round(float(np.mean(lltd_arr)), 3),
        "std_lltd_pct": round(float(np.std(lltd_arr)), 3),
        "mean_outside_camber_deg": round(float(np.mean(cam_out_arr)), 3),
        "min_outside_camber_deg": round(float(np.min(cam_out_arr)), 3),
        "mean_camber_err_deg": round(float(np.mean(cam_err_arr)), 3),
        "camber_utilization_pct": round(camber_util_pct, 2),
        # Powertrain & Safety
        "p_dc_max_kw": round(float(np.max(p_dc_arr)), 2),
        "p_dc_min_kw": round(float(np.min(p_dc_arr)), 2),
        "hv_violations": int(hv_violations),
        "bspd_violations": int(bspd_violations),
        "max_dmz_dbeta_nm_rad": round(float(np.max(dmz_arr)), 2),
        "yaw_stable": bool(float(np.max(dmz_arr)) < 0.0),
        "min_clear_m": round(float(min(clear_list)), 4) if clear_list else None,
        "cones_hit": int(cones_hit),
    }

    if record_timeseries:
        metrics["timeseries"] = ts_rows
    return metrics


def _apply_mass_penalty(
    theta_base: np.ndarray,
    delta_mass_kg: float,
) -> np.ndarray:
    """
    Apply realistic hardware mass penalty (+delta_mass_kg) to theta, scaling
    chassis rotational inertias proportionally for physical consistency.
    """
    th = theta_base.copy()
    m0 = float(th[PARAM_IDX["mass"]])
    m_new = m0 + float(delta_mass_kg)
    ratio = m_new / max(m0, 1.0)
    th[PARAM_IDX["mass"]] = m_new
    th[PARAM_IDX["i_xx"]] *= ratio
    th[PARAM_IDX["i_yy"]] *= ratio
    th[PARAM_IDX["i_zz"]] *= ratio
    return th


def _interpolate_breakeven_kg(
    masses_kg: List[float],
    times_s: List[float],
    ref_time_s: float,
) -> Optional[float]:
    """
    Find delta_m [kg] where time(delta_m) crosses ref_time_s (linear interpolation).
    Returns 0.0 if already slower at 0 kg, or extrapolated/capped value if faster across full sweep.
    """
    if times_s[0] >= ref_time_s:
        return 0.0
    for i in range(len(masses_kg) - 1):
        t0, t1 = times_s[i], times_s[i + 1]
        if t0 <= ref_time_s <= t1 and abs(t1 - t0) > 1e-12:
            frac = (ref_time_s - t0) / (t1 - t0)
            return round(float(masses_kg[i] + frac * (masses_kg[i + 1] - masses_kg[i])), 3)
    # Still faster at max swept mass: linearly extrapolate from last two points
    slope = (times_s[-1] - times_s[0]) / max(masses_kg[-1] - masses_kg[0], 1e-6)
    if slope > 1e-6:
        return round(float((ref_time_s - times_s[0]) / slope), 3)
    return None


def run_suspension_architecture_benchmark(
    report_path: Path = DEFAULT_REPORT_PATH,
    dt: float = 0.005,
) -> Dict[str, Any]:
    """Execute the full controlled suspension architecture benchmark suite."""
    t0_wall = time.perf_counter()
    print("=== Controlled Physics Benchmark: 4 Suspension Architectures (CasADi 14-DOF) ===", flush=True)
    print(f"Vehicle: TeR27 4WD | Tires: R25B | Grip: FSG_RUBBERED={FSG_RUBBERED} (COLD_TEST_OCT={COLD_TEST_OCT})", flush=True)

    ctrl = Diff4WDController(alqp_iters=8)
    theta_ctrl_nom = ctrl_dict_to_vec()
    fn_ctrl_skidpad = _build_4wd_controller_fn(ctrl, event="skidpad", dt=dt)
    fn_ctrl_autocross = _build_4wd_controller_fn(ctrl, event="autocross", dt=dt)

    arch_ids = list(SUSPENSION_MODES) + ["decoupled_heave_roll_tuned"]
    models: Dict[str, DifferentiableVehicle14Dof] = {}
    kin_fns: Dict[str, ca.Function] = {}
    arch_meta: Dict[str, Dict[str, Any]] = {}

    for aid in arch_ids:
        m, meta = _create_arch_model(aid, dt=dt)
        models[aid] = m
        kin_fns[aid] = _build_kinematics_evaluator(m)
        arch_meta[aid] = meta

    # ------------------------------------------------------------------
    # 1. Core Controlled Benchmark across all 4 (+1 tuned) architectures
    # ------------------------------------------------------------------
    print("\n[1/3] Running Skidpad & Autocross controlled benchmarks (identical base mass=290.0 kg)...", flush=True)
    results_skidpad: Dict[str, Dict[str, Any]] = {}
    results_autocross: Dict[str, Dict[str, Any]] = {}
    timeseries_export: Dict[str, Dict[str, Any]] = {}

    for aid in arch_ids:
        m = models[aid]
        th_nom = m.default_theta()
        res_sk = run_event_simulation(
            m, kin_fns[aid], fn_ctrl_skidpad, "skidpad", th_nom, theta_ctrl_nom,
            dt=dt, record_timeseries=True, ts_stride=10,
        )
        res_ax = run_event_simulation(
            m, kin_fns[aid], fn_ctrl_autocross, "autocross", th_nom, theta_ctrl_nom,
            dt=dt, record_timeseries=True, ts_stride=12,
        )
        timeseries_export[aid] = {
            "skidpad": res_sk.pop("timeseries"),
            "autocross": res_ax.pop("timeseries"),
        }
        results_skidpad[aid] = res_sk
        results_autocross[aid] = res_ax

        print(
            f"  {aid:27s} | Skidpad: {res_sk['event_time_s']:.4f}s (ay={res_sk['max_ay_g']:.2f}g, clr={res_sk['min_clear_m']:.3f}m, "
            f"std_rh_f={res_sk['std_rh_f_mm']:.2f}mm, cam={res_sk['camber_utilization_pct']:.1f}%) | "
            f"Autocross: {res_ax['event_time_s']:.4f}s (std_rh_f={res_ax['std_rh_f_mm']:.2f}mm, min_rh_f={res_ax['min_rh_f_mm']:.2f}mm, "
            f"pitch={res_ax['pitch_amp_deg']:.2f}deg, ClA={res_ax['mean_ClA_eff']:.3f}, dCoP={res_ax['delta_CoP_pct']:.2f}%)",
            flush=True,
        )

    # ------------------------------------------------------------------
    # 2. Anti-Dive (56.9%) / Anti-Squat (41.2%) Ablation Study
    # ------------------------------------------------------------------
    print("\n[2/3] Running Anti-Dive (56.9%) & Anti-Squat (41.2%) ablation on Autocross...", flush=True)
    anti_geom_ablation: Dict[str, Dict[str, Any]] = {}
    for aid in ("standard_pushrod", "3d_double_wishbone_lut", "decoupled_heave_roll", "decoupled_heave_roll_tuned"):
        m = models[aid]
        th_no_anti = m.default_theta()
        th_no_anti[PARAM_IDX["anti_dive_f"]] = 0.0
        th_no_anti[PARAM_IDX["anti_squat_r"]] = 0.0
        res_no_anti = run_event_simulation(
            m, kin_fns[aid], fn_ctrl_autocross, "autocross", th_no_anti, theta_ctrl_nom,
            dt=dt, record_timeseries=False,
        )
        res_with_anti = results_autocross[aid]
        anti_geom_ablation[aid] = {
            "without_anti_geom_0pct": res_no_anti,
            "with_ter27_anti_geom_56p9_41p2": res_with_anti,
            "delta_time_ms": round((res_with_anti["event_time_s"] - res_no_anti["event_time_s"]) * 1000.0, 2),
            "delta_pitch_amp_deg": round(res_with_anti["pitch_amp_deg"] - res_no_anti["pitch_amp_deg"], 4),
            "delta_std_rh_f_mm": round(res_with_anti["std_rh_f_mm"] - res_no_anti["std_rh_f_mm"], 3),
            "delta_min_rh_f_mm": round(res_with_anti["min_rh_f_mm"] - res_no_anti["min_rh_f_mm"], 3),
            "delta_CoP_pct_change": round(res_with_anti["delta_CoP_pct"] - res_no_anti["delta_CoP_pct"], 3),
            "delta_ClA_eff": round(res_with_anti["mean_ClA_eff"] - res_no_anti["mean_ClA_eff"], 4),
        }
        ab = anti_geom_ablation[aid]
        print(
            f"  {aid:27s} | anti-dive/squat effect: dT={ab['delta_time_ms']:+.1f}ms, "
            f"dPitch={ab['delta_pitch_amp_deg']:+.3f}deg, dStdRH_f={ab['delta_std_rh_f_mm']:+.2f}mm, "
            f"dMinRH_f={ab['delta_min_rh_f_mm']:+.2f}mm, dCoP={ab['delta_CoP_pct_change']:+.2f}%",
            flush=True,
        )

    # ------------------------------------------------------------------
    # 3. Hardware Mass Penalty Sweep & Net Lap-Time Break-Even Analysis
    # ------------------------------------------------------------------
    print("\n[3/3] Running hardware mass penalty sensitivity sweep (+0.0 to +4.0 kg) for decoupled_heave_roll...", flush=True)
    delta_masses_kg = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    ref_pushrod_sk = results_skidpad["standard_pushrod"]["event_time_s"]
    ref_pushrod_ax = results_autocross["standard_pushrod"]["event_time_s"]
    ref_3dw_sk = results_skidpad["3d_double_wishbone_lut"]["event_time_s"]
    ref_3dw_ax = results_autocross["3d_double_wishbone_lut"]["event_time_s"]

    mass_sweep: Dict[str, Any] = {}
    for aid in ("decoupled_heave_roll", "decoupled_heave_roll_tuned"):
        m = models[aid]
        th_nom = m.default_theta()
        sk_times: List[float] = []
        ax_times: List[float] = []
        comb_times: List[float] = []
        rows: List[Dict[str, Any]] = []

        for dm in delta_masses_kg:
            th_m = _apply_mass_penalty(th_nom, dm)
            r_sk = run_event_simulation(
                m, kin_fns[aid], fn_ctrl_skidpad, "skidpad", th_m, theta_ctrl_nom,
                dt=dt, record_timeseries=False,
            )
            r_ax = run_event_simulation(
                m, kin_fns[aid], fn_ctrl_autocross, "autocross", th_m, theta_ctrl_nom,
                dt=dt, record_timeseries=False,
            )
            t_sk = r_sk["event_time_s"]
            t_ax = r_ax["event_time_s"]
            t_comb = t_sk + t_ax
            sk_times.append(t_sk)
            ax_times.append(t_ax)
            comb_times.append(t_comb)
            rows.append({
                "delta_mass_kg": dm,
                "total_mass_kg": round(290.0 + dm, 2),
                "skidpad_time_s": t_sk,
                "autocross_time_s": t_ax,
                "combined_time_s": round(t_comb, 5),
                "delta_vs_pushrod_skidpad_ms": round((t_sk - ref_pushrod_sk) * 1000.0, 2),
                "delta_vs_pushrod_autocross_ms": round((t_ax - ref_pushrod_ax) * 1000.0, 2),
                "delta_vs_3dw_skidpad_ms": round((t_sk - ref_3dw_sk) * 1000.0, 2),
                "delta_vs_3dw_autocross_ms": round((t_ax - ref_3dw_ax) * 1000.0, 2),
                "autocross_min_rh_f_mm": r_ax["min_rh_f_mm"],
                "autocross_std_rh_f_mm": r_ax["std_rh_f_mm"],
                "autocross_mean_ClA_eff": r_ax["mean_ClA_eff"],
            })

        # Compute mass sensitivity slopes [ms/kg]
        sk_ms_per_kg = round(((sk_times[-1] - sk_times[0]) / delta_masses_kg[-1]) * 1000.0, 2)
        ax_ms_per_kg = round(((ax_times[-1] - ax_times[0]) / delta_masses_kg[-1]) * 1000.0, 2)

        be_vs_pushrod_sk = _interpolate_breakeven_kg(delta_masses_kg, sk_times, ref_pushrod_sk)
        be_vs_pushrod_ax = _interpolate_breakeven_kg(delta_masses_kg, ax_times, ref_pushrod_ax)
        be_vs_pushrod_comb = _interpolate_breakeven_kg(
            delta_masses_kg, comb_times, ref_pushrod_sk + ref_pushrod_ax
        )
        be_vs_3dw_sk = _interpolate_breakeven_kg(delta_masses_kg, sk_times, ref_3dw_sk)
        be_vs_3dw_ax = _interpolate_breakeven_kg(delta_masses_kg, ax_times, ref_3dw_ax)
        be_vs_3dw_comb = _interpolate_breakeven_kg(
            delta_masses_kg, comb_times, ref_3dw_sk + ref_3dw_ax
        )

        mass_sweep[aid] = {
            "delta_masses_kg": delta_masses_kg,
            "sensitivity_skidpad_ms_per_kg": sk_ms_per_kg,
            "sensitivity_autocross_ms_per_kg": ax_ms_per_kg,
            "breakeven_mass_kg": {
                "vs_standard_pushrod_skidpad": be_vs_pushrod_sk,
                "vs_standard_pushrod_autocross": be_vs_pushrod_ax,
                "vs_standard_pushrod_combined": be_vs_pushrod_comb,
                "vs_3d_double_wishbone_skidpad": be_vs_3dw_sk,
                "vs_3d_double_wishbone_autocross": be_vs_3dw_ax,
                "vs_3d_double_wishbone_combined": be_vs_3dw_comb,
            },
            "curve": rows,
        }
        print(
            f"  {aid:27s} | mass slope: Skidpad={sk_ms_per_kg:+.2f} ms/kg, Autocross={ax_ms_per_kg:+.2f} ms/kg | "
            f"Break-even vs pushrod: AX={be_vs_pushrod_ax} kg, Comb={be_vs_pushrod_comb} kg | "
            f"vs 3D-DW: AX={be_vs_3dw_ax} kg, Comb={be_vs_3dw_comb} kg",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Build Comparative Summary Tables & Engineering Verdict
    # ------------------------------------------------------------------
    comp_table_skidpad = []
    comp_table_autocross = []
    for aid in arch_ids:
        rs = results_skidpad[aid]
        ra = results_autocross[aid]
        comp_table_skidpad.append({
            "architecture": aid,
            "event_time_s": rs["event_time_s"],
            "delta_vs_pushrod_ms": round((rs["event_time_s"] - ref_pushrod_sk) * 1000.0, 2),
            "max_ay_g": rs["max_ay_g"],
            "std_rh_f_mm": rs["std_rh_f_mm"],
            "std_rh_r_mm": rs["std_rh_r_mm"],
            "min_rh_f_mm": rs["min_rh_f_mm"],
            "roll_amp_deg": rs["roll_amp_deg"],
            "mean_ClA_eff": rs["mean_ClA_eff"],
            "delta_CoP_pct": rs["delta_CoP_pct"],
            "mean_lltd_pct": rs["mean_lltd_pct"],
            "std_lltd_pct": rs["std_lltd_pct"],
            "camber_utilization_pct": rs["camber_utilization_pct"],
            "min_clear_m": rs["min_clear_m"],
            "cones_hit": rs["cones_hit"],
        })
        comp_table_autocross.append({
            "architecture": aid,
            "event_time_s": ra["event_time_s"],
            "delta_vs_pushrod_ms": round((ra["event_time_s"] - ref_pushrod_ax) * 1000.0, 2),
            "max_vx_ms": ra["max_vx_ms"],
            "max_ay_g": ra["max_ay_g"],
            "std_rh_f_mm": ra["std_rh_f_mm"],
            "std_rh_r_mm": ra["std_rh_r_mm"],
            "min_rh_f_mm": ra["min_rh_f_mm"],
            "pitch_amp_deg": ra["pitch_amp_deg"],
            "roll_amp_deg": ra["roll_amp_deg"],
            "mean_ClA_eff": ra["mean_ClA_eff"],
            "ClA_retention_pct": ra["ClA_retention_pct"],
            "delta_CoP_pct": ra["delta_CoP_pct"],
            "mean_lltd_pct": ra["mean_lltd_pct"],
            "std_lltd_pct": ra["std_lltd_pct"],
            "camber_utilization_pct": ra["camber_utilization_pct"],
        })

    # Verify all safety invariants across all runs
    all_hv_zero = all(
        results_skidpad[a]["hv_violations"] == 0 and results_autocross[a]["hv_violations"] == 0
        for a in arch_ids
    )
    all_bspd_zero = all(
        results_skidpad[a]["bspd_violations"] == 0 and results_autocross[a]["bspd_violations"] == 0
        for a in arch_ids
    )
    all_cones_zero = all(results_skidpad[a]["cones_hit"] == 0 for a in arch_ids)
    all_clear_ok = all(
        (results_skidpad[a]["min_clear_m"] or 0.0) >= MIN_CONE_CLEAR_M for a in arch_ids
    )
    all_yaw_stable = all(
        results_skidpad[a]["yaw_stable"] and results_autocross[a]["yaw_stable"] for a in arch_ids
    )

    elapsed_s = time.perf_counter() - t0_wall
    report: Dict[str, Any] = {
        "benchmark_metadata": {
            "vehicle_id": "ter27_4wd",
            "model_id": MODEL_DIFFERENTIABLE_CASADI_14DOF,
            "tire_compound": "R25B",
            "grip_profile": "FSG_RUBBERED",
            "fsg_rubbered_lambda_muy": FSG_RUBBERED,
            "cold_test_oct_lambda_muy": COLD_TEST_OCT,
            "base_mass_kg": 290.0,
            "anti_dive_front_pct": 56.9,
            "anti_squat_rear_pct": 41.2,
            "cl_a_effective_nominal_m2": round(float(models["3d_double_wishbone_lut"].cfg.cl_a), 4),
            "cd_a_m2": float(models["3d_double_wishbone_lut"].cfg.cd_a),
            "aero_balance_front_pct": 48.0,
            "stall_rh_f_mm": 18.0,
            "dt_s": dt,
            "autocross_circuit_length_m": AUTOCROSS_TARGET_S_M,
            "wall_time_s": round(elapsed_s, 2),
        },
        "safety_invariants": {
            "all_hv_dc_within_minus25_plus80_kw": all_hv_zero,
            "all_bspd_zero_violations": all_bspd_zero,
            "all_skidpad_zero_cones": all_cones_zero,
            "all_skidpad_min_clear_ge_0p150m": all_clear_ok,
            "all_yaw_stable_dmz_dbeta_lt_0": all_yaw_stable,
        },
        "architectures": arch_meta,
        "comparative_tables": {
            "skidpad": comp_table_skidpad,
            "autocross": comp_table_autocross,
        },
        "detailed_results": {
            "skidpad": results_skidpad,
            "autocross": results_autocross,
        },
        "anti_dive_anti_squat_ablation": anti_geom_ablation,
        "mass_penalty_breakeven": mass_sweep,
        "timeseries": timeseries_export,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\nSaved benchmark report to {report_path} ({report_path.stat().st_size / 1024.0:.1f} KB) in {elapsed_s:.2f}s", flush=True)
    return report


if __name__ == "__main__":
    run_suspension_architecture_benchmark()
