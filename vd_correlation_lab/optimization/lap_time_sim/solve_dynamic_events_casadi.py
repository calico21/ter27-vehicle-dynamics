#!/usr/bin/env python3
"""
Reusable CasADi 3.7.2 + IPOPT Dynamic Events OCP Solver for Tecnun e-Racing TeR27 4WD.

Provides exact smooth C^2 optimal control solvers for Formula Student dynamic events:
  1. Official FSG Skidpad Stationary Radius & 4-Wheel Torque Optimization
  2. Official FSG Skidpad Full Figure-8 OCP (Entry -> 2R -> Crossover -> 2L -> Exit)
  3. Official FS 75 m Acceleration OCP (4-wheel slip + 80 kW HV DC ceiling)

Rules & Physics Integrity:
  - Official FSG Skidpad: R_inner = 7.625 m, R_outer = 10.625 m, centers at (0, +-9.125) m.
  - Strict 4-wheel cone clearance >= 0.15 m (R_eff >= 7.625 + 0.615 + 0.15 = 8.390 m).
  - Calibrated Hoosier 16.0x7.5-10 R20 / R25B MF 6.2 tires only (no mu inflation).
  - Strict HV DC power envelope [-25.0, +80.0] kW and motor torque/speed envelope.
  - Directional stability check dMz/dbeta < 0 at steady-state equilibrium.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import casadi as ca
import numpy as np

ORACLE_DIR = Path(__file__).resolve().parent
LAB_DIR = ORACLE_DIR.parents[1]
REPO_ROOT = LAB_DIR.parent
CORE_DIR = LAB_DIR / "core"
for _p in (str(REPO_ROOT), str(LAB_DIR), str(CORE_DIR), str(ORACLE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from vd_correlation_lab.models.vehicle_14dof.ter27_casadi_model import (  # noqa: E402
    SMOOTHING_PARAMS,
    TeR27CasADiVehicle,
    default_design_params,
    smax,
)
from vd_correlation_lab.models.vehicle_14dof.ter27_physics import (  # noqa: E402
    GRIP_PROFILES,
    TRACKSIDE_SETUP_BOUNDS,
    PacejkaMF62,
    TeR27PhysicsModel,
    load_ter27_params,
    rake_to_aero_modifiers,
)

R_INNER_M: float = 7.625
R_OUTER_M: float = 10.625
R_CENTER_M: float = 9.125
CONE_MARGIN_M: float = 0.150
HALF_TRACK_M: float = 0.615
R_EFF_MIN_M: float = R_INNER_M + HALF_TRACK_M + CONE_MARGIN_M  # 8.390 m


class SkidpadSetupCasADiVehicle(TeR27CasADiVehicle):
    """
    TeR27CasADiVehicle extension adding smooth static front/rear toe angles
    (toe_f_deg, toe_r_deg; positive = toe-in) and static rake (rake_mm in [-8, +8] mm)
    for trackside setup co-optimization without altering base ter27_casadi_model.py.
    """

    def ackermann_steer_smooth(self, delta: Any) -> Any:
        base = super().ackermann_steer_smooth(delta)
        toe_f_deg = self._dp("toe_f_deg", 0.0)
        toe_r_deg = self._dp("toe_r_deg", 0.0)
        toe_f_rad = toe_f_deg * (math.pi / 180.0)
        toe_r_rad = toe_r_deg * (math.pi / 180.0)
        # ISO 8855: positive toe-in -> left wheel points right (-), right wheel points left (+)
        return ca.vertcat(
            base[0] - toe_f_rad,
            base[1] + toe_f_rad,
            base[2] - toe_r_rad,
            base[3] + toe_r_rad,
        )


def _apply_rake_to_design_params(dp: Dict[str, Any], base_cop_front: float) -> None:
    """Map rake_mm (float or CasADi SX/MX) smoothly onto cl_scale, cd_scale, cop_front_frac."""
    if "rake_mm" not in dp:
        return
    r = dp["rake_mm"]
    cl_mod = 1.0 + 0.0080 * r - 0.00055 * r * r
    cd_mod = 1.0 + 0.0035 * r + 0.00030 * r * r
    cop_mod = base_cop_front + 0.0040 * r
    dp["cl_scale"] = dp.get("cl_scale", 1.0) * cl_mod
    dp["cd_scale"] = dp.get("cd_scale", 1.0) * cd_mod
    dp["cop_front_frac"] = cop_mod


def resolve_profile_config(grip_profile: str) -> Dict[str, Any]:
    """Resolve standardized grip profile parameters anchored to ter27_physics.GRIP_PROFILES."""
    key = grip_profile.strip().upper()
    if key in ("TTC_PEAK", "TTC_LAB_PEAK", "TTC"):
        gp = GRIP_PROFILES["TTC_LAB_PEAK"]
        return {
            "name": "TTC_PEAK",
            "physics_profile": "TTC_LAB_PEAK",
            "env_mu_scale": float(gp["env_mu_scale"]),
            "env_aero_cl_scale": float(gp["env_aero_cl_scale"]),
            "track_temp_c": 77.5,
            "tire_temp_c": 77.5,
        }
    if key in ("COLD_TEST_OCT", "COLD_OCT"):
        gp = GRIP_PROFILES["COLD_TEST_OCT"]
        return {
            "name": "COLD_TEST_OCT",
            "physics_profile": "COLD_TEST_OCT",
            "env_mu_scale": float(gp["env_mu_scale"]),
            "env_aero_cl_scale": float(gp["env_aero_cl_scale"]),
            "track_temp_c": float(gp.get("track_temp_c", 18.0)),
            "tire_temp_c": float(gp.get("tire_temp_c", 35.0)),
        }
    if key in ("FSG_RUBBERED", "FSG"):
        gp = GRIP_PROFILES["FSG_RUBBERED"]
        return {
            "name": "FSG_RUBBERED",
            "physics_profile": "FSG_RUBBERED",
            "env_mu_scale": float(gp["env_mu_scale"]),
            "env_aero_cl_scale": float(gp["env_aero_cl_scale"]),
            "track_temp_c": float(gp.get("track_temp_c", 45.0)),
            "tire_temp_c": float(gp.get("tire_temp_c", 75.0)),
        }
    if key in ("FSG_RUBBERED_LEGACY_0P88", "LEGACY_0P88"):
        return {
            "name": "FSG_RUBBERED_LEGACY_0P88",
            "physics_profile": "FSG_RUBBERED",
            "env_mu_scale": 0.88,
            "env_aero_cl_scale": 0.78,
            "track_temp_c": 45.0,
            "tire_temp_c": 75.0,
        }
    if key in ("COLD_LOW_GRIP", "COLD", "COLD_18C"):
        gp = GRIP_PROFILES["COLD_LOW_GRIP"]
        return {
            "name": "COLD_LOW_GRIP",
            "physics_profile": "COLD_LOW_GRIP",
            "env_mu_scale": float(gp["env_mu_scale"]),
            "env_aero_cl_scale": float(gp["env_aero_cl_scale"]),
            "track_temp_c": float(gp.get("track_temp_c", 18.0)),
            "tire_temp_c": float(gp.get("tire_temp_c", 22.0)),
        }
    raise ValueError(f"Unsupported grip_profile '{grip_profile}'")


def build_casadi_vehicle_for_compound(
    grip_profile: str = "FSG_RUBBERED",
    compound: str = "R20",
    tire_temp_c: Optional[float] = None,
    design_overrides: Optional[Dict[str, Any]] = None,
) -> SkidpadSetupCasADiVehicle:
    """Build a smooth C^2 SkidpadSetupCasADiVehicle for the specified grip profile, compound, and setup."""
    cfg = resolve_profile_config(grip_profile)
    p = load_ter27_params()
    comp_upper = compound.strip().upper()
    tire = PacejkaMF62(comp_upper)
    t_tire = cfg["tire_temp_c"] if tire_temp_c is None else float(tire_temp_c)

    dp: Dict[str, Any] = dict(default_design_params(p))
    dp["toe_f_deg"] = 0.0
    dp["toe_r_deg"] = 0.0
    if design_overrides:
        dp.update(design_overrides)
        if "brake_bias" in design_overrides and "brake_bias_front" not in design_overrides:
            dp["brake_bias_front"] = design_overrides["brake_bias"]
        _apply_rake_to_design_params(dp, float(p.aero_balance_front))

    x_w = np.array([p.a_cog, p.a_cog, -p.b_cog, -p.b_cog], dtype=float)
    y_w = np.array(
        [0.5 * p.track_front, -0.5 * p.track_front, 0.5 * p.track_rear, -0.5 * p.track_rear],
        dtype=float,
    )
    bf = p.brake_bias_front
    hyd_split = np.array([0.5 * bf, 0.5 * bf, 0.5 * (1.0 - bf), 0.5 * (1.0 - bf)], dtype=float)
    side_sign = np.array([1.0, -1.0, 1.0, -1.0], dtype=float)

    env_mu = float(cfg["env_mu_scale"])
    env_cl = float(cfg["env_aero_cl_scale"])

    return SkidpadSetupCasADiVehicle(
        grip_profile=cfg["physics_profile"],
        params=p,
        tire=tire,
        env_mu_scale=env_mu,
        env_aero_cl_scale=env_cl,
        lmux=p.lambda_mux_base * env_mu,
        lmuy=p.lambda_muy_base * env_mu,
        x_w=x_w,
        y_w=y_w,
        hyd_split=hyd_split,
        side_sign=side_sign,
        smoothing=dict(SMOOTHING_PARAMS),
        temp_c=float(cfg["track_temp_c"]),
        tire_temp_c=t_tire,
        design_params=dp,
    )


def build_physics_model_for_compound(
    grip_profile: str = "FSG_RUBBERED",
    compound: str = "R20",
    design_overrides: Optional[Dict[str, float]] = None,
) -> TeR27PhysicsModel:
    """Build exact non-smoothed TeR27PhysicsModel for independent verification."""
    cfg = resolve_profile_config(grip_profile)
    model = TeR27PhysicsModel(profile="TTC_LAB_PEAK")
    model.p.tire_compound = compound.strip().upper()
    model.env_mu_scale = float(cfg["env_mu_scale"])
    model.env_aero_cl_scale = float(cfg["env_aero_cl_scale"])
    model.lmux = model.p.lambda_mux_base * model.env_mu_scale
    model.lmuy = model.p.lambda_muy_base * model.env_mu_scale
    model.tire = PacejkaMF62(model.p.tire_compound)
    if design_overrides:
        if "lltd" in design_overrides or "k_roll_scale" in design_overrides:
            k_tot_base = model.p.k_roll_f + model.p.k_roll_r
            k_scale = float(design_overrides.get("k_roll_scale", 1.0))
            lltd = float(design_overrides.get("lltd", model.p.lltd_elastic_f))
            model.p.k_roll_f = lltd * k_tot_base * k_scale
            model.p.k_roll_r = (1.0 - lltd) * k_tot_base * k_scale
            model.p.lltd_elastic_f = lltd
        if "camber_f_deg" in design_overrides:
            model.p.static_camber_f_rad = math.radians(float(design_overrides["camber_f_deg"]))
        if "camber_r_deg" in design_overrides:
            model.p.static_camber_r_rad = math.radians(float(design_overrides["camber_r_deg"]))
        if "toe_f_deg" in design_overrides:
            model.p.static_toe_f_rad = math.radians(float(design_overrides["toe_f_deg"]))
        if "toe_r_deg" in design_overrides:
            model.p.static_toe_r_rad = math.radians(float(design_overrides["toe_r_deg"]))
        if "brake_bias_front" in design_overrides:
            model.p.brake_bias_front = float(design_overrides["brake_bias_front"])
        elif "brake_bias" in design_overrides:
            model.p.brake_bias_front = float(design_overrides["brake_bias"])
        if "rake_mm" in design_overrides:
            cl_mod, cd_mod, cop_f = rake_to_aero_modifiers(float(design_overrides["rake_mm"]), model.p.aero_balance_front)
            model.p.cl_a *= cl_mod
            model.p.cd_a *= cd_mod
            model.p.aero_balance_front = cop_f
            rake_m = float(design_overrides["rake_mm"]) * 1e-3
            model.p.static_rh_f = 0.035 - 0.5 * rake_m
            model.p.static_rh_r = 0.035 + 0.5 * rake_m
    return model


def _wheel_world_positions(
    X: Any, Y: Any, psi: Any, x_w: np.ndarray, y_w: np.ndarray
) -> Tuple[List[Any], List[Any]]:
    cp = ca.cos(psi)
    sp = ca.sin(psi)
    xw_list = [X + float(x_w[i]) * cp - float(y_w[i]) * sp for i in range(4)]
    yw_list = [Y + float(x_w[i]) * sp + float(y_w[i]) * cp for i in range(4)]
    return xw_list, yw_list


def compute_skidpad_cone_clearance_np(
    X: float,
    Y: float,
    psi: float,
    x_w: np.ndarray,
    y_w: np.ndarray,
    circle_side: str = "left",
) -> Tuple[float, np.ndarray]:
    """
    Compute exact minimum cone clearance (m) across all 4 wheels for a given circle.
    circle_side: 'left' -> center (0, +9.125), 'right' -> center (0, -9.125).
    """
    yc = R_CENTER_M if circle_side == "left" else -R_CENTER_M
    cp, sp = math.cos(psi), math.sin(psi)
    clearances = np.zeros(4, dtype=float)
    for i in range(4):
        xw = X + float(x_w[i]) * cp - float(y_w[i]) * sp
        yw = Y + float(x_w[i]) * sp + float(y_w[i]) * cp
        d = math.hypot(xw, yw - yc)
        clearances[i] = min(d - R_INNER_M, R_OUTER_M - d)
    return float(np.min(clearances)), clearances


def compute_directional_stability_dmz_dbeta(
    veh: TeR27CasADiVehicle,
    vx: float,
    vy: float,
    wz: float,
    delta: float,
    kappa_4: np.ndarray,
    ax_lt: float,
    ay_lt: float,
    eps_beta: float = 0.004,
) -> float:
    """
    Evaluate numerical directional stability derivative dMz/dbeta (Nm/deg) around
    the operating point at fixed (V, wz, delta, kappa_4). Must be < 0 for stability.
    """
    V = math.hypot(vx, vy)
    beta0 = math.atan2(vy, max(vx, 0.5))

    def _mz_at_beta(b: float) -> float:
        vx_b = V * math.cos(b)
        vy_b = V * math.sin(b)
        ev = veh.eval_forces_from_kappa(vx_b, vy_b, wz, delta, kappa_4, 0.0, ax_lt, ay_lt)
        return float(ev["Mz_tot"])

    mz_p = _mz_at_beta(beta0 + eps_beta)
    mz_m = _mz_at_beta(beta0 - eps_beta)
    dmz_drad = (mz_p - mz_m) / (2.0 * eps_beta)
    return float(dmz_drad * (math.pi / 180.0))


def _compute_mu_util_4(
    veh: TeR27CasADiVehicle,
    kap_4: np.ndarray,
    alpha_4: Any,
    fz_4: Any,
    delta_w_4: Any,
    phi: Any,
    ax_lt: float,
    fx_w_4: Any,
    fy_w_4: Any,
    mu_thermal_4: Any,
) -> np.ndarray:
    gam_4 = veh.cambers_smooth(delta_w_4, phi, ax_lt)
    _, _, _, mu_y_4 = veh.eval_tire_mf62(
        kap_4, alpha_4, fz_4, gam_4, veh.side_sign, mu_thermal=mu_thermal_4
    )
    fz_np = np.array(fz_4).flatten()
    ftot_np = np.hypot(np.array(fx_w_4).flatten(), np.array(fy_w_4).flatten())
    muy_np = np.array(mu_y_4).flatten()
    return ftot_np / np.maximum(muy_np * fz_np, 1.0)


def _implied_arb_stiffness_nm_m(p: Any, lltd: float, k_roll_scale: float) -> Tuple[float, float]:
    """Invert axle roll stiffness back to effective ARB stiffness (N/m at wheel) for reporting."""
    k_tot = (p.k_roll_f + p.k_roll_r) * float(k_roll_scale)
    k_rf_tgt = float(lltd) * k_tot
    k_rr_tgt = (1.0 - float(lltd)) * k_tot

    def _inv_axle(k_roll_tgt: float, track: float, k_w: float) -> float:
        k_series = k_roll_tgt / max(0.5 * track * track, 1e-6)
        if k_series >= 0.98 * p.k_tire_z:
            return 25000.0
        k_susp = (k_series * p.k_tire_z) / max(p.k_tire_z - k_series, 1.0)
        return float(max(0.0, 0.5 * (k_susp - k_w)))

    return (
        _inv_axle(k_rf_tgt, p.track_front, p.k_wheel_f),
        _inv_axle(k_rr_tgt, p.track_rear, p.k_wheel_r),
    )


def solve_skidpad_stationary_casadi(
    grip_profile: str = "FSG_RUBBERED",
    compound: str = "R20",
    r_fixed: Optional[float] = None,
    open_diff: bool = False,
    turn_dir: str = "left",
    r_bounds: Tuple[float, float] = (R_EFF_MIN_M, 9.60),
    design_overrides: Optional[Dict[str, float]] = None,
    optimize_setup: bool = False,
    optimize_trackside_setup: bool = False,
    equal_front_drive: bool = False,
) -> Dict[str, Any]:
    """
    Solve the stationary optimal skidpad cornering NLP with CasADi + IPOPT.
    Maximizes cornering speed (minimizes t_circle = 2*pi*R / V) over
    [V, delta, beta, kappa_FL, kappa_FR, kappa_RL, kappa_RR, R]
    and optionally co-optimizes the 7 mechanical trackside setup parameters:
    [lltd, camber_f_deg, camber_r_deg, toe_f_deg, toe_r_deg, brake_bias, rake_mm].
    """
    do_opt_setup = bool(optimize_setup or optimize_trackside_setup)
    p_nom = load_ter27_params()
    dp_init: Dict[str, Any] = dict(design_overrides) if design_overrides else {}
    s_sc: Optional[ca.SX] = None
    setup_scale = np.array([0.5, 2.0, 2.0, 0.5, 0.5, 0.6, 8.0], dtype=float)
    if do_opt_setup:
        s_sc = ca.SX.sym("s_sc", 7)
        s_val = s_sc * ca.DM(setup_scale)
        dp_init["lltd"] = s_val[0]
        dp_init["k_roll_scale"] = 1.0
        dp_init["camber_f_deg"] = s_val[1]
        dp_init["camber_r_deg"] = s_val[2]
        dp_init["toe_f_deg"] = s_val[3]
        dp_init["toe_r_deg"] = s_val[4]
        dp_init["brake_bias_front"] = s_val[5]
        dp_init["rake_mm"] = s_val[6]

    veh = build_casadi_vehicle_for_compound(grip_profile, compound, design_overrides=dp_init if dp_init else None)
    p = veh.params
    mg = p.mass * p.gravity
    mgl = mg * p.wheelbase
    p_max_kw = (p.p_dc_max_w - 150.0) * 1e-3
    p_min_kw = (p.p_dc_min_w + 800.0) * 1e-3  # safety margin for exact physics parity (>= -25.0 kW)
    sgn = 1.0 if turn_dir == "left" else -1.0

    scale = np.array([10.0, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 10.0], dtype=float)
    x_sc = ca.SX.sym("x_sc", 8)
    x = x_sc * ca.DM(scale)

    V = x[0]
    delta = x[1]
    beta = x[2]
    kappa_4 = x[3:7]
    R = x[7]

    vx = V * ca.cos(beta)
    vy = V * ca.sin(beta)
    r = sgn * V / R
    ax_ss = -r * vy
    ay_ss = r * vx

    ev = veh.eval_forces_from_kappa(vx, vy, r, delta, kappa_4, 0.0, ax_ss, ay_ss)
    eq_fx = (ev["Fx_tot"] - p.mass * ax_ss) / mg
    eq_fy = (ev["Fy_tot"] - p.mass * ay_ss) / mg
    eq_mz = ev["Mz_tot"] / mgl

    t_mot = ev["T_motor_4"]
    t_max = ev["T_max_4"]
    t_min = ev["T_min_4"]
    p_ecu_kw = ev["P_dc_ecu_kw"]
    p_drv_kw = ev["P_dc_drv_kw"]

    # Exact 4-wheel cone clearance when car velocity is along +x at (0, sgn*(9.125 - R)) -> psi = -beta
    psi = -beta
    yc_cone = sgn * R_CENTER_M
    y_cg_offset = sgn * (R_CENTER_M - R)
    xw_list, yw_list = _wheel_world_positions(0.0, y_cg_offset, psi, veh.x_w, veh.y_w)
    g_cone: List[Any] = []
    for i in range(4):
        d_cone = ca.sqrt(xw_list[i] ** 2 + (yw_list[i] - yc_cone) ** 2)
        g_cone.append(d_cone - (R_INNER_M + CONE_MARGIN_M))
        g_cone.append((R_OUTER_M - CONE_MARGIN_M) - d_cone)

    g_list: List[Any] = [
        eq_fx,
        eq_fy,
        eq_mz,
        (t_max - t_mot) / 100.0,
        (t_mot - t_min) / 100.0,
        (p_max_kw - p_ecu_kw) / 10.0,
        (p_max_kw - p_drv_kw) / 10.0,
        (p_ecu_kw - p_min_kw) / 10.0,
        (p_drv_kw - p_min_kw) / 10.0,
    ] + g_cone
    lbg = [0.0, 0.0, 0.0] + [0.0] * 12 + [0.0] * 8
    ubg = [0.0, 0.0, 0.0] + [ca.inf] * 12 + [ca.inf] * 8

    if open_diff:
        sf = p.front_drive_ratio
        sr = 1.0 - sf
        g_list.extend(
            [
                (t_mot[0] - t_mot[1]) / 100.0,
                (t_mot[2] - t_mot[3]) / 100.0,
                (sr * (t_mot[0] + t_mot[1]) - sf * (t_mot[2] + t_mot[3])) / 100.0,
            ]
        )
        lbg.extend([0.0, 0.0, 0.0])
        ubg.extend([0.0, 0.0, 0.0])
    elif equal_front_drive:
        g_list.append((t_mot[0] - t_mot[1]) / 100.0)
        lbg.append(0.0)
        ubg.append(0.0)

    t_circle = 2.0 * math.pi * R / V
    reg_setup = 0.0
    if do_opt_setup and s_sc is not None:
        # Tiny regularization on brake_bias (inactive in pure steady state) and toe
        reg_setup = (
            1e-6 * ((s_val[5] - p_nom.brake_bias_front) ** 2)
            + 1e-6 * (s_val[3] ** 2 + s_val[4] ** 2)
        )
    x_decision = ca.vertcat(x_sc, s_sc) if (do_opt_setup and s_sc is not None) else x_sc
    nlp = {"x": x_decision, "f": t_circle + reg_setup, "g": ca.vertcat(*g_list)}
    opts = {
        "print_time": 0,
        "ipopt.print_level": 0,
        "ipopt.linear_solver": "mumps",
        "ipopt.hessian_approximation": "exact",
        "ipopt.tol": 1e-8,
        "ipopt.acceptable_tol": 1e-6,
        "ipopt.max_iter": 350,
        "ipopt.sb": "yes",
    }
    solver = ca.nlpsol("skidpad_stat", "ipopt", nlp, opts)

    r_lo = float(r_fixed) if r_fixed is not None else float(r_bounds[0])
    r_hi = float(r_fixed) if r_fixed is not None else float(r_bounds[1])
    r_init = 0.5 * (r_lo + r_hi) if r_fixed is None else float(r_fixed)
    d_lo, d_hi = (0.04, 0.38) if sgn > 0 else (-0.38, -0.04)
    d_init = sgn * 0.22

    lbx = [4.0 / 10.0, d_lo / 0.3, -0.25 / 0.1, -0.18 / 0.1, -0.18 / 0.1, -0.18 / 0.1, -0.18 / 0.1, r_lo / 10.0]
    ubx = [18.0 / 10.0, d_hi / 0.3, 0.25 / 0.1, 0.18 / 0.1, 0.18 / 0.1, 0.18 / 0.1, 0.18 / 0.1, r_hi / 10.0]
    x0 = [10.0 / 10.0, d_init / 0.3, 0.0 / 0.1, 0.005 / 0.1, 0.005 / 0.1, 0.01 / 0.1, 0.01 / 0.1, r_init / 10.0]

    if do_opt_setup:
        b = TRACKSIDE_SETUP_BOUNDS
        # [lltd, camber_f_deg, camber_r_deg, toe_f_deg, toe_r_deg, brake_bias, rake_mm]
        s_lb = np.array([
            b["lltd"][0], b["camber_f_deg"][0], b["camber_r_deg"][0],
            b["toe_f_deg"][0], b["toe_r_deg"][0], b["brake_bias"][0], b["rake_mm"][0],
        ], dtype=float) / setup_scale
        s_ub = np.array([
            b["lltd"][1], b["camber_f_deg"][1], b["camber_r_deg"][1],
            b["toe_f_deg"][1], b["toe_r_deg"][1], b["brake_bias"][1], b["rake_mm"][1],
        ], dtype=float) / setup_scale
        s_0 = np.array([
            np.clip(p_nom.lltd_elastic_f, b["lltd"][0], b["lltd"][1]),
            np.clip(math.degrees(p_nom.static_camber_f_rad), b["camber_f_deg"][0], b["camber_f_deg"][1]),
            np.clip(math.degrees(p_nom.static_camber_r_rad), b["camber_r_deg"][0], b["camber_r_deg"][1]),
            0.0,
            0.0,
            np.clip(p_nom.brake_bias_front, b["brake_bias"][0], b["brake_bias"][1]),
            0.0,
        ], dtype=float) / setup_scale
        lbx.extend(s_lb.tolist())
        ubx.extend(s_ub.tolist())
        x0.extend(s_0.tolist())

    sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    stats = solver.stats()
    x_raw = np.array(sol["x"]).flatten()
    xs = x_raw[:8] * scale
    V_s, d_s, b_s, k_s, R_s = float(xs[0]), float(xs[1]), float(xs[2]), xs[3:7], float(xs[7])
    vx_s = V_s * math.cos(b_s)
    vy_s = V_s * math.sin(b_s)
    r_s = sgn * V_s / R_s
    ax_s = -r_s * vy_s
    ay_s = r_s * vx_s

    resolved_dp: Dict[str, float] = dict(design_overrides) if design_overrides else {}
    opt_setup_dict: Optional[Dict[str, float]] = None
    if do_opt_setup:
        s_opt = x_raw[8:15] * setup_scale
        resolved_dp = {
            "lltd": float(s_opt[0]),
            "k_roll_scale": 1.0,
            "camber_f_deg": float(s_opt[1]),
            "camber_r_deg": float(s_opt[2]),
            "toe_f_deg": float(s_opt[3]),
            "toe_r_deg": float(s_opt[4]),
            "brake_bias_front": float(s_opt[5]),
            "brake_bias": float(s_opt[5]),
            "rake_mm": float(s_opt[6]),
        }
        arb_f, arb_r = _implied_arb_stiffness_nm_m(p_nom, resolved_dp["lltd"], resolved_dp["k_roll_scale"])
        opt_setup_dict = {
            **resolved_dp,
            "arb_stiffness_front_nm_m": round(arb_f, 1),
            "arb_stiffness_rear_nm_m": round(arb_r, 1),
        }

    # Build numeric vehicle with resolved setup for clean post-evaluation & stability check
    veh_eval = build_casadi_vehicle_for_compound(grip_profile, compound, design_overrides=resolved_dp if resolved_dp else None)
    phys = build_physics_model_for_compound(grip_profile, compound, design_overrides=resolved_dp if resolved_dp else None)

    ev_s = veh_eval.eval_forces_from_kappa(vx_s, vy_s, r_s, d_s, k_s, 0.0, ax_s, ay_s)
    t_mot_s = np.array(ev_s["T_motor_4"]).flatten()
    fz_s = np.array(ev_s["Fz_4"]).flatten()
    mu_util_4 = _compute_mu_util_4(
        veh_eval, k_s, ev_s["alpha_4"], ev_s["Fz_4"], ev_s["delta_w_4"],
        ev_s["phi"], ax_s, ev_s["Fx_w_4"], ev_s["Fy_w_4"], ev_s["mu_thermal_4"],
    )

    # Independent verification on exact non-smoothed ter27_physics
    phys_res = phys.evaluate_state_from_kappa(vx_s, vy_s, r_s, d_s, k_s, ax_s, ay_s, n_load_iter=15)
    p_phys_kw = float(phys.compute_hv_power_w(phys_res["torques"], phys_res["omega_w"]) * 1e-3)

    min_clear, wheel_clear = compute_skidpad_cone_clearance_np(
        0.0, sgn * (R_CENTER_M - R_s), -b_s, veh_eval.x_w, veh_eval.y_w, circle_side=turn_dir
    )
    dmz_dbeta = compute_directional_stability_dmz_dbeta(
        veh_eval, vx_s, vy_s, r_s, d_s, k_s, ax_s, ay_s
    )

    out_dict: Dict[str, Any] = {
        "success": bool(stats.get("success", False)),
        "return_status": str(stats.get("return_status", "")),
        "grip_profile": grip_profile,
        "compound": compound,
        "turn_dir": turn_dir,
        "open_diff": open_diff,
        "equal_front_drive": equal_front_drive,
        "optimize_setup": do_opt_setup,
        "optimize_trackside_setup": do_opt_setup,
        "r_eff_m": R_s,
        "t_circle_s": float(2.0 * math.pi * R_s / V_s),
        "v_ms": V_s,
        "v_kmh": V_s * 3.6,
        "ay_g": abs(ay_s) / p.gravity,
        "beta_deg": math.degrees(b_s),
        "delta_deg": math.degrees(d_s),
        "wz_rads": r_s,
        "T_motor_4_nm": [float(v) for v in t_mot_s],
        "kappa_4": [float(v) for v in k_s],
        "fz_4_n": [float(v) for v in fz_s],
        "mu_util_4": [float(v) for v in mu_util_4],
        "mz_tv_nm": float(ev_s["Mz_tv_long"]),
        "p_dc_ecu_kw": float(ev_s["P_dc_ecu_kw"]),
        "p_dc_phys_kw": p_phys_kw,
        "min_cone_clear_m": min_clear,
        "wheel_clear_4_m": [float(v) for v in wheel_clear],
        "dmz_dbeta_nm_deg": dmz_dbeta,
    }
    if opt_setup_dict is not None:
        out_dict["optimal_setup"] = opt_setup_dict
    return out_dict


def build_figure8_skidpad_track(
    l_entry: float = 10.0,
    l_exit: float = 8.0,
    n_entry: int = 8,
    n_circle: int = 24,
    n_exit: int = 6,
) -> Dict[str, Any]:
    """
    Build the exact FSG Figure-8 Skidpad reference path:
      Entry straight -> 2 Right circles -> 2 Left circles -> Exit straight.
    Circles are tangent at (0, 0) with centers (0, -9.125) and (0, +9.125).
    """
    c_len = 2.0 * math.pi * R_CENTER_M
    k_right = -1.0 / R_CENTER_M
    k_left = +1.0 / R_CENTER_M

    s_list: List[np.ndarray] = []
    k_list: List[np.ndarray] = []
    phase_list: List[np.ndarray] = []

    s_cur = 0.0
    s_seg = np.linspace(s_cur, s_cur + l_entry, n_entry, endpoint=False)
    s_list.append(s_seg)
    k_list.append(np.zeros(n_entry))
    phase_list.append(np.full(n_entry, 0, dtype=int))
    s_cur += l_entry

    for ph_id, k_val in [(1, k_right), (2, k_right), (3, k_left), (4, k_left)]:
        s_seg = np.linspace(s_cur, s_cur + c_len, n_circle, endpoint=False)
        s_list.append(s_seg)
        k_list.append(np.full(n_circle, k_val))
        phase_list.append(np.full(n_circle, ph_id, dtype=int))
        s_cur += c_len

    s_seg = np.linspace(s_cur, s_cur + l_exit, n_exit + 1, endpoint=True)
    s_list.append(s_seg)
    k_list.append(np.zeros(n_exit + 1))
    phase_list.append(np.full(n_exit + 1, 5, dtype=int))

    s_grid = np.concatenate(s_list)
    kappa_ref = np.concatenate(k_list)
    phase = np.concatenate(phase_list)
    M = len(s_grid)

    x_ref = np.zeros(M, dtype=float)
    y_ref = np.zeros(M, dtype=float)
    psi_ref = np.zeros(M, dtype=float)

    for i in range(M):
        ph = int(phase[i])
        si = float(s_grid[i])
        if ph == 0:
            x_ref[i] = si - l_entry
            y_ref[i] = 0.0
            psi_ref[i] = 0.0
        elif ph in (1, 2):
            th = (si - l_entry) / R_CENTER_M
            x_ref[i] = R_CENTER_M * math.sin(th)
            y_ref[i] = -R_CENTER_M + R_CENTER_M * math.cos(th)
            psi_ref[i] = -th
        elif ph in (3, 4):
            th = (si - l_entry - 2.0 * c_len) / R_CENTER_M
            x_ref[i] = R_CENTER_M * math.sin(th)
            y_ref[i] = R_CENTER_M - R_CENTER_M * math.cos(th)
            psi_ref[i] = th
        else:
            x_ref[i] = si - (l_entry + 4.0 * c_len)
            y_ref[i] = 0.0
            psi_ref[i] = 0.0

    return {
        "s_grid": s_grid,
        "ds": np.diff(s_grid),
        "kappa_ref": kappa_ref,
        "phase": phase,
        "x_ref": x_ref,
        "y_ref": y_ref,
        "psi_ref": psi_ref,
        "idx_r2_start": int(n_entry + n_circle),
        "idx_r2_end": int(n_entry + 2 * n_circle),
        "idx_l2_start": int(n_entry + 3 * n_circle),
        "idx_l2_end": int(n_entry + 4 * n_circle),
        "idx_cross": int(n_entry + 2 * n_circle),
    }


def solve_skidpad_figure8_casadi(
    grip_profile: str = "FSG_RUBBERED",
    compound: str = "R20",
    v_entry_ms: float = 4.0,
    n_circle: int = 24,
    design_overrides: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Solve the full Official FSG Skidpad Figure-8 OCP with CasADi + IPOPT:
      Entry launch -> 2 Right circles -> Ballistic Figure-8 crossover -> 2 Left circles -> Exit.
    """
    veh = build_casadi_vehicle_for_compound(grip_profile, compound, design_overrides=design_overrides)
    phys = build_physics_model_for_compound(grip_profile, compound, design_overrides=design_overrides)
    p = veh.params
    track = build_figure8_skidpad_track(n_circle=n_circle)

    s_grid = track["s_grid"]
    ds = track["ds"]
    kappa_ref = track["kappa_ref"]
    phase = track["phase"]
    x_ref = track["x_ref"]
    y_ref = track["y_ref"]
    psi_ref = track["psi_ref"]
    M = len(s_grid)

    st_r = solve_skidpad_stationary_casadi(grip_profile, compound, turn_dir="right", design_overrides=design_overrides)
    st_l = solve_skidpad_stationary_casadi(grip_profile, compound, turn_dir="left", design_overrides=design_overrides)

    nx = 13
    z_scale = np.array([10.0, 2.0, 1.0, 1.0, 0.5, 0.3, 0.1, 0.1, 0.1, 0.1, 1000.0, 10.0, 10.0], dtype=float)
    Z_sc = ca.SX.sym("Z_sc", nx, M)
    Z = Z_sc * ca.DM(z_scale)
    p_max_kw = (p.p_dc_max_w - 150.0) * 1e-3
    p_min_kw = (p.p_dc_min_w + 800.0) * 1e-3  # safety margin for exact physics parity (>= -25.0 kW)

    d_state_list: List[Any] = []
    inv_sd_list: List[Any] = []
    g_list: List[Any] = []
    lbg: List[float] = []
    ubg: List[float] = []

    z0 = Z[:, 0]
    g_list.extend([(z0[0] - v_entry_ms) / 10.0, z0[1] / 2.0, z0[2], z0[3], z0[4] / 0.5])
    lbg.extend([0.0] * 5)
    ubg.extend([0.0] * 5)

    state_scale = ca.DM(z_scale[0:5])

    for k in range(M):
        zk = Z[:, k]
        vx, vy, r, ey, ep = zk[0], zk[1], zk[2], zk[3], zk[4]
        delta, kap_4, f_brk = zk[5], zk[6:10], zk[10]
        ax_lt, ay_lt = zk[11], zk[12]

        ev = veh.eval_forces_from_kappa(vx, vy, r, delta, kap_4, f_brk, ax_lt, ay_lt)
        ax_phys = ev["Fx_tot"] / p.mass
        ay_phys = ev["Fy_tot"] / p.mass

        vx_dot = ax_phys + r * vy
        vy_dot = ay_phys - r * vx
        r_dot = ev["Mz_tot"] / p.izz_total

        cos_ep = ca.cos(ep)
        sin_ep = ca.sin(ep)
        k_ref_k = float(kappa_ref[k])
        denom_geom = smax(1.0 - k_ref_k * ey, 0.25, 1e-3)
        s_dot = (vx * cos_ep - vy * sin_ep) / denom_geom
        ey_dot = vx * sin_ep + vy * cos_ep
        ep_dot = r - k_ref_k * s_dot

        inv_sd = 1.0 / smax(s_dot, 0.5, 1e-3)
        inv_sd_list.append(inv_sd)
        d_state_list.append(
            ca.vertcat(vx_dot, vy_dot, r_dot, ey_dot, ep_dot) * inv_sd
        )

        g_list.append((ax_lt - ax_phys) / p.gravity)
        g_list.append((ay_lt - ay_phys) / p.gravity)
        lbg.extend([0.0, 0.0])
        ubg.extend([0.0, 0.0])

        t_mot, t_max, t_min = ev["T_motor_4"], ev["T_max_4"], ev["T_min_4"]
        p_ecu, p_drv = ev["P_dc_ecu_kw"], ev["P_dc_drv_kw"]
        g_list.extend([
            (t_max - t_mot) / 100.0,
            (t_mot - t_min) / 100.0,
            (p_max_kw - p_ecu) / 10.0,
            (p_max_kw - p_drv) / 10.0,
            (p_ecu - p_min_kw) / 10.0,
            (p_drv - p_min_kw) / 10.0,
        ])
        lbg.extend([0.0] * 12)
        ubg.extend([ca.inf] * 12)

        ph = int(phase[k])
        if ph in (1, 2, 3, 4):
            xr, yr, pr = float(x_ref[k]), float(y_ref[k]), float(psi_ref[k])
            X_cg = xr - ey * math.sin(pr)
            Y_cg = yr + ey * math.cos(pr)
            psi_cg = pr + ep
            xw_l, yw_l = _wheel_world_positions(X_cg, Y_cg, psi_cg, veh.x_w, veh.y_w)
            yc_cone = -R_CENTER_M if ph in (1, 2) else R_CENTER_M
            for i in range(4):
                d_c = ca.sqrt(xw_l[i] ** 2 + (yw_l[i] - yc_cone) ** 2)
                g_list.append(d_c - (R_INNER_M + CONE_MARGIN_M))
                g_list.append((R_OUTER_M - CONE_MARGIN_M) - d_c)
                lbg.extend([0.0, 0.0])
                ubg.extend([ca.inf, ca.inf])

    dt_seg_list: List[Any] = []
    for k in range(M - 1):
        hk = float(ds[k])
        defect = (Z[0:5, k + 1] - Z[0:5, k] - 0.5 * hk * (d_state_list[k] + d_state_list[k + 1])) / state_scale
        g_list.append(defect)
        lbg.extend([0.0] * 5)
        ubg.extend([0.0] * 5)
        dt_seg_list.append(0.5 * hk * (inv_sd_list[k] + inv_sd_list[k + 1]))

        ddelta = Z[5, k + 1] - Z[5, k]
        dkap = Z[6:10, k + 1] - Z[6:10, k]
        g_list.append(ddelta / (0.80 * hk))
        lbg.append(-1.0)
        ubg.append(1.0)
        g_list.append(dkap / (0.15 * hk))
        lbg.extend([-1.0] * 4)
        ubg.extend([1.0] * 4)

    t_total_expr = ca.sum1(ca.vertcat(*dt_seg_list))
    reg_expr = 0.0
    for k in range(M - 1):
        dd = (Z[5, k + 1] - Z[5, k]) / 0.3
        dk = (Z[6:10, k + 1] - Z[6:10, k]) / 0.1
        reg_expr += 1e-4 * (dd * dd) + 1e-4 * ca.sumsqr(dk) + 1e-5 * ((Z[10, k] / 1000.0) ** 2)

    lb_node = np.array([2.0, -4.0, -3.0, -1.35, -0.7, -0.38, -0.18, -0.18, -0.18, -0.18, 0.0, -25.0, -25.0]) / z_scale
    ub_node = np.array([22.0, 4.0, 3.0, 1.35, 0.7, 0.38, 0.18, 0.18, 0.18, 0.18, 2500.0, 25.0, 25.0]) / z_scale
    lbx = np.tile(lb_node, (M, 1)).T
    ubx = np.tile(ub_node, (M, 1)).T

    z_r = np.array([
        st_r["v_ms"] * math.cos(math.radians(st_r["beta_deg"])),
        st_r["v_ms"] * math.sin(math.radians(st_r["beta_deg"])),
        st_r["wz_rads"],
        -(R_CENTER_M - st_r["r_eff_m"]),
        -math.radians(st_r["beta_deg"]),
        math.radians(st_r["delta_deg"]),
        *st_r["kappa_4"],
        0.0, 0.0, -st_r["ay_g"] * p.gravity,
    ])
    z_l = np.array([
        st_l["v_ms"] * math.cos(math.radians(st_l["beta_deg"])),
        st_l["v_ms"] * math.sin(math.radians(st_l["beta_deg"])),
        st_l["wz_rads"],
        +(R_CENTER_M - st_l["r_eff_m"]),
        -math.radians(st_l["beta_deg"]),
        math.radians(st_l["delta_deg"]),
        *st_l["kappa_4"],
        0.0, 0.0, +st_l["ay_g"] * p.gravity,
    ])
    z_init = np.zeros((nx, M), dtype=float)
    s_cross = float(s_grid[track["idx_cross"]])
    s_enter = float(s_grid[track["idx_r2_start"] - n_circle])
    s_exit_start = float(s_grid[track["idx_l2_end"]])
    for k in range(M):
        sk = float(s_grid[k])
        ph = int(phase[k])
        if ph == 0:
            frac = sk / max(s_enter, 1.0)
            z_init[:, k] = (1.0 - frac) * np.array([v_entry_ms, 0, 0, 0, 0, 0, 0.03, 0.03, 0.05, 0.05, 0, 3.0, 0]) + frac * z_r
        elif ph in (1, 2, 3, 4):
            w = 0.5 * (1.0 + math.tanh((sk - s_cross) / 3.5))
            z_init[:, k] = (1.0 - w) * z_r + w * z_l
        else:
            frac = (sk - s_exit_start) / max(float(s_grid[-1]) - s_exit_start, 1.0)
            z_init[:, k] = (1.0 - frac) * z_l + frac * np.array([st_l["v_ms"], 0, 0, 0, 0, 0, 0.02, 0.02, 0.02, 0.02, 0, 2.0, 0])
    z_init_sc = np.clip(z_init / z_scale[:, None], lbx + 1e-4, ubx - 1e-4)

    nlp = {"x": ca.reshape(Z_sc, nx * M, 1), "f": t_total_expr + reg_expr, "g": ca.vertcat(*g_list)}
    opts = {
        "print_time": 0,
        "ipopt.print_level": 0,
        "ipopt.linear_solver": "mumps",
        "ipopt.hessian_approximation": "exact",
        "ipopt.tol": 1e-6,
        "ipopt.acceptable_tol": 1e-4,
        "ipopt.max_iter": 800,
        "ipopt.mu_strategy": "adaptive",
        "ipopt.sb": "yes",
    }
    solver = ca.nlpsol("skidpad_fig8", "ipopt", nlp, opts)
    sol = solver(
        x0=z_init_sc.reshape((-1, 1), order="F"),
        lbx=lbx.reshape((-1, 1), order="F"),
        ubx=ubx.reshape((-1, 1), order="F"),
        lbg=lbg,
        ubg=ubg,
    )
    stats = solver.stats()
    z_opt = np.array(sol["x"]).reshape((nx, M), order="F") * z_scale[:, None]

    dt_segs = np.zeros(M - 1, dtype=float)
    t_arr = np.zeros(M, dtype=float)
    X_arr = np.zeros(M, dtype=float)
    Y_arr = np.zeros(M, dtype=float)
    psi_arr = np.zeros(M, dtype=float)
    T4_arr = np.zeros((M, 4), dtype=float)
    pdc_phys = np.zeros(M, dtype=float)
    mz_tv_arr = np.zeros(M, dtype=float)
    ay_g_arr = np.zeros(M, dtype=float)
    beta_deg_arr = np.zeros(M, dtype=float)
    mu_util_arr = np.zeros((M, 4), dtype=float)
    clear_arr = np.full(M, np.nan, dtype=float)
    sdot_arr = np.zeros(M, dtype=float)

    for k in range(M):
        vx, vy, r, ey, ep = z_opt[0:5, k]
        sdot_arr[k] = (vx * math.cos(ep) - vy * math.sin(ep)) / max(1.0 - float(kappa_ref[k]) * ey, 0.25)
        delta, kap_4, f_brk = float(z_opt[5, k]), z_opt[6:10, k], float(z_opt[10, k])
        ax_lt, ay_lt = float(z_opt[11, k]), float(z_opt[12, k])

        xr, yr, pr = float(x_ref[k]), float(y_ref[k]), float(psi_ref[k])
        X_arr[k] = xr - ey * math.sin(pr)
        Y_arr[k] = yr + ey * math.cos(pr)
        psi_arr[k] = pr + ep

        ev = veh.eval_forces_from_kappa(vx, vy, r, delta, kap_4, f_brk, ax_lt, ay_lt)
        T4_arr[k] = np.array(ev["T_motor_4"]).flatten()
        mz_tv_arr[k] = float(ev["Mz_tv_long"])
        ay_g_arr[k] = float(ev["Fy_tot"]) / (p.mass * p.gravity)
        beta_deg_arr[k] = math.degrees(math.atan2(vy, max(vx, 0.5)))

        mu_util_arr[k] = _compute_mu_util_4(
            veh, kap_4, ev["alpha_4"], ev["Fz_4"], ev["delta_w_4"],
            ev["phi"], ax_lt, ev["Fx_w_4"], ev["Fy_w_4"], ev["mu_thermal_4"],
        )

        pres = phys.evaluate_state_from_kappa(vx, vy, r, delta, kap_4, ax_lt, ay_lt, n_load_iter=15)
        fz_p = pres["fz"]
        rho_rad = np.minimum(fz_p / p.k_tire_z, 0.5 * p.wheel_radius)
        r_load_p = p.wheel_radius - rho_rad
        bf = p.brake_bias_front
        hyd_split_p = np.array([0.5 * bf, 0.5 * bf, 0.5 * (1.0 - bf), 0.5 * (1.0 - bf)])
        t_motor_phys = pres["torques"] + f_brk * hyd_split_p * r_load_p
        pdc_phys[k] = float(phys.compute_hv_power_w(t_motor_phys, pres["omega_w"]) * 1e-3)

        ph = int(phase[k])
        if ph in (1, 2):
            clear_arr[k], _ = compute_skidpad_cone_clearance_np(
                X_arr[k], Y_arr[k], psi_arr[k], veh.x_w, veh.y_w, "right"
            )
        elif ph in (3, 4):
            clear_arr[k], _ = compute_skidpad_cone_clearance_np(
                X_arr[k], Y_arr[k], psi_arr[k], veh.x_w, veh.y_w, "left"
            )

        if k < M - 1:
            s_dot_k = (vx * math.cos(ep) - vy * math.sin(ep)) / max(1.0 - float(kappa_ref[k]) * ey, 0.25)
            vx1, vy1, _, ey1, ep1 = z_opt[0:5, k + 1]
            s_dot_1 = (vx1 * math.cos(ep1) - vy1 * math.sin(ep1)) / max(1.0 - float(kappa_ref[k + 1]) * ey1, 0.25)
            dt_segs[k] = 0.5 * float(ds[k]) * (1.0 / max(s_dot_k, 0.5) + 1.0 / max(s_dot_1, 0.5))
            t_arr[k + 1] = t_arr[k] + dt_segs[k]

    t_right = float(t_arr[track["idx_r2_end"]] - t_arr[track["idx_r2_start"]])
    t_left = float(t_arr[track["idx_l2_end"]] - t_arr[track["idx_l2_start"]])
    t_mean = 0.5 * (t_right + t_left)

    mask_timed = (phase == 2) | (phase == 4)

    # Periodic (steady-lap) metrics: the as-timed lap above ends at the crossover timing line,
    # where the OCP borrows speed (surge) that is paid back in the untimed next circle. The
    # periodic lap time uses the mid-circle plateau of the preceding full circle (R1 / L1).
    c_len_ref = 2.0 * math.pi * R_CENTER_M

    def _plateau(ph_id: int) -> np.ndarray:
        idx = np.where(phase == ph_id)[0]
        n = len(idx)
        return idx[int(0.35 * n):int(0.65 * n)]

    idx_r = _plateau(1)
    idx_l = _plateau(3)
    t_right_per = float(c_len_ref / np.median(sdot_arr[idx_r]))
    t_left_per = float(c_len_ref / np.median(sdot_arr[idx_l]))
    plat = np.concatenate([idx_r, idx_l])
    vtot = np.hypot(z_opt[0], z_opt[1])

    return {
        "t_right_periodic_s": t_right_per,
        "t_left_periodic_s": t_left_per,
        "t_mean_periodic_s": 0.5 * (t_right_per + t_left_per),
        "v_plateau_ms": float(np.mean(vtot[plat])),
        "ay_plateau_g": float(np.mean(np.abs(ay_g_arr[plat]))),
        "beta_plateau_deg_R": float(np.mean(beta_deg_arr[idx_r])),
        "beta_plateau_deg_L": float(np.mean(beta_deg_arr[idx_l])),
        "T_motor_plateau_R_nm": [float(v) for v in np.mean(T4_arr[idx_r], axis=0)],
        "T_motor_plateau_L_nm": [float(v) for v in np.mean(T4_arr[idx_l], axis=0)],
        "mz_tv_plateau_R_nm": float(np.mean(mz_tv_arr[idx_r])),
        "mz_tv_plateau_L_nm": float(np.mean(mz_tv_arr[idx_l])),
        "mu_util_plateau_4": [float(v) for v in np.mean(mu_util_arr[plat], axis=0)],
        "p_dc_plateau_kw": float(np.mean(pdc_phys[plat])),
        "crossover_surge_v_max_ms": float(np.max(vtot[track["idx_cross"] - 8:track["idx_cross"] + 3])),
        "success": bool(stats.get("success", False)),
        "return_status": str(stats.get("return_status", "")),
        "grip_profile": grip_profile,
        "compound": compound,
        "t_right_s": t_right,
        "t_left_s": t_left,
        "t_mean_s": t_mean,
        "t_total_s": float(t_arr[-1]),
        "v_mean_timed_ms": float(np.mean(np.hypot(z_opt[0, mask_timed], z_opt[1, mask_timed]))),
        "ay_mean_timed_g": float(np.mean(np.abs(ay_g_arr[mask_timed]))),
        "beta_mean_abs_deg": float(np.mean(np.abs(beta_deg_arr[mask_timed]))),
        "p_dc_min_kw": float(np.min(pdc_phys)),
        "p_dc_max_kw": float(np.max(pdc_phys)),
        "hv_violations": int(np.sum((pdc_phys < -25.0) | (pdc_phys > 80.0))),
        "min_cone_clear_m": float(np.nanmin(clear_arr)),
        "mu_util_timed_mean_4": [float(v) for v in np.mean(mu_util_arr[mask_timed], axis=0)],
        "crossover_v_min_ms": float(np.min(np.hypot(
            z_opt[0, track["idx_cross"] - 3:track["idx_cross"] + 4],
            z_opt[1, track["idx_cross"] - 3:track["idx_cross"] + 4],
        ))),
        "crossover_p_dc_min_kw": float(np.min(
            pdc_phys[track["idx_cross"] - 3:track["idx_cross"] + 4]
        )),
        "trace": {
            "s_m": s_grid.tolist(),
            "t_s": t_arr.tolist(),
            "phase": phase.tolist(),
            "X_m": X_arr.tolist(),
            "Y_m": Y_arr.tolist(),
            "psi_rad": psi_arr.tolist(),
            "vx_ms": z_opt[0].tolist(),
            "vy_ms": z_opt[1].tolist(),
            "wz_rads": z_opt[2].tolist(),
            "delta_deg": np.degrees(z_opt[5]).tolist(),
            "beta_deg": beta_deg_arr.tolist(),
            "ay_g": ay_g_arr.tolist(),
            "T_motor_4_nm": T4_arr.tolist(),
            "Mz_tv_nm": mz_tv_arr.tolist(),
            "P_dc_kw": pdc_phys.tolist(),
            "mu_util_4": mu_util_arr.tolist(),
            "cone_clear_m": [None if np.isnan(v) else float(v) for v in clear_arr],
        },
    }


def solve_accel_75m_casadi(
    grip_profile: str = "FSG_RUBBERED",
    compound: str = "R20",
    n_steps: int = 35,
    design_overrides: Optional[Dict[str, float]] = None,
    optimize_trackside_setup: bool = False,
) -> Dict[str, Any]:
    """
    Solve the Official FS 0-75 m Acceleration OCP with CasADi + IPOPT (time-domain).
    Enforces strict 80.0 kW HV DC ceiling and real motor torque/speed envelope.
    Optionally co-optimizes the 7 mechanical trackside setup parameters:
    [lltd, camber_f_deg, camber_r_deg, toe_f_deg, toe_r_deg, brake_bias, rake_mm].
    """
    p_nom = load_ter27_params()
    dp_init: Dict[str, Any] = dict(design_overrides) if design_overrides else {}
    s_sc: Optional[ca.SX] = None
    setup_scale = np.array([0.5, 2.0, 2.0, 0.5, 0.5, 0.6, 8.0], dtype=float)
    if optimize_trackside_setup:
        s_sc = ca.SX.sym("s_sc", 7)
        s_val = s_sc * ca.DM(setup_scale)
        dp_init["lltd"] = s_val[0]
        dp_init["k_roll_scale"] = 1.0
        dp_init["camber_f_deg"] = s_val[1]
        dp_init["camber_r_deg"] = s_val[2]
        dp_init["toe_f_deg"] = s_val[3]
        dp_init["toe_r_deg"] = s_val[4]
        dp_init["brake_bias_front"] = s_val[5]
        dp_init["rake_mm"] = s_val[6]

    veh = build_casadi_vehicle_for_compound(grip_profile, compound, design_overrides=dp_init if dp_init else None)
    p = veh.params
    p_max_kw = (p.p_dc_max_w - 100.0) * 1e-3

    # Scaled variables: T_75 (1), and per node [s_m/50, vx/20, kf/0.1, kr/0.1, ax_lt/10]
    T_sc = ca.SX.sym("T_sc")
    T_75 = T_sc * 4.0
    nx = 5
    z_scale = np.array([50.0, 20.0, 0.1, 0.1, 10.0], dtype=float)
    Z_sc = ca.SX.sym("Z_sc", nx, n_steps + 1)
    Z = Z_sc * ca.DM(z_scale)
    dt = T_75 / n_steps

    g_list: List[Any] = [Z[0, 0] / 50.0, (Z[1, 0] - 1.5) / 20.0, (Z[0, n_steps] - 75.0) / 50.0]
    lbg: List[float] = [0.0, 0.0, 0.0]
    ubg: List[float] = [0.0, 0.0, 0.0]

    xdot_list: List[Any] = []
    for k in range(n_steps + 1):
        vx_k = Z[1, k]
        kf_k, kr_k = Z[2, k], Z[3, k]
        ax_k = Z[4, k]
        kap_k = ca.vertcat(kf_k, kf_k, kr_k, kr_k)
        ev = veh.eval_forces_from_kappa(vx_k, 0.0, 0.0, 0.0, kap_k, 0.0, ax_k, 0.0)
        ax_b = ev["Fx_tot"] / p.mass
        xdot_list.append(ca.vertcat(vx_k, ax_b))

        g_list.extend([
            (ax_k - ax_b) / 10.0,
            (ev["T_max_4"] - ev["T_motor_4"]) / 100.0,
            ev["T_motor_4"] / 100.0,
            (p_max_kw - ev["P_dc_ecu_kw"]) / 10.0,
            (p_max_kw - ev["P_dc_drv_kw"]) / 10.0,
        ])
        lbg.extend([0.0] + [0.0] * 10)
        ubg.extend([0.0] + [ca.inf] * 10)

    state_sc = ca.DM([50.0, 20.0])
    for k in range(n_steps):
        defect = (Z[0:2, k + 1] - Z[0:2, k] - 0.5 * dt * (xdot_list[k] + xdot_list[k + 1])) / state_sc
        g_list.append(defect)
        lbg.extend([0.0, 0.0])
        ubg.extend([0.0, 0.0])

    reg = 1e-4 * ca.sumsqr((Z[2:4, 1:] - Z[2:4, :-1]) / 0.1)
    if optimize_trackside_setup and s_sc is not None:
        # Tiny regularization on lateral-inactive vars (lltd, brake_bias) for clean conditioning
        reg = reg + 1e-5 * ((s_val[0] - p_nom.lltd_elastic_f) ** 2 + (s_val[5] - p_nom.brake_bias_front) ** 2)
    x_vec = ca.vertcat(T_sc, ca.reshape(Z_sc, nx * (n_steps + 1), 1))
    if optimize_trackside_setup and s_sc is not None:
        x_vec = ca.vertcat(x_vec, s_sc)
    nlp = {"x": x_vec, "f": T_75 + reg, "g": ca.vertcat(*g_list)}
    opts = {
        "print_time": 0,
        "ipopt.print_level": 0,
        "ipopt.linear_solver": "mumps",
        "ipopt.hessian_approximation": "exact",
        "ipopt.tol": 1e-7,
        "ipopt.max_iter": 400,
        "ipopt.sb": "yes",
    }
    solver = ca.nlpsol("accel75_ocp", "ipopt", nlp, opts)

    lb_n = np.array([0.0, 0.2, 0.0, 0.0, -2.0]) / z_scale
    ub_n = np.array([80.0, 45.0, 0.14, 0.14, 20.0]) / z_scale
    lbx = np.concatenate([[2.5 / 4.0], np.tile(lb_n, (n_steps + 1, 1)).T.reshape(-1, order="F")])
    ubx = np.concatenate([[8.0 / 4.0], np.tile(ub_n, (n_steps + 1, 1)).T.reshape(-1, order="F")])
    z0 = np.zeros((nx, n_steps + 1), dtype=float)
    frac = np.linspace(0.0, 1.0, n_steps + 1)
    z0[0] = 75.0 * frac * frac
    z0[1] = 1.5 + 30.0 * frac
    z0[2] = 0.08
    z0[3] = 0.09
    z0[4] = 8.0
    x0 = np.concatenate([[4.0 / 4.0], (z0 / z_scale[:, None]).reshape(-1, order="F")])

    if optimize_trackside_setup:
        b = TRACKSIDE_SETUP_BOUNDS
        s_lb = np.array([
            b["lltd"][0], b["camber_f_deg"][0], b["camber_r_deg"][0],
            b["toe_f_deg"][0], b["toe_r_deg"][0], b["brake_bias"][0], b["rake_mm"][0],
        ], dtype=float) / setup_scale
        s_ub = np.array([
            b["lltd"][1], b["camber_f_deg"][1], b["camber_r_deg"][1],
            b["toe_f_deg"][1], b["toe_r_deg"][1], b["brake_bias"][1], b["rake_mm"][1],
        ], dtype=float) / setup_scale
        s_0 = np.array([
            np.clip(p_nom.lltd_elastic_f, b["lltd"][0], b["lltd"][1]),
            np.clip(math.degrees(p_nom.static_camber_f_rad), b["camber_f_deg"][0], b["camber_f_deg"][1]),
            np.clip(math.degrees(p_nom.static_camber_r_rad), b["camber_r_deg"][0], b["camber_r_deg"][1]),
            0.0,
            0.0,
            np.clip(p_nom.brake_bias_front, b["brake_bias"][0], b["brake_bias"][1]),
            0.0,
        ], dtype=float) / setup_scale
        lbx = np.concatenate([lbx, s_lb])
        ubx = np.concatenate([ubx, s_ub])
        x0 = np.concatenate([x0, s_0])

    sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    stats = solver.stats()
    x_opt = np.array(sol["x"]).flatten()
    t75 = float(x_opt[0] * 4.0)
    n_traj = 1 + nx * (n_steps + 1)
    z_opt = x_opt[1:n_traj].reshape((nx, n_steps + 1), order="F") * z_scale[:, None]

    resolved_dp: Dict[str, float] = dict(design_overrides) if design_overrides else {}
    opt_setup_dict: Optional[Dict[str, float]] = None
    if optimize_trackside_setup:
        s_opt = x_opt[n_traj:n_traj + 7] * setup_scale
        resolved_dp = {
            "lltd": float(s_opt[0]),
            "k_roll_scale": 1.0,
            "camber_f_deg": float(s_opt[1]),
            "camber_r_deg": float(s_opt[2]),
            "toe_f_deg": float(s_opt[3]),
            "toe_r_deg": float(s_opt[4]),
            "brake_bias_front": float(s_opt[5]),
            "brake_bias": float(s_opt[5]),
            "rake_mm": float(s_opt[6]),
        }
        arb_f, arb_r = _implied_arb_stiffness_nm_m(p_nom, resolved_dp["lltd"], 1.0)
        opt_setup_dict = {
            **resolved_dp,
            "arb_stiffness_front_nm_m": round(arb_f, 1),
            "arb_stiffness_rear_nm_m": round(arb_r, 1),
        }

    veh_eval = build_casadi_vehicle_for_compound(grip_profile, compound, design_overrides=resolved_dp if resolved_dp else None)
    pdc_arr = np.zeros(n_steps + 1, dtype=float)
    ax_arr = np.zeros(n_steps + 1, dtype=float)
    for k in range(n_steps + 1):
        ev_n = veh_eval.eval_forces_from_kappa(
            float(z_opt[1, k]), 0.0, 0.0, 0.0,
            np.array([z_opt[2, k], z_opt[2, k], z_opt[3, k], z_opt[3, k]]),
            0.0, float(z_opt[4, k]), 0.0,
        )
        pdc_arr[k] = float(ev_n["P_dc_ecu_kw"])
        ax_arr[k] = float(ev_n["Fx_tot"]) / (p.mass * p.gravity)

    out: Dict[str, Any] = {
        "success": bool(stats.get("success", False)),
        "converged_valid": bool(stats.get("success", False)) and 2.6 < t75 < 7.9,
        "grip_profile": grip_profile,
        "compound": compound,
        "optimize_trackside_setup": optimize_trackside_setup,
        "time_75m_s": t75,
        "v_exit_kmh": float(z_opt[1, -1] * 3.6),
        "peak_ax_g": float(np.max(ax_arr)),
        "max_hv_power_kw": float(np.max(pdc_arr)),
    }
    if opt_setup_dict is not None:
        out["optimal_setup"] = opt_setup_dict
    return out


if __name__ == "__main__":
    import argparse
    import json as _json

    parser = argparse.ArgumentParser(description="CasADi Dynamic Events OCP (Skidpad & Acceleration 75 m)")
    parser.add_argument("--event", type=str, default="all", choices=["skidpad", "accel", "all"])
    parser.add_argument("--grip-profile", type=str, default="FSG_RUBBERED")
    parser.add_argument("--compound", type=str, default="R25B")
    parser.add_argument("--optimize-setup", action="store_true", help="Co-optimize 7 trackside setup parameters")
    args = parser.parse_args()

    out_cli: Dict[str, Any] = {}
    if args.event in ("skidpad", "all"):
        out_cli["skidpad"] = solve_skidpad_stationary_casadi(
            grip_profile=args.grip_profile,
            compound=args.compound,
            optimize_trackside_setup=args.optimize_setup,
        )
    if args.event in ("accel", "all"):
        out_cli["accel_75m"] = solve_accel_75m_casadi(
            grip_profile=args.grip_profile,
            compound=args.compound,
            optimize_trackside_setup=args.optimize_setup,
        )
    print(_json.dumps(out_cli, indent=2), flush=True)
