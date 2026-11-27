#!/usr/bin/env python3
"""
Real FSG_RUBBERED Track Grip Calibration Anchored to TeR26 RWD Skidpad = 5.50 s.

Physical Rationale:
  - TeR26 RWD October track SysID (`ter26_sysid_params.json`, `COLD_TEST_OCT`)
    identified cold-asphalt grip (`mu_y_scale = 0.7739` on nominal 0.64 ->
    `lambda_muy = 0.4953`), which was propagated into `ter27_4wd/tires.json`
    as `lambda_muy = 0.47`, `lambda_mux = 0.68`.
  - Applying an additional 0.88 multiplier on top of `0.47` (`0.4136`) double-
    penalized grip for rubbered Hockenheim FSG asphalt, making TeR26 RWD ~6.01 s
    and TeR27 4WD ~5.45 s.
  - This script keeps `COLD_TEST_OCT` (0.774) completely untouched and solves via
    SciPy Brent root-finding + CasADi 3.7.2 / IPOPT on the validated TeR26 RWD
    3D double-wishbone model (`TER21 FINAL.xlsx`, 46% F weight dist, open diff,
    official FSG Skidpad cones D_inner=15.25 m, D_outer=21.25 m, margin >= 0.15 m)
    for the exact Hockenheim rubbered grip multiplier `mu_fsg_scale` (referenced
    to the shared `TeR27` Hoosier R20 base `lambda_mux=0.68, lambda_muy=0.47`)
    and bulk tire temperature `T_fsg_c` that reproduces:
        t_skidpad(TeR26 RWD) = 5.500 s +/- 0.03 s.

Output:
  - outputs/validation_vault/03_ter27_torque_vectoring_and_maps/fsg_grip_calibration_from_ter26.json
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import casadi as ca
import numpy as np
from scipy.optimize import brentq

ORACLE_DIR = Path(__file__).resolve().parent
LAB_DIR = ORACLE_DIR.parents[1]
REPO_ROOT = LAB_DIR.parent
VAULT_03_DIR = REPO_ROOT / "outputs" / "validation_vault" / "03_ter27_torque_vectoring_and_maps"
OUT_JSON = VAULT_03_DIR / "fsg_grip_calibration_from_ter26.json"
SYSID_JSON = LAB_DIR / "results" / "ter26_extended_correlation" / "ter26_sysid_params.json"

for _p in (str(REPO_ROOT), str(LAB_DIR), str(ORACLE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from vd_correlation_lab.models.vehicle_14dof.ter26_high_fidelity_model import (  # noqa: E402
    TeR26HighFidelityCasADi,
    default_sysid_theta,
    load_ter26_params,
    theta_dict_to_vec,
)
from vd_correlation_lab.models.vehicle_14dof.ter27_casadi_model import thermal_grip_multiplier_smooth  # noqa: E402
from vd_correlation_lab.models.vehicle_14dof.ter27_physics import load_ter27_params  # noqa: E402

# Official FSG Skidpad geometry
D_INNER_M: float = 15.25
D_OUTER_M: float = 21.25
R_INNER_M: float = 0.5 * D_INNER_M   # 7.625 m
R_OUTER_M: float = 0.5 * D_OUTER_M   # 10.625 m
R_CENTER_M: float = 9.125            # Official circle centerline radius
CONE_MARGIN_M: float = 0.150         # Minimum 4-wheel clearance to cones

# Hockenheim FSG August rubbered track thermal & aero state
T_TRACK_FSG_C: float = 45.0
T_TIRE_FSG_C: float = 75.0           # Warm Hoosier R20 bulk temp in optimal 70-85 C window
ENV_AERO_CL_SCALE_FSG: float = 0.78  # Yawed skidpad aero efficiency
TARGET_TER26_SKIDPAD_S: float = 5.500


def _compute_cone_clearance_4(
    r_eff: float, beta_rad: float, x_w: np.ndarray, y_w: np.ndarray
) -> float:
    """Exact minimum 4-wheel cone clearance (m) on a left skidpad circle."""
    psi = -beta_rad
    cp, sp = math.cos(psi), math.sin(psi)
    y_cg = R_CENTER_M - r_eff
    min_c = 1e9
    for i in range(4):
        xw = float(x_w[i]) * cp - float(y_w[i]) * sp
        yw = y_cg + float(x_w[i]) * sp + float(y_w[i]) * cp
        d = math.hypot(xw, yw - R_CENTER_M)
        min_c = min(min_c, d - R_INNER_M, R_OUTER_M - d)
    return float(min_c)


def solve_ter26_rwd_skidpad_ss(
    model: TeR26HighFidelityCasADi,
    sysid_params: Dict[str, float],
    lmux_base_ref: float,
    lmuy_base_ref: float,
    mu_fsg_scale: float,
    t_tire_c: float = T_TIRE_FSG_C,
    cl_scale: float = ENV_AERO_CL_SCALE_FSG,
    r_fixed: Optional[float] = R_CENTER_M,
) -> Dict[str, Any]:
    """
    Solve steady-state skidpad equilibrium for validated TeR26 RWD (open diff,
    3D double-wishbone kinematics, 46% F weight dist, CAD h_cg = 0.270 m).
    """
    p = model.params
    th = dict(default_sysid_theta(p))
    # Map shared reference tire base * mu_fsg_scale onto TeR26 MF6.2 scale factors
    lmux_eff = lmux_base_ref * mu_fsg_scale
    lmuy_eff = lmuy_base_ref * mu_fsg_scale
    th["mu_x_scale"] = lmux_eff / p.lambda_mux_base
    th["mu_y_scale"] = lmuy_eff / p.lambda_muy_base
    # Preserve validated TeR26 SysID cornering stiffness, camber & steering compliance
    th["kya_scale"] = float(sysid_params["kya_scale"])
    th["camber_scale"] = float(sysid_params["camber_scale"])
    th["k_steer_comp"] = float(sysid_params["k_steer_comp"])
    th["h_cg"] = float(p.h_cg)  # 0.270 m CAD nominal
    th["cl_a"] = float(sysid_params["cl_a"]) * float(cl_scale)
    th["cd_a"] = float(sysid_params["cd_a"])
    th_vec = theta_dict_to_vec(th)

    x = ca.MX.sym("x", 5)  # [V, delta, beta, ax_cmd_drive, R]
    V, delta, beta, ax_cmd, R = x[0], x[1], x[2], x[3], x[4]
    vx = V * ca.cos(beta)
    vy = V * ca.sin(beta)
    wz = V / R
    ax_ss = -wz * vy
    ay_ss = wz * vx

    af_kin = delta - ca.atan2(vy + p.a_cog * wz, vx)
    ar_kin = -ca.atan2(vy - p.b_cog * wz, vx)
    state = ca.vertcat(vx, vy, wz, af_kin, ar_kin)
    u = ca.vertcat(delta, ax_cmd, ax_ss, ay_ss)
    t4 = ca.DM([float(t_tire_c)] * 4)
    xdot, _ = model.fn_ode(state, u, t4, th_vec)

    # 4-wheel cone clearance constraints
    psi = -beta
    cp, sp = ca.cos(psi), ca.sin(psi)
    y_cg = R_CENTER_M - R
    g_cone = []
    for i in range(4):
        xw = float(model.x_w[i]) * cp - float(model.y_w[i]) * sp
        yw = y_cg + float(model.x_w[i]) * sp + float(model.y_w[i]) * cp
        d = ca.sqrt(xw * xw + (yw - R_CENTER_M) ** 2)
        g_cone.append(d - (R_INNER_M + CONE_MARGIN_M))
        g_cone.append((R_OUTER_M - CONE_MARGIN_M) - d)

    g = ca.vertcat(xdot[0], xdot[1], xdot[2], *g_cone)
    nlp = {"x": x, "f": 2.0 * math.pi * R / V, "g": g}
    opts = {
        "print_time": 0,
        "ipopt.print_level": 0,
        "ipopt.sb": "yes",
        "ipopt.tol": 1e-8,
        "ipopt.acceptable_tol": 1e-6,
        "ipopt.max_iter": 300,
    }
    solver = ca.nlpsol("ter26_skidpad", "ipopt", nlp, opts)

    r_lo = 8.30 if r_fixed is None else float(r_fixed)
    r_hi = 9.60 if r_fixed is None else float(r_fixed)
    r_init = 8.45 if r_fixed is None else float(r_fixed)

    sol = solver(
        x0=[10.5, 0.22, -0.02, 0.4, r_init],
        lbx=[4.0, 0.05, -0.22, -1.0, r_lo],
        ubx=[18.0, 0.38, 0.22, 4.0, r_hi],
        lbg=[0.0, 0.0, 0.0] + [0.0] * 8,
        ubg=[0.0, 0.0, 0.0] + [ca.inf] * 8,
    )
    stats = solver.stats()
    xs = np.array(sol["x"]).flatten()
    v_s, d_s, b_s, ax_s, r_s = [float(v) for v in xs]
    t_circle = 2.0 * math.pi * r_s / v_s
    ay_g = (v_s * v_s / r_s) / p.gravity
    min_clear = _compute_cone_clearance_4(r_s, b_s, model.x_w, model.y_w)

    return {
        "success": bool(stats.get("success", False)),
        "return_status": str(stats.get("return_status", "")),
        "mu_fsg_scale": float(mu_fsg_scale),
        "t_tire_c": float(t_tire_c),
        "mu_thermal": float(thermal_grip_multiplier_smooth(t_tire_c)),
        "lmux_eff": float(lmux_eff),
        "lmuy_eff": float(lmuy_eff),
        "t_skidpad_s": float(t_circle),
        "v_ms": float(v_s),
        "v_kmh": float(v_s * 3.6),
        "ay_g": float(ay_g),
        "r_eff_m": float(r_s),
        "beta_deg": float(math.degrees(b_s)),
        "delta_deg": float(math.degrees(d_s)),
        "ax_cmd_ms2": float(ax_s),
        "min_clear_m": float(min_clear),
    }


def calibrate_fsg_rubbered() -> Dict[str, Any]:
    t0 = time.perf_counter()
    print("=== FSG_RUBBERED Grip Calibration from TeR26 RWD (Target = 5.500 s) ===", flush=True)

    p26 = load_ter26_params(use_cad_3d=True)
    p27 = load_ter27_params()
    sysid_doc = json.loads(SYSID_JSON.read_text(encoding="utf-8"))
    sysid_params = sysid_doc["identified_params"]

    model26 = TeR26HighFidelityCasADi(
        params=p26, use_3d_kinematics=True, use_thermal_coupling=True
    )

    lmux_ref = float(p27.lambda_mux_base)  # 0.68
    lmuy_ref = float(p27.lambda_muy_base)  # 0.47

    # 1. Verify COLD_TEST_OCT reference (untouched October SysID: mu_y_scale=0.7739 on 0.64)
    cold_oct_mu_equiv = float(p26.lambda_muy_base) / lmuy_ref  # 0.4953 / 0.47 = 1.0538
    cold_oct_res = solve_ter26_rwd_skidpad_ss(
        model26,
        sysid_params,
        lmux_ref,
        lmuy_ref,
        mu_fsg_scale=cold_oct_mu_equiv,
        t_tire_c=35.0,  # Cold October asphalt tire bulk temp
        cl_scale=0.78,
        r_fixed=R_CENTER_M,
    )
    print(
        f"[COLD_TEST_OCT ref] T_tire=35.0C (mu_th={cold_oct_res['mu_thermal']:.3f}) -> "
        f"TeR26 RWD t={cold_oct_res['t_skidpad_s']:.3f}s ay={cold_oct_res['ay_g']:.3f}g",
        flush=True,
    )

    # 2. Check legacy double-penalized FSG_RUBBERED (0.88 on 0.47 -> 0.4136)
    legacy_res = solve_ter26_rwd_skidpad_ss(
        model26,
        sysid_params,
        lmux_ref,
        lmuy_ref,
        mu_fsg_scale=0.88,
        t_tire_c=T_TIRE_FSG_C,
        cl_scale=ENV_AERO_CL_SCALE_FSG,
        r_fixed=R_CENTER_M,
    )
    print(
        f"[Legacy FSG 0.88]   lmuy_eff={legacy_res['lmuy_eff']:.4f} -> "
        f"TeR26 RWD t={legacy_res['t_skidpad_s']:.3f}s ay={legacy_res['ay_g']:.3f}g (too slow by +{legacy_res['t_skidpad_s'] - TARGET_TER26_SKIDPAD_S:.3f}s)",
        flush=True,
    )

    # 3. Exact Brent root-finding for mu_fsg_scale giving TeR26 RWD = 5.500 s at R=9.125 m
    def _residual(mu_sc: float) -> float:
        r = solve_ter26_rwd_skidpad_ss(
            model26,
            sysid_params,
            lmux_ref,
            lmuy_ref,
            mu_fsg_scale=float(mu_sc),
            t_tire_c=T_TIRE_FSG_C,
            cl_scale=ENV_AERO_CL_SCALE_FSG,
            r_fixed=R_CENTER_M,
        )
        return float(r["t_skidpad_s"] - TARGET_TER26_SKIDPAD_S)

    mu_fsg_star = float(brentq(_residual, 0.90, 1.30, xtol=1e-5, rtol=1e-5))
    cal_center = solve_ter26_rwd_skidpad_ss(
        model26,
        sysid_params,
        lmux_ref,
        lmuy_ref,
        mu_fsg_scale=mu_fsg_star,
        t_tire_c=T_TIRE_FSG_C,
        cl_scale=ENV_AERO_CL_SCALE_FSG,
        r_fixed=R_CENTER_M,
    )
    cal_tight = solve_ter26_rwd_skidpad_ss(
        model26,
        sysid_params,
        lmux_ref,
        lmuy_ref,
        mu_fsg_scale=mu_fsg_star,
        t_tire_c=T_TIRE_FSG_C,
        cl_scale=ENV_AERO_CL_SCALE_FSG,
        r_fixed=None,
    )

    print(
        f"[Calibrated FSG_RUBBERED] mu_fsg_scale={mu_fsg_star:.4f} | T_fsg={T_TIRE_FSG_C:.1f}C | "
        f"lmuy_eff={cal_center['lmuy_eff']:.4f} lmux_eff={cal_center['lmux_eff']:.4f}",
        flush=True,
    )
    print(
        f"  TeR26 RWD @ R=9.125m: t={cal_center['t_skidpad_s']:.4f}s | ay={cal_center['ay_g']:.3f}g | "
        f"clear={cal_center['min_clear_m']:.3f}m ({cal_center['return_status']})",
        flush=True,
    )
    print(
        f"  TeR26 RWD @ R*={cal_tight['r_eff_m']:.3f}m: t={cal_tight['t_skidpad_s']:.4f}s | "
        f"ay={cal_tight['ay_g']:.3f}g | clear={cal_tight['min_clear_m']:.3f}m ({cal_tight['return_status']})",
        flush=True,
    )

    report = {
        "calibration_method": "SciPy brentq + CasADi 3.7.2/IPOPT on validated TeR26 RWD 3D double-wishbone model",
        "target_ter26_rwd_skidpad_s": TARGET_TER26_SKIDPAD_S,
        "tolerance_s": 0.03,
        "mu_fsg_scale": round(mu_fsg_star, 6),
        "T_fsg_c": T_TIRE_FSG_C,
        "T_track_fsg_c": T_TRACK_FSG_C,
        "env_aero_cl_scale": ENV_AERO_CL_SCALE_FSG,
        "mu_thermal_at_T_fsg": round(cal_center["mu_thermal"], 6),
        "ter27_base_lambda_mux": lmux_ref,
        "ter27_base_lambda_muy": lmuy_ref,
        "fsg_effective_lambda_mux": round(cal_center["lmux_eff"], 6),
        "fsg_effective_lambda_muy": round(cal_center["lmuy_eff"], 6),
        "ter26_rwd_skidpad_s": round(cal_center["t_skidpad_s"], 4),
        "ter26_ay_g": round(cal_center["ay_g"], 4),
        "ter26_r_eff_m": round(cal_center["r_eff_m"], 4),
        "ter26_v_ms": round(cal_center["v_ms"], 4),
        "ter26_beta_deg": round(cal_center["beta_deg"], 4),
        "ter26_delta_deg": round(cal_center["delta_deg"], 4),
        "min_clear_m": round(cal_center["min_clear_m"], 4),
        "ter26_rwd_tight_radius_reference": {
            "t_skidpad_s": round(cal_tight["t_skidpad_s"], 4),
            "r_eff_m": round(cal_tight["r_eff_m"], 4),
            "ay_g": round(cal_tight["ay_g"], 4),
            "min_clear_m": round(cal_tight["min_clear_m"], 4),
        },
        "cold_test_oct_preserved": {
            "sysid_mu_y_scale_untouched": round(float(sysid_params["mu_y_scale"]), 6),
            "ter26_tires_json_lambda_muy_untouched": round(float(p26.lambda_muy_base), 6),
            "ter26_cold_oct_skidpad_s": round(cold_oct_res["t_skidpad_s"], 4),
            "ter26_cold_oct_ay_g": round(cold_oct_res["ay_g"], 4),
        },
        "legacy_0p88_comparison": {
            "legacy_env_mu_scale": 0.88,
            "legacy_lmuy_eff": round(legacy_res["lmuy_eff"], 6),
            "legacy_ter26_rwd_skidpad_s": round(legacy_res["t_skidpad_s"], 4),
            "legacy_ter26_ay_g": round(legacy_res["ay_g"], 4),
        },
        "geometry": {
            "d_inner_m": D_INNER_M,
            "d_outer_m": D_OUTER_M,
            "r_inner_m": R_INNER_M,
            "r_outer_m": R_OUTER_M,
            "r_center_m": R_CENTER_M,
            "cone_margin_min_m": CONE_MARGIN_M,
        },
        "runtime_s": round(time.perf_counter() - t0, 2),
    }

    VAULT_03_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved: {OUT_JSON}", flush=True)
    return report


if __name__ == "__main__":
    calibrate_fsg_rubbered()
