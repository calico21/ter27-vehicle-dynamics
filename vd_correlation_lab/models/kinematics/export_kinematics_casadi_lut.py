"""
Export Exact 3D TeR26 Kinematics Lookup Tables & Verify C^2 CasADi Cubic B-Spline Surrogate.

Generates in `vd_correlation_lab/results/ter26_extended_correlation/`:
  - `ter26_kinematics_lut.npz`
  - `ter26_kinematics_3d_report.json`

And validates `ter26_kinematics_casadi_loader.py` against the exact 3D solver on held-out
off-grid test points across the full travel `z_w in [-40, +40] mm` and steering rack
`d_rack in [-45, +45] mm` domain, asserting:
  - Loop closure residual < 1e-6 mm
  - Off-grid B-Spline max error < 1e-4 deg (angles) and < 1e-4 mm (displacements)
  - C^2 symbolic CasADi Jacobian & Hessian differentiability.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import casadi as ca
import numpy as np

# Ensure repository root is importable
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vd_correlation_lab.models.kinematics.ter26_double_wishbone_3d import (  # noqa: E402
    TeR26Kinematics3D,
)

OUT_DIR = REPO_ROOT / "vd_correlation_lab" / "results" / "ter26_extended_correlation"
NPZ_PATH = OUT_DIR / "ter26_kinematics_lut.npz"
REPORT_PATH = OUT_DIR / "ter26_kinematics_3d_report.json"


def build_and_export_luts(
    n_z: int = 41,
    n_rack: int = 45,
    n_mean_steer: int = 61,
    z_min_mm: float = -40.0,
    z_max_mm: float = 40.0,
    rack_min_mm: float = -45.0,
    rack_max_mm: float = 45.0,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    kin = TeR26Kinematics3D()

    z_grid_m = np.linspace(z_min_mm * 1e-3, z_max_mm * 1e-3, n_z)
    rack_grid_m = np.linspace(rack_min_mm * 1e-3, rack_max_mm * 1e-3, n_rack)
    mid_z = n_z // 2
    mid_r = n_rack // 2

    # Allocate Front 2D arrays: shape (n_z, n_rack)
    shape_f = (n_z, n_rack)
    fl_camber_deg = np.zeros(shape_f, dtype=np.float64)
    fl_steer_deg = np.zeros(shape_f, dtype=np.float64)
    fl_toe_deg = np.zeros(shape_f, dtype=np.float64)
    fl_damper_disp_mm = np.zeros(shape_f, dtype=np.float64)
    fl_mr_spring = np.zeros(shape_f, dtype=np.float64)
    fl_arb_angle_rad = np.zeros(shape_f, dtype=np.float64)
    fl_mr_arb_rad_m = np.zeros(shape_f, dtype=np.float64)
    fl_rc_z_mm = np.zeros(shape_f, dtype=np.float64)
    fl_svsa_slope = np.zeros(shape_f, dtype=np.float64)
    fl_mech_trail_mm = np.zeros(shape_f, dtype=np.float64)
    fl_scrub_radius_mm = np.zeros(shape_f, dtype=np.float64)
    fl_cp_dx_mm = np.zeros(shape_f, dtype=np.float64)
    fl_cp_dy_mm = np.zeros(shape_f, dtype=np.float64)

    max_grid_res_mm = 0.0

    # Sweep Front Left from center outward for warm starts
    r_order = list(range(mid_r, n_rack)) + list(range(mid_r - 1, -1, -1))
    z_up = list(range(mid_z, n_z))
    z_dn = list(range(mid_z - 1, -1, -1))

    x_center = np.zeros(3, dtype=np.float64)
    tr_center = 0.0
    ta_center = 0.0
    for ir in r_order:
        if ir == mid_r - 1:
            x_center = np.zeros(3, dtype=np.float64)
            tr_center = 0.0
            ta_center = 0.0
        r_val = float(rack_grid_m[ir])
        p0 = kin.fl.solve_pose(0.0, r_val, x_center, tr_center, ta_center)
        x_center = p0["x_state"]
        tr_center = p0["theta_rck_rad"]
        ta_center = p0["theta_arb_rad"]

        for z_seq in (z_up, z_dn):
            x_cur = x_center.copy()
            tr_cur = tr_center
            ta_cur = ta_center
            for iz in z_seq:
                z_val = float(z_grid_m[iz])
                sol = kin.fl.solve_pose_with_derivatives(z_val, r_val, x_cur, tr_cur, ta_cur)
                x_cur = sol["x_state"]
                tr_cur = sol["theta_rck_rad"]
                ta_cur = sol["theta_arb_rad"]
                max_grid_res_mm = max(max_grid_res_mm, sol["max_residual_mm"])

                fl_camber_deg[iz, ir] = sol["camber_deg"]
                fl_steer_deg[iz, ir] = sol["steer_deg"]
                fl_toe_deg[iz, ir] = sol["toe_deg"]
                fl_damper_disp_mm[iz, ir] = sol["damper_disp_mm"]
                fl_mr_spring[iz, ir] = sol["mr_spring"]
                fl_arb_angle_rad[iz, ir] = sol["arb_angle_rad"]
                fl_mr_arb_rad_m[iz, ir] = sol["mr_arb_rad_m"]
                fl_rc_z_mm[iz, ir] = sol["roll_center_z_mm"]
                fl_svsa_slope[iz, ir] = sol["svsa_slope"]
                fl_mech_trail_mm[iz, ir] = sol["mechanical_trail_mm"]
                fl_scrub_radius_mm[iz, ir] = sol["scrub_radius_mm"]
                fl_cp_dx_mm[iz, ir] = sol["cp_dx_m"] * 1e3
                fl_cp_dy_mm[iz, ir] = sol["cp_dy_m"] * 1e3

    # Allocate Rear 1D arrays: shape (n_z,)
    rl_camber_deg = np.zeros(n_z, dtype=np.float64)
    rl_toe_deg = np.zeros(n_z, dtype=np.float64)
    rl_damper_disp_mm = np.zeros(n_z, dtype=np.float64)
    rl_mr_spring = np.zeros(n_z, dtype=np.float64)
    rl_arb_angle_rad = np.zeros(n_z, dtype=np.float64)
    rl_mr_arb_rad_m = np.zeros(n_z, dtype=np.float64)
    rl_rc_z_mm = np.zeros(n_z, dtype=np.float64)
    rl_svsa_slope = np.zeros(n_z, dtype=np.float64)
    rl_cp_dx_mm = np.zeros(n_z, dtype=np.float64)
    rl_cp_dy_mm = np.zeros(n_z, dtype=np.float64)

    for z_seq in (z_up, z_dn):
        x_cur = np.zeros(3, dtype=np.float64)
        tr_cur = 0.0
        ta_cur = 0.0
        for iz in z_seq:
            z_val = float(z_grid_m[iz])
            sol = kin.rl.solve_pose_with_derivatives(z_val, 0.0, x_cur, tr_cur, ta_cur)
            x_cur = sol["x_state"]
            tr_cur = sol["theta_rck_rad"]
            ta_cur = sol["theta_arb_rad"]
            max_grid_res_mm = max(max_grid_res_mm, sol["max_residual_mm"])

            rl_camber_deg[iz] = sol["camber_deg"]
            rl_toe_deg[iz] = sol["toe_deg"]
            rl_damper_disp_mm[iz] = sol["damper_disp_mm"]
            rl_mr_spring[iz] = sol["mr_spring"]
            rl_arb_angle_rad[iz] = sol["arb_angle_rad"]
            rl_mr_arb_rad_m[iz] = sol["mr_arb_rad_m"]
            rl_rc_z_mm[iz] = sol["roll_center_z_mm"]
            rl_svsa_slope[iz] = sol["svsa_slope"]
            rl_cp_dx_mm[iz] = sol["cp_dx_m"] * 1e3
            rl_cp_dy_mm[iz] = sol["cp_dy_m"] * 1e3

    # Build calibrated 1D table: mean_steer_rad -> d_rack_m (at z_w = 0)
    # Find max mean steer achieved at rack = +44 mm (inside [-45, +45] mm)
    p_fl_max = kin.fl.solve_pose(0.0, 0.044)
    p_fr_max = kin.fr.solve_pose(0.0, 0.044)
    max_mean_steer_rad = 0.5 * (p_fl_max["steer_rad"] + p_fr_max["steer_rad"])
    mean_steer_grid_rad = np.linspace(-max_mean_steer_rad, max_mean_steer_rad, n_mean_steer)
    rack_from_mean_steer_m = np.zeros(n_mean_steer, dtype=np.float64)
    mid_s = n_mean_steer // 2

    for s_seq in (list(range(mid_s + 1, n_mean_steer)), list(range(mid_s - 1, -1, -1))):
        r_g = 0.0
        x_l = np.zeros(3, dtype=np.float64)
        x_r = np.zeros(3, dtype=np.float64)
        for idx in s_seq:
            tgt = float(mean_steer_grid_rad[idx])
            r = r_g
            for _ in range(25):
                pl = kin.fl.solve_pose(0.0, r, x_l)
                pr = kin.fr.solve_pose(0.0, r, x_r)
                x_l = pl["x_state"]
                x_r = pr["x_state"]
                ms = 0.5 * (pl["steer_rad"] + pr["steer_rad"])
                err = ms - tgt
                if abs(err) < 1e-12:
                    break
                dr = 1e-5
                plp = kin.fl.solve_pose(0.0, r + dr, x_l)
                prp = kin.fr.solve_pose(0.0, r + dr, x_r)
                dms = (0.5 * (plp["steer_rad"] + prp["steer_rad"]) - ms) / dr
                r -= err / dms
            r_g = r
            rack_from_mean_steer_m[idx] = r

    # Static & stiffness report, full-domain verification, Ackermann & ride curves
    static_summary = kin.compute_static_and_stiffness_report()
    domain_verification = kin.verify_full_domain()
    ackermann_curve = kin.compute_ackermann_curve(z_w_m=0.0)

    # Ride sweep summary curves (-40 to +40 mm in 10 mm steps)
    ride_z_mm = [-40.0, -30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 40.0]
    front_ride_curve: List[Dict[str, float]] = []
    rear_ride_curve: List[Dict[str, float]] = []
    for zmm in ride_z_mm:
        sf = kin.fl.solve_pose_with_derivatives(zmm * 1e-3, 0.0)
        sr = kin.rl.solve_pose_with_derivatives(zmm * 1e-3, 0.0)
        front_ride_curve.append(
            {
                "z_w_mm": zmm,
                "camber_deg": sf["camber_deg"],
                "toe_deg": sf["toe_deg"],
                "damper_disp_mm": sf["damper_disp_mm"],
                "mr_spring": sf["mr_spring"],
                "mr_arb_rad_m": sf["mr_arb_rad_m"],
                "rc_z_mm": sf["roll_center_z_mm"],
                "caster_deg": sf["caster_deg"],
                "kpi_deg": sf["kpi_deg"],
                "scrub_radius_mm": sf["scrub_radius_mm"],
                "mech_trail_mm": sf["mechanical_trail_mm"],
                "max_residual_mm": sf["max_residual_mm"],
            }
        )
        rear_ride_curve.append(
            {
                "z_w_mm": zmm,
                "camber_deg": sr["camber_deg"],
                "toe_deg": sr["toe_deg"],
                "damper_disp_mm": sr["damper_disp_mm"],
                "mr_spring": sr["mr_spring"],
                "mr_arb_rad_m": sr["mr_arb_rad_m"],
                "rc_z_mm": sr["roll_center_z_mm"],
                "max_residual_mm": sr["max_residual_mm"],
            }
        )

    metadata = {
        "vehicle": "TeR26 (TER21 FINAL.xlsx)",
        "coordinate_system": "ISO 8855 axle-local (+X forward, +Y left, +Z up, ground Z=0)",
        "couplings": {
            "front": {"push_pull": "LowerAArm", "u_bar": "Rocker"},
            "rear": {"push_pull": "UpperAArm", "u_bar": "Rocker"},
        },
        "grid": {
            "n_z": n_z,
            "n_rack": n_rack,
            "n_mean_steer": n_mean_steer,
            "z_min_mm": z_min_mm,
            "z_max_mm": z_max_mm,
            "rack_min_mm": rack_min_mm,
            "rack_max_mm": rack_max_mm,
            "max_mean_steer_deg": math.degrees(max_mean_steer_rad),
            "max_grid_residual_mm": max_grid_res_mm,
        },
        "params": kin.params,
        "static_summary": static_summary,
    }

    # Save NPZ lookup tables
    np.savez_compressed(
        NPZ_PATH,
        z_grid_m=z_grid_m,
        rack_grid_m=rack_grid_m,
        mean_steer_grid_rad=mean_steer_grid_rad,
        rack_from_mean_steer_m=rack_from_mean_steer_m,
        fl_camber_deg=fl_camber_deg,
        fl_steer_deg=fl_steer_deg,
        fl_toe_deg=fl_toe_deg,
        fl_damper_disp_mm=fl_damper_disp_mm,
        fl_mr_spring=fl_mr_spring,
        fl_arb_angle_rad=fl_arb_angle_rad,
        fl_mr_arb_rad_m=fl_mr_arb_rad_m,
        fl_rc_z_mm=fl_rc_z_mm,
        fl_svsa_slope=fl_svsa_slope,
        fl_mech_trail_mm=fl_mech_trail_mm,
        fl_scrub_radius_mm=fl_scrub_radius_mm,
        fl_cp_dx_mm=fl_cp_dx_mm,
        fl_cp_dy_mm=fl_cp_dy_mm,
        rl_camber_deg=rl_camber_deg,
        rl_toe_deg=rl_toe_deg,
        rl_damper_disp_mm=rl_damper_disp_mm,
        rl_mr_spring=rl_mr_spring,
        rl_arb_angle_rad=rl_arb_angle_rad,
        rl_mr_arb_rad_m=rl_mr_arb_rad_m,
        rl_rc_z_mm=rl_rc_z_mm,
        rl_svsa_slope=rl_svsa_slope,
        rl_cp_dx_mm=rl_cp_dx_mm,
        rl_cp_dy_mm=rl_cp_dy_mm,
        metadata_json=np.array(json.dumps(metadata)),
    )

    # =========================================================================
    # Held-Out Off-Grid C^2 CasADi B-Spline Validation
    # =========================================================================
    sys.path.insert(0, str(OUT_DIR))
    from ter26_kinematics_casadi_loader import load_ter26_kinematics  # noqa: E402

    loader = load_ter26_kinematics(NPZ_PATH)

    rng = np.random.default_rng(20260418)
    n_test_front = 120
    n_test_rear = 60
    z_test_f = rng.uniform(-0.0395, 0.0395, size=n_test_front)
    r_test_f = rng.uniform(-0.0445, 0.0445, size=n_test_front)
    z_test_r = rng.uniform(-0.0395, 0.0395, size=n_test_rear)

    err_fl_camber_deg = 0.0
    err_fr_camber_deg = 0.0
    err_fl_steer_deg = 0.0
    err_fr_steer_deg = 0.0
    err_fl_damper_mm = 0.0
    err_fl_rc_mm = 0.0
    err_fl_mr = 0.0
    err_fl_mr_arb = 0.0
    err_fl_trail_mm = 0.0
    err_fl_scrub_mm = 0.0

    for zv, rv in zip(z_test_f, r_test_f):
        ref_l = kin.fl.solve_pose_with_derivatives(float(zv), float(rv))
        ref_r = kin.fr.solve_pose_with_derivatives(float(zv), float(rv))

        err_fl_camber_deg = max(err_fl_camber_deg, abs(float(loader.fl_camber_deg(zv, rv)) - ref_l["camber_deg"]))
        err_fr_camber_deg = max(err_fr_camber_deg, abs(float(loader.fr_camber_deg(zv, rv)) - ref_r["camber_deg"]))
        err_fl_steer_deg = max(err_fl_steer_deg, abs(float(loader.fl_steer_deg(zv, rv)) - ref_l["steer_deg"]))
        err_fr_steer_deg = max(err_fr_steer_deg, abs(float(loader.fr_steer_deg(zv, rv)) - ref_r["steer_deg"]))
        err_fl_damper_mm = max(err_fl_damper_mm, abs(float(loader.fl_damper_disp_mm(zv, rv)) - ref_l["damper_disp_mm"]))
        err_fl_rc_mm = max(err_fl_rc_mm, abs(float(loader.fl_rc_z_mm(zv, rv)) - ref_l["roll_center_z_mm"]))
        err_fl_mr = max(err_fl_mr, abs(float(loader.fl_mr_spring(zv, rv)) - ref_l["mr_spring"]))
        err_fl_mr_arb = max(err_fl_mr_arb, abs(float(loader.fl_mr_arb_rad_m(zv, rv)) - ref_l["mr_arb_rad_m"]))
        err_fl_trail_mm = max(err_fl_trail_mm, abs(float(loader.fl_mech_trail_mm(zv, rv)) - ref_l["mechanical_trail_mm"]))
        err_fl_scrub_mm = max(err_fl_scrub_mm, abs(float(loader.fl_scrub_radius_mm(zv, rv)) - ref_l["scrub_radius_mm"]))

    err_rl_camber_deg = 0.0
    err_rl_toe_deg = 0.0
    err_rl_damper_mm = 0.0
    err_rl_rc_mm = 0.0
    err_rl_mr = 0.0
    err_rl_mr_arb = 0.0

    for zv in z_test_r:
        ref = kin.rl.solve_pose_with_derivatives(float(zv), 0.0)
        err_rl_camber_deg = max(err_rl_camber_deg, abs(float(loader.rear_camber_deg(zv)) - ref["camber_deg"]))
        err_rl_toe_deg = max(err_rl_toe_deg, abs(float(loader.rear_toe_deg(zv)) - ref["toe_deg"]))
        err_rl_damper_mm = max(err_rl_damper_mm, abs(float(loader.rear_damper_disp_mm(zv)) - ref["damper_disp_mm"]))
        err_rl_rc_mm = max(err_rl_rc_mm, abs(float(loader.rear_rc_z_mm(zv)) - ref["roll_center_z_mm"]))
        err_rl_mr = max(err_rl_mr, abs(float(loader.rear_mr_spring(zv)) - ref["mr_spring"]))
        err_rl_mr_arb = max(err_rl_mr_arb, abs(float(loader.rear_mr_arb_rad_m(zv)) - ref["mr_arb_rad_m"]))

    # Mean steer -> rack inverse mapping off-grid check
    err_rack_inv_mm = 0.0
    for st_deg in np.linspace(-25.0, 25.0, 41):
        st_rad = math.radians(float(st_deg))
        r_pred = float(loader.rack_from_mean_steer_m(st_rad))
        pl = kin.fl.solve_pose(0.0, r_pred)
        pr = kin.fr.solve_pose(0.0, r_pred)
        st_actual = 0.5 * (pl["steer_rad"] + pr["steer_rad"])
        # Convert steer error to equivalent rack error
        err_rack_inv_mm = max(err_rack_inv_mm, abs(math.degrees(st_actual - st_rad)))

    # Verify C^2 symbolic CasADi Jacobian and Hessian evaluation
    z_sym = ca.MX.sym("z_w")
    r_sym = ca.MX.sym("d_rack")
    u_sym = ca.vertcat(z_sym, r_sym)
    g_sym = loader.fl_camber_rad(z_sym, r_sym)
    H_sym, grad_sym = ca.hessian(g_sym, u_sym)
    f_hess = ca.Function("test_c2", [u_sym], [g_sym, grad_sym, H_sym])
    g_val, grad_val, H_val = f_hess([0.015, 0.018])
    c2_hessian_finite = bool(
        np.all(np.isfinite(np.array(g_val)))
        and np.all(np.isfinite(np.array(grad_val)))
        and np.all(np.isfinite(np.array(H_val)))
    )

    max_angle_err_deg = max(
        err_fl_camber_deg, err_fr_camber_deg, err_fl_steer_deg, err_fr_steer_deg,
        err_rl_camber_deg, err_rl_toe_deg, err_rack_inv_mm,
    )
    max_disp_err_mm = max(
        err_fl_damper_mm, err_fl_rc_mm, err_fl_trail_mm, err_fl_scrub_mm,
        err_rl_damper_mm, err_rl_rc_mm,
    )

    bspline_validation = {
        "n_test_front_offgrid": n_test_front,
        "n_test_rear_offgrid": n_test_rear,
        "max_err_fl_camber_deg": err_fl_camber_deg,
        "max_err_fr_camber_deg": err_fr_camber_deg,
        "max_err_fl_steer_deg": err_fl_steer_deg,
        "max_err_fr_steer_deg": err_fr_steer_deg,
        "max_err_fl_damper_mm": err_fl_damper_mm,
        "max_err_fl_rc_mm": err_fl_rc_mm,
        "max_err_fl_mr_spring": err_fl_mr,
        "max_err_fl_mr_arb_rad_m": err_fl_mr_arb,
        "max_err_fl_mech_trail_mm": err_fl_trail_mm,
        "max_err_fl_scrub_radius_mm": err_fl_scrub_mm,
        "max_err_rl_camber_deg": err_rl_camber_deg,
        "max_err_rl_toe_deg": err_rl_toe_deg,
        "max_err_rl_damper_mm": err_rl_damper_mm,
        "max_err_rl_rc_mm": err_rl_rc_mm,
        "max_err_rl_mr_spring": err_rl_mr,
        "max_err_rl_mr_arb_rad_m": err_rl_mr_arb,
        "max_err_mean_steer_inv_deg": err_rack_inv_mm,
        "max_angle_err_deg": max_angle_err_deg,
        "max_disp_err_mm": max_disp_err_mm,
        "pass_angle_lt_1e4_deg": bool(max_angle_err_deg < 1e-4),
        "pass_disp_lt_1e4_mm": bool(max_disp_err_mm < 1e-4),
        "c2_hessian_finite": c2_hessian_finite,
    }

    report = {
        "metadata": metadata,
        "domain_verification": domain_verification,
        "bspline_validation": bspline_validation,
        "ackermann_curve_zw0": ackermann_curve,
        "front_ride_curve": front_ride_curve,
        "rear_ride_curve": rear_ride_curve,
        "elapsed_s": time.perf_counter() - t0,
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    rep = build_and_export_luts()
    print("=== EXPORT COMPLETE ===", flush=True)
    print(f"NPZ:    {NPZ_PATH}", flush=True)
    print(f"Report: {REPORT_PATH}", flush=True)
    print(f"Time:   {rep['elapsed_s']:.2f} s", flush=True)
    print("=== DOMAIN CLOSURE VERIFICATION ===", flush=True)
    print(json.dumps(rep["domain_verification"], indent=2), flush=True)
    print("=== OFF-GRID C^2 CASADI B-SPLINE VALIDATION ===", flush=True)
    print(json.dumps(rep["bspline_validation"], indent=2), flush=True)

    assert rep["domain_verification"]["closure_pass_lt_1e6_mm"], "Loop closure > 1e-6 mm!"
    assert rep["bspline_validation"]["pass_angle_lt_1e4_deg"], "B-Spline angle error >= 1e-4 deg!"
    assert rep["bspline_validation"]["pass_disp_lt_1e4_mm"], "B-Spline disp error >= 1e-4 mm!"
    assert rep["bspline_validation"]["c2_hessian_finite"], "CasADi C^2 Hessian check failed!"
    print("ALL ASSERTIONS PASSED.", flush=True)
