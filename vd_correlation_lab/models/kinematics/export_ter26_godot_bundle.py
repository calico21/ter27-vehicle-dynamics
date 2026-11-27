"""
Export Validated TeR26 Godot Asset Bundle (`ter26_validated_spec.json`).

Consolidates:
  1. SysID-identified parameters from `ter26_sysid_params.json`.
  2. Exact 3D spatial double-wishbone kinematics & stiffness metrics from
     `ter26_double_wishbone_3d.py` (`TER21 FINAL.xlsx`) and `ter26_kinematics_lut.npz`.
  3. All 17 left and right 3D hardpoints in ISO/ROS (axle-ground and CG-relative)
     and Godot (+X right, +Y up, -Z forward) coordinate frames, plus the exact
     `double_wishbone_kinematics` dictionary consumed by `DoubleWishboneKinematics.gd`.

Pure ASCII output, zero emojis.
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

from vd_correlation_lab.models.kinematics.ter26_double_wishbone_3d import (  # noqa: E402
    FRONT_LEFT_HARDPOINTS_MM,
    REAR_LEFT_HARDPOINTS_MM,
    TER26_VEHICLE_PARAMS,
    TeR26Kinematics3D,
)

SYSID_PATH = (
    REPO_ROOT
    / "vd_correlation_lab"
    / "results"
    / "ter26_extended_correlation"
    / "ter26_sysid_params.json"
)
NPZ_PATH = (
    REPO_ROOT
    / "vd_correlation_lab"
    / "results"
    / "ter26_extended_correlation"
    / "ter26_kinematics_lut.npz"
)
OUT_PATH = (
    REPO_ROOT
    / "src"
    / "mirena_sim"
    / "Assets"
    / "Vehicles"
    / "TeR26"
    / "ter26_validated_spec.json"
)

HARDPOINT_NAMES_17: Tuple[str, ...] = (
    "CHAS_LowFor",
    "CHAS_LowAft",
    "CHAS_UppFor",
    "CHAS_UppAft",
    "UPRI_LowPnt",
    "UPRI_UppPnt",
    "CHAS_TiePnt",
    "UPRI_TiePnt",
    "NSMA_PPAttPnt",
    "CHAS_AttPnt",
    "CHAS_RocAxi",
    "CHAS_RocPiv",
    "ROCK_RodPnt",
    "ROCK_CoiPnt",
    "NSMA_UBarAttPnt",
    "UBAR_AttPnt",
    "CHAS_PivPnt",
)

HARDPOINT_TO_GODOT_KEY: Dict[str, str] = {
    "CHAS_LowFor": "lca_front",
    "CHAS_LowAft": "lca_rear",
    "UPRI_LowPnt": "lbj",
    "CHAS_UppFor": "uca_front",
    "CHAS_UppAft": "uca_rear",
    "UPRI_UppPnt": "ubj",
    "NSMA_PPAttPnt": "pushrod_wishbone",
    "ROCK_RodPnt": "pushrod_rocker",
    "UPRI_TiePnt": "tie_outer",
    "CHAS_TiePnt": "tie_inner",
    "CHAS_AttPnt": "damper_chassis",
    "ROCK_CoiPnt": "damper_rocker",
    "CHAS_RocAxi": "rocker_axis_p1",
    "CHAS_RocPiv": "rocker_pivot",
    "NSMA_UBarAttPnt": "arb_link_rocker",
    "UBAR_AttPnt": "arb_arm_tip",
    "CHAS_PivPnt": "arb_chassis_piv",
}


def ros_to_godot(v: List[float]) -> List[float]:
    """Convert ISO/ROS (+X fwd, +Y left, +Z up) to Godot (+X right, +Y up, -Z fwd)."""
    return [round(-float(v[1]), 6), round(float(v[2]), 6), round(-float(v[0]), 6)]


def build_hardpoints_frames(
    hp_mm: Dict[str, Tuple[float, float, float]],
    is_front: bool,
    is_right: bool,
    wheelbase_m: float,
    a_cog_m: float,
    h_cg_m: float,
) -> Dict[str, Dict[str, List[float]]]:
    """Return all 17 hardpoints in axle-ground ROS, front-axle ROS, CG ROS, and CG Godot."""
    y_sgn = -1.0 if is_right else 1.0
    x_axle_shift = 0.0 if is_front else -wheelbase_m

    iso_axle: Dict[str, List[float]] = {}
    iso_front_axle: Dict[str, List[float]] = {}
    iso_cg: Dict[str, List[float]] = {}
    godot_cg: Dict[str, List[float]] = {}

    for k in HARDPOINT_NAMES_17:
        x_mm, y_mm, z_mm = hp_mm[k]
        ax = [round(x_mm * 1e-3, 6), round(y_mm * y_sgn * 1e-3, 6), round(z_mm * 1e-3, 6)]
        fax = [round(ax[0] + x_axle_shift, 6), ax[1], ax[2]]
        cg = [round(fax[0] - a_cog_m, 6), fax[1], round(fax[2] - h_cg_m, 6)]
        iso_axle[k] = ax
        iso_front_axle[k] = fax
        iso_cg[k] = cg
        godot_cg[k] = ros_to_godot(cg)

    return {
        "iso_ros_axle_ground_m": iso_axle,
        "iso_ros_front_axle_ground_m": iso_front_axle,
        "iso_ros_cg_m": iso_cg,
        "godot_cg_m": godot_cg,
    }


def build_mirena_sim_kinematics_dict(
    wheelbase_m: float,
    half_track_f_m: float,
    half_track_r_m: float,
    r_wheel_m: float,
    spring_f_n_m: float,
    spring_r_n_m: float,
) -> Dict[str, Any]:
    """Build exact `double_wishbone_kinematics` dict for DoubleWishboneKinematics.gd."""

    def _side_dict(
        hp_mm: Dict[str, Tuple[float, float, float]],
        is_front: bool,
        half_track: float,
        spring_k: float,
    ) -> Dict[str, Any]:
        x_shift = 0.0 if is_front else -wheelbase_m
        out: Dict[str, Any] = {}
        for hp_name, gd_name in HARDPOINT_TO_GODOT_KEY.items():
            x_mm, y_mm, z_mm = hp_mm[hp_name]
            out[gd_name] = [
                round(x_mm * 1e-3 + x_shift, 6),
                round(y_mm * 1e-3, 6),
                round(z_mm * 1e-3, 6),
            ]
        out["wheel_center"] = [round(x_shift, 6), round(half_track, 6), round(r_wheel_m, 6)]
        out["contact_patch"] = [round(x_shift, 6), round(half_track, 6), 0.0]
        out["pushrod_on_lower"] = bool(is_front)
        out["static_camber_deg"] = -0.5 if is_front else -1.0
        out["static_toe_deg"] = 0.0
        out["spring_stiffness_n_m"] = float(spring_k)
        return out

    return {
        "use_3d_kinematics": True,
        "frame": "iso_ros_left_from_front_axle_ground_m",
        "wheelbase_m": wheelbase_m,
        "track_front_m": round(2.0 * half_track_f_m, 6),
        "track_rear_m": round(2.0 * half_track_r_m, 6),
        "wheel_radius_m": r_wheel_m,
        "front_left": _side_dict(FRONT_LEFT_HARDPOINTS_MM, True, half_track_f_m, spring_f_n_m),
        "rear_left": _side_dict(REAR_LEFT_HARDPOINTS_MM, False, half_track_r_m, spring_r_n_m),
    }


def export_bundle(out_path: Path = OUT_PATH) -> Dict[str, Any]:
    if not SYSID_PATH.exists():
        raise FileNotFoundError(f"Missing SysID params: {SYSID_PATH}")
    if not NPZ_PATH.exists():
        raise FileNotFoundError(f"Missing Kinematics LUT NPZ: {NPZ_PATH}")

    sysid_raw = json.loads(SYSID_PATH.read_text(encoding="utf-8"))
    ident = sysid_raw["identified_params"]

    kin = TeR26Kinematics3D()
    static_rep = kin.compute_static_and_stiffness_report()
    domain_ver = kin.verify_full_domain()
    ack_curve = kin.compute_ackermann_curve(z_w_m=0.0)
    assert domain_ver["closure_pass_lt_1e6_mm"], "TeR26 3D closure failed < 1e-6 mm!"

    L = 1.550
    fwd = 0.46
    left_frac = 0.50
    half_tf = 0.600
    half_tr = 0.590
    a_cog = L * (1.0 - fwd)
    b_cog = L * fwd
    mass_kg = 300.0
    sprung_mass_kg = 269.0
    unsprung_mass_total_kg = 31.0
    unsprung_mass_corner_kg = 7.75
    r_wheel_m = 0.2032
    h_cg_m = round(float(ident["h_cg"]), 6)  # 0.210 m
    h_sprung_cg_m = (mass_kg * h_cg_m - unsprung_mass_total_kg * r_wheel_m) / sprung_mass_kg
    i_zz_total = round(float(ident["i_zz"]), 5)  # 94.05 kg*m^2
    wheel_asm_inertia = 0.30
    steiner_f = 2.0 * unsprung_mass_corner_kg * (a_cog ** 2 + half_tf ** 2)
    steiner_r = 2.0 * unsprung_mass_corner_kg * (b_cog ** 2 + half_tr ** 2)
    i_zz_sprung = i_zz_total - steiner_f - steiner_r - 4.0 * wheel_asm_inertia

    mu_x_scale = float(ident["mu_x_scale"])
    mu_y_scale = float(ident["mu_y_scale"])
    kya_scale = float(ident["kya_scale"])
    camber_scale = float(ident["camber_scale"])
    cl_a = float(ident["cl_a"])
    cd_a = float(ident["cd_a"])
    c_damp_scale = float(ident["c_damp_scale"])
    k_steer_comp = float(ident["k_steer_comp"])
    sigma_y = float(ident["sigma_y"])
    i_steer = 1.2963

    lambda_mux_base = 0.78
    lambda_muy_base = 0.64
    lambda_mux_eff = lambda_mux_base * mu_x_scale
    lambda_muy_eff = lambda_muy_base * mu_y_scale

    # Godot SuspensionSystem._apply_axle_arb_3d uses k_arb_wheel = k_bar_nm_rad * mr_arb^2
    # so that K_phi_arb = k_arb_wheel * track^2 = k_bar_nm_rad * (mr_arb * track)^2
    rad2deg = math.pi / 180.0
    k_ubar_nm_rad = 5.0 / rad2deg
    mr_arb_f = float(static_rep["front"]["mr_arb_rad_per_m"])
    mr_arb_r = float(static_rep["rear"]["mr_arb_rad_per_m"])
    arb_stiffness_front_godot = k_ubar_nm_rad * (mr_arb_f ** 2)
    arb_stiffness_rear_godot = k_ubar_nm_rad * (mr_arb_r ** 2)

    # LLTD computed at SysID h_CG = 0.210 m as well as CAD h_CG = 0.270 m
    h_rc_f_m = float(static_rep["front"]["roll_center_height_mm"]) * 1e-3
    h_rc_r_m = float(static_rep["rear"]["roll_center_height_mm"]) * 1e-3
    el_f = float(static_rep["vehicle"]["elastic_roll_stiffness_front_frac"])
    h_roll_sysid = h_cg_m - (fwd * h_rc_f_m + (1.0 - fwd) * h_rc_r_m)
    dfz_f_u = (el_f * h_roll_sysid + fwd * h_rc_f_m) / (2.0 * half_tf)
    dfz_r_u = ((1.0 - el_f) * h_roll_sysid + (1.0 - fwd) * h_rc_r_m) / (2.0 * half_tr)
    lltd_sysid_pct = (dfz_f_u / (dfz_f_u + dfz_r_u)) * 100.0

    # Load NPZ 2D/1D kinematic LUTs
    npz = np.load(NPZ_PATH, allow_pickle=False)

    def _rlist(arr: np.ndarray, decimals: int = 6) -> Any:
        return np.round(np.asarray(arr, dtype=np.float64), decimals).tolist()

    fl_camber_deg_2d = np.asarray(npz["fl_camber_deg"], dtype=np.float64)
    fl_steer_deg_2d = np.asarray(npz["fl_steer_deg"], dtype=np.float64)
    # Mirror property: FR(z, d_rack) = mirror(FL(z, -d_rack))
    fr_camber_deg_2d = fl_camber_deg_2d[:, ::-1]
    fr_steer_deg_2d = -fl_steer_deg_2d[:, ::-1]

    kinematic_luts = {
        "axes": {
            "z_grid_m": _rlist(npz["z_grid_m"], 6),
            "rack_grid_m": _rlist(npz["rack_grid_m"], 6),
            "mean_steer_grid_rad": _rlist(npz["mean_steer_grid_rad"], 6),
            "rack_from_mean_steer_m": _rlist(npz["rack_from_mean_steer_m"], 6),
        },
        "front_2d_shape": [int(fl_camber_deg_2d.shape[0]), int(fl_camber_deg_2d.shape[1])],
        "front_2d": {
            "fl_camber_deg": _rlist(fl_camber_deg_2d, 5),
            "fr_camber_deg": _rlist(fr_camber_deg_2d, 5),
            "fl_steer_deg": _rlist(fl_steer_deg_2d, 5),
            "fr_steer_deg": _rlist(fr_steer_deg_2d, 5),
            "fl_toe_deg": _rlist(npz["fl_toe_deg"], 5),
            "fl_mr_spring": _rlist(npz["fl_mr_spring"], 6),
            "fl_mr_arb_rad_m": _rlist(npz["fl_mr_arb_rad_m"], 6),
            "fl_damper_disp_mm": _rlist(npz["fl_damper_disp_mm"], 5),
            "fl_rc_z_mm": _rlist(npz["fl_rc_z_mm"], 4),
            "fl_svsa_slope": _rlist(npz["fl_svsa_slope"], 6),
            "fl_mech_trail_mm": _rlist(npz["fl_mech_trail_mm"], 4),
            "fl_scrub_radius_mm": _rlist(npz["fl_scrub_radius_mm"], 4),
        },
        "rear_1d_shape": [int(npz["rl_camber_deg"].shape[0])],
        "rear_1d": {
            "rl_camber_deg": _rlist(npz["rl_camber_deg"], 5),
            "rl_toe_deg": _rlist(npz["rl_toe_deg"], 5),
            "rl_mr_spring": _rlist(npz["rl_mr_spring"], 6),
            "rl_mr_arb_rad_m": _rlist(npz["rl_mr_arb_rad_m"], 6),
            "rl_damper_disp_mm": _rlist(npz["rl_damper_disp_mm"], 5),
            "rl_rc_z_mm": _rlist(npz["rl_rc_z_mm"], 4),
            "rl_svsa_slope": _rlist(npz["rl_svsa_slope"], 6),
        },
    }

    hardpoints = {
        "front_left": build_hardpoints_frames(FRONT_LEFT_HARDPOINTS_MM, True, False, L, a_cog, h_cg_m),
        "front_right": build_hardpoints_frames(FRONT_LEFT_HARDPOINTS_MM, True, True, L, a_cog, h_cg_m),
        "rear_left": build_hardpoints_frames(REAR_LEFT_HARDPOINTS_MM, False, False, L, a_cog, h_cg_m),
        "rear_right": build_hardpoints_frames(REAR_LEFT_HARDPOINTS_MM, False, True, L, a_cog, h_cg_m),
    }

    mirena_dw = build_mirena_sim_kinematics_dict(
        wheelbase_m=L,
        half_track_f_m=half_tf,
        half_track_r_m=half_tr,
        r_wheel_m=r_wheel_m,
        spring_f_n_m=44000.0,
        spring_r_n_m=53000.0,
    )

    bundle: Dict[str, Any] = {
        "metadata": {
            "vehicle_id": "ter26_rwd",
            "display_name": "TeR26 Validado (RWD)",
            "source_cad": "TER21 FINAL.xlsx",
            "source_sysid": "vd_correlation_lab/results/ter26_extended_correlation/ter26_sysid_params.json",
            "source_kinematics_lut": "vd_correlation_lab/results/ter26_extended_correlation/ter26_kinematics_lut.npz",
            "coordinate_conventions": {
                "iso_ros": "+X forward, +Y left, +Z up",
                "godot": "+X right (-Y_ros), +Y up (+Z_ros), -Z forward (-X_ros)",
                "vehicle_camber": "negative = top of tire leans inward (both L and R)",
                "vehicle_toe": "positive = toe-in",
            },
            "domain_closure_verification": domain_ver,
        },
        "sysid_params": {
            "wheelbase_m": L,
            "front_weight_distribution": fwd,
            "rear_weight_distribution": round(1.0 - fwd, 6),
            "left_weight_distribution": left_frac,
            "a_cog_m": round(a_cog, 6),
            "b_cog_m": round(b_cog, 6),
            "half_track_f_m": half_tf,
            "half_track_r_m": half_tr,
            "track_width_f_m": round(2.0 * half_tf, 6),
            "track_width_r_m": round(2.0 * half_tr, 6),
            "mass_kg": mass_kg,
            "sprung_mass_kg": sprung_mass_kg,
            "unsprung_mass_total_kg": unsprung_mass_total_kg,
            "unsprung_mass_corner_kg": unsprung_mass_corner_kg,
            "wheel_radius_m": r_wheel_m,
            "wheel_assembly_inertia_kg_m2": wheel_asm_inertia,
            "h_cg_m": h_cg_m,
            "h_sprung_cg_m": round(h_sprung_cg_m, 6),
            "i_zz_total_kg_m2": round(i_zz_total, 4),
            "i_zz_sprung_kg_m2": round(i_zz_sprung, 6),
            "brake_bias_front": 0.70,
            "drive_bias_front": 0.0,
            "mu_x_scale": round(mu_x_scale, 6),
            "mu_y_scale": round(mu_y_scale, 6),
            "lambda_mux_base": lambda_mux_base,
            "lambda_muy_base": lambda_muy_base,
            "lambda_mux_effective": round(lambda_mux_eff, 6),
            "lambda_muy_effective": round(lambda_muy_eff, 6),
            "kya_scale": round(kya_scale, 6),
            "camber_scale": round(camber_scale, 6),
            "cl_a_m2": round(cl_a, 6),
            "cd_a_m2": round(cd_a, 6),
            "c_damp_scale": round(c_damp_scale, 6),
            "k_steer_comp_nm_rad": round(k_steer_comp, 4),
            "sigma_y_m": round(sigma_y, 6),
            "i_steer": i_steer,
        },
        "suspension_summary": {
            "couplings": {
                "front_pushrod_on_lower": True,
                "rear_pushrod_on_lower": False,
                "u_bar": "Rocker",
            },
            "spring_rate_front_n_mm": 44.0,
            "spring_rate_rear_n_mm": 53.0,
            "wheel_rate_front_n_mm": round(float(static_rep["front"]["wheel_rate_n_mm"]), 4),
            "wheel_rate_rear_n_mm": round(float(static_rep["rear"]["wheel_rate_n_mm"]), 4),
            "mr_spring_front": round(float(static_rep["front"]["mr_spring"]), 6),
            "mr_spring_rear": round(float(static_rep["rear"]["mr_spring"]), 6),
            "mr_arb_front_rad_m": round(mr_arb_f, 6),
            "mr_arb_rear_rad_m": round(mr_arb_r, 6),
            "ubar_stiffness_front_nm_deg": 5.0,
            "ubar_stiffness_rear_nm_deg": 5.0,
            "godot_arb_stiffness_front_n_m": round(arb_stiffness_front_godot, 4),
            "godot_arb_stiffness_rear_n_m": round(arb_stiffness_rear_godot, 4),
            "k_phi_spring_front_nm_deg": round(float(static_rep["front"]["k_phi_spring_nm_deg"]), 4),
            "k_phi_spring_rear_nm_deg": round(float(static_rep["rear"]["k_phi_spring_nm_deg"]), 4),
            "k_phi_arb_front_nm_deg": round(float(static_rep["front"]["k_phi_arb_nm_deg"]), 4),
            "k_phi_arb_rear_nm_deg": round(float(static_rep["rear"]["k_phi_arb_nm_deg"]), 4),
            "k_phi_total_front_nm_deg": round(float(static_rep["front"]["k_phi_total_nm_deg"]), 4),
            "k_phi_total_rear_nm_deg": round(float(static_rep["rear"]["k_phi_total_nm_deg"]), 4),
            "h_rc_front_mm": round(float(static_rep["front"]["roll_center_height_mm"]), 4),
            "h_rc_rear_mm": round(float(static_rep["rear"]["roll_center_height_mm"]), 4),
            "anti_dive_pct": round(float(static_rep["front"]["anti_dive_pct_at_70pct_bb"]), 4),
            "anti_squat_pct": round(float(static_rep["rear"]["anti_squat_pct_rwd"]), 4),
            "lltd_mechanical_cad_hcg270_pct": round(float(static_rep["vehicle"]["lltd_mechanical_pct"]), 4),
            "lltd_mechanical_sysid_hcg210_pct": round(lltd_sysid_pct, 4),
            "caster_front_deg": round(float(static_rep["front"]["caster_deg"]), 4),
            "kpi_front_deg": round(float(static_rep["front"]["kpi_deg"]), 4),
            "scrub_radius_front_mm": round(float(static_rep["front"]["scrub_radius_mm"]), 4),
            "mechanical_trail_front_mm": round(float(static_rep["front"]["mechanical_trail_mm"]), 4),
            "static_camber_front_deg": -0.5,
            "static_camber_rear_deg": -1.0,
            "static_toe_front_deg": 0.0,
            "static_toe_rear_deg": 0.0,
            "ackermann_curve": ack_curve,
        },
        "hardpoints_17": hardpoints,
        "mirena_sim_double_wishbone_kinematics": mirena_dw,
        "kinematic_luts": kinematic_luts,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_text = json.dumps(bundle, indent=2, ensure_ascii=True) + "\n"
    assert out_text.isascii(), "Exported JSON must be 100% pure ASCII!"
    out_path.write_text(out_text, encoding="utf-8")
    return bundle


if __name__ == "__main__":
    print("Exporting TeR26 validated Godot bundle...", flush=True)
    b = export_bundle()
    sp = b["sysid_params"]
    ss = b["suspension_summary"]
    print(f"Wrote: {OUT_PATH}", flush=True)
    print(
        f"SysID: L={sp['wheelbase_m']} m, FWD={sp['front_weight_distribution']}, "
        f"h_CG={sp['h_cg_m']} m, I_zz={sp['i_zz_total_kg_m2']} kg*m^2, "
        f"lmux={sp['lambda_mux_effective']}, lmuy={sp['lambda_muy_effective']}, "
        f"kya={sp['kya_scale']}, camber={sp['camber_scale']}, "
        f"CL*A={sp['cl_a_m2']}, CD*A={sp['cd_a_m2']}, "
        f"k_comp={sp['k_steer_comp_nm_rad']}, sigma_y={sp['sigma_y_m']}",
        flush=True,
    )
    print(
        f"Susp: kw={ss['wheel_rate_front_n_mm']}/{ss['wheel_rate_rear_n_mm']} N/mm, "
        f"h_RC={ss['h_rc_front_mm']}/{ss['h_rc_rear_mm']} mm, "
        f"AD={ss['anti_dive_pct']}%, AS={ss['anti_squat_pct']}%, "
        f"LLTD(CAD)={ss['lltd_mechanical_cad_hcg270_pct']}%, "
        f"LLTD(SysID)={ss['lltd_mechanical_sysid_hcg210_pct']}%",
        flush=True,
    )
