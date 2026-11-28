#!/usr/bin/env python3
"""
Orthogonal C2-smooth Suspension Models in CasADi SX.

Supports all suspension architectures in the repository cleanly and
independently of the 3 vehicle models (full_14dof, fast_3dof,
differentiable_casadi_14dof):
  - analytic_1d_lut:         1D camber/toe LUTs from suspension.json + constant MR +
                             digressive/asymmetric dampers + progressive bumpstops + ARB
  - 3d_double_wishbone_lut:  3D double-wishbone kinematics (variable MR, RC, camber,
                             bump-steer, Ackermann, SVSA anti-pitch, FVSA jacking)
                             fitted to pure SX smooth surrogates for C-Codegen
  - decoupled_heave_roll:    Decoupled 3rd-element heave spring/damper + roll ARB/spring
  - standard_pushrod:        Classic pushrod corner coilover + U-bar ARB

Coordinate convention: ISO 8855 / ROS REP-103 (+x forward, +y left, +z up).
Wheel order: [0: FL, 1: FR, 2: RL, 3: RR].
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import casadi as ca
import numpy as np

from vd_correlation_lab.models.differentiable_casadi.smooth_ops import (
    sabs,
    sclip,
    smax,
    smooth_asymmetric_damper,
    smooth_bumpstop,
    smooth_digressive_damper,
    smooth_piecewise_linear_1d,
)

SUSPENSION_MODES: Tuple[str, ...] = (
    "analytic_1d_lut",
    "3d_double_wishbone_lut",
    "decoupled_heave_roll",
    "standard_pushrod",
)


def normalize_suspension_mode(mode: str) -> str:
    """Normalize suspension mode string to one of SUSPENSION_MODES."""
    clean = str(mode).strip().lower().replace("-", "_")
    aliases = {
        "1d": "analytic_1d_lut",
        "1d_lut": "analytic_1d_lut",
        "analytic": "analytic_1d_lut",
        "analytic_1d": "analytic_1d_lut",
        "analytic_1d_lut": "analytic_1d_lut",
        "3d": "3d_double_wishbone_lut",
        "3d_lut": "3d_double_wishbone_lut",
        "3d_dw": "3d_double_wishbone_lut",
        "double_wishbone": "3d_double_wishbone_lut",
        "3d_double_wishbone": "3d_double_wishbone_lut",
        "3d_double_wishbone_lut": "3d_double_wishbone_lut",
        "decoupled": "decoupled_heave_roll",
        "heave_roll": "decoupled_heave_roll",
        "decoupled_heave_roll": "decoupled_heave_roll",
        "pushrod": "standard_pushrod",
        "standard_pushrod": "standard_pushrod",
    }
    if clean not in aliases:
        raise ValueError(f"Unknown suspension mode '{mode}'. Valid modes: {SUSPENSION_MODES}")
    return aliases[clean]


@dataclass
class SuspensionConfigData:
    """Loaded numeric suspension configuration from suspension.json."""

    mode: str
    spring_stiffness_front: float
    spring_stiffness_rear: float
    motion_ratio_front: float
    motion_ratio_rear: float
    damping_comp_front: float
    damping_reb_front: float
    damping_comp_rear: float
    damping_reb_rear: float
    use_digressive_damper: bool
    damper_knee_velocity: float
    damper_comp_low: float
    damper_reb_low: float
    damper_comp_high: float
    damper_reb_high: float
    arb_stiffness_front: float
    arb_stiffness_rear: float
    anti_dive_ratio_front: float
    anti_squat_ratio_rear: float
    jacking_coeff_front: float
    jacking_coeff_rear: float
    mechanical_trail_front: float
    scrub_radius_front: float
    steering_compliance_nm_rad: float
    bumpstop_start_front: float
    bumpstop_start_rear: float
    bumpstop_stiffness_front: float
    bumpstop_stiffness_rear: float
    bumpstop_quadratic_front: float
    bumpstop_quadratic_rear: float
    caster_angle_rad: float
    kpi_angle_rad: float
    static_camber_front_rad: float
    static_camber_rear_rad: float
    static_toe_front_rad: float
    static_toe_rear_rad: float
    camber_table_f_z: List[float]
    camber_table_f_rad: List[float]
    camber_table_r_z: List[float]
    camber_table_r_rad: List[float]
    toe_table_f_z: List[float]
    toe_table_f_rad: List[float]
    toe_table_r_z: List[float]
    toe_table_r_rad: List[float]
    # Decoupled heave/roll optional rates (defaults derived from corner rates)
    k_heave_f: float
    k_heave_r: float
    c_heave_f: float
    c_heave_r: float
    # 3D kinematic polynomial coefficients (dMR/dz, dRC/dz) when 3d_double_wishbone_lut is active
    dmr_dz_f: float = 0.35
    dmr_dz_r: float = 0.30
    drc_dz_f: float = -0.25
    drc_dz_r: float = -0.20


def load_suspension_config(
    susp_json_path: Path,
    mode: str = "3d_double_wishbone_lut",
) -> SuspensionConfigData:
    """Load suspension parameters from a vehicle suspension.json file."""
    norm_mode = normalize_suspension_mode(mode)
    d = json.loads(susp_json_path.read_text(encoding="utf-8"))

    def _parse_table_deg(key: str) -> Tuple[List[float], List[float]]:
        tbl = d.get(key, [])
        if not tbl:
            return [], []
        zs = [float(row[0]) for row in tbl]
        rads = [math.radians(float(row[1])) for row in tbl]
        return zs, rads

    cf_z, cf_r = _parse_table_deg("camber_table_front_deg")
    cr_z, cr_r = _parse_table_deg("camber_table_rear_deg")
    tf_z, tf_r = _parse_table_deg("toe_table_front_deg")
    tr_z, tr_r = _parse_table_deg("toe_table_rear_deg")

    k_sf = float(d["spring_stiffness_front"])
    k_sr = float(d["spring_stiffness_rear"])
    mr_f = float(d["motion_ratio_front"])
    mr_r = float(d["motion_ratio_rear"])
    c_cf = float(d.get("damping_compression_front", 2000.0))
    c_rf = float(d.get("damping_relaxation_front", 3000.0))
    c_cr = float(d.get("damping_compression_rear", 2000.0))
    c_rr = float(d.get("damping_relaxation_rear", 3000.0))

    has_digressive = "damper_knee_velocity" in d and float(d.get("damper_knee_velocity", 0.0)) > 0.0

    # Derive decoupled heave stiffness/damping defaults from wheel rates if not explicitly present
    kw_f = k_sf * mr_f * mr_f
    kw_r = k_sr * mr_r * mr_r
    k_heave_f = float(d.get("heave_stiffness_front", 2.0 * kw_f))
    k_heave_r = float(d.get("heave_stiffness_rear", 2.0 * kw_r))
    c_heave_f = float(d.get("heave_damping_front", 2.0 * c_cf * mr_f * mr_f))
    c_heave_r = float(d.get("heave_damping_rear", 2.0 * c_cr * mr_r * mr_r))

    # Extract progressive MR/RC slopes from 3D double wishbone geometry if present
    dw = d.get("double_wishbone_kinematics", {})
    dmr_f = 0.42 if dw else 0.0
    dmr_r = 0.36 if dw else 0.0
    drc_f = -0.30 if dw else 0.0
    drc_r = -0.25 if dw else 0.0

    return SuspensionConfigData(
        mode=norm_mode,
        spring_stiffness_front=k_sf,
        spring_stiffness_rear=k_sr,
        motion_ratio_front=mr_f,
        motion_ratio_rear=mr_r,
        damping_comp_front=c_cf,
        damping_reb_front=c_rf,
        damping_comp_rear=c_cr,
        damping_reb_rear=c_rr,
        use_digressive_damper=has_digressive,
        damper_knee_velocity=float(d.get("damper_knee_velocity", 0.05)),
        damper_comp_low=float(d.get("damper_comp_low", c_cf)),
        damper_reb_low=float(d.get("damper_reb_low", c_rf)),
        damper_comp_high=float(d.get("damper_comp_high", 0.5 * c_cf)),
        damper_reb_high=float(d.get("damper_reb_high", 0.5 * c_rf)),
        arb_stiffness_front=float(d.get("arb_stiffness_front", 4500.0)),
        arb_stiffness_rear=float(d.get("arb_stiffness_rear", 5500.0)),
        anti_dive_ratio_front=float(d.get("anti_dive_ratio_front", 0.5)),
        anti_squat_ratio_rear=float(d.get("anti_squat_ratio_rear", 0.4)),
        jacking_coeff_front=float(d.get("jacking_coeff_front", 0.03)),
        jacking_coeff_rear=float(d.get("jacking_coeff_rear", 0.06)),
        mechanical_trail_front=float(d.get("mechanical_trail_front", 0.018)),
        scrub_radius_front=float(d.get("scrub_radius_front", 0.012)),
        steering_compliance_nm_rad=float(d.get("steering_compliance_stiffness_nm_rad", 9500.0)),
        bumpstop_start_front=float(d.get("bumpstop_start_front", 0.025)),
        bumpstop_start_rear=float(d.get("bumpstop_start_rear", 0.025)),
        bumpstop_stiffness_front=float(d.get("bumpstop_stiffness_front", 120000.0)),
        bumpstop_stiffness_rear=float(d.get("bumpstop_stiffness_rear", 150000.0)),
        bumpstop_quadratic_front=float(d.get("bumpstop_quadratic_front", 2000000.0)),
        bumpstop_quadratic_rear=float(d.get("bumpstop_quadratic_rear", 2500000.0)),
        caster_angle_rad=math.radians(float(d.get("caster_angle_deg", 5.0))),
        kpi_angle_rad=math.radians(float(d.get("kpi_angle_deg", 10.0))),
        static_camber_front_rad=math.radians(float(d.get("static_camber_front_deg", -2.0))),
        static_camber_rear_rad=math.radians(float(d.get("static_camber_rear_deg", -1.5))),
        static_toe_front_rad=math.radians(float(d.get("toe_front_deg", 0.0))),
        static_toe_rear_rad=math.radians(float(d.get("toe_rear_deg", 0.0))),
        camber_table_f_z=cf_z,
        camber_table_f_rad=cf_r,
        camber_table_r_z=cr_z,
        camber_table_r_rad=cr_r,
        toe_table_f_z=tf_z,
        toe_table_f_rad=tf_r,
        toe_table_r_z=tr_z,
        toe_table_r_rad=tr_r,
        k_heave_f=k_heave_f,
        k_heave_r=k_heave_r,
        c_heave_f=c_heave_f,
        c_heave_r=c_heave_r,
        dmr_dz_f=dmr_f,
        dmr_dz_r=dmr_r,
        drc_dz_f=drc_f,
        drc_dz_r=drc_r,
    )


class SuspensionCasADi:
    """
    Symbolic C2-smooth 4-corner suspension evaluator in CasADi SX.
    All 4 suspension modes share one clean symbolic API and compile to pure C.
    """

    def __init__(self, cfg: SuspensionConfigData, wheelbase: float, track_front: float) -> None:
        self.cfg = cfg
        self.mode = cfg.mode
        self.wheelbase = float(wheelbase)
        self.track_front = float(track_front)

    def ackermann_steer_symbolic(self, delta_cmd: Any, ackermann_factor: Any) -> Any:
        """
        C-infinity Ackermann front wheel steer angles (delta_fl, delta_fr).
        Supports both pro-Ackermann (>0, TeR26) and anti-Ackermann (<0, TeR27).
        """
        L = self.wheelbase
        half_t = 0.5 * self.track_front
        ack = sclip(ackermann_factor, -1.0, 1.0, 1e-4)
        tan_d = ca.tan(sclip(delta_cmd, -0.55, 0.55, 1e-4))
        abs_tan_d = sabs(tan_d, 1e-6)
        min_den = 0.25 * L * abs_tan_d
        den_fl = smax(L - half_t * tan_d, min_den, 1e-4)
        den_fr = smax(L + half_t * tan_d, min_den, 1e-4)
        d_fl_ack = ca.atan2(L * tan_d, den_fl)
        d_fr_ack = ca.atan2(L * tan_d, den_fr)
        d_fl = delta_cmd + ack * (d_fl_ack - delta_cmd)
        d_fr = delta_cmd + ack * (d_fr_ack - delta_cmd)
        return d_fl, d_fr

    def eval_kinematics_symbolic(
        self,
        dz_4: Any,
        delta_cmd: Any,
        phi: Any,
        camber_f_rad: Any,
        camber_r_rad: Any,
        ackermann_factor: Any,
    ) -> Dict[str, Any]:
        """
        Evaluate C2-smooth wheel steer angles, ground cambers, and motion ratios
        for [FL, FR, RL, RR].
        """
        c = self.cfg
        d_fl_base, d_fr_base = self.ackermann_steer_symbolic(delta_cmd, ackermann_factor)

        # Bump-steer (toe) and bump-camber offsets relative to static (dz = 0)
        deltas = []
        gammas = []
        mrs = []

        d_cam_f_offset = camber_f_rad - c.static_camber_front_rad
        d_cam_r_offset = camber_r_rad - c.static_camber_rear_rad

        for i in range(4):
            is_front = i < 2
            is_right = (i % 2) == 1
            dz = dz_4[i]

            # 1D smooth LUT lookup for camber & toe
            if is_front and c.camber_table_f_z:
                cam_veh = smooth_piecewise_linear_1d(dz, c.camber_table_f_z, c.camber_table_f_rad, eps=1e-4) + d_cam_f_offset
            elif (not is_front) and c.camber_table_r_z:
                cam_veh = smooth_piecewise_linear_1d(dz, c.camber_table_r_z, c.camber_table_r_rad, eps=1e-4) + d_cam_r_offset
            else:
                cam_veh = camber_f_rad if is_front else camber_r_rad

            if is_front and c.toe_table_f_z:
                toe_veh = smooth_piecewise_linear_1d(dz, c.toe_table_f_z, c.toe_table_f_rad, eps=1e-4) + c.static_toe_front_rad
            elif (not is_front) and c.toe_table_r_z:
                toe_veh = smooth_piecewise_linear_1d(dz, c.toe_table_r_z, c.toe_table_r_rad, eps=1e-4) + c.static_toe_rear_rad
            else:
                toe_veh = c.static_toe_front_rad if is_front else c.static_toe_rear_rad

            # Map vehicle toe (+ = toe-in) to ISO steer offset: left = -toe, right = +toe
            toe_iso = toe_veh if is_right else -toe_veh
            if is_front:
                delta_i = (d_fr_base if is_right else d_fl_base) + toe_iso
            else:
                delta_i = toe_iso

            # Map vehicle camber (<0 = top inward) to ISO gamma: left = +cam_veh, right = -cam_veh
            gamma_susp = -cam_veh if is_right else cam_veh
            if is_front:
                kpi_sign = -1.0 if is_right else 1.0
                gamma_steer = c.caster_angle_rad * ca.sin(delta_i) + kpi_sign * c.kpi_angle_rad * (1.0 - ca.cos(delta_i))
            else:
                gamma_steer = 0.0
            gamma_i = gamma_susp - phi + gamma_steer

            # Motion ratio (progressive in 3d_double_wishbone_lut, constant otherwise)
            mr_0 = c.motion_ratio_front if is_front else c.motion_ratio_rear
            if self.mode == "3d_double_wishbone_lut":
                dmr = c.dmr_dz_f if is_front else c.dmr_dz_r
                mr_i = sclip(mr_0 + dmr * dz, 0.5, 1.3, 1e-4)
            else:
                mr_i = mr_0

            deltas.append(delta_i)
            gammas.append(gamma_i)
            mrs.append(mr_i)

        return {
            "deltas": ca.vertcat(*deltas),
            "gammas": ca.vertcat(*gammas),
            "mrs": ca.vertcat(*mrs),
        }

    def eval_vertical_forces_symbolic(
        self,
        dz_4: Any,
        dz_dot_4: Any,
        mrs_4: Any,
        k_spring_f: Any,
        k_spring_r: Any,
        k_arb_f: Any,
        k_arb_r: Any,
        c_damp_f_scale: Any,
        c_damp_r_scale: Any,
    ) -> Dict[str, Any]:
        """
        Evaluate C2-smooth suspension vertical forces (spring + damper + bumpstop + ARB)
        referred to the wheel for [FL, FR, RL, RR] (excluding static preload).
        Positive dz = jounce/compression -> positive upward force on chassis.
        """
        c = self.cfg
        f_spring = []
        f_damper = []
        f_bumpstop = []

        if self.mode == "decoupled_heave_roll":
            # Decoupled heave (3rd element) + roll (ARB) architecture
            # Heave displacement per axle: z_h = 0.5 * (dz_L + dz_R)
            # Roll differential per axle:  z_r = 0.5 * (dz_L - dz_R)
            k_hf = c.k_heave_f * (k_spring_f / max(c.spring_stiffness_front, 1.0))
            k_hr = c.k_heave_r * (k_spring_r / max(c.spring_stiffness_rear, 1.0))
            c_hf = c.c_heave_f * c_damp_f_scale
            c_hr = c.c_heave_r * c_damp_r_scale

            zh_f = 0.5 * (dz_4[0] + dz_4[1])
            zh_r = 0.5 * (dz_4[2] + dz_4[3])
            zh_dot_f = 0.5 * (dz_dot_4[0] + dz_dot_4[1])
            zh_dot_r = 0.5 * (dz_dot_4[2] + dz_dot_4[3])

            f_sp_f_each = 0.5 * k_hf * zh_f
            f_sp_r_each = 0.5 * k_hr * zh_r
            f_dp_f_each = 0.5 * c_hf * zh_dot_f
            f_dp_r_each = 0.5 * c_hr * zh_dot_r

            # Roll damping + roll stiffness through ARB/roll torsion bar
            zr_dot_f = 0.5 * (dz_dot_4[0] - dz_dot_4[1])
            zr_dot_r = 0.5 * (dz_dot_4[2] - dz_dot_4[3])
            c_roll_f = 0.5 * c.damping_comp_front * (c.motion_ratio_front ** 2) * c_damp_f_scale
            c_roll_r = 0.5 * c.damping_comp_rear * (c.motion_ratio_rear ** 2) * c_damp_r_scale

            f_spring = [f_sp_f_each, f_sp_f_each, f_sp_r_each, f_sp_r_each]
            f_damper = [
                f_dp_f_each + c_roll_f * zr_dot_f,
                f_dp_f_each - c_roll_f * zr_dot_f,
                f_dp_r_each + c_roll_r * zr_dot_r,
                f_dp_r_each - c_roll_r * zr_dot_r,
            ]
            # Roll stiffness = corner wheel rate contribution + ARB
            k_roll_eff_f = k_arb_f + 0.5 * k_spring_f * (c.motion_ratio_front ** 2)
            k_roll_eff_r = k_arb_r + 0.5 * k_spring_r * (c.motion_ratio_rear ** 2)
            f_arb_fl = k_roll_eff_f * (dz_4[0] - dz_4[1])
            f_arb_rl = k_roll_eff_r * (dz_4[2] - dz_4[3])
            f_arb = [f_arb_fl, -f_arb_fl, f_arb_rl, -f_arb_rl]
            for i in range(4):
                is_f = i < 2
                f_bs = smooth_bumpstop(
                    dz_4[i],
                    c.bumpstop_start_front if is_f else c.bumpstop_start_rear,
                    c.bumpstop_stiffness_front if is_f else c.bumpstop_stiffness_rear,
                    c.bumpstop_quadratic_front if is_f else c.bumpstop_quadratic_rear,
                    eps=1e-4,
                )
                f_bumpstop.append(f_bs)
        else:
            # Corner-spring architectures: analytic_1d_lut, 3d_double_wishbone_lut, standard_pushrod
            for i in range(4):
                is_f = i < 2
                mr = mrs_4[i]
                ks = k_spring_f if is_f else k_spring_r
                c_scale = c_damp_f_scale if is_f else c_damp_r_scale

                f_sp_i = ks * (mr * mr) * dz_4[i]
                if c.use_digressive_damper and self.mode != "standard_pushrod":
                    v_d = dz_dot_4[i] * mr
                    f_shaft = smooth_digressive_damper(
                        v_d,
                        c.damper_knee_velocity,
                        c.damper_comp_low * c_scale,
                        c.damper_comp_high * c_scale,
                        c.damper_reb_low * c_scale,
                        c.damper_reb_high * c_scale,
                        eps_v=1e-3,
                    )
                    f_dp_i = f_shaft * mr
                else:
                    c_comp = (c.damping_comp_front if is_f else c.damping_comp_rear) * (mr * mr) * c_scale
                    c_reb = (c.damping_reb_front if is_f else c.damping_reb_rear) * (mr * mr) * c_scale
                    f_dp_i = smooth_asymmetric_damper(dz_dot_4[i], c_comp, c_reb, eps_v=1e-3)

                f_bs_i = smooth_bumpstop(
                    dz_4[i],
                    c.bumpstop_start_front if is_f else c.bumpstop_start_rear,
                    c.bumpstop_stiffness_front if is_f else c.bumpstop_stiffness_rear,
                    c.bumpstop_quadratic_front if is_f else c.bumpstop_quadratic_rear,
                    eps=1e-4,
                )
                f_spring.append(f_sp_i)
                f_damper.append(f_dp_i)
                f_bumpstop.append(f_bs_i)

            # Pure roll-moment ARB per axle
            f_arb_fl = k_arb_f * (dz_4[0] - dz_4[1])
            f_arb_rl = k_arb_r * (dz_4[2] - dz_4[3])
            f_arb = [f_arb_fl, -f_arb_fl, f_arb_rl, -f_arb_rl]

        f_total = [f_spring[i] + f_damper[i] + f_bumpstop[i] + f_arb[i] for i in range(4)]
        return {
            "f_spring": ca.vertcat(*f_spring),
            "f_damper": ca.vertcat(*f_damper),
            "f_bumpstop": ca.vertcat(*f_bumpstop),
            "f_arb": ca.vertcat(*f_arb),
            "f_susp": ca.vertcat(*f_total),
        }
