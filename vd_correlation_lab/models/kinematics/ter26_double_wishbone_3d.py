"""
Exact 3D Spatial Double-Wishbone Suspension Kinematics Solver for TeR26 (TER21 FINAL.xlsx).

Coordinate convention (ISO 8855 / OptimumKinematics [[1,0,0],[0,1,0],[0,0,1]]):
  - Axle-local origin (0, 0, 0) at axle centerline on the ground plane (Z = 0).
  - +X forward, +Y left, +Z up.
  - Right-side hardpoints are exact mirror images in Y (Y_R = -Y_L).
  - Wheel travel z_w: vertical contact-patch displacement in m (bump > 0, droop < 0).
  - Steering rack displacement d_rack: lateral rack shift in m (+Y = left -> steers both
    front wheels left, delta_FL > 0, delta_FR > 0).
  - Camber (vehicle convention): negative = top of tire leans inward toward vehicle centerline
    for both left and right wheels.
  - Toe (vehicle convention): positive = toe-in (front of tire points inward toward centerline).

Couplings from TER21 FINAL.xlsx:
  - Front: Push-Pull attached to LowerAArm; U-Bar attached to Rocker.
  - Rear:  Push-Pull attached to UpperAArm; U-Bar attached to Rocker.

Loop closure accuracy:
  - Main 5-link spatial closure (LCA, UCA, kingpin, tie-rod, contact patch Z): damped
    Newton-Raphson to < 1e-12 m (< 1e-9 mm).
  - Pushrod-to-rocker and rocker-to-U-Bar closures: exact analytical circle-sphere
    intersection refined with Newton-Raphson to < 1e-13 m (< 1e-10 mm).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# Exact Hardpoint & Vehicle Parameters from TER21 FINAL.xlsx (mm, deg, N/mm, Nm/deg)
# =============================================================================

TER26_VEHICLE_PARAMS: Dict[str, float] = {
    "wheelbase_mm": 1550.0,
    "brake_bias_front": 0.70,
    "drive_bias_front": 0.00,
    "front_weight_frac": 0.46,
    "left_weight_frac": 0.50,
    "h_cg_mm": 270.0,
    "rim_dia_mm": 254.0,
    "tire_dia_mm": 406.4,
    "tire_radius_mm": 203.2,
    "tire_width_mm": 190.5,
    "half_track_front_mm": 600.0,
    "half_track_rear_mm": 590.0,
    "static_camber_front_deg": -0.5,
    "static_camber_rear_deg": -1.0,
    "static_toe_front_deg": 0.0,
    "static_toe_rear_deg": 0.0,
    "spring_rate_front_n_mm": 44.0,
    "spring_rate_rear_n_mm": 53.0,
    "ubar_stiffness_front_nm_deg": 5.0,
    "ubar_stiffness_rear_nm_deg": 5.0,
    "steering_ratio_mm_per_rev": 101.6,
}

FRONT_LEFT_HARDPOINTS_MM: Dict[str, Tuple[float, float, float]] = {
    "CHAS_LowFor": (32.774, 245.0, 126.645),
    "CHAS_LowAft": (-144.948, 245.0, 128.98),
    "CHAS_UppFor": (109.916, 245.0, 247.31),
    "CHAS_UppAft": (-119.867, 245.0, 228.048),
    "UPRI_LowPnt": (4.797, 565.853, 134.6),
    "UPRI_UppPnt": (-4.993, 551.138, 274.6),
    "CHAS_TiePnt": (95.0, 144.78, 190.0),
    "UPRI_TiePnt": (67.176, 571.546, 223.748),
    "NSMA_PPAttPnt": (2.305, 510.78, 175.346),
    "CHAS_AttPnt": (-24.664, 25.0, 616.288),
    "CHAS_RocAxi": (-62.865, 195.0, 585.645),
    "CHAS_RocPiv": (-22.939, 195.0, 588.087),
    "ROCK_RodPnt": (-24.666, 240.998, 616.322),
    "ROCK_CoiPnt": (-26.618, 184.318, 648.229),
    "NSMA_UBarAttPnt": (-23.22, 221.131, 592.684),
    "UBAR_AttPnt": (-19.558, 221.131, 532.796),
    "CHAS_PivPnt": (50.312, 221.131, 537.069),
}

REAR_LEFT_HARDPOINTS_MM: Dict[str, Tuple[float, float, float]] = {
    "CHAS_LowFor": (89.75, 307.165, 126.164),
    "CHAS_LowAft": (-91.92, 307.564, 130.722),
    "CHAS_UppFor": (80.43, 318.0, 231.596),
    "CHAS_UppAft": (-77.79, 318.0, 251.211),
    "UPRI_LowPnt": (0.0, 535.385, 130.0),
    "UPRI_UppPnt": (0.0, 498.993, 265.0),
    "CHAS_TiePnt": (-95.0, 312.75, 190.4),
    "UPRI_TiePnt": (-70.0, 550.0, 205.0),
    "NSMA_PPAttPnt": (2.55, 468.425, 285.585),
    "CHAS_AttPnt": (214.16, 246.477, 492.514),
    "CHAS_RocAxi": (73.17, 245.855, 409.696),
    "CHAS_RocPiv": (65.03, 263.84, 437.032),
    "ROCK_RodPnt": (19.49, 265.694, 422.107),
    "ROCK_CoiPnt": (57.13, 207.56, 471.016),
    "NSMA_UBarAttPnt": (32.96, 230.398, 448.71),
    "UBAR_AttPnt": (-193.28, 230.398, 381.672),
    "CHAS_PivPnt": (-173.31, 230.398, 314.58),
}


# =============================================================================
# 3D Geometry Primitives (Pure NumPy, float64)
# =============================================================================

def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-15:
        return np.zeros(3, dtype=np.float64)
    return v / n


def _rot_vec(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation of vector v around unit axis by angle (rad)."""
    c = math.cos(angle)
    s = math.sin(angle)
    return v * c + np.cross(axis, v) * s + axis * (float(np.dot(axis, v)) * (1.0 - c))


def _rot_point(p: np.ndarray, origin: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotate point p around line (origin, unit axis) by angle (rad)."""
    return origin + _rot_vec(p - origin, axis, angle)


def _rot_matrix_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    """3x3 Rodrigues rotation matrix about unit axis by angle (rad)."""
    c = math.cos(angle)
    s = math.sin(angle)
    ux, uy, uz = float(axis[0]), float(axis[1]), float(axis[2])
    K = np.array(
        [
            [0.0, -uz, uy],
            [uz, 0.0, -ux],
            [-uy, ux, 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) * c + s * K + (1.0 - c) * np.outer(axis, axis)


def _rot_align_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimal 3x3 rotation matrix R such that R @ a = b (for unit vectors a, b)."""
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s2 = float(np.dot(v, v))
    if s2 < 1e-28:
        if c > 0.0:
            return np.eye(3, dtype=np.float64)
        # 180-deg fallback
        ortho = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(a[0]) > 0.9:
            ortho = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        ax = _normalize(np.cross(a, ortho))
        return _rot_matrix_axis(ax, math.pi)
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    K = np.array(
        [
            [0.0, -vz, vy],
            [vz, 0.0, -vx],
            [-vy, ux := vx, 0.0],
        ],
        dtype=np.float64,
    )
    # Fix K[2, 1] = vx (explicitly)
    K[2, 1] = vx
    K[1, 2] = -vx
    return np.eye(3, dtype=np.float64) + K + (K @ K) / (1.0 + c)


def _solve_circle_sphere(
    p0: np.ndarray,
    origin: np.ndarray,
    axis: np.ndarray,
    target: np.ndarray,
    target_dist: float,
    theta_ref: float = 0.0,
) -> Tuple[float, np.ndarray, float]:
    """
    Exact closed-form intersection of:
      - Circle traced by rotating p0 around (origin, unit axis) by angle theta
      - Sphere centered at target with radius target_dist.

    Returns (theta, rotated_point, abs_residual_m), selecting the branch closest to theta_ref.
    """
    r0 = p0 - origin
    d = target - origin
    r_par = float(np.dot(r0, axis)) * axis
    r_perp = r0 - r_par
    w = np.cross(axis, r_perp)  # orthogonal to axis and r_perp, same norm as r_perp

    # p(th) - target = (r_par - d) + cos(th)*r_perp + sin(th)*w
    # ||p(th) - target||^2 = target_dist^2
    # => A * cos(th) + B * sin(th) = C
    delta_par = r_par - d
    A = 2.0 * float(np.dot(delta_par, r_perp))
    B = 2.0 * float(np.dot(delta_par, w))
    C = target_dist * target_dist - float(np.dot(delta_par, delta_par)) - float(np.dot(r_perp, r_perp))

    R_ab = math.hypot(A, B)
    if R_ab < 1e-15:
        pt = _rot_point(p0, origin, axis, theta_ref)
        err = abs(float(np.linalg.norm(pt - target)) - target_dist)
        return theta_ref, pt, err

    ratio = max(-1.0, min(1.0, C / R_ab))
    phi = math.atan2(B, A)
    acos_val = math.acos(ratio)

    cand1 = phi + acos_val
    cand2 = phi - acos_val

    def _wrap_diff(a: float, b: float) -> float:
        df = (a - b + math.pi) % (2.0 * math.pi) - math.pi
        return abs(df)

    # Wrap candidates near theta_ref
    def _nearest(cand: float, ref: float) -> float:
        return ref + ((cand - ref + math.pi) % (2.0 * math.pi) - math.pi)

    c1 = _nearest(cand1, theta_ref)
    c2 = _nearest(cand2, theta_ref)
    th = c1 if _wrap_diff(c1, theta_ref) <= _wrap_diff(c2, theta_ref) else c2

    # One Newton polish step for machine precision
    pt = _rot_point(p0, origin, axis, th)
    for _ in range(3):
        diff = pt - target
        dist = float(np.linalg.norm(diff))
        err = dist - target_dist
        if abs(err) < 1e-15:
            break
        vel = np.cross(axis, pt - origin)
        derr = float(np.dot(diff, vel)) / max(dist, 1e-15)
        if abs(derr) < 1e-15:
            break
        th -= err / derr
        pt = _rot_point(p0, origin, axis, th)

    err_final = abs(float(np.linalg.norm(pt - target)) - target_dist)
    return th, pt, err_final


# =============================================================================
# Corner Geometry & Exact 3D Solver
# =============================================================================

@dataclass
class Corner3D:
    """Single corner double-wishbone + pushrod + rocker + U-Bar geometry (meters)."""

    name: str
    is_front: bool
    is_right: bool
    pushrod_on_lower: bool
    half_track_m: float
    wheel_radius_m: float
    static_camber_rad: float
    static_toe_rad: float
    wheelbase_m: float
    h_cg_m: float
    brake_bias_front: float
    drive_bias_front: float

    # Hardpoints (m)
    lca_front: np.ndarray
    lca_rear: np.ndarray
    uca_front: np.ndarray
    uca_rear: np.ndarray
    lbj: np.ndarray
    ubj: np.ndarray
    tie_inner: np.ndarray
    tie_outer: np.ndarray
    pr_wb: np.ndarray
    dm_chas: np.ndarray
    rck_axi: np.ndarray
    rck_piv: np.ndarray
    pr_rck: np.ndarray
    dm_rck: np.ndarray
    arb_rck: np.ndarray
    arb_tip: np.ndarray
    arb_piv: np.ndarray

    # Derived quantities
    u_lca: np.ndarray = field(init=False)
    u_uca: np.ndarray = field(init=False)
    u_rck: np.ndarray = field(init=False)
    u_arb: np.ndarray = field(init=False)
    lca_mid: np.ndarray = field(init=False)
    uca_mid: np.ndarray = field(init=False)

    len_kp: float = field(init=False)
    len_tie: float = field(init=False)
    len_pr: float = field(init=False)
    len_dm0: float = field(init=False)
    len_al: float = field(init=False)
    len_lca_f: float = field(init=False)
    len_lca_r: float = field(init=False)
    len_uca_f: float = field(init=False)
    len_uca_r: float = field(init=False)

    contact_patch_0: np.ndarray = field(init=False)
    wheel_center_0: np.ndarray = field(init=False)
    ez0: np.ndarray = field(init=False)
    R_u0: np.ndarray = field(init=False)
    wc_loc: np.ndarray = field(init=False)
    to_loc: np.ndarray = field(init=False)
    spin_loc: np.ndarray = field(init=False)
    arb_jounce_sign: float = field(init=False, default=1.0)

    def __post_init__(self) -> None:
        self.u_lca = _normalize(self.lca_rear - self.lca_front)
        self.u_uca = _normalize(self.uca_rear - self.uca_front)
        self.u_rck = _normalize(self.rck_axi - self.rck_piv)
        self.u_arb = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        self.lca_mid = 0.5 * (self.lca_front + self.lca_rear)
        self.uca_mid = 0.5 * (self.uca_front + self.uca_rear)

        self.len_kp = float(np.linalg.norm(self.ubj - self.lbj))
        self.len_tie = float(np.linalg.norm(self.tie_outer - self.tie_inner))
        self.len_pr = float(np.linalg.norm(self.pr_wb - self.pr_rck))
        self.len_dm0 = float(np.linalg.norm(self.dm_chas - self.dm_rck))
        self.len_al = float(np.linalg.norm(self.arb_tip - self.arb_rck))
        self.len_lca_f = float(np.linalg.norm(self.lbj - self.lca_front))
        self.len_lca_r = float(np.linalg.norm(self.lbj - self.lca_rear))
        self.len_uca_f = float(np.linalg.norm(self.ubj - self.uca_front))
        self.len_uca_r = float(np.linalg.norm(self.ubj - self.uca_rear))

        y_sgn = -1.0 if self.is_right else 1.0
        self.contact_patch_0 = np.array([0.0, y_sgn * self.half_track_m, 0.0], dtype=np.float64)

        # Static wheel spin axis pointing car-left (+Y) with static camber & toe
        # Vehicle camber < 0 => top inward:
        #   Left (+Y):  s_ax.z = -sin(gamma_0) > 0
        #   Right (-Y): s_ax.z = +sin(gamma_0) < 0
        # Vehicle toe > 0 (toe-in):
        #   Left (+Y):  steer_iso = -toe_0 => s_ax rotated by -toe_0 about +Z
        #   Right (-Y): steer_iso = +toe_0 => s_ax rotated by +toe_0 about +Z
        g0 = self.static_camber_rad
        steer0 = self.static_toe_rad if self.is_right else -self.static_toe_rad
        s_ax_0 = np.array(
            [
                -math.sin(steer0) * math.cos(g0),
                math.cos(steer0) * math.cos(g0),
                (math.sin(g0) if self.is_right else -math.sin(g0)),
            ],
            dtype=np.float64,
        )
        s_ax_0 = _normalize(s_ax_0)
        down = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        rad_dn_0 = _normalize(down - float(np.dot(down, s_ax_0)) * s_ax_0)
        # Contact patch is at (0, y_sgn * half_track, 0) at static design pose
        self.wheel_center_0 = self.contact_patch_0 - self.wheel_radius_m * rad_dn_0

        # Static upright orthogonal frame
        self.ez0 = (self.ubj - self.lbj) / self.len_kp
        ey0 = _normalize(np.cross(self.ez0, self.tie_outer - self.lbj))
        ex0 = _normalize(np.cross(ey0, self.ez0))
        self.R_u0 = np.column_stack([ex0, ey0, self.ez0])
        R_u0_T = self.R_u0.T
        self.wc_loc = R_u0_T @ (self.wheel_center_0 - self.lbj)
        self.to_loc = R_u0_T @ (self.tie_outer - self.lbj)
        self.spin_loc = R_u0_T @ s_ax_0

        # Determine ARB sign so d(arb_angle)/d(z_w) > 0 in bump
        sol_p = self.solve_pose(z_w=0.005, d_rack=0.0)
        if sol_p["arb_angle_raw_rad"] < 0.0:
            self.arb_jounce_sign = -1.0
        else:
            self.arb_jounce_sign = 1.0

    @classmethod
    def from_hardpoints_mm(
        cls,
        name: str,
        hp_mm: Dict[str, Tuple[float, float, float]],
        is_front: bool,
        is_right: bool,
        pushrod_on_lower: bool,
        params: Dict[str, float] = TER26_VEHICLE_PARAMS,
    ) -> "Corner3D":
        y_sgn = -1.0 if is_right else 1.0

        def _p(key: str) -> np.ndarray:
            x, y, z = hp_mm[key]
            return np.array([x * 1e-3, y * y_sgn * 1e-3, z * 1e-3], dtype=np.float64)

        half_track = (
            params["half_track_front_mm"] if is_front else params["half_track_rear_mm"]
        ) * 1e-3
        static_camber = math.radians(
            params["static_camber_front_deg"] if is_front else params["static_camber_rear_deg"]
        )
        static_toe = math.radians(
            params["static_toe_front_deg"] if is_front else params["static_toe_rear_deg"]
        )

        return cls(
            name=name,
            is_front=is_front,
            is_right=is_right,
            pushrod_on_lower=pushrod_on_lower,
            half_track_m=half_track,
            wheel_radius_m=params["tire_radius_mm"] * 1e-3,
            static_camber_rad=static_camber,
            static_toe_rad=static_toe,
            wheelbase_m=params["wheelbase_mm"] * 1e-3,
            h_cg_m=params["h_cg_mm"] * 1e-3,
            brake_bias_front=params["brake_bias_front"],
            drive_bias_front=params["drive_bias_front"],
            lca_front=_p("CHAS_LowFor"),
            lca_rear=_p("CHAS_LowAft"),
            uca_front=_p("CHAS_UppFor"),
            uca_rear=_p("CHAS_UppAft"),
            lbj=_p("UPRI_LowPnt"),
            ubj=_p("UPRI_UppPnt"),
            tie_inner=_p("CHAS_TiePnt"),
            tie_outer=_p("UPRI_TiePnt"),
            pr_wb=_p("NSMA_PPAttPnt"),
            dm_chas=_p("CHAS_AttPnt"),
            rck_axi=_p("CHAS_RocAxi"),
            rck_piv=_p("CHAS_RocPiv"),
            pr_rck=_p("ROCK_RodPnt"),
            dm_rck=_p("ROCK_CoiPnt"),
            arb_rck=_p("NSMA_UBarAttPnt"),
            arb_tip=_p("UBAR_AttPnt"),
            arb_piv=_p("CHAS_PivPnt"),
        )

    def _eval_main_residual(
        self, x: np.ndarray, z_w: float, tie_in_tgt: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        th_lca, th_uca, psi_kp = float(x[0]), float(x[1]), float(x[2])
        lbj_p = _rot_point(self.lbj, self.lca_front, self.u_lca, th_lca)
        ubj_p = _rot_point(self.ubj, self.uca_front, self.u_uca, th_uca)
        kp_vec = ubj_p - lbj_p
        dist_kp = float(np.linalg.norm(kp_vec))
        ez = kp_vec / max(dist_kp, 1e-12)

        R_align = _rot_align_vectors(self.ez0, ez)
        R_spin = _rot_matrix_axis(ez, psi_kp)
        R_u = R_spin @ R_align @ self.R_u0

        to_p = lbj_p + R_u @ self.to_loc
        wc_p = lbj_p + R_u @ self.wc_loc
        s_ax = _normalize(R_u @ self.spin_loc)
        down = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        rad_dn = _normalize(down - float(np.dot(down, s_ax)) * s_ax)
        cp_p = wc_p + self.wheel_radius_m * rad_dn

        f1 = dist_kp - self.len_kp
        f2 = float(np.linalg.norm(to_p - tie_in_tgt)) - self.len_tie
        f3 = (cp_p[2] - self.contact_patch_0[2]) - z_w
        return np.array([f1, f2, f3], dtype=np.float64), lbj_p, ubj_p, to_p, wc_p, cp_p, s_ax

    def solve_pose(
        self,
        z_w: float,
        d_rack: float = 0.0,
        init_guess: Optional[np.ndarray] = None,
        theta_rck_ref: float = 0.0,
        theta_arb_ref: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Exact 3D spatial loop closure at vertical wheel travel z_w (m) and rack shift d_rack (m).
        """
        rack_y = d_rack if self.is_front else 0.0
        tie_in_tgt = self.tie_inner + np.array([0.0, rack_y, 0.0], dtype=np.float64)
        x = np.zeros(3, dtype=np.float64) if init_guess is None else np.array(init_guess, dtype=np.float64).copy()

        r, lbj_p, ubj_p, to_p, wc_p, cp_p, s_ax = self._eval_main_residual(x, z_w, tie_in_tgt)
        r_norm = float(np.linalg.norm(r))
        eps_j = 1e-7

        for _ in range(40):
            if r_norm < 1e-13:
                break
            J = np.zeros((3, 3), dtype=np.float64)
            for j in range(3):
                xp = x.copy()
                xm = x.copy()
                xp[j] += eps_j
                xm[j] -= eps_j
                rp = self._eval_main_residual(xp, z_w, tie_in_tgt)[0]
                rm = self._eval_main_residual(xm, z_w, tie_in_tgt)[0]
                J[:, j] = (rp - rm) / (2.0 * eps_j)

            try:
                dx = np.linalg.solve(J, -r)
            except np.linalg.LinAlgError:
                dx = np.linalg.lstsq(J, -r, rcond=None)[0]

            dx_norm = float(np.linalg.norm(dx))
            if dx_norm > 0.25:
                dx *= 0.25 / dx_norm

            alpha = 1.0
            accepted = False
            for _ls in range(10):
                x_try = x + alpha * dx
                res_try = self._eval_main_residual(x_try, z_w, tie_in_tgt)
                rn_try = float(np.linalg.norm(res_try[0]))
                if rn_try < r_norm or rn_try < 1e-13:
                    x = x_try
                    r, lbj_p, ubj_p, to_p, wc_p, cp_p, s_ax = res_try
                    r_norm = rn_try
                    accepted = True
                    break
                alpha *= 0.5
            if not accepted:
                x = x + alpha * dx
                r, lbj_p, ubj_p, to_p, wc_p, cp_p, s_ax = self._eval_main_residual(x, z_w, tie_in_tgt)
                r_norm = float(np.linalg.norm(r))

        # Pushrod attachment rotates rigidly with LowerAArm (Front) or UpperAArm (Rear)
        if self.pushrod_on_lower:
            pr_wb_p = _rot_point(self.pr_wb, self.lca_front, self.u_lca, float(x[0]))
        else:
            pr_wb_p = _rot_point(self.pr_wb, self.uca_front, self.u_uca, float(x[1]))

        # Exact 3D rocker closure around CHAS_RocPiv -> CHAS_RocAxi
        th_rck, pr_rck_p, err_pr = _solve_circle_sphere(
            self.pr_rck, self.rck_piv, self.u_rck, pr_wb_p, self.len_pr, theta_rck_ref
        )
        dm_rck_p = _rot_point(self.dm_rck, self.rck_piv, self.u_rck, th_rck)
        len_dm = float(np.linalg.norm(self.dm_chas - dm_rck_p))
        damper_disp_m = self.len_dm0 - len_dm  # positive = compression

        # Exact 3D U-Bar closure (Rocker -> drop link -> U-Bar blade rotating around +Y)
        arb_rck_p = _rot_point(self.arb_rck, self.rck_piv, self.u_rck, th_rck)
        th_arb, arb_tip_p, err_arb = _solve_circle_sphere(
            self.arb_tip, self.arb_piv, self.u_arb, arb_rck_p, self.len_al, theta_arb_ref
        )

        # Link closure residual audit (m)
        err_kp = abs(float(np.linalg.norm(ubj_p - lbj_p)) - self.len_kp)
        err_tie = abs(float(np.linalg.norm(to_p - tie_in_tgt)) - self.len_tie)
        err_zw = abs((cp_p[2] - self.contact_patch_0[2]) - z_w)
        err_lca_f = abs(float(np.linalg.norm(lbj_p - self.lca_front)) - self.len_lca_f)
        err_lca_r = abs(float(np.linalg.norm(lbj_p - self.lca_rear)) - self.len_lca_r)
        err_uca_f = abs(float(np.linalg.norm(ubj_p - self.uca_front)) - self.len_uca_f)
        err_uca_r = abs(float(np.linalg.norm(ubj_p - self.uca_rear)) - self.len_uca_r)
        max_err_m = max(err_kp, err_tie, err_zw, err_pr, err_arb, err_lca_f, err_lca_r, err_uca_f, err_uca_r)

        # Wheel orientation angles
        # Vehicle camber: negative = top inward for BOTH left and right
        horiz = math.hypot(float(s_ax[0]), float(s_ax[1]))
        camber_veh_rad = math.atan2(float(s_ax[2]) if self.is_right else -float(s_ax[2]), horiz)
        fwd_w = _normalize(np.cross(s_ax, np.array([0.0, 0.0, 1.0], dtype=np.float64)))
        steer_iso_rad = math.atan2(float(fwd_w[1]), float(fwd_w[0]))
        # Vehicle toe: positive = toe-in (front inward)
        toe_veh_rad = steer_iso_rad if self.is_right else -steer_iso_rad

        # Dynamic kingpin metrics
        kp_now = ubj_p - lbj_p
        caster_rad = math.atan2(-float(kp_now[0]), float(kp_now[2]))
        kpi_rad = math.atan2(
            float(kp_now[1]) if self.is_right else -float(kp_now[1]),
            float(kp_now[2]),
        )
        t_g = (cp_p[2] - lbj_p[2]) / kp_now[2]
        kp_g = lbj_p + t_g * kp_now
        kp_minus_cp = kp_g - cp_p
        lat_w = _normalize(np.cross(np.array([0.0, 0.0, 1.0], dtype=np.float64), fwd_w))
        lat_outboard = -lat_w if self.is_right else lat_w
        trail_m = float(np.dot(kp_minus_cp, fwd_w))
        scrub_m = float(np.dot(-kp_minus_cp, lat_outboard))

        # Exact 3D FVSA (intersection of LCA & UCA planes with X = cp.x) & Roll Center
        fvsa_ic_y, fvsa_ic_z, rc_z_m = self._compute_fvsa_and_rc(lbj_p, ubj_p, cp_p)
        # Exact 3D SVSA (intersection of LCA & UCA planes with Y = cp.y) & Anti-Dive/Squat
        svsa_ic_x, svsa_ic_z, svsa_slope = self._compute_svsa(lbj_p, ubj_p, cp_p)

        return {
            "x_state": x,
            "theta_rck_rad": th_rck,
            "theta_arb_rad": th_arb,
            "arb_angle_raw_rad": th_arb,
            "arb_angle_rad": self.arb_jounce_sign * th_arb,
            "max_residual_m": max_err_m,
            "max_residual_mm": max_err_m * 1e3,
            "err_kp_mm": err_kp * 1e3,
            "err_tie_mm": err_tie * 1e3,
            "err_zw_mm": err_zw * 1e3,
            "err_pr_mm": err_pr * 1e3,
            "err_arb_mm": err_arb * 1e3,
            "camber_rad": camber_veh_rad,
            "camber_deg": math.degrees(camber_veh_rad),
            "steer_rad": steer_iso_rad,
            "steer_deg": math.degrees(steer_iso_rad),
            "toe_rad": toe_veh_rad,
            "toe_deg": math.degrees(toe_veh_rad),
            "damper_disp_m": damper_disp_m,
            "damper_disp_mm": damper_disp_m * 1e3,
            "damper_len_m": len_dm,
            "cp_dx_m": float(cp_p[0] - self.contact_patch_0[0]),
            "cp_dy_m": float(cp_p[1] - self.contact_patch_0[1]),
            "caster_deg": math.degrees(caster_rad),
            "kpi_deg": math.degrees(kpi_rad),
            "mechanical_trail_m": trail_m,
            "mechanical_trail_mm": trail_m * 1e3,
            "scrub_radius_m": scrub_m,
            "scrub_radius_mm": scrub_m * 1e3,
            "fvsa_ic_y_m": fvsa_ic_y,
            "fvsa_ic_z_m": fvsa_ic_z,
            "roll_center_z_m": rc_z_m,
            "roll_center_z_mm": rc_z_m * 1e3,
            "svsa_ic_x_m": svsa_ic_x,
            "svsa_ic_z_m": svsa_ic_z,
            "svsa_slope": svsa_slope,
            "lbj": lbj_p,
            "ubj": ubj_p,
            "wheel_center": wc_p,
            "contact_patch": cp_p,
        }

    def solve_pose_with_derivatives(
        self,
        z_w: float,
        d_rack: float = 0.0,
        init_guess: Optional[np.ndarray] = None,
        theta_rck_ref: float = 0.0,
        theta_arb_ref: float = 0.0,
        h_z: float = 5e-4,
    ) -> Dict[str, Any]:
        """Exact 3D pose + 2nd-order central-difference derivatives wrt z_w."""
        c0 = self.solve_pose(z_w, d_rack, init_guess, theta_rck_ref, theta_arb_ref)
        cp = self.solve_pose(z_w + h_z, d_rack, c0["x_state"], c0["theta_rck_rad"], c0["theta_arb_rad"])
        cm = self.solve_pose(z_w - h_z, d_rack, c0["x_state"], c0["theta_rck_rad"], c0["theta_arb_rad"])

        mr_spring = (cp["damper_disp_m"] - cm["damper_disp_m"]) / (2.0 * h_z)
        d_mr_dz = (cp["damper_disp_m"] - 2.0 * c0["damper_disp_m"] + cm["damper_disp_m"]) / (h_z * h_z)
        mr_arb = (cp["arb_angle_rad"] - cm["arb_angle_rad"]) / (2.0 * h_z)
        camber_gain_deg_m = (cp["camber_deg"] - cm["camber_deg"]) / (2.0 * h_z)
        bump_steer_deg_m = (cp["toe_deg"] - cm["toe_deg"]) / (2.0 * h_z)

        c0["mr_spring"] = mr_spring
        c0["d_mr_spring_dz"] = d_mr_dz
        c0["mr_arb_rad_m"] = mr_arb
        c0["mr_arb_deg_m"] = math.degrees(mr_arb)
        c0["camber_gain_deg_m"] = camber_gain_deg_m
        c0["bump_steer_deg_m"] = bump_steer_deg_m
        c0["max_residual_mm"] = max(c0["max_residual_mm"], cp["max_residual_mm"], cm["max_residual_mm"])
        return c0

    def _compute_fvsa_and_rc(
        self, lbj_p: np.ndarray, ubj_p: np.ndarray, cp_p: np.ndarray
    ) -> Tuple[float, float, float]:
        """Exact 3D intersection of LCA plane and UCA plane with wheel transverse plane X = cp.x."""
        n_lca = np.cross(self.lca_rear - self.lca_front, lbj_p - self.lca_front)
        d_lca = float(np.dot(n_lca, lbj_p))
        n_uca = np.cross(self.uca_rear - self.uca_front, ubj_p - self.uca_front)
        d_uca = float(np.dot(n_uca, ubj_p))
        n_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        d_x = float(cp_p[0])

        A = np.vstack([n_lca, n_uca, n_x])
        det = float(np.linalg.det(A))
        if abs(det) > 1e-12:
            ic_pt = np.linalg.solve(A, np.array([d_lca, d_uca, d_x], dtype=np.float64))
            ic_y, ic_z = float(ic_pt[1]), float(ic_pt[2])
        else:
            # Parallel wishbones fallback
            y_dir = 1.0 if self.is_right else -1.0
            ic_y = float(cp_p[1]) + y_dir * 1e4
            ic_z = float(cp_p[2])

        dy_arm = ic_y - float(cp_p[1])
        rc_z = 0.0
        if abs(dy_arm) > 1e-6:
            t_rc = -float(cp_p[1]) / dy_arm
            rc_z = t_rc * (ic_z - float(cp_p[2]))
        return ic_y, ic_z, rc_z

    def _compute_svsa(
        self, lbj_p: np.ndarray, ubj_p: np.ndarray, cp_p: np.ndarray
    ) -> Tuple[float, float, float]:
        """Exact 3D intersection of LCA plane and UCA plane with wheel longitudinal plane Y = cp.y."""
        n_lca = np.cross(self.lca_rear - self.lca_front, lbj_p - self.lca_front)
        d_lca = float(np.dot(n_lca, lbj_p))
        n_uca = np.cross(self.uca_rear - self.uca_front, ubj_p - self.uca_front)
        d_uca = float(np.dot(n_uca, ubj_p))
        n_y = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        d_y = float(cp_p[1])

        A = np.vstack([n_lca, n_uca, n_y])
        det = float(np.linalg.det(A))
        ic_x = float(cp_p[0]) - (1.0 if self.is_front else -1.0)
        ic_z = float(cp_p[2])
        slope = 0.0
        if abs(det) > 1e-12:
            ic_pt = np.linalg.solve(A, np.array([d_lca, d_uca, d_y], dtype=np.float64))
            ic_x, ic_z = float(ic_pt[0]), float(ic_pt[2])
            dx_arm = (float(cp_p[0]) - ic_x) if self.is_front else (ic_x - float(cp_p[0]))
            if abs(dx_arm) > 1e-6:
                slope = (ic_z - float(cp_p[2])) / dx_arm
        return ic_x, ic_z, slope


# =============================================================================
# Full TeR26 Vehicle Kinematics Assembly & Analysis
# =============================================================================

class TeR26Kinematics3D:
    """Full 4-corner exact 3D spatial double-wishbone kinematics model for TeR26."""

    def __init__(self, params: Optional[Dict[str, float]] = None) -> None:
        self.params = dict(TER26_VEHICLE_PARAMS if params is None else params)
        self.fl = Corner3D.from_hardpoints_mm(
            "FL", FRONT_LEFT_HARDPOINTS_MM, is_front=True, is_right=False, pushrod_on_lower=True, params=self.params
        )
        self.fr = Corner3D.from_hardpoints_mm(
            "FR", FRONT_LEFT_HARDPOINTS_MM, is_front=True, is_right=True, pushrod_on_lower=True, params=self.params
        )
        self.rl = Corner3D.from_hardpoints_mm(
            "RL", REAR_LEFT_HARDPOINTS_MM, is_front=False, is_right=False, pushrod_on_lower=False, params=self.params
        )
        self.rr = Corner3D.from_hardpoints_mm(
            "RR", REAR_LEFT_HARDPOINTS_MM, is_front=False, is_right=True, pushrod_on_lower=False, params=self.params
        )

    def compute_static_and_stiffness_report(self) -> Dict[str, Any]:
        """Compute complete static metrics, ride stiffnesses, roll stiffnesses, and LLTD."""
        sf = self.fl.solve_pose_with_derivatives(0.0, 0.0)
        sr = self.rl.solve_pose_with_derivatives(0.0, 0.0)

        L = self.params["wheelbase_mm"] * 1e-3
        h_cg = self.params["h_cg_mm"] * 1e-3
        t_f = 2.0 * self.params["half_track_front_mm"] * 1e-3
        t_r = 2.0 * self.params["half_track_rear_mm"] * 1e-3
        w_f = self.params["front_weight_frac"]
        w_r = 1.0 - w_f
        bb_f = self.params["brake_bias_front"]
        db_f = self.params["drive_bias_front"]

        # Anti-dive (%) and Anti-squat (%)
        anti_dive_100_pct = sf["svsa_slope"] * (L / h_cg) * 100.0
        anti_dive_pct = bb_f * anti_dive_100_pct
        anti_squat_100_pct = sr["svsa_slope"] * (L / h_cg) * 100.0
        anti_squat_pct = (1.0 - db_f) * anti_squat_100_pct

        # Spring wheel rates (N/mm and N/m)
        ks_f_n_mm = self.params["spring_rate_front_n_mm"]
        ks_r_n_mm = self.params["spring_rate_rear_n_mm"]
        mr_f = sf["mr_spring"]
        mr_r = sr["mr_spring"]
        kw_f_n_mm = ks_f_n_mm * (mr_f ** 2)
        kw_r_n_mm = ks_r_n_mm * (mr_r ** 2)
        kw_f_n_m = kw_f_n_mm * 1e3
        kw_r_n_m = kw_r_n_mm * 1e3

        # Spring roll stiffnesses: K_phi_spring = 0.5 * K_w * t^2 [Nm/rad] -> * (pi/180) [Nm/deg]
        rad2deg = math.pi / 180.0
        k_phi_spring_f_nm_rad = 0.5 * kw_f_n_m * (t_f ** 2)
        k_phi_spring_r_nm_rad = 0.5 * kw_r_n_m * (t_r ** 2)

        # U-Bar roll stiffness:
        # In pure roll phi (rad), z_L = +0.5*t*phi, z_R = -0.5*t*phi
        # Twist delta_theta = theta_L - theta_R = mr_arb * t * phi
        # Torque T_bar = K_bar_nm_rad * delta_theta
        # Wheel force F_w = T_bar * mr_arb; roll moment M_phi = F_w * t = K_bar_nm_rad * (mr_arb * t)^2 * phi
        k_bar_f_nm_deg = self.params["ubar_stiffness_front_nm_deg"]
        k_bar_r_nm_deg = self.params["ubar_stiffness_rear_nm_deg"]
        k_bar_f_nm_rad = k_bar_f_nm_deg / rad2deg
        k_bar_r_nm_rad = k_bar_r_nm_deg / rad2deg

        mr_arb_f = sf["mr_arb_rad_m"]
        mr_arb_r = sr["mr_arb_rad_m"]
        k_phi_arb_f_nm_rad = k_bar_f_nm_rad * ((mr_arb_f * t_f) ** 2)
        k_phi_arb_r_nm_rad = k_bar_r_nm_rad * ((mr_arb_r * t_r) ** 2)
        k_arb_wheel_f_n_m = 2.0 * k_bar_f_nm_rad * (mr_arb_f ** 2)
        k_arb_wheel_r_n_m = 2.0 * k_bar_r_nm_rad * (mr_arb_r ** 2)

        k_phi_f_nm_rad = k_phi_spring_f_nm_rad + k_phi_arb_f_nm_rad
        k_phi_r_nm_rad = k_phi_spring_r_nm_rad + k_phi_arb_r_nm_rad
        k_phi_f_nm_deg = k_phi_f_nm_rad * rad2deg
        k_phi_r_nm_deg = k_phi_r_nm_rad * rad2deg

        # Elastic roll stiffness share & exact mechanical LLTD
        elastic_front_share = k_phi_f_nm_rad / (k_phi_f_nm_rad + k_phi_r_nm_rad)
        h_rc_f = sf["roll_center_z_m"]
        h_rc_r = sr["roll_center_z_m"]
        h_roll = h_cg - (w_f * h_rc_f + w_r * h_rc_r)

        # Lateral load transfer per unit (m * a_y):
        # dFz_f / (m*ay) = (1/t_f) * (elastic_front_share * h_roll + w_f * h_rc_f)
        # dFz_r / (m*ay) = (1/t_r) * ((1 - elastic_front_share) * h_roll + w_r * h_rc_r)
        dfz_f_unit = (elastic_front_share * h_roll + w_f * h_rc_f) / t_f
        dfz_r_unit = ((1.0 - elastic_front_share) * h_roll + w_r * h_rc_r) / t_r
        lltd_mechanical = dfz_f_unit / (dfz_f_unit + dfz_r_unit)

        return {
            "front": {
                "static_camber_deg": sf["camber_deg"],
                "static_toe_deg": sf["toe_deg"],
                "camber_gain_deg_per_m": sf["camber_gain_deg_m"],
                "bump_steer_deg_per_m": sf["bump_steer_deg_m"],
                "mr_spring": mr_f,
                "d_mr_spring_dz_per_m": sf["d_mr_spring_dz"],
                "mr_arb_rad_per_m": mr_arb_f,
                "mr_arb_deg_per_m": sf["mr_arb_deg_m"],
                "roll_center_height_mm": sf["roll_center_z_mm"],
                "fvsa_ic_y_mm": sf["fvsa_ic_y_m"] * 1e3,
                "fvsa_ic_z_mm": sf["fvsa_ic_z_m"] * 1e3,
                "svsa_ic_x_mm": sf["svsa_ic_x_m"] * 1e3,
                "svsa_ic_z_mm": sf["svsa_ic_z_m"] * 1e3,
                "svsa_slope": sf["svsa_slope"],
                "anti_dive_pct_at_70pct_bb": anti_dive_pct,
                "anti_dive_pct_at_100pct_bb": anti_dive_100_pct,
                "caster_deg": sf["caster_deg"],
                "kpi_deg": sf["kpi_deg"],
                "scrub_radius_mm": sf["scrub_radius_mm"],
                "mechanical_trail_mm": sf["mechanical_trail_mm"],
                "spring_rate_n_mm": ks_f_n_mm,
                "wheel_rate_n_mm": kw_f_n_mm,
                "arb_wheel_rate_n_mm": k_arb_wheel_f_n_m * 1e-3,
                "k_phi_spring_nm_deg": k_phi_spring_f_nm_rad * rad2deg,
                "k_phi_arb_nm_deg": k_phi_arb_f_nm_rad * rad2deg,
                "k_phi_total_nm_deg": k_phi_f_nm_deg,
            },
            "rear": {
                "static_camber_deg": sr["camber_deg"],
                "static_toe_deg": sr["toe_deg"],
                "camber_gain_deg_per_m": sr["camber_gain_deg_m"],
                "bump_steer_deg_per_m": sr["bump_steer_deg_m"],
                "mr_spring": mr_r,
                "d_mr_spring_dz_per_m": sr["d_mr_spring_dz"],
                "mr_arb_rad_per_m": mr_arb_r,
                "mr_arb_deg_per_m": sr["mr_arb_deg_m"],
                "roll_center_height_mm": sr["roll_center_z_mm"],
                "fvsa_ic_y_mm": sr["fvsa_ic_y_m"] * 1e3,
                "fvsa_ic_z_mm": sr["fvsa_ic_z_m"] * 1e3,
                "svsa_ic_x_mm": sr["svsa_ic_x_m"] * 1e3,
                "svsa_ic_z_mm": sr["svsa_ic_z_m"] * 1e3,
                "svsa_slope": sr["svsa_slope"],
                "anti_squat_pct_rwd": anti_squat_pct,
                "caster_deg": sr["caster_deg"],
                "kpi_deg": sr["kpi_deg"],
                "scrub_radius_mm": sr["scrub_radius_mm"],
                "mechanical_trail_mm": sr["mechanical_trail_mm"],
                "spring_rate_n_mm": ks_r_n_mm,
                "wheel_rate_n_mm": kw_r_n_mm,
                "arb_wheel_rate_n_mm": k_arb_wheel_r_n_m * 1e-3,
                "k_phi_spring_nm_deg": k_phi_spring_r_nm_rad * rad2deg,
                "k_phi_arb_nm_deg": k_phi_arb_r_nm_rad * rad2deg,
                "k_phi_total_nm_deg": k_phi_r_nm_deg,
            },
            "vehicle": {
                "wheelbase_mm": self.params["wheelbase_mm"],
                "track_front_mm": 2.0 * self.params["half_track_front_mm"],
                "track_rear_mm": 2.0 * self.params["half_track_rear_mm"],
                "h_cg_mm": self.params["h_cg_mm"],
                "h_roll_mm": h_roll * 1e3,
                "elastic_roll_stiffness_front_frac": elastic_front_share,
                "lltd_mechanical": lltd_mechanical,
                "lltd_mechanical_pct": lltd_mechanical * 100.0,
            },
        }

    def compute_ackermann_curve(
        self,
        z_w_m: float = 0.0,
        mean_steer_deg_targets: Optional[List[float]] = None,
    ) -> List[Dict[str, float]]:
        """
        Compute exact Ackermann curve (delta_FL, delta_FR, d_rack, % Ackermann)
        for mean steer angles from 0 to 20 deg (left turn, d_rack >= 0).
        """
        if mean_steer_deg_targets is None:
            mean_steer_deg_targets = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]

        L = self.params["wheelbase_mm"] * 1e-3
        t_f = 2.0 * self.params["half_track_front_mm"] * 1e-3
        ratio_mm_rev = self.params["steering_ratio_mm_per_rev"]

        rows: List[Dict[str, float]] = []
        r_guess = 0.0
        x_fl = np.zeros(3, dtype=np.float64)
        x_fr = np.zeros(3, dtype=np.float64)

        for target_deg in mean_steer_deg_targets:
            target_rad = math.radians(target_deg)
            if abs(target_deg) < 1e-12:
                r = 0.0
                p_fl = self.fl.solve_pose(z_w_m, 0.0, x_fl)
                p_fr = self.fr.solve_pose(z_w_m, 0.0, x_fr)
            else:
                r = r_guess
                for _ in range(25):
                    p_fl = self.fl.solve_pose(z_w_m, r, x_fl)
                    p_fr = self.fr.solve_pose(z_w_m, r, x_fr)
                    x_fl = p_fl["x_state"]
                    x_fr = p_fr["x_state"]
                    mean_st = 0.5 * (p_fl["steer_rad"] + p_fr["steer_rad"])
                    err = mean_st - target_rad
                    if abs(err) < 1e-11:
                        break
                    dr = 1e-5
                    p_fl_p = self.fl.solve_pose(z_w_m, r + dr, x_fl)
                    p_fr_p = self.fr.solve_pose(z_w_m, r + dr, x_fr)
                    dmean = (0.5 * (p_fl_p["steer_rad"] + p_fr_p["steer_rad"]) - mean_st) / dr
                    r -= err / dmean
                r_guess = r
                p_fl = self.fl.solve_pose(z_w_m, r, x_fl)
                p_fr = self.fr.solve_pose(z_w_m, r, x_fr)

            d_in = p_fl["steer_rad"]
            d_out = p_fr["steer_rad"]
            mean_deg = 0.5 * (p_fl["steer_deg"] + p_fr["steer_deg"])
            if d_out > 1e-6:
                d_in_ideal = math.atan(1.0 / (1.0 / math.tan(d_out) - t_f / L))
                ack_pct = ((d_in - d_out) / (d_in_ideal - d_out)) * 100.0
            else:
                ack_pct = 0.0

            sw_deg = (r * 1e3 / ratio_mm_rev) * 360.0
            rows.append(
                {
                    "mean_steer_deg": mean_deg,
                    "d_rack_mm": r * 1e3,
                    "steering_wheel_deg": sw_deg,
                    "delta_FL_deg": p_fl["steer_deg"],
                    "delta_FR_deg": p_fr["steer_deg"],
                    "delta_diff_deg": p_fl["steer_deg"] - p_fr["steer_deg"],
                    "ackermann_pct": ack_pct,
                    "camber_FL_deg": p_fl["camber_deg"],
                    "camber_FR_deg": p_fr["camber_deg"],
                    "max_residual_mm": max(p_fl["max_residual_mm"], p_fr["max_residual_mm"]),
                }
            )
        return rows

    def verify_full_domain(
        self,
        z_min_mm: float = -40.0,
        z_max_mm: float = 40.0,
        n_z: int = 33,
        rack_min_mm: float = -45.0,
        rack_max_mm: float = 45.0,
        n_rack: int = 37,
    ) -> Dict[str, Any]:
        """Verify exact 3D loop closure across full travel and full rack sweep."""
        z_grid = np.linspace(z_min_mm * 1e-3, z_max_mm * 1e-3, n_z)
        r_grid = np.linspace(rack_min_mm * 1e-3, rack_max_mm * 1e-3, n_rack)
        mid_z = n_z // 2
        mid_r = n_rack // 2

        max_res_fl = 0.0
        max_res_fr = 0.0
        max_res_rl = 0.0
        max_res_rr = 0.0
        max_lr_bump_sym_err = 0.0
        max_lr_steer_antisym_err = 0.0

        # Front sweep (warm-started from center outward)
        r_order = list(range(mid_r, n_rack)) + list(range(mid_r - 1, -1, -1))
        z_up = list(range(mid_z, n_z))
        z_dn = list(range(mid_z - 1, -1, -1))

        x_center_fl = np.zeros(3, dtype=np.float64)
        x_center_fr = np.zeros(3, dtype=np.float64)
        for ir in r_order:
            if ir == mid_r - 1:
                x_center_fl = np.zeros(3, dtype=np.float64)
                x_center_fr = np.zeros(3, dtype=np.float64)
            r_val = float(r_grid[ir])
            p0_fl = self.fl.solve_pose(0.0, r_val, x_center_fl)
            p0_fr = self.fr.solve_pose(0.0, r_val, x_center_fr)
            x_center_fl = p0_fl["x_state"]
            x_center_fr = p0_fr["x_state"]

            for z_seq in (z_up, z_dn):
                xf_l = x_center_fl.copy()
                xf_r = x_center_fr.copy()
                tr_l, ta_l = p0_fl["theta_rck_rad"], p0_fl["theta_arb_rad"]
                tr_r, ta_r = p0_fr["theta_rck_rad"], p0_fr["theta_arb_rad"]
                for iz in z_seq:
                    zv = float(z_grid[iz])
                    pl = self.fl.solve_pose(zv, r_val, xf_l, tr_l, ta_l)
                    pr = self.fr.solve_pose(zv, r_val, xf_r, tr_r, ta_r)
                    xf_l, tr_l, ta_l = pl["x_state"], pl["theta_rck_rad"], pl["theta_arb_rad"]
                    xf_r, tr_r, ta_r = pr["x_state"], pr["theta_rck_rad"], pr["theta_arb_rad"]
                    max_res_fl = max(max_res_fl, pl["max_residual_mm"])
                    max_res_fr = max(max_res_fr, pr["max_residual_mm"])
                    if ir == mid_r:
                        max_lr_bump_sym_err = max(
                            max_lr_bump_sym_err,
                            abs(pl["camber_deg"] - pr["camber_deg"]),
                            abs(pl["toe_deg"] - pr["toe_deg"]),
                            abs(pl["damper_disp_mm"] - pr["damper_disp_mm"]),
                        )

        # Steer antisymmetry check: FL(z, +r) vs FR(z, -r)
        for zv in (-0.030, 0.0, 0.030):
            for rv in (0.015, 0.035):
                pl = self.fl.solve_pose(zv, rv)
                pr = self.fr.solve_pose(zv, -rv)
                max_lr_steer_antisym_err = max(
                    max_lr_steer_antisym_err,
                    abs(pl["steer_deg"] + pr["steer_deg"]),
                    abs(pl["camber_deg"] - pr["camber_deg"]),
                )

        # Rear sweep
        for z_seq in (z_up, z_dn):
            xr_l = np.zeros(3, dtype=np.float64)
            xr_r = np.zeros(3, dtype=np.float64)
            tr_l, ta_l = 0.0, 0.0
            tr_r, ta_r = 0.0, 0.0
            for iz in z_seq:
                zv = float(z_grid[iz])
                pl = self.rl.solve_pose(zv, 0.0, xr_l, tr_l, ta_l)
                pr = self.rr.solve_pose(zv, 0.0, xr_r, tr_r, ta_r)
                xr_l, tr_l, ta_l = pl["x_state"], pl["theta_rck_rad"], pl["theta_arb_rad"]
                xr_r, tr_r, ta_r = pr["x_state"], pr["theta_rck_rad"], pr["theta_arb_rad"]
                max_res_rl = max(max_res_rl, pl["max_residual_mm"])
                max_res_rr = max(max_res_rr, pr["max_residual_mm"])
                max_lr_bump_sym_err = max(
                    max_lr_bump_sym_err,
                    abs(pl["camber_deg"] - pr["camber_deg"]),
                    abs(pl["toe_deg"] - pr["toe_deg"]),
                    abs(pl["damper_disp_mm"] - pr["damper_disp_mm"]),
                )

        max_overall = max(max_res_fl, max_res_fr, max_res_rl, max_res_rr)
        return {
            "z_range_mm": [z_min_mm, z_max_mm],
            "d_rack_range_mm": [rack_min_mm, rack_max_mm],
            "grid_points_front": n_z * n_rack,
            "grid_points_rear": n_z,
            "max_residual_fl_mm": max_res_fl,
            "max_residual_fr_mm": max_res_fr,
            "max_residual_rl_mm": max_res_rl,
            "max_residual_rr_mm": max_res_rr,
            "max_residual_overall_mm": max_overall,
            "closure_pass_lt_1e6_mm": bool(max_overall < 1e-6),
            "max_lr_bump_symmetry_err": max_lr_bump_sym_err,
            "max_lr_steer_antisymmetry_err": max_lr_steer_antisym_err,
        }


if __name__ == "__main__":
    kin = TeR26Kinematics3D()
    rep = kin.compute_static_and_stiffness_report()
    ver = kin.verify_full_domain()
    ack = kin.compute_ackermann_curve()
    print("=== STATIC & STIFFNESS REPORT ===", flush=True)
    print(json.dumps(rep, indent=2), flush=True)
    print("=== FULL DOMAIN VERIFICATION ===", flush=True)
    print(json.dumps(ver, indent=2), flush=True)
    print("=== ACKERMANN CURVE (z_w = 0) ===", flush=True)
    for row in ack:
        print(
            f"mean={row['mean_steer_deg']:5.2f} deg | rack={row['d_rack_mm']:6.2f} mm | "
            f"SW={row['steering_wheel_deg']:6.2f} deg | FL={row['delta_FL_deg']:6.3f} | "
            f"FR={row['delta_FR_deg']:6.3f} | Ack={row['ackermann_pct']:6.2f}% | "
            f"res={row['max_residual_mm']:.2e} mm",
            flush=True,
        )
