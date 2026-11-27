#!/usr/bin/env python3
"""
High-Fidelity Smooth C^2 CasADi RWD Vehicle Model for Tecnun eRacing TeR26.

Couples:
  - Exact 3D double-wishbone spatial kinematics C^2 cubic B-Spline LUTs from
    `ter26_kinematics_casadi_loader.py` (camber, Ackermann delta_FL/FR, bump-steer,
    spring MR, U-bar ARB MR, roll center heights h_RC_f/r, anti-dive/anti-squat
    SVSA slopes, instantaneous axle roll stiffnesses K_phi_f/r, mechanical LLTD).
  - Hoosier 16.0x7.5-10 R20 Pacejka MF 6.2 combined-slip tire model (smooth C^2).
  - First-order tire lateral relaxation length (sigma_y).
  - Steering column/rack compliance (k_steer_comp) from pneumatic + mechanical trail.
  - Thermal grip coupling (per-corner tire temperature multiplier).
  - RWD powertrain (open/locked rear drive torque distribution, hydraulic brake bias).
  - Parametric SysID vector (11 physical parameters) for CasADi + IPOPT Multiple-Shooting.

Coordinate convention: ISO 8855 / ROS REP-103 (+x forward, +y left, +z up).
Wheel order: [0: FL, 1: FR, 2: RL, 3: RR].
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import casadi as ca
import numpy as np

CORE_DIR = Path(__file__).resolve().parent
LAB_DIR = CORE_DIR.parents[1]
REPO_ROOT = LAB_DIR.parent
CONFIG_DIR = REPO_ROOT / "src" / "mirena_sim" / "Scenes" / "Vehicle" / "Config" / "vehicles" / "ter26_rwd"
MF62_DIR = REPO_ROOT / "src" / "mirena_sim" / "Scenes" / "Vehicle" / "Dynamics" / "Tires" / "MF62"
KIN_DIR = LAB_DIR / "results" / "ter26_extended_correlation"

for _p in (str(REPO_ROOT), str(CORE_DIR), str(KIN_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from vd_correlation_lab.models.vehicle_14dof.ter27_physics import PacejkaMF62  # noqa: E402
from vd_correlation_lab.models.vehicle_14dof.ter27_casadi_model import (  # noqa: E402
    sabs,
    sclip,
    smax,
    smin,
    sneg,
    spos,
    ssign,
    thermal_grip_multiplier_smooth,
)
from ter26_kinematics_casadi_loader import TeR26KinematicsCasADi, load_ter26_kinematics  # noqa: E402


# Ordered 11-parameter SysID vector
SYSID_PARAM_NAMES: Tuple[str, ...] = (
    "mu_x_scale",       # Longitudinal peak grip multiplier (scales lambda_mux)
    "mu_y_scale",       # Lateral peak grip multiplier (scales lambda_muy)
    "kya_scale",        # Cornering stiffness scale (scales MF6.2 PKY1 / K_y_alpha)
    "camber_scale",     # Camber thrust & sensitivity scale
    "h_cg",             # CG height [m] (CAD nominal 0.270 m)
    "i_zz",             # Total yaw moment of inertia [kg*m^2]
    "cl_a",             # Aerodynamic downforce CL*A [m^2]
    "cd_a",             # Aerodynamic drag CD*A [m^2]
    "c_damp_scale",     # Effective yaw/roll damping & load-transfer damping scale
    "k_steer_comp",     # Steering compliance stiffness [Nm/rad]
    "sigma_y",          # Tire lateral relaxation length [m]
)


@dataclass
class TeR26Params:
    """Nominal TeR26 physical parameters combining repo JSONs and exact 3D kinematics CAD."""

    mass: float = 300.0
    sprung_mass: float = 269.0
    unsprung_mass_front: float = 7.74
    unsprung_mass_rear: float = 7.76
    unsprung_mass_total: float = 31.0
    gravity: float = 9.80665

    wheelbase: float = 1.550
    track_front: float = 1.200
    track_rear: float = 1.180
    front_weight_dist: float = 0.46  # CAD/brief nominal (46% F / 54% R)
    a_cog: float = 0.837             # L * (1 - 0.46)
    b_cog: float = 0.713             # L * 0.46
    h_cg: float = 0.270              # CAD/brief nominal
    h_sprung_cg: float = 0.2777      # (m*h_cg - m_u*r_w) / m_s

    i_xx: float = 13.8
    i_yy: float = 74.5
    i_zz: float = 73.0

    # Suspension springs & U-bars from TER21 FINAL.xlsx kinematics
    k_spring_f: float = 44000.0      # N/m (44 N/mm)
    k_spring_r: float = 53000.0      # N/m (53 N/mm)
    k_ubar_f_nm_rad: float = 5.0 * 180.0 / math.pi  # 5.0 Nm/deg
    k_ubar_r_nm_rad: float = 5.0 * 180.0 / math.pi  # 5.0 Nm/deg
    k_tire_z_f: float = 67020.0      # N/m
    k_tire_z_r: float = 93700.0      # N/m
    c_damp_f: float = 2323.8         # Ns/m
    c_damp_r: float = 3201.6         # Ns/m

    # Steering & Brakes
    max_steer_rad: float = 0.38
    steer_ratio: float = 1.255
    k_steer_comp: float = 9500.0     # Nm/rad
    brake_bias_front: float = 0.68
    brake_mu: float = 1.15
    f_brk_max_n: float = 6000.0      # Full hydraulic brake force at brake=1.0

    # Aero
    air_density: float = 1.18
    cd_a: float = 1.35
    cl_a: float = 4.554
    aero_balance_front: float = 0.45

    # Tires (Hoosier 16.0x7.5-10 R20)
    wheel_radius: float = 0.2032
    wheel_inertia: float = 0.35
    rolling_resistance_coeff: float = 0.01
    tire_compound: str = "R20"
    lambda_mux_base: float = 0.78
    lambda_muy_base: float = 0.64
    p_infl_kpa: float = 82.737       # 12 psi nominal Hoosier R20
    sigma_y_nom: float = 0.22        # Lateral relaxation length [m]

    # Powertrain (RWD)
    power_limit_watts: float = 80000.0
    motor_peak_torque: float = 21.0
    motor_max_rpm: float = 20000.0
    base_speed_rpm: float = 8800.0
    gear_ratio: float = 11.5
    drivetrain_efficiency: float = 0.95
    default_trq_limit_nm: float = 120.0


def load_ter26_params(config_dir: Path = CONFIG_DIR, use_cad_3d: bool = True) -> TeR26Params:
    """Load TeR26 parameters from repo config JSONs, optionally applying 3D CAD nominal specs."""
    veh = json.loads((config_dir / "vehicle.json").read_text(encoding="utf-8"))
    aero = json.loads((config_dir / "aero.json").read_text(encoding="utf-8"))
    pt = json.loads((config_dir / "powertrain.json").read_text(encoding="utf-8"))
    susp = json.loads((config_dir / "suspension.json").read_text(encoding="utf-8"))
    tires = json.loads((config_dir / "tires.json").read_text(encoding="utf-8"))
    brakes = json.loads((config_dir / "brakes.json").read_text(encoding="utf-8"))

    mass = float(veh["mass"])
    m_s = float(veh["sprung_mass"])
    m_u_tot = float(veh["unsprung_mass_total"])
    r_w = float(tires["wheel_radius"])
    L = float(veh["wheelbase"])

    if use_cad_3d:
        fwd = 0.46
        h_cg = 0.270
        k_sf = 44000.0
        k_sr = 53000.0
        bb_f = 0.70
    else:
        fwd = float(veh["front_weight_distribution"])
        h_cg = float(veh["cg_height"])
        k_sf = float(susp["spring_stiffness_front"])
        k_sr = float(susp["spring_stiffness_rear"])
        bb_f = float(brakes["brake_bias_front"])

    h_s = max((mass * h_cg - m_u_tot * r_w) / max(m_s, 1.0), 0.15)
    return TeR26Params(
        mass=mass,
        sprung_mass=m_s,
        unsprung_mass_front=float(veh.get("unsprung_mass_front", 7.74)),
        unsprung_mass_rear=float(veh.get("unsprung_mass_rear", 7.76)),
        unsprung_mass_total=m_u_tot,
        gravity=float(veh.get("gravity", 9.81)),
        wheelbase=L,
        track_front=float(veh["track_front"]),
        track_rear=float(veh["track_rear"]),
        front_weight_dist=fwd,
        a_cog=L * (1.0 - fwd),
        b_cog=L * fwd,
        h_cg=h_cg,
        h_sprung_cg=h_s,
        i_xx=float(veh["inertia"]["ixx"]),
        i_yy=float(veh["inertia"]["iyy"]),
        i_zz=float(veh["inertia"]["izz"]),
        k_spring_f=k_sf,
        k_spring_r=k_sr,
        k_ubar_f_nm_rad=5.0 * 180.0 / math.pi,
        k_ubar_r_nm_rad=5.0 * 180.0 / math.pi,
        k_tire_z_f=float(tires.get("tire_vertical_stiffness_front", 67020.0)),
        k_tire_z_r=float(tires.get("tire_vertical_stiffness_rear", 93700.0)),
        c_damp_f=float(susp.get("damping_compression_front", 2323.8)),
        c_damp_r=float(susp.get("damping_compression_rear", 3201.6)),
        max_steer_rad=float(veh.get("max_steer_angle", 0.38)),
        steer_ratio=float(veh.get("steering_ratio", 1.255)),
        k_steer_comp=float(susp.get("steering_compliance_stiffness_nm_rad", 9500.0)),
        brake_bias_front=bb_f,
        brake_mu=float(brakes.get("brake_mu", 1.15)),
        f_brk_max_n=3300.0,
        air_density=float(aero.get("air_density", 1.18)),
        cd_a=float(aero.get("cd_a", 1.35)),
        cl_a=float(aero.get("cl_a", 4.554)),
        aero_balance_front=float(aero.get("aero_balance_front", 0.45)),
        wheel_radius=r_w,
        wheel_inertia=float(tires.get("wheel_inertia", 0.35)),
        rolling_resistance_coeff=float(tires.get("rolling_resistance_coeff", 0.01)),
        tire_compound=str(tires.get("tire_compound", "R20")),
        lambda_mux_base=float(tires.get("lambda_mux", 0.78)),
        lambda_muy_base=float(tires.get("lambda_muy", 0.64)),
        sigma_y_nom=0.22,
        power_limit_watts=float(pt.get("power_limit_watts", 80000.0)),
        motor_peak_torque=float(pt.get("motor_peak_torque", 21.0)),
        motor_max_rpm=float(pt.get("motor_max_rpm", 20000.0)),
        base_speed_rpm=float(pt.get("base_speed_rpm", 8800.0)),
        gear_ratio=float(pt.get("gear_ratio", 11.5)),
        drivetrain_efficiency=float(pt.get("drivetrain_efficiency", 0.95)),
    )


def default_sysid_theta(params: Optional[TeR26Params] = None) -> Dict[str, float]:
    """Return nominal values for the 11 SysID parameters."""
    p = load_ter26_params(use_cad_3d=True) if params is None else params
    return {
        "mu_x_scale": 1.0,
        "mu_y_scale": 1.0,
        "kya_scale": 1.0,
        "camber_scale": 1.0,
        "h_cg": float(p.h_cg),
        "i_zz": float(p.i_zz),
        "cl_a": float(p.cl_a),
        "cd_a": float(p.cd_a),
        "c_damp_scale": 1.0,
        "k_steer_comp": float(p.k_steer_comp),
        "sigma_y": float(p.sigma_y_nom),
    }


def theta_dict_to_vec(d: Dict[str, float]) -> np.ndarray:
    return np.array([float(d[k]) for k in SYSID_PARAM_NAMES], dtype=np.float64)


def theta_vec_to_dict(v: np.ndarray) -> Dict[str, float]:
    return {k: float(v[i]) for i, k in enumerate(SYSID_PARAM_NAMES)}


class TeR26HighFidelityCasADi:
    """
    Smooth C^2 CasADi RWD TeR26 vehicle dynamics model coupled with exact 3D
    double-wishbone B-Spline kinematics and Hoosier MF 6.2 tires.
    """

    def __init__(
        self,
        params: Optional[TeR26Params] = None,
        kinematics: Optional[TeR26KinematicsCasADi] = None,
        use_3d_kinematics: bool = True,
        use_thermal_coupling: bool = True,
    ) -> None:
        self.params = load_ter26_params(use_cad_3d=use_3d_kinematics) if params is None else params
        self.kin = load_ter26_kinematics() if (kinematics is None and use_3d_kinematics) else kinematics
        self.use_3d_kinematics = bool(use_3d_kinematics and self.kin is not None)
        self.use_thermal_coupling = bool(use_thermal_coupling)

        self.tire = PacejkaMF62(self.params.tire_compound)

        p = self.params
        self.x_w = np.array([p.a_cog, p.a_cog, -p.b_cog, -p.b_cog], dtype=np.float64)
        self.y_w = np.array([0.5 * p.track_front, -0.5 * p.track_front, 0.5 * p.track_rear, -0.5 * p.track_rear], dtype=np.float64)
        self.side_sign = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float64)

        self.smoothing = {
            "eps_fz_n": 15.0,
            "eps_dfz": 0.05,
            "eps_gamma_rad": 2e-3,
            "eps_camber_f": 0.02,
            "eps_sign_slip": 5e-3,
            "eps_e_clip": 0.05,
            "eps_g_floor": 0.01,
            "eps_speed_ms": 0.5,
            "eps_alpha_eq_sq": 1e-6,
        }

        # Build compiled CasADi functions
        self._build_casadi_functions()

    @staticmethod
    def _shape_cos(b: Any, c: Any, e: Any, x: Any) -> Any:
        bx = b * x
        return ca.cos(c * ca.atan(bx - e * (bx - ca.atan(bx))))

    def eval_tire_mf62_symbolic(
        self,
        kappa: Any,
        alpha: Any,
        fz: Any,
        gamma: Any,
        side: Any,
        mu_x_scale: Any,
        mu_y_scale: Any,
        kya_scale: Any,
        camber_scale: Any,
        mu_thermal: Any = 1.0,
    ) -> Tuple[Any, Any, Any]:
        """Smooth C^2 Pacejka MF 6.2 combined-slip evaluator with SysID scale factors."""
        t = self.tire
        s = self.smoothing
        p = self.params

        fz_pos = spos(fz, s["eps_fz_n"])
        dfz_raw = (fz_pos - t.fnomin) / t.fnomin
        dfz = smin(dfz_raw, t.dfz_max, s["eps_dfz"]) if math.isfinite(t.dfz_max) else dfz_raw
        dpi = (p.p_infl_kpa - t.pinom) / t.pinom

        gamma_eff = gamma * camber_scale
        abs_gamma = sabs(gamma_eff, s["eps_gamma_rad"])
        gamma_sq = gamma_eff * gamma_eff

        lmux_eff = p.lambda_mux_base * mu_x_scale * mu_thermal
        lmuy_eff = p.lambda_muy_base * mu_y_scale * mu_thermal

        # Pure lateral Fy0
        cy = t.PCY1
        camber_f_y = smax(1.0 - t.PDY3 * gamma_sq, 0.25, s["eps_camber_f"])
        mu_y = (
            (t.PDY1 + t.PDY2 * dfz)
            * (1.0 + t.PPY3 * dpi + t.PPY4 * dpi * dpi)
            * camber_f_y
            * lmuy_eff
        )
        dy = mu_y * fz_pos
        shy = side * (t.PHY1 + t.PHY2 * dfz) + t.PHY3 * gamma_eff
        svy = side * fz_pos * (t.PVY1 + t.PVY2 * dfz) + fz_pos * (t.PVY3 + t.PVY4 * dfz) * gamma_eff
        alpha_y = alpha + shy
        kya = (
            kya_scale
            * t.PKY1
            * t.fnomin
            * (1.0 + t.PPY1 * dpi)
            * (1.0 - t.PKY3 * abs_gamma)
            * ca.sin(2.0 * ca.atan(fz_pos / (t.PKY2 * t.fnomin * (1.0 + t.PPY2 * dpi) + 1e-4)))
        )
        by = kya / (cy * dy + 1e-4)
        ey_raw = (t.PEY1 + t.PEY2 * dfz) * (
            1.0 + t.PEY5 * gamma_sq - (t.PEY3 + t.PEY4 * gamma_eff) * ssign(alpha_y, s["eps_sign_slip"])
        )
        ey = sclip(ey_raw, -10.0, 1.0, s["eps_e_clip"])
        y = by * alpha_y
        fy0 = dy * ca.sin(cy * ca.atan(y - ey * (y - ca.atan(y)))) + svy

        # Pure longitudinal Fx0
        cx = t.PCX1
        camber_f_x = smax(1.0 - t.PDX3 * gamma_sq, 0.25, s["eps_camber_f"])
        mu_x = (
            (t.PDX1 + t.PDX2 * dfz)
            * (1.0 + t.PPX3 * dpi + t.PPX4 * dpi * dpi)
            * camber_f_x
            * lmux_eff
        )
        dx = mu_x * fz_pos
        kx = kappa + t.PHX1 + t.PHX2 * dfz
        svx = fz_pos * (t.PVX1 + t.PVX2 * dfz)
        kxk = (
            fz_pos
            * (t.PKX1 + t.PKX2 * dfz)
            * ca.exp(t.PKX3 * dfz)
            * (1.0 + t.PPX1 * dpi + t.PPX2 * dpi * dpi)
        )
        bx = kxk / (cx * dx + 1e-4)
        ex_raw = (t.PEX1 + t.PEX2 * dfz + t.PEX3 * dfz * dfz) * (
            1.0 - t.PEX4 * ssign(kx, s["eps_sign_slip"])
        )
        ex = sclip(ex_raw, -10.0, 1.0, s["eps_e_clip"])
        x = bx * kx
        fx0 = dx * ca.sin(cx * ca.atan(x - ex * (x - ca.atan(x)))) + svx

        # Combined slip weighting
        bxa = (t.RBX1 + t.RBX3 * gamma_sq) * ca.cos(ca.atan(t.RBX2 * kappa))
        exa = sclip(t.REX1 + t.REX2 * dfz, -10.0, 1.0, s["eps_e_clip"])
        gxa_den = smax(self._shape_cos(bxa, t.RCX1, exa, t.RHX1), 1e-4, 1e-5)
        gxa = smax(self._shape_cos(bxa, t.RCX1, exa, alpha + t.RHX1) / gxa_den, 0.0, s["eps_g_floor"])

        byk = (t.RBY1 + t.RBY4 * gamma_sq) * ca.cos(ca.atan(t.RBY2 * (alpha - t.RBY3)))
        eyk = sclip(t.REY1 + t.REY2 * dfz, -10.0, 1.0, s["eps_e_clip"])
        shyk = t.RHY1 + t.RHY2 * dfz
        gyk_den = smax(self._shape_cos(byk, t.RCY1, eyk, shyk), 1e-4, 1e-5)
        gyk = smax(self._shape_cos(byk, t.RCY1, eyk, kappa + shyk) / gyk_den, 0.0, s["eps_g_floor"])

        dvyk = (
            mu_y
            * fz_pos
            * (t.RVY1 + t.RVY2 * dfz + t.RVY3 * gamma_eff)
            * ca.cos(ca.atan(t.RVY4 * alpha))
        )
        svyk = dvyk * ca.sin(t.RVY5 * ca.atan(t.RVY6 * kappa))

        fx = gxa * fx0
        fy = gyk * fy0 + svyk

        # Pneumatic trail & self-aligning torque Mz
        sht = t.QHZ1 + t.QHZ2 * dfz + (t.QHZ3 + t.QHZ4 * dfz) * gamma_eff
        alpha_t = alpha + sht
        kx_eq = (kxk / (kya + 1e-3)) * kappa
        alpha_t_mag = ca.sqrt(alpha_t * alpha_t + kx_eq * kx_eq + s["eps_alpha_eq_sq"])
        alpha_t_signed = ssign(alpha_t, s["eps_sign_slip"]) * alpha_t_mag
        dt_t = (
            (fz_pos / t.fnomin)
            * t.r0
            * (t.QDZ1 + t.QDZ2 * dfz)
            * (1.0 + t.PPZ1 * dpi)
            * (1.0 + t.QDZ3 * abs_gamma + t.QDZ4 * gamma_sq)
        )
        ct = t.QCZ1
        bt = (t.QBZ1 + t.QBZ2 * dfz + t.QBZ3 * dfz * dfz) * (
            1.0 + t.QBZ4 * gamma_eff + t.QBZ5 * abs_gamma
        )
        t_arg = bt * alpha_t_mag
        et_raw = (t.QEZ1 + t.QEZ2 * dfz + t.QEZ3 * dfz * dfz) * (
            1.0 + (t.QEZ4 + t.QEZ5 * gamma_eff) * (2.0 / math.pi) * ca.atan(bt * ct * alpha_t_signed)
        )
        et = sclip(et_raw, -10.0, 1.0, s["eps_e_clip"])
        trail = dt_t * ca.cos(ct * ca.atan(t_arg - et * (t_arg - ca.atan(t_arg)))) * ca.cos(alpha)
        mz = -trail * (fy - svyk)
        return fx, fy, mz

    def eval_dynamics_symbolic(
        self,
        vx: Any,
        vy: Any,
        wz: Any,
        alpha_f_state: Any,
        alpha_r_state: Any,
        delta_cmd: Any,
        ax_cmd_drive: Any,
        ax_lt: Any,
        ay_lt: Any,
        t_tire_4: Any,
        theta: Any,
    ) -> Dict[str, Any]:
        """
        Evaluate full 4-corner forces, moments, wheel bumps, kinematics, and state derivatives.

        Args:
            vx, vy, wz: Body velocities [m/s, m/s, rad/s]
            alpha_f_state, alpha_r_state: Relaxed axle slip angle states [rad]
            delta_cmd: Mean front wheel steer command [rad] (+ = left)
            ax_cmd_drive: Net longitudinal drive/brake command force / mass [m/s^2]
                          (positive = RWD rear drive, negative = hydraulic braking)
            ax_lt, ay_lt: Longitudinal/lateral acceleration for load transfer [m/s^2]
            t_tire_4: 4-vector of tire temperatures [FL, FR, RL, RR] in deg C
            theta: 11-vector of SysID parameters (SYSID_PARAM_NAMES order)
        """
        p = self.params
        s = self.smoothing

        mu_x_scale = theta[0]
        mu_y_scale = theta[1]
        kya_scale = theta[2]
        camber_scale = theta[3]
        h_cg = theta[4]
        i_zz = theta[5]
        cl_a = theta[6]
        cd_a = theta[7]
        c_damp_scale = theta[8]
        k_steer_comp = theta[9]
        sigma_y = theta[10]

        # 1. Aerodynamics
        v_sq = vx * vx + vy * vy
        speed = ca.sqrt(v_sq + s["eps_speed_ms"] ** 2)
        q_dyn = 0.5 * p.air_density * v_sq
        f_drag_x = -0.5 * p.air_density * cd_a * speed * vx
        f_drag_y = -0.5 * p.air_density * cd_a * speed * vy
        f_down_f = q_dyn * cl_a * p.aero_balance_front
        f_down_r = q_dyn * cl_a * (1.0 - p.aero_balance_front)

        # 2. Longitudinal load transfer & static/aero vertical loads
        mg = p.mass * p.gravity
        fz_f_stat = mg * (p.b_cog / p.wheelbase)
        fz_r_stat = mg * (p.a_cog / p.wheelbase)
        dfz_long = (p.mass * ax_lt * h_cg) / p.wheelbase
        fz_f_tot = spos(fz_f_stat - dfz_long + f_down_f, s["eps_fz_n"])
        fz_r_tot = spos(fz_r_stat + dfz_long + f_down_r, s["eps_fz_n"])

        # 3. Suspension geometry, roll stiffness, LLTD, and wheel bump deflections
        if self.use_3d_kinematics and self.kin is not None:
            # Nominal static MRs for initial bump estimate (1-step fixed point for C^2 speed)
            st_f = self.kin.static_summary["front"]
            st_r = self.kin.static_summary["rear"]
            kw_f0 = p.k_spring_f * (st_f["mr_spring"] ** 2)
            kw_r0 = p.k_spring_r * (st_r["mr_spring"] ** 2)
            karb_f0 = p.k_ubar_f_nm_rad * (st_f["mr_arb_rad_per_m"] ** 2)
            karb_r0 = p.k_ubar_r_nm_rad * (st_r["mr_arb_rad_per_m"] ** 2)
            h_rc_f0 = st_f["roll_center_height_mm"] * 1e-3
            h_rc_r0 = st_r["roll_center_height_mm"] * 1e-3
            ad_0 = st_f["anti_dive_pct_at_70pct_bb"] * 0.01
            as_0 = st_r["anti_squat_pct_rwd"] * 0.01
        else:
            kw_f0 = p.k_spring_f * (0.87719 ** 2)
            kw_r0 = p.k_spring_r * (0.86207 ** 2)
            karb_f0 = 332.8
            karb_r0 = 3718.0
            h_rc_f0 = 0.040
            h_rc_r0 = 0.06028
            ad_0 = 0.52
            as_0 = 0.45

        # Series tire + wheel rate for roll stiffness
        ktot_f0 = ((kw_f0 + 2.0 * karb_f0) * p.k_tire_z_f) / (kw_f0 + 2.0 * karb_f0 + p.k_tire_z_f)
        ktot_r0 = ((kw_r0 + 2.0 * karb_r0) * p.k_tire_z_r) / (kw_r0 + 2.0 * karb_r0 + p.k_tire_z_r)
        k_phi_f0 = 0.5 * p.track_front * p.track_front * ktot_f0
        k_phi_r0 = 0.5 * p.track_rear * p.track_rear * ktot_r0
        k_phi_tot0 = k_phi_f0 + k_phi_r0

        m_s = p.sprung_mass
        m_u_f = 2.0 * p.unsprung_mass_front
        m_u_r = 2.0 * p.unsprung_mass_rear
        h_s = p.h_sprung_cg + (h_cg - p.h_cg)
        h_rc_cg0 = h_rc_f0 + (h_rc_r0 - h_rc_f0) * (p.a_cog / p.wheelbase)
        h_roll_arm0 = smax(h_s - h_rc_cg0, 0.01, 1e-4)
        roll_denom0 = smax(k_phi_tot0 - m_s * p.gravity * h_roll_arm0, 100.0, 1.0)
        phi0 = (m_s * ay_lt * h_roll_arm0) / roll_denom0

        # Anti-dive (under braking ax < 0) & anti-squat (under accel ax > 0) reduce pitch bump
        dfz_brk = spos(-dfz_long, 10.0)
        dfz_acc = spos(dfz_long, 10.0)
        z_heave_f = (0.5 * f_down_f + 0.5 * dfz_brk * (1.0 - ad_0) - 0.5 * dfz_acc) / max(kw_f0, 1000.0)
        z_heave_r = (0.5 * f_down_r + 0.5 * dfz_acc * (1.0 - as_0) - 0.5 * dfz_brk) / max(kw_r0, 1000.0)

        # Roll wheel bump: phi > 0 (left turn, ay > 0) compresses right (+z) and rebounds left (-z)
        z_fl = sclip(z_heave_f - 0.5 * p.track_front * phi0, -0.036, 0.036, 1e-3)
        z_fr = sclip(z_heave_f + 0.5 * p.track_front * phi0, -0.036, 0.036, 1e-3)
        z_rl = sclip(z_heave_r - 0.5 * p.track_rear * phi0, -0.036, 0.036, 1e-3)
        z_rr = sclip(z_heave_r + 0.5 * p.track_rear * phi0, -0.036, 0.036, 1e-3)

        # 4. Exact 3D kinematics B-Spline lookup at (z_w, d_rack)
        delta_clamped = sclip(delta_cmd, -0.48, 0.48, 1e-3)
        if self.use_3d_kinematics and self.kin is not None:
            d_rack0 = self.kin.rack_from_mean_steer_m(delta_clamped)
            d_rack0 = sclip(d_rack0, -0.042, 0.042, 1e-4)

            # Evaluate instantaneous MRs and RCs from 3D B-Splines
            mr_sp_f = 0.5 * (self.kin.fl_mr_spring(z_fl, d_rack0) + self.kin.fr_mr_spring(z_fr, d_rack0))
            mr_sp_r = 0.5 * (self.kin.rear_mr_spring(z_rl) + self.kin.rear_mr_spring(z_rr))
            mr_arb_f = 0.5 * (self.kin.fl_mr_arb_rad_m(z_fl, d_rack0) + self.kin.fr_mr_arb_rad_m(z_fr, d_rack0))
            mr_arb_r = 0.5 * (self.kin.rear_mr_arb_rad_m(z_rl) + self.kin.rear_mr_arb_rad_m(z_rr))
            h_rc_f = 0.5 * (self.kin.fl_rc_z_m(z_fl, d_rack0) + self.kin.fr_rc_z_m(z_fr, d_rack0))
            h_rc_r = 0.5 * (self.kin.rear_rc_z_m(z_rl) + self.kin.rear_rc_z_m(z_rr))
            trail_mech_f = 0.5 * (
                self.kin.fl_mech_trail_mm(z_fl, d_rack0) + self.kin.fl_mech_trail_mm(z_fr, -d_rack0)
            ) * 1e-3

            kw_f = p.k_spring_f * mr_sp_f * mr_sp_f
            kw_r = p.k_spring_r * mr_sp_r * mr_sp_r
            karb_f = p.k_ubar_f_nm_rad * mr_arb_f * mr_arb_f
            karb_r = p.k_ubar_r_nm_rad * mr_arb_r * mr_arb_r

            ktot_f = ((kw_f + 2.0 * karb_f) * p.k_tire_z_f) / (kw_f + 2.0 * karb_f + p.k_tire_z_f)
            ktot_r = ((kw_r + 2.0 * karb_r) * p.k_tire_z_r) / (kw_r + 2.0 * karb_r + p.k_tire_z_r)
            k_phi_f = 0.5 * p.track_front * p.track_front * ktot_f
            k_phi_r = 0.5 * p.track_rear * p.track_rear * ktot_r
        else:
            h_rc_f = h_rc_f0
            h_rc_r = h_rc_r0
            k_phi_f = k_phi_f0
            k_phi_r = k_phi_r0
            trail_mech_f = 0.013

        # Refined roll angle & exact 4-corner normal forces
        k_phi_tot = k_phi_f + k_phi_r
        h_rc_cg = h_rc_f + (h_rc_r - h_rc_f) * (p.a_cog / p.wheelbase)
        h_roll_arm = smax(h_s - h_rc_cg, 0.01, 1e-4)
        roll_denom = smax(k_phi_tot - m_s * p.gravity * h_roll_arm, 100.0, 1.0)
        phi = (m_s * ay_lt * h_roll_arm) / roll_denom

        dfz_lat_f_raw = (
            (k_phi_f * phi) / p.track_front
            + (m_s * (p.b_cog / p.wheelbase) * ay_lt * h_rc_f) / p.track_front
            + (m_u_f * ay_lt * p.wheel_radius) / p.track_front
        )
        dfz_lat_r_raw = (
            (k_phi_r * phi) / p.track_rear
            + (m_s * (p.a_cog / p.wheelbase) * ay_lt * h_rc_r) / p.track_rear
            + (m_u_r * ay_lt * p.wheel_radius) / p.track_rear
        )
        dfz_lat_f = sclip(dfz_lat_f_raw, -0.5 * fz_f_tot, 0.5 * fz_f_tot, s["eps_fz_n"])
        dfz_lat_r = sclip(dfz_lat_r_raw, -0.5 * fz_r_tot, 0.5 * fz_r_tot, s["eps_fz_n"])

        fz = ca.vertcat(
            spos(0.5 * fz_f_tot - dfz_lat_f, s["eps_fz_n"]),
            spos(0.5 * fz_f_tot + dfz_lat_f, s["eps_fz_n"]),
            spos(0.5 * fz_r_tot - dfz_lat_r, s["eps_fz_n"]),
            spos(0.5 * fz_r_tot + dfz_lat_r, s["eps_fz_n"]),
        )

        # 5. Steering compliance (1st-order algebraic compliance from front lateral force * trail)
        # Approximate front axle lateral force from ay_lt * m * b / L for smooth compliance deflection
        fy_f_est = p.mass * (p.b_cog / p.wheelbase) * ay_lt
        tau_steer_est = fy_f_est * (trail_mech_f + 0.015)  # mech trail + ~15mm pneumatic trail
        delta_comp = sclip(tau_steer_est / smax(k_steer_comp, 1000.0, 50.0), -0.06, 0.06, 1e-4)
        delta_eff = sclip(delta_cmd - delta_comp, -0.48, 0.48, 1e-3)

        # Exact 3D Ackermann + bump steer + camber
        if self.use_3d_kinematics and self.kin is not None:
            d_rack = sclip(self.kin.rack_from_mean_steer_m(delta_eff), -0.042, 0.042, 1e-4)
            delta_fl = self.kin.fl_steer_rad(z_fl, d_rack)
            delta_fr = self.kin.fr_steer_rad(z_fr, d_rack)
            # Rear toe > 0 is toe-in: RL points right (-delta), RR points left (+delta)
            delta_rl = -self.kin.rear_toe_rad(z_rl)
            delta_rr = self.kin.rear_toe_rad(z_rr)
            # ISO 8855 ground camber
            g_fl = self.kin.fl_camber_rad(z_fl, d_rack) - phi
            g_fr = -self.kin.fr_camber_rad(z_fr, d_rack) - phi
            g_rl = self.kin.rear_camber_rad(z_rl) - phi
            g_rr = -self.kin.rear_camber_rad(z_rr) - phi
        else:
            # Linear Ackermann + static camber baseline
            ack_k = -0.1
            delta_fl = delta_eff * (1.0 + ack_k * 0.1 * delta_eff)
            delta_fr = delta_eff * (1.0 - ack_k * 0.1 * delta_eff)
            delta_rl = 0.0
            delta_rr = 0.0
            cam_f0 = math.radians(-2.1)
            cam_r0 = math.radians(-1.4)
            g_fl = cam_f0 - phi
            g_fr = -cam_f0 - phi
            g_rl = cam_r0 - phi
            g_rr = -cam_r0 - phi

        deltas = [delta_fl, delta_fr, delta_rl, delta_rr]
        gammas = [g_fl, g_fr, g_rl, g_rr]

        # 6. Kinematic slip angles at all 4 corners + relaxation-length offset
        # Compute axle-mean kinematic slip angles and apply relaxation state offset cleanly
        vx_safe = smax(vx, 1.5, 0.3)
        alpha_kin_list = []
        v_wx_list = []
        v_wy_list = []
        for i in range(4):
            v_ix = vx - wz * self.y_w[i]
            v_iy = vy + wz * self.x_w[i]
            cd = ca.cos(deltas[i])
            sd = ca.sin(deltas[i])
            v_wx = v_ix * cd + v_iy * sd
            v_wy = -v_ix * sd + v_iy * cd
            v_wx_safe = smax(v_wx, 1.2, 0.3)
            # ISO slip angle: alpha = -atan(v_wy / v_wx) so positive delta -> positive alpha -> positive Fy
            alpha_kin_i = -ca.atan2(v_wy, v_wx_safe)
            alpha_kin_list.append(alpha_kin_i)
            v_wx_list.append(v_wx_safe)
            v_wy_list.append(v_wy)

        alpha_f_kin = 0.5 * (alpha_kin_list[0] + alpha_kin_list[1])
        alpha_r_kin = 0.5 * (alpha_kin_list[2] + alpha_kin_list[3])
        # Preserve left/right Ackermann slip differential while relaxing the axle mean
        d_alpha_f = alpha_f_state - alpha_f_kin
        d_alpha_r = alpha_r_state - alpha_r_kin
        alphas_eff = [
            alpha_kin_list[0] + d_alpha_f,
            alpha_kin_list[1] + d_alpha_f,
            alpha_kin_list[2] + d_alpha_r,
            alpha_kin_list[3] + d_alpha_r,
        ]

        # Relaxation length ODEs: d(alpha)/dt = (vx / sigma_y) * (alpha_kin - alpha_state)
        sig_y_safe = smax(sigma_y, 0.03, 0.005)
        alpha_f_dot = (vx_safe / sig_y_safe) * (alpha_f_kin - alpha_f_state)
        alpha_r_dot = (vx_safe / sig_y_safe) * (alpha_r_kin - alpha_r_state)

        # 7. Longitudinal force demand -> smooth wheel slip ratios kappa_i
        # Positive ax_cmd_drive -> RWD (RL, RR); Negative -> hydraulic braking split by brake_bias_front
        f_cmd_tot = p.mass * ax_cmd_drive
        f_drive_tot = spos(f_cmd_tot, 20.0)
        f_brake_tot = spos(-f_cmd_tot, 20.0)

        f_req = [
            -0.5 * p.brake_bias_front * f_brake_tot,
            -0.5 * p.brake_bias_front * f_brake_tot,
            0.5 * f_drive_tot - 0.5 * (1.0 - p.brake_bias_front) * f_brake_tot,
            0.5 * f_drive_tot - 0.5 * (1.0 - p.brake_bias_front) * f_brake_tot,
        ]

        # 8. Evaluate MF 6.2 at all 4 wheels
        fx_body_tot = 0.0
        fy_body_tot = 0.0
        mz_body_tot = 0.0
        fx_w_list = []
        fy_w_list = []
        mz_w_list = []
        kappa_list = []

        t = self.tire
        dpi = (p.p_infl_kpa - t.pinom) / t.pinom

        for i in range(4):
            mu_th_i = thermal_grip_multiplier_smooth(t_tire_4[i]) if self.use_thermal_coupling else 1.0
            fz_i = fz[i]
            fz_pos_i = spos(fz_i, s["eps_fz_n"])
            dfz_i = smin((fz_pos_i - t.fnomin) / t.fnomin, t.dfz_max, s["eps_dfz"]) if math.isfinite(t.dfz_max) else (fz_pos_i - t.fnomin) / t.fnomin
            g_eff_i = gammas[i] * camber_scale
            camber_fx_i = smax(1.0 - t.PDX3 * g_eff_i * g_eff_i, 0.25, s["eps_camber_f"])
            mu_x_i = (
                (t.PDX1 + t.PDX2 * dfz_i)
                * (1.0 + t.PPX3 * dpi + t.PPX4 * dpi * dpi)
                * camber_fx_i
                * (p.lambda_mux_base * mu_x_scale * mu_th_i)
            )
            dx_i = smax(mu_x_i * fz_pos_i, 80.0, 15.0)
            kxk_i = (
                fz_pos_i
                * (t.PKX1 + t.PKX2 * dfz_i)
                * ca.exp(t.PKX3 * dfz_i)
                * (1.0 + t.PPX1 * dpi + t.PPX2 * dpi * dpi)
            )
            cx_i = t.PCX1
            bx_i = kxk_i / (cx_i * dx_i + 1e-4)
            # Smooth analytic Pacejka inverse: Fx0(kappa) = Dx * sin(Cx * atan(Bx * kappa)) = f_req
            fx_ratio = sclip(f_req[i] / dx_i, -0.94, 0.94, 0.04)
            arg_tan = ca.asin(fx_ratio) / max(cx_i, 0.5)
            arg_tan_safe = sclip(arg_tan, -1.25, 1.25, 0.05)
            kappa_i = sclip(ca.tan(arg_tan_safe) / smax(bx_i, 1.0, 0.1) - (t.PHX1 + t.PHX2 * dfz_i), -0.22, 0.22, 0.01)

            fx_i, fy_i, mz_i = self.eval_tire_mf62_symbolic(
                kappa=kappa_i,
                alpha=alphas_eff[i],
                fz=fz_i,
                gamma=gammas[i],
                side=self.side_sign[i],
                mu_x_scale=mu_x_scale,
                mu_y_scale=mu_y_scale,
                kya_scale=kya_scale,
                camber_scale=camber_scale,
                mu_thermal=mu_th_i,
            )
            # Rolling resistance
            fx_i = fx_i - p.rolling_resistance_coeff * fz_i * ssign(v_wx_list[i], 0.3)

            cd = ca.cos(deltas[i])
            sd = ca.sin(deltas[i])
            fx_b = fx_i * cd - fy_i * sd
            fy_b = fx_i * sd + fy_i * cd

            fx_body_tot = fx_body_tot + fx_b
            fy_body_tot = fy_body_tot + fy_b
            mz_body_tot = mz_body_tot + (self.x_w[i] * fy_b - self.y_w[i] * fx_b) + mz_i

            fx_w_list.append(fx_i)
            fy_w_list.append(fy_i)
            mz_w_list.append(mz_i)
            kappa_list.append(kappa_i)

        # Effective chassis/suspension yaw damping term (scaled by c_damp_scale)
        # Models damper-induced transient yaw/roll energy dissipation
        c_yaw_eff = 18.0 * (c_damp_scale - 1.0)
        mz_body_tot = mz_body_tot - c_yaw_eff * wz

        fx_tot = fx_body_tot + f_drag_x
        fy_tot = fy_body_tot + f_drag_y

        ax_body = fx_tot / p.mass
        ay_body = fy_tot / p.mass

        # Planar state derivatives
        vx_dot = ax_body + wz * vy
        vy_dot = ay_body - wz * vx
        wz_dot = mz_body_tot / smax(i_zz, 20.0, 1.0)

        return {
            "vx_dot": vx_dot,
            "vy_dot": vy_dot,
            "wz_dot": wz_dot,
            "alpha_f_dot": alpha_f_dot,
            "alpha_r_dot": alpha_r_dot,
            "ax": ax_body,
            "ay": ay_body,
            "mz": mz_body_tot,
            "fz": fz,
            "phi": phi,
            "z_w": ca.vertcat(z_fl, z_fr, z_rl, z_rr),
            "deltas": ca.vertcat(delta_fl, delta_fr, delta_rl, delta_rr),
            "gammas": ca.vertcat(g_fl, g_fr, g_rl, g_rr),
            "fx_w": ca.vertcat(*fx_w_list),
            "fy_w": ca.vertcat(*fy_w_list),
            "kappa": ca.vertcat(*kappa_list),
            "alphas": ca.vertcat(*alphas_eff),
            "lltd_elastic": k_phi_f / smax(k_phi_tot, 1.0, 0.1),
        }

    def _build_casadi_functions(self) -> None:
        """Compile CasADi MX functions for fast RK4 simulation and IPOPT Multiple-Shooting."""
        # Note: ca.interpolant('bspline') requires MX graphs
        x = ca.MX.sym("x", 5)        # [vx, vy, wz, alpha_f, alpha_r]
        u = ca.MX.sym("u", 4)        # [delta_cmd, ax_cmd_drive, ax_lt, ay_lt]
        t4 = ca.MX.sym("t_tire", 4)  # [T_FL, T_FR, T_RL, T_RR] deg C
        th = ca.MX.sym("theta", len(SYSID_PARAM_NAMES))

        res = self.eval_dynamics_symbolic(
            vx=x[0], vy=x[1], wz=x[2],
            alpha_f_state=x[3], alpha_r_state=x[4],
            delta_cmd=u[0], ax_cmd_drive=u[1], ax_lt=u[2], ay_lt=u[3],
            t_tire_4=t4, theta=th,
        )

        xdot = ca.vertcat(
            res["vx_dot"], res["vy_dot"], res["wz_dot"],
            res["alpha_f_dot"], res["alpha_r_dot"],
        )
        y_obs = ca.vertcat(res["ax"], res["ay"], res["mz"], res["phi"])

        self.fn_ode = ca.Function(
            "ter26_ode",
            [x, u, t4, th],
            [xdot, y_obs],
            ["x", "u", "t_tire", "theta"],
            ["xdot", "y_obs"],
        )

        self.fn_full_diag = ca.Function(
            "ter26_full_diag",
            [x, u, t4, th],
            [
                xdot, res["ax"], res["ay"], res["mz"], res["phi"],
                res["fz"], res["z_w"], res["deltas"], res["gammas"],
                res["fx_w"], res["fy_w"], res["kappa"], res["alphas"],
                res["lltd_elastic"],
            ],
        )

        # Single RK4 step for slow chassis states [vx, vy, wz] + exact A-stable exponential step for [alpha_f, alpha_r]
        dt = ca.MX.sym("dt")
        # Evaluate kinematic slip angles at current state for exact exponential relaxation update
        def _step_alpha_exact(x_in: Any, u_in: Any, th_in: Any, dt_in: Any) -> Tuple[Any, Any]:
            vx_s = smax(x_in[0], 1.5, 0.3)
            sig_y = smax(th_in[10], 0.03, 0.005)
            decay = ca.exp(-dt_in * vx_s / sig_y)
            # Compute alpha_kin from current (vx, vy, wz, delta_cmd)
            v_f_y = x_in[1] + self.params.a_cog * x_in[2]
            v_r_y = x_in[1] - self.params.b_cog * x_in[2]
            af_kin = u_in[0] - ca.atan2(v_f_y, vx_s)
            ar_kin = -ca.atan2(v_r_y, vx_s)
            af_next = af_kin + (x_in[3] - af_kin) * decay
            ar_next = ar_kin + (x_in[4] - ar_kin) * decay
            return af_next, ar_next

        k1, y1 = self.fn_ode(x, u, t4, th)
        af_half, ar_half = _step_alpha_exact(x, u, th, 0.5 * dt)
        x2 = ca.vertcat(x[0:3] + 0.5 * dt * k1[0:3], af_half, ar_half)
        k2, _ = self.fn_ode(x2, u, t4, th)
        x3 = ca.vertcat(x[0:3] + 0.5 * dt * k2[0:3], af_half, ar_half)
        k3, _ = self.fn_ode(x3, u, t4, th)
        af_full, ar_full = _step_alpha_exact(x, u, th, dt)
        x4 = ca.vertcat(x[0:3] + dt * k3[0:3], af_full, ar_full)
        k4, _ = self.fn_ode(x4, u, t4, th)
        x_next = ca.vertcat(
            x[0:3] + (dt / 6.0) * (k1[0:3] + 2.0 * k2[0:3] + 2.0 * k3[0:3] + k4[0:3]),
            af_full,
            ar_full,
        )
        self.fn_rk4 = ca.Function(
            "ter26_rk4",
            [x, u, t4, th, dt],
            [x_next, y1],
            ["x", "u", "t_tire", "theta", "dt"],
            ["x_next", "y_obs"],
        )

    def map_pedals_to_ax_cmd(
        self,
        vx_ms: float,
        throttle: float,
        brake: float,
        trq_limit_nm: float = 120.0,
    ) -> float:
        """Map conditioned telemetry pedals (throttle, brake, trq_limit_nm) to net longitudinal cmd [m/s^2]."""
        p = self.params
        v_safe = max(vx_ms, 0.5)
        omega_w = v_safe / p.wheel_radius
        t_wheel_max_pwr = (p.power_limit_watts * p.drivetrain_efficiency) / max(omega_w, 1.0)
        t_wheel_cmd = min(2.0 * p.motor_peak_torque * p.gear_ratio * p.drivetrain_efficiency, t_wheel_max_pwr)
        trq_scale = min(max(trq_limit_nm / 120.0, 0.2), 1.5)
        f_drive = throttle * (t_wheel_cmd * trq_scale) / p.wheel_radius
        f_brake = brake * p.f_brk_max_n
        # Lift-off coasting resistance (0.15 Nm motor drag + drivetrain losses when off-throttle)
        f_coast = 140.0 if (throttle < 0.02 and brake < 0.02 and vx_ms > 1.5) else 0.0
        return (f_drive - f_brake - f_coast) / p.mass

    def map_pedals_to_ax_cmd_array(
        self,
        vx_ms: np.ndarray,
        throttle: np.ndarray,
        brake: np.ndarray,
        trq_limit_nm: np.ndarray,
    ) -> np.ndarray:
        """Vectorized pedal-to-longitudinal-command mapping [m/s^2]."""
        p = self.params
        v_safe = np.maximum(vx_ms, 0.5)
        omega_w = v_safe / p.wheel_radius
        t_wheel_max_pwr = (p.power_limit_watts * p.drivetrain_efficiency) / np.maximum(omega_w, 1.0)
        t_wheel_nom = 2.0 * p.motor_peak_torque * p.gear_ratio * p.drivetrain_efficiency
        t_wheel_cmd = np.minimum(t_wheel_nom, t_wheel_max_pwr)
        trq_scale = np.clip(trq_limit_nm / 120.0, 0.2, 1.5)
        f_drive = np.clip(throttle, 0.0, 1.0) * (t_wheel_cmd * trq_scale) / p.wheel_radius
        f_brake = np.clip(brake, 0.0, 1.0) * p.f_brk_max_n
        coast_mask = (throttle < 0.02) & (brake < 0.02) & (vx_ms > 1.5)
        f_coast = np.where(coast_mask, 140.0, 0.0)
        return (f_drive - f_brake - f_coast) / p.mass

    def simulate_trajectory(
        self,
        t_s: np.ndarray,
        delta_rad: np.ndarray,
        ax_cmd_drive: np.ndarray,
        vx_meas: np.ndarray,
        wz_meas: np.ndarray,
        t_tire_4: Optional[np.ndarray] = None,
        theta: Optional[Union[Dict[str, float], np.ndarray]] = None,
        horizon_reset_s: float = 0.0,
        horizon_observer_tau_s: float = 0.0,
        use_measured_vx: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Forward-simulate the TeR26 model along a telemetry time series using RK4.

        Args:
            t_s: Time vector [s]
            delta_rad: Mean front wheel steer angle [rad] (+ = left)
            ax_cmd_drive: Longitudinal drive/brake acceleration command [m/s^2]
            vx_meas: Measured longitudinal speed [m/s] (used for init & optional horizon reset)
            wz_meas: Measured yaw rate [rad/s] (used for init & optional horizon reset)
            t_tire_4: Optional (N, 4) tire temperatures [FL, FR, RL, RR] in deg C
            theta: Parameter dict or 11-vector (default nominal)
            horizon_reset_s: If > 0, hard-reset states to measured (vx, wz) every horizon_reset_s s
            horizon_observer_tau_s: If > 0, apply Godot-style exponential vx observer with time constant tau_s
            use_measured_vx: If True, hold vx to vx_meas (isolates lateral/yaw dynamics)
        """
        n = len(t_s)
        if theta is None:
            th_vec = theta_dict_to_vec(default_sysid_theta(self.params))
        elif isinstance(theta, dict):
            th_vec = theta_dict_to_vec(theta)
        else:
            th_vec = np.asarray(theta, dtype=np.float64)

        if t_tire_4 is None:
            t4_arr = np.full((n, 4), 55.0, dtype=np.float64)
        else:
            t4_arr = np.asarray(t_tire_4, dtype=np.float64)

        vx_sim = np.zeros(n, dtype=np.float64)
        vy_sim = np.zeros(n, dtype=np.float64)
        wz_sim = np.zeros(n, dtype=np.float64)
        ax_sim = np.zeros(n, dtype=np.float64)
        ay_sim = np.zeros(n, dtype=np.float64)
        phi_sim = np.zeros(n, dtype=np.float64)
        fz_sim = np.zeros((n, 4), dtype=np.float64)
        zw_sim = np.zeros((n, 4), dtype=np.float64)
        kappa_sim = np.zeros((n, 4), dtype=np.float64)
        fx_w_sim = np.zeros((n, 4), dtype=np.float64)
        fy_w_sim = np.zeros((n, 4), dtype=np.float64)
        alphas_sim = np.zeros((n, 4), dtype=np.float64)
        wheel_rpm_sim = np.zeros((n, 4), dtype=np.float64)

        x_curr = np.array([max(float(vx_meas[0]), 0.5), 0.0, float(wz_meas[0]), 0.0, 0.0], dtype=np.float64)
        ax_lt = 0.0
        ay_lt = 0.0
        last_reset_t = float(t_s[0])

        for k in range(n):
            if horizon_reset_s > 0.0 and (float(t_s[k]) - last_reset_t) >= horizon_reset_s:
                x_curr[0] = max(float(vx_meas[k]), 0.5)
                x_curr[1] = 0.0
                x_curr[2] = float(wz_meas[k])
                x_curr[3] = 0.0
                x_curr[4] = 0.0
                last_reset_t = float(t_s[k])
            if use_measured_vx:
                x_curr[0] = max(float(vx_meas[k]), 0.5)

            u_k = np.array([float(delta_rad[k]), float(ax_cmd_drive[k]), ax_lt, ay_lt], dtype=np.float64)
            t4_k = t4_arr[k]

            diag = self.fn_full_diag(x_curr, u_k, t4_k, th_vec)
            ax_k = float(diag[1])
            ay_k = float(diag[2])
            phi_k = float(diag[4])
            fz_k = np.asarray(diag[5], dtype=np.float64).ravel()
            zw_k = np.asarray(diag[6], dtype=np.float64).ravel()
            fx_wk = np.asarray(diag[9], dtype=np.float64).ravel()
            fy_wk = np.asarray(diag[10], dtype=np.float64).ravel()
            kap_k = np.asarray(diag[11], dtype=np.float64).ravel()
            alp_k = np.asarray(diag[12], dtype=np.float64).ravel()

            vx_sim[k] = x_curr[0]
            vy_sim[k] = x_curr[1]
            wz_sim[k] = x_curr[2]
            ax_sim[k] = ax_k
            ay_sim[k] = ay_k
            phi_sim[k] = phi_k
            fz_sim[k] = fz_k
            zw_sim[k] = zw_k
            fx_w_sim[k] = fx_wk
            fy_w_sim[k] = fy_wk
            kappa_sim[k] = kap_k
            alphas_sim[k] = alp_k

            for i in range(4):
                v_ix = max(x_curr[0] - x_curr[2] * self.y_w[i], 0.0)
                omega_i = v_ix * (1.0 + kap_k[i]) / self.params.wheel_radius
                wheel_rpm_sim[k, i] = omega_i * 60.0 / (2.0 * math.pi)

            ax_lt = 0.5 * ax_lt + 0.5 * float(np.clip(ax_k, -25.0, 25.0))
            ay_lt = 0.5 * ay_lt + 0.5 * float(np.clip(ay_k, -25.0, 25.0))

            if k + 1 < n:
                dt_k = float(t_s[k + 1] - t_s[k])
                if dt_k <= 0.0 or dt_k > 0.25:
                    x_curr = np.array([max(float(vx_meas[k + 1]), 0.5), 0.0, float(wz_meas[k + 1]), 0.0, 0.0], dtype=np.float64)
                    ax_lt = 0.0
                    ay_lt = 0.0
                    last_reset_t = float(t_s[k + 1])
                else:
                    x_next, _ = self.fn_rk4(x_curr, u_k, t4_k, th_vec, dt_k)
                    x_curr = np.asarray(x_next, dtype=np.float64).ravel()
                    if horizon_observer_tau_s > 0.0:
                        alpha_h = 1.0 - math.exp(-dt_k / horizon_observer_tau_s)
                        x_curr[0] += (max(float(vx_meas[k + 1]), 0.5) - x_curr[0]) * alpha_h
                    x_curr[0] = float(np.clip(x_curr[0], 0.5, 45.0))
                    x_curr[1] = float(np.clip(x_curr[1], -10.0, 10.0))
                    x_curr[2] = float(np.clip(x_curr[2], -4.0, 4.0))
                    x_curr[3] = float(np.clip(x_curr[3], -0.45, 0.45))
                    x_curr[4] = float(np.clip(x_curr[4], -0.45, 0.45))

        return {
            "vx_sim": vx_sim,
            "vy_sim": vy_sim,
            "wz_sim": wz_sim,
            "ax_sim": ax_sim,
            "ay_sim": ay_sim,
            "phi_sim": phi_sim,
            "fz_sim": fz_sim,
            "zw_sim": zw_sim,
            "fx_w_sim": fx_w_sim,
            "fy_w_sim": fy_w_sim,
            "kappa_sim": kappa_sim,
            "alphas_sim": alphas_sim,
            "wheel_rpm_sim": wheel_rpm_sim,
        }


def _self_test() -> None:
    print("[TeR26HighFidelityCasADi] Running self-test (3D kinematics + C^2 derivatives)...", flush=True)
    model = TeR26HighFidelityCasADi(use_3d_kinematics=True, use_thermal_coupling=True)
    th0 = theta_dict_to_vec(default_sysid_theta(model.params))
    x0 = np.array([15.0, -0.2, 0.8, 0.05, 0.04], dtype=np.float64)
    u0 = np.array([math.radians(6.0), 1.5, 1.5, 10.0], dtype=np.float64)
    t4 = np.array([58.0, 62.0, 60.0, 65.0], dtype=np.float64)

    xdot, yobs = model.fn_ode(x0, u0, t4, th0)
    print(f"  ODE eval: xdot={np.asarray(xdot).ravel().round(4)}, ax={float(yobs[0]):.3f}, ay={float(yobs[1]):.3f}, Mz={float(yobs[2]):.2f}, phi={math.degrees(float(yobs[3])):.3f} deg", flush=True)

    # Verify exact C^2 Jacobian & Hessian wrt theta
    x_s = ca.MX.sym("x", 5)
    u_s = ca.MX.sym("u", 4)
    t_s = ca.MX.sym("t4", 4)
    th_s = ca.MX.sym("th", len(SYSID_PARAM_NAMES))
    _, y_s = model.fn_ode(x_s, u_s, t_s, th_s)
    cost = y_s[0] ** 2 + y_s[1] ** 2 + 1e-4 * y_s[2] ** 2
    jac_fn = ca.Function("jac_th", [x_s, u_s, t_s, th_s], [ca.jacobian(cost, th_s)])
    hes_fn = ca.Function("hes_th", [x_s, u_s, t_s, th_s], [ca.hessian(cost, th_s)[0]])
    j_val = np.asarray(jac_fn(x0, u0, t4, th0), dtype=np.float64)
    h_val = np.asarray(hes_fn(x0, u0, t4, th0), dtype=np.float64)
    assert np.all(np.isfinite(j_val)), "Non-finite Jacobian!"
    assert np.all(np.isfinite(h_val)), "Non-finite Hessian!"
    print(f"  C^2 Jacobian & Hessian wrt 11 SysID params: PASS (||J||={np.linalg.norm(j_val):.3e}, ||H||={np.linalg.norm(h_val):.3e})", flush=True)

    # Quick 2-second sine-steer simulation test
    t_arr = np.arange(0.0, 2.0, 0.01)
    del_arr = np.radians(5.0) * np.sin(2.0 * np.pi * 0.5 * t_arr)
    ax_arr = np.full_like(t_arr, 1.0)
    vx_arr = np.full_like(t_arr, 14.0)
    wz_arr = np.zeros_like(t_arr)
    sim = model.simulate_trajectory(t_arr, del_arr, ax_arr, vx_arr, wz_arr)
    print(f"  2.0s sine-steer sim: max |ay|={np.max(np.abs(sim['ay_sim'])):.2f} m/s^2, max |wz|={np.max(np.abs(sim['wz_sim'])):.3f} rad/s, min Fz={np.min(sim['fz_sim']):.1f} N", flush=True)
    print("[TeR26HighFidelityCasADi] Self-test PASSED.", flush=True)


if __name__ == "__main__":
    _self_test()
