#!/usr/bin/env python3
"""
C2-smooth Differentiable 14-DOF CasADi SX Vehicle Dynamics Model (Option 3).

Supports both TeR26 (RWD) and TeR27 (4WD, [-25, +80] kW, I^2*R(T)/FW losses,
56.9% anti-dive, 41.2% anti-squat) with orthogonal suspension modes
(analytic_1d_lut, 3d_double_wishbone_lut, decoupled_heave_roll, standard_pushrod)
and embedded Hoosier 16.0x7.5-10 R20 / R25B MF 6.2 + PINN tires.

State vector x (NX = 32):
  0..5:   X, Y, Z, phi, theta_p, psi            (chassis 6-DOF pose)
  6..11:  vx, vy, vz, wx, wy, wz                (chassis 6-DOF body velocities)
  12..15: z_u_FL, z_u_FR, z_u_RL, z_u_RR        (4 unsprung vertical displacements [m])
  16..19: dz_u_FL, dz_u_FR, dz_u_RL, dz_u_RR    (4 unsprung vertical velocities [m/s])
  20..23: omega_FL, omega_FR, omega_RL, omega_RR (4 wheel angular velocities [rad/s])
  24..27: alpha_rel_FL..RR                       (4 lateral carcass relaxation states [rad])
  28..31: kappa_rel_FL..RR                       (4 longitudinal carcass relaxation states [-])

Control input vector u (NU = 9):
  0:      delta_cmd                              (mean front steer command [rad], + left)
  1..4:   T_wheel_FL..RR                         (motor drive/regen wheel torque [Nm])
  5..8:   T_brk_FL..RR                           (hydraulic brake wheel torque magnitude [Nm] >= 0)

Parameter vector theta (NTHETA = 28):
  See PARAM_NAMES below (mass, inertia, CG, springs, ARBs, dampers, RCs,
  anti-dive/anti-squat, camber, Ackermann, aero, grip, cornering stiffness,
  carcass relaxation lengths, steering compliance).

Disturbance vector w (NW = 10):
  0:      d_mu_scale                             (fractional grip perturbation, mu_eff = lambda_mu * (1 + d_mu_scale))
  1..3:   d_fx_ext, d_fy_ext, d_mz_ext           (external force/moment disturbances [N, N, Nm])
  4..7:   z_road_FL..RR                          (road vertical elevation under each wheel [m])
  8:      t_tire_c                               (bulk tire temperature [deg C])
  9:      t_motor_c                              (motor stator winding temperature [deg C])

Output vector y (NY = 36):
  See OUTPUT_NAMES below (ax, ay, az, Fz_4, Fx_4, Fy_4, Mz_4, alpha_4, kappa_4,
  delta_4, P_dc_kw, P_loss_kw, rh_f, rh_r, beta).

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
    smin,
    spos,
    thermal_grip_multiplier_smooth,
)
from vd_correlation_lab.models.differentiable_casadi.suspension_casadi import (
    SUSPENSION_MODES,
    SuspensionCasADi,
    SuspensionConfigData,
    load_suspension_config,
    normalize_suspension_mode,
)
from vd_correlation_lab.models.differentiable_casadi.tire_mf62_pinn_casadi import (
    COLD_TEST_OCT,
    FSG_RUBBERED,
    TireMF62PINNCasADi,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
VEHICLES_CONFIG_DIR = REPO_ROOT / "src" / "mirena_sim" / "Scenes" / "Vehicle" / "Config" / "vehicles"

# ---------------------------------------------------------------------------
# Canonical vector layouts (immutable public contract for downstream modules)
# ---------------------------------------------------------------------------

STATE_NAMES: Tuple[str, ...] = (
    "X", "Y", "Z", "phi", "theta_p", "psi",
    "vx", "vy", "vz", "wx", "wy", "wz",
    "z_u_FL", "z_u_FR", "z_u_RL", "z_u_RR",
    "dz_u_FL", "dz_u_FR", "dz_u_RL", "dz_u_RR",
    "omega_FL", "omega_FR", "omega_RL", "omega_RR",
    "alpha_rel_FL", "alpha_rel_FR", "alpha_rel_RL", "alpha_rel_RR",
    "kappa_rel_FL", "kappa_rel_FR", "kappa_rel_RL", "kappa_rel_RR",
)

INPUT_NAMES: Tuple[str, ...] = (
    "delta_cmd",
    "T_wheel_FL", "T_wheel_FR", "T_wheel_RL", "T_wheel_RR",
    "T_brk_FL", "T_brk_FR", "T_brk_RL", "T_brk_RR",
)

PARAM_NAMES: Tuple[str, ...] = (
    "mass",
    "i_xx",
    "i_yy",
    "i_zz",
    "h_cg",
    "front_weight_dist",
    "k_spring_f",
    "k_spring_r",
    "k_arb_f",
    "k_arb_r",
    "c_damp_f_scale",
    "c_damp_r_scale",
    "h_rc_f",
    "h_rc_r",
    "anti_dive_f",
    "anti_squat_r",
    "camber_f_rad",
    "camber_r_rad",
    "ackermann_factor",
    "cl_a",
    "cd_a",
    "aero_balance_front",
    "lambda_mux",
    "lambda_muy",
    "kya_scale",
    "sigma_y",
    "sigma_x",
    "k_steer_comp",
)

DISTURBANCE_NAMES: Tuple[str, ...] = (
    "d_mu_scale",
    "d_fx_ext",
    "d_fy_ext",
    "d_mz_ext",
    "z_road_FL",
    "z_road_FR",
    "z_road_RL",
    "z_road_RR",
    "t_tire_c",
    "t_motor_c",
)

OUTPUT_NAMES: Tuple[str, ...] = (
    "ax", "ay", "az",
    "Fz_FL", "Fz_FR", "Fz_RL", "Fz_RR",
    "Fx_FL", "Fx_FR", "Fx_RL", "Fx_RR",
    "Fy_FL", "Fy_FR", "Fy_RL", "Fy_RR",
    "Mz_FL", "Mz_FR", "Mz_RL", "Mz_RR",
    "alpha_FL", "alpha_FR", "alpha_RL", "alpha_RR",
    "kappa_FL", "kappa_FR", "kappa_RL", "kappa_RR",
    "delta_FL", "delta_FR", "delta_RL", "delta_RR",
    "P_dc_kw", "P_loss_kw", "rh_f", "rh_r", "beta",
)

NX: int = len(STATE_NAMES)          # 32
NU: int = len(INPUT_NAMES)          # 9
NTHETA: int = len(PARAM_NAMES)      # 28
NW: int = len(DISTURBANCE_NAMES)    # 10
NY: int = len(OUTPUT_NAMES)         # 36

STATE_IDX: Dict[str, int] = {k: i for i, k in enumerate(STATE_NAMES)}
INPUT_IDX: Dict[str, int] = {k: i for i, k in enumerate(INPUT_NAMES)}
PARAM_IDX: Dict[str, int] = {k: i for i, k in enumerate(PARAM_NAMES)}
DIST_IDX: Dict[str, int] = {k: i for i, k in enumerate(DISTURBANCE_NAMES)}
OUTPUT_IDX: Dict[str, int] = {k: i for i, k in enumerate(OUTPUT_NAMES)}

# Standardized grip profiles (COLD_TEST_OCT = 0.47 and FSG_RUBBERED = 0.57 preserved intact)
GRIP_PROFILE_CONFIGS: Dict[str, Dict[str, float]] = {
    "COLD_TEST_OCT": {
        "lambda_muy": COLD_TEST_OCT,
        "env_mu_scale": 1.0,
        "env_aero_cl_scale": 0.78,
        "t_tire_c": 35.0,
    },
    "FSG_RUBBERED": {
        "lambda_muy": FSG_RUBBERED,
        "env_mu_scale": round(FSG_RUBBERED / COLD_TEST_OCT, 6),
        "env_aero_cl_scale": 0.78,
        "t_tire_c": 75.0,
    },
    "TTC_LAB_PEAK": {
        "lambda_muy": 0.47,
        "env_mu_scale": 1.0,
        "env_aero_cl_scale": 1.0,
        "t_tire_c": 75.0,
    },
    "FSG_REAL_TRACK": {
        "lambda_muy": 0.47 * 0.86,
        "env_mu_scale": 0.86,
        "env_aero_cl_scale": 0.64,
        "t_tire_c": 45.0,
    },
}


@dataclass
class Vehicle14DofConfig:
    """Complete physical configuration for TeR26 (RWD) or TeR27 (4WD)."""

    vehicle_id: str
    drive_type: str  # "RWD" or "4WD"
    suspension_mode: str
    tire_compound: str
    use_pinn: bool

    # Mass & geometry
    mass: float
    sprung_mass: float
    unsprung_mass_front: float
    unsprung_mass_rear: float
    gravity: float
    wheelbase: float
    track_front: float
    track_rear: float
    h_cg: float
    h_sprung_cg: float
    front_weight_dist: float
    i_xx: float
    i_yy: float
    i_zz: float
    i_xz: float

    # Suspension
    susp: SuspensionConfigData
    h_rc_f: float
    h_rc_r: float
    ackermann_factor: float
    max_steer_angle: float

    # Aero
    air_density: float
    cl_a: float
    cd_a: float
    aero_balance_front: float
    static_rh_f: float
    static_rh_r: float
    stall_rh_f: float
    env_aero_cl_scale: float

    # Tires
    wheel_radius: float
    wheel_eff_inertia: float
    k_tire_z_f: float
    k_tire_z_r: float
    c_tire_z: float
    rolling_resistance_coeff: float
    slip_reg_speed: float
    p_infl_kpa: float
    lambda_mux: float
    lambda_muy: float
    kya_scale: float
    sigma_y: float
    sigma_x: float

    # Powertrain & In-Hub Electrical Losses
    gear_ratio: float
    motor_peak_torque: float
    motor_max_rpm: float
    base_speed_rpm: float
    v_dc_nom_v: float
    kt_nm_per_arms: float
    r_phase_25c_ohm: float
    alpha_cu_per_k: float
    v_ce0_v: float
    r_on_ohm: float
    f_sw_hz: float
    k_sw_j_per_a: float
    id_max_arms: float
    k_fw_gain: float
    c_iron_w_per_rads2: float
    eta_gear: float
    p_dc_max_kw: float
    p_dc_min_kw: float
    brake_bias_front: float


def load_vehicle_14dof_config(
    vehicle_id: str = "ter27_4wd",
    suspension_mode: str = "3d_double_wishbone_lut",
    tire_compound: Optional[str] = None,
    use_pinn: bool = True,
    grip_profile: Optional[str] = None,
) -> Vehicle14DofConfig:
    """Load complete vehicle configuration directly from MirenaSim JSON files."""
    vid = str(vehicle_id).strip().lower()
    if vid in ("ter26", "ter26_rwd", "rwd"):
        vid = "ter26_rwd"
    elif vid in ("ter27", "ter27_4wd", "4wd"):
        vid = "ter27_4wd"
    else:
        raise ValueError(f"Unknown vehicle_id '{vehicle_id}'. Expected 'ter26_rwd' or 'ter27_4wd'.")

    cfg_dir = VEHICLES_CONFIG_DIR / vid
    veh = json.loads((cfg_dir / "vehicle.json").read_text(encoding="utf-8"))
    aero = json.loads((cfg_dir / "aero.json").read_text(encoding="utf-8"))
    pt = json.loads((cfg_dir / "powertrain.json").read_text(encoding="utf-8"))
    tires = json.loads((cfg_dir / "tires.json").read_text(encoding="utf-8"))
    brakes = json.loads((cfg_dir / "brakes.json").read_text(encoding="utf-8"))
    susp_cfg = load_suspension_config(cfg_dir / "suspension.json", mode=suspension_mode)

    mass = float(veh["mass"])
    sprung_mass = float(veh["sprung_mass"])
    m_u_f = float(veh.get("unsprung_mass_front", veh.get("unsprung_mass", 7.75)))
    m_u_r = float(veh.get("unsprung_mass_rear", veh.get("unsprung_mass", 7.75)))
    L = float(veh["wheelbase"])
    tf = float(veh["track_front"])
    tr = float(veh["track_rear"])
    fwd = float(veh["front_weight_distribution"])
    a_cog = L * (1.0 - fwd)
    b_cog = L * fwd

    w_asm_i = float(pt.get("wheel_assembly_inertia", tires.get("wheel_inertia", 0.35)))
    G = float(pt["gear_ratio"])
    i_rot = float(pt.get("motor_rotor_inertia", 0.0012))
    i_eff = w_asm_i + G * G * i_rot

    izz_sprung = float(veh["inertia"]["izz"])
    if vid == "ter27_4wd":
        steiner_f = 2.0 * m_u_f * (a_cog * a_cog + (0.5 * tf) ** 2)
        steiner_r = 2.0 * m_u_r * (b_cog * b_cog + (0.5 * tr) ** 2)
        i_zz = izz_sprung + steiner_f + steiner_r + 4.0 * w_asm_i
        drive_type = "4WD"
        p_dc_min_kw = -float(brakes.get("p_max_charge_w", 25000.0)) * 1e-3
    else:
        i_zz = izz_sprung
        drive_type = "RWD"
        p_dc_min_kw = -float(brakes.get("p_max_charge_w", 22000.0)) * 1e-3

    compound = str(tire_compound or tires.get("tire_compound", "R20")).upper()
    lmux = float(tires.get("lambda_mux", 0.68))
    lmuy = float(tires.get("lambda_muy", COLD_TEST_OCT))
    env_aero_cl_scale = 1.0

    if grip_profile is not None:
        gp_key = str(grip_profile).upper()
        if gp_key not in GRIP_PROFILE_CONFIGS:
            raise ValueError(f"Unknown grip_profile '{grip_profile}'. Valid: {list(GRIP_PROFILE_CONFIGS)}")
        gp = GRIP_PROFILE_CONFIGS[gp_key]
        lmux = lmux * gp["env_mu_scale"]
        lmuy = lmuy * gp["env_mu_scale"]
        env_aero_cl_scale = gp["env_aero_cl_scale"]

    return Vehicle14DofConfig(
        vehicle_id=vid,
        drive_type=drive_type,
        suspension_mode=susp_cfg.mode,
        tire_compound=compound,
        use_pinn=bool(use_pinn),
        mass=mass,
        sprung_mass=sprung_mass,
        unsprung_mass_front=m_u_f,
        unsprung_mass_rear=m_u_r,
        gravity=float(veh.get("gravity", 9.81)),
        wheelbase=L,
        track_front=tf,
        track_rear=tr,
        h_cg=float(veh["cg_height"]),
        h_sprung_cg=float(veh.get("sprung_cg_height", veh["cg_height"])),
        front_weight_dist=fwd,
        i_xx=float(veh["inertia"]["ixx"]),
        i_yy=float(veh["inertia"]["iyy"]),
        i_zz=i_zz,
        i_xz=float(veh["inertia"].get("ixz", 0.0)),
        susp=susp_cfg,
        h_rc_f=float(veh.get("roll_center_height_front", 0.02)),
        h_rc_r=float(veh.get("roll_center_height_rear", 0.04)),
        ackermann_factor=float(veh.get("ackermann_factor", -0.11)),
        max_steer_angle=float(veh.get("max_steer_angle", 0.38)),
        air_density=float(aero.get("air_density", 1.225)),
        cl_a=float(aero.get("cl_a", 4.5)) * env_aero_cl_scale,
        cd_a=float(aero.get("cd_a", 1.35)),
        aero_balance_front=float(aero.get("aero_balance_front", 0.48)),
        static_rh_f=float(aero.get("static_ride_height_front", 0.035)),
        static_rh_r=float(aero.get("static_ride_height_rear", 0.035)),
        stall_rh_f=float(aero.get("stall_ride_height_front", 0.018)),
        env_aero_cl_scale=env_aero_cl_scale,
        wheel_radius=float(tires.get("wheel_radius", 0.2032)),
        wheel_eff_inertia=i_eff,
        k_tire_z_f=float(tires.get("tire_vertical_stiffness_front", tires.get("tire_vertical_stiffness", 115000.0))),
        k_tire_z_r=float(tires.get("tire_vertical_stiffness_rear", tires.get("tire_vertical_stiffness", 115000.0))),
        c_tire_z=250.0,
        rolling_resistance_coeff=float(tires.get("rolling_resistance_coeff", 0.015)),
        slip_reg_speed=float(tires.get("slip_regularization_speed", 0.5)),
        p_infl_kpa=83.0,
        lambda_mux=lmux,
        lambda_muy=lmuy,
        kya_scale=float(tires.get("lambda_kya", 1.0)),
        sigma_y=float(tires.get("tire_relaxation_sigma_y", 0.18)),
        sigma_x=0.12,
        gear_ratio=G,
        motor_peak_torque=float(pt.get("motor_peak_torque", 25.0)),
        motor_max_rpm=float(pt.get("motor_max_rpm", 18600.0)),
        base_speed_rpm=float(pt.get("base_speed_rpm", 9500.0)),
        v_dc_nom_v=float(pt.get("battery_nominal_voltage_v", 540.0)),
        kt_nm_per_arms=0.26,
        r_phase_25c_ohm=0.135,
        alpha_cu_per_k=0.00393,
        v_ce0_v=0.85,
        r_on_ohm=0.012,
        f_sw_hz=16000.0,
        k_sw_j_per_a=1.85e-4,
        id_max_arms=55.0,
        k_fw_gain=1.0,
        c_iron_w_per_rads2=1.2e-4,
        eta_gear=0.965,
        p_dc_max_kw=float(pt.get("max_dc_power_w", pt.get("power_limit_watts", 80000.0))) * 1e-3,
        p_dc_min_kw=p_dc_min_kw,
        brake_bias_front=float(brakes.get("brake_bias_front", 0.58)),
    )


class DifferentiableVehicle14Dof:
    """
    C2-smooth 14-DOF CasADi SX vehicle dynamics model with RK4 integrator,
    exact AD Jacobians (A_x, B_u, S_theta, G_w), and C-Codegen support.
    """

    def __init__(
        self,
        vehicle_id: str = "ter27_4wd",
        suspension_mode: str = "3d_double_wishbone_lut",
        tire_compound: Optional[str] = None,
        use_pinn: bool = True,
        grip_profile: Optional[str] = None,
        dt: float = 0.005,
        rk4_substeps: int = 2,
        use_thermal_coupling: bool = False,
    ) -> None:
        self.cfg = load_vehicle_14dof_config(
            vehicle_id=vehicle_id,
            suspension_mode=suspension_mode,
            tire_compound=tire_compound,
            use_pinn=use_pinn,
            grip_profile=grip_profile,
        )
        self.dt = float(dt)
        self.rk4_substeps = max(int(rk4_substeps), 1)
        self.use_thermal_coupling = bool(use_thermal_coupling)

        self.tire = TireMF62PINNCasADi(
            compound=self.cfg.tire_compound,
            use_pinn=self.cfg.use_pinn,
        )
        self.susp = SuspensionCasADi(
            cfg=self.cfg.susp,
            wheelbase=self.cfg.wheelbase,
            track_front=self.cfg.track_front,
        )

        # Build symbolic CasADi SX functions
        self._build_casadi_functions()

    # ------------------------------------------------------------------
    # Default state / input / parameter / disturbance builders
    # ------------------------------------------------------------------

    def default_theta(self) -> np.ndarray:
        """Return nominal 28-element parameter vector theta for this vehicle."""
        c = self.cfg
        return np.array([
            c.mass,
            c.i_xx,
            c.i_yy,
            c.i_zz,
            c.h_cg,
            c.front_weight_dist,
            c.susp.spring_stiffness_front,
            c.susp.spring_stiffness_rear,
            c.susp.arb_stiffness_front,
            c.susp.arb_stiffness_rear,
            1.0,  # c_damp_f_scale
            1.0,  # c_damp_r_scale
            c.h_rc_f,
            c.h_rc_r,
            c.susp.anti_dive_ratio_front,
            c.susp.anti_squat_ratio_rear,
            c.susp.static_camber_front_rad,
            c.susp.static_camber_rear_rad,
            c.ackermann_factor,
            c.cl_a,
            c.cd_a,
            c.aero_balance_front,
            c.lambda_mux,
            c.lambda_muy,
            c.kya_scale,
            c.sigma_y,
            c.sigma_x,
            c.susp.steering_compliance_nm_rad,
        ], dtype=np.float64)

    def theta_dict(self, theta: Optional[np.ndarray] = None) -> Dict[str, float]:
        arr = self.default_theta() if theta is None else np.asarray(theta, dtype=np.float64)
        return {k: float(arr[i]) for i, k in enumerate(PARAM_NAMES)}

    def default_state(self, vx: float = 0.0, vy: float = 0.0, wz: float = 0.0) -> np.ndarray:
        """Return equilibrium initial 32-element state vector x at given velocity."""
        x = np.zeros(NX, dtype=np.float64)
        x[STATE_IDX["Z"]] = self.cfg.h_cg
        x[STATE_IDX["vx"]] = float(vx)
        x[STATE_IDX["vy"]] = float(vy)
        x[STATE_IDX["wz"]] = float(wz)
        # Initialize wheel angular velocities to free-rolling speed
        om0 = float(vx) / max(self.cfg.wheel_radius, 0.05)
        x[20:24] = om0
        return x

    def default_input(self, delta_cmd: float = 0.0) -> np.ndarray:
        u = np.zeros(NU, dtype=np.float64)
        u[0] = float(delta_cmd)
        return u

    def default_disturbance(self, t_tire_c: float = 75.0, t_motor_c: float = 55.0) -> np.ndarray:
        w = np.zeros(NW, dtype=np.float64)
        w[DIST_IDX["t_tire_c"]] = float(t_tire_c)
        w[DIST_IDX["t_motor_c"]] = float(t_motor_c)
        return w

    def inputs_from_driver(
        self,
        delta_cmd: float,
        throttle: float = 0.0,
        brake: float = 0.0,
        max_brake_torque_total_nm: float = 1200.0,
    ) -> np.ndarray:
        """Convert normalized driver commands (delta [rad], throttle [0..1], brake [0..1]) to u[9]."""
        c = self.cfg
        th = float(np.clip(throttle, 0.0, 1.0))
        br = float(np.clip(brake, 0.0, 1.0))
        t_max_wheel = c.motor_peak_torque * c.gear_ratio
        u = np.zeros(NU, dtype=np.float64)
        u[0] = float(delta_cmd)
        if c.drive_type == "RWD":
            u[1:3] = 0.0
            u[3:5] = th * t_max_wheel
        else:
            u[1:3] = th * t_max_wheel * 0.7
            u[3:5] = th * t_max_wheel
        bf = c.brake_bias_front
        u[5:7] = 0.5 * bf * br * max_brake_torque_total_nm
        u[7:9] = 0.5 * (1.0 - bf) * br * max_brake_torque_total_nm
        return u

    # ------------------------------------------------------------------
    # Symbolic 14-DOF Physics ODE
    # ------------------------------------------------------------------

    def eval_powertrain_losses_symbolic(
        self,
        t_wheel_4: Any,
        omega_4: Any,
        t_motor_c: Any,
    ) -> Tuple[Any, Any]:
        """
        Smooth C2 in-hub PMSM copper I^2*R(T) + inverter conduction/switching +
        field-weakening Id + iron/gearbox loss map matching alqp_power_abs_pitch.py.
        Returns (P_dc_tot_kw, P_loss_tot_kw).
        """
        c = self.cfg
        inv_ng_kt = 1.0 / (c.gear_ratio * c.kt_nm_per_arms)
        a1_inv = 3.0 * c.v_ce0_v + c.k_sw_j_per_a * c.f_sw_hz
        c0_gear = 1.0 - c.eta_gear
        om_base = c.base_speed_rpm * (2.0 * math.pi / 60.0)

        t_clamped = sclip(t_motor_c, -20.0, 180.0, 0.1)
        r_ph = c.r_phase_25c_ohm * (1.0 + c.alpha_cu_per_k * (t_clamped - 25.0))

        p_dc_tot = 0.0
        p_loss_tot = 0.0
        start_idx = 2 if c.drive_type == "RWD" else 0
        for i in range(start_idx, 4):
            tw = t_wheel_4[i]
            om_w = omega_4[i]
            om_m = sabs(om_w, 1e-2) * c.gear_ratio
            iq = sabs(tw, 1e-3) * inv_ng_kt
            fw_frac = spos(1.0 - om_base / smax(om_m, 1.0, 0.2), 1e-4)
            id_fw = c.k_fw_gain * fw_frac * c.id_max_arms
            i_rms = ca.sqrt(iq * iq + id_fw * id_fw + 1e-6)

            p_cu = 3.0 * (i_rms * i_rms) * r_ph
            p_inv = a1_inv * i_rms + 3.0 * c.r_on_ohm * (i_rms * i_rms)
            p_mech = tw * om_w
            p_iron_gear = c.c_iron_w_per_rads2 * (om_m * om_m) + c0_gear * sabs(p_mech, 1.0)
            p_loss_i = p_cu + p_inv + p_iron_gear
            p_dc_i = p_mech + p_loss_i

            p_loss_tot = p_loss_tot + p_loss_i
            p_dc_tot = p_dc_tot + p_dc_i

        return p_dc_tot * 1e-3, p_loss_tot * 1e-3

    def eval_ode_symbolic(
        self,
        x: ca.SX,
        u: ca.SX,
        theta: ca.SX,
        w: ca.SX,
    ) -> Tuple[ca.SX, ca.SX]:
        """
        Evaluate continuous-time 14-DOF state derivative x_dot (32,) and outputs y (36,).
        All expressions are strictly C-infinity smooth in CasADi SX.
        """
        c = self.cfg

        # Unpack states
        X_pos = x[0]
        Y_pos = x[1]
        Z_pos = x[2]
        phi = x[3]
        theta_p = x[4]
        psi = x[5]
        vx = x[6]
        vy = x[7]
        vz = x[8]
        wx = x[9]
        wy = x[10]
        wz = x[11]
        z_u = x[12:16]
        dz_u = x[16:20]
        omega = x[20:24]
        alpha_rel = x[24:28]
        kappa_rel = x[28:32]

        # Unpack controls
        delta_cmd = u[0]
        t_motor_raw = u[1:5]
        t_brk_mag = u[5:9]

        # Enforce RWD mask if vehicle is RWD
        if c.drive_type == "RWD":
            t_motor = ca.vertcat(0.0, 0.0, t_motor_raw[2], t_motor_raw[3])
        else:
            t_motor = t_motor_raw

        # Unpack parameters
        mass = theta[0]
        i_xx = theta[1]
        i_yy = theta[2]
        i_zz = theta[3]
        h_cg = theta[4]
        fwd = sclip(theta[5], 0.25, 0.75, 1e-4)
        k_spring_f = theta[6]
        k_spring_r = theta[7]
        k_arb_f = theta[8]
        k_arb_r = theta[9]
        c_damp_f_scale = theta[10]
        c_damp_r_scale = theta[11]
        h_rc_f = theta[12]
        h_rc_r = theta[13]
        anti_dive_f = theta[14]
        anti_squat_r = theta[15]
        camber_f_rad = theta[16]
        camber_r_rad = theta[17]
        ackermann_factor = theta[18]
        cl_a = theta[19]
        cd_a = theta[20]
        aero_bal_f = sclip(theta[21], 0.1, 0.9, 1e-4)
        lambda_mux = theta[22]
        lambda_muy = theta[23]
        kya_scale = theta[24]
        sigma_y = theta[25]
        sigma_x = theta[26]
        k_steer_comp = theta[27]

        # Unpack disturbances
        d_mu_scale = w[0]
        d_fx_ext = w[1]
        d_fy_ext = w[2]
        d_mz_ext = w[3]
        z_road = w[4:8]
        t_tire_c = w[8]
        t_motor_c = w[9]

        # Derived mass & geometry
        L = c.wheelbase
        tf = c.track_front
        tr = c.track_rear
        a_cog = L * (1.0 - fwd)
        b_cog = L * fwd
        m_u_list = [c.unsprung_mass_front, c.unsprung_mass_front, c.unsprung_mass_rear, c.unsprung_mass_rear]
        m_u_tot = 2.0 * (c.unsprung_mass_front + c.unsprung_mass_rear)
        m_s = smax(mass - m_u_tot, 50.0, 0.5)
        h_s = (mass * h_cg - m_u_tot * c.wheel_radius) / m_s
        x_w = [a_cog, a_cog, -b_cog, -b_cog]
        y_w = [0.5 * tf, -0.5 * tf, 0.5 * tr, -0.5 * tr]
        side_sign = [1.0, -1.0, 1.0, -1.0]
        k_tz_list = [c.k_tire_z_f, c.k_tire_z_f, c.k_tire_z_r, c.k_tire_z_r]

        # 1. Corner chassis vertical displacements & velocities relative to static design height
        z_cg_rel = Z_pos - h_cg
        cp = ca.cos(theta_p)
        sp = ca.sin(theta_p)
        sr = ca.sin(phi)
        cr = ca.cos(phi)

        dz_susp_list = []
        dz_susp_dot_list = []
        for i in range(4):
            z_c_i = z_cg_rel + y_w[i] * cp * sr - x_w[i] * sp
            z_c_dot_i = vz + wx * y_w[i] - wy * x_w[i]
            dz_susp_list.append(z_u[i] - z_c_i)
            dz_susp_dot_list.append(dz_u[i] - z_c_dot_i)
        dz_susp = ca.vertcat(*dz_susp_list)
        dz_susp_dot = ca.vertcat(*dz_susp_dot_list)

        # Dynamic ride heights (front & rear axle center)
        rh_f = c.static_rh_f + z_cg_rel - a_cog * sp
        rh_r = c.static_rh_r + z_cg_rel + b_cog * sp

        # 2. Aerodynamics (with smooth front-diffuser stall modulation near stall_rh_f)
        v_sq = vx * vx + vy * vy
        speed = ca.sqrt(v_sq + 1e-4)
        q_dyn = 0.5 * c.air_density * v_sq
        stall_ratio = sclip((rh_f - 0.5 * c.stall_rh_f) / max(0.5 * c.stall_rh_f, 1e-3), 0.35, 1.0, 1e-3)
        f_drag_x = -0.5 * c.air_density * cd_a * speed * vx
        f_drag_y = -0.5 * c.air_density * cd_a * speed * vy
        f_down_tot = q_dyn * cl_a * stall_ratio
        f_down_f = f_down_tot * aero_bal_f
        f_down_r = f_down_tot * (1.0 - aero_bal_f)

        # 3. Suspension kinematics & vertical forces
        kin = self.susp.eval_kinematics_symbolic(
            dz_4=dz_susp,
            delta_cmd=delta_cmd,
            phi=phi,
            camber_f_rad=camber_f_rad,
            camber_r_rad=camber_r_rad,
            ackermann_factor=ackermann_factor,
        )
        deltas_raw = kin["deltas"]
        gammas = kin["gammas"]
        mrs = kin["mrs"]

        susp_f = self.susp.eval_vertical_forces_symbolic(
            dz_4=dz_susp,
            dz_dot_4=dz_susp_dot,
            mrs_4=mrs,
            k_spring_f=k_spring_f,
            k_spring_r=k_spring_r,
            k_arb_f=k_arb_f,
            k_arb_r=k_arb_r,
            c_damp_f_scale=c_damp_f_scale,
            c_damp_r_scale=c_damp_r_scale,
        )
        f_susp_dyn = susp_f["f_susp"]

        # Static preloads (exact equilibrium at Z=h_cg, z_u=0)
        f_sp_stat_f = 0.5 * m_s * c.gravity * (b_cog / L)
        f_sp_stat_r = 0.5 * m_s * c.gravity * (a_cog / L)
        f_sp_stat = [f_sp_stat_f, f_sp_stat_f, f_sp_stat_r, f_sp_stat_r]
        f_susp_tot = [f_sp_stat[i] + f_susp_dyn[i] for i in range(4)]

        # Tire vertical normal forces (elastic tire sidewall + static preload)
        fz_tire_base = []
        for i in range(4):
            fz_stat_i = f_sp_stat[i] + m_u_list[i] * c.gravity
            fz_raw_i = fz_stat_i + k_tz_list[i] * (z_road[i] - z_u[i]) - c.c_tire_z * dz_u[i]
            fz_tire_base.append(spos(fz_raw_i, 0.5))

        # 4. Steering compliance (smooth algebraic kingpin compliance on front wheels)
        # Approximate using relaxed lateral slip stiffness for algebraic smoothness
        mu_th = thermal_grip_multiplier_smooth(t_tire_c) if self.use_thermal_coupling else 1.0
        mu_pert = smax(1.0 + d_mu_scale, 0.15, 1e-3)
        lmux_eff = lambda_mux * mu_pert * mu_th
        lmuy_eff = lambda_muy * mu_pert * mu_th

        # 5. Wheel hub kinematics, slip angles, slip ratios, and MF6.2 + PINN forces
        fx_w_list = []
        fy_w_list = []
        mz_w_list = []
        fx_c_list = []
        fy_c_list = []
        alpha_kin_list = []
        kappa_kin_list = []
        r_load_list = []
        delta_eff_list = []
        u_den_list = []

        # First pass: wheel kinematics & tire evaluation at base Fz
        for i in range(4):
            delta_i = deltas_raw[i]
            cd = ca.cos(delta_i)
            sd = ca.sin(delta_i)
            v_hub_x = vx - wz * y_w[i]
            v_hub_y = vy + wz * x_w[i]
            v_wx = v_hub_x * cd + v_hub_y * sd
            v_wy = -v_hub_x * sd + v_hub_y * cd
            u_den = smax(sabs(v_wx, 1e-3), c.slip_reg_speed, 1e-3)

            alpha_kin_i = -ca.atan2(v_wy, u_den)
            rho_rad_i = smin(fz_tire_base[i] / k_tz_list[i], 0.45 * c.wheel_radius, 1e-4)
            r_load_i = c.wheel_radius - rho_rad_i
            r_eff_i = c.wheel_radius - rho_rad_i / 3.0
            kappa_kin_i = sclip((omega[i] * r_eff_i - v_wx) / u_den, -1.0, 1.0, 1e-3)

            # Evaluate MF 6.2 + PINN at relaxed carcass slips
            fx_i, fy_i, mz_i, _, _ = self.tire.eval_combined_symbolic(
                kappa=kappa_rel[i],
                alpha=alpha_rel[i],
                fz=fz_tire_base[i],
                gamma=gammas[i],
                side=side_sign[i],
                lmux=lmux_eff,
                lmuy=lmuy_eff,
                p_kpa=c.p_infl_kpa,
                kya_scale=kya_scale,
                camber_scale=1.0,
                pinn_scale=1.0,
            )

            # Front steering compliance adjustment (elastic understeer)
            if i < 2:
                tau_kp = mz_i - fy_i * c.susp.mechanical_trail_front
                d_comp = sclip(tau_kp / smax(k_steer_comp, 500.0, 10.0), -0.08, 0.08, 1e-4)
                delta_eff_i = delta_i + d_comp
                # Re-project slip angle target with compliant steer
                cd_e = ca.cos(delta_eff_i)
                sd_e = ca.sin(delta_eff_i)
                v_wx_e = v_hub_x * cd_e + v_hub_y * sd_e
                v_wy_e = -v_hub_x * sd_e + v_hub_y * cd_e
                u_den_e = smax(sabs(v_wx_e, 1e-3), c.slip_reg_speed, 1e-3)
                alpha_kin_i = -ca.atan2(v_wy_e, u_den_e)
                cd, sd, u_den = cd_e, sd_e, u_den_e
            else:
                delta_eff_i = delta_i

            # Rolling resistance
            f_rr_i = c.rolling_resistance_coeff * fz_tire_base[i] * ca.tanh(v_wx / 0.5)
            fx_net_w = fx_i - f_rr_i

            fx_c_i = fx_net_w * cd - fy_i * sd
            fy_c_i = fx_net_w * sd + fy_i * cd

            fx_w_list.append(fx_i)
            fy_w_list.append(fy_i)
            mz_w_list.append(mz_i)
            fx_c_list.append(fx_c_i)
            fy_c_list.append(fy_c_i)
            alpha_kin_list.append(alpha_kin_i)
            kappa_kin_list.append(kappa_kin_i)
            r_load_list.append(r_load_i)
            delta_eff_list.append(delta_eff_i)
            u_den_list.append(u_den)

        # 6. Geometric non-elastic load transfer (roll-center jacking + anti-dive/anti-squat)
        sprung_ratio = m_s / smax(mass, 1.0, 0.1)
        fy_f_tot = fy_c_list[0] + fy_c_list[1]
        fy_r_tot = fy_c_list[2] + fy_c_list[3]
        fx_f_tot = fx_c_list[0] + fx_c_list[1]
        fx_r_tot = fx_c_list[2] + fx_c_list[3]

        dfz_rc_f = (sprung_ratio * fy_f_tot * h_rc_f) / tf
        dfz_rc_r = (sprung_ratio * fy_r_tot * h_rc_r) / tr
        dfz_anti = 0.5 * (h_cg / L) * (anti_dive_f * fx_f_tot + anti_squat_r * fx_r_tot)

        dfz_geom = [
            -dfz_rc_f - dfz_anti,
            +dfz_rc_f - dfz_anti,
            -dfz_rc_r + dfz_anti,
            +dfz_rc_r + dfz_anti,
        ]
        fz_eff = [spos(fz_tire_base[i] + dfz_geom[i], 0.5) for i in range(4)]

        # 7. Net forces & moments on 6-DOF chassis (ISO 8855 body frame)
        Fx_body = sum(fx_c_list) + f_drag_x + d_fx_ext
        Fy_body = sum(fy_c_list) + f_drag_y + d_fy_ext
        Fz_body = sum(f_susp_tot) - f_down_f - f_down_r - m_s * c.gravity

        # Roll moment Mx around +x:
        #   Restoring suspension moment: sum(y_i * F_susp_tot_i)
        #   Lateral overturning moment above roll centers: fy_f*(h_s - h_rc_f) + fy_r*(h_s - h_rc_r)
        Mx_susp = sum(y_w[i] * f_susp_tot[i] for i in range(4))
        Mx_lat = fy_f_tot * smax(h_s - h_rc_f, 0.0, 1e-4) + fy_r_tot * smax(h_s - h_rc_r, 0.0, 1e-4)
        Mx_body = Mx_susp + Mx_lat

        # Pitch moment My around +y:
        #   Restoring suspension moment: -sum(x_i * F_susp_tot_i)
        #   Aero pitch moment: a_cog * f_down_f - b_cog * f_down_r (front downforce pitches nose down -> +My)
        #   Longitudinal moment reduced by anti-dive (56.9% TeR27) & anti-squat (41.2% TeR27)
        My_susp = -sum(x_w[i] * f_susp_tot[i] for i in range(4))
        My_aero = a_cog * f_down_f - b_cog * f_down_r
        My_long = -h_cg * ((1.0 - anti_dive_f) * fx_f_tot + (1.0 - anti_squat_r) * fx_r_tot)
        My_body = My_susp + My_aero + My_long

        # Yaw moment Mz around +z:
        Mz_body = sum(x_w[i] * fy_c_list[i] - y_w[i] * fx_c_list[i] + mz_w_list[i] for i in range(4)) + d_mz_ext

        # 8. Newton-Euler 6-DOF chassis derivatives
        ax_body = Fx_body / mass
        ay_body = Fy_body / mass
        az_body = Fz_body / m_s

        # Body-frame linear velocity derivatives (including gravity projection on roll/pitch)
        vx_dot = ax_body + wz * vy - wy * vz + c.gravity * sp
        vy_dot = ay_body - wz * vx + wx * vz - c.gravity * cp * sr
        vz_dot = az_body + wy * vx - wx * vy

        # Rotational Euler equations with Ixz coupling
        rhs_x = Mx_body - (i_zz - i_yy) * wy * wz
        rhs_y = My_body - (i_xx - i_zz) * wz * wx
        rhs_z = Mz_body - (i_yy - i_xx) * wx * wy
        det_xz = smax(i_xx * i_zz - c.i_xz * c.i_xz, 1.0, 1e-2)
        wx_dot = (i_zz * rhs_x + c.i_xz * rhs_z) / det_xz
        wy_dot = rhs_y / smax(i_yy, 1.0, 1e-2)
        wz_dot = (c.i_xz * rhs_x + i_xx * rhs_z) / det_xz

        # Inertial kinematics (Z-Y-X Euler)
        cy = ca.cos(psi)
        sy = ca.sin(psi)
        X_dot = (cy * cp) * vx + (cy * sp * sr - sy * cr) * vy + (cy * sp * cr + sy * sr) * vz
        Y_dot = (sy * cp) * vx + (sy * sp * sr + cy * cr) * vy + (sy * sp * cr - cy * sr) * vz
        Z_dot = -sp * vx + (cp * sr) * vy + (cp * cr) * vz

        cp_safe = smax(cp, 0.2, 1e-3)
        tp = sp / cp_safe
        q_sr_r_cr = wy * sr + wz * cr
        phi_dot = wx + q_sr_r_cr * tp
        theta_p_dot = wy * cr - wz * sr
        psi_dot = q_sr_r_cr / cp_safe

        # 9. 4 Unsprung mass vertical ODEs
        z_u_dot = dz_u
        dz_u_dot_list = []
        for i in range(4):
            dz_u_dot_i = (fz_eff[i] - f_susp_tot[i] - m_u_list[i] * c.gravity) / m_u_list[i]
            dz_u_dot_list.append(dz_u_dot_i)
        dz_u_dot = ca.vertcat(*dz_u_dot_list)

        # 10. 4 Wheel spin ODEs
        omega_dot_list = []
        for i in range(4):
            om_tanh = ca.tanh(omega[i] / 0.5)
            t_brk_signed = -spos(t_brk_mag[i], 1e-3) * om_tanh
            t_rr = c.rolling_resistance_coeff * fz_eff[i] * r_load_list[i] * om_tanh
            om_dot_i = (t_motor[i] + t_brk_signed - fx_w_list[i] * r_load_list[i] - t_rr) / c.wheel_eff_inertia
            omega_dot_list.append(om_dot_i)
        omega_dot = ca.vertcat(*omega_dot_list)

        # 11. Carcass relaxation ODEs
        sig_y_safe = smax(sigma_y, 0.02, 1e-3)
        sig_x_safe = smax(sigma_x, 0.02, 1e-3)
        alpha_rel_dot = ca.vertcat(*[
            (u_den_list[i] / sig_y_safe) * (alpha_kin_list[i] - alpha_rel[i]) for i in range(4)
        ])
        kappa_rel_dot = ca.vertcat(*[
            (u_den_list[i] / sig_x_safe) * (kappa_kin_list[i] - kappa_rel[i]) for i in range(4)
        ])

        x_dot = ca.vertcat(
            X_dot, Y_dot, Z_dot, phi_dot, theta_p_dot, psi_dot,
            vx_dot, vy_dot, vz_dot, wx_dot, wy_dot, wz_dot,
            z_u_dot, dz_u_dot, omega_dot, alpha_rel_dot, kappa_rel_dot,
        )

        # 12. Powertrain HV DC power & losses
        p_dc_kw, p_loss_kw = self.eval_powertrain_losses_symbolic(t_motor, omega, t_motor_c)

        # 13. Outputs vector y (36,)
        vx_safe = smax(vx, 0.5, 0.05)
        beta = ca.atan2(vy, vx_safe)
        az_imu = az_body + c.gravity * cp * cr

        outputs = ca.vertcat(
            ax_body, ay_body, az_imu,
            *fz_eff,
            *fx_w_list,
            *fy_w_list,
            *mz_w_list,
            *[alpha_rel[i] for i in range(4)],
            *[kappa_rel[i] for i in range(4)],
            *delta_eff_list,
            p_dc_kw, p_loss_kw, rh_f, rh_r, beta,
        )
        return x_dot, outputs

    # ------------------------------------------------------------------
    # Fixed-step RK4 integrator & Jacobian builder
    # ------------------------------------------------------------------

    def _build_casadi_functions(self) -> None:
        """Build CasADi SX functions for ODE, RK4 step, and exact AD Jacobians."""
        x = ca.SX.sym("x", NX)
        u = ca.SX.sym("u", NU)
        theta = ca.SX.sym("theta", NTHETA)
        w = ca.SX.sym("w", NW)
        dt_sym = ca.SX.sym("dt")

        x_dot, outputs_0 = self.eval_ode_symbolic(x, u, theta, w)
        self.fn_ode = ca.Function(
            "f_ode",
            [x, u, theta, w],
            [x_dot, outputs_0],
            ["x", "u", "theta", "w"],
            ["x_dot", "outputs"],
        )

        # Build RK4 step with rk4_substeps for variable dt_sym
        h = dt_sym / float(self.rk4_substeps)
        x_cur = x
        y_cur = outputs_0
        for _ in range(self.rk4_substeps):
            k1, y_cur = self.fn_ode(x_cur, u, theta, w)
            k2, _ = self.fn_ode(x_cur + 0.5 * h * k1, u, theta, w)
            k3, _ = self.fn_ode(x_cur + 0.5 * h * k2, u, theta, w)
            k4, _ = self.fn_ode(x_cur + h * k3, u, theta, w)
            x_cur = x_cur + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        _, y_next = self.fn_ode(x_cur, u, theta, w)

        self.fn_step_dt = ca.Function(
            "step_dynamics_dt",
            [x, u, theta, w, dt_sym],
            [x_cur, y_next],
            ["x", "u", "theta", "w", "dt"],
            ["x_next", "outputs"],
        )

        # Fixed-dt versions (exact signature: step_dynamics(x, u, theta, w))
        x_next_fixed, y_next_fixed = self.fn_step_dt(x, u, theta, w, self.dt)
        self.fn_step = ca.Function(
            "step_dynamics",
            [x, u, theta, w],
            [x_next_fixed, y_next_fixed],
            ["x", "u", "theta", "w"],
            ["x_next", "outputs"],
        )

        # Exact AD Jacobians for fixed-dt step
        A_x = ca.jacobian(x_next_fixed, x)
        B_u = ca.jacobian(x_next_fixed, u)
        S_theta = ca.jacobian(x_next_fixed, theta)
        G_w = ca.jacobian(x_next_fixed, w)
        self.fn_jacobians = ca.Function(
            "step_jacobians",
            [x, u, theta, w],
            [A_x, B_u, S_theta, G_w],
            ["x", "u", "theta", "w"],
            ["A_x", "B_u", "S_theta", "G_w"],
        )

        # Exact AD Jacobians for variable-dt step
        A_x_dt = ca.jacobian(x_cur, x)
        B_u_dt = ca.jacobian(x_cur, u)
        S_theta_dt = ca.jacobian(x_cur, theta)
        G_w_dt = ca.jacobian(x_cur, w)
        self.fn_jacobians_dt = ca.Function(
            "step_jacobians_dt",
            [x, u, theta, w, dt_sym],
            [A_x_dt, B_u_dt, S_theta_dt, G_w_dt],
            ["x", "u", "theta", "w", "dt"],
            ["A_x", "B_u", "S_theta", "G_w"],
        )

        # Output Jacobians for adjoint/EKF/Tube-MPC observation linearization
        C_x = ca.jacobian(y_next_fixed, x)
        D_u = ca.jacobian(y_next_fixed, u)
        O_theta = ca.jacobian(y_next_fixed, theta)
        O_w = ca.jacobian(y_next_fixed, w)
        self.fn_output_jacobians = ca.Function(
            "output_jacobians",
            [x, u, theta, w],
            [C_x, D_u, O_theta, O_w],
            ["x", "u", "theta", "w"],
            ["C_x", "D_u", "O_theta", "O_w"],
        )

    # ------------------------------------------------------------------
    # Convenient Python runtime API
    # ------------------------------------------------------------------

    def step(
        self,
        x: np.ndarray,
        u: np.ndarray,
        theta: Optional[np.ndarray] = None,
        w: Optional[np.ndarray] = None,
        dt: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Advance state x by one RK4 step. Returns (x_next, outputs) as numpy arrays."""
        th = self.default_theta() if theta is None else np.asarray(theta, dtype=np.float64)
        wv = self.default_disturbance() if w is None else np.asarray(w, dtype=np.float64)
        if dt is None:
            xn, yn = self.fn_step(x, u, th, wv)
        else:
            xn, yn = self.fn_step_dt(x, u, th, wv, float(dt))
        return np.asarray(xn, dtype=np.float64).ravel(), np.asarray(yn, dtype=np.float64).ravel()

    def jacobians(
        self,
        x: np.ndarray,
        u: np.ndarray,
        theta: Optional[np.ndarray] = None,
        w: Optional[np.ndarray] = None,
        dt: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute exact AD Jacobians (A_x, B_u, S_theta, G_w) as numpy arrays."""
        th = self.default_theta() if theta is None else np.asarray(theta, dtype=np.float64)
        wv = self.default_disturbance() if w is None else np.asarray(w, dtype=np.float64)
        if dt is None:
            ax, bu, st, gw = self.fn_jacobians(x, u, th, wv)
        else:
            ax, bu, st, gw = self.fn_jacobians_dt(x, u, th, wv, float(dt))
        return (
            np.asarray(ax, dtype=np.float64),
            np.asarray(bu, dtype=np.float64),
            np.asarray(st, dtype=np.float64),
            np.asarray(gw, dtype=np.float64),
        )
