#!/usr/bin/env python3
"""
Exact shared physics model for Tecnun e-Racing TeR27 4WD (MirenaSim).

Loads parameters directly from:
  - src/mirena_sim/Scenes/Vehicle/Config/vehicles/ter27_4wd/*.json
  - src/mirena_sim/Scenes/Vehicle/Dynamics/Tires/MF62/hoosier_16x7.5_10_<compound>_mf62.json

Coordinate convention: ISO 8855 / ROS REP-103 (+x forward, +y left, +z up).
Wheel order: [0: FL, 1: FR, 2: RL, 3: RR].
Matches MirenaSim (Pacejka62.gd, TireModel.gd, AeroModel.gd, SuspensionSystem.gd,
ChassisDynamics.gd, DrivetrainModel.gd, Full14DofModel.gd, EcuTorqueVectoring4Wd.gd).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "src" / "mirena_sim" / "Scenes" / "Vehicle" / "Config" / "vehicles" / "ter27_4wd"
MF62_DIR = REPO_ROOT / "src" / "mirena_sim" / "Scenes" / "Vehicle" / "Dynamics" / "Tires" / "MF62"

# Standardized grip profiles:
# - COLD_TEST_OCT: intact at lambda_muy = 0.47 (35 C, TeR26 Oct telemetry 98.4% omega_z correlation)
# - FSG_RUBBERED: lambda_muy = 0.57 for fresh Hoosier R25B at 75 C nominal (70-85 C window),
#   with proportional scaling for R20 preserving TTC R25B/R20 ratio (PDY1 2.5608 vs 2.4744).
FSG_RUBBERED_LAMBDA_MUY_TARGET: float = 0.57
COLD_TEST_OCT_LAMBDA_MUY_TARGET: float = 0.47
FSG_RUBBERED_ENV_MU_SCALE: float = round(FSG_RUBBERED_LAMBDA_MUY_TARGET / COLD_TEST_OCT_LAMBDA_MUY_TARGET, 6)

GRIP_PROFILES: Dict[str, Dict[str, float]] = {
    "TTC_LAB_PEAK": {"env_mu_scale": 1.00, "env_aero_cl_scale": 1.00},
    "FSG_REAL_TRACK": {"env_mu_scale": 0.86, "env_aero_cl_scale": 0.64},
    "COLD_TEST_OCT": {
        "env_mu_scale": 1.00,
        "env_aero_cl_scale": 0.78,
        "track_temp_c": 18.0,
        "tire_temp_c": 35.0,
        "lambda_muy_target": COLD_TEST_OCT_LAMBDA_MUY_TARGET,
    },
    "FSG_RUBBERED": {
        "env_mu_scale": FSG_RUBBERED_ENV_MU_SCALE,
        "env_aero_cl_scale": 0.78,
        "track_temp_c": 45.0,
        "tire_temp_c": 75.0,
        "lambda_muy_target": FSG_RUBBERED_LAMBDA_MUY_TARGET,
    },
    "COLD_LOW_GRIP": {
        "env_mu_scale": 0.72,
        "env_aero_cl_scale": 0.64,
        "track_temp_c": 18.0,
        "tire_temp_c": 22.0,
    },
}

# Mechanical trackside setup optimization bounds (shared across CasADi solvers & presets)
TRACKSIDE_SETUP_BOUNDS: Dict[str, Tuple[float, float]] = {
    "lltd": (0.35, 0.62),
    "camber_f_deg": (-2.5, -0.2),
    "camber_r_deg": (-2.2, -0.2),
    "toe_f_deg": (-0.5, 0.5),
    "toe_r_deg": (0.0, 0.6),
    "brake_bias": (0.55, 0.78),
    "rake_mm": (-8.0, 8.0),
}


def rake_to_aero_modifiers(rake_mm: float, base_cop_front: float = 0.48) -> Tuple[float, float, float]:
    """
    Smooth physical mapping from static rake (rh_rear - rh_front in mm, [-8, +8])
    to (cl_scale, cd_scale, cop_front_frac) for TeR27 undertray + diffuser.
    Positive rake increases front-down pitch: shifts CoP forward (+0.004/mm),
    increases diffuser downforce smoothly up to ~+7 mm, with modest drag increase.
    """
    r = float(rake_mm)
    cl_scale = 1.0 + 0.0080 * r - 0.00055 * r * r
    cd_scale = 1.0 + 0.0035 * r + 0.00030 * r * r
    cop_front = float(np.clip(base_cop_front + 0.0040 * r, 0.40, 0.56))
    return float(cl_scale), float(cd_scale), cop_front


@dataclass
class TeR27Params:
    # Vehicle geometry & mass
    mass: float
    sprung_mass: float
    unsprung_mass: float
    unsprung_mass_total: float
    gravity: float
    wheelbase: float
    track_front: float
    track_rear: float
    h_cog: float
    h_sprung_cog: float
    h_rc_f: float
    h_rc_r: float
    front_weight_dist: float
    a_cog: float
    b_cog: float
    max_steer_angle: float
    ackermann_factor: float
    ixx: float
    iyy: float
    izz_sprung: float
    izz_total: float
    ixz: float

    # Suspension & kinematics
    k_spring_f: float
    k_spring_r: float
    mr_f: float
    mr_r: float
    k_arb_f: float
    k_arb_r: float
    k_tire_z: float
    k_wheel_f: float
    k_wheel_r: float
    k_roll_f: float
    k_roll_r: float
    lltd_elastic_f: float
    h_rc_cg: float
    h_roll_arm: float
    static_camber_f_rad: float
    static_camber_r_rad: float
    caster_rad: float
    kpi_rad: float
    mech_trail_f: float
    k_steer_comp: float
    anti_dive_f: float
    anti_squat_r: float

    # Aero
    air_density: float
    cd_a: float
    cl_a: float
    aero_balance_front: float
    cp_height: float
    static_rh_f: float
    static_rh_r: float
    stall_rh_f: float

    # Tires
    wheel_radius: float
    wheel_inertia: float
    wheel_eff_inertia: float
    rolling_resistance_coeff: float
    slip_reg_speed: float
    tire_compound: str
    lambda_mux_base: float
    lambda_muy_base: float
    p_infl_kpa: float

    # Powertrain & HV
    p_dc_max_w: float
    p_dc_min_w: float  # negative = max regen charge (-25000 W)
    motor_peak_torque: float
    motor_max_rpm: float
    base_speed_rpm: float
    motor_taper_ratio: float
    gear_ratio: float
    eta_drive: float
    eta_regen: float
    t_regen_max_wheel: float
    v_oc_nom: float
    v_nom: float
    r_int_ohm: float
    front_drive_ratio: float
    brake_mu: float
    brake_bias_front: float
    static_toe_f_rad: float = 0.0
    static_toe_r_rad: float = 0.0


def load_ter27_params(config_dir: Path = CONFIG_DIR) -> TeR27Params:
    veh = json.loads((config_dir / "vehicle.json").read_text(encoding="utf-8"))
    aero = json.loads((config_dir / "aero.json").read_text(encoding="utf-8"))
    pt = json.loads((config_dir / "powertrain.json").read_text(encoding="utf-8"))
    susp = json.loads((config_dir / "suspension.json").read_text(encoding="utf-8"))
    tires = json.loads((config_dir / "tires.json").read_text(encoding="utf-8"))
    brakes = json.loads((config_dir / "brakes.json").read_text(encoding="utf-8"))
    tv_cfg = json.loads((config_dir / "tv_4wd_config.json").read_text(encoding="utf-8"))

    mass = float(veh["mass"])
    sprung_mass = float(veh["sprung_mass"])
    unsprung_mass = float(veh["unsprung_mass"])
    unsprung_mass_total = float(veh["unsprung_mass_total"])
    gravity = float(veh["gravity"])
    L = float(veh["wheelbase"])
    tf = float(veh["track_front"])
    tr = float(veh["track_rear"])
    h_cog = float(veh["cg_height"])
    h_s = float(veh["sprung_cg_height"])
    h_rc_f = float(veh["roll_center_height_front"])
    h_rc_r = float(veh["roll_center_height_rear"])
    fwd = float(veh["front_weight_distribution"])
    a_cog = L * (1.0 - fwd)
    b_cog = L * fwd

    izz_sprung = float(veh["inertia"]["izz"])
    w_asm_i = float(pt["wheel_assembly_inertia"])
    steiner_f = 2.0 * unsprung_mass * (a_cog * a_cog + (0.5 * tf) ** 2)
    steiner_r = 2.0 * unsprung_mass * (b_cog * b_cog + (0.5 * tr) ** 2)
    izz_total = izz_sprung + steiner_f + steiner_r + 4.0 * w_asm_i

    k_sf = float(susp["spring_stiffness_front"])
    k_sr = float(susp["spring_stiffness_rear"])
    mr_f = float(susp["motion_ratio_front"])
    mr_r = float(susp["motion_ratio_rear"])
    k_arb_f = float(susp["arb_stiffness_front"])
    k_arb_r = float(susp["arb_stiffness_rear"])
    k_tz = float(tires["tire_vertical_stiffness"])

    k_wf = k_sf * mr_f * mr_f
    k_wr = k_sr * mr_r * mr_r
    # Series ride rate including tire vertical stiffness for roll stiffness calculation
    k_rf = (k_wf * k_tz) / (k_wf + k_tz)
    k_rr = (k_wr * k_tz) / (k_wr + k_tz)
    # Axle roll stiffnesses: K_phi = 0.5 * t^2 * (k_ride + 2*k_arb_eff)
    # ARB acts in parallel with spring, in series with tire
    k_tot_f = ((k_wf + 2.0 * k_arb_f) * k_tz) / (k_wf + 2.0 * k_arb_f + k_tz)
    k_tot_r = ((k_wr + 2.0 * k_arb_r) * k_tz) / (k_wr + 2.0 * k_arb_r + k_tz)
    k_roll_f = 0.5 * tf * tf * k_tot_f
    k_roll_r = 0.5 * tr * tr * k_tot_r
    lltd_elastic_f = k_roll_f / (k_roll_f + k_roll_r)

    h_rc_cg = h_rc_f + (h_rc_r - h_rc_f) * (a_cog / L)
    h_roll_arm = max(0.0, h_s - h_rc_cg)

    G = float(pt["gear_ratio"])
    i_rot = float(pt["motor_rotor_inertia"])
    i_eff = w_asm_i + G * G * i_rot

    el_tv = tv_cfg.get("electrical", {})
    p_dc_max = float(pt.get("max_dc_power_w", pt.get("power_limit_watts", 80000.0)))
    p_chg_max = float(brakes.get("p_max_charge_w", el_tv.get("p_max_charge_w", 25000.0)))

    # Initial OCV at soc=0.95 from DrivetrainModel.compute_ocv
    v_full = float(pt.get("battery_ocv_full_v", 600.0))
    v_empty = max(v_full - 160.0, 0.70 * v_full)
    soc0 = float(pt.get("battery_initial_soc", 0.95))
    span = max(v_full - v_empty, 0.0)
    knee = 0.18 * span * (1.0 - math.exp(-10.0 * soc0)) / (1.0 - math.exp(-10.0))
    v_oc_nom = v_empty + knee + 0.82 * span * soc0

    return TeR27Params(
        mass=mass,
        sprung_mass=sprung_mass,
        unsprung_mass=unsprung_mass,
        unsprung_mass_total=unsprung_mass_total,
        gravity=gravity,
        wheelbase=L,
        track_front=tf,
        track_rear=tr,
        h_cog=h_cog,
        h_sprung_cog=h_s,
        h_rc_f=h_rc_f,
        h_rc_r=h_rc_r,
        front_weight_dist=fwd,
        a_cog=a_cog,
        b_cog=b_cog,
        max_steer_angle=float(veh["max_steer_angle"]),
        ackermann_factor=float(veh["ackermann_factor"]),
        ixx=float(veh["inertia"]["ixx"]),
        iyy=float(veh["inertia"]["iyy"]),
        izz_sprung=izz_sprung,
        izz_total=izz_total,
        ixz=float(veh["inertia"]["ixz"]),
        k_spring_f=k_sf,
        k_spring_r=k_sr,
        mr_f=mr_f,
        mr_r=mr_r,
        k_arb_f=k_arb_f,
        k_arb_r=k_arb_r,
        k_tire_z=k_tz,
        k_wheel_f=k_wf,
        k_wheel_r=k_wr,
        k_roll_f=k_roll_f,
        k_roll_r=k_roll_r,
        lltd_elastic_f=lltd_elastic_f,
        h_rc_cg=h_rc_cg,
        h_roll_arm=h_roll_arm,
        static_camber_f_rad=math.radians(float(susp["static_camber_front_deg"])),
        static_camber_r_rad=math.radians(float(susp["static_camber_rear_deg"])),
        caster_rad=math.radians(float(susp["caster_angle_deg"])),
        kpi_rad=math.radians(float(susp["kpi_angle_deg"])),
        mech_trail_f=float(susp["mechanical_trail_front"]),
        k_steer_comp=float(susp["steering_compliance_stiffness_nm_rad"]),
        anti_dive_f=float(susp["anti_dive_ratio_front"]),
        anti_squat_r=float(susp["anti_squat_ratio_rear"]),
        air_density=float(aero["air_density"]),
        cd_a=float(aero["cd_a"]),
        cl_a=float(aero["cl_a"]),
        aero_balance_front=float(aero["aero_balance_front"]),
        cp_height=float(aero["cp_height"]),
        static_rh_f=float(aero["static_ride_height_front"]),
        static_rh_r=float(aero["static_ride_height_rear"]),
        stall_rh_f=float(aero["stall_ride_height_front"]),
        wheel_radius=float(tires["wheel_radius"]),
        wheel_inertia=float(tires["wheel_inertia"]),
        wheel_eff_inertia=i_eff,
        rolling_resistance_coeff=float(tires["rolling_resistance_coeff"]),
        slip_reg_speed=float(tires["slip_regularization_speed"]),
        tire_compound=str(tires["tire_compound"]).upper(),
        lambda_mux_base=float(tires["lambda_mux"]),
        lambda_muy_base=float(tires["lambda_muy"]),
        p_infl_kpa=83.0,
        p_dc_max_w=p_dc_max,
        p_dc_min_w=-p_chg_max,
        motor_peak_torque=float(pt["motor_peak_torque"]),
        motor_max_rpm=float(pt["motor_max_rpm"]),
        base_speed_rpm=float(pt["base_speed_rpm"]),
        motor_taper_ratio=float(pt["motor_taper_ratio"]),
        gear_ratio=G,
        eta_drive=float(pt["drivetrain_efficiency"]),
        eta_regen=float(el_tv.get("eta_regen", pt["drivetrain_efficiency"])),
        t_regen_max_wheel=float(el_tv.get("t_regen_max_wheel_nm", 150.0)),
        v_oc_nom=v_oc_nom,
        v_nom=float(pt.get("battery_nominal_voltage_v", 540.0)),
        r_int_ohm=float(pt.get("battery_internal_resistance_ohm", 0.14)),
        front_drive_ratio=float(pt.get("front_drive_torque_ratio", 0.35)),
        brake_mu=float(brakes["brake_mu"]),
        brake_bias_front=float(brakes["brake_bias_front"]),
    )


class PacejkaMF62:
    """Exact vectorized Python port of MirenaSim Pacejka62.gd (Schema v2)."""

    def __init__(self, compound: str = "R20", mf62_dir: Path = MF62_DIR) -> None:
        fname = f"hoosier_16x7.5_10_{compound.lower()}_mf62.json"
        path = mf62_dir / fname
        d = json.loads(path.read_text(encoding="utf-8"))
        self.d = d
        self.compound = str(d["COMPOUND"]).upper()
        self.fnomin = float(d["FNOMIN"])
        self.pinom = float(d["PINOM"])
        self.r0 = float(d["UNLOADED_RADIUS"])
        self.dfz_max = (float(d["FZ_VALID_MAX"]) - self.fnomin) / self.fnomin if "FZ_VALID_MAX" in d else np.inf

        # Store all MF62 coefficients as float attributes
        for k, v in d.items():
            if isinstance(v, (int, float)):
                setattr(self, k, float(v))

    @staticmethod
    def _shape_cos(b, c, e, x):
        bx = b * x
        return np.cos(c * np.arctan(bx - e * (bx - np.arctan(bx))))

    def compute_combined_forces_vec(
        self,
        kappa: np.ndarray,
        alpha: np.ndarray,
        fz: np.ndarray,
        gamma: np.ndarray,
        lmux: float,
        lmuy: float,
        is_right: np.ndarray,
        p_kpa: float = 83.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Vectorized exact MF 6.2 combined-slip force evaluation.
        Returns (Fx, Fy, Mz, mu_y_eff) for arrays of arbitrary matching shape (e.g. (4,) or (N, 4)).
        """
        fz_pos = np.maximum(fz, 0.0)
        valid = fz_pos > 1e-3
        fz_safe = np.where(valid, fz_pos, 1.0)

        dfz = np.minimum((fz_safe - self.fnomin) / self.fnomin, self.dfz_max)
        dpi = (p_kpa - self.pinom) / self.pinom
        side = np.where(is_right, -1.0, 1.0)

        # Pure lateral Fy0
        cy = self.PCY1
        camber_f_y = np.maximum(1.0 - self.PDY3 * gamma * gamma, 0.2)
        mu_y = (self.PDY1 + self.PDY2 * dfz) * (1.0 + self.PPY3 * dpi + self.PPY4 * dpi * dpi) * camber_f_y * lmuy
        dy = mu_y * fz_safe
        shy = side * (self.PHY1 + self.PHY2 * dfz) + self.PHY3 * gamma
        svy = side * fz_safe * (self.PVY1 + self.PVY2 * dfz) + fz_safe * (self.PVY3 + self.PVY4 * dfz) * gamma
        alpha_y = alpha + shy
        kya = (
            self.PKY1
            * self.fnomin
            * (1.0 + self.PPY1 * dpi)
            * (1.0 - self.PKY3 * np.abs(gamma))
            * np.sin(2.0 * np.arctan(fz_safe / (self.PKY2 * self.fnomin * (1.0 + self.PPY2 * dpi) + 1e-4)))
        )
        by = kya / (cy * dy + 1e-4)
        # Smooth sign approximation is NOT used in Pacejka62.gd; PEY3/PEY4 are 0.0 in R20 anyway
        ey = np.clip(
            (self.PEY1 + self.PEY2 * dfz)
            * (1.0 + self.PEY5 * gamma * gamma - (self.PEY3 + self.PEY4 * gamma) * np.sign(alpha_y)),
            -10.0,
            1.0,
        )
        y = by * alpha_y
        fy0 = dy * np.sin(cy * np.arctan(y - ey * (y - np.arctan(y)))) + svy

        # Pure longitudinal Fx0
        cx = self.PCX1
        camber_f_x = np.maximum(1.0 - self.PDX3 * gamma * gamma, 0.2)
        mu_x = (self.PDX1 + self.PDX2 * dfz) * (1.0 + self.PPX3 * dpi + self.PPX4 * dpi * dpi) * camber_f_x * lmux
        dx = mu_x * fz_safe
        kx = kappa + self.PHX1 + self.PHX2 * dfz
        svx = fz_safe * (self.PVX1 + self.PVX2 * dfz)
        kxk = (
            fz_safe
            * (self.PKX1 + self.PKX2 * dfz)
            * np.exp(self.PKX3 * dfz)
            * (1.0 + self.PPX1 * dpi + self.PPX2 * dpi * dpi)
        )
        bx = kxk / (cx * dx + 1e-4)
        ex = np.clip(
            (self.PEX1 + self.PEX2 * dfz + self.PEX3 * dfz * dfz) * (1.0 - self.PEX4 * np.sign(kx)),
            -10.0,
            1.0,
        )
        x = bx * kx
        fx0 = dx * np.sin(cx * np.arctan(x - ex * (x - np.arctan(x)))) + svx

        # Combined slip weighting
        bxa = (self.RBX1 + self.RBX3 * gamma * gamma) * np.cos(np.arctan(self.RBX2 * kappa))
        exa = np.clip(self.REX1 + self.REX2 * dfz, -10.0, 1.0)
        gxa = np.maximum(
            self._shape_cos(bxa, self.RCX1, exa, alpha + self.RHX1)
            / np.maximum(self._shape_cos(bxa, self.RCX1, exa, self.RHX1), 1e-4),
            0.0,
        )

        byk = (self.RBY1 + self.RBY4 * gamma * gamma) * np.cos(np.arctan(self.RBY2 * (alpha - self.RBY3)))
        eyk = np.clip(self.REY1 + self.REY2 * dfz, -10.0, 1.0)
        shyk = self.RHY1 + self.RHY2 * dfz
        gyk = np.maximum(
            self._shape_cos(byk, self.RCY1, eyk, kappa + shyk)
            / np.maximum(self._shape_cos(byk, self.RCY1, eyk, shyk), 1e-4),
            0.0,
        )

        dvyk = mu_y * fz_safe * (self.RVY1 + self.RVY2 * dfz + self.RVY3 * gamma) * np.cos(np.arctan(self.RVY4 * alpha))
        svyk = dvyk * np.sin(self.RVY5 * np.arctan(self.RVY6 * kappa))

        fx = np.where(valid, gxa * fx0, 0.0)
        fy = np.where(valid, gyk * fy0 + svyk, 0.0)

        # Self-aligning torque Mz
        shf = shy + svy / (kya + 1e-3)
        sht = self.QHZ1 + self.QHZ2 * dfz + (self.QHZ3 + self.QHZ4 * dfz) * gamma
        alpha_t = alpha + sht
        alpha_r = alpha + shf
        kx_eq = (kxk / (kya + 1e-3)) * kappa
        sgn_at = np.where(alpha_t >= 0.0, 1.0, -1.0)
        sgn_ar = np.where(alpha_r >= 0.0, 1.0, -1.0)
        alpha_t_eq = sgn_at * np.sqrt(alpha_t * alpha_t + kx_eq * kx_eq + 1e-12)
        alpha_r_eq = sgn_ar * np.sqrt(alpha_r * alpha_r + kx_eq * kx_eq + 1e-12)

        dt = (
            (fz_safe / self.fnomin)
            * self.r0
            * (self.QDZ1 + self.QDZ2 * dfz)
            * (1.0 + self.PPZ1 * dpi)
            * (1.0 + self.QDZ3 * np.abs(gamma) + self.QDZ4 * gamma * gamma)
        )
        ct = self.QCZ1
        bt = (self.QBZ1 + self.QBZ2 * dfz + self.QBZ3 * dfz * dfz) * (
            1.0 + self.QBZ4 * gamma + self.QBZ5 * np.abs(gamma)
        )
        t_arg = bt * alpha_t_eq
        et = np.clip(
            (self.QEZ1 + self.QEZ2 * dfz + self.QEZ3 * dfz * dfz)
            * (1.0 + (self.QEZ4 + self.QEZ5 * gamma) * (2.0 / np.pi) * np.arctan(bt * ct * alpha_t_eq)),
            -10.0,
            1.0,
        )
        trail = dt * np.cos(ct * np.arctan(t_arg - et * (t_arg - np.arctan(t_arg)))) * np.cos(alpha)
        mz_prime = -trail * (fy - svyk)

        dr = (
            fz_safe
            * self.r0
            * (side * (self.QDZ6 + self.QDZ7 * dfz) + (self.QDZ8 + self.QDZ9 * dfz) * (1.0 + self.PPZ2 * dpi) * gamma)
            * np.cos(alpha)
        )
        br = self.QBZ9 + self.QBZ10 * by * cy
        mzr = dr * np.cos(np.arctan(br * alpha_r_eq))

        s_arm = self.r0 * (self.SSZ1 + self.SSZ2 * (fy / self.fnomin) + (self.SSZ3 + self.SSZ4 * dfz) * gamma)
        mz = np.where(valid, mz_prime + mzr + s_arm * fx, 0.0)

        return fx, fy, mz, np.where(valid, mu_y, 0.0)


class TeR27PhysicsModel:
    """
    Unified TeR27 4WD planar + load-transfer + MF6.2 + Aero + HV Powertrain model
    matching MirenaSim (Full14DofModel.gd / DrivetrainModel.gd / Pacejka62.gd).
    """

    IS_RIGHT = np.array([False, True, False, True], dtype=bool)
    IS_FRONT = np.array([True, True, False, False], dtype=bool)

    def __init__(self, profile: str = "FSG_REAL_TRACK") -> None:
        if profile not in GRIP_PROFILES:
            raise ValueError(f"Unknown grip profile '{profile}'. Expected one of {list(GRIP_PROFILES)}")
        self.profile = profile
        self.p = load_ter27_params()
        self.tire = PacejkaMF62(self.p.tire_compound)
        self.env_mu_scale = GRIP_PROFILES[profile]["env_mu_scale"]
        self.env_aero_cl_scale = GRIP_PROFILES[profile]["env_aero_cl_scale"]
        self.lmux = self.p.lambda_mux_base * self.env_mu_scale
        self.lmuy = self.p.lambda_muy_base * self.env_mu_scale

        # Wheel coordinates relative to CoG in ISO 8855 (+x fwd, +y left)
        self.x_w = np.array([self.p.a_cog, self.p.a_cog, -self.p.b_cog, -self.p.b_cog], dtype=float)
        self.y_w = np.array(
            [0.5 * self.p.track_front, -0.5 * self.p.track_front, 0.5 * self.p.track_rear, -0.5 * self.p.track_rear],
            dtype=float,
        )

    def ackermann_steer(self, delta: float) -> np.ndarray:
        """Exact port of SuspensionSystem.compute_ackermann_steer for [FL, FR, RL, RR] with optional static toe."""
        abs_s = abs(delta)
        ack = np.clip(self.p.ackermann_factor, -1.0, 1.0)
        tf = float(getattr(self.p, "static_toe_f_rad", 0.0))
        tr = float(getattr(self.p, "static_toe_r_rad", 0.0))
        if abs_s < 1e-6 or abs(ack) < 1e-9:
            return np.array([delta - tf, delta + tf, -tr, tr], dtype=float)
        half_t = 0.5 * self.p.track_front
        r_turn = self.p.wheelbase / math.tan(abs_s)
        sgn = 1.0 if delta > 0.0 else -1.0

        def _one(is_left: bool) -> float:
            is_inner = (delta > 0.0 and is_left) or (delta < 0.0 and not is_left)
            r_w = max(r_turn - half_t, 0.25 * self.p.wheelbase) if is_inner else (r_turn + half_t)
            ack_mag = math.atan2(self.p.wheelbase, r_w)
            return sgn * (abs_s + ack * (ack_mag - abs_s))

        return np.array([_one(True) - tf, _one(False) + tf, -tr, tr], dtype=float)

    def aero_forces(self, vx: float, vy: float = 0.0) -> Tuple[float, float, float, float]:
        """
        Computes aerodynamic forces in ISO 8855 (+x fwd, +y left).
        Returns (F_drag_x, F_drag_y, F_down_front, F_down_rear) where F_down_* >= 0.
        Matches AeroModel.gd + Full14DofModel.gd env_aero_cl_scale.
        """
        speed = math.hypot(vx, vy)
        if speed < 0.1:
            return 0.0, 0.0, 0.0, 0.0
        q = 0.5 * self.p.air_density * speed * speed
        f_drag_mag = q * self.p.cd_a
        f_drag_x = -f_drag_mag * (vx / speed)
        f_drag_y = -f_drag_mag * (vy / speed)
        f_down_f = q * self.p.cl_a * self.p.aero_balance_front * self.env_aero_cl_scale
        f_down_r = q * self.p.cl_a * (1.0 - self.p.aero_balance_front) * self.env_aero_cl_scale
        return f_drag_x, f_drag_y, f_down_f, f_down_r

    def compute_normal_forces(
        self,
        ax: float,
        ay: float,
        vx: float,
        vy: float = 0.0,
    ) -> Tuple[np.ndarray, float]:
        """
        Computes steady-state 4-wheel normal loads Fz [FL, FR, RL, RR] (N) and roll angle phi (rad)
        with exact elastic + geometric roll-center + unsprung-mass lateral load transfer,
        longitudinal load transfer, and axle-split aerodynamic downforce.
        Positive ay (left turn) transfers load from Left (+y) to Right (-y).
        Positive ax (acceleration) transfers load from Front to Rear.
        """
        p = self.p
        _, _, f_down_f, f_down_r = self.aero_forces(vx, vy)
        mg = p.mass * p.gravity
        fz_f_stat = mg * (p.b_cog / p.wheelbase)
        fz_r_stat = mg * (p.a_cog / p.wheelbase)

        # Longitudinal load transfer (includes aero pitching moment effect around CoG via axle downforce)
        dfz_long = (p.mass * ax * p.h_cog) / p.wheelbase
        fz_f_tot = max(0.0, fz_f_stat - dfz_long + f_down_f)
        fz_r_tot = max(0.0, fz_r_stat + dfz_long + f_down_r)

        # Lateral load transfer: sprung elastic (roll stiffness split) + sprung geometric (roll centers) + unsprung
        m_s = p.sprung_mass
        m_u_axle = 0.5 * p.unsprung_mass_total
        # Roll angle phi (+ right down when ay > 0)
        k_roll_tot = p.k_roll_f + p.k_roll_r
        phi = (m_s * ay * p.h_roll_arm) / max(k_roll_tot - m_s * p.gravity * p.h_roll_arm, 1.0)

        # Elastic roll moment per axle
        m_roll_f = p.k_roll_f * phi
        m_roll_r = p.k_roll_r * phi
        dfz_el_f = m_roll_f / p.track_front
        dfz_el_r = m_roll_r / p.track_rear

        # Geometric roll-center load transfer (using axle weight fractions for sprung lateral force)
        dfz_rc_f = (m_s * (p.b_cog / p.wheelbase) * ay * p.h_rc_f) / p.track_front
        dfz_rc_r = (m_s * (p.a_cog / p.wheelbase) * ay * p.h_rc_r) / p.track_rear

        # Unsprung mass load transfer
        dfz_u_f = (m_u_axle * ay * p.wheel_radius) / p.track_front
        dfz_u_r = (m_u_axle * ay * p.wheel_radius) / p.track_rear

        dfz_lat_f = dfz_el_f + dfz_rc_f + dfz_u_f
        dfz_lat_r = dfz_el_r + dfz_rc_r + dfz_u_r

        # Clamp lateral transfer per axle so neither wheel drops below 0 while conserving axle load
        dfz_lat_f = np.clip(dfz_lat_f, -0.5 * fz_f_tot, 0.5 * fz_f_tot)
        dfz_lat_r = np.clip(dfz_lat_r, -0.5 * fz_r_tot, 0.5 * fz_r_tot)

        fz = np.array(
            [
                0.5 * fz_f_tot - dfz_lat_f,
                0.5 * fz_f_tot + dfz_lat_f,
                0.5 * fz_r_tot - dfz_lat_r,
                0.5 * fz_r_tot + dfz_lat_r,
            ],
            dtype=float,
        )
        return np.maximum(fz, 0.0), float(phi)

    def compute_cambers(self, delta_w: np.ndarray, phi: float) -> np.ndarray:
        """
        Effective ground camber gamma (rad) in ISO 8855 for [FL, FR, RL, RR].
        Matches SuspensionSystem.compute_ground_camber_iso.
        """
        p = self.p
        # Static camber: negative vehicle camber = top inward -> left: -gamma, right: +gamma
        g_stat = np.array(
            [p.static_camber_f_rad, -p.static_camber_f_rad, p.static_camber_r_rad, -p.static_camber_r_rad],
            dtype=float,
        )
        g_steer = np.zeros(4, dtype=float)
        # FL (left, kpi_sign=+1), FR (right, kpi_sign=-1)
        g_steer[0] = p.caster_rad * math.sin(delta_w[0]) + p.kpi_rad * (1.0 - math.cos(delta_w[0]))
        g_steer[1] = p.caster_rad * math.sin(delta_w[1]) - p.kpi_rad * (1.0 - math.cos(delta_w[1]))
        return g_stat - phi + g_steer

    def wheel_motor_limits(self, omega_w: np.ndarray, v_dc: float = -1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Exact per-wheel drive torque upper bound T_max >= 0 and regen lower bound T_min <= 0 (Nm at wheel).
        Matches DrivetrainModel.gd & EcuTorqueVectoring4Wd.gd field weakening + speed taper.
        """
        p = self.p
        v_eff = v_dc if v_dc > 0.0 else p.v_oc_nom
        v_scale = np.clip(v_eff / p.v_nom, 0.5, 1.25)
        omega_base_m = p.base_speed_rpm * (2.0 * math.pi / 60.0) * v_scale
        omega_max_m = p.motor_max_rpm * (2.0 * math.pi / 60.0)
        omega_taper_start = omega_max_m * p.motor_taper_ratio
        p_motor_peak = p.motor_peak_torque * omega_base_m

        om_m = np.abs(omega_w) * p.gear_ratio
        t_env = np.where(om_m > omega_base_m, np.minimum(p.motor_peak_torque, p_motor_peak / np.maximum(om_m, 1e-3)), p.motor_peak_torque)
        speed_scale = np.where(
            om_m <= omega_taper_start,
            1.0,
            np.where(om_m >= omega_max_m, 0.0, 1.0 - (om_m - omega_taper_start) / max(omega_max_m - omega_taper_start, 1e-5)),
        )
        # In EcuTorqueVectoring4Wd.gd, motor torque at wheel is t_env * gear_ratio * speed_scale
        t_max = t_env * p.gear_ratio * speed_scale
        t_min = -np.minimum(p.t_regen_max_wheel, t_max)
        return t_max, t_min

    def compute_hv_power_w(self, torques: np.ndarray, omega_w: np.ndarray) -> float:
        """
        Net HV DC electrical power (W) matching DrivetrainModel.step_hv_and_inverters:
        P_mech_i = T_i * omega_i
        if P_mech_i >= 0: P_elec_i = P_mech_i / eta_drive
        if P_mech_i <  0: P_elec_i = P_mech_i * eta_regen (or eta_drive in DrivetrainModel)
        Note: using eta_drive=0.90 for drive and eta_regen=0.88 for regen is strictly conservative
        for the +80 kW ceiling (less regen credit) and matches EcuTorqueVectoring4Wd.gd.
        To guarantee 0 violations in BOTH DrivetrainModel (eta=0.90) and ECU (eta_regen=0.88),
        we also provide compute_hv_power_drivetrain_w().
        """
        p_mech = torques * omega_w
        p_elec = np.where(p_mech >= 0.0, p_mech / self.p.eta_drive, p_mech * self.p.eta_regen)
        return float(np.sum(p_elec))

    def compute_hv_power_drivetrain_w(self, torques: np.ndarray, omega_w: np.ndarray) -> float:
        """Exact DrivetrainModel.step_hv_and_inverters DC power (uses drivetrain_efficiency=0.90 both ways)."""
        p_mech = torques * omega_w
        p_elec = np.where(p_mech >= 0.0, p_mech / self.p.eta_drive, p_mech * self.p.eta_drive)
        return float(np.sum(p_elec))

    def wheel_velocities_and_alphas(
        self, vx: float, vy: float, r: float, delta_w: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes per-wheel longitudinal velocity v_wx, lateral velocity v_wy, and slip angle alpha (rad)
        in ISO 8855 matching WheelDynamics.compute_slip.
        """
        v_hub_x = vx - r * self.y_w
        v_hub_y = vy + r * self.x_w
        cos_d = np.cos(delta_w)
        sin_d = np.sin(delta_w)
        v_wx = v_hub_x * cos_d + v_hub_y * sin_d
        v_wy = -v_hub_x * sin_d + v_hub_y * cos_d
        u_den = np.maximum(np.abs(v_wx), self.p.slip_reg_speed)
        alpha = -np.arctan2(v_wy, u_den)
        return v_wx, v_wy, alpha

    def solve_kappa_for_fx_target(
        self,
        fx_target: np.ndarray,
        alpha: np.ndarray,
        fz: np.ndarray,
        gamma: np.ndarray,
        n_iter: int = 8,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Given desired wheel longitudinal forces fx_target[4] (from wheel torque T_i / r_w - F_rr_i),
        inverts MF 6.2 Fx(kappa, alpha, fz) to find steady-state kappa[4] on the stable side of the
        slip curve (|kappa| <= kappa_peak), clamping fx_target if it exceeds tire Fx peak.
        Returns (kappa, Fx_actual, Fy_actual, Mz_tire).
        """
        # Vectorized Newton/bisection hybrid on stable branch [-0.14, +0.14]
        k_lo = np.full(4, -0.14, dtype=float)
        k_hi = np.full(4, 0.14, dtype=float)
        # First evaluate at endpoints to clamp unattainable targets cleanly
        fx_lo, _, _, _ = self.tire.compute_combined_forces_vec(k_lo, alpha, fz, gamma, self.lmux, self.lmuy, self.IS_RIGHT, self.p.p_infl_kpa)
        fx_hi, _, _, _ = self.tire.compute_combined_forces_vec(k_hi, alpha, fz, gamma, self.lmux, self.lmuy, self.IS_RIGHT, self.p.p_infl_kpa)
        tgt = np.clip(fx_target, np.minimum(fx_lo, fx_hi), np.maximum(fx_lo, fx_hi))

        # Bisection (12 steps gives < 1e-4 kappa accuracy and is 100% monotone & branch-safe)
        lo = k_lo.copy()
        hi = k_hi.copy()
        for _ in range(14):
            mid = 0.5 * (lo + hi)
            fx_m, _, _, _ = self.tire.compute_combined_forces_vec(mid, alpha, fz, gamma, self.lmux, self.lmuy, self.IS_RIGHT, self.p.p_infl_kpa)
            go_right = fx_m < tgt
            lo = np.where(go_right, mid, lo)
            hi = np.where(go_right, hi, mid)
        kappa = 0.5 * (lo + hi)
        fx, fy, mz, _ = self.tire.compute_combined_forces_vec(kappa, alpha, fz, gamma, self.lmux, self.lmuy, self.IS_RIGHT, self.p.p_infl_kpa)
        return kappa, fx, fy, mz

    def evaluate_state_from_kappa(
        self,
        vx: float,
        vy: float,
        r: float,
        delta: float,
        kappa: np.ndarray,
        ax_est: float,
        ay_est: float,
        n_load_iter: int = 3,
    ) -> Dict[str, object]:
        """
        Evaluates full steady-state / quasi-steady vehicle forces, moments, wheel torques,
        and HV power given (vx, vy, r, delta, kappa[4]), iterating load transfer to self-consistency.
        """
        delta_w = self.ackermann_steer(delta)
        v_wx, v_wy, alpha = self.wheel_velocities_and_alphas(vx, vy, r, delta_w)
        f_drag_x, f_drag_y, f_down_f, f_down_r = self.aero_forces(vx, vy)

        ax_cur, ay_cur = float(ax_est), float(ay_est)
        fz = np.zeros(4)
        phi = 0.0
        fx_w = np.zeros(4)
        fy_w = np.zeros(4)
        mz_tire = np.zeros(4)
        mu_y_eff = np.zeros(4)
        fx_c = np.zeros(4)
        fy_c = np.zeros(4)
        cos_d = np.cos(delta_w)
        sin_d = np.sin(delta_w)

        for _ in range(max(n_load_iter, 1)):
            fz, phi = self.compute_normal_forces(ax_cur, ay_cur, vx, vy)
            gamma = self.compute_cambers(delta_w, phi)
            fx_w, fy_w, mz_tire, mu_y_eff = self.tire.compute_combined_forces_vec(
                kappa, alpha, fz, gamma, self.lmux, self.lmuy, self.IS_RIGHT, self.p.p_infl_kpa
            )
            fx_c = fx_w * cos_d - fy_w * sin_d
            fy_c = fx_w * sin_d + fy_w * cos_d
            ax_new = (float(np.sum(fx_c)) + f_drag_x) / self.p.mass
            ay_new = (float(np.sum(fy_c)) + f_drag_y) / self.p.mass
            if abs(ax_new - ax_cur) < 1e-4 and abs(ay_new - ay_cur) < 1e-4:
                ax_cur, ay_cur = ax_new, ay_new
                break
            ax_cur = 0.5 * ax_cur + 0.5 * ax_new
            ay_cur = 0.5 * ay_cur + 0.5 * ay_new

        # Net forces and yaw moment in ISO 8855
        sum_fx = float(np.sum(fx_c)) + f_drag_x
        sum_fy = float(np.sum(fy_c)) + f_drag_y
        # Yaw moment: r_pos x F = x_w * Fy_c - y_w * Fx_c + Mz_tire
        mz_from_fy = float(np.sum(self.x_w * fy_c))
        mz_from_fx = float(np.sum(-self.y_w * fx_c))
        mz_from_tire = float(np.sum(mz_tire))
        sum_mz = mz_from_fy + mz_from_fx + mz_from_tire

        # Loaded radius & effective rolling radius (WheelDynamics.update_radial_deflection)
        rho_rad = np.minimum(fz / self.p.k_tire_z, 0.5 * self.p.wheel_radius)
        r_load = self.p.wheel_radius - rho_rad
        r_eff = self.p.wheel_radius - rho_rad / 3.0

        # Steady-state wheel torque: T_i = (Fx_w + F_rr) * r_load
        f_rr = self.p.rolling_resistance_coeff * fz * np.tanh(v_wx / 0.5)
        torques = (fx_w + f_rr) * r_load

        # Wheel angular velocity from kappa = (omega * r_eff - v_wx) / max(|v_wx|, v_eps)
        u_den = np.maximum(np.abs(v_wx), self.p.slip_reg_speed)
        omega_w = (v_wx + kappa * u_den) / np.maximum(r_eff, 0.1)

        p_dc_ecu_w = self.compute_hv_power_w(torques, omega_w)
        p_dc_drv_w = self.compute_hv_power_drivetrain_w(torques, omega_w)
        t_max, t_min = self.wheel_motor_limits(omega_w)

        # Direct TV yaw moment contribution from longitudinal forces (including Fx_f * sin(delta) * a_cog)
        mz_tv_long = float(
            np.sum(-self.y_w * (fx_w * cos_d)) + np.sum(self.x_w * (fx_w * sin_d))
        )
        fy_from_fx_sin_delta = float(np.sum(fx_w[:2] * sin_d[:2]))

        return {
            "sum_fx": sum_fx,
            "sum_fy": sum_fy,
            "sum_mz": sum_mz,
            "ax": sum_fx / self.p.mass,
            "ay": sum_fy / self.p.mass,
            "fz": fz,
            "phi": phi,
            "delta_w": delta_w,
            "alpha": alpha,
            "kappa": kappa,
            "fx_w": fx_w,
            "fy_w": fy_w,
            "fx_c": fx_c,
            "fy_c": fy_c,
            "mz_tire": mz_tire,
            "mu_y_eff": mu_y_eff,
            "torques": torques,
            "omega_w": omega_w,
            "t_max": t_max,
            "t_min": t_min,
            "p_dc_ecu_w": p_dc_ecu_w,
            "p_dc_drv_w": p_dc_drv_w,
            "f_drag_x": f_drag_x,
            "f_drag_y": f_drag_y,
            "f_down_f": f_down_f,
            "f_down_r": f_down_r,
            "f_rr_sum": float(np.sum(f_rr * cos_d)),
            "mz_from_fy": mz_from_fy,
            "mz_from_fx": mz_from_fx,
            "mz_from_tire": mz_from_tire,
            "mz_tv_long": mz_tv_long,
            "fy_from_fx_sin_delta": fy_from_fx_sin_delta,
        }


if __name__ == "__main__":
    for prof in ("FSG_REAL_TRACK", "TTC_LAB_PEAK"):
        m = TeR27PhysicsModel(profile=prof)
        res = m.evaluate_state_from_kappa(
            vx=15.0,
            vy=-0.3,
            r=1.0,
            delta=math.radians(8.0),
            kappa=np.array([0.01, 0.03, 0.01, 0.04]),
            ax_est=0.0,
            ay_est=15.0,
        )
        print(
            f"[{prof}] LMUX={m.lmux:.4f} LMUY={m.lmuy:.4f} CL_eff={m.p.cl_a * m.env_aero_cl_scale:.2f} | "
            f"Fz={np.round(res['fz'], 1)} | ay={res['ay']/9.81:.3f}g | Mz={res['sum_mz']:.1f} Nm | "
            f"P_dc={res['p_dc_ecu_w']/1e3:.2f} kW",
            flush=True,
        )
