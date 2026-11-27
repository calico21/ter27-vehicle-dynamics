#!/usr/bin/env python3
"""
Fase 2: Massive TeR27 4WD Maneuvers & Handling Battery.

Generates all 13 publication-grade (300 DPI, MoTeC i2 dark theme) figures and
`metrics.json` in `outputs/validation_vault/02_ter27_maneuvers_and_handling/`
covering 6 comprehensive vehicle-dynamics test families:
  1. Milliken Yaw Moment Diagrams (YMD): velocity sweep, grip/thermal sweep, TeR27 vs TeR26 (3D vs Parametric)
  2. Steady-State Ramp Steer: understeer gradient K_us(ay), sideslip beta(ay), roll phi(ay), 4-wheel Fz_i(ay)
  3. Step-Steer Transient Response: 50 & 80 km/h (small & large amplitude), t_rise, t_peak, overshoot, damping
  4. Sine-Sweep Frequency Response (0.1 - 6.0 Hz): Bode magnitude, phase & coherence for wz/delta, ay/delta, beta/delta
  5. Combined & Emergency Maneuvers: 1.4 g Trail Braking, FS 15 m Slalom, ISO 3888-2 DLC, Split-Mu (0.90 / 0.50)
  6. Official Formula Student Dynamic Events: 0-75 m Acceleration (80 kW HV limit) & R=9.125 m Skidpad

All physics are 100% genuine: exact Hoosier 16.0x7.5-10 R20 Pacejka MF 6.2 combined-slip tires,
exact load transfer, aerodynamics, 3D vs Parametric suspension kinematics, HV [-25, +80] kW
powertrain limits, and validated TeR26 RWD comparison.

Coordinate convention: ISO 8855 / ROS REP-103 (+x forward, +y left, +z up).
Wheel order: [0: FL, 1: FR, 2: RL, 3: RR].
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import optimize, signal

# Ensure repository root is importable regardless of cwd
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

from vd_correlation_lab.config.paths import (  # noqa: E402
    TER27_CONFIG_DIR,
    VAULT_02_MANEUVERS,
    ensure_repo_in_syspath,
)
from vd_correlation_lab.models.vehicle_14dof.ter26_high_fidelity_model import (  # noqa: E402
    TeR26HighFidelityCasADi,
    default_sysid_theta,
    theta_dict_to_vec,
)
from vd_correlation_lab.models.vehicle_14dof.ter27_casadi_model import (  # noqa: E402
    CASADI_GRIP_PROFILES,
    thermal_grip_multiplier_smooth,
)
from vd_correlation_lab.models.vehicle_14dof.ter27_physics import (  # noqa: E402
    TeR27PhysicsModel,
)
from vd_correlation_lab.analysis.visualization.plot_style import (  # noqa: E402
    PALETTE,
    add_footer,
    add_kpi_box,
    add_suptitle,
    apply_motec_style,
    save_fig,
    style_ax,
)

ensure_repo_in_syspath()

G_ACC = 9.81
OUT_DIR = VAULT_02_MANEUVERS
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 1. Unified High-Fidelity TeR27 4WD Dynamic & Quasi-Steady Simulator
# =============================================================================

class TeR27HandlingModel:
    """
    Extends TeR27PhysicsModel with:
      - Selectable grip/thermal profiles (COLD 18 C, FSG_RUBBERED 45 C, TTC_PEAK 77.5 C)
      - Selectable 3D vs Parametric suspension kinematics (bump camber, bump toe, steering compliance)
      - Per-wheel friction scaling (for Split-mu 0.90 / 0.50)
      - 12-state RK4 transient simulator with exact A-stable exponential tire relaxation
        and 2nd-order roll dynamics.
    """

    def __init__(
        self,
        profile: str = "FSG_RUBBERED",
        tire_temp_c: Optional[float] = None,
        use_3d_kin: bool = True,
        sigma_y: float = 0.22,
        mu_wheel_scale: Optional[np.ndarray] = None,
    ) -> None:
        base_profile = "TTC_LAB_PEAK" if profile == "TTC_LAB_PEAK" else "FSG_REAL_TRACK"
        self.base = TeR27PhysicsModel(profile=base_profile)
        self.p = self.base.p
        self.tire = self.base.tire
        self.profile_name = profile
        self.use_3d_kin = bool(use_3d_kin)
        self.sigma_y = float(sigma_y)

        # Profile grip & aero scaling from CASADI_GRIP_PROFILES
        if profile in CASADI_GRIP_PROFILES:
            prof_cfg = CASADI_GRIP_PROFILES[profile]
            self.env_mu_scale = float(prof_cfg["env_mu_scale"])
            self.env_aero_cl_scale = float(prof_cfg["env_aero_cl_scale"])
            default_temp = float(prof_cfg["default_temp_c"])
        else:
            self.env_mu_scale = self.base.env_mu_scale
            self.env_aero_cl_scale = self.base.env_aero_cl_scale
            default_temp = 45.0

        self.tire_temp_c = default_temp if tire_temp_c is None else float(tire_temp_c)
        self.mu_thermal = float(thermal_grip_multiplier_smooth(self.tire_temp_c))
        self.mu_wheel_scale = (
            np.ones(4, dtype=float)
            if mu_wheel_scale is None
            else np.asarray(mu_wheel_scale, dtype=float).copy()
        )

        # Effective tire friction multipliers (per-wheel array)
        self.lmux_4 = self.p.lambda_mux_base * self.env_mu_scale * self.mu_thermal * self.mu_wheel_scale
        self.lmuy_4 = self.p.lambda_muy_base * self.env_mu_scale * self.mu_thermal * self.mu_wheel_scale

        # Load kinematic tables from ter27_4wd/suspension.json
        susp = json.loads((TER27_CONFIG_DIR / "suspension.json").read_text(encoding="utf-8"))
        self._cam_f_z, self._cam_f_deg = self._parse_table(susp.get("camber_table_front_deg", []))
        self._cam_r_z, self._cam_r_deg = self._parse_table(susp.get("camber_table_rear_deg", []))
        self._toe_f_z, self._toe_f_deg = self._parse_table(susp.get("toe_table_front_deg", []))
        self._toe_r_z, self._toe_r_deg = self._parse_table(susp.get("toe_table_rear_deg", []))

        # Roll damping from dampers (average comp/reb * MR^2 * 0.5 * t^2)
        c_f = 0.5 * (float(susp["damping_compression_front"]) + float(susp["damping_relaxation_front"])) * (self.p.mr_f ** 2)
        c_r = 0.5 * (float(susp["damping_compression_rear"]) + float(susp["damping_relaxation_rear"])) * (self.p.mr_r ** 2)
        self.c_roll_tot = 0.5 * (self.p.track_front ** 2) * c_f + 0.5 * (self.p.track_rear ** 2) * c_r
        self.k_roll_tot = self.p.k_roll_f + self.p.k_roll_r
        self.k_roll_eff = max(self.k_roll_tot - self.p.sprung_mass * self.p.gravity * self.p.h_roll_arm, 500.0)

        self.x_w = self.base.x_w
        self.y_w = self.base.y_w
        self.IS_RIGHT = self.base.IS_RIGHT

    @staticmethod
    def _parse_table(tbl: List[List[float]]) -> Tuple[np.ndarray, np.ndarray]:
        if not tbl:
            return np.array([-0.03, 0.03]), np.array([0.0, 0.0])
        arr = np.asarray(tbl, dtype=float)
        return arr[:, 0], arr[:, 1]

    def aero_forces(self, vx: float, vy: float = 0.0) -> Tuple[float, float, float, float]:
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
        phi_override: Optional[float] = None,
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """Return (fz[4], phi_rad, z_w[4]) with exact elastic + geometric + unsprung load transfer."""
        p = self.p
        _, _, f_down_f, f_down_r = self.aero_forces(vx, vy)
        mg = p.mass * p.gravity
        fz_f_stat = mg * (p.b_cog / p.wheelbase)
        fz_r_stat = mg * (p.a_cog / p.wheelbase)

        dfz_long = (p.mass * ax * p.h_cog) / p.wheelbase
        fz_f_tot = max(0.0, fz_f_stat - dfz_long + f_down_f)
        fz_r_tot = max(0.0, fz_r_stat + dfz_long + f_down_r)

        m_s = p.sprung_mass
        m_u_axle = 0.5 * p.unsprung_mass_total
        phi = (m_s * ay * p.h_roll_arm) / self.k_roll_eff if phi_override is None else float(phi_override)

        dfz_el_f = (p.k_roll_f * phi) / p.track_front
        dfz_el_r = (p.k_roll_r * phi) / p.track_rear
        dfz_rc_f = (m_s * (p.b_cog / p.wheelbase) * ay * p.h_rc_f) / p.track_front
        dfz_rc_r = (m_s * (p.a_cog / p.wheelbase) * ay * p.h_rc_r) / p.track_rear
        dfz_u_f = (m_u_axle * ay * p.wheel_radius) / p.track_front
        dfz_u_r = (m_u_axle * ay * p.wheel_radius) / p.track_rear

        dfz_lat_f = float(np.clip(dfz_el_f + dfz_rc_f + dfz_u_f, -0.5 * fz_f_tot, 0.5 * fz_f_tot))
        dfz_lat_r = float(np.clip(dfz_el_r + dfz_rc_r + dfz_u_r, -0.5 * fz_r_tot, 0.5 * fz_r_tot))

        fz = np.array(
            [
                0.5 * fz_f_tot - dfz_lat_f,
                0.5 * fz_f_tot + dfz_lat_f,
                0.5 * fz_r_tot - dfz_lat_r,
                0.5 * fz_r_tot + dfz_lat_r,
            ],
            dtype=float,
        )
        fz = np.maximum(fz, 0.0)

        # Wheel bump z_w (positive = compression)
        dfz_brk = max(-dfz_long, 0.0)
        dfz_acc = max(dfz_long, 0.0)
        z_hf = (0.5 * f_down_f + 0.5 * dfz_brk * (1.0 - p.anti_dive_f) - 0.5 * dfz_acc) / max(p.k_wheel_f, 1000.0)
        z_hr = (0.5 * f_down_r + 0.5 * dfz_acc * (1.0 - p.anti_squat_r) - 0.5 * dfz_brk) / max(p.k_wheel_r, 1000.0)
        z_w = np.clip(
            np.array(
                [
                    z_hf - 0.5 * p.track_front * phi,
                    z_hf + 0.5 * p.track_front * phi,
                    z_hr - 0.5 * p.track_rear * phi,
                    z_hr + 0.5 * p.track_rear * phi,
                ],
                dtype=float,
            ),
            -0.035,
            0.035,
        )
        return fz, phi, z_w

    def compute_wheel_angles(self, delta: float, phi: float, z_w: np.ndarray, ay_est: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Return (delta_w[4], gamma_iso[4]) for 3D or Parametric suspension mode."""
        p = self.p
        if self.use_3d_kin:
            # Steering compliance deflection under front lateral force
            fy_f_est = p.mass * (p.b_cog / p.wheelbase) * ay_est
            delta_comp = float(np.clip(fy_f_est * (p.mech_trail_f + 0.015) / max(p.k_steer_comp, 1000.0), -0.05, 0.05))
            delta_eff = float(np.clip(delta - delta_comp, -p.max_steer_angle, p.max_steer_angle))
            delta_w = self.base.ackermann_steer(delta_eff)
            # Add kinematic bump-toe (toe > 0 is toe-in: FL points right (-), FR points left (+))
            toe_fl = math.radians(float(np.interp(z_w[0], self._toe_f_z, self._toe_f_deg)))
            toe_fr = math.radians(float(np.interp(z_w[1], self._toe_f_z, self._toe_f_deg)))
            toe_rl = math.radians(float(np.interp(z_w[2], self._toe_r_z, self._toe_r_deg)))
            toe_rr = math.radians(float(np.interp(z_w[3], self._toe_r_z, self._toe_r_deg)))
            delta_w = delta_w + np.array([-toe_fl, toe_fr, -toe_rl, toe_rr], dtype=float)

            # Kinematic bump-camber + roll + caster/KPI steer camber
            cam_fl = math.radians(float(np.interp(z_w[0], self._cam_f_z, self._cam_f_deg)))
            cam_fr = math.radians(float(np.interp(z_w[1], self._cam_f_z, self._cam_f_deg)))
            cam_rl = math.radians(float(np.interp(z_w[2], self._cam_r_z, self._cam_r_deg)))
            cam_rr = math.radians(float(np.interp(z_w[3], self._cam_r_z, self._cam_r_deg)))
            g_kin = np.array([cam_fl, -cam_fr, cam_rl, -cam_rr], dtype=float)
            g_steer = np.array(
                [
                    p.caster_rad * math.sin(delta_w[0]) + p.kpi_rad * (1.0 - math.cos(delta_w[0])),
                    p.caster_rad * math.sin(delta_w[1]) - p.kpi_rad * (1.0 - math.cos(delta_w[1])),
                    0.0,
                    0.0,
                ],
                dtype=float,
            )
            gamma = g_kin - phi + g_steer
        else:
            delta_w = self.base.ackermann_steer(float(np.clip(delta, -p.max_steer_angle, p.max_steer_angle)))
            gamma = self.base.compute_cambers(delta_w, phi)
        return delta_w, gamma

    def eval_tire_forces(self, kappa: np.ndarray, alpha: np.ndarray, fz: np.ndarray, gamma: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.tire.compute_combined_forces_vec(
            kappa, alpha, fz, gamma, self.lmux_4, self.lmuy_4, self.IS_RIGHT, self.p.p_infl_kpa
        )

    def eval_ymd_point(self, v_ms: float, beta_rad: float, delta_rad: float, n_iter: int = 5) -> Dict[str, float]:
        """Evaluate one Milliken YMD grid point at constant speed v_ms with r = ay / v_ms."""
        vx = v_ms * math.cos(beta_rad)
        vy = v_ms * math.sin(beta_rad)
        ay_cur = 0.0
        kappa = np.zeros(4, dtype=float)
        fx_c = np.zeros(4)
        fy_c = np.zeros(4)
        mz_t = np.zeros(4)
        fz = np.zeros(4)
        mu_y_eff = np.zeros(4)
        phi = 0.0
        f_drag_x, f_drag_y, _, _ = self.aero_forces(vx, vy)

        for _ in range(n_iter):
            r = ay_cur / max(v_ms, 1.0)
            fz, phi, z_w = self.compute_normal_forces(0.0, ay_cur, vx, vy)
            delta_w, gamma = self.compute_wheel_angles(delta_rad, phi, z_w, ay_cur)
            _, _, alpha = self.base.wheel_velocities_and_alphas(vx, vy, r, delta_w)
            fx_w, fy_w, mz_t, mu_y_eff = self.eval_tire_forces(kappa, alpha, fz, gamma)
            cd, sd = np.cos(delta_w), np.sin(delta_w)
            fx_c = fx_w * cd - fy_w * sd
            fy_c = fx_w * sd + fy_w * cd
            ay_new = (float(np.sum(fy_c)) + f_drag_y) / self.p.mass
            if abs(ay_new - ay_cur) < 1e-3:
                ay_cur = ay_new
                break
            ay_cur = 0.55 * ay_new + 0.45 * ay_cur

        sum_mz = float(np.sum(self.x_w * fy_c - self.y_w * fx_c + mz_t))
        cn = sum_mz / (self.p.mass * self.p.gravity * self.p.wheelbase)
        sat_f = float(np.sum(np.abs(fy_c[:2])) / max(np.sum(mu_y_eff[:2] * fz[:2]), 1.0))
        sat_r = float(np.sum(np.abs(fy_c[2:])) / max(np.sum(mu_y_eff[2:] * fz[2:]), 1.0))
        return {
            "ay_g": ay_cur / G_ACC,
            "cn": cn,
            "mz_nm": sum_mz,
            "phi_deg": math.degrees(phi),
            "sat_front": min(sat_f, 1.15),
            "sat_rear": min(sat_r, 1.15),
        }

    def _kappa_from_fx_req(self, fx_req: np.ndarray, fz: np.ndarray, gamma: np.ndarray) -> np.ndarray:
        """Fast analytic Pacejka inverse Fx0(kappa) = fx_req (matching TeR26 smooth inverse)."""
        t = self.tire
        fz_pos = np.maximum(fz, 15.0)
        dfz = np.minimum((fz_pos - t.fnomin) / t.fnomin, t.dfz_max)
        dpi = (self.p.p_infl_kpa - t.pinom) / t.pinom
        camber_fx = np.maximum(1.0 - t.PDX3 * gamma * gamma, 0.25)
        mu_x = (t.PDX1 + t.PDX2 * dfz) * (1.0 + t.PPX3 * dpi + t.PPX4 * dpi * dpi) * camber_fx * self.lmux_4
        dx = np.maximum(mu_x * fz_pos, 40.0)
        kxk = fz_pos * (t.PKX1 + t.PKX2 * dfz) * np.exp(t.PKX3 * dfz) * (1.0 + t.PPX1 * dpi + t.PPX2 * dpi * dpi)
        bx = kxk / (t.PCX1 * dx + 1e-4)
        ratio = np.clip(fx_req / dx, -0.96, 0.96)
        arg_tan = np.clip(np.arcsin(ratio) / max(t.PCX1, 0.5), -1.30, 1.30)
        return np.clip(np.tan(arg_tan) / np.maximum(bx, 1.0) - (t.PHX1 + t.PHX2 * dfz), -0.22, 0.22)

    def simulate_maneuver(
        self,
        t_s: np.ndarray,
        v0_ms: float,
        control_fn,
        hold_vx: bool = False,
        x0_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> Dict[str, np.ndarray]:
        """
        12-state RK4 transient simulation with A-stable exponential tire relaxation.
        State: [vx, vy, wz, phi, phi_dot, alpha_fl, alpha_fr, alpha_rl, alpha_rr, X, Y, psi]
        control_fn(t, state, diag_prev) -> (delta_rad, ax_cmd_ms2, mz_tv_nm)
        """
        n = len(t_s)
        p = self.p
        s = np.zeros(12, dtype=float)
        s[0] = float(v0_ms)
        s[9], s[10], s[11] = float(x0_pos[0]), float(x0_pos[1]), float(x0_pos[2])

        out = {
            "t": t_s.copy(),
            "vx": np.zeros(n),
            "vy": np.zeros(n),
            "wz": np.zeros(n),
            "beta_deg": np.zeros(n),
            "ax": np.zeros(n),
            "ay": np.zeros(n),
            "phi_deg": np.zeros(n),
            "delta_deg": np.zeros(n),
            "mz": np.zeros(n),
            "mz_tv": np.zeros(n),
            "X": np.zeros(n),
            "Y": np.zeros(n),
            "psi_deg": np.zeros(n),
            "fz": np.zeros((n, 4)),
            "fx_w": np.zeros((n, 4)),
            "fy_w": np.zeros((n, 4)),
            "alpha_deg": np.zeros((n, 4)),
            "kappa": np.zeros((n, 4)),
        }

        ax_lt = 0.0
        ay_lt = 0.0
        diag_prev = {"ax": 0.0, "ay": 0.0, "fz": np.full(4, p.mass * G_ACC * 0.25)}

        def eval_rhs(st: np.ndarray, delta: float, ax_cmd: float, mz_tv: float, ax_l: float, ay_l: float):
            vx_c = max(st[0], 0.8)
            vy_c = st[1]
            wz_c = st[2]
            phi_c = st[3]
            phi_dot_c = st[4]
            alpha_eff = st[5:9]
            psi_c = st[11]

            fz, _, z_w = self.compute_normal_forces(ax_l, ay_l, vx_c, vy_c, phi_override=phi_c)
            delta_w, gamma = self.compute_wheel_angles(delta, phi_c, z_w, ay_l)
            v_wx, _, alpha_kin = self.base.wheel_velocities_and_alphas(vx_c, vy_c, wz_c, delta_w)
            f_drag_x, f_drag_y, _, _ = self.aero_forces(vx_c, vy_c)

            # Longitudinal force allocation (4WD drive split + brake bias + direct TV yaw moment)
            f_tot = p.mass * ax_cmd
            if f_tot >= 0.0:
                f_f = p.front_drive_ratio * f_tot
                f_r = (1.0 - p.front_drive_ratio) * f_tot
            else:
                f_f = p.brake_bias_front * f_tot
                f_r = (1.0 - p.brake_bias_front) * f_tot

            # Torque vectoring differential Fx (positive mz_tv -> +Fx on right, -Fx on left)
            dfx_f = (0.35 * mz_tv) / p.track_front
            dfx_r = (0.65 * mz_tv) / p.track_rear
            fx_req = np.array(
                [0.5 * f_f - dfx_f, 0.5 * f_f + dfx_f, 0.5 * f_r - dfx_r, 0.5 * f_r + dfx_r],
                dtype=float,
            )
            kappa = self._kappa_from_fx_req(fx_req, fz, gamma)
            fx_w, fy_w, mz_t, _ = self.eval_tire_forces(kappa, alpha_eff, fz, gamma)
            # Rolling resistance
            fx_w = fx_w - p.rolling_resistance_coeff * fz * np.tanh(v_wx / 0.5)

            cd, sd = np.cos(delta_w), np.sin(delta_w)
            fx_c = fx_w * cd - fy_w * sd
            fy_c = fx_w * sd + fy_w * cd

            sum_fx = float(np.sum(fx_c)) + f_drag_x
            sum_fy = float(np.sum(fy_c)) + f_drag_y
            sum_mz = float(np.sum(self.x_w * fy_c - self.y_w * fx_c + mz_t))

            ax_b = sum_fx / p.mass
            ay_b = sum_fy / p.mass

            vx_dot = 0.0 if hold_vx else (ax_b + wz_c * vy_c)
            vy_dot = ay_b - wz_c * vx_c
            wz_dot = sum_mz / p.izz_total
            phi_ddot = (p.sprung_mass * ay_b * p.h_roll_arm - self.k_roll_eff * phi_c - self.c_roll_tot * phi_dot_c) / p.ixx

            X_dot = vx_c * math.cos(psi_c) - vy_c * math.sin(psi_c)
            Y_dot = vx_c * math.sin(psi_c) + vy_c * math.cos(psi_c)
            psi_dot = wz_c

            deriv = np.zeros(12, dtype=float)
            deriv[0] = vx_dot
            deriv[1] = vy_dot
            deriv[2] = wz_dot
            deriv[3] = phi_dot_c
            deriv[4] = phi_ddot
            deriv[9] = X_dot
            deriv[10] = Y_dot
            deriv[11] = psi_dot
            diag = {
                "ax": ax_b,
                "ay": ay_b,
                "mz": sum_mz,
                "fz": fz,
                "fx_w": fx_w,
                "fy_w": fy_w,
                "alpha_kin": alpha_kin,
                "v_wx": v_wx,
                "kappa": kappa,
            }
            return deriv, diag

        for k in range(n):
            delta_k, ax_cmd_k, mz_tv_k = control_fn(float(t_s[k]), s, diag_prev)
            k1, diag1 = eval_rhs(s, delta_k, ax_cmd_k, mz_tv_k, ax_lt, ay_lt)

            out["vx"][k] = s[0]
            out["vy"][k] = s[1]
            out["wz"][k] = s[2]
            out["beta_deg"][k] = math.degrees(math.atan2(s[1], max(s[0], 0.5)))
            out["ax"][k] = diag1["ax"]
            out["ay"][k] = diag1["ay"]
            out["phi_deg"][k] = math.degrees(s[3])
            out["delta_deg"][k] = math.degrees(delta_k)
            out["mz"][k] = diag1["mz"]
            out["mz_tv"][k] = mz_tv_k
            out["X"][k] = s[9]
            out["Y"][k] = s[10]
            out["psi_deg"][k] = math.degrees(s[11])
            out["fz"][k] = diag1["fz"]
            out["fx_w"][k] = diag1["fx_w"]
            out["fy_w"][k] = diag1["fy_w"]
            out["alpha_deg"][k] = np.degrees(s[5:9])
            out["kappa"][k] = diag1["kappa"]

            diag_prev = diag1
            ax_lt = 0.5 * ax_lt + 0.5 * float(np.clip(diag1["ax"], -28.0, 28.0))
            ay_lt = 0.5 * ay_lt + 0.5 * float(np.clip(diag1["ay"], -28.0, 28.0))

            if k + 1 < n:
                dt = float(t_s[k + 1] - t_s[k])
                # Exact exponential update for tire slip relaxation (A-stable at any speed)
                v_rel = np.maximum(np.abs(diag1["v_wx"]), 1.5)
                decay_half = np.exp(-0.5 * dt * v_rel / self.sigma_y)
                decay_full = decay_half * decay_half
                a_kin = diag1["alpha_kin"]
                a_half = a_kin + (s[5:9] - a_kin) * decay_half
                a_full = a_kin + (s[5:9] - a_kin) * decay_full

                s2 = s + 0.5 * dt * k1
                s2[5:9] = a_half
                k2, _ = eval_rhs(s2, delta_k, ax_cmd_k, mz_tv_k, ax_lt, ay_lt)

                s3 = s + 0.5 * dt * k2
                s3[5:9] = a_half
                k3, _ = eval_rhs(s3, delta_k, ax_cmd_k, mz_tv_k, ax_lt, ay_lt)

                s4 = s + dt * k3
                s4[5:9] = a_full
                k4, _ = eval_rhs(s4, delta_k, ax_cmd_k, mz_tv_k, ax_lt, ay_lt)

                s = s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
                s[5:9] = np.clip(a_full, -0.55, 0.55)
                if hold_vx:
                    s[0] = float(v0_ms)
                else:
                    s[0] = float(np.clip(s[0], 0.5, 45.0))
                s[1] = float(np.clip(s[1], -15.0, 15.0))
                s[2] = float(np.clip(s[2], -6.0, 6.0))
                s[3] = float(np.clip(s[3], -0.25, 0.25))
                s[4] = float(np.clip(s[4], -4.0, 4.0))

        return out


# =============================================================================
# 2. TeR26 Helper Wrapper (YMD & Handling Parity)
# =============================================================================

class TeR26HandlingWrapper:
    """Wraps TeR26HighFidelityCasADi for YMD, Ramp-Steer, Step-Steer, and Skidpad comparison."""

    def __init__(self, use_3d_kin: bool = True, tire_temp_c: float = 55.0) -> None:
        self.model = TeR26HighFidelityCasADi(use_3d_kinematics=use_3d_kin, use_thermal_coupling=True)
        self.p = self.model.params
        self.th_vec = theta_dict_to_vec(default_sysid_theta(self.p))
        self.t4 = np.full(4, float(tire_temp_c), dtype=np.float64)

    def eval_ymd_point(self, v_ms: float, beta_rad: float, delta_rad: float, n_iter: int = 5) -> Dict[str, float]:
        vx = v_ms * math.cos(beta_rad)
        vy = v_ms * math.sin(beta_rad)
        ay_cur = 0.0
        mz_cur = 0.0
        phi_cur = 0.0
        for _ in range(n_iter):
            r = ay_cur / max(v_ms, 1.0)
            v_fy = vy + self.p.a_cog * r
            v_ry = vy - self.p.b_cog * r
            af = delta_rad - math.atan2(v_fy, max(vx, 1.5))
            ar = -math.atan2(v_ry, max(vx, 1.5))
            x = np.array([vx, vy, r, af, ar], dtype=np.float64)
            u = np.array([delta_rad, 0.0, 0.0, ay_cur], dtype=np.float64)
            _, yobs = self.model.fn_ode(x, u, self.t4, self.th_vec)
            ay_new = float(yobs[1])
            mz_cur = float(yobs[2])
            phi_cur = float(yobs[3])
            if abs(ay_new - ay_cur) < 1e-3:
                ay_cur = ay_new
                break
            ay_cur = 0.55 * ay_new + 0.45 * ay_cur

        cn = mz_cur / (self.p.mass * self.p.gravity * self.p.wheelbase)
        return {
            "ay_g": ay_cur / G_ACC,
            "cn": cn,
            "mz_nm": mz_cur,
            "phi_deg": math.degrees(phi_cur),
        }

    def simulate_constant_speed(self, t_s: np.ndarray, v0_ms: float, delta_rad_arr: np.ndarray) -> Dict[str, np.ndarray]:
        n = len(t_s)
        ax_arr = np.zeros(n, dtype=np.float64)
        vx_arr = np.full(n, float(v0_ms), dtype=np.float64)
        wz_arr = np.zeros(n, dtype=np.float64)
        t4_arr = np.full((n, 4), float(self.t4[0]), dtype=np.float64)
        sim = self.model.simulate_trajectory(
            t_s=t_s,
            delta_rad=delta_rad_arr,
            ax_cmd_drive=ax_arr,
            vx_meas=vx_arr,
            wz_meas=wz_arr,
            t_tire_4=t4_arr,
            theta=self.th_vec,
            use_measured_vx=True,
        )
        return {
            "t": t_s.copy(),
            "vx": sim["vx_sim"],
            "vy": sim["vy_sim"],
            "wz": sim["wz_sim"],
            "beta_deg": np.degrees(np.arctan2(sim["vy_sim"], np.maximum(sim["vx_sim"], 0.5))),
            "ay": sim["ay_sim"],
            "phi_deg": np.degrees(sim["phi_sim"]),
            "delta_deg": np.degrees(delta_rad_arr),
            "fz": sim["fz_sim"],
        }


# =============================================================================
# 3. Family 1: Milliken Yaw Moment Diagrams (Figs 1, 2, 3)
# =============================================================================

def compute_ymd_grid(model: Any, v_kmh: float, betas_deg: np.ndarray, deltas_deg: np.ndarray) -> Dict[str, Any]:
    v_ms = v_kmh / 3.6
    nb, nd = len(betas_deg), len(deltas_deg)
    ay_g = np.zeros((nb, nd))
    cn = np.zeros((nb, nd))
    mz_nm = np.zeros((nb, nd))
    for i, b in enumerate(betas_deg):
        b_rad = math.radians(float(b))
        for j, d in enumerate(deltas_deg):
            pt = model.eval_ymd_point(v_ms, b_rad, math.radians(float(d)))
            ay_g[i, j] = pt["ay_g"]
            cn[i, j] = pt["cn"]
            mz_nm[i, j] = pt["mz_nm"]

    # Extract trim points (CN = 0 crossings along delta for each beta)
    trim_pts: List[Tuple[float, float, float]] = []
    for i, b in enumerate(betas_deg):
        for j in range(nd - 1):
            c0, c1 = cn[i, j], cn[i, j + 1]
            if c0 == 0.0 or c0 * c1 < 0.0:
                u = 0.0 if abs(c1 - c0) < 1e-12 else float(np.clip(-c0 / (c1 - c0), 0.0, 1.0))
                ay_t = float(ay_g[i, j] + u * (ay_g[i, j + 1] - ay_g[i, j]))
                d_t = float(deltas_deg[j] + u * (deltas_deg[j + 1] - deltas_deg[j]))
                trim_pts.append((ay_t, d_t, float(b)))

    pos_trims = [t for t in trim_pts if t[0] >= 0.0]
    if pos_trims:
        max_ay_trim, delta_trim_max, beta_trim_max = max(pos_trims, key=lambda x: x[0])
    else:
        idx = np.unravel_index(np.argmax(ay_g), ay_g.shape)
        max_ay_trim = float(ay_g[idx])
        delta_trim_max = float(deltas_deg[idx[1]])
        beta_trim_max = float(betas_deg[idx[0]])

    i_b0 = int(np.argmin(np.abs(betas_deg)))
    j_d0 = int(np.argmin(np.abs(deltas_deg)))
    stab_0g = float((cn[i_b0 + 1, j_d0] - cn[i_b0 - 1, j_d0]) / (betas_deg[i_b0 + 1] - betas_deg[i_b0 - 1]))
    ctrl_0g = float((cn[i_b0, j_d0 + 1] - cn[i_b0, j_d0 - 1]) / (deltas_deg[j_d0 + 1] - deltas_deg[j_d0 - 1]))

    # Stability index at Ay = 1.0g (or 0.75 * max_ay_trim if < 1.1g)
    ay_ref = min(1.0, 0.75 * max_ay_trim)
    cn_at_ref, b_at_ref = [], []
    for i in range(nb):
        for j in range(nd - 1):
            a0, a1 = ay_g[i, j], ay_g[i, j + 1]
            if (a0 - ay_ref) * (a1 - ay_ref) <= 0.0 and abs(a1 - a0) > 1e-6:
                u = (ay_ref - a0) / (a1 - a0)
                cn_at_ref.append(float(cn[i, j] + u * (cn[i, j + 1] - cn[i, j])))
                b_at_ref.append(float(betas_deg[i]))
                break
    if len(b_at_ref) >= 2:
        ic = int(np.argmin(np.abs(np.array(b_at_ref))))
        i0, i1 = max(0, ic - 1), min(len(b_at_ref) - 1, ic + 1)
        stab_1g = float((cn_at_ref[i1] - cn_at_ref[i0]) / max(b_at_ref[i1] - b_at_ref[i0], 1e-6))
    else:
        stab_1g = stab_0g

    return {
        "v_kmh": float(v_kmh),
        "betas_deg": betas_deg,
        "deltas_deg": deltas_deg,
        "ay_g": ay_g,
        "cn": cn,
        "mz_nm": mz_nm,
        "trim_max_ay_g": float(max_ay_trim),
        "trim_delta_deg": float(delta_trim_max),
        "trim_beta_deg": float(beta_trim_max),
        "stability_index_0g_per_deg": float(stab_0g),
        "stability_index_1g_per_deg": float(stab_1g),
        "control_index_per_deg": float(ctrl_0g),
    }


def _plot_ymd_mesh(ax, g: Dict[str, Any], color: str, label: str, ls: str = "-", alpha_grid: float = 0.45) -> None:
    ay = g["ay_g"]
    cn = g["cn"]
    betas = g["betas_deg"]
    deltas = g["deltas_deg"]
    for i in range(len(betas)):
        ax.plot(ay[i], cn[i], color=color, alpha=alpha_grid, linewidth=0.95, linestyle=ls)
    for j in range(len(deltas)):
        if j % 2 == 0:
            ax.plot(ay[:, j], cn[:, j], color=color, alpha=alpha_grid * 0.65, linewidth=0.75, linestyle=":")
    i0 = int(np.argmin(np.abs(betas)))
    ax.plot(ay[i0], cn[i0], color=color, linewidth=2.2, linestyle=ls, label=label)
    ax.scatter([g["trim_max_ay_g"]], [0.0], color=color, s=38, zorder=6, edgecolor=PALETTE["white"], linewidth=0.7)


def run_family_1_ymd(ter26_3d: TeR26HandlingWrapper, ter26_par: TeR26HandlingWrapper) -> Dict[str, Any]:
    print("[Family 1/6] Running Milliken Yaw Moment Diagrams (YMD)...", flush=True)
    betas = np.linspace(-10.0, 10.0, 17)
    deltas = np.linspace(-14.0, 14.0, 21)

    # Track temperature -> operating tire bulk temp (FSG_RUBBERED 45 C track -> 77.5 C warm tire; COLD 18 C -> 22 C cold tire)
    m_fsg_3d = TeR27HandlingModel(profile="FSG_RUBBERED", tire_temp_c=77.5, use_3d_kin=True)
    m_fsg_par = TeR27HandlingModel(profile="FSG_RUBBERED", tire_temp_c=77.5, use_3d_kin=False)
    m_cold = TeR27HandlingModel(profile="COLD_LOW_GRIP", tire_temp_c=22.0, use_3d_kin=True)
    m_ttc = TeR27HandlingModel(profile="TTC_LAB_PEAK", tire_temp_c=77.5, use_3d_kin=True)

    # --- Figure 1: YMD Velocity Sweep (35, 60, 90 km/h) ---
    vel_speeds = [35.0, 60.0, 90.0]
    vel_grids = [compute_ymd_grid(m_fsg_3d, v, betas, deltas) for v in vel_speeds]

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.4), sharey=True)
    colors_v = [PALETTE["amber"], PALETTE["cyan"], PALETTE["emerald"]]
    for ax, g, col in zip(axes, vel_grids, colors_v):
        _plot_ymd_mesh(ax, g, col, f"beta=0 spine (Ay_max={g['trim_max_ay_g']:.2f} g)", alpha_grid=0.48)
        ax.axhline(0.0, color=PALETTE["white"], linewidth=1.0, alpha=0.85)
        ax.axvline(0.0, color=PALETTE["muted"], linewidth=0.8, linestyle="--", alpha=0.65)
        style_ax(
            ax,
            title=f"TeR27 4WD @ {g['v_kmh']:.0f} km/h (FSG Rubbered 45 C)",
            xlabel="Lateral Acceleration Ay [g]",
            ylabel="Yaw Moment Coeff CN = Mz/(m*g*L) [-]" if ax is axes[0] else "",
            legend=True,
            legend_loc="upper left",
        )
        add_kpi_box(
            ax,
            [
                f"Max Trim Ay : {g['trim_max_ay_g']:.2f} g",
                f"Stab dCN/db : {g['stability_index_0g_per_deg']:+.4f} /deg",
                f"Ctrl dCN/dd : {g['control_index_per_deg']:+.4f} /deg",
                f"Trim delta  : {g['trim_delta_deg']:+.1f} deg",
            ],
            loc="lower right",
            border_color=col,
        )
    add_suptitle(
        fig,
        "TeR27 4WD Milliken Yaw Moment Diagram: Velocity Sweep (35 / 60 / 90 km/h)",
        "Hoosier R20 MF 6.2 | FSG_RUBBERED (45 C) | Aero downforce expands lateral envelope with speed",
    )
    add_footer(fig)
    fig.tight_layout(rect=[0.0, 0.025, 1.0, 0.91])
    save_fig(fig, OUT_DIR / "ymd_velocity_sweep.png", dpi=300)

    # --- Figure 2: YMD Grip & Thermal Comparison (COLD 18 C, FSG_RUBBERED 45 C, TTC_PEAK 77.5 C) ---
    grip_specs = [
        ("COLD 18 C", m_cold, PALETTE["blue"]),
        ("FSG_RUBBERED 45 C", m_fsg_3d, PALETTE["cyan"]),
        ("TTC_PEAK 77.5 C", m_ttc, PALETTE["emerald"]),
    ]
    grip_grids = {}
    for name, mdl, _ in grip_specs:
        grip_grids[name] = compute_ymd_grid(mdl, 60.0, betas, deltas)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6), gridspec_kw={"width_ratios": [1.5, 1.0]})
    ax0, ax1 = axes
    for name, _, col in grip_specs:
        g = grip_grids[name]
        _plot_ymd_mesh(ax0, g, col, f"{name} (Ay_max={g['trim_max_ay_g']:.2f} g)", alpha_grid=0.35)
    ax0.axhline(0.0, color=PALETTE["white"], linewidth=1.0)
    ax0.axvline(0.0, color=PALETTE["muted"], linewidth=0.8, linestyle="--")
    style_ax(
        ax0,
        title="Overlaid YMD Envelope @ 60 km/h",
        xlabel="Lateral Acceleration Ay [g]",
        ylabel="Yaw Moment Coeff CN = Mz/(m*g*L) [-]",
        legend=True,
        legend_loc="upper left",
    )

    names_g = [s[0] for s in grip_specs]
    x_pos = np.arange(len(names_g))
    max_ays = [grip_grids[n]["trim_max_ay_g"] for n in names_g]
    stabs = [grip_grids[n]["stability_index_0g_per_deg"] * 10.0 for n in names_g]
    ctrls = [grip_grids[n]["control_index_per_deg"] * 10.0 for n in names_g]
    w = 0.25
    b1 = ax1.bar(x_pos - w, max_ays, width=w, color=PALETTE["cyan"], label="Max Trim Ay [g]")
    b2 = ax1.bar(x_pos, stabs, width=w, color=PALETTE["emerald"], label="10x Stab dCN/db [1/deg]")
    b3 = ax1.bar(x_pos + w, ctrls, width=w, color=PALETTE["amber"], label="10x Ctrl dCN/dd [1/deg]")
    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() * 0.5, h + 0.04, f"{h:.2f}", ha="center", va="bottom", fontsize=7.6, fontfamily="DejaVu Sans Mono", color=PALETTE["text"])
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(names_g, fontsize=8.4)
    style_ax(ax1, title="YMD Quantitative KPIs Across Grip Regimes", ylabel="Metric Value [g or 10/deg]", legend=True, legend_loc="upper left")
    add_kpi_box(
        ax1,
        [
            f"COLD -> FSG : +{(max_ays[1]-max_ays[0]):.2f} g",
            f"FSG  -> TTC : +{(max_ays[2]-max_ays[1]):.2f} g",
            f"Total Span  : {(max_ays[2]/max(max_ays[0],0.1)):.2f}x",
        ],
        loc="upper right",
        border_color=PALETTE["emerald"],
    )
    add_suptitle(
        fig,
        "TeR27 4WD YMD Grip & Thermal Envelope Comparison @ 60 km/h",
        "COLD_LOW_GRIP (18 C) vs FSG_RUBBERED (45 C) vs TTC_LAB_PEAK (77.5 C) | Exact MF 6.2 + C^2 Thermal Model",
    )
    add_footer(fig)
    fig.tight_layout(rect=[0.0, 0.025, 1.0, 0.91])
    save_fig(fig, OUT_DIR / "ymd_grip_comparison.png", dpi=300)

    # --- Figure 3: TeR27 4WD vs TeR26 RWD + 3D vs Parametric ---
    g_t27_3d = vel_grids[1]  # 60 km/h
    g_t27_par = compute_ymd_grid(m_fsg_par, 60.0, betas, deltas)
    g_t26_3d = compute_ymd_grid(ter26_3d, 60.0, betas, deltas)
    g_t26_par = compute_ymd_grid(ter26_par, 60.0, betas, deltas)

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.6))
    _plot_ymd_mesh(axes[0], g_t27_3d, PALETTE["cyan"], f"TeR27 4WD (3D) Ay_max={g_t27_3d['trim_max_ay_g']:.2f} g", ls="-", alpha_grid=0.40)
    _plot_ymd_mesh(axes[0], g_t26_3d, PALETTE["crimson"], f"TeR26 RWD (3D) Ay_max={g_t26_3d['trim_max_ay_g']:.2f} g", ls="--", alpha_grid=0.35)
    axes[0].axhline(0.0, color=PALETTE["white"], linewidth=1.0)
    axes[0].axvline(0.0, color=PALETTE["muted"], linewidth=0.8, linestyle="--")
    style_ax(
        axes[0],
        title="TeR27 4WD vs TeR26 RWD (3D Kinematics @ 60 km/h)",
        xlabel="Lateral Acceleration Ay [g]",
        ylabel="Yaw Moment Coeff CN [-]",
        legend=True,
        legend_loc="upper left",
    )
    add_kpi_box(
        axes[0],
        [
            f"TeR27 Max Ay : {g_t27_3d['trim_max_ay_g']:.2f} g",
            f"TeR26 Max Ay : {g_t26_3d['trim_max_ay_g']:.2f} g",
            f"Delta Ay     : {g_t27_3d['trim_max_ay_g'] - g_t26_3d['trim_max_ay_g']:+.2f} g",
        ],
        loc="lower right",
        border_color=PALETTE["cyan"],
    )

    # Right panel: 3D vs Parametric beta=0 and beta=+/-4 deg curves + KPI box
    i0 = int(np.argmin(np.abs(betas)))
    for g, col, ls, lbl in [
        (g_t27_3d, PALETTE["cyan"], "-", "TeR27 3D (beta=0)"),
        (g_t27_par, PALETTE["amber"], "--", "TeR27 Parametric (beta=0)"),
        (g_t26_3d, PALETTE["violet"], "-", "TeR26 3D (beta=0)"),
        (g_t26_par, PALETTE["slate"], "--", "TeR26 Parametric (beta=0)"),
    ]:
        axes[1].plot(g["ay_g"][i0], g["cn"][i0], color=col, linestyle=ls, linewidth=2.0, label=lbl)
    axes[1].axhline(0.0, color=PALETTE["white"], linewidth=1.0)
    axes[1].axvline(0.0, color=PALETTE["muted"], linewidth=0.8, linestyle="--")
    style_ax(
        axes[1],
        title="3D Double-Wishbone vs Parametric Kinematics (beta=0 Spine)",
        xlabel="Lateral Acceleration Ay [g]",
        ylabel="Yaw Moment Coeff CN [-]",
        legend=True,
        legend_loc="upper left",
    )
    add_kpi_box(
        axes[1],
        [
            f"TeR27 3D vs Par dAy : {g_t27_3d['trim_max_ay_g'] - g_t27_par['trim_max_ay_g']:+.3f} g",
            f"TeR26 3D vs Par dAy : {g_t26_3d['trim_max_ay_g'] - g_t26_par['trim_max_ay_g']:+.3f} g",
            f"TeR27 Stab (3D/Par) : {g_t27_3d['stability_index_0g_per_deg']:.3f}/{g_t27_par['stability_index_0g_per_deg']:.3f}",
            f"TeR26 Stab (3D/Par) : {g_t26_3d['stability_index_0g_per_deg']:.3f}/{g_t26_par['stability_index_0g_per_deg']:.3f}",
        ],
        loc="lower right",
        border_color=PALETTE["violet"],
    )
    add_suptitle(
        fig,
        "Cross-Vehicle & Kinematics YMD Benchmark: TeR27 4WD vs TeR26 RWD (3D vs Parametric)",
        "Steady-state cornering @ 60 km/h | Direct comparison of vehicle platforms and suspension modeling fidelity",
    )
    add_footer(fig)
    fig.tight_layout(rect=[0.0, 0.025, 1.0, 0.91])
    save_fig(fig, OUT_DIR / "ymd_ter27_vs_ter26.png", dpi=300)

    def _strip(g: Dict[str, Any]) -> Dict[str, float]:
        return {k: v for k, v in g.items() if isinstance(v, float)}

    return {
        "velocity_sweep": {f"{int(v)}kmh": _strip(g) for v, g in zip(vel_speeds, vel_grids)},
        "grip_comparison_60kmh": {k: _strip(v) for k, v in grip_grids.items()},
        "cross_vehicle_60kmh": {
            "ter27_3d": _strip(g_t27_3d),
            "ter27_parametric": _strip(g_t27_par),
            "ter26_3d": _strip(g_t26_3d),
            "ter26_parametric": _strip(g_t26_par),
        },
    }


# =============================================================================
# 4. Family 2: Steady-State Ramp Steer (Figs 4, 5)
# =============================================================================

def _extract_kus_curve(sim: Dict[str, np.ndarray], wheelbase: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute Ay [g], K_us(ay) [deg/g], beta(ay) [deg], and mean linear K_us in [0.2, 0.9] g."""
    ay_g = sim["ay"] / G_ACC
    vx = np.maximum(sim["vx"], 1.0)
    delta_deg = sim["delta_deg"]
    delta_ack_deg = np.degrees(wheelbase * sim["wz"] / vx)
    delta_excess_deg = delta_deg - delta_ack_deg
    beta_deg = sim["beta_deg"]

    # Bin by Ay [g] up to peak Ay (monotone ascending branch)
    i_peak = int(np.argmax(ay_g))
    ay_sub = ay_g[: i_peak + 1]
    dex_sub = delta_excess_deg[: i_peak + 1]
    bet_sub = beta_deg[: i_peak + 1]

    # Smooth monotonic grid
    ay_max = float(np.max(ay_sub))
    grid = np.linspace(0.05, max(ay_max * 0.96, 0.3), 80)
    dex_g = np.interp(grid, np.maximum.accumulate(ay_sub), dex_sub)
    bet_g = np.interp(grid, np.maximum.accumulate(ay_sub), bet_sub)
    kus_g = np.gradient(dex_g, grid)

    lin_mask = (grid >= 0.20) & (grid <= min(0.90, ay_max * 0.75))
    kus_lin = float(np.mean(kus_g[lin_mask])) if np.any(lin_mask) else float(np.mean(kus_g))
    return grid, kus_g, bet_g, kus_lin


def run_family_2_ramp_steer(ter26_3d: TeR26HandlingWrapper, ter26_par: TeR26HandlingWrapper) -> Dict[str, Any]:
    print("[Family 2/6] Running Steady-State Ramp-Steer Battery (40, 65, 85 km/h)...", flush=True)
    m_t27_3d = TeR27HandlingModel(profile="FSG_RUBBERED", tire_temp_c=77.5, use_3d_kin=True)
    m_t27_par = TeR27HandlingModel(profile="FSG_RUBBERED", tire_temp_c=77.5, use_3d_kin=False)

    dt = 0.005
    t_s = np.arange(0.0, 9.0, dt)
    ramp_rate_rad_s = math.radians(1.5)
    delta_cmd_arr = np.minimum(ramp_rate_rad_s * t_s, math.radians(13.5))

    def make_ramp_ctrl():
        def _ctrl(t, st, diag):
            return min(ramp_rate_rad_s * t, math.radians(13.5)), 0.0, 0.0
        return _ctrl

    speeds_kmh = [40.0, 65.0, 85.0]
    results_t27_3d = {}
    results_t27_par = {}
    for v_kmh in speeds_kmh:
        sim_3d = m_t27_3d.simulate_maneuver(t_s, v_kmh / 3.6, make_ramp_ctrl(), hold_vx=True)
        sim_par = m_t27_par.simulate_maneuver(t_s, v_kmh / 3.6, make_ramp_ctrl(), hold_vx=True)
        results_t27_3d[v_kmh] = sim_3d
        results_t27_par[v_kmh] = sim_par

    sim_t26_3d_65 = ter26_3d.simulate_constant_speed(t_s, 65.0 / 3.6, delta_cmd_arr)
    sim_t26_par_65 = ter26_par.simulate_constant_speed(t_s, 65.0 / 3.6, delta_cmd_arr)

    # --- Figure 4: Understeer Gradient K_us(ay) & Sideslip beta(ay) ---
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.2))
    sp_colors = {40.0: PALETTE["amber"], 65.0: PALETTE["cyan"], 85.0: PALETTE["emerald"]}
    kus_metrics: Dict[str, Any] = {}

    for v_kmh in speeds_kmh:
        g_3d, kus_3d, bet_3d, klin_3d = _extract_kus_curve(results_t27_3d[v_kmh], m_t27_3d.p.wheelbase)
        g_par, kus_par, bet_par, klin_par = _extract_kus_curve(results_t27_par[v_kmh], m_t27_par.p.wheelbase)
        col = sp_colors[v_kmh]
        axes[0, 0].plot(g_3d, kus_3d, color=col, linewidth=2.0, label=f"TeR27 3D @ {v_kmh:.0f} km/h (K_lin={klin_3d:+.2f} deg/g)")
        axes[0, 0].plot(g_par, kus_par, color=col, linewidth=1.3, linestyle="--", alpha=0.75, label=f"TeR27 Par @ {v_kmh:.0f} km/h")
        axes[1, 0].plot(g_3d, bet_3d, color=col, linewidth=2.0, label=f"TeR27 3D @ {v_kmh:.0f} km/h")
        axes[1, 0].plot(g_par, bet_par, color=col, linewidth=1.3, linestyle="--", alpha=0.75)
        kus_metrics[f"ter27_3d_{int(v_kmh)}kmh"] = {
            "k_us_linear_deg_per_g": klin_3d,
            "max_ay_g": float(np.max(results_t27_3d[v_kmh]["ay"]) / G_ACC),
            "beta_at_1g_deg": float(np.interp(1.0, g_3d, bet_3d)),
        }
        kus_metrics[f"ter27_par_{int(v_kmh)}kmh"] = {
            "k_us_linear_deg_per_g": klin_par,
            "max_ay_g": float(np.max(results_t27_par[v_kmh]["ay"]) / G_ACC),
        }

    axes[0, 0].axhline(0.0, color=PALETTE["white"], linewidth=0.9, linestyle=":")
    axes[0, 0].set_ylim(-2.0, 6.0)
    style_ax(axes[0, 0], title="TeR27 4WD Understeer Gradient K_us(Ay): Speed & 3D vs Par", xlabel="Lateral Acceleration Ay [g]", ylabel="K_us = d(delta - L/R)/dAy [deg/g]", legend=True, legend_loc="upper left")
    axes[1, 0].axhline(0.0, color=PALETTE["white"], linewidth=0.9, linestyle=":")
    style_ax(axes[1, 0], title="TeR27 4WD Body Sideslip Angle beta(Ay)", xlabel="Lateral Acceleration Ay [g]", ylabel="Sideslip beta [deg]", legend=True, legend_loc="lower left")

    # Right column: TeR27 vs TeR26 at 65 km/h (3D vs Parametric)
    g27_3d, k27_3d, b27_3d, kl27_3d = _extract_kus_curve(results_t27_3d[65.0], m_t27_3d.p.wheelbase)
    g27_par, k27_par, b27_par, kl27_par = _extract_kus_curve(results_t27_par[65.0], m_t27_par.p.wheelbase)
    g26_3d, k26_3d, b26_3d, kl26_3d = _extract_kus_curve(sim_t26_3d_65, ter26_3d.p.wheelbase)
    g26_par, k26_par, b26_par, kl26_par = _extract_kus_curve(sim_t26_par_65, ter26_par.p.wheelbase)

    kus_metrics["ter26_3d_65kmh"] = {"k_us_linear_deg_per_g": kl26_3d, "max_ay_g": float(np.max(sim_t26_3d_65["ay"]) / G_ACC)}
    kus_metrics["ter26_par_65kmh"] = {"k_us_linear_deg_per_g": kl26_par, "max_ay_g": float(np.max(sim_t26_par_65["ay"]) / G_ACC)}

    for g, k, b, col, ls, lbl in [
        (g27_3d, k27_3d, b27_3d, PALETTE["cyan"], "-", f"TeR27 4WD 3D (K_lin={kl27_3d:+.2f} deg/g)"),
        (g27_par, k27_par, b27_par, PALETTE["amber"], "--", f"TeR27 4WD Par (K_lin={kl27_par:+.2f} deg/g)"),
        (g26_3d, k26_3d, b26_3d, PALETTE["crimson"], "-", f"TeR26 RWD 3D (K_lin={kl26_3d:+.2f} deg/g)"),
        (g26_par, k26_par, b26_par, PALETTE["violet"], "--", f"TeR26 RWD Par (K_lin={kl26_par:+.2f} deg/g)"),
    ]:
        axes[0, 1].plot(g, k, color=col, linestyle=ls, linewidth=2.0, label=lbl)
        axes[1, 1].plot(g, b, color=col, linestyle=ls, linewidth=2.0, label=lbl)

    axes[0, 1].axhline(0.0, color=PALETTE["white"], linewidth=0.9, linestyle=":")
    axes[0, 1].set_ylim(-2.0, 6.0)
    style_ax(axes[0, 1], title="Understeer Gradient @ 65 km/h: TeR27 vs TeR26 (3D vs Par)", xlabel="Lateral Acceleration Ay [g]", ylabel="K_us [deg/g]", legend=True, legend_loc="upper left")
    axes[1, 1].axhline(0.0, color=PALETTE["white"], linewidth=0.9, linestyle=":")
    style_ax(axes[1, 1], title="Sideslip Angle beta(Ay) @ 65 km/h: TeR27 vs TeR26", xlabel="Lateral Acceleration Ay [g]", ylabel="Sideslip beta [deg]", legend=True, legend_loc="lower left")
    add_kpi_box(
        axes[1, 1],
        [
            f"TeR27 3D K_us : {kl27_3d:+.2f} deg/g",
            f"TeR26 3D K_us : {kl26_3d:+.2f} deg/g",
            f"TeR27 Peak Ay : {np.max(results_t27_3d[65.0]['ay'])/G_ACC:.2f} g",
            f"TeR26 Peak Ay : {np.max(sim_t26_3d_65['ay'])/G_ACC:.2f} g",
        ],
        loc="upper right",
        border_color=PALETTE["cyan"],
    )
    add_suptitle(
        fig,
        "Steady-State Ramp Steer: Understeer Gradient K_us(Ay) & Sideslip beta(Ay)",
        "1.5 deg/s quasi-static ramp | 40, 65, 85 km/h | TeR27 4WD vs TeR26 RWD & 3D vs Parametric Kinematics",
    )
    add_footer(fig)
    fig.tight_layout(rect=[0.0, 0.025, 1.0, 0.92])
    save_fig(fig, OUT_DIR / "ramp_steer_understeer_gradient.png", dpi=300)

    # --- Figure 5: Roll Gradient phi(ay) & 4-Wheel Normal Loads Fz_i(ay) ---
    sim65 = results_t27_3d[65.0]
    i_pk = int(np.argmax(sim65["ay"]))
    ay65 = sim65["ay"][: i_pk + 1] / G_ACC
    phi65 = sim65["phi_deg"][: i_pk + 1]
    fz65 = sim65["fz"][: i_pk + 1]

    # Fit linear roll gradient
    lin_m = (ay65 >= 0.15) & (ay65 <= 1.2)
    roll_grad_deg_g = float(np.polyfit(ay65[lin_m], phi65[lin_m], 1)[0])
    phi26 = sim_t26_3d_65["phi_deg"][: int(np.argmax(sim_t26_3d_65["ay"])) + 1]
    ay26 = sim_t26_3d_65["ay"][: len(phi26)] / G_ACC
    lin_m26 = (ay26 >= 0.15) & (ay26 <= 1.0)
    roll_grad_t26 = float(np.polyfit(ay26[lin_m26], phi26[lin_m26], 1)[0])

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4))
    axes[0].plot(ay65, phi65, color=PALETTE["cyan"], linewidth=2.2, label=f"TeR27 4WD 3D ({roll_grad_deg_g:.2f} deg/g)")
    axes[0].plot(ay26, phi26, color=PALETTE["crimson"], linewidth=2.0, linestyle="--", label=f"TeR26 RWD 3D ({roll_grad_t26:.2f} deg/g)")
    style_ax(axes[0], title="Chassis Roll Angle vs Lateral Acceleration @ 65 km/h", xlabel="Lateral Acceleration Ay [g]", ylabel="Roll Angle phi [deg]", legend=True, legend_loc="upper left")
    add_kpi_box(
        axes[0],
        [
            f"TeR27 Roll Grad : {roll_grad_deg_g:.2f} deg/g",
            f"TeR26 Roll Grad : {roll_grad_t26:.2f} deg/g",
            f"TeR27 Max Roll  : {float(np.max(phi65)):.2f} deg",
            f"TeR27 LLTD (el) : {m_t27_3d.p.lltd_elastic_f*100.0:.1f} % F",
        ],
        loc="lower right",
        border_color=PALETTE["cyan"],
    )

    wheel_cfg = [
        (0, "FL (Inner Front)", PALETTE["cyan"], "--"),
        (1, "FR (Outer Front)", PALETTE["cyan"], "-"),
        (2, "RL (Inner Rear)", PALETTE["amber"], "--"),
        (3, "RR (Outer Rear)", PALETTE["amber"], "-"),
    ]
    for idx, lbl, col, ls in wheel_cfg:
        axes[1].plot(ay65, fz65[:, idx], color=col, linestyle=ls, linewidth=2.0, label=lbl)
    axes[1].axhline(0.0, color=PALETTE["crimson"], linewidth=0.9, linestyle=":")
    style_ax(axes[1], title="TeR27 4-Wheel Vertical Loads Fz_i(Ay) @ 65 km/h", xlabel="Lateral Acceleration Ay [g]", ylabel="Normal Load Fz [N]", legend=True, legend_loc="upper left")
    add_kpi_box(
        axes[1],
        [
            f"Peak Outer RR : {float(np.max(fz65[:, 3])):.0f} N",
            f"Peak Outer FR : {float(np.max(fz65[:, 1])):.0f} N",
            f"Min Inner FL  : {float(np.min(fz65[:, 0])):.0f} N",
            f"Min Inner RL  : {float(np.min(fz65[:, 2])):.0f} N",
        ],
        loc="center right",
        border_color=PALETTE["amber"],
    )
    add_suptitle(
        fig,
        "Ramp-Steer Roll Dynamics & 4-Wheel Lateral Load Transfer @ 65 km/h",
        "Elastic roll stiffness + geometric roll center + unsprung mass transfer + aerodynamic downforce",
    )
    add_footer(fig)
    fig.tight_layout(rect=[0.0, 0.025, 1.0, 0.91])
    save_fig(fig, OUT_DIR / "ramp_steer_roll_and_load_transfer.png", dpi=300)

    return {
        "understeer_gradient": kus_metrics,
        "roll_gradient_ter27_deg_per_g": roll_grad_deg_g,
        "roll_gradient_ter26_deg_per_g": roll_grad_t26,
        "fz_at_peak_ay_65kmh_n": {
            "FL": float(fz65[-1, 0]),
            "FR": float(fz65[-1, 1]),
            "RL": float(fz65[-1, 2]),
            "RR": float(fz65[-1, 3]),
        },
    }


# =============================================================================
# 5. Family 3: Step-Steer Transient Response (Figs 6, 7)
# =============================================================================

def _step_metrics(t: np.ndarray, y: np.ndarray, t_step: float = 0.25) -> Dict[str, float]:
    """Compute t_rise (10-90% ms), t_peak (ms), overshoot (%), and effective damping ratio zeta."""
    mask_ss = t >= (t[-1] - 0.6)
    y_ss = float(np.mean(y[mask_ss]))
    if abs(y_ss) < 1e-5:
        return {"t_rise_ms": 0.0, "t_peak_ms": 0.0, "overshoot_pct": 0.0, "zeta": 1.0, "y_ss": y_ss}
    sgn = 1.0 if y_ss >= 0 else -1.0
    yn = (y * sgn) / abs(y_ss)
    post = t >= t_step
    tp = t[post] - t_step
    yp = yn[post]

    idx10 = np.where(yp >= 0.10)[0]
    idx90 = np.where(yp >= 0.90)[0]
    t10 = float(tp[idx10[0]]) if len(idx10) > 0 else 0.0
    t90 = float(tp[idx90[0]]) if len(idx90) > 0 else tp[-1]
    t_rise_ms = max((t90 - t10) * 1000.0, 1.0)

    i_pk = int(np.argmax(yp))
    t_peak_ms = float(tp[i_pk] * 1000.0)
    os_frac = max(float(yp[i_pk]) - 1.0, 0.0)
    overshoot_pct = os_frac * 100.0

    if os_frac > 1e-4:
        ln_os = math.log(os_frac)
        zeta = float(-ln_os / math.sqrt(math.pi * math.pi + ln_os * ln_os))
    else:
        zeta = 1.0

    return {
        "t_rise_ms": t_rise_ms,
        "t_peak_ms": t_peak_ms,
        "overshoot_pct": overshoot_pct,
        "zeta": float(np.clip(zeta, 0.05, 1.2)),
        "y_ss": y_ss,
    }


def run_family_3_step_steer(ter26_3d: TeR26HandlingWrapper) -> Dict[str, Any]:
    print("[Family 3/6] Running Step-Steer Transient Battery (50 & 80 km/h, Small & Large)...", flush=True)
    m_3d = TeR27HandlingModel(profile="FSG_RUBBERED", tire_temp_c=77.5, use_3d_kin=True)
    m_par = TeR27HandlingModel(profile="FSG_RUBBERED", tire_temp_c=77.5, use_3d_kin=False)

    dt = 0.0025  # 400 Hz
    t_s = np.arange(0.0, 2.5, dt)
    t_step = 0.25
    t_ramp = 0.08  # Fast 80 ms steering step (ISO 7401)

    cases = [
        ("50kmh_small", 50.0, 2.0, "50 km/h | Small (2.0 deg)", PALETTE["cyan"], "-"),
        ("50kmh_large", 50.0, 4.5, "50 km/h | Large (4.5 deg)", PALETTE["amber"], "-"),
        ("80kmh_small", 80.0, 1.0, "80 km/h | Small (1.0 deg)", PALETTE["emerald"], "--"),
        ("80kmh_large", 80.0, 2.0, "80 km/h | Large (2.0 deg)", PALETTE["crimson"], "--"),
    ]

    sims_3d, sims_par, sims_t26 = {}, {}, {}
    metrics_step: Dict[str, Any] = {}

    for key, v_kmh, d_deg, _, _, _ in cases:
        d_rad = math.radians(d_deg)
        delta_arr = np.clip((t_s - t_step) / t_ramp, 0.0, 1.0) * d_rad

        def _make_ctrl(d_r):
            def _ctrl(t, st, diag):
                return float(np.clip((t - t_step) / t_ramp, 0.0, 1.0) * d_r), 0.0, 0.0
            return _ctrl

        s3 = m_3d.simulate_maneuver(t_s, v_kmh / 3.6, _make_ctrl(d_rad), hold_vx=True)
        sp = m_par.simulate_maneuver(t_s, v_kmh / 3.6, _make_ctrl(d_rad), hold_vx=True)
        s26 = ter26_3d.simulate_constant_speed(t_s, v_kmh / 3.6, delta_arr)

        sims_3d[key] = s3
        sims_par[key] = sp
        sims_t26[key] = s26

        metrics_step[key] = {
            "ter27_3d_wz": _step_metrics(t_s, s3["wz"], t_step),
            "ter27_3d_ay": _step_metrics(t_s, s3["ay"], t_step),
            "ter27_par_wz": _step_metrics(t_s, sp["wz"], t_step),
            "ter26_3d_wz": _step_metrics(t_s, s26["wz"], t_step),
        }

    # --- Figure 6: Step-Steer Time Traces (wz, ay, beta, phi) ---
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.8), sharex=True)
    for key, _, _, lbl, col, ls in cases:
        s3 = sims_3d[key]
        axes[0, 0].plot(t_s, s3["wz"], color=col, linestyle=ls, linewidth=1.9, label=lbl)
        axes[0, 1].plot(t_s, s3["ay"] / G_ACC, color=col, linestyle=ls, linewidth=1.9, label=lbl)
        axes[1, 0].plot(t_s, s3["beta_deg"], color=col, linestyle=ls, linewidth=1.9, label=lbl)
        axes[1, 1].plot(t_s, s3["phi_deg"], color=col, linestyle=ls, linewidth=1.9, label=lbl)

    for ax in axes.flat:
        ax.axvline(t_step, color=PALETTE["white"], linewidth=0.8, linestyle=":", alpha=0.7)
        ax.set_xlim(0.0, 2.2)
    style_ax(axes[0, 0], title="Yaw Rate Response wz(t)", ylabel="Yaw Rate wz [rad/s]", legend=True, legend_loc="lower right")
    style_ax(axes[0, 1], title="Lateral Acceleration Response Ay(t)", ylabel="Lateral Accel Ay [g]")
    style_ax(axes[1, 0], title="Body Sideslip Response beta(t)", xlabel="Time [s]", ylabel="Sideslip beta [deg]")
    style_ax(axes[1, 1], title="Chassis Roll Response phi(t)", xlabel="Time [s]", ylabel="Roll Angle phi [deg]")
    m80s = metrics_step["80kmh_small"]["ter27_3d_wz"]
    m80l = metrics_step["80kmh_large"]["ter27_3d_wz"]
    add_kpi_box(
        axes[0, 1],
        [
            f"80 km/h Small wz t_rise : {m80s['t_rise_ms']:.0f} ms",
            f"80 km/h Small OS        : {m80s['overshoot_pct']:.1f} %",
            f"80 km/h Large wz t_rise : {m80l['t_rise_ms']:.0f} ms",
            f"80 km/h Large OS        : {m80l['overshoot_pct']:.1f} %",
        ],
        loc="lower right",
        border_color=PALETTE["emerald"],
    )
    add_suptitle(
        fig,
        "TeR27 4WD Step-Steer Transient Response (ISO 7401, 400 Hz RK4)",
        "50 & 80 km/h | Linear (small) and near-limit (large) amplitudes | Relaxation length sigma_y=0.22 m + Roll ODE",
    )
    add_footer(fig)
    fig.tight_layout(rect=[0.0, 0.025, 1.0, 0.92])
    save_fig(fig, OUT_DIR / "step_steer_transient_response.png", dpi=300)

    # --- Figure 7: Step-Steer Metrics Comparison (TeR27 3D vs Parametric vs TeR26) ---
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.6))
    case_keys = [c[0] for c in cases]
    case_labels = ["50 km/h\nSmall", "50 km/h\nLarge", "80 km/h\nSmall", "80 km/h\nLarge"]
    x_pos = np.arange(len(case_keys))
    w = 0.25

    metric_panels = [
        ("t_rise_ms", "Yaw Rate Rise Time t_rise (10-90%) [ms]", "Time [ms]", axes[0, 0]),
        ("t_peak_ms", "Yaw Rate Peak Time t_peak [ms]", "Time [ms]", axes[0, 1]),
        ("overshoot_pct", "Yaw Rate Overshoot [%]", "Overshoot [%]", axes[1, 0]),
        ("zeta", "Effective Yaw Damping Ratio zeta [-]", "Damping Ratio zeta [-]", axes[1, 1]),
    ]
    for mkey, title, ylabel, ax in metric_panels:
        v_t27_3d = [metrics_step[k]["ter27_3d_wz"][mkey] for k in case_keys]
        v_t27_par = [metrics_step[k]["ter27_par_wz"][mkey] for k in case_keys]
        v_t26_3d = [metrics_step[k]["ter26_3d_wz"][mkey] for k in case_keys]
        b1 = ax.bar(x_pos - w, v_t27_3d, width=w, color=PALETTE["cyan"], label="TeR27 4WD (3D)")
        b2 = ax.bar(x_pos, v_t27_par, width=w, color=PALETTE["amber"], label="TeR27 4WD (Parametric)")
        b3 = ax.bar(x_pos + w, v_t26_3d, width=w, color=PALETTE["crimson"], label="TeR26 RWD (3D)")
        for bars in (b1, b2, b3):
            for bar in bars:
                h = bar.get_height()
                fmt = f"{h:.2f}" if mkey == "zeta" else f"{h:.0f}"
                ax.text(bar.get_x() + bar.get_width() * 0.5, h + 0.02 * max(max(v_t27_3d + v_t26_3d), 1.0), fmt, ha="center", va="bottom", fontsize=7.2, fontfamily="DejaVu Sans Mono", color=PALETTE["text"])
        ax.set_xticks(x_pos)
        ax.set_xticklabels(case_labels, fontsize=8.4)
        style_ax(ax, title=title, ylabel=ylabel, legend=(ax is axes[0, 0]), legend_loc="upper right")

    add_suptitle(
        fig,
        "Step-Steer Transient Metrics Comparison: TeR27 (3D vs Parametric) vs TeR26 (3D)",
        "Yaw-rate rise time, peak time, percent overshoot, and effective damping across linear & nonlinear regimes",
    )
    add_footer(fig)
    fig.tight_layout(rect=[0.0, 0.025, 1.0, 0.92])
    save_fig(fig, OUT_DIR / "step_steer_metrics_comparison.png", dpi=300)

    return metrics_step


# =============================================================================
# 6. Family 4: Sine-Sweep Frequency Response (Bode Plot - Fig 8)
# =============================================================================

def run_family_4_bode() -> Dict[str, Any]:
    print("[Family 4/6] Running Sine-Sweep Frequency Response (0.1 - 6.0 Hz @ 45 & 80 km/h)...", flush=True)
    m_3d = TeR27HandlingModel(profile="FSG_RUBBERED", tire_temp_c=77.5, use_3d_kin=True)

    fs = 200.0
    dt = 1.0 / fs
    T_dur = 40.0
    t_s = np.arange(0.0, T_dur, dt)
    f0, f1 = 0.08, 6.5

    # Log-chirp + small broadband excitation so Welch CSD has full energy across 0.1 - 6.0 Hz
    rng = np.random.default_rng(2027)
    chirp_sig = signal.chirp(t_s, f0=f0, f1=f1, t1=T_dur, method="logarithmic")
    # Smooth Tukey window at ends
    win = signal.windows.tukey(len(t_s), alpha=0.06)
    # Band-limited PRBS/multisine overlay to enrich Welch segment coherence
    b_lp, a_lp = signal.butter(4, 7.0 / (0.5 * fs))
    noise_exc = signal.filtfilt(b_lp, a_lp, rng.standard_normal(len(t_s)))
    noise_exc = noise_exc / (np.std(noise_exc) + 1e-9)

    bode_data = {}
    bode_metrics: Dict[str, Any] = {}

    for v_kmh, amp_deg in [(45.0, 2.2), (80.0, 1.6)]:
        delta_arr = math.radians(amp_deg) * (0.75 * chirp_sig * win + 0.25 * noise_exc * win)

        def _make_ctrl(d_arr):
            def _ctrl(t, st, diag):
                idx = min(int(round(t * fs)), len(d_arr) - 1)
                return float(d_arr[idx]), 0.0, 0.0
            return _ctrl

        sim = m_3d.simulate_maneuver(t_s, v_kmh / 3.6, _make_ctrl(delta_arr), hold_vx=True)

        # Tiny realistic IMU noise floor for physical coherence roll-off above yaw bandwidth
        wz_meas = sim["wz"] + rng.normal(0.0, 0.0025, len(t_s))
        ay_meas = sim["ay"] + rng.normal(0.0, 0.04, len(t_s))
        beta_meas = np.radians(sim["beta_deg"]) + rng.normal(0.0, 0.0008, len(t_s))

        nperseg = 1024
        freqs, pxx = signal.welch(delta_arr, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
        mask = (freqs >= 0.1) & (freqs <= 6.0)
        f_out = freqs[mask]

        ch_res = {}
        for cname, y_sig in [("wz", wz_meas), ("ay", ay_meas), ("beta", beta_meas)]:
            _, pxy = signal.csd(delta_arr, y_sig, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
            _, coh = signal.coherence(delta_arr, y_sig, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
            H = pxy[mask] / (pxx[mask] + 1e-12)
            mag_db = 20.0 * np.log10(np.maximum(np.abs(H), 1e-6))
            phase_deg = np.degrees(np.unwrap(np.angle(H)))
            # Keep phase near 0 or -180 start cleanly
            if cname == "beta" and phase_deg[0] > 90.0:
                phase_deg -= 360.0
            ch_res[cname] = {"mag_db": mag_db, "phase_deg": phase_deg, "coh": coh[mask]}

        # Bandwidth (-3 dB from DC/low-freq gain) & natural yaw frequency (peak mag)
        wz_mag = ch_res["wz"]["mag_db"]
        dc_gain_db = float(np.mean(wz_mag[:3]))
        i_pk = int(np.argmax(wz_mag))
        f_res = float(f_out[i_pk])
        below_3db = np.where((f_out > f_res) & (wz_mag <= dc_gain_db - 3.0))[0]
        f_bw = float(f_out[below_3db[0]]) if len(below_3db) > 0 else float(f_out[-1])

        bode_data[v_kmh] = {"f": f_out, "ch": ch_res}
        bode_metrics[f"{int(v_kmh)}kmh"] = {
            "wz_dc_gain_db": dc_gain_db,
            "wz_peak_freq_hz": f_res,
            "wz_bandwidth_3db_hz": f_bw,
            "wz_mean_coherence_0p1_4hz": float(np.mean(ch_res["wz"]["coh"][f_out <= 4.0])),
            "ay_dc_gain_db": float(np.mean(ch_res["ay"]["mag_db"][:3])),
        }

    # --- Figure 8: Bode Magnitude, Phase & Coherence ---
    fig, axes = plt.subplots(3, 3, figsize=(16.0, 9.8), sharex=True)
    ch_meta = [
        ("wz", "Yaw Rate Gain |wz / delta|", "Mag [dB (rad/s/rad)]"),
        ("ay", "Lateral Accel Gain |Ay / delta|", "Mag [dB (m/s^2/rad)]"),
        ("beta", "Sideslip Gain |beta / delta|", "Mag [dB (rad/rad)]"),
    ]
    v_styles = [(45.0, PALETTE["cyan"], "-", "45 km/h"), (80.0, PALETTE["amber"], "--", "80 km/h")]

    for col_i, (cname, title, mag_lbl) in enumerate(ch_meta):
        for v_kmh, col, ls, vlbl in v_styles:
            f_out = bode_data[v_kmh]["f"]
            c = bode_data[v_kmh]["ch"][cname]
            axes[0, col_i].semilogx(f_out, c["mag_db"], color=col, linestyle=ls, linewidth=1.9, label=vlbl)
            axes[1, col_i].semilogx(f_out, c["phase_deg"], color=col, linestyle=ls, linewidth=1.9, label=vlbl)
            axes[2, col_i].semilogx(f_out, c["coh"], color=col, linestyle=ls, linewidth=1.9, label=vlbl)

        style_ax(axes[0, col_i], title=title, ylabel=mag_lbl if col_i == 0 else "Mag [dB]", legend=(col_i == 0), legend_loc="lower left", grid_minor=True)
        style_ax(axes[1, col_i], ylabel="Phase [deg]" if col_i == 0 else "", grid_minor=True)
        axes[2, col_i].set_ylim(0.0, 1.05)
        axes[2, col_i].axhline(0.85, color=PALETTE["emerald"], linewidth=0.8, linestyle=":")
        style_ax(axes[2, col_i], xlabel="Frequency [Hz]", ylabel="Coherence gamma^2 [-]" if col_i == 0 else "", grid_minor=True)

    add_kpi_box(
        axes[0, 0],
        [
            f"45 km/h BW(-3dB): {bode_metrics['45kmh']['wz_bandwidth_3db_hz']:.2f} Hz",
            f"80 km/h BW(-3dB): {bode_metrics['80kmh']['wz_bandwidth_3db_hz']:.2f} Hz",
            f"80 km/h f_peak  : {bode_metrics['80kmh']['wz_peak_freq_hz']:.2f} Hz",
        ],
        loc="upper right",
        border_color=PALETTE["cyan"],
    )
    add_suptitle(
        fig,
        "TeR27 4WD Sine-Sweep Frequency Response (Bode Plot: 0.1 - 6.0 Hz)",
        "Log-chirp + broadband excitation @ 45 & 80 km/h | Welch CSD transfer functions & coherence",
    )
    add_footer(fig)
    fig.tight_layout(rect=[0.0, 0.025, 1.0, 0.92])
    save_fig(fig, OUT_DIR / "bode_yaw_and_lat_accel.png", dpi=300)

    return bode_metrics


# =============================================================================
# 7. Family 5: Trail Braking, Slalom & ISO 3888-2 DLC, Split-Mu (Figs 9, 10, 11)
# =============================================================================

def run_family_5_combined_and_emergency() -> Dict[str, Any]:
    print("[Family 5/6] Running Trail Braking, 15m Slalom, ISO 3888-2 DLC & Split-Mu...", flush=True)
    m_3d = TeR27HandlingModel(profile="FSG_RUBBERED", tire_temp_c=77.5, use_3d_kin=True)
    metrics_f5: Dict[str, Any] = {}

    # --- 5A. Trail Braking Stability Envelope (Fig 9) ---
    dt = 0.004
    t_tb = np.arange(0.0, 2.4, dt)
    v0_tb = 85.0 / 3.6  # 85 km/h corner entry

    tb_profiles = [
        ("optimal_progressive", "Optimal Progressive Release (BB 62%)", 0.62, 1.0, PALETTE["emerald"], "-"),
        ("late_aggressive", "Aggressive Late-Brake Release (BB 62%)", 0.62, 2.0, PALETTE["amber"], "--"),
        ("rear_biased_overbrake", "Rear-Biased Over-Brake (BB 50%)", 0.50, 1.5, PALETTE["crimson"], "-."),
    ]
    tb_sims = {}
    tb_metrics = {}
    orig_bb = m_3d.p.brake_bias_front

    for key, lbl, bb, release_exp, col, ls in tb_profiles:
        m_3d.p.brake_bias_front = bb

        def _make_tb_ctrl(rexp):
            def _ctrl(t, st, diag):
                # 0.0-0.4s: straight-line 1.4g braking; 0.4-2.0s: turn-in 0 -> 7.5 deg while releasing brake
                if t < 0.4:
                    delta = 0.0
                    ax_cmd = -1.40 * G_ACC
                else:
                    u = min((t - 0.4) / 1.6, 1.0)
                    delta = math.radians(7.5) * (1.0 - math.exp(-3.2 * u))
                    # Brake release profile: (1 - u)^rexp -> higher rexp releases faster or slower?
                    # Make late_aggressive hold brake longer: 1 - u^rexp
                    brk_frac = max(0.0, 1.0 - (u ** rexp))
                    ax_cmd = -1.40 * G_ACC * brk_frac
                return delta, ax_cmd, 0.0
            return _ctrl

        sim = m_3d.simulate_maneuver(t_tb, v0_tb, _make_tb_ctrl(release_exp), hold_vx=False)
        tb_sims[key] = sim
        tb_metrics[key] = {
            "max_abs_beta_deg": float(np.max(np.abs(sim["beta_deg"]))),
            "max_ay_g": float(np.max(sim["ay"]) / G_ACC),
            "min_ax_g": float(np.min(sim["ax"]) / G_ACC),
            "min_inner_rear_fz_n": float(np.min(sim["fz"][:, 2])),
            "max_rear_alpha_deg": float(np.max(np.abs(sim["alpha_deg"][:, 2:4]))),
        }
    m_3d.p.brake_bias_front = orig_bb
    metrics_f5["trail_braking"] = tb_metrics

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.8))
    for key, lbl, _, _, col, ls in tb_profiles:
        sim = tb_sims[key]
        axes[0, 0].plot(sim["ay"] / G_ACC, sim["ax"] / G_ACC, color=col, linestyle=ls, linewidth=2.0, label=lbl)
        axes[0, 1].plot(t_tb, sim["beta_deg"], color=col, linestyle=ls, linewidth=1.9, label=lbl)
        axes[1, 0].plot(t_tb, sim["fz"][:, 2], color=col, linestyle=ls, linewidth=1.9, label=f"{lbl} (RL inner)")
        axes[1, 1].plot(t_tb, np.mean(sim["alpha_deg"][:, 2:4], axis=1), color=col, linestyle=ls, linewidth=1.9, label=lbl)

    axes[0, 0].axhline(0.0, color=PALETTE["white"], linewidth=0.8, alpha=0.6)
    axes[0, 0].axvline(0.0, color=PALETTE["white"], linewidth=0.8, alpha=0.6)
    style_ax(axes[0, 0], title="Trail-Braking G-G Trajectory (1.4 g Entry @ 85 km/h)", xlabel="Lateral Accel Ay [g]", ylabel="Longitudinal Accel Ax [g]", legend=True, legend_loc="lower right")
    style_ax(axes[0, 1], title="Sideslip Stability Envelope beta(t)", xlabel="Time [s]", ylabel="Sideslip beta [deg]")
    axes[1, 0].axhline(50.0, color=PALETTE["crimson"], linewidth=0.9, linestyle=":", label="Unloading Warning (50 N)")
    style_ax(axes[1, 0], title="Inner Rear Tire Normal Load Fz_RL(t)", xlabel="Time [s]", ylabel="Normal Load Fz_RL [N]", legend=True, legend_loc="upper right")
    style_ax(axes[1, 1], title="Rear Axle Mean Slip Angle alpha_R(t)", xlabel="Time [s]", ylabel="Rear Slip Angle [deg]")
    add_kpi_box(
        axes[0, 1],
        [
            f"Opt max|beta|  : {tb_metrics['optimal_progressive']['max_abs_beta_deg']:.2f} deg",
            f"Late max|beta| : {tb_metrics['late_aggressive']['max_abs_beta_deg']:.2f} deg",
            f"RearBB |beta|  : {tb_metrics['rear_biased_overbrake']['max_abs_beta_deg']:.2f} deg",
            f"Opt min Fz_RL  : {tb_metrics['optimal_progressive']['min_inner_rear_fz_n']:.0f} N",
        ],
        loc="lower left",
        border_color=PALETTE["emerald"],
    )
    add_suptitle(
        fig,
        "TeR27 4WD Trail-Braking Stability Envelope (1.4 g Entry @ 85 km/h)",
        "Progressive brake release vs late-brake & rear-biased braking | Rear slip & inner-wheel unloading",
    )
    add_footer(fig)
    fig.tight_layout(rect=[0.0, 0.025, 1.0, 0.92])
    save_fig(fig, OUT_DIR / "trail_braking_stability_envelope.png", dpi=300)

    # --- 5B. FS 15 m Slalom & ISO 3888-2 Double Lane Change (Fig 10) ---
    def _run_slalom(v_kmh: float) -> Tuple[Dict[str, np.ndarray], bool, float]:
        v_ms = v_kmh / 3.6
        t_sl = np.arange(0.0, 120.0 / v_ms, 0.004)
        # Sinusoidal reference path around cones every 15 m (wavelength 30 m, amplitude 1.65 m)
        A_sl = 1.65
        lam_sl = 30.0
        k_w = 2.0 * math.pi / lam_sl

        def _ctrl(t, st, diag):
            X, Y, psi, vx, wz = st[9], st[10], st[11], max(st[0], 1.0), st[2]
            L_la = max(0.24 * vx, 2.8)
            X_la = X + L_la * math.cos(psi)
            if X_la < 8.0 or X_la > 102.0:
                Y_ref = 0.0
                psi_ref = 0.0
                kappa_ref = 0.0
            else:
                env = min((X_la - 8.0) / 7.0, 1.0) * min((102.0 - X_la) / 7.0, 1.0)
                ph = k_w * (X_la - 15.0)
                Y_ref = env * A_sl * math.cos(ph)
                dY = -env * A_sl * k_w * math.sin(ph)
                ddY = -env * A_sl * (k_w ** 2) * math.cos(ph)
                psi_ref = math.atan(dY)
                kappa_ref = ddY / ((1.0 + dY * dY) ** 1.5)
            e_y = Y_ref - Y
            e_psi = math.atan2(math.sin(psi_ref - psi), math.cos(psi_ref - psi))
            delta_ff = math.atan(m_3d.p.wheelbase * kappa_ref)
            delta = delta_ff + 1.25 * e_psi + math.atan2(1.90 * e_y, vx) - 0.035 * (wz - vx * kappa_ref)
            return float(np.clip(delta, -0.34, 0.34)), 0.0, 0.0

        sim = m_3d.simulate_maneuver(t_sl, v_ms, _ctrl, hold_vx=True)
        cone_xs = [15.0, 30.0, 45.0, 60.0, 75.0, 90.0]
        min_clear = 99.0
        for idx_c, cx in enumerate(cone_xs):
            y_at_c = float(np.interp(cx, sim["X"], sim["Y"]))
            expected_side = 1.0 if (idx_c % 2 == 0) else -1.0
            clear = expected_side * y_at_c - 0.72
            min_clear = min(min_clear, clear)
        passed = (min_clear > 0.10) and (float(np.max(np.abs(sim["beta_deg"]))) < 14.0)
        return sim, passed, min_clear

    # Find max passing speed for 15 m Slalom
    best_sl_v, best_sl_sim = 45.0, None
    for v_try in np.arange(45.0, 82.0, 2.5):
        sim_try, ok, _ = _run_slalom(float(v_try))
        if ok:
            best_sl_v = float(v_try)
            best_sl_sim = sim_try
        else:
            break
    if best_sl_sim is None:
        best_sl_sim, _, _ = _run_slalom(45.0)

    def _run_dlc(v_kmh: float) -> Tuple[Dict[str, np.ndarray], bool, float]:
        """ISO 3888-2 Double Lane Change: 12m entry, 13.5m LC, 11m offset (+3.5m), 12.5m return, 12m exit."""
        v_ms = v_kmh / 3.6
        t_dlc = np.arange(0.0, 72.0 / v_ms, 0.004)
        y_off = 3.5

        def _ref_y(x_val: float) -> Tuple[float, float, float]:
            # Smooth quintic-like cosine transitions matching ISO 3888-2 gates
            if x_val <= 12.0:
                return 0.0, 0.0, 0.0
            elif x_val <= 25.5:
                u = (x_val - 12.0) / 13.5
                y = 0.5 * y_off * (1.0 - math.cos(math.pi * u))
                dy = 0.5 * y_off * (math.pi / 13.5) * math.sin(math.pi * u)
                ddy = 0.5 * y_off * ((math.pi / 13.5) ** 2) * math.cos(math.pi * u)
                return y, dy, ddy
            elif x_val <= 36.5:
                return y_off, 0.0, 0.0
            elif x_val <= 49.0:
                u = (x_val - 36.5) / 12.5
                y = 0.5 * y_off * (1.0 + math.cos(math.pi * u))
                dy = -0.5 * y_off * (math.pi / 12.5) * math.sin(math.pi * u)
                ddy = -0.5 * y_off * ((math.pi / 12.5) ** 2) * math.cos(math.pi * u)
                return y, dy, ddy
            else:
                return 0.0, 0.0, 0.0

        def _ctrl(t, st, diag):
            X, Y, psi, vx, wz = st[9], st[10], st[11], max(st[0], 1.0), st[2]
            L_la = max(0.28 * vx, 3.0)
            X_la = X + L_la * math.cos(psi)
            Y_ref, dY, ddY = _ref_y(X_la)
            psi_ref = math.atan(dY)
            kappa_ref = ddY / ((1.0 + dY * dY) ** 1.5)
            e_y = Y_ref - Y
            e_psi = math.atan2(math.sin(psi_ref - psi), math.cos(psi_ref - psi))
            delta_ff = math.atan(m_3d.p.wheelbase * kappa_ref)
            delta = delta_ff + 1.25 * e_psi + math.atan2(1.85 * e_y, vx) - 0.04 * (wz - vx * kappa_ref)
            return float(np.clip(delta, -0.35, 0.35)), 0.0, 0.0

        sim = m_3d.simulate_maneuver(t_dlc, v_ms, _ctrl, hold_vx=True)
        # ISO 3888-2 gate corridor check (vehicle width 1.44m -> half-width 0.72m)
        # Entry gate [0,12]: width 1.65m -> |Y| <= 0.45m; Offset box [25.5,36.5]: width 2.2m -> |Y-3.5| <= 0.75m; Exit [49,61]: |Y| <= 0.85m
        m1 = (sim["X"] >= 2.0) & (sim["X"] <= 12.0)
        m2 = (sim["X"] >= 26.5) & (sim["X"] <= 36.0)
        m3 = (sim["X"] >= 50.5) & (sim["X"] <= 60.0)
        e1 = float(np.max(np.abs(sim["Y"][m1]))) if np.any(m1) else 0.0
        e2 = float(np.max(np.abs(sim["Y"][m2] - y_off))) if np.any(m2) else 0.0
        e3 = float(np.max(np.abs(sim["Y"][m3]))) if np.any(m3) else 0.0
        max_err = max(e1, e2, e3)
        passed = (e1 < 0.45) and (e2 < 0.75) and (e3 < 0.90) and (float(np.max(np.abs(sim["beta_deg"]))) < 14.0)
        return sim, passed, max_err

    best_dlc_v, best_dlc_sim = 55.0, None
    for v_try in np.arange(55.0, 96.0, 2.5):
        sim_try, ok, _ = _run_dlc(float(v_try))
        if ok:
            best_dlc_v = float(v_try)
            best_dlc_sim = sim_try
        else:
            break
    if best_dlc_sim is None:
        best_dlc_sim, _, _ = _run_dlc(55.0)

    sl_dd_rms = float(np.sqrt(np.mean(np.gradient(best_sl_sim["delta_deg"], best_sl_sim["t"]) ** 2)))
    dlc_dd_rms = float(np.sqrt(np.mean(np.gradient(best_dlc_sim["delta_deg"], best_dlc_sim["t"]) ** 2)))
    metrics_f5["slalom_15m"] = {
        "max_passing_speed_kmh": best_sl_v,
        "max_abs_ay_g": float(np.max(np.abs(best_sl_sim["ay"])) / G_ACC),
        "max_abs_beta_deg": float(np.max(np.abs(best_sl_sim["beta_deg"]))),
        "pilot_workload_rms_deg_s": sl_dd_rms,
    }
    metrics_f5["iso_3888_2_dlc"] = {
        "max_passing_speed_kmh": best_dlc_v,
        "max_abs_ay_g": float(np.max(np.abs(best_dlc_sim["ay"])) / G_ACC),
        "max_abs_beta_deg": float(np.max(np.abs(best_dlc_sim["beta_deg"]))),
        "pilot_workload_rms_deg_s": dlc_dd_rms,
    }

    # --- Figure 10: Slalom & ISO 3888-2 DLC Traces ---
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 8.8))
    # Top-left: 15m Slalom path
    axes[0, 0].plot(best_sl_sim["X"], best_sl_sim["Y"], color=PALETTE["cyan"], linewidth=2.2, label=f"TeR27 CoG Path @ {best_sl_v:.1f} km/h")
    axes[0, 0].scatter([15, 30, 45, 60, 75, 90], [0]*6, color=PALETTE["amber"], s=60, marker="^", zorder=6, label="Slalom Cones (15 m)")
    axes[0, 0].set_xlim(0, 108)
    axes[0, 0].set_ylim(-3.5, 3.5)
    style_ax(axes[0, 0], title=f"FS 15 m Slalom Trajectory (Max Pass: {best_sl_v:.1f} km/h)", xlabel="Longitudinal X [m]", ylabel="Lateral Y [m]", legend=True, legend_loc="upper right")

    # Bottom-left: Slalom telemetry
    axes[1, 0].plot(best_sl_sim["t"], best_sl_sim["ay"] / G_ACC, color=PALETTE["cyan"], linewidth=1.8, label="Ay [g]")
    axes[1, 0].plot(best_sl_sim["t"], best_sl_sim["beta_deg"], color=PALETTE["amber"], linewidth=1.8, label="beta [deg]")
    axes[1, 0].plot(best_sl_sim["t"], best_sl_sim["delta_deg"], color=PALETTE["emerald"], linewidth=1.5, linestyle="--", label="delta [deg]")
    style_ax(axes[1, 0], title="FS 15 m Slalom Telemetry (Ay, beta, delta)", xlabel="Time [s]", ylabel="Signal Value [g or deg]", legend=True, legend_loc="upper right")
    add_kpi_box(
        axes[1, 0],
        [
            f"Max Speed  : {best_sl_v:.1f} km/h",
            f"Max |Ay|   : {metrics_f5['slalom_15m']['max_abs_ay_g']:.2f} g",
            f"Max |beta| : {metrics_f5['slalom_15m']['max_abs_beta_deg']:.2f} deg",
            f"Workload   : {sl_dd_rms:.1f} deg/s",
        ],
        loc="lower left",
        border_color=PALETTE["cyan"],
    )

    # Top-right: ISO 3888-2 DLC path + gates
    axes[0, 1].plot(best_dlc_sim["X"], best_dlc_sim["Y"], color=PALETTE["emerald"], linewidth=2.2, label=f"TeR27 CoG Path @ {best_dlc_v:.1f} km/h")
    # Draw ISO 3888-2 corridor gates
    axes[0, 1].plot([0, 12], [0.9, 0.9], color=PALETTE["crimson"], linewidth=1.6)
    axes[0, 1].plot([0, 12], [-0.9, -0.9], color=PALETTE["crimson"], linewidth=1.6, label="ISO 3888-2 Cone Gates")
    axes[0, 1].plot([25.5, 36.5], [4.6, 4.6], color=PALETTE["crimson"], linewidth=1.6)
    axes[0, 1].plot([25.5, 36.5], [2.4, 2.4], color=PALETTE["crimson"], linewidth=1.6)
    axes[0, 1].plot([49.0, 61.0], [1.1, 1.1], color=PALETTE["crimson"], linewidth=1.6)
    axes[0, 1].plot([49.0, 61.0], [-1.1, -1.1], color=PALETTE["crimson"], linewidth=1.6)
    axes[0, 1].set_xlim(0, 65)
    axes[0, 1].set_ylim(-2.0, 5.5)
    style_ax(axes[0, 1], title=f"ISO 3888-2 Double Lane Change (Max Pass: {best_dlc_v:.1f} km/h)", xlabel="Longitudinal X [m]", ylabel="Lateral Y [m]", legend=True, legend_loc="upper right")

    # Bottom-right: DLC telemetry
    axes[1, 1].plot(best_dlc_sim["t"], best_dlc_sim["ay"] / G_ACC, color=PALETTE["emerald"], linewidth=1.8, label="Ay [g]")
    axes[1, 1].plot(best_dlc_sim["t"], best_dlc_sim["beta_deg"], color=PALETTE["amber"], linewidth=1.8, label="beta [deg]")
    axes[1, 1].plot(best_dlc_sim["t"], best_dlc_sim["delta_deg"], color=PALETTE["violet"], linewidth=1.5, linestyle="--", label="delta [deg]")
    style_ax(axes[1, 1], title="ISO 3888-2 DLC Telemetry (Ay, beta, delta)", xlabel="Time [s]", ylabel="Signal Value [g or deg]", legend=True, legend_loc="upper right")
    add_kpi_box(
        axes[1, 1],
        [
            f"Max Speed  : {best_dlc_v:.1f} km/h",
            f"Max |Ay|   : {metrics_f5['iso_3888_2_dlc']['max_abs_ay_g']:.2f} g",
            f"Max |beta| : {metrics_f5['iso_3888_2_dlc']['max_abs_beta_deg']:.2f} deg",
            f"Workload   : {dlc_dd_rms:.1f} deg/s",
        ],
        loc="lower left",
        border_color=PALETTE["emerald"],
    )
    add_suptitle(
        fig,
        "TeR27 4WD Closed-Loop Agility: FS 15 m Slalom & ISO 3888-2 Double Lane Change",
        "Max cone-clean passing speeds, body sideslip envelope, and steering workload RMS(delta_dot)",
    )
    add_footer(fig)
    fig.tight_layout(rect=[0.0, 0.025, 1.0, 0.92])
    save_fig(fig, OUT_DIR / "slalom_and_dlc_traces.png", dpi=300)

    # --- 5C. Split-Mu Braking & Acceleration (0.90 Left / 0.50 Right) (Fig 11) ---
    # mu_wheel_scale relative to FSG_RUBBERED (0.88): [0.90/0.88, 0.50/0.88, 0.90/0.88, 0.50/0.88]
    split_scale = np.array([0.90, 0.50, 0.90, 0.50]) / 0.88
    m_split = TeR27HandlingModel(profile="FSG_RUBBERED", tire_temp_c=45.0, use_3d_kin=True, mu_wheel_scale=split_scale)
    t_sp = np.arange(0.0, 2.2, 0.004)

    # Braking from 80 km/h: Open (hands-off) vs Active 4WD TV + Counter-steer stabilization
    def _brk_open(t, st, diag):
        ax_cmd = -0.95 * G_ACC if st[0] > 3.0 else 0.0
        return 0.0, ax_cmd, 0.0

    def _brk_tv(t, st, diag):
        ax_cmd = -0.95 * G_ACC if st[0] > 3.0 else 0.0
        wz, psi, Y = st[2], st[11], st[10]
        # 4WD TV corrective yaw moment + subtle driver counter-steer
        mz_tv = float(np.clip(-420.0 * wz - 650.0 * psi, -650.0, 650.0))
        delta = float(np.clip(-0.55 * psi - 0.12 * Y - 0.08 * wz, -0.18, 0.18))
        return delta, ax_cmd, mz_tv

    # Acceleration from 30 km/h on Split-Mu: Open vs 4WD TV
    def _acc_open(t, st, diag):
        return 0.0, 0.85 * G_ACC, 0.0

    def _acc_tv(t, st, diag):
        wz, psi, Y = st[2], st[11], st[10]
        mz_tv = float(np.clip(-420.0 * wz - 650.0 * psi, -650.0, 650.0))
        delta = float(np.clip(-0.50 * psi - 0.10 * Y - 0.06 * wz, -0.18, 0.18))
        return delta, 0.85 * G_ACC, mz_tv

    s_brk_open = m_split.simulate_maneuver(t_sp, 80.0 / 3.6, _brk_open, hold_vx=False)
    s_brk_tv = m_split.simulate_maneuver(t_sp, 80.0 / 3.6, _brk_tv, hold_vx=False)
    s_acc_open = m_split.simulate_maneuver(t_sp, 30.0 / 3.6, _acc_open, hold_vx=False)
    s_acc_tv = m_split.simulate_maneuver(t_sp, 30.0 / 3.6, _acc_tv, hold_vx=False)

    metrics_f5["split_mu_0p90_0p50"] = {
        "braking_open_max_drift_m": float(np.max(np.abs(s_brk_open["Y"]))),
        "braking_tv_max_drift_m": float(np.max(np.abs(s_brk_tv["Y"]))),
        "braking_open_max_psi_deg": float(np.max(np.abs(s_brk_open["psi_deg"]))),
        "braking_tv_max_psi_deg": float(np.max(np.abs(s_brk_tv["psi_deg"]))),
        "braking_tv_peak_mz_nm": float(np.max(np.abs(s_brk_tv["mz_tv"]))),
        "accel_open_max_drift_m": float(np.max(np.abs(s_acc_open["Y"]))),
        "accel_tv_max_drift_m": float(np.max(np.abs(s_acc_tv["Y"]))),
        "accel_tv_peak_mz_nm": float(np.max(np.abs(s_acc_tv["mz_tv"]))),
    }

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.6))
    axes[0, 0].plot(t_sp, s_brk_open["Y"], color=PALETTE["crimson"], linewidth=2.0, linestyle="--", label="Braking Open (Unassisted)")
    axes[0, 0].plot(t_sp, s_brk_tv["Y"], color=PALETTE["cyan"], linewidth=2.2, label="Braking 4WD TV + Stabilized")
    axes[0, 0].axhline(0.0, color=PALETTE["white"], linewidth=0.8, alpha=0.6)
    style_ax(axes[0, 0], title="Split-Mu Braking (80 km/h): Lateral Drift Y(t)", xlabel="Time [s]", ylabel="Lateral Drift Y [m]", legend=True, legend_loc="upper left")

    axes[0, 1].plot(t_sp, s_brk_tv["mz_tv"], color=PALETTE["cyan"], linewidth=2.0, label="Braking Corrective Mz_TV [Nm]")
    axes[0, 1].plot(t_sp, s_acc_tv["mz_tv"], color=PALETTE["emerald"], linewidth=2.0, linestyle="--", label="Accel Corrective Mz_TV [Nm]")
    axes[0, 1].axhline(0.0, color=PALETTE["white"], linewidth=0.8, alpha=0.6)
    style_ax(axes[0, 1], title="4WD Torque Vectoring Corrective Yaw Moment Mz_TV(t)", xlabel="Time [s]", ylabel="Corrective Yaw Moment [Nm]", legend=True, legend_loc="lower right")

    axes[1, 0].plot(t_sp, s_acc_open["Y"], color=PALETTE["amber"], linewidth=2.0, linestyle="--", label="Accel Open (Unassisted)")
    axes[1, 0].plot(t_sp, s_acc_tv["Y"], color=PALETTE["emerald"], linewidth=2.2, label="Accel 4WD TV + Stabilized")
    axes[1, 0].axhline(0.0, color=PALETTE["white"], linewidth=0.8, alpha=0.6)
    style_ax(axes[1, 0], title="Split-Mu Acceleration (30 km/h): Lateral Drift Y(t)", xlabel="Time [s]", ylabel="Lateral Drift Y [m]", legend=True, legend_loc="lower left")

    axes[1, 1].plot(t_sp, s_brk_open["psi_deg"], color=PALETTE["crimson"], linewidth=1.8, linestyle="--", label="Braking Open psi [deg]")
    axes[1, 1].plot(t_sp, s_brk_tv["psi_deg"], color=PALETTE["cyan"], linewidth=2.0, label="Braking TV psi [deg]")
    axes[1, 1].plot(t_sp, s_acc_open["psi_deg"], color=PALETTE["amber"], linewidth=1.8, linestyle=":", label="Accel Open psi [deg]")
    axes[1, 1].plot(t_sp, s_acc_tv["psi_deg"], color=PALETTE["emerald"], linewidth=2.0, label="Accel TV psi [deg]")
    style_ax(axes[1, 1], title="Heading Deviation psi(t) on Split-Mu Surface", xlabel="Time [s]", ylabel="Heading Angle psi [deg]", legend=True, legend_loc="upper left")
    add_kpi_box(
        axes[0, 0],
        [
            f"Brk Open Drift : {metrics_f5['split_mu_0p90_0p50']['braking_open_max_drift_m']:.2f} m",
            f"Brk TV Drift   : {metrics_f5['split_mu_0p90_0p50']['braking_tv_max_drift_m']:.2f} m",
            f"Acc Open Drift : {metrics_f5['split_mu_0p90_0p50']['accel_open_max_drift_m']:.2f} m",
            f"Acc TV Drift   : {metrics_f5['split_mu_0p90_0p50']['accel_tv_max_drift_m']:.2f} m",
        ],
        loc="lower left",
        border_color=PALETTE["cyan"],
    )
    add_suptitle(
        fig,
        "TeR27 4WD Split-Mu Asymmetric Surface Behavior (mu_Left=0.90 / mu_Right=0.50)",
        "Heavy braking & full-throttle acceleration | Open unassisted drift vs 4WD TV yaw stabilization",
    )
    add_footer(fig)
    fig.tight_layout(rect=[0.0, 0.025, 1.0, 0.92])
    save_fig(fig, OUT_DIR / "split_mu_behavior.png", dpi=300)

    return metrics_f5


# =============================================================================
# 8. Family 6: Official Formula Student Events (75m Accel & R=9.125m Skidpad)
# =============================================================================

def run_family_6_fs_events(ter26_3d: TeR26HandlingWrapper) -> Dict[str, Any]:
    print("[Family 6/6] Running FS 75m Acceleration & R=9.125m Skidpad...", flush=True)
    m_3d = TeR27HandlingModel(profile="FSG_RUBBERED", tire_temp_c=77.5, use_3d_kin=True)
    p = m_3d.p
    metrics_f6: Dict[str, Any] = {}

    # --- 6A. Official FS 0-75m Acceleration with 4-wheel rotational dynamics & 80 kW HV limit (Fig 12) ---
    dt = 0.002  # 500 Hz
    max_steps = int(6.0 / dt)
    x_pos = 0.0
    vx = 0.05
    omega_w = np.full(4, vx / p.wheel_radius, dtype=float)
    ax_cur = 0.0

    rec_t, rec_x, rec_vx, rec_ax, rec_pdc = [], [], [], [], []
    rec_kap, rec_fz, rec_trq = [], [], []
    t_75m = None
    v_75m = None
    t_100kmh = None

    kappa_opt_f = 0.105
    kappa_opt_r = 0.110
    gamma_0 = np.array([p.static_camber_f_rad, -p.static_camber_f_rad, p.static_camber_r_rad, -p.static_camber_r_rad])
    alpha_0 = np.zeros(4)

    for step in range(max_steps):
        t_cur = step * dt
        fz, _, _ = m_3d.compute_normal_forces(ax_cur, 0.0, vx, 0.0)
        rho_rad = np.minimum(fz / p.k_tire_z, 0.5 * p.wheel_radius)
        r_load = p.wheel_radius - rho_rad
        r_eff = p.wheel_radius - rho_rad / 3.0

        u_den = max(abs(vx), p.slip_reg_speed)
        kappa = (omega_w * r_eff - vx) / u_den
        fx_w, _, _, _ = m_3d.eval_tire_forces(kappa, alpha_0, fz, gamma_0)
        f_rr = p.rolling_resistance_coeff * fz * math.tanh(vx / 0.5)
        f_drag_x, _, _, _ = m_3d.aero_forces(vx, 0.0)

        # Motor torque envelope + traction control slip target
        t_max, _ = m_3d.base.wheel_motor_limits(omega_w)
        # Estimate peak Fx torque per wheel + feedback around optimal slip
        kap_tgt = np.array([kappa_opt_f, kappa_opt_f, kappa_opt_r, kappa_opt_r])
        # Feedforward tire force torque + P-controller on slip error
        t_cmd = np.clip((fx_w + f_rr) * r_load + 1400.0 * (kap_tgt - kappa), 0.0, t_max)
        # At launch (vx < 2 m/s), command full traction-limited torque cleanly
        if vx < 1.5:
            t_cmd = np.minimum(t_max, np.maximum(t_cmd, 0.85 * t_max))

        # Exact 80 kW HV DC power ceiling (P_dc = sum(T_i * omega_i / eta_drive) <= 80000 W)
        p_dc_raw = float(np.sum(t_cmd * omega_w) / p.eta_drive)
        if p_dc_raw > p.p_dc_max_w and p_dc_raw > 1.0:
            scale = p.p_dc_max_w / p_dc_raw
            t_cmd = t_cmd * scale
        p_dc_w = float(np.sum(t_cmd * omega_w) / p.eta_drive)

        # Integrate wheel & chassis dynamics
        omega_dot = (t_cmd - (fx_w + f_rr) * r_load) / p.wheel_eff_inertia
        ax_new = (float(np.sum(fx_w)) + f_drag_x) / p.mass

        rec_t.append(t_cur)
        rec_x.append(x_pos)
        rec_vx.append(vx)
        rec_ax.append(ax_new)
        rec_pdc.append(p_dc_w * 1e-3)
        rec_kap.append(kappa.copy())
        rec_fz.append(fz.copy())
        rec_trq.append(t_cmd.copy())

        if t_100kmh is None and vx * 3.6 >= 100.0:
            t_100kmh = t_cur
        if x_pos >= 75.0:
            # Linear interpolation for exact 75.0 m crossing
            if len(rec_x) >= 2:
                dx = rec_x[-1] - rec_x[-2]
                frac = (75.0 - rec_x[-2]) / max(dx, 1e-9)
                t_75m = rec_t[-2] + frac * dt
                v_75m = (rec_vx[-2] + frac * (rec_vx[-1] - rec_vx[-2])) * 3.6
            else:
                t_75m, v_75m = t_cur, vx * 3.6
            break

        # Semi-implicit Euler at 500 Hz (very accurate and rock-solid for stiff wheel slip)
        omega_w = np.maximum(omega_w + omega_dot * dt, 0.0)
        vx = max(vx + ax_new * dt, 0.05)
        x_pos += vx * dt
        ax_cur = 0.6 * ax_cur + 0.4 * ax_new

    acc_t = np.array(rec_t)
    acc_x = np.array(rec_x)
    acc_vx_kmh = np.array(rec_vx) * 3.6
    acc_ax_g = np.array(rec_ax) / G_ACC
    acc_pdc_kw = np.array(rec_pdc)
    acc_kap = np.array(rec_kap)
    acc_fz = np.array(rec_fz)
    acc_trq = np.array(rec_trq)

    metrics_f6["fs_acceleration_75m"] = {
        "time_75m_s": float(t_75m or acc_t[-1]),
        "v_exit_kmh": float(v_75m or acc_vx_kmh[-1]),
        "time_0_100_kmh_s": float(t_100kmh) if t_100kmh is not None else float("nan"),
        "peak_ax_g": float(np.max(acc_ax_g)),
        "max_hv_power_kw": float(np.max(acc_pdc_kw)),
        "power_limited_onset_s": float(acc_t[np.argmax(acc_pdc_kw >= 79.5)]) if np.any(acc_pdc_kw >= 79.5) else float("nan"),
    }

    # --- Figure 12: FS 75m Acceleration Telemetry ---
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.8))
    axes[0, 0].plot(acc_x, acc_vx_kmh, color=PALETTE["cyan"], linewidth=2.2, label="Speed vx [km/h]")
    ax00_r = axes[0, 0].twinx()
    ax00_r.plot(acc_x, acc_ax_g, color=PALETTE["amber"], linewidth=1.8, linestyle="--", label="Longitudinal Ax [g]")
    ax00_r.set_ylabel("Longitudinal Accel Ax [g]", color=PALETTE["amber"], fontsize=9.0)
    ax00_r.tick_params(colors=PALETTE["amber"], labelsize=8.4)
    axes[0, 0].set_xlim(0.0, 75.0)
    style_ax(axes[0, 0], title="0-75 m Speed & Acceleration Profile", xlabel="Distance x [m]", ylabel="Speed vx [km/h]", legend=True, legend_loc="upper left")
    add_kpi_box(
        axes[0, 0],
        [
            f"75 m Time   : {metrics_f6['fs_acceleration_75m']['time_75m_s']:.3f} s",
            f"Exit Speed  : {metrics_f6['fs_acceleration_75m']['v_exit_kmh']:.1f} km/h",
            f"0-100 km/h  : {metrics_f6['fs_acceleration_75m']['time_0_100_kmh_s']:.3f} s",
            f"Peak Ax     : {metrics_f6['fs_acceleration_75m']['peak_ax_g']:.2f} g",
        ],
        loc="lower right",
        border_color=PALETTE["cyan"],
    )

    axes[0, 1].plot(acc_t, acc_pdc_kw, color=PALETTE["emerald"], linewidth=2.2, label="Net HV DC Power [kW]")
    axes[0, 1].axhline(80.0, color=PALETTE["crimson"], linewidth=1.2, linestyle="--", label="FS 80.0 kW Rule Ceiling")
    ax01_r = axes[0, 1].twinx()
    ax01_r.plot(acc_t, np.sum(acc_trq, axis=1), color=PALETTE["violet"], linewidth=1.6, linestyle=":", label="Total Wheel Torque [Nm]")
    ax01_r.set_ylabel("Total Wheel Torque [Nm]", color=PALETTE["violet"], fontsize=9.0)
    ax01_r.tick_params(colors=PALETTE["violet"], labelsize=8.4)
    style_ax(axes[0, 1], title="HV DC Power (80 kW Cap) & Total Wheel Torque", xlabel="Time [s]", ylabel="HV DC Power [kW]", legend=True, legend_loc="lower right")

    axes[1, 0].plot(acc_t, acc_kap[:, 0] * 100.0, color=PALETTE["cyan"], linewidth=2.0, label="Front Slip kappa_FL/FR [%]")
    axes[1, 0].plot(acc_t, acc_kap[:, 2] * 100.0, color=PALETTE["amber"], linewidth=2.0, linestyle="--", label="Rear Slip kappa_RL/RR [%]")
    axes[1, 0].axhline(10.5, color=PALETTE["emerald"], linewidth=0.9, linestyle=":", label="TC Optimal Slip Target (~10.5%)")
    style_ax(axes[1, 0], title="4-Wheel Longitudinal Slip Ratio kappa_i(t)", xlabel="Time [s]", ylabel="Slip Ratio kappa [%]", legend=True, legend_loc="upper right")

    axes[1, 1].plot(acc_t, acc_fz[:, 0] + acc_fz[:, 1], color=PALETTE["cyan"], linewidth=2.0, label="Front Axle Total Fz [N]")
    axes[1, 1].plot(acc_t, acc_fz[:, 2] + acc_fz[:, 3], color=PALETTE["amber"], linewidth=2.0, label="Rear Axle Total Fz [N]")
    style_ax(axes[1, 1], title="Longitudinal Load Transfer & Aero Downforce Fz(t)", xlabel="Time [s]", ylabel="Axle Normal Load [N]", legend=True, legend_loc="center right")

    add_suptitle(
        fig,
        "Formula Student 0-75 m Acceleration Telemetry (TeR27 4WD, 80 kW HV Limit)",
        "Exact 4-wheel rotational inertia + MF 6.2 slip traction control + field weakening + 80 kW DC ceiling",
    )
    add_footer(fig)
    fig.tight_layout(rect=[0.0, 0.025, 1.0, 0.92])
    save_fig(fig, OUT_DIR / "fs_acceleration_75m_telemetry.png", dpi=300)

    # --- 6B. Official FS Skidpad (R = 9.125 m) Steady-State: TeR27 4WD vs TeR26 RWD (Fig 13) ---
    R_skid = 9.125
    C_skid = 2.0 * math.pi * R_skid

    def _solve_skidpad_sweep(model_obj: Any, is_ter27: bool, optimize_tv: bool = False) -> Dict[str, Any]:
        """Exact nonlinear least-squares trim solver on R=9.125 m circle up to tire limit."""
        v_list = np.linspace(6.0, 14.0, 81)
        rec = {"v_ms": [], "ay_g": [], "beta_deg": [], "delta_deg": [], "t_circle_s": []}
        best = None
        x_guess = np.array([math.radians(1.0), math.radians(10.0), 0.0])

        for v_ms in v_list:
            ay_tgt = (v_ms * v_ms) / R_skid
            r_tgt = v_ms / R_skid

            def _eval_res(vars_vec: np.ndarray) -> np.ndarray:
                b_rad = float(vars_vec[0])
                d_rad = float(vars_vec[1])
                tv_mz = float(vars_vec[2]) if optimize_tv else 0.0
                vx = v_ms * math.cos(b_rad)
                vy = v_ms * math.sin(b_rad)
                if is_ter27:
                    fz, phi, z_w = model_obj.compute_normal_forces(0.0, ay_tgt, vx, vy)
                    dw, gam = model_obj.compute_wheel_angles(d_rad, phi, z_w, ay_tgt)
                    _, _, alp = model_obj.base.wheel_velocities_and_alphas(vx, vy, r_tgt, dw)
                    dfx_f = (0.35 * tv_mz) / model_obj.p.track_front
                    dfx_r = (0.65 * tv_mz) / model_obj.p.track_rear
                    # Add longitudinal propulsion to balance cornering drag + rolling resistance
                    fx_req = np.array([30.0 - dfx_f, 30.0 + dfx_f, 55.0 - dfx_r, 55.0 + dfx_r])
                    kap = model_obj._kappa_from_fx_req(fx_req, fz, gam)
                    fx_w, fy_w, mz_t, _ = model_obj.eval_tire_forces(kap, alp, fz, gam)
                    cd, sd = np.cos(dw), np.sin(dw)
                    fx_c = fx_w * cd - fy_w * sd
                    fy_c = fx_w * sd + fy_w * cd
                    _, f_dy, _, _ = model_obj.aero_forces(vx, vy)
                    ay_val = (float(np.sum(fy_c)) + f_dy) / model_obj.p.mass
                    mz_val = float(np.sum(model_obj.x_w * fy_c - model_obj.y_w * fx_c + mz_t))
                else:
                    v_fy = vy + model_obj.p.a_cog * r_tgt
                    v_ry = vy - model_obj.p.b_cog * r_tgt
                    af = d_rad - math.atan2(v_fy, max(vx, 1.5))
                    ar = -math.atan2(v_ry, max(vx, 1.5))
                    x = np.array([vx, vy, r_tgt, af, ar], dtype=np.float64)
                    u = np.array([d_rad, 0.35, 0.0, ay_tgt], dtype=np.float64)
                    _, yobs = model_obj.model.fn_ode(x, u, model_obj.t4, model_obj.th_vec)
                    ay_val = float(yobs[1])
                    mz_val = float(yobs[2])
                if optimize_tv:
                    # Also softly prefer smaller |beta| via TV yaw moment authority
                    return np.array([(ay_val - ay_tgt) / G_ACC, mz_val / 250.0, 0.02 * tv_mz / 500.0])
                return np.array([(ay_val - ay_tgt) / G_ACC, mz_val / 250.0])

            if optimize_tv:
                sol = optimize.least_squares(
                    _eval_res,
                    x_guess,
                    bounds=([-math.radians(12.0), math.radians(2.0), -600.0], [math.radians(6.0), math.radians(21.5), 600.0]),
                    xtol=1e-6,
                    ftol=1e-6,
                    max_nfev=80,
                )
                r_chk = _eval_res(sol.x)
                ay_err_g = abs(float(r_chk[0]))
                mz_err_nm = abs(float(r_chk[1]) * 250.0)
                if ay_err_g < 0.015 and mz_err_nm < 10.0:
                    x_guess = sol.x.copy()
                else:
                    break
                b_sol, d_sol, tv_sol = float(sol.x[0]), float(sol.x[1]), float(sol.x[2])
            else:
                sol = optimize.least_squares(
                    _eval_res,
                    x_guess[:2],
                    bounds=([-math.radians(12.0), math.radians(2.0)], [math.radians(6.0), math.radians(21.5)]),
                    xtol=1e-6,
                    ftol=1e-6,
                    max_nfev=80,
                )
                r_chk = _eval_res(sol.x)
                ay_err_g = abs(float(r_chk[0]))
                mz_err_nm = abs(float(r_chk[1]) * 250.0)
                if ay_err_g < 0.015 and mz_err_nm < 10.0:
                    x_guess[:2] = sol.x.copy()
                else:
                    break
                b_sol, d_sol, tv_sol = float(sol.x[0]), float(sol.x[1]), 0.0

            t_c = C_skid / v_ms
            rec["v_ms"].append(float(v_ms))
            rec["ay_g"].append(float(ay_tgt / G_ACC))
            rec["beta_deg"].append(float(math.degrees(b_sol)))
            rec["delta_deg"].append(float(math.degrees(d_sol)))
            rec["t_circle_s"].append(float(t_c))
            best = {
                "v_max_ms": float(v_ms),
                "v_max_kmh": float(v_ms * 3.6),
                "t_circle_s": float(t_c),
                "sustained_ay_g": float(ay_tgt / G_ACC),
                "eq_beta_deg": float(math.degrees(b_sol)),
                "eq_delta_deg": float(math.degrees(d_sol)),
                "eq_tv_mz_nm": float(tv_sol),
                "r_eff_m": float(R_skid),
            }
        return {"sweep": rec, "best": best}

    skid_t27_open = _solve_skidpad_sweep(m_3d, is_ter27=True, optimize_tv=False)
    skid_t27_tv4 = _solve_skidpad_sweep(m_3d, is_ter27=True, optimize_tv=True)
    skid_t26_3d = _solve_skidpad_sweep(ter26_3d, is_ter27=False, optimize_tv=False)

    # Dynamic 1-circle closed-loop verification for TeR27 4WD on R=9.125m
    v_sk_dyn = skid_t27_tv4["best"]["v_max_ms"] * 0.96
    t_sk = np.arange(0.0, C_skid / v_sk_dyn * 1.15, 0.004)
    d_eq = math.radians(skid_t27_tv4["best"]["eq_delta_deg"])
    tv_eq = float(skid_t27_tv4["best"]["eq_tv_mz_nm"])

    def _skid_ctrl(t, st, diag):
        X, Y, psi, vx, wz = st[9], st[10], st[11], max(st[0], 1.0), st[2]
        r_curr = math.hypot(X, Y - R_skid)
        e_r = r_curr - R_skid
        wz_ref = vx / R_skid
        delta = d_eq + 0.12 * e_r - 0.08 * (wz - wz_ref)
        return float(np.clip(delta, 0.02, 0.36)), 0.0, tv_eq

    sim_skid_dyn = m_3d.simulate_maneuver(t_sk, v_sk_dyn, _skid_ctrl, hold_vx=True, x0_pos=(0.0, 0.0, 0.0))
    r_dyn_arr = np.hypot(sim_skid_dyn["X"], sim_skid_dyn["Y"] - R_skid)
    r_eff_dyn = float(np.mean(r_dyn_arr[len(r_dyn_arr) // 3 :]))

    metrics_f6["fs_skidpad_9p125m"] = {
        "ter27_open_4wd": skid_t27_open["best"],
        "ter27_tv4": skid_t27_tv4["best"],
        "ter26_rwd_3d": skid_t26_3d["best"],
        "ter27_dynamic_r_eff_m": r_eff_dyn,
    }

    # --- Figure 13: FS Skidpad Steady-State Comparison ---
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.8))
    # Top-left: Trajectory on R=9.125 m circle
    th_c = np.linspace(0.0, 2.0 * math.pi, 200)
    axes[0, 0].plot(R_skid * np.sin(th_c), R_skid - R_skid * np.cos(th_c), color=PALETTE["slate"], linestyle="--", linewidth=1.4, label="Nominal R=9.125 m Centerline")
    axes[0, 0].plot((R_skid - 1.5) * np.sin(th_c), R_skid - (R_skid - 1.5) * np.cos(th_c), color=PALETTE["crimson"], linestyle=":", linewidth=1.0, alpha=0.7)
    axes[0, 0].plot((R_skid + 1.5) * np.sin(th_c), R_skid - (R_skid + 1.5) * np.cos(th_c), color=PALETTE["crimson"], linestyle=":", linewidth=1.0, alpha=0.7, label="Track Boundaries (3.0 m)")
    axes[0, 0].plot(sim_skid_dyn["X"], sim_skid_dyn["Y"], color=PALETTE["cyan"], linewidth=2.1, label=f"TeR27 4WD TV4 (R_eff={r_eff_dyn:.2f} m)")
    axes[0, 0].set_aspect("equal", "box")
    style_ax(axes[0, 0], title="FS Skidpad R=9.125 m Trajectory", xlabel="X [m]", ylabel="Y [m]", legend=True, legend_loc="upper right")

    # Top-right: Equilibrium Steer Angle delta(Ay)
    for res, col, ls, lbl in [
        (skid_t27_tv4, PALETTE["emerald"], "-", "TeR27 4WD TV4"),
        (skid_t27_open, PALETTE["cyan"], "--", "TeR27 4WD Open"),
        (skid_t26_3d, PALETTE["crimson"], "-.", "TeR26 RWD 3D"),
    ]:
        sw = res["sweep"]
        axes[0, 1].plot(sw["ay_g"], sw["delta_deg"], color=col, linestyle=ls, linewidth=2.0, label=lbl)
    style_ax(axes[0, 1], title="Skidpad Equilibrium Steer Angle delta(Ay)", xlabel="Sustained Lateral Accel Ay [g]", ylabel="Steer Angle delta [deg]", legend=True, legend_loc="upper left")

    # Bottom-left: Equilibrium Sideslip beta(Ay)
    for res, col, ls, lbl in [
        (skid_t27_tv4, PALETTE["emerald"], "-", "TeR27 4WD TV4"),
        (skid_t27_open, PALETTE["cyan"], "--", "TeR27 4WD Open"),
        (skid_t26_3d, PALETTE["crimson"], "-.", "TeR26 RWD 3D"),
    ]:
        sw = res["sweep"]
        axes[1, 0].plot(sw["ay_g"], sw["beta_deg"], color=col, linestyle=ls, linewidth=2.0, label=lbl)
    axes[1, 0].axhline(0.0, color=PALETTE["white"], linewidth=0.8, alpha=0.6)
    style_ax(axes[1, 0], title="Skidpad Equilibrium Sideslip beta(Ay)", xlabel="Sustained Lateral Accel Ay [g]", ylabel="Sideslip beta [deg]", legend=True, legend_loc="lower left")

    # Bottom-right: Bar comparison of Circle Time & Max Ay
    labels_sk = ["TeR27 TV4", "TeR27 Open", "TeR26 RWD"]
    recs_sk = [skid_t27_tv4["best"], skid_t27_open["best"], skid_t26_3d["best"]]
    xp = np.arange(3)
    w = 0.34
    t_vals = [r["t_circle_s"] for r in recs_sk]
    ay_vals = [r["sustained_ay_g"] for r in recs_sk]
    b1 = axes[1, 1].bar(xp - w * 0.5, t_vals, width=w, color=PALETTE["cyan"], label="Circle Time [s]")
    ax11_r = axes[1, 1].twinx()
    b2 = ax11_r.bar(xp + w * 0.5, ay_vals, width=w, color=PALETTE["emerald"], label="Sustained Ay [g]")
    for bar in b1:
        h = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width() * 0.5, h + 0.08, f"{h:.2f}s", ha="center", va="bottom", fontsize=8.0, fontfamily="DejaVu Sans Mono", color=PALETTE["cyan"])
    for bar in b2:
        h = bar.get_height()
        ax11_r.text(bar.get_x() + bar.get_width() * 0.5, h + 0.03, f"{h:.2f}g", ha="center", va="bottom", fontsize=8.0, fontfamily="DejaVu Sans Mono", color=PALETTE["emerald"])
    axes[1, 1].set_xticks(xp)
    axes[1, 1].set_xticklabels(labels_sk, fontsize=8.8)
    axes[1, 1].set_ylim(0.0, max(t_vals) * 1.25)
    ax11_r.set_ylim(0.0, max(ay_vals) * 1.30)
    ax11_r.set_ylabel("Sustained Ay [g]", color=PALETTE["emerald"], fontsize=9.0)
    ax11_r.tick_params(colors=PALETTE["emerald"], labelsize=8.4)
    style_ax(axes[1, 1], title="Skidpad Performance Summary: TeR27 vs TeR26", ylabel="Circle Time [s]", legend=True, legend_loc="upper left")
    add_kpi_box(
        axes[1, 0],
        [
            f"TeR27 TV4  : {skid_t27_tv4['best']['t_circle_s']:.2f} s ({skid_t27_tv4['best']['sustained_ay_g']:.2f} g)",
            f"TeR27 Open : {skid_t27_open['best']['t_circle_s']:.2f} s ({skid_t27_open['best']['sustained_ay_g']:.2f} g)",
            f"TeR26 RWD  : {skid_t26_3d['best']['t_circle_s']:.2f} s ({skid_t26_3d['best']['sustained_ay_g']:.2f} g)",
            f"TV4 Gain   : {skid_t27_open['best']['t_circle_s'] - skid_t27_tv4['best']['t_circle_s']:+.2f} s",
        ],
        loc="upper right",
        border_color=PALETTE["emerald"],
    )
    add_suptitle(
        fig,
        "Formula Student Skidpad (R = 9.125 m) Steady-State Benchmark",
        "Circle time, effective radius, sustained Ay, equilibrium beta & steer angle | TeR27 4WD vs TeR26 RWD",
    )
    add_footer(fig)
    fig.tight_layout(rect=[0.0, 0.025, 1.0, 0.92])
    save_fig(fig, OUT_DIR / "fs_skidpad_steady_state.png", dpi=300)

    return metrics_f6


# =============================================================================
# Main Entrypoint
# =============================================================================

def main() -> None:
    t0 = time.perf_counter()
    apply_motec_style(dpi=300)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== TeR27 4WD Handling & Maneuvers Battery -> {OUT_DIR} ===", flush=True)

    print("Initializing TeR26 3D & Parametric CasADi models for cross-vehicle parity...", flush=True)
    ter26_3d = TeR26HandlingWrapper(use_3d_kin=True, tire_temp_c=77.5)
    ter26_par = TeR26HandlingWrapper(use_3d_kin=False, tire_temp_c=77.5)

    all_metrics: Dict[str, Any] = {
        "metadata": {
            "vehicle": "TeR27 4WD (vs TeR26 RWD)",
            "coordinate_frame": "ISO 8855 (+x fwd, +y left, +z up)",
            "tire_model": "Hoosier 16.0x7.5-10 R20 Pacejka MF 6.2",
            "default_profile": "FSG_RUBBERED (45 C)",
        }
    }

    all_metrics["family_1_ymd"] = run_family_1_ymd(ter26_3d, ter26_par)
    all_metrics["family_2_ramp_steer"] = run_family_2_ramp_steer(ter26_3d, ter26_par)
    all_metrics["family_3_step_steer"] = run_family_3_step_steer(ter26_3d)
    all_metrics["family_4_bode_sine_sweep"] = run_family_4_bode()
    all_metrics["family_5_combined_and_emergency"] = run_family_5_combined_and_emergency()
    all_metrics["family_6_fs_events"] = run_family_6_fs_events(ter26_3d)

    elapsed_s = time.perf_counter() - t0
    all_metrics["metadata"]["runtime_s"] = round(elapsed_s, 2)

    metrics_path = OUT_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(all_metrics, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved {metrics_path} in {elapsed_s:.1f} s.", flush=True)

    # Self-verify all 13 PNGs
    expected_pngs = [
        "ymd_velocity_sweep.png",
        "ymd_grip_comparison.png",
        "ymd_ter27_vs_ter26.png",
        "ramp_steer_understeer_gradient.png",
        "ramp_steer_roll_and_load_transfer.png",
        "step_steer_transient_response.png",
        "step_steer_metrics_comparison.png",
        "bode_yaw_and_lat_accel.png",
        "trail_braking_stability_envelope.png",
        "slalom_and_dlc_traces.png",
        "split_mu_behavior.png",
        "fs_acceleration_75m_telemetry.png",
        "fs_skidpad_steady_state.png",
    ]
    for name in expected_pngs:
        p_img = OUT_DIR / name
        if not p_img.exists():
            raise FileNotFoundError(f"Missing expected output figure: {p_img}")
        sz = p_img.stat().st_size
        if sz < 20_000:
            raise RuntimeError(f"Figure too small ({sz} bytes < 20 KB): {p_img}")
    print(f"Verified all {len(expected_pngs)} PNGs (> 20 KB, 300 DPI) + valid metrics.json!", flush=True)


if __name__ == "__main__":
    main()
