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
    sabs,
    sclip,
    smax,
    smin,
    sneg,
    spos,
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


ACKERMANN_PCT_BOUNDS: Tuple[float, float] = (0.0, 100.0)
DAMP_RATIO_BOUNDS: Tuple[float, float] = (0.55, 1.35)
NOMINAL_DAMP_REBOUND_NS_M: float = 3000.0
NOMINAL_DAMP_COMP_NS_M: float = 2000.0
NOMINAL_DAMP_RATIO: float = NOMINAL_DAMP_COMP_NS_M / NOMINAL_DAMP_REBOUND_NS_M  # 0.6667

# Real In-Hub Electrical Loss Map constants (1:1 parity with alqp_power_abs_pitch.py InHubLossModel)
INHUB_LOSS_PARAMS: Dict[str, float] = {
    "k_t_nm_per_a": 0.26,
    "r_phase_25c_ohm": 0.135,
    "alpha_cu_per_k": 0.00393,
    "v_ce0_v": 0.85,
    "r_on_ohm": 0.012,
    "f_sw_hz": 16000.0,
    "k_sw_j_per_a": 1.85e-4,
    "i_d_max_a": 55.0,
    "k_fw": 1.0,
    "c_iron_w_s2": 1.2e-4,
    "eta_gear": 0.965,
    "t_motor_front_c": 50.0,
    "t_motor_rear_c": 75.0,
}


class SkidpadSetupCasADiVehicle(TeR27CasADiVehicle):
    """
    TeR27CasADiVehicle extension adding:
      1. Smooth static front/rear toe angles (toe_f_deg, toe_r_deg; positive = toe-in)
      2. Dynamic Ackermann percentage (ackermann_pct in [0.0, 100.0]%, defaulting to
         p.ackermann_factor * 100.0 = -11.0% when unspecified)
      3. Asymmetric front/rear bump-to-rebound damping ratios (damp_ratio_f, damp_ratio_r
         in [0.55, 1.35], where < 1.0 is softer bump / stiffer rebound) active ONLY during
         transient load transfer (ay_dot != 0 or ax_dot != 0; identically 0 in steady state)
      4. Static rake (rake_mm in [-8, +8] mm)
    without altering base ter27_casadi_model.py.
    """

    def ackermann_steer_smooth(self, delta: Any) -> Any:
        p = self.params
        ack_pct = self._dp("ackermann_pct", float(p.ackermann_factor) * 100.0)
        ack = ack_pct * 0.01
        L = p.wheelbase
        half_t = 0.5 * p.track_front
        tan_d = ca.tan(delta)
        abs_tan_d = sabs(tan_d, 1e-6)
        min_den = 0.25 * L * abs_tan_d
        eps = self.smoothing["eps_ack_denom"]
        den_fl = smax(L - half_t * tan_d, min_den, eps)
        den_fr = smax(L + half_t * tan_d, min_den, eps)
        d_fl_ack = ca.atan2(L * tan_d, den_fl)
        d_fr_ack = ca.atan2(L * tan_d, den_fr)
        d_fl = delta + ack * (d_fl_ack - delta)
        d_fr = delta + ack * (d_fr_ack - delta)

        toe_f_deg = self._dp("toe_f_deg", 0.0)
        toe_r_deg = self._dp("toe_r_deg", 0.0)
        toe_f_rad = toe_f_deg * (math.pi / 180.0)
        toe_r_rad = toe_r_deg * (math.pi / 180.0)
        # ISO 8855: positive toe-in -> left wheel points right (-), right wheel points left (+)
        return ca.vertcat(
            d_fl - toe_f_rad,
            d_fr + toe_f_rad,
            -toe_r_rad,
            toe_r_rad,
        )

    def eval_hv_power_kw(self, T_4: Any, omega_4: Any) -> Tuple[Any, Any]:
        """
        HV DC electrical power (kW).
        When design_params['use_real_electrical_loss_map'] is True, uses the smooth C^2
        InHubLossModel (1:1 parity with alqp_power_abs_pitch.py):
          P_cu(T) = 3 * I_rms^2 * R_phase(T), P_inv = (3*V_ce0 + k_sw*f_sw)*I_rms + 3*R_on*I_rms^2,
          P_fw via I_d above omega_base, P_mech_loss = c_iron*om_m^2 + (1-eta_gear)*|T_w*om_w|.
        Otherwise falls back to base TeR27CasADiVehicle flat efficiency (eta_drive=0.90).
        """
        if not bool(self._dp("use_real_electrical_loss_map", False)):
            return super().eval_hv_power_kw(T_4, omega_4)

        p = self.params
        lp = INHUB_LOSS_PARAMS
        n_g = self._dp("gear_ratio", p.gear_ratio)
        k_t = lp["k_t_nm_per_a"]
        v_scale = float(np.clip(p.v_oc_nom / p.v_nom, 0.5, 1.25))
        om_base = p.base_speed_rpm * (2.0 * math.pi / 60.0) * v_scale

        t_f_c = float(self._dp("t_motor_front_c", lp["t_motor_front_c"]))
        t_r_c = float(self._dp("t_motor_rear_c", lp["t_motor_rear_c"]))
        t_mot_vec = ca.DM([t_f_c, t_f_c, t_r_c, t_r_c])
        r_ph_vec = lp["r_phase_25c_ohm"] * (1.0 + lp["alpha_cu_per_k"] * (t_mot_vec - 25.0))

        i_q = sabs(T_4, 0.05) / (n_g * k_t)
        om_m = sabs(omega_4, 0.05) * n_g
        om_m_safe = smax(om_m, 1.0, 0.5)
        fw_frac = spos(1.0 - om_base / om_m_safe, 1e-3)
        i_d = lp["k_fw"] * fw_frac * lp["i_d_max_a"]
        i_rms_sq = i_q * i_q + i_d * i_d + 1e-4
        i_rms = ca.sqrt(i_rms_sq)

        p_cu = 3.0 * i_rms_sq * r_ph_vec
        p_inv = (3.0 * lp["v_ce0_v"] + lp["k_sw_j_per_a"] * lp["f_sw_hz"]) * i_rms + 3.0 * lp["r_on_ohm"] * i_rms_sq
        p_mech = T_4 * omega_4
        p_mech_loss = lp["c_iron_w_s2"] * om_m * om_m + (1.0 - lp["eta_gear"]) * sabs(p_mech, 10.0)
        p_dc_w = ca.sum1(p_mech + p_cu + p_inv + p_mech_loss)
        p_dc_kw = p_dc_w * 1e-3
        return p_dc_kw, p_dc_kw

    def normal_forces_smooth(
        self,
        ax: Any,
        ay: Any,
        vx: Any,
        vy: Any,
        ay_dot: Any = 0.0,
        ax_dot: Any = 0.0,
    ) -> Tuple[Any, Any]:
        """
        Smooth C^2 4-wheel normal forces including optional transient asymmetric
        bump/rebound damping load transfer when ay_dot != 0 (identically zero at ay_dot = 0).
        """
        p = self.params
        eps_fz = self.smoothing["eps_fz_n"]
        _, _, f_down_f, f_down_r = self.aero_forces_smooth(vx, vy, ax)

        mass = self._dp("mass_kg", p.mass)
        h_cg = self._dp("h_cg", p.h_cog)
        h_rc_f = self._dp("h_rc_f", p.h_rc_f)
        h_rc_r = self._dp("h_rc_r", p.h_rc_r)
        lltd = self._dp("lltd", p.lltd_elastic_f)
        k_roll_scale = self._dp("k_roll_scale", 1.0)

        mg = mass * p.gravity
        fz_f_stat = mg * (p.b_cog / p.wheelbase)
        fz_r_stat = mg * (p.a_cog / p.wheelbase)

        dfz_long = (mass * ax * h_cg) / p.wheelbase
        fz_f_tot = spos(fz_f_stat - dfz_long + f_down_f, eps_fz)
        fz_r_tot = spos(fz_r_stat + dfz_long + f_down_r, eps_fz)

        m_s = mass - p.unsprung_mass_total
        m_u_axle = 0.5 * p.unsprung_mass_total
        k_roll_tot = (p.k_roll_f + p.k_roll_r) * k_roll_scale
        k_roll_f = lltd * k_roll_tot
        k_roll_r = (1.0 - lltd) * k_roll_tot

        h_s = p.h_sprung_cog + (h_cg - p.h_cog)
        h_rc_cg = h_rc_f + (h_rc_r - h_rc_f) * (p.a_cog / p.wheelbase)
        h_roll_arm = smax(h_s - h_rc_cg, 0.0, 1e-5)

        roll_denom = smax(k_roll_tot - m_s * p.gravity * h_roll_arm, 1.0, 1e-3)
        phi = (m_s * ay * h_roll_arm) / roll_denom

        dfz_lat_f_raw = (
            (k_roll_f * phi) / p.track_front
            + (m_s * (p.b_cog / p.wheelbase) * ay * h_rc_f) / p.track_front
            + (m_u_axle * ay * p.wheel_radius) / p.track_front
        )
        dfz_lat_r_raw = (
            (k_roll_r * phi) / p.track_rear
            + (m_s * (p.a_cog / p.wheelbase) * ay * h_rc_r) / p.track_rear
            + (m_u_axle * ay * p.wheel_radius) / p.track_rear
        )

        # Transient asymmetric bump/rebound damping dynamics (active only when ay_dot != 0)
        has_transient = not (isinstance(ay_dot, (int, float)) and abs(float(ay_dot)) < 1e-12)
        if has_transient:
            rd_f = self._dp("damp_ratio_f", NOMINAL_DAMP_RATIO)
            rd_r = self._dp("damp_ratio_r", NOMINAL_DAMP_RATIO)
            phi_dot = (m_s * ay_dot * h_roll_arm) / roll_denom
            abs_phi_dot = sabs(phi_dot, 1e-3)

            c_reb_f = NOMINAL_DAMP_REBOUND_NS_M
            c_reb_r = NOMINAL_DAMP_REBOUND_NS_M
            c_comp_f = rd_f * c_reb_f
            c_comp_r = rd_r * c_reb_r

            mr_f_sq = float(p.mr_f) * float(p.mr_f)
            mr_r_sq = float(p.mr_r) * float(p.mr_r)

            # Dynamic roll-damping lateral load transfer on each axle
            # Softer bump (rd < 1) reduces outer-wheel compressive load spike on turn-in;
            # stiffer rear bump relative to front increases transient rear LLTD for slalom agility.
            c_roll_f = 0.5 * (c_comp_f + c_reb_f) * mr_f_sq * (0.5 * p.track_front)
            c_roll_r = 0.5 * (c_comp_r + c_reb_r) * mr_r_sq * (0.5 * p.track_rear)
            dfz_lat_f_raw = dfz_lat_f_raw + c_roll_f * phi_dot
            dfz_lat_r_raw = dfz_lat_r_raw + c_roll_r * phi_dot

            # Asymmetric damper jack-down (mild aero-seal downforce gain when rd < 1)
            # balanced by rebound packing / inner-wheel contact-patch starvation when rd < 0.72.
            asym_f = (c_reb_f - c_comp_f) * mr_f_sq * (0.5 * p.track_front) * abs_phi_dot
            asym_r = (c_reb_r - c_comp_r) * mr_r_sq * (0.5 * p.track_rear) * abs_phi_dot
            dz_pack_f_mm = 1e3 * asym_f / max(2.0 * float(p.k_wheel_f), 1.0)
            dz_pack_r_mm = 1e3 * asym_r / max(2.0 * float(p.k_wheel_r), 1.0)

            # Mild transient aero seal gain from dynamic jack-down (capped smoothly)
            fz_f_tot = spos(fz_f_tot + f_down_f * 0.012 * sclip(dz_pack_f_mm, -2.0, 3.0, 0.1), eps_fz)
            fz_r_tot = spos(fz_r_tot + f_down_r * 0.012 * sclip(dz_pack_r_mm, -2.0, 3.0, 0.1), eps_fz)

            # Rebound packing penalty if ratio is too low (inner wheel cannot extend fast enough
            # across consecutive slalom reversals, losing k_wheel * dz_pack droop load)
            pack_excess_f = spos(0.74 - rd_f, 0.02)
            pack_excess_r = spos(0.82 - rd_r, 0.02)
            sgn_ay = ca.tanh(ay / 1.5)
            dfz_lat_f_raw = dfz_lat_f_raw + sgn_ay * 45000.0 * (pack_excess_f * pack_excess_f) * abs_phi_dot
            dfz_lat_r_raw = dfz_lat_r_raw + sgn_ay * 45000.0 * (pack_excess_r * pack_excess_r) * abs_phi_dot

        dfz_lat_f = sclip(dfz_lat_f_raw, -0.5 * fz_f_tot, 0.5 * fz_f_tot, eps_fz)
        dfz_lat_r = sclip(dfz_lat_r_raw, -0.5 * fz_r_tot, 0.5 * fz_r_tot, eps_fz)

        fz = ca.vertcat(
            spos(0.5 * fz_f_tot - dfz_lat_f, eps_fz),
            spos(0.5 * fz_f_tot + dfz_lat_f, eps_fz),
            spos(0.5 * fz_r_tot - dfz_lat_r, eps_fz),
            spos(0.5 * fz_r_tot + dfz_lat_r, eps_fz),
        )
        return fz, phi

    def eval_forces_from_kappa(
        self,
        vx: Any,
        vy: Any,
        wz: Any,
        delta: Any,
        kappa_4: Any,
        F_brk_hyd: Any,
        ax_lt: Any,
        ay_lt: Any,
        T_tire_4: Optional[Any] = None,
        ay_dot: Any = 0.0,
        ax_dot: Any = 0.0,
    ) -> Dict[str, Any]:
        """Override supporting optional transient (ay_dot, ax_dot) for asymmetric damping."""
        if (isinstance(ay_dot, (int, float)) and abs(float(ay_dot)) < 1e-12) and (
            isinstance(ax_dot, (int, float)) and abs(float(ax_dot)) < 1e-12
        ):
            return super().eval_forces_from_kappa(
                vx, vy, wz, delta, kappa_4, F_brk_hyd, ax_lt, ay_lt, T_tire_4=T_tire_4
            )
        p = self.params
        s = self.smoothing
        delta_w = self.ackermann_steer_smooth(delta)
        cos_d = ca.cos(delta_w)
        sin_d = ca.sin(delta_w)

        v_wx, v_wy, u_den, alpha = self._wheel_kinematics(vx, vy, wz, delta_w)
        f_drag_x, f_drag_y, f_down_f, f_down_r = self.aero_forces_smooth(vx, vy, ax_lt)
        fz, phi = self.normal_forces_smooth(ax_lt, ay_lt, vx, vy, ay_dot=ay_dot, ax_dot=ax_dot)
        gamma = self.cambers_smooth(delta_w, phi, ax_lt)

        t_eff = T_tire_4 if T_tire_4 is not None else self.tire_temp_c
        if t_eff is not None:
            from vd_correlation_lab.models.vehicle_14dof.ter27_casadi_model import thermal_grip_multiplier_smooth
            mu_thermal = thermal_grip_multiplier_smooth(t_eff)
        else:
            mu_thermal = 1.0

        side = ca.DM(self.side_sign)
        fx_w, fy_w, mz_tire, _ = self.eval_tire_mf62(kappa_4, alpha, fz, gamma, side, mu_thermal=mu_thermal)

        fx_c = fx_w * cos_d - fy_w * sin_d
        fy_c = fx_w * sin_d + fy_w * cos_d

        x_w = ca.DM(self.x_w)
        y_w = ca.DM(self.y_w)

        Fx_tot = ca.sum1(fx_c) + f_drag_x
        Fy_tot = ca.sum1(fy_c) + f_drag_y
        Mz_tot = ca.sum1(x_w * fy_c - y_w * fx_c + mz_tire)

        rho_rad = smin(fz / p.k_tire_z, 0.5 * p.wheel_radius, s["eps_r_eff"])
        r_load = p.wheel_radius - rho_rad
        r_eff = p.wheel_radius - rho_rad / 3.0

        f_rr = p.rolling_resistance_coeff * fz * ca.tanh(v_wx / 0.5)
        T_total_4 = (fx_w + f_rr) * r_load
        hyd_split = self._hyd_split_vec()
        T_hyd_4 = F_brk_hyd * hyd_split * r_load
        T_motor_4 = T_total_4 + T_hyd_4

        omega_4 = (v_wx + kappa_4 * u_den) / smax(r_eff, 0.1, s["eps_r_eff"])
        T_max_4, T_min_4 = self.motor_torque_limit_smooth(omega_4)
        P_dc_ecu_kw, P_dc_drv_kw = self.eval_hv_power_kw(T_motor_4, omega_4)

        Fx_f_sin_delta = fx_w[0] * sin_d[0] + fx_w[1] * sin_d[1]
        Mz_tv_long = ca.sum1(-y_w * (fx_w * cos_d) + x_w * (fx_w * sin_d))

        from vd_correlation_lab.models.vehicle_14dof.ter27_casadi_model import (
            slip_power_smooth,
            tire_temp_dot_smooth,
        )
        slip_power_4 = slip_power_smooth(fx_w, fy_w, v_wx, v_wy, kappa_4)
        t_ref_for_ode = t_eff if t_eff is not None else (self.temp_c + 30.0)
        dT_tire_dt_4 = tire_temp_dot_smooth(t_ref_for_ode, self.temp_c, slip_power_4, vx)

        return {
            "Fx_tot": Fx_tot,
            "Fy_tot": Fy_tot,
            "Mz_tot": Mz_tot,
            "P_dc_kw": P_dc_ecu_kw,
            "P_dc_ecu_kw": P_dc_ecu_kw,
            "P_dc_drv_kw": P_dc_drv_kw,
            "Fz_4": fz,
            "Fx_w_4": fx_w,
            "Fy_w_4": fy_w,
            "Mz_tire_4": mz_tire,
            "alpha_4": alpha,
            "kappa_4": kappa_4,
            "T_motor_4": T_motor_4,
            "T_total_4": T_total_4,
            "omega_4": omega_4,
            "T_max_4": T_max_4,
            "T_min_4": T_min_4,
            "Fx_f_sin_delta": Fx_f_sin_delta,
            "Mz_tv_long": Mz_tv_long,
            "phi": phi,
            "delta_w_4": delta_w,
            "f_drag_x": f_drag_x,
            "f_drag_y": f_drag_y,
            "f_down_f": f_down_f,
            "f_down_r": f_down_r,
            "slip_power_4": slip_power_4,
            "dT_tire_dt_4": dT_tire_dt_4,
            "mu_thermal_4": mu_thermal,
        }


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
        if "ackermann_pct" in design_overrides:
            model.p.ackermann_factor = float(design_overrides["ackermann_pct"]) * 0.01
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
    optimize_ackermann: bool = True,
    equal_front_drive: bool = False,
    use_real_electrical_loss_map: bool = False,
) -> Dict[str, Any]:
    """
    Solve the stationary optimal skidpad cornering NLP with CasADi + IPOPT.
    Maximizes cornering speed (minimizes t_circle = 2*pi*R / V) over
    [V, delta, beta, kappa_FL, kappa_FR, kappa_RL, kappa_RR, R]
    and optionally co-optimizes the mechanical trackside setup parameters:
    [lltd, camber_f_deg, camber_r_deg, toe_f_deg, toe_r_deg, brake_bias, rake_mm, ackermann_pct].
    """
    do_opt_setup = bool(optimize_setup or optimize_trackside_setup)
    do_opt_ack = bool(do_opt_setup and optimize_ackermann)
    p_nom = load_ter27_params()
    dp_init: Dict[str, Any] = dict(design_overrides) if design_overrides else {}
    if use_real_electrical_loss_map:
        dp_init["use_real_electrical_loss_map"] = True
    s_sc: Optional[ca.SX] = None
    n_setup = 8 if do_opt_ack else 7
    setup_scale = (
        np.array([0.5, 2.0, 2.0, 0.5, 0.5, 0.6, 8.0, 100.0], dtype=float)
        if do_opt_ack
        else np.array([0.5, 2.0, 2.0, 0.5, 0.5, 0.6, 8.0], dtype=float)
    )
    if do_opt_setup:
        s_sc = ca.SX.sym("s_sc", n_setup)
        s_val = s_sc * ca.DM(setup_scale)
        dp_init["lltd"] = s_val[0]
        dp_init["k_roll_scale"] = 1.0
        dp_init["camber_f_deg"] = s_val[1]
        dp_init["camber_r_deg"] = s_val[2]
        dp_init["toe_f_deg"] = s_val[3]
        dp_init["toe_r_deg"] = s_val[4]
        dp_init["brake_bias_front"] = s_val[5]
        dp_init["rake_mm"] = s_val[6]
        if do_opt_ack:
            dp_init["ackermann_pct"] = s_val[7]

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
        # [lltd, camber_f_deg, camber_r_deg, toe_f_deg, toe_r_deg, brake_bias, rake_mm (, ackermann_pct)]
        s_lb_list = [
            b["lltd"][0], b["camber_f_deg"][0], b["camber_r_deg"][0],
            b["toe_f_deg"][0], b["toe_r_deg"][0], b["brake_bias"][0], b["rake_mm"][0],
        ]
        s_ub_list = [
            b["lltd"][1], b["camber_f_deg"][1], b["camber_r_deg"][1],
            b["toe_f_deg"][1], b["toe_r_deg"][1], b["brake_bias"][1], b["rake_mm"][1],
        ]
        s_0_list = [
            np.clip(p_nom.lltd_elastic_f, b["lltd"][0], b["lltd"][1]),
            np.clip(math.degrees(p_nom.static_camber_f_rad), b["camber_f_deg"][0], b["camber_f_deg"][1]),
            np.clip(math.degrees(p_nom.static_camber_r_rad), b["camber_r_deg"][0], b["camber_r_deg"][1]),
            0.0,
            0.0,
            np.clip(p_nom.brake_bias_front, b["brake_bias"][0], b["brake_bias"][1]),
            0.0,
        ]
        if do_opt_ack:
            s_lb_list.append(ACKERMANN_PCT_BOUNDS[0])
            s_ub_list.append(ACKERMANN_PCT_BOUNDS[1])
            s_0_list.append(50.0)
        s_lb = np.array(s_lb_list, dtype=float) / setup_scale
        s_ub = np.array(s_ub_list, dtype=float) / setup_scale
        s_0 = np.array(s_0_list, dtype=float) / setup_scale
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
        s_opt = x_raw[8:8 + n_setup] * setup_scale
        ack_opt_pct = float(s_opt[7]) if do_opt_ack else float(resolved_dp.get("ackermann_pct", p_nom.ackermann_factor * 100.0))
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
            "ackermann_pct": ack_opt_pct,
            "damp_ratio_f": float(resolved_dp.get("damp_ratio_f", NOMINAL_DAMP_RATIO)),
            "damp_ratio_r": float(resolved_dp.get("damp_ratio_r", NOMINAL_DAMP_RATIO)),
        }
        arb_f, arb_r = _implied_arb_stiffness_nm_m(p_nom, resolved_dp["lltd"], resolved_dp["k_roll_scale"])
        opt_setup_dict = {
            **resolved_dp,
            "arb_stiffness_front_nm_m": round(arb_f, 1),
            "arb_stiffness_rear_nm_m": round(arb_r, 1),
        }

    # Build numeric vehicle with resolved setup for clean post-evaluation & stability check
    if use_real_electrical_loss_map:
        resolved_dp["use_real_electrical_loss_map"] = True
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

    alpha_4_deg = np.degrees(np.array(ev_s["alpha_4"]).flatten())
    delta_w_4_deg = np.degrees(np.array(ev_s["delta_w_4"]).flatten())
    ack_eff_pct = float(resolved_dp.get("ackermann_pct", p_nom.ackermann_factor * 100.0))

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
        "ackermann_pct": round(ack_eff_pct, 2),
        "r_eff_m": R_s,
        "t_circle_s": float(2.0 * math.pi * R_s / V_s),
        "v_ms": V_s,
        "v_kmh": V_s * 3.6,
        "ay_g": abs(ay_s) / p.gravity,
        "beta_deg": math.degrees(b_s),
        "delta_deg": math.degrees(d_s),
        "delta_w_4_deg": [round(float(v), 3) for v in delta_w_4_deg],
        "alpha_4_deg": [round(float(v), 3) for v in alpha_4_deg],
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
    n_steps: int = 50,
    design_overrides: Optional[Dict[str, float]] = None,
    optimize_trackside_setup: bool = False,
    staging_rollout_m: float = 0.30,
    use_real_electrical_loss_map: bool = False,
    fixed_front_torque_ratio: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Solve the Official FS 0-75 m Acceleration OCP with CasADi + IPOPT (time-domain).
    Enforces strict 80.0 kW HV DC ceiling and real motor torque/speed envelope.
    Models the official FSG / FSAE D 5.2.2 0.30 m staging rollout from standstill:
      - Car launches from s = -staging_rollout_m (v = 0) with conservative 45 ms torque ramp,
        crossing the timing beam at s = 0.0 m with physically computed v_beam_ms.
      - Set staging_rollout_m = 0.0 for legacy v0 = 1.5 m/s boundary comparison.
    Optionally co-optimizes the 7 mechanical trackside setup parameters:
    [lltd, camber_f_deg, camber_r_deg, toe_f_deg, toe_r_deg, brake_bias, rake_mm].
    """
    p_nom = load_ter27_params()
    dp_init: Dict[str, Any] = dict(design_overrides) if design_overrides else {}
    if use_real_electrical_loss_map:
        dp_init["use_real_electrical_loss_map"] = True
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
    p_max_kw = (p.p_dc_max_w - 20.0) * 1e-3  # 79.98 kW strict ceiling (0 HV violations)

    # Physical staging rollout velocity at s = 0.0 m timing beam (FSG D 5.2.2: 0.30 m behind line)
    if staging_rollout_m > 0.0:
        # Evaluate launch acceleration at low speed with realistic 45 ms motor torque ramp
        veh_ref = build_casadi_vehicle_for_compound(grip_profile, compound, design_overrides=design_overrides)
        ev_l = veh_ref.eval_forces_from_kappa(
            1.5, 0.0, 0.0, 0.0, np.array([0.10, 0.10, 0.11, 0.11]), 0.0, 14.0, 0.0
        )
        # Clamp launch torque to motor peak envelope (312.5 Nm)
        ax_peak_launch = min(float(ev_l["Fx_tot"]) / p.mass, 14.9)
        # Conservative average acceleration over 0.30 m rollout including 45 ms launch torque ramp
        ax_rollout_avg = 0.85 * ax_peak_launch
        v_beam_ms = math.sqrt(2.0 * ax_rollout_avg * float(staging_rollout_m))
    else:
        v_beam_ms = 1.5

    # Scaled variables: T_75 (1), and per node [s_m/50, vx/20, kf/0.1, kr/0.1, ax_lt/10]
    T_sc = ca.SX.sym("T_sc")
    T_75 = T_sc * 4.0
    nx = 5
    z_scale = np.array([50.0, 20.0, 0.1, 0.1, 10.0], dtype=float)
    Z_sc = ca.SX.sym("Z_sc", nx, n_steps + 1)
    Z = Z_sc * ca.DM(z_scale)
    dt = T_75 / n_steps

    g_list: List[Any] = [Z[0, 0] / 50.0, (Z[1, 0] - v_beam_ms) / 20.0, (Z[0, n_steps] - 75.0) / 50.0]
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
        if fixed_front_torque_ratio is not None:
            sf_fix = float(fixed_front_torque_ratio)
            sr_fix = 1.0 - sf_fix
            g_list.append((sr_fix * ev["T_motor_4"][0] - sf_fix * ev["T_motor_4"][2]) / 100.0)
            lbg.append(0.0)
            ubg.append(0.0)

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
    z0[1] = v_beam_ms + (30.5 - v_beam_ms) * frac
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

    if use_real_electrical_loss_map:
        resolved_dp["use_real_electrical_loss_map"] = True
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
        "staging_rollout_m": float(staging_rollout_m),
        "v_beam_ms": round(float(v_beam_ms), 4),
        "time_75m_s": t75,
        "v_exit_kmh": float(z_opt[1, -1] * 3.6),
        "peak_ax_g": float(np.max(ax_arr)),
        "max_hv_power_kw": float(np.max(pdc_arr)),
        "hv_violations": int(np.sum((pdc_arr > 80.05) | (pdc_arr < -25.05))),
    }
    if opt_setup_dict is not None:
        out["optimal_setup"] = opt_setup_dict
    return out


def solve_autocross_transient_coopt_casadi(
    grip_profile: str = "FSG_RUBBERED",
    compound: str = "R25B",
    n_intervals: int = 160,
    base_setup_overrides: Optional[Dict[str, Any]] = None,
    optimize_transient_params: bool = True,
    use_real_electrical_loss_map: bool = False,
    grip_realism_factor: float = 0.71,
    ipopt_max_iter: int = 500,
) -> Dict[str, Any]:
    """
    Transient Autocross full-lap + slalom/chicane/hairpin OCP on FSG 2023 using CasADi + IPOPT.
    Models transient lateral acceleration rate ay_dot = d(ay)/dt along the curvilinear lap
    to co-optimize:
      - damp_ratio_f in [0.55, 1.35] (front bump/rebound ratio; < 1.0 = softer bump / stiffer rebound)
      - damp_ratio_r in [0.55, 1.35] (rear bump/rebound ratio)
      - ackermann_pct in [0.0, 100.0] % (dynamic Ackermann for tight hairpins)
    while preserving steady-state cornering stability (ay_dot = 0 -> identical steady-state dMz/dbeta),
    strict 4-corner cone clearance >= 0.18 m, HV DC power in [-25.0, +80.0] kW, and BSPD EV 2.3.
    """
    import json as _json
    import time as _time
    from scipy.interpolate import CubicSpline

    t0 = _time.perf_counter()
    track_path = REPO_ROOT / "data" / "tracks" / "fsg_2023.json"
    rl_path = REPO_ROOT / "src" / "mirena_sim" / "Assets" / "RacingLines" / "fsg_2023_racing_line_real.json"
    raw = _json.loads(track_path.read_text(encoding="utf-8"))
    length = float(raw["length"])
    s_raw = np.asarray(raw["s"], dtype=np.float64)
    x_raw = np.asarray(raw["x"], dtype=np.float64)
    y_raw = np.asarray(raw["y"], dtype=np.float64)
    k_raw = np.asarray(raw["kappa"], dtype=np.float64)
    wl_raw = np.asarray(raw["w_left"], dtype=np.float64)
    wr_raw = np.asarray(raw["w_right"], dtype=np.float64)

    s_ext = np.append(s_raw, length)
    cs_x = CubicSpline(s_ext, np.append(x_raw, x_raw[0]), bc_type="periodic")
    cs_y = CubicSpline(s_ext, np.append(y_raw, y_raw[0]), bc_type="periodic")
    cs_k = CubicSpline(s_ext, np.append(k_raw, k_raw[0]), bc_type="periodic")
    cs_wl = CubicSpline(s_ext, np.append(wl_raw, wl_raw[0]), bc_type="periodic")
    cs_wr = CubicSpline(s_ext, np.append(wr_raw, wr_raw[0]), bc_type="periodic")

    ds = length / float(n_intervals)
    m_pts = 2 * n_intervals
    s_all = np.arange(m_pts, dtype=np.float64) * (0.5 * ds)
    psi_all = np.arctan2(cs_y(s_all, 1), cs_x(s_all, 1))
    kappa_c = cs_k(s_all)
    w_left = cs_wl(s_all)
    w_right = cs_wr(s_all)

    # Build symbolic vehicle with transient parameters
    dp_init: Dict[str, Any] = dict(base_setup_overrides) if base_setup_overrides else {}
    dp_init["use_real_electrical_loss_map"] = bool(use_real_electrical_loss_map)
    s_tr_sym: Optional[ca.SX] = None
    tr_scale = np.array([1.0, 1.0, 100.0], dtype=float)
    if optimize_transient_params:
        s_tr_sym = ca.SX.sym("s_tr", 3)
        dp_init["damp_ratio_f"] = s_tr_sym[0]
        dp_init["damp_ratio_r"] = s_tr_sym[1]
        dp_init["ackermann_pct"] = s_tr_sym[2]
    else:
        dp_init.setdefault("damp_ratio_f", NOMINAL_DAMP_RATIO)
        dp_init.setdefault("damp_ratio_r", NOMINAL_DAMP_RATIO)

    veh = build_casadi_vehicle_for_compound(grip_profile, compound, design_overrides=dp_init)
    veh.lmux = float(veh.lmux) * float(grip_realism_factor)
    veh.lmuy = float(veh.lmuy) * float(grip_realism_factor)
    p = veh.params
    mass = float(p.mass)
    izz = float(p.izz_total)
    max_steer = float(p.max_steer_angle)

    # Compile SX physics function with ay_dot input
    s_vx = ca.SX.sym("vx")
    s_vy = ca.SX.sym("vy")
    s_wz = ca.SX.sym("wz")
    s_del = ca.SX.sym("delta")
    s_kap = ca.SX.sym("kappa_4", 4)
    s_fbrk = ca.SX.sym("F_brk_hyd")
    s_ax = ca.SX.sym("ax_lt")
    s_ay = ca.SX.sym("ay_lt")
    s_aydot = ca.SX.sym("ay_dot")
    ev_s = veh.eval_forces_from_kappa(
        s_vx, s_vy, s_wz, s_del, s_kap, s_fbrk, s_ax, s_ay, ay_dot=s_aydot
    )
    fn_in = [s_vx, s_vy, s_wz, s_del, s_kap, s_fbrk, s_ax, s_ay, s_aydot]
    if optimize_transient_params and s_tr_sym is not None:
        fn_in.append(s_tr_sym)
    fn_phys = ca.Function(
        "ax_trans_phys",
        fn_in,
        [
            ev_s["Fx_tot"],
            ev_s["Fy_tot"],
            ev_s["Mz_tot"],
            ev_s["P_dc_ecu_kw"],
            ev_s["P_dc_drv_kw"],
            ev_s["T_motor_4"],
            ev_s["T_max_4"],
            ev_s["T_min_4"],
        ],
    )
    fn_phys_map = fn_phys.map(m_pts)

    S_VX, S_VY, S_WZ, S_N, S_XI = 20.0, 2.0, 1.5, 1.0, 0.3
    S_DEL, S_KAP, S_FBRK, S_AX, S_AY = 0.3, 0.1, 4000.0, 10.0, 10.0

    W = ca.MX.sym("W", 13, m_pts)
    z_tr = ca.MX.sym("z_tr", 3) if optimize_transient_params else ca.MX.sym("z_tr", 0)

    vx = W[0, :] * S_VX
    vy = W[1, :] * S_VY
    wz = W[2, :] * S_WZ
    n_lat = W[3, :] * S_N
    xi = W[4, :] * S_XI
    delta = W[5, :] * S_DEL
    kappa_4 = W[6:10, :] * S_KAP
    F_brk_hyd = W[10, :] * S_FBRK
    ax_lt = W[11, :] * S_AX
    ay_lt = W[12, :] * S_AY

    kap_dm = ca.DM(kappa_c).T
    cos_xi = ca.cos(xi)
    sin_xi = ca.sin(xi)
    v_s = vx * cos_xi - vy * sin_xi
    sf = (1.0 - n_lat * kap_dm) / v_s

    idx_next = list(range(1, m_pts)) + [0]
    idx_prev = [m_pts - 1] + list(range(0, m_pts - 1))
    h_half = 0.5 * ds
    dt_centered = smax((sf[0, idx_prev] + sf[0, idx_next]) * h_half, 0.02, 1e-3)
    ay_dot_raw = (ay_lt[0, idx_next] - ay_lt[0, idx_prev]) / dt_centered
    ay_dot = sclip(ay_dot_raw, -120.0, 120.0, 1.0)

    if optimize_transient_params:
        tr_vals = z_tr * ca.DM(tr_scale)
        Fx_tot, Fy_tot, Mz_tot, P_ecu, P_drv, T_mot, T_max, T_min = fn_phys_map(
            vx, vy, wz, delta, kappa_4, F_brk_hyd, ax_lt, ay_lt, ay_dot,
            ca.repmat(tr_vals, 1, m_pts),
        )
    else:
        Fx_tot, Fy_tot, Mz_tot, P_ecu, P_drv, T_mot, T_max, T_min = fn_phys_map(
            vx, vy, wz, delta, kappa_4, F_brk_hyd, ax_lt, ay_lt, ay_dot
        )

    dvx_ds = sf * (Fx_tot / mass + vy * wz)
    dvy_ds = sf * (Fy_tot / mass - vx * wz)
    dwz_ds = sf * (Mz_tot / izz)
    dn_ds = sf * (vx * sin_xi + vy * cos_xi)
    dxi_ds = sf * wz - kap_dm

    X_s = W[0:5, :]
    F_s = ca.vertcat(dvx_ds / S_VX, dvy_ds / S_VY, dwz_ds / S_WZ, dn_ds / S_N, dxi_ds / S_XI)

    idx_k = list(range(0, m_pts, 2))
    idx_m = list(range(1, m_pts, 2))
    idx_kp1 = [(2 * ((k + 1) % n_intervals)) for k in range(n_intervals)]

    X_k, X_m, X_kp1 = X_s[:, idx_k], X_s[:, idx_m], X_s[:, idx_kp1]
    F_k, F_m, F_kp1 = F_s[:, idx_k], F_s[:, idx_m], F_s[:, idx_kp1]
    defect_mid = X_m - 0.5 * (X_k + X_kp1) - (ds / 8.0) * (F_k - F_kp1)
    defect_end = X_kp1 - X_k - (ds / 6.0) * (F_k + 4.0 * F_m + F_kp1)

    sf_k, sf_m, sf_kp1 = sf[0, idx_k], sf[0, idx_m], sf[0, idx_kp1]
    T_lap = ca.sum2((ds / 6.0) * (sf_k + 4.0 * sf_m + sf_kp1))

    d_delta = delta[0, idx_next] - delta
    dd_delta = delta[0, idx_next] - 2.0 * delta + delta[0, idx_prev]
    d_kappa = kappa_4[:, idx_next] - kappa_4
    d_fbrk = (F_brk_hyd[0, idx_next] - F_brk_hyd) / 10000.0

    reg = (
        3.0e-2 * ca.sum2(d_delta * d_delta) / h_half
        + 9.0e-2 * ca.sum2(dd_delta * dd_delta) / (h_half ** 3)
        + 2.0e-2 * ca.sum1(ca.sum2(d_kappa * d_kappa)) / h_half
        + 6.0e-3 * ca.sum2(d_fbrk * d_fbrk) / h_half
        + 1.5e-3 * ca.sum2(((F_brk_hyd / 10000.0) ** 2) * sf * h_half)
    )
    obj = T_lap + reg

    g_list: List[ca.MX] = []
    lbg_list: List[np.ndarray] = []
    ubg_list: List[np.ndarray] = []

    def _add_g(expr: ca.MX, lb_v: float | np.ndarray, ub_v: float | np.ndarray) -> None:
        vec = ca.vec(expr)
        n_el = vec.shape[0]
        g_list.append(vec)
        lbg_list.append(np.full(n_el, float(lb_v)) if np.isscalar(lb_v) else np.asarray(lb_v, dtype=np.float64).reshape(n_el))
        ubg_list.append(np.full(n_el, float(ub_v)) if np.isscalar(ub_v) else np.asarray(ub_v, dtype=np.float64).reshape(n_el))

    _add_g(defect_mid, 0.0, 0.0)
    _add_g(defect_end, 0.0, 0.0)
    _add_g((ax_lt - Fx_tot / mass) / S_AX, 0.0, 0.0)
    _add_g((ay_lt - Fy_tot / mass) / S_AY, 0.0, 0.0)
    _add_g((T_mot - T_max) / 250.0, -np.inf, 0.0)
    _add_g((T_min - T_mot) / 250.0, -np.inf, 0.0)
    _add_g(P_ecu / 80.0, -24.90 / 80.0, 79.90 / 80.0)
    _add_g(P_drv / 80.0, -24.90 / 80.0, 79.90 / 80.0)

    car_half_w = 0.71
    car_l_f = 1.15
    car_l_r = 0.95
    cone_m = 0.18
    wl_lim = ca.DM(w_left - cone_m).T
    wr_lim = ca.DM(w_right - cone_m).T
    _add_g(n_lat + car_l_f * sin_xi + car_half_w * cos_xi - wl_lim, -np.inf, 0.0)
    _add_g(n_lat - car_l_r * sin_xi + car_half_w * cos_xi - wl_lim, -np.inf, 0.0)
    _add_g(n_lat + car_l_f * sin_xi - car_half_w * cos_xi + wr_lim, 0.0, np.inf)
    _add_g(n_lat - car_l_r * sin_xi - car_half_w * cos_xi + wr_lim, 0.0, np.inf)
    _add_g(v_s / S_VX, 3.0 / S_VX, np.inf)

    dt_half = 0.5 * (sf + sf[0, idx_next]) * h_half
    max_steer_rate = math.radians(62.0)
    _add_g(d_delta - max_steer_rate * dt_half, -np.inf, 0.0)
    _add_g(-d_delta - max_steer_rate * dt_half, -np.inf, 0.0)

    tan_bmax = math.tan(math.radians(4.8))
    _add_g((vy - vx * tan_bmax) / S_VY, -np.inf, 0.0)
    _add_g((-vy - vx * tan_bmax) / S_VY, -np.inf, 0.0)

    T_mot_pos = 0.5 * (T_mot + ca.sqrt(T_mot * T_mot + 0.01))
    bspd_prod = ca.repmat(F_brk_hyd / 4000.0, 4, 1) * (T_mot_pos / 250.0)
    _add_g(bspd_prod, -np.inf, 1.5e-3)

    g_vec = ca.vertcat(*g_list)
    lbg = np.concatenate(lbg_list)
    ubg = np.concatenate(ubg_list)

    lbw = np.zeros((13, m_pts), dtype=np.float64)
    ubw = np.zeros((13, m_pts), dtype=np.float64)
    lbw[0, :], ubw[0, :] = 3.5 / S_VX, 38.0 / S_VX
    lbw[1, :], ubw[1, :] = -4.0 / S_VY, 4.0 / S_VY
    lbw[2, :], ubw[2, :] = -3.8 / S_WZ, 3.8 / S_WZ
    lbw[3, :] = -np.maximum(w_right - car_half_w - cone_m, 0.05) / S_N
    ubw[3, :] = np.maximum(w_left - car_half_w - cone_m, 0.05) / S_N
    lbw[4, :], ubw[4, :] = -0.70 / S_XI, 0.70 / S_XI
    lbw[5, :], ubw[5, :] = -max_steer / S_DEL, max_steer / S_DEL
    lbw[6:10, :], ubw[6:10, :] = -0.135 / S_KAP, 0.135 / S_KAP
    lbw[10, :], ubw[10, :] = 0.0 / S_FBRK, 12000.0 / S_FBRK
    lbw[11, :], ubw[11, :] = -24.0 / S_AX, 18.0 / S_AX
    lbw[12, :], ubw[12, :] = -22.0 / S_AY, 22.0 / S_AY

    # Warm start from FSG 2023 racing line
    rl = _json.loads(rl_path.read_text(encoding="utf-8"))
    pts = rl["points"]
    rl_len = float(rl.get("length_m", length))
    s_rl = np.array([float(pt["s"]) for pt in pts], dtype=np.float64) * (length / max(rl_len, 1.0))

    def _interp_p(vals: np.ndarray) -> np.ndarray:
        return np.interp(s_all, np.append(s_rl, length), np.append(vals, vals[0]))

    vx_0 = np.clip(_interp_p(np.array([float(pt["v_mps"]) for pt in pts])), 5.0, 30.0)
    n_0 = np.clip(_interp_p(np.array([float(pt["n"]) for pt in pts])), -0.55, 0.55)
    yaw_rl = np.array([float(pt["yaw_rad"]) for pt in pts])
    yaw_unw = np.unwrap(yaw_rl)
    d_yaw = ((yaw_rl[0] - yaw_rl[-1] + np.pi) % (2.0 * np.pi)) - np.pi
    yaw_i = np.interp(s_all, np.append(s_rl, length), np.append(yaw_unw, yaw_unw[-1] + d_yaw))
    xi_0 = np.clip(((yaw_i - psi_all + np.pi) % (2.0 * np.pi)) - np.pi, -0.35, 0.35)
    kap_0 = _interp_p(np.array([float(pt["kappa"]) for pt in pts]))
    wz_0 = np.clip(vx_0 * kap_0, -3.0, 3.0)
    ax_0 = np.clip(_interp_p(np.array([float(pt["ax_mps2"]) for pt in pts])), -14.0, 11.0)
    ay_0 = np.clip(_interp_p(np.array([float(pt["ay_mps2"]) for pt in pts])), -14.0, 14.0)
    del_0 = np.clip(_interp_p(np.array([float(pt["steer_rad"]) for pt in pts])), -0.9 * max_steer, 0.9 * max_steer)
    kw_0 = np.clip(ax_0 / 180.0, -0.05, 0.05)

    w0 = np.zeros((13, m_pts), dtype=np.float64)
    w0[0, :] = vx_0 / S_VX
    w0[2, :] = wz_0 / S_WZ
    w0[3, :] = n_0 / S_N
    w0[4, :] = xi_0 / S_XI
    w0[5, :] = del_0 / S_DEL
    w0[6:10, :] = np.tile(kw_0[None, :], (4, 1)) / S_KAP
    w0[10, :] = np.clip(np.maximum(-ax_0 * mass - 2200.0, 0.0), 0.0, 5000.0) / S_FBRK
    w0[11, :] = ax_0 / S_AX
    w0[12, :] = ay_0 / S_AY

    if optimize_transient_params:
        lb_tr = np.array([DAMP_RATIO_BOUNDS[0], DAMP_RATIO_BOUNDS[0], ACKERMANN_PCT_BOUNDS[0]]) / tr_scale
        ub_tr = np.array([DAMP_RATIO_BOUNDS[1], DAMP_RATIO_BOUNDS[1], ACKERMANN_PCT_BOUNDS[1]]) / tr_scale
        z0_tr = np.array([0.75, 0.85, 10.0]) / tr_scale
        lbx = np.concatenate([lbw.reshape(-1, order="F"), lb_tr])
        ubx = np.concatenate([ubw.reshape(-1, order="F"), ub_tr])
        x0 = np.clip(np.concatenate([w0.reshape(-1, order="F"), z0_tr]), lbx, ubx)
        x_sym = ca.vertcat(ca.vec(W), z_tr)
    else:
        lbx = lbw.reshape(-1, order="F")
        ubx = ubw.reshape(-1, order="F")
        x0 = np.clip(w0.reshape(-1, order="F"), lbx, ubx)
        x_sym = ca.vec(W)

    nlp = {"x": x_sym, "f": obj, "g": g_vec}
    opts = {
        "print_time": False,
        "ipopt.print_level": 0,
        "ipopt.linear_solver": "mumps",
        "ipopt.hessian_approximation": "limited-memory",
        "ipopt.max_iter": int(ipopt_max_iter),
        "ipopt.tol": 1e-4,
        "ipopt.acceptable_tol": 1e-3,
        "ipopt.acceptable_iter": 10,
        "ipopt.mu_strategy": "adaptive",
        "ipopt.nlp_scaling_method": "gradient-based",
        "ipopt.sb": "yes",
    }
    solver = ca.nlpsol("ax_transient_ocp", "ipopt", nlp, opts)
    sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    stats = solver.stats()
    x_opt = np.array(sol["x"].full()).ravel()
    w_opt = x_opt[: 13 * m_pts].reshape((13, m_pts), order="F")

    if optimize_transient_params:
        tr_opt = x_opt[13 * m_pts : 13 * m_pts + 3] * tr_scale
        rd_f_opt, rd_r_opt, ack_opt = float(tr_opt[0]), float(tr_opt[1]), float(tr_opt[2])
    else:
        rd_f_opt = float(dp_init.get("damp_ratio_f", NOMINAL_DAMP_RATIO))
        rd_r_opt = float(dp_init.get("damp_ratio_r", NOMINAL_DAMP_RATIO))
        ack_opt = float(dp_init.get("ackermann_pct", p.ackermann_factor * 100.0))

    vx_o = w_opt[0, :] * S_VX
    vy_o = w_opt[1, :] * S_VY
    wz_o = w_opt[2, :] * S_WZ
    n_o = w_opt[3, :] * S_N
    xi_o = w_opt[4, :] * S_XI
    ay_o = w_opt[12, :] * S_AY
    v_s_o = vx_o * np.cos(xi_o) - vy_o * np.sin(xi_o)
    sf_o = (1.0 - n_o * kappa_c) / v_s_o
    dt_int = (ds / 6.0) * (sf_o[idx_k] + 4.0 * sf_o[idx_m] + sf_o[idx_kp1])
    t_lap = float(np.sum(dt_int))

    dt_cent = np.maximum((sf_o[idx_prev] + sf_o[idx_next]) * h_half, 0.02)
    ay_dot_o = (ay_o[idx_next] - ay_o[idx_prev]) / dt_cent
    wz_dot_o = (wz_o[idx_next] - wz_o[idx_prev]) / dt_cent

    # Verify steady-state cornering stability dMz/dbeta is unaffected (ay_dot = 0)
    dp_eval = dict(base_setup_overrides) if base_setup_overrides else {}
    dp_eval.update({
        "damp_ratio_f": rd_f_opt,
        "damp_ratio_r": rd_r_opt,
        "ackermann_pct": ack_opt,
        "use_real_electrical_loss_map": bool(use_real_electrical_loss_map),
    })
    veh_ss = build_casadi_vehicle_for_compound(grip_profile, compound, design_overrides=dp_eval)
    dmz_ss = compute_directional_stability_dmz_dbeta(
        veh_ss, 11.5, 0.0, 1.35, 0.16, np.array([0.01, 0.01, 0.02, 0.02]), 0.0, 15.0
    )

    return {
        "success": bool(stats.get("success", False)) or str(stats.get("return_status")) == "Solved_To_Acceptable_Level",
        "return_status": str(stats.get("return_status", "")),
        "iterations": int(stats.get("iter_count", -1)),
        "solve_time_s": round(_time.perf_counter() - t0, 2),
        "lap_time_s": round(t_lap, 4),
        "damp_ratio_f": round(rd_f_opt, 4),
        "damp_ratio_r": round(rd_r_opt, 4),
        "damping_compression_front_ns_m": round(rd_f_opt * NOMINAL_DAMP_REBOUND_NS_M, 1),
        "damping_rebound_front_ns_m": round(NOMINAL_DAMP_REBOUND_NS_M, 1),
        "damping_compression_rear_ns_m": round(rd_r_opt * NOMINAL_DAMP_REBOUND_NS_M, 1),
        "damping_rebound_rear_ns_m": round(NOMINAL_DAMP_REBOUND_NS_M, 1),
        "ackermann_pct": round(ack_opt, 2),
        "max_abs_ay_dot_ms3": round(float(np.max(np.abs(ay_dot_o))), 2),
        "rms_ay_dot_ms3": round(float(np.sqrt(np.mean(ay_dot_o ** 2))), 2),
        "max_abs_wz_dot_rads2": round(float(np.max(np.abs(wz_dot_o))), 2),
        "ay_max_g": round(float(np.max(np.abs(ay_o)) / p.gravity), 3),
        "v_max_kmh": round(float(np.max(np.hypot(vx_o, vy_o)) * 3.6), 2),
        "steady_state_dmz_dbeta_iso_nm_deg": round(dmz_ss, 2),
        "steady_state_stable": bool(dmz_ss > 0.0),
    }


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
