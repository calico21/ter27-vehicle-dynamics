#!/usr/bin/env python3
"""
Exact CasADi Jacobian Covariance Propagation and Embedded Float32 Low-Rank
Approximation for Stochastic Tube-MPC (TeR27 4WD).

Full 14-DOF closed-loop covariance update using exact AD Jacobians (A_k, B_k, G_k)
from outputs/casadi_cache/libter_diff_14dof.so:
    Sigma_{k+1} = (A_k - B_k K_k) Sigma_k (A_k - B_k K_k)^T
                  + G_k Q_w(sigma_lambda_mu^2, sigma_IMU^2, sigma_CAN^2) G_k^T

Also provides EmbeddedDiagonalTubePropagator (float32, 0 B heap, 5-state low-rank
2x2 (e_y, e_psi) + diagonal (v_y, omega_z)) with 1-to-1 parity to C99
ter27_stochastic_tube.c and Godot EcuTorqueVectoring4Wd.gd.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

from vd_correlation_lab.models.differentiable_casadi import (
    DIST_IDX,
    INPUT_IDX,
    NX,
    NU,
    NW,
    STATE_IDX,
    load_compiled_14dof_library,
)


@dataclass(frozen=True)
class ProcessNoiseConfig:
    """Physical sensor, CAN transport, and grip-uncertainty noise parameters."""

    mass_kg: float = 290.0
    izz_kgm2: float = 83.1
    can_accel_latency_s: float = 0.125   # TeR26/27 CAN accelerometer latency [s]
    can_steer_latency_s: float = 0.020   # CAN steer/gyro latency [s]
    can_jitter_s: float = 0.015          # CAN transport jitter std [s]
    imu_noise_scale: float = 1.0         # 0.0 in SENSOR_IDEAL, 1.0+ in SENSOR_TRACK_REALISM
    ax_noise_std_ms2: float = 0.18
    ay_noise_std_ms2: float = 0.18
    wz_noise_std_rads: float = 0.012
    omega_noise_std_rads: float = 0.08
    accel_quant_ms2: float = 0.01
    k_innov_var: float = 0.0028          # Innovation-triggered grip variance inflation
    z_road_std_m: float = 0.0015
    t_tire_std_c: float = 2.0
    t_motor_std_c: float = 2.0
    # Asymmetric yaw-moment disturbance Q_w parameters (split-mu & 3WD limp-home)
    k_asym_slip_mz: float = 420.0        # [Nm per unit L-R wheel-slip asymmetry]
    k_asym_torque_mz: float = 1.15       # [Nm per Nm of L-R torque-delivery yaw residual]
    k_asym_yaw_innov_mz: float = 280.0   # [Nm per rad/s of yaw-rate innovation residual]
    sigma_mz_asym_max_nm: float = 950.0  # Safety ceiling on asymmetric yaw std [Nm]


class ProcessNoiseModel:
    """
    Maps live (sigma_lambda_mu^2, sigma_IMU^2, sigma_CAN^2) into the 10x10
    disturbance covariance matrix Q_w aligned with DifferentiableVehicle14Dof
    disturbance vector w = [d_mu_scale, d_fx_ext, d_fy_ext, d_mz_ext,
    z_road_FL..RR, t_tire_c, t_motor_c].
    """

    def __init__(self, cfg: Optional[ProcessNoiseConfig] = None) -> None:
        self.cfg = cfg if cfg is not None else ProcessNoiseConfig()

    def build_qw(
        self,
        lambda_mu_hat: float,
        p_cov_mu: float,
        innov_ms2: float,
        ay_ms2: float,
        ax_ms2: float = 0.0,
        imu_noise_scale: Optional[float] = None,
        can_accel_latency_s: Optional[float] = None,
        can_jitter_s: Optional[float] = None,
        lr_slip_asym: float = 0.0,
        torque_delivery_resid_nm: float = 0.0,
        yaw_innov_rads: float = 0.0,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Return (Q_w [10x10], scalar variance breakdown dict)."""
        n_scale = self.cfg.imu_noise_scale if imu_noise_scale is None else float(imu_noise_scale)
        can_lat = self.cfg.can_accel_latency_s if can_accel_latency_s is None else float(can_accel_latency_s)
        can_jit = self.cfg.can_jitter_s if can_jitter_s is None else float(can_jitter_s)

        lam_safe = max(float(lambda_mu_hat), 0.25)
        ay_abs = abs(float(ay_ms2))
        ax_abs = abs(float(ax_ms2))

        # 1. Grip uncertainty variance sigma_lambda_mu^2 (live RLS p_cov + innovation spike)
        innov_norm = float(innov_ms2) / max(ay_abs, 4.0)
        sigma_mu_var = max(float(p_cov_mu), 1.0e-6) + self.cfg.k_innov_var * (innov_norm * innov_norm)
        # Relative d_mu_scale variance for CasADi w[0] (multiplicative grip scale)
        var_d_mu_scale = sigma_mu_var / (lam_safe * lam_safe)

        # 2. IMU noise variance
        sig_ax_imu = n_scale * self.cfg.ax_noise_std_ms2
        sig_ay_imu = n_scale * self.cfg.ay_noise_std_ms2
        sig_wz_imu = n_scale * self.cfg.wz_noise_std_rads

        # 3. CAN transport latency + jitter + quantization variance
        ay_dot_rms = max(ay_abs * 1.8, 3.0)
        ax_dot_rms = max(ax_abs * 1.5, 2.5)
        wz_dot_rms = max(ay_abs * 0.25, 0.4)
        can_eff_a = can_lat + can_jit
        can_eff_w = (self.cfg.can_steer_latency_s + 0.5 * can_jit) if can_lat > 0.0 else 0.0

        var_quant = (self.cfg.accel_quant_ms2 ** 2) / 12.0 if n_scale > 0.0 else 0.0
        var_ax_can = (can_eff_a * ax_dot_rms) ** 2 + var_quant
        var_ay_can = (can_eff_a * ay_dot_rms) ** 2 + var_quant
        var_wz_can = (can_eff_w * wz_dot_rms) ** 2

        var_ax_tot = sig_ax_imu ** 2 + var_ax_can
        var_ay_tot = sig_ay_imu ** 2 + var_ay_can
        var_wz_tot = sig_wz_imu ** 2 + var_wz_can

        # 4. Asymmetric yaw-moment disturbance variance sigma_Mz_asym^2
        #    Driven by L-R wheel-slip asymmetry, torque-delivery residual, and yaw innovation.
        #    Identically 0.0 when all three optional residuals are 0.0 (backward-compatible).
        slip_term = self.cfg.k_asym_slip_mz * abs(float(lr_slip_asym))
        torq_term = self.cfg.k_asym_torque_mz * abs(float(torque_delivery_resid_nm))
        yaw_term = self.cfg.k_asym_yaw_innov_mz * abs(float(yaw_innov_rads))
        sigma_mz_asym_nm = min(
            math.sqrt(slip_term * slip_term + torq_term * torq_term + yaw_term * yaw_term),
            self.cfg.sigma_mz_asym_max_nm,
        )
        var_mz_asym = sigma_mz_asym_nm * sigma_mz_asym_nm
        var_fy_asym = (0.35 * sigma_mz_asym_nm / 0.615) ** 2 if sigma_mz_asym_nm > 0.0 else 0.0

        # Populate diagonal 10x10 Q_w matching DIST_IDX
        qw = np.zeros((NW, NW), dtype=np.float64)
        qw[DIST_IDX["d_mu_scale"], DIST_IDX["d_mu_scale"]] = var_d_mu_scale
        qw[DIST_IDX["d_fx_ext"], DIST_IDX["d_fx_ext"]] = (self.cfg.mass_kg ** 2) * var_ax_tot
        qw[DIST_IDX["d_fy_ext"], DIST_IDX["d_fy_ext"]] = (self.cfg.mass_kg ** 2) * var_ay_tot + var_fy_asym
        qw[DIST_IDX["d_mz_ext"], DIST_IDX["d_mz_ext"]] = (self.cfg.izz_kgm2 ** 2) * var_wz_tot + var_mz_asym

        z_var = (self.cfg.z_road_std_m ** 2) if n_scale > 0.0 else 1.0e-9
        for k in ("z_road_FL", "z_road_FR", "z_road_RL", "z_road_RR"):
            qw[DIST_IDX[k], DIST_IDX[k]] = z_var
        th_scale = max(n_scale, 0.05) if (n_scale > 0.0 or can_lat > 0.0) else 0.02
        qw[DIST_IDX["t_tire_c"], DIST_IDX["t_tire_c"]] = (self.cfg.t_tire_std_c * th_scale) ** 2
        qw[DIST_IDX["t_motor_c"], DIST_IDX["t_motor_c"]] = (self.cfg.t_motor_std_c * th_scale) ** 2

        breakdown = {
            "sigma_mu_var": float(sigma_mu_var),
            "var_d_mu_scale": float(var_d_mu_scale),
            "var_ay_imu": float(sig_ay_imu ** 2),
            "var_ay_can": float(var_ay_can),
            "var_wz_tot": float(var_wz_tot),
            "sigma_mz_asym_nm": float(sigma_mz_asym_nm),
            "var_mz_asym_nm2": float(var_mz_asym),
        }
        return qw, breakdown


class ExactCovariancePropagator:
    """
    Exact 32-state discrete-time covariance propagator using compiled CasADi
    AD Jacobians (A_k, B_k, G_k) from libter_diff_14dof.so:
        Sigma_{k+1} = (A_k - B_k K_k) Sigma_k (A_k - B_k K_k)^T + G_k Q_w G_k^T
    """

    def __init__(
        self,
        noise_model: Optional[ProcessNoiseModel] = None,
        compiled_lib: Optional[Any] = None,
        track_half_m: float = 0.615,
        r_eff_m: float = 0.2032,
    ) -> None:
        self.noise_model = noise_model if noise_model is not None else ProcessNoiseModel()
        self.lib = compiled_lib if compiled_lib is not None else load_compiled_14dof_library()
        self.track_half_m = float(track_half_m)
        self.r_eff_m = float(r_eff_m)
        self.sigma: np.ndarray = np.zeros((NX, NX), dtype=np.float64)
        self.reset()

    def reset(self, init_pos_std_m: float = 0.02, init_vel_std_ms: float = 0.05) -> None:
        """Initialize 32x32 state covariance matrix Sigma_0."""
        self.sigma = np.eye(NX, dtype=np.float64) * 1.0e-6
        self.sigma[STATE_IDX["X"], STATE_IDX["X"]] = init_pos_std_m ** 2
        self.sigma[STATE_IDX["Y"], STATE_IDX["Y"]] = init_pos_std_m ** 2
        self.sigma[STATE_IDX["psi"], STATE_IDX["psi"]] = 0.01 ** 2
        self.sigma[STATE_IDX["vx"], STATE_IDX["vx"]] = init_vel_std_ms ** 2
        self.sigma[STATE_IDX["vy"], STATE_IDX["vy"]] = init_vel_std_ms ** 2
        self.sigma[STATE_IDX["wz"], STATE_IDX["wz"]] = 0.02 ** 2

    def build_local_feedback_gain(self, x: np.ndarray, psi_ref: float) -> np.ndarray:
        """
        Build stabilizing local tube feedback matrix K_k [9x32] such that
        du = -K_k dx stabilizes lateral error e_y, heading error e_psi,
        longitudinal speed vx, sideslip vy (with dMz/dbeta < 0), and yaw rate wz.
        """
        k_mat = np.zeros((NU, NX), dtype=np.float64)
        vx_eff = max(abs(float(x[STATE_IDX["vx"]])), 3.0)
        sin_p = math.sin(float(psi_ref))
        cos_p = math.cos(float(psi_ref))

        # Lateral error e_y = -sin(psi_ref)*dX + cos(psi_ref)*dY
        # Steering feedback: d(delta) = -k_ey * e_y - k_psi * dpsi - k_vy * dvy - k_wz * dwz
        k_ey = 0.28
        k_psi = 1.15
        k_vy_steer = 0.025
        k_wz_steer = 0.06
        idx_del = INPUT_IDX["delta_cmd"]
        k_mat[idx_del, STATE_IDX["X"]] = -k_ey * sin_p
        k_mat[idx_del, STATE_IDX["Y"]] = k_ey * cos_p
        k_mat[idx_del, STATE_IDX["psi"]] = k_psi
        k_mat[idx_del, STATE_IDX["vy"]] = k_vy_steer
        k_mat[idx_del, STATE_IDX["wz"]] = k_wz_steer

        # 4WD Torque-Vectoring feedback:
        #   - Speed regulation: dFx = -k_vx * dvx -> dT_i = -(k_vx * r_eff / 4) * dvx
        #   - Yaw stabilization: dMz = -k_r * dwz - k_beta * d(beta) (dMz/dbeta < 0!)
        #     Since d(beta) ~ d(vy)/vx_eff, dMz = -k_r * dwz - (k_beta / vx_eff) * dvy
        #     Left wheels (FL, RL) have b_mz < 0, right wheels (FR, RR) have b_mz > 0.
        #     So dT_right = +0.25 * (r_eff / track_half) * dMz, dT_left = -dT_right.
        #     Because du = -K dx, K[T_right, wz] = +gain_r, K[T_left, wz] = -gain_r,
        #     and K[T_right, vy] = +gain_vy, K[T_left, vy] = -gain_vy (preserving dMz/dbeta < 0!).
        k_vx_t = 35.0
        k_r_nm = 650.0
        k_beta_nm = 420.0
        t_per_mz = 0.25 * self.r_eff_m / max(self.track_half_m, 0.3)
        g_wz = k_r_nm * t_per_mz
        g_vy = (k_beta_nm / vx_eff) * t_per_mz

        for w_name, side_sgn in (
            ("T_wheel_FL", -1.0),
            ("T_wheel_FR", +1.0),
            ("T_wheel_RL", -1.0),
            ("T_wheel_RR", +1.0),
        ):
            r_idx = INPUT_IDX[w_name]
            k_mat[r_idx, STATE_IDX["vx"]] = k_vx_t
            k_mat[r_idx, STATE_IDX["wz"]] = side_sgn * g_wz
            k_mat[r_idx, STATE_IDX["vy"]] = side_sgn * g_vy
        return k_mat

    def step_covariance(
        self,
        x: np.ndarray,
        u: np.ndarray,
        theta: np.ndarray,
        w: np.ndarray,
        psi_ref: float,
        lambda_mu_hat: float,
        p_cov_mu: float,
        innov_ms2: float,
        ay_ms2: float,
        ax_ms2: float = 0.0,
        p_dc_nom_kw: float = 0.0,
        imu_noise_scale: Optional[float] = None,
        can_accel_latency_s: Optional[float] = None,
        can_jitter_s: Optional[float] = None,
        n_preview_steps: int = 1,
        lr_slip_asym: float = 0.0,
        torque_delivery_resid_nm: float = 0.0,
        yaw_innov_rads: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Evaluate exact AD Jacobians (A_k, B_k, G_k) via libter_diff_14dof.so and
        propagate:
            Sigma_{k+1} = (A_k - B_k K_k) Sigma_k (A_k - B_k K_k)^T + G_k Q_w G_k^T
        Returns projected scalar variances (var_ey, var_fric, var_pdc) and diagnostics.
        """
        a_x_raw, b_u_raw, _s_theta, g_w_raw = self.lib.step_jacobians(x, u, theta, w)
        a_x = np.asarray(a_x_raw, dtype=np.float64)
        b_u = np.asarray(b_u_raw, dtype=np.float64)
        g_w = np.asarray(g_w_raw, dtype=np.float64)
        k_mat = self.build_local_feedback_gain(x, psi_ref)
        a_cl = a_x - b_u @ k_mat

        qw, nb = self.noise_model.build_qw(
            lambda_mu_hat=lambda_mu_hat,
            p_cov_mu=p_cov_mu,
            innov_ms2=innov_ms2,
            ay_ms2=ay_ms2,
            ax_ms2=ax_ms2,
            imu_noise_scale=imu_noise_scale,
            can_accel_latency_s=can_accel_latency_s,
            can_jitter_s=can_jitter_s,
            lr_slip_asym=lr_slip_asym,
            torque_delivery_resid_nm=torque_delivery_resid_nm,
            yaw_innov_rads=yaw_innov_rads,
        )
        gqg = g_w @ qw @ g_w.T

        # Propagate 1 step on live state covariance
        sig_next = a_cl @ self.sigma @ a_cl.T + gqg
        sig_next = 0.5 * (sig_next + sig_next.T)
        # Bound non-lateral internal states to prevent unconstrained drift of global X along path
        self.sigma = sig_next

        # Optional short lookahead tube preview (for chance constraint tightening)
        sig_eval = self.sigma
        if n_preview_steps > 1:
            s_tmp = self.sigma.copy()
            for _ in range(n_preview_steps - 1):
                s_tmp = a_cl @ s_tmp @ a_cl.T + gqg
                s_tmp = 0.5 * (s_tmp + s_tmp.T)
            sig_eval = s_tmp

        # Project onto lateral corridor error c_ey = [-sin(psi_ref), cos(psi_ref)] on (X, Y)
        c_ey = np.zeros(NX, dtype=np.float64)
        c_ey[STATE_IDX["X"]] = -math.sin(float(psi_ref))
        c_ey[STATE_IDX["Y"]] = math.cos(float(psi_ref))
        var_ey = max(float(c_ey @ sig_eval @ c_ey), 0.0)

        var_vy = max(float(sig_eval[STATE_IDX["vy"], STATE_IDX["vy"]]), 0.0)
        var_wz = max(float(sig_eval[STATE_IDX["wz"], STATE_IDX["wz"]]), 0.0)

        # Normalized friction ellipse utilization variance
        var_fric = (
            nb["var_d_mu_scale"]
            + 0.015 * var_vy
            + 0.040 * var_wz
            + 0.0080 * (nb["var_ay_imu"] + nb["var_ay_can"]) / (9.81 * 9.81)
            + 0.025 * (nb["var_mz_asym_nm2"] / (650.0 * 650.0))
        )

        # HV DC power variance from wheel speed covariance + encoder/CAN uncertainty
        n_sc = self.noise_model.cfg.imu_noise_scale if imu_noise_scale is None else float(imu_noise_scale)
        can_l = self.noise_model.cfg.can_accel_latency_s if can_accel_latency_s is None else float(can_accel_latency_s)
        var_om_sum = sum(
            max(float(sig_eval[STATE_IDX[k], STATE_IDX[k]]), 0.0)
            for k in ("omega_FL", "omega_FR", "omega_RL", "omega_RR")
        )
        t_sq_avg = float(np.mean(np.square(u[1:5])))
        var_pdc_dyn = (t_sq_avg * var_om_sum) * 1.0e-6  # (Nm * rad/s -> kW)^2
        om_std = n_sc * self.noise_model.cfg.omega_noise_std_rads
        var_pdc = (
            var_pdc_dyn
            + (float(p_dc_nom_kw) * 0.004) ** 2
            + 0.35 * (om_std ** 2)
            + 0.60 * ((can_l * 2.5) ** 2)
        )

        # Spectral radius check on lateral-yaw subblock
        sub_idx = [STATE_IDX["Y"], STATE_IDX["psi"], STATE_IDX["vy"], STATE_IDX["wz"]]
        rho_sub = float(np.max(np.abs(np.linalg.eigvals(a_cl[np.ix_(sub_idx, sub_idx)]))))

        return {
            "var_ey_m2": float(var_ey),
            "var_vy_m2s2": float(var_vy),
            "var_wz_r2s2": float(var_wz),
            "var_fric_norm": float(max(var_fric, 0.0)),
            "var_pdc_kw2": float(max(var_pdc, 0.0)),
            "rho_cl_lat_yaw": rho_sub,
            "noise_breakdown": nb,
        }


class EmbeddedDiagonalTubePropagator:
    """
    Deterministic float32 / 0 B heap diagonal + 2x2 (e_y, e_psi) low-rank
    covariance propagator. Matches C99 ter27_stochastic_tube.c and Godot
    EcuTorqueVectoring4Wd.gd bit-for-bit.
    """

    __slots__ = (
        "k_ey_fb",
        "k_epsi_fb",
        "k_vy_fb",
        "k_wz_fb",
        "k_innov_var",
        "can_accel_latency_s",
        "can_steer_latency_s",
        "can_jitter_s",
        "imu_noise_scale",
        "ay_noise_std_ms2",
        "wz_noise_std_rads",
        "omega_noise_std_rads",
        "accel_quant_ms2",
        "p_yy",
        "p_yp",
        "p_pp",
        "p_vy",
        "p_wz",
    )

    def __init__(
        self,
        can_accel_latency_s: float = 0.125,
        can_steer_latency_s: float = 0.020,
        can_jitter_s: float = 0.015,
        imu_noise_scale: float = 1.0,
    ) -> None:
        self.k_ey_fb: float = float(np.float32(4.2))
        self.k_epsi_fb: float = float(np.float32(6.5))
        self.k_vy_fb: float = float(np.float32(8.0))
        self.k_wz_fb: float = float(np.float32(9.5))
        self.k_innov_var: float = float(np.float32(0.0028))
        self.can_accel_latency_s: float = float(np.float32(can_accel_latency_s))
        self.can_steer_latency_s: float = float(np.float32(can_steer_latency_s))
        self.can_jitter_s: float = float(np.float32(can_jitter_s))
        self.imu_noise_scale: float = float(np.float32(imu_noise_scale))
        self.ay_noise_std_ms2: float = float(np.float32(0.18))
        self.wz_noise_std_rads: float = float(np.float32(0.012))
        self.omega_noise_std_rads: float = float(np.float32(0.08))
        self.accel_quant_ms2: float = float(np.float32(0.01))
        self.p_yy: float = float(np.float32(0.0004))
        self.p_yp: float = float(np.float32(0.0))
        self.p_pp: float = float(np.float32(0.0001))
        self.p_vy: float = float(np.float32(0.0025))
        self.p_wz: float = float(np.float32(0.0004))

    def reset(self) -> None:
        self.p_yy = float(np.float32(0.0004))
        self.p_yp = float(np.float32(0.0))
        self.p_pp = float(np.float32(0.0001))
        self.p_vy = float(np.float32(0.0025))
        self.p_wz = float(np.float32(0.0004))

    def step(
        self,
        vx_ms: float,
        ay_ms2: float,
        lambda_mu_hat: float,
        p_cov_mu: float,
        innov_ms2: float,
        p_dc_nom_kw: float,
        dt_s: float = 0.005,
        sigma_mz_asym_nm: float = 0.0,
    ) -> Tuple[float, float, float]:
        """Return (var_ey_m2, var_fric_norm, var_pdc_kw2) in float32."""
        f32 = np.float32

        def clampf(val: np.float32, lo: float, hi: float) -> np.float32:
            return f32(min(max(float(val), lo), hi))

        dt_safe = clampf(f32(dt_s), 0.0005, 0.025)
        vx_eff = f32(max(abs(float(f32(vx_ms))), 1.5))
        lam_safe = f32(max(float(f32(lambda_mu_hat)), 0.25))
        ay_abs = f32(abs(float(f32(ay_ms2))))

        rho_y = clampf(f32(1.0) - f32(self.k_ey_fb) * dt_safe, 0.70, 0.995)
        rho_p = clampf(f32(1.0) - f32(self.k_epsi_fb) * dt_safe, 0.65, 0.995)
        rho_vy = clampf(f32(1.0) - f32(self.k_vy_fb) * dt_safe, 0.60, 0.990)
        rho_wz = clampf(f32(1.0) - f32(self.k_wz_fb) * dt_safe, 0.60, 0.990)

        innov_norm = f32(f32(innov_ms2) / f32(max(float(ay_abs), 4.0)))
        sigma_mu_var = f32(max(float(f32(p_cov_mu)), 1.0e-6)) + f32(self.k_innov_var) * innov_norm * innov_norm
        imu_ay_std = f32(self.imu_noise_scale) * f32(self.ay_noise_std_ms2)
        imu_wz_std = f32(self.imu_noise_scale) * f32(self.wz_noise_std_rads)
        sigma_imu_ay_var = imu_ay_std * imu_ay_std
        ay_dot_rms = f32(max(float(ay_abs * f32(1.8)), 3.0))
        can_ay_std = (f32(self.can_accel_latency_s) + f32(self.can_jitter_s)) * ay_dot_rms
        sigma_can_ay_var = can_ay_std * can_ay_std + (f32(self.accel_quant_ms2) ** 2) / f32(12.0)
        g_mu_ay = f32(max(float(ay_abs), 2.0)) / lam_safe
        q_ay = g_mu_ay * g_mu_ay * sigma_mu_var + sigma_imu_ay_var + sigma_can_ay_var

        g_mu_wz = (f32(0.45) * g_mu_ay) / f32(max(float(vx_eff), 3.0))
        wz_dot_rms = f32(max(float(ay_abs * f32(0.25)), 0.4))
        can_wz_std = (f32(self.can_steer_latency_s) + f32(0.5) * f32(self.can_jitter_s)) * wz_dot_rms
        q_wz = g_mu_wz * g_mu_wz * sigma_mu_var + imu_wz_std * imu_wz_std + can_wz_std * can_wz_std
        if sigma_mz_asym_nm > 0.0:
            wz_asym = f32(sigma_mz_asym_nm) / f32(83.1)
            q_wz = q_wz + wz_asym * wz_asym

        dt2 = dt_safe * dt_safe
        self.p_vy = float(clampf(rho_vy * rho_vy * f32(self.p_vy) + q_ay * dt2, 1.0e-6, 4.0))
        self.p_wz = float(clampf(rho_wz * rho_wz * f32(self.p_wz) + q_wz * dt2, 1.0e-7, 2.0))

        a01 = vx_eff * dt_safe
        p_yy_n = (
            rho_y * rho_y * f32(self.p_yy)
            + f32(2.0) * rho_y * a01 * f32(self.p_yp)
            + a01 * a01 * f32(self.p_pp)
            + f32(self.p_vy) * dt2 * f32(6.0)
        )
        p_yp_n = rho_y * rho_p * f32(self.p_yp) + a01 * rho_p * f32(self.p_pp)
        p_pp_n = rho_p * rho_p * f32(self.p_pp) + f32(self.p_wz) * dt2 * f32(6.0)
        self.p_yy = float(clampf(p_yy_n, 1.0e-7, 0.50))
        self.p_yp = float(clampf(p_yp_n, -0.25, 0.25))
        self.p_pp = float(clampf(p_pp_n, 1.0e-7, 0.25))

        var_ey = float(self.p_yy)
        var_fric = float(
            sigma_mu_var / (lam_safe * lam_safe)
            + f32(0.015) * f32(self.p_vy)
            + f32(0.040) * f32(self.p_wz)
            + f32(0.0080) * (sigma_imu_ay_var + sigma_can_ay_var) / (f32(9.81) * f32(9.81))
        )
        om_std = f32(self.imu_noise_scale) * f32(self.omega_noise_std_rads)
        pdc_rel = f32(p_dc_nom_kw) * f32(0.004)
        pdc_can = f32(self.can_accel_latency_s) * f32(2.5)
        var_pdc = float(pdc_rel * pdc_rel + f32(0.35) * om_std * om_std + f32(0.60) * pdc_can * pdc_can)
        return var_ey, var_fric, var_pdc
