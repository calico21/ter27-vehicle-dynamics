#!/usr/bin/env python3
"""
Stochastic Tube-MPC Controller for TeR27 4WD.

Combines:
  - OnlineGripEstimator (live lambda_mu_hat & p_cov = sigma_lambda_mu^2)
  - ExactCovariancePropagator (CasADi AD Jacobians A_k, B_k, G_k) or
    EmbeddedDiagonalTubePropagator (float32 / 0 B heap STM32F405 parity)
  - ChanceConstraintEvaluator (eps = 1e-4, >99.99% no off-track guarantee)
  - InHubLossModel + optimize_dc_power_allocation + RegenAbsController
    (strict [-25.0, +80.0] kW HV DC cap & BSPD EV 2.3 compliance)
  - Directional stability guarantee dMz/dbeta < 0
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Optional

import numpy as np

from vd_correlation_lab.control.sensors_ekf.online_mu_and_relaxation import (
    LAMBDA_MU_COLD_TEST_OCT,
    OnlineGripEstimator,
)
from vd_correlation_lab.control.stochastic_tube.chance_constraints import (
    ChanceBackoffResult,
    ChanceConstraintConfig,
    ChanceConstraintEvaluator,
)
from vd_correlation_lab.control.stochastic_tube.covariance_propagator import (
    EmbeddedDiagonalTubePropagator,
    ExactCovariancePropagator,
    ProcessNoiseConfig,
    ProcessNoiseModel,
)
from vd_correlation_lab.control.torque_vectoring.alqp_power_abs_pitch import (
    InHubLossModel,
    RegenAbsController,
    optimize_dc_power_allocation,
)
from vd_correlation_lab.models.differentiable_casadi import NU, STATE_IDX


@dataclass
class TubeMPCConfig:
    """Closed-loop controller parameters for TeR27 4WD Stochastic Tube-MPC."""

    stochastic_enabled: bool = True
    use_exact_jacobians: bool = True
    jacobian_stride: int = 2               # Compute exact 32x32 CasADi Jacobians every N ticks
    preview_horizon_steps: int = 6         # Multi-step covariance preview for chance tightening
    mass_kg: float = 290.0
    wheelbase_m: float = 1.540
    track_m: float = 1.230
    r_eff_m: float = 0.2032
    t_wheel_max_nm: float = 312.5
    t_regen_max_nm: float = 150.0
    max_steer_rad: float = 0.45
    # Steering & lateral corridor tracking gains
    k_stanley_ey: float = 0.55
    k_stanley_epsi: float = 1.15
    k_ey_dot: float = 0.14
    # Yaw & sideslip stabilization (strictly dMz/dbeta < 0)
    k_yaw_r: float = 720.0
    k_beta_stab: float = 680.0             # Mz_stab = -k_beta_stab * beta -> dMz/dbeta = -680 < 0
    k_beta_dot: float = 120.0
    # Longitudinal speed tracking
    k_v_p: float = 820.0
    # Nominal lateral acceleration scaling (mu_eff = mu_y_per_lam * lambda_mu_hat)
    mu_y_per_lambda: float = 2.68          # FSG_RUBBERED 0.57 -> ~1.53 g; COLD_TEST_OCT 0.47 -> ~1.26 g
    # Additive asymmetric yaw-disturbance Q_w & 3WD limp-home tube handling
    asymmetric_qw_enabled: bool = False
    asym_yaw_damp_boost: float = 0.85
    asym_beta_stab_boost: float = 0.65
    asym_corridor_gain_boost: float = 0.65
    asym_speed_backoff_gain: float = 0.12
    limp_home_detect_thresh_nm: float = 18.0
    limp_home_confirm_ticks: int = 3
    can_torque_delay_ticks: int = 4


class StochasticTubeMPCController:
    """
    Closed-loop Stochastic Tube-MPC / 4WD Torque-Vectoring controller.

    When `stochastic_enabled=False`, runs as an apples-to-apples deterministic
    controller (zero covariance backoff, operating at nominal friction limit).
    When `stochastic_enabled=True`, propagates state covariance with exact
    CasADi Jacobians (or embedded float32 diagonal approximation) and tightens:
      - Safe corridor / cornering speed reference (cone clearance >= 0.150 m)
      - AL-QP per-wheel friction ellipse effective radius
      - HV DC power cap [-25.0, +80.0] kW
    """

    def __init__(
        self,
        cfg: Optional[TubeMPCConfig] = None,
        chance_cfg: Optional[ChanceConstraintConfig] = None,
        noise_cfg: Optional[ProcessNoiseConfig] = None,
        compiled_lib: Optional[Any] = None,
        lambda_mu_init: float = LAMBDA_MU_COLD_TEST_OCT,
        p_init_mu: float = 8.0e-4,
    ) -> None:
        self.cfg = cfg if cfg is not None else TubeMPCConfig()
        self.chance_eval = ChanceConstraintEvaluator(chance_cfg)
        self.noise_cfg = noise_cfg if noise_cfg is not None else ProcessNoiseConfig()
        self.noise_model = ProcessNoiseModel(self.noise_cfg)
        self.exact_prop = (
            ExactCovariancePropagator(
                noise_model=self.noise_model,
                compiled_lib=compiled_lib,
                track_half_m=0.5 * self.cfg.track_m,
                r_eff_m=self.cfg.r_eff_m,
            )
            if self.cfg.use_exact_jacobians
            else None
        )
        self.embedded_prop = EmbeddedDiagonalTubePropagator(
            can_accel_latency_s=self.noise_cfg.can_accel_latency_s,
            can_steer_latency_s=self.noise_cfg.can_steer_latency_s,
            can_jitter_s=self.noise_cfg.can_jitter_s,
            imu_noise_scale=self.noise_cfg.imu_noise_scale,
        )
        is_ideal_sensor = (
            self.noise_cfg.imu_noise_scale <= 1e-6
            and self.noise_cfg.can_accel_latency_s <= 1e-6
        )
        self.grip_est = OnlineGripEstimator(
            mass_kg=self.cfg.mass_kg,
            lambda_mu_init=lambda_mu_init,
            p_init=p_init_mu,
            r_meas=0.04 if is_ideal_sensor else 1.20,
            rls_forgetting=0.992 if is_ideal_sensor else 0.945,
            q_proc=1.0e-6 if is_ideal_sensor else 1.5e-4,
            p_min=1.0e-6 if is_ideal_sensor else 1.0e-5,
        )
        self.loss_model = InHubLossModel()
        self.regen_abs = RegenAbsController(
            r_eff_m=self.cfg.r_eff_m,
            track_f_m=self.cfg.track_m,
            track_r_m=self.cfg.track_m,
        )
        self._tick: int = 0
        self._last_var_ey: float = 0.0004
        self._last_var_fric: float = 0.0002
        self._last_var_pdc: float = 0.01
        self._last_rho_cl: float = 0.95
        self._last_p_dc_kw: float = 0.0
        self._ey_prev: float = 0.0
        # Asymmetric yaw-disturbance & 3WD limp-home estimator state
        self._lr_slip_asym_filt: float = 0.0
        self._torque_resid_filt_nm: float = 0.0
        self._yaw_innov_filt_rads: float = 0.0
        self._last_sigma_mz_asym_nm: float = 0.0
        self._motor_fault_mask: np.ndarray = np.zeros(4, dtype=bool)
        self._motor_fault_counts: np.ndarray = np.zeros(4, dtype=np.int32)
        self._t_cmd_hist: list = [np.zeros(4, dtype=np.float64) for _ in range(max(self.cfg.can_torque_delay_ticks, 1) + 1)]

    def reset(
        self,
        lambda_mu_init: Optional[float] = None,
        p_init_mu: Optional[float] = None,
        init_pos_std_m: float = 0.02,
    ) -> None:
        if p_init_mu is not None:
            self.grip_est.p_init = float(np.float32(p_init_mu))
        self.grip_est.reset(lambda_mu_init=lambda_mu_init)
        if self.exact_prop is not None:
            self.exact_prop.reset(init_pos_std_m=init_pos_std_m)
        self.embedded_prop.reset()
        self.regen_abs.reset()
        self._tick = 0
        self._last_var_ey = init_pos_std_m ** 2
        self._last_var_fric = 0.0002
        self._last_var_pdc = 0.01
        self._last_rho_cl = 0.95
        self._last_p_dc_kw = 0.0
        self._ey_prev = 0.0
        self._lr_slip_asym_filt = 0.0
        self._torque_resid_filt_nm = 0.0
        self._yaw_innov_filt_rads = 0.0
        self._last_sigma_mz_asym_nm = 0.0
        self._motor_fault_mask = np.zeros(4, dtype=bool)
        self._motor_fault_counts = np.zeros(4, dtype=np.int32)
        self._t_cmd_hist = [np.zeros(4, dtype=np.float64) for _ in range(max(self.cfg.can_torque_delay_ticks, 1) + 1)]

    def _allocate_3wd_limp_home(
        self,
        fx_req: float,
        mz_req: float,
        t_ell_4: np.ndarray,
        is_braking: bool,
    ) -> np.ndarray:
        """
        Priority-weighted analytical 3-wheel torque allocation under motor fault.
        Prioritizes yaw moment Mz_req (directional stability) over Fx_req,
        guarantees zero torque on faulted wheel(s), and respects BSPD EV 2.3.
        """
        healthy = ~self._motor_fault_mask
        n_ok = int(np.sum(healthy))
        if n_ok == 0:
            return np.zeros(4, dtype=np.float64)

        idx = np.flatnonzero(healthy)
        side_sgn = np.array([-1.0, +1.0, -1.0, +1.0], dtype=np.float64)
        inv_r = 1.0 / max(self.cfg.r_eff_m, 0.1)
        b_mz = (0.5 * self.cfg.track_m) * inv_r

        # Build 2 x n_ok effectiveness matrix A: [Fx; Mz] = A @ T_healthy
        a_mat = np.zeros((2, n_ok), dtype=np.float64)
        a_mat[0, :] = inv_r
        a_mat[1, :] = side_sgn[idx] * b_mz

        # Prioritize yaw moment Mz_req while preserving balanced longitudinal drive
        w_vec = np.array([0.65, 3.2], dtype=np.float64)
        aw = a_mat * w_vec[:, None]
        h_mat = a_mat.T @ aw + 1.0e-3 * np.eye(n_ok, dtype=np.float64)
        rhs = aw.T @ np.array([fx_req, mz_req], dtype=np.float64)
        t_h = np.linalg.solve(h_mat, rhs)

        t_out = np.zeros(4, dtype=np.float64)
        t_out[idx] = t_h
        if is_braking:
            t_out = np.clip(t_out, -np.minimum(self.cfg.t_regen_max_nm, t_ell_4), 0.0)
        else:
            t_out = np.clip(t_out, -np.minimum(self.cfg.t_regen_max_nm, t_ell_4), t_ell_4)
        t_out[self._motor_fault_mask] = 0.0
        return t_out

    def step(
        self,
        x_true: np.ndarray,
        u_prev: np.ndarray,
        theta_nom: np.ndarray,
        w_nom: np.ndarray,
        meas: Dict[str, Any],
        e_y: float,
        e_psi: float,
        psi_ref: float,
        kappa_ref: float,
        v_target_nom: float,
        fx_nom_est_n: float,
        fy_nom_est_n: float,
        dt: float = 0.005,
        imu_noise_scale_override: Optional[float] = None,
        can_latency_override_s: Optional[float] = None,
        can_jitter_override_s: Optional[float] = None,
        t_meas_4: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Compute one 200 Hz control tick returning u[9] and full tube telemetry."""
        vx_m = max(float(meas["vx"]), 0.5)
        vy_m = float(meas["vy"])
        wz_m = float(meas["wz"])
        ax_m = float(meas["ax"])
        ay_m = float(meas["ay"])
        fz_4 = np.maximum(np.asarray(meas["fz_4"], dtype=np.float64), 60.0)
        om_4 = np.asarray(meas["omega_4"], dtype=np.float64)
        beta_m = float(meas["beta_hat"])

        # 1. Online recursive grip estimation (live lambda_mu_hat & p_cov = sigma_lambda_mu^2)
        grip_out = self.grip_est.step(
            ax_comp_ms2=ax_m,
            ay_comp_ms2=ay_m,
            fx_nom_n=fx_nom_est_n,
            fy_nom_n=fy_nom_est_n,
            dt_s=dt,
            fz_ref_n=float(np.mean(fz_4)),
        )
        lam_hat = float(grip_out["lambda_mu_hat"])
        p_cov_mu = float(grip_out["p_cov"])
        innov_ms2 = float(grip_out["innovation_ms2"])

        # 1b. Asymmetric yaw-disturbance innovation residuals (when enabled)
        lr_slip_asym = 0.0
        torque_resid_nm = 0.0
        yaw_innov_rads = 0.0
        if self.cfg.asymmetric_qw_enabled and self.cfg.stochastic_enabled:
            # Left-right wheel-speed / slip asymmetry minus kinematic yaw-rate differential
            dv_kin = -wz_m * self.cfg.track_m
            dv_meas = 0.5 * ((om_4[0] + om_4[2]) - (om_4[1] + om_4[3])) * self.cfg.r_eff_m
            lr_slip_raw = abs(dv_meas - dv_kin) / max(vx_m, 3.0)
            lr_slip_raw = max(0.0, lr_slip_raw - 0.025)  # deadband above encoder noise & cornering compliance
            self._lr_slip_asym_filt = 0.65 * self._lr_slip_asym_filt + 0.35 * lr_slip_raw

            # Torque-delivery yaw-moment residual & 3WD motor-loss detector
            # Compare delayed inverter feedback t_meas_4(t) against delay-aligned t_cmd_4(t - tau_can)
            mz_resid_raw = 0.0
            t_cmd_aligned = self._t_cmd_hist[0]
            if t_meas_4 is not None and self._tick >= max(self.cfg.can_torque_delay_ticks + 2, 6):
                tm = np.asarray(t_meas_4, dtype=np.float64)
                dt_err = tm - t_cmd_aligned
                mz_resid_raw = abs(
                    (0.5 * self.cfg.track_m / max(self.cfg.r_eff_m, 0.1))
                    * ((dt_err[1] + dt_err[3]) - (dt_err[0] + dt_err[2]))
                )
                mz_resid_raw = max(0.0, mz_resid_raw - 15.0)
                for wi in range(4):
                    if (
                        abs(float(t_cmd_aligned[wi])) > self.cfg.limp_home_detect_thresh_nm
                        and abs(float(tm[wi])) < 2.0
                    ):
                        self._motor_fault_counts[wi] += 1
                        if self._motor_fault_counts[wi] >= self.cfg.limp_home_confirm_ticks:
                            self._motor_fault_mask[wi] = True
                    else:
                        self._motor_fault_counts[wi] = max(0, int(self._motor_fault_counts[wi]) - 1)
            self._torque_resid_filt_nm = 0.55 * self._torque_resid_filt_nm + 0.45 * mz_resid_raw

            # Yaw-rate innovation residual vs kinematic bicycle expectation (deadband rejects normal slalom lag)
            delta_meas = float(meas.get("delta", u_prev[0]))
            r_kin = vx_m * math.tan(delta_meas) / max(self.cfg.wheelbase_m * (1.0 + 0.0012 * vx_m * vx_m), 0.5)
            yaw_innov_raw = max(0.0, abs(wz_m - r_kin) - 0.22)
            self._yaw_innov_filt_rads = 0.65 * self._yaw_innov_filt_rads + 0.35 * yaw_innov_raw

            lr_slip_asym = self._lr_slip_asym_filt
            torque_resid_nm = self._torque_resid_filt_nm
            yaw_innov_rads = self._yaw_innov_filt_rads

            # Compute immediate sigma_mz_asym_nm for embedded propagator & active yaw tube
            _, nb_tmp = self.noise_model.build_qw(
                lambda_mu_hat=lam_hat,
                p_cov_mu=p_cov_mu,
                innov_ms2=innov_ms2,
                ay_ms2=ay_m,
                ax_ms2=ax_m,
                imu_noise_scale=imu_noise_scale_override,
                can_accel_latency_s=can_latency_override_s,
                can_jitter_s=can_jitter_override_s,
                lr_slip_asym=lr_slip_asym,
                torque_delivery_resid_nm=torque_resid_nm,
                yaw_innov_rads=yaw_innov_rads,
            )
            self._last_sigma_mz_asym_nm = float(nb_tmp["sigma_mz_asym_nm"])
        else:
            self._last_sigma_mz_asym_nm = 0.0

        # 2. Covariance propagation (exact CasADi Jacobians + embedded float32 parity check)
        emb_var_ey, emb_var_fric, emb_var_pdc = self.embedded_prop.step(
            vx_ms=vx_m,
            ay_ms2=ay_m,
            lambda_mu_hat=lam_hat,
            p_cov_mu=p_cov_mu,
            innov_ms2=innov_ms2,
            p_dc_nom_kw=self._last_p_dc_kw,
            dt_s=dt,
            sigma_mz_asym_nm=self._last_sigma_mz_asym_nm,
        )

        if self.cfg.stochastic_enabled:
            if self.exact_prop is not None and (self._tick % max(self.cfg.jacobian_stride, 1) == 0):
                prop = self.exact_prop.step_covariance(
                    x=x_true,
                    u=u_prev,
                    theta=theta_nom,
                    w=w_nom,
                    psi_ref=psi_ref,
                    lambda_mu_hat=lam_hat,
                    p_cov_mu=p_cov_mu,
                    innov_ms2=innov_ms2,
                    ay_ms2=ay_m,
                    ax_ms2=ax_m,
                    p_dc_nom_kw=self._last_p_dc_kw,
                    imu_noise_scale=imu_noise_scale_override,
                    can_accel_latency_s=can_latency_override_s,
                    can_jitter_s=can_jitter_override_s,
                    n_preview_steps=self.cfg.preview_horizon_steps,
                    lr_slip_asym=lr_slip_asym,
                    torque_delivery_resid_nm=torque_resid_nm,
                    yaw_innov_rads=yaw_innov_rads,
                )
                self._last_var_ey = prop["var_ey_m2"]
                self._last_var_fric = prop["var_fric_norm"]
                self._last_var_pdc = prop["var_pdc_kw2"]
                self._last_rho_cl = prop["rho_cl_lat_yaw"]
            elif self.exact_prop is None:
                self._last_var_ey = emb_var_ey
                self._last_var_fric = emb_var_fric
                self._last_var_pdc = emb_var_pdc

            backoff: ChanceBackoffResult = self.chance_eval.evaluate_from_variances(
                var_ey_m2=self._last_var_ey,
                var_fric_norm=self._last_var_fric,
                var_pdc_kw2=self._last_var_pdc,
            )
            # Also compute embedded backoff for exact vs embedded comparison
            emb_backoff = self.chance_eval.evaluate_from_variances(
                var_ey_m2=emb_var_ey,
                var_fric_norm=emb_var_fric,
                var_pdc_kw2=emb_var_pdc,
            )
        else:
            # Deterministic baseline: no covariance backoff (operates at full nominal limit)
            backoff = ChanceBackoffResult(
                z_track=self.chance_eval.z_track,
                z_friction=self.chance_eval.z_friction,
                z_power=self.chance_eval.z_power,
                sigma_ey_m=0.0,
                sigma_fric_norm=0.0,
                sigma_pdc_kw=0.0,
                b_cone_m=0.0,
                b_friction=0.0,
                b_pdc_kw=0.0,
                corridor_half_margin_m=self.chance_eval.corridor_body_clearance_m,
                effective_cone_limit_m=self.chance_eval.corridor_body_clearance_m - self.chance_eval.cfg.min_cone_clearance_m,
                friction_ellipse_scale=0.995,
                p_dc_max_eff_kw=79.95,
                p_dc_regen_eff_kw=-24.95,
                v_ref_scale=1.0,
            )
            emb_backoff = backoff

        # 3. Steering law: curvature feedforward + Stanley lateral/heading feedback + tube corridor barrier
        ey_dot = float(np.clip((e_y - self._ey_prev) / max(dt, 1e-4), -10.0, 10.0)) if self._tick > 0 else 0.0
        self._ey_prev = float(e_y)

        asym_sev = (
            min(self._last_sigma_mz_asym_nm / 450.0, 1.0)
            if (self.cfg.asymmetric_qw_enabled and self.cfg.stochastic_enabled)
            else 0.0
        )

        delta_ff = math.atan(self.cfg.wheelbase_m * kappa_ref)
        # Tube-aware corridor barrier gain increases smoothly when |e_y| approaches effective_cone_limit_m
        ey_lim = max(backoff.effective_cone_limit_m, 0.08)
        corridor_prox = max(0.0, (abs(e_y) - 0.55 * ey_lim) / max(0.45 * ey_lim, 0.05))
        k_ey_eff = self.cfg.k_stanley_ey * (
            1.0
            + (1.4 * corridor_prox if self.cfg.stochastic_enabled else 0.0)
            + self.cfg.asym_corridor_gain_boost * asym_sev
        )
        k_epsi_eff = self.cfg.k_stanley_epsi * (1.0 + 0.35 * asym_sev)
        delta_fb = -math.atan2(k_ey_eff * e_y, max(vx_m, 3.5)) - k_epsi_eff * e_psi - self.cfg.k_ey_dot * ey_dot
        delta_cmd = float(np.clip(delta_ff + delta_fb, -self.cfg.max_steer_rad, self.cfg.max_steer_rad))

        # 4. Safe reference speed from grip belief & chance-constraint tube tightening
        mu_y_est = self.cfg.mu_y_per_lambda * lam_hat
        # Account for downforce at current speed
        f_aero = 0.5 * 1.225 * 6.0 * vx_m * vx_m
        g_eff = 9.81 + f_aero / self.cfg.mass_kg
        ay_cap_phys = mu_y_est * g_eff * backoff.friction_ellipse_scale
        if abs(kappa_ref) > 1e-4:
            v_corner_cap = math.sqrt(max(ay_cap_phys / abs(kappa_ref), 1.0))
        else:
            v_corner_cap = v_target_nom

        # In stochastic mode, also slow proactively if corridor error + tube approaches limit
        v_safe = min(v_target_nom, v_corner_cap) * backoff.v_ref_scale
        if self.cfg.stochastic_enabled and corridor_prox > 0.25:
            v_safe *= max(0.75, 1.0 - 0.20 * min(corridor_prox - 0.25, 1.2))
        if asym_sev > 0.0:
            v_safe *= max(0.74, 1.0 - self.cfg.asym_speed_backoff_gain * asym_sev)

        fx_req = float(np.clip(self.cfg.k_v_p * (v_safe - vx_m), -3800.0, 4200.0))

        # 5. Yaw moment command with strict directional stability dMz/dbeta < 0
        r_ref_raw = vx_m * kappa_ref
        r_cap = (ay_cap_phys / max(vx_m, 3.0))
        r_ref = float(np.clip(r_ref_raw, -r_cap, r_cap))

        # Analytical sideslip derivative dMz/dbeta strictly negative
        beta_dot_est = (ay_m / max(vx_m, 2.0)) - wz_m
        k_yaw_r_eff = self.cfg.k_yaw_r * (1.0 + self.cfg.asym_yaw_damp_boost * asym_sev)
        k_beta_stab_eff = self.cfg.k_beta_stab * (1.0 + self.cfg.asym_beta_stab_boost * asym_sev)
        mz_yaw = k_yaw_r_eff * (r_ref - wz_m)
        mz_beta = -k_beta_stab_eff * beta_m - self.cfg.k_beta_dot * beta_dot_est
        dmz_dbeta = -float(k_beta_stab_eff)  # strictly < 0
        mz_lim = 950.0 if asym_sev > 0.0 else 650.0
        mz_req = float(np.clip(mz_yaw + mz_beta, -mz_lim, mz_lim))

        # 6. 4WD AL-QP style friction-ellipse-tightened torque allocation
        f_cap_4 = backoff.friction_ellipse_scale * mu_y_est * fz_4
        # Estimate lateral load fraction from ay_m
        ay_util = min(abs(ay_m) / max(mu_y_est * g_eff, 2.0), 0.94)
        fx_cap_4 = f_cap_4 * math.sqrt(max(1.0 - ay_util * ay_util, 0.04))
        t_ell_4 = np.minimum(fx_cap_4 * self.cfg.r_eff_m, self.cfg.t_wheel_max_nm)

        is_braking = fx_req < -150.0
        if self.cfg.asymmetric_qw_enabled and np.any(self._motor_fault_mask):
            t_cmd_4 = self._allocate_3wd_limp_home(
                fx_req=fx_req,
                mz_req=mz_req,
                t_ell_4=t_ell_4,
                is_braking=is_braking,
            )
        else:
            # Distribute Fx_req proportionally to Fz and add differential Mz_req
            fz_sum = max(float(np.sum(fz_4)), 100.0)
            t_fx_4 = (fx_req * (fz_4 / fz_sum)) * self.cfg.r_eff_m
            dt_mz = 0.25 * mz_req * self.cfg.r_eff_m / max(0.5 * self.cfg.track_m, 0.3)
            t_cmd_4 = t_fx_4 + np.array([-dt_mz, +dt_mz, -dt_mz, +dt_mz], dtype=np.float64)

            if is_braking:
                # Strict BSPD EV 2.3: never positive motor torque when braking
                t_cmd_4 = np.clip(t_cmd_4, -np.minimum(self.cfg.t_regen_max_nm, t_ell_4), 0.0)
            else:
                t_cmd_4 = np.clip(t_cmd_4, -np.minimum(self.cfg.t_regen_max_nm, t_ell_4), t_ell_4)

        # 7. Strict HV DC Power Cap [-25.0, +80.0] kW tightened by chance-constraint backoff
        p_max_w = backoff.p_dc_max_eff_kw * 1000.0
        p_eval = self.loss_model.evaluate_4(t_cmd_4, om_4)
        if float(p_eval["p_dc_kw"]) * 1000.0 > p_max_w:
            # Preserve yaw differential if possible, then water-fill under p_max_w
            pos_mask = t_cmd_4 > 0.0
            if np.any(pos_mask):
                alloc = optimize_dc_power_allocation(
                    loss_model=self.loss_model,
                    omega_w_4=om_4,
                    t_max_4=np.maximum(t_cmd_4, 0.0),
                    p_dc_limit_w=p_max_w,
                    fz_4=fz_4,
                )
                t_cmd_4 = np.where(pos_mask, alloc["t_wheel_4"], t_cmd_4)
        if self.cfg.asymmetric_qw_enabled and np.any(self._motor_fault_mask):
            t_cmd_4[self._motor_fault_mask] = 0.0

        # Apply per-wheel RegenAbsController under braking (BSPD EV 2.3 & strict -25.0 kW cap)
        t_brk_4 = np.zeros(4, dtype=np.float64)
        if is_braking:
            rabs_out = self.regen_abs.step(
                t_cmd_4=t_cmd_4,
                omega_4=om_4,
                vx_ms=vx_m,
                fz_4=fz_4,
                lambda_mu_hat=lam_hat,
                fx_req=fx_req,
                dt=dt,
                wz_rads=wz_m,
                loss_model=self.loss_model,
            )
            t_cmd_4 = np.minimum(rabs_out["t_wheel_4"], 0.0)
            if self.cfg.asymmetric_qw_enabled and np.any(self._motor_fault_mask):
                t_cmd_4[self._motor_fault_mask] = 0.0
                # If one front wheel is faulted, balance the opposite front wheel to avoid braking yaw pull
                if self._motor_fault_mask[0] and not self._motor_fault_mask[1]:
                    t_cmd_4[1] = 0.0
                elif self._motor_fault_mask[1] and not self._motor_fault_mask[0]:
                    t_cmd_4[0] = 0.0
            # Enforce tightened stochastic regen cap if stricter than -24.95 kW
            p_reg_lim_w = abs(backoff.p_dc_regen_eff_kw) * 1000.0
            p_now_w = float(self.loss_model.evaluate_4(t_cmd_4, om_4)["p_dc_kw"]) * 1000.0
            if p_now_w < -p_reg_lim_w and p_now_w < -1.0:
                sc = float(np.clip(p_reg_lim_w / (-p_now_w), 0.0, 1.0))
                t_cmd_4 *= sc
            # Remainder braking force handled by hydraulic brakes (58/42 bias)
            fx_regen = float(np.sum(t_cmd_4)) / self.cfg.r_eff_m
            fx_hyd_need = max(0.0, -fx_req - max(-fx_regen, 0.0))
            t_hyd_f = 0.5 * 0.58 * fx_hyd_need * self.cfg.r_eff_m
            t_hyd_r = 0.5 * 0.42 * fx_hyd_need * self.cfg.r_eff_m
            t_brk_4 = np.array([t_hyd_f, t_hyd_f, t_hyd_r, t_hyd_r], dtype=np.float64)

        p_final = self.loss_model.evaluate_4(t_cmd_4, om_4)
        self._last_p_dc_kw = float(p_final["p_dc_kw"])
        self._t_cmd_hist.pop(0)
        self._t_cmd_hist.append(t_cmd_4.copy())
        self._tick += 1

        u_cmd = np.zeros(NU, dtype=np.float64)
        u_cmd[0] = delta_cmd
        u_cmd[1:5] = t_cmd_4
        u_cmd[5:9] = t_brk_4

        bspd_viol = bool(is_braking and np.any(t_cmd_4 > 1e-6))
        hv_viol = bool(self._last_p_dc_kw > 80.001 or self._last_p_dc_kw < -25.001)

        return {
            "u_cmd": u_cmd,
            "delta_cmd": delta_cmd,
            "t_wheel_4": t_cmd_4,
            "t_brk_4": t_brk_4,
            "v_safe": v_safe,
            "fx_req": fx_req,
            "mz_req": mz_req,
            "dMz_dbeta": dmz_dbeta,
            "lambda_mu_hat": lam_hat,
            "p_cov_mu": p_cov_mu,
            "innov_ms2": innov_ms2,
            "p_dc_kw": self._last_p_dc_kw,
            "hv_violation": hv_viol,
            "bspd_violation": bspd_viol,
            "rho_cl_lat_yaw": self._last_rho_cl,
            "backoff": backoff,
            "embedded_backoff": emb_backoff,
            "sigma_mz_asym_nm": self._last_sigma_mz_asym_nm,
            "lr_slip_asym": lr_slip_asym,
            "torque_delivery_resid_nm": torque_resid_nm,
            "yaw_innov_rads": yaw_innov_rads,
            "motor_fault_mask": [bool(v) for v in self._motor_fault_mask],
        }
