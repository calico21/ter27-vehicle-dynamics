#!/usr/bin/env python3
"""
Unit tests for TeR27 4WD advanced embedded control modules:
  1. Online grip estimator (lambda_mu_hat) convergence (0.47 <-> 0.57 & COLD_18C),
     zero oscillation, cruise gate, and directional stability preservation (dMz/dbeta < 0).
  2. Per-wheel 200 Hz regenerative ABS wheel-lock prevention under strict BSPD EV 2.3
     (T_i <= 0 under braking) and -25.0 kW HV DC charge ceiling.
  3. Real in-hub electrical loss map (I^2*R(T) + inverter + field-weakening Id)
     net mechanical kW gain at strict 80.0 kW HV DC ceiling.
  4. Active aero-pitch longitudinal split stability (56.9% anti-dive / 41.2% anti-squat,
     zero authority in Skidpad) & Hoosier tire relaxation length precompensator.
  5. STM32F405 @ 168 MHz / 200 Hz CPU load gate (< 8.5%, < 425 us / 5000 us).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from vd_correlation_lab.control.sensors_ekf.online_mu_and_relaxation import (
    HOOSIER_SIGMA_ALPHA_M,
    HOOSIER_SIGMA_KAPPA_M,
    LAMBDA_MU_COLD_18C,
    LAMBDA_MU_COLD_TEST_OCT,
    LAMBDA_MU_FSG_RUBBERED,
    OnlineGripEstimator,
    TireRelaxationCompensator,
)
from vd_correlation_lab.control.skidpad.skidpad_tv_controller import (
    SkidpadMaxAttackController,
)
from vd_correlation_lab.control.torque_vectoring.alqp_power_abs_pitch import (
    AeroPitchController,
    InHubLossModel,
    RegenAbsController,
    optimize_dc_power_allocation,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
VAULT_08_METRICS = (
    REPO_ROOT
    / "outputs"
    / "validation_vault"
    / "08_codegen_sil_and_godot_parity"
    / "metrics.json"
)


def _run_grip_step(
    lam_init: float,
    lam_true: float,
    noisy: bool = False,
    duration_s: float = 2.0,
    dt_s: float = 0.005,
) -> tuple[list[float], float, float]:
    """Simulate a 200 Hz lateral cornering step for OnlineGripEstimator."""
    est = OnlineGripEstimator(lambda_mu_init=lam_init)
    rng = np.random.default_rng(2027)
    n_steps = int(round(duration_s / dt_s))
    ay_nom = 10.5  # m/s^2 nominal lateral accel at lambda_mu = 0.47
    fy_nom = ay_nom * est.mass_kg
    ay_true = ay_nom * (lam_true / LAMBDA_MU_COLD_TEST_OCT)

    trace: list[float] = []
    for _ in range(n_steps):
        if noisy:
            ax_m = float(np.round(0.08 + rng.normal(0.0, 0.18), 2))
            ay_m = float(np.round(ay_true - 0.06 + rng.normal(0.0, 0.18), 2))
        else:
            ax_m = 0.0
            ay_m = ay_true
        lam_hat, _, _, _, _ = est.step_scalar(
            ax_comp_ms2=ax_m,
            ay_comp_ms2=ay_m,
            fx_nom_n=0.0,
            fy_nom_n=fy_nom,
            dt_s=dt_s,
        )
        trace.append(lam_hat)

    tail = np.asarray(trace[int(0.5 * n_steps) :], dtype=float)
    ss_rmse = float(np.sqrt(np.mean((tail - lam_true) ** 2)))
    if lam_true >= lam_init:
        overshoot = float(max(0.0, max(trace) - lam_true))
    else:
        overshoot = float(max(0.0, lam_true - min(trace)))
    return trace, ss_rmse, overshoot


def test_online_grip_estimator_convergence_and_stability() -> None:
    """Verify 0.47 <-> 0.57 and COLD_18C convergence, zero oscillation, cruise gate, and dMz/dbeta < 0."""
    # 1. Noiseless & noisy transitions
    for lam_init, lam_true in (
        (LAMBDA_MU_COLD_TEST_OCT, LAMBDA_MU_FSG_RUBBERED),
        (LAMBDA_MU_FSG_RUBBERED, LAMBDA_MU_COLD_TEST_OCT),
        (LAMBDA_MU_FSG_RUBBERED, LAMBDA_MU_COLD_18C),
    ):
        trace_clean, rmse_clean, ov_clean = _run_grip_step(lam_init, lam_true, noisy=False)
        assert abs(trace_clean[-1] - lam_true) < 0.005, (
            f"Clean convergence failed {lam_init}->{lam_true}: final={trace_clean[-1]:.4f}"
        )
        assert rmse_clean < 0.005
        assert ov_clean < 0.002, f"Unexpected overshoot {ov_clean:.5f} on {lam_init}->{lam_true}"

        trace_noisy, rmse_noisy, _ = _run_grip_step(lam_init, lam_true, noisy=True)
        assert rmse_noisy < 0.015, (
            f"Noisy steady-state RMSE too high {lam_init}->{lam_true}: {rmse_noisy:.5f}"
        )
        assert abs(trace_noisy[-1] - lam_true) < 0.02

    # 2. Straight-line cruise gate: zero drift below accel_gate_ms2 (3.5 m/s^2)
    est = OnlineGripEstimator(lambda_mu_init=LAMBDA_MU_COLD_TEST_OCT)
    for _ in range(200):
        lam_hat, _, _, _, active = est.step_scalar(
            ax_comp_ms2=0.8,
            ay_comp_ms2=0.5,
            fx_nom_n=300.0,
            fy_nom_n=100.0,
            dt_s=0.005,
        )
        assert not active
    assert abs(lam_hat - LAMBDA_MU_COLD_TEST_OCT) < 1e-6

    # 3. Integrated SkidpadMaxAttackController preserves directional stability dMz/dbeta < 0
    ctrl = SkidpadMaxAttackController()
    out_ss = ctrl.step(
        fx_req=600.0,
        delta_rad=0.18,
        vx_ms=10.5,
        vy_ms=-0.12,
        wz_rads=1.22,
        ay_meas=12.8,
        fz_4=np.array([380.0, 1050.0, 440.0, 1280.0]),
        mu_avg=1.35,
        dt=0.005,
    )
    assert float(out_ss["dMz_dbeta"]) < 0.0
    assert -25.0 <= float(out_ss["p_dc_kw"]) <= 80.0


def test_regen_abs_prevents_lockup_with_bspd_and_hv_limits() -> None:
    """Verify per-wheel 200 Hz regen ABS prevents inner-front lockup under BSPD EV 2.3 and -25 kW."""
    abs_ctrl = RegenAbsController()
    loss_model = InHubLossModel()
    dt = 0.005
    r_eff = 0.2032
    j_wheel = 0.38  # kg*m^2 effective wheel+drivetrain inertia

    # Trail-braking scenario: unloaded inner-front wheel (FL = 210 N)
    fz_4 = np.array([210.0, 960.0, 420.0, 810.0], dtype=float)
    t_req_4 = np.array([-135.0, -135.0, -110.0, -110.0], dtype=float)
    vx = 20.0
    omega_no_abs = np.full(4, vx / r_eff, dtype=float)
    omega_abs = np.full(4, vx / r_eff, dtype=float)

    lock_ticks_no_abs = 0
    lock_ticks_abs = 0
    min_kappa_abs = 0.0
    bspd_violations = 0
    hv_violations = 0

    for _ in range(160):
        # Uncontrolled baseline (same regen torque without ABS)
        v_ref = max(vx, 1.5)
        kap_no = np.clip((omega_no_abs * r_eff - vx) / v_ref, -1.0, 1.0)
        if np.any(kap_no < -0.20):
            lock_ticks_no_abs += 1
        # Simple tire braking torque saturation mu * Fz * r_eff
        mu_tire = 0.57 * 2.1
        fx_no = np.clip(kap_no / 0.085, -1.0, 1.0) * mu_tire * fz_4
        omega_no_abs = np.maximum(0.0, omega_no_abs + ((t_req_4 - fx_no * r_eff) / j_wheel) * dt)

        # Controlled with RegenAbsController
        out = abs_ctrl.step(
            t_cmd_4=t_req_4,
            omega_4=omega_abs,
            vx_ms=vx,
            fz_4=fz_4,
            lambda_mu_hat=LAMBDA_MU_FSG_RUBBERED,
            fx_req=-3200.0,
            dt=dt,
            wz_rads=0.45,
            loss_model=loss_model,
        )
        t_out = out["t_wheel_4"]
        if np.any(t_out > 1e-6):
            bspd_violations += 1
        p_dc_kw = float(loss_model.evaluate_4(t_out, omega_abs)["p_dc_kw"])
        if p_dc_kw < -25.001:
            hv_violations += 1

        kap_abs = out["kappa_4"]
        min_kappa_abs = min(min_kappa_abs, float(np.min(kap_abs)))
        if np.any(kap_abs < -0.20):
            lock_ticks_abs += 1

        fx_abs = np.clip(kap_abs / 0.085, -1.0, 1.0) * mu_tire * fz_4
        omega_abs = np.maximum(0.0, omega_abs + ((t_out - fx_abs * r_eff) / j_wheel) * dt)
        vx = max(5.0, vx - 7.5 * dt)

    assert bspd_violations == 0, "BSPD EV 2.3 violated under braking"
    assert hv_violations == 0, "HV -25.0 kW regen limit violated"
    assert lock_ticks_no_abs > 50, "Baseline scenario must provoke inner-front lockup"
    assert lock_ticks_abs == 0, f"Regen ABS failed to prevent lockup: {lock_ticks_abs} ticks"
    assert min_kappa_abs > -0.15, f"Peak negative slip too deep: {min_kappa_abs:.4f}"


def test_inhub_electrical_loss_map_80kw_dc_gain() -> None:
    """Verify real I^2*R(T)/FW allocator increases net mechanical kW at strict 80.0 kW DC."""
    loss_model = InHubLossModel()
    t_max_4 = np.full(4, 312.5, dtype=float)
    omega_4 = np.array([81.2, 81.2, 81.2, 81.2], dtype=float)
    temps_4 = np.array([45.0, 45.0, 98.0, 98.0], dtype=float)  # hot rear motors

    opt = optimize_dc_power_allocation(
        loss_model=loss_model,
        omega_w_4=omega_4,
        t_max_4=t_max_4,
        temp_c_4=temps_4,
        p_dc_limit_w=79950.0,
    )
    assert opt["is_power_limited"] is True
    assert opt["p_dc_kw"] <= 80.0, f"HV DC ceiling exceeded: {opt['p_dc_kw']:.3f} kW"

    # Compare against fixed 35/65 front/rear split scaled to exact 79.95 kW DC
    split_weights = np.array([0.35, 0.35, 0.65, 0.65], dtype=float)
    lo, hi = 0.0, 500.0
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        p_mid = float(loss_model.evaluate_4(split_weights * mid, omega_4, temps_4)["p_dc_kw"])
        if p_mid <= 79.95:
            lo = mid
        else:
            hi = mid
    res_fixed = loss_model.evaluate_4(split_weights * lo, omega_4, temps_4)

    gain_kw = float(opt["p_mech_kw"]) - float(res_fixed["p_mech_kw"])
    assert gain_kw > 0.50, f"Expected > 0.50 kW net mech gain vs 35/65 split, got {gain_kw:.4f} kW"

    # Cooler front motors must receive more torque than hotter rear motors
    tw = opt["t_wheel_4"]
    assert tw[0] > tw[2] and tw[1] > tw[3]

    # Verify field-weakening Id activates above base speed (9500 rpm / 12.5 = 79.6 rad/s)
    id_low = loss_model.field_weakening_id(np.full(4, 40.0))
    id_high = loss_model.field_weakening_id(np.full(4, 120.0))
    assert float(np.max(id_low)) == 0.0
    assert float(np.min(id_high)) > 10.0


def test_aero_pitch_and_tire_relaxation_precompensator() -> None:
    """Verify active aero-pitch stability (65-105 km/h, zero in Skidpad) & relaxation lead-lag."""
    # 1. AeroPitchController
    aero = AeroPitchController()

    # Exact zero authority at Skidpad speed (~11 m/s < 15 m/s)
    st_skid = aero.compute_split_adjustment(vx_ms=11.5, fx_req_n=-1500.0, base_front_share=0.50)
    assert abs(st_skid["delta_s_front"]) < 1e-9

    # High-speed braking (95 km/h = 26.39 m/s): reduces nose-down dive and protects front RH
    st_base_brk = aero.estimate_platform_state(vx_ms=26.39, fx_front_n=-1800.0, fx_rear_n=-1800.0)
    st_ctrl_brk = aero.compute_split_adjustment(vx_ms=26.39, fx_req_n=-3600.0, base_front_share=0.50)
    assert st_ctrl_brk["delta_s_front"] > 0.0
    assert abs(st_ctrl_brk["theta_deg"]) < abs(st_base_brk["theta_deg"])
    assert st_ctrl_brk["front_rh_mm"] > st_base_brk["front_rh_mm"]

    # High-speed acceleration (85 km/h = 23.61 m/s): shifts drive rearward to reduce squat
    st_base_acc = aero.estimate_platform_state(vx_ms=23.61, fx_front_n=1400.0, fx_rear_n=1400.0)
    st_ctrl_acc = aero.compute_split_adjustment(vx_ms=23.61, fx_req_n=2800.0, base_front_share=0.50)
    assert st_ctrl_acc["delta_s_front"] < 0.0
    assert abs(st_ctrl_acc["theta_deg"]) < abs(st_base_acc["theta_deg"])

    # 2. TireRelaxationCompensator
    comp = TireRelaxationCompensator()
    assert abs(comp.sigma_alpha_m - HOOSIER_SIGMA_ALPHA_M) < 1e-6
    assert abs(comp.sigma_kappa_m - HOOSIER_SIGMA_KAPPA_M) < 1e-6

    # Exact zero lead above 12.0 m/s
    comp.reset()
    for k in range(40):
        mz_in = 150.0 * math.sin(2.0 * math.pi * 1.5 * k * 0.005)
        mz_out, fx_out, kap_out = comp.step_scalar(
            vx_ms=14.0,
            mz_ff_nm=mz_in,
            fx_launch_n=2500.0,
            kappa_target=0.10,
            dt_s=0.005,
            dist_m=20.0,
        )
        assert abs(mz_out - mz_in) < 1e-4
        assert abs(fx_out - 2500.0) < 1e-3
        assert abs(kap_out - 0.10) < 1e-5

    # Low-speed hairpin exit (vx = 5.5 m/s): lead compensator adds phase lead on rising Mz_ff
    comp.reset()
    mz_in_ramp = 0.0
    lead_seen = False
    for k in range(30):
        mz_in_ramp = 250.0 * (k / 29.0)
        mz_out, _, _ = comp.step_scalar(vx_ms=5.5, mz_ff_nm=mz_in_ramp, dt_s=0.005)
        if k > 5 and mz_out > mz_in_ramp + 2.0:
            lead_seen = True
    assert lead_seen, "TireRelaxationCompensator failed to provide phase lead at vx=5.5 m/s"

    # Integrated controller runs cleanly at high speed with aero-pitch active
    ctrl = SkidpadMaxAttackController()
    out_hs = ctrl.step(
        fx_req=-2500.0,
        delta_rad=0.04,
        vx_ms=25.0,
        vy_ms=0.0,
        wz_rads=0.15,
        ay_meas=3.5,
        fz_4=np.array([900.0, 900.0, 520.0, 520.0]),
        mu_avg=1.40,
        dt=0.005,
    )
    assert -25.0 <= float(out_hs["p_dc_kw"]) <= 80.0
    assert np.all(np.asarray(out_hs["t_motor_4"]) <= 1e-6)


def test_stm32f405_cpu_load_budget() -> None:
    """Verify STM32F405 @ 168 MHz / 200 Hz FPU float32 CPU load stays strictly < 8.5% (< 425 us)."""
    assert VAULT_08_METRICS.exists(), f"Missing Vault 08 metrics: {VAULT_08_METRICS}"
    data = json.loads(VAULT_08_METRICS.read_text(encoding="utf-8"))
    c99 = data["c99_sil"]
    assert c99["accuracy"]["pass"] is True
    assert float(c99["accuracy"]["torque_max_abs_err"]) == 0.0
    assert float(c99["accuracy"]["kappa_max_abs_err"]) == 0.0

    stm = c99["stm32f405_168mhz_model"]
    fpu = stm["fpu_f32"]
    assert fpu["pass"] is True
    assert float(fpu["total_tick_us"]) < 425.0, (
        f"STM32F405 tick time {fpu['total_tick_us']:.2f} us exceeds 425 us limit"
    )
    assert float(fpu["cpu_load_pct"]) < 8.5, (
        f"STM32F405 CPU load {fpu['cpu_load_pct']:.2f}% exceeds 8.5% limit"
    )
