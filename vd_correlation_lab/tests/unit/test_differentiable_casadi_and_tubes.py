#!/usr/bin/env python3
"""
Comprehensive Verification Suite for Differentiable CasADi 14-DOF,
Adjoint Optimization, Stochastic Tube-MPC, and Validation Vault Suite 11.

Covers:
  1. Orthogonal selection of all 3 vehicle models x all 4 suspension modes
     (Python registry, FastAPI endpoints, and ter27.py CLI).
  2. AD vs central finite-difference Jacobian & adjoint gradient check (< 1e-6).
  3. Empirical TeR26 correlation vs real track telemetry (omega_z R >= 0.984)
     and adjoint-gradient-only SysID parameter recovery (< 2%).
  4. Stochastic Tube-MPC propagation in sudden FSG_RUBBERED (0.57) ->
     COLD_TEST_OCT (0.47) grip drop + SENSOR_TRACK_REALISM (125 ms CAN):
     0 cones hit, min_clear >= 0.150 m, 0 HV/BSPD violations, dMz/dbeta < 0.
  5. STM32F405 @ 168 MHz / 200 Hz CPU budget (< 8.5% / < 425 us) & 0 B heap C99 SIL.
  6. Validation Vault Suite 11 (5 figures @ 300 DPI > 20 KB + pure ASCII).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app
from vd_correlation_lab.models.differentiable_casadi import (
    COLD_TEST_OCT,
    FSG_RUBBERED,
    load_compiled_14dof_library,
)
from vd_correlation_lab.models.registry import (
    SUSPENSION_MODES,
    VEHICLE_MODELS,
    create_vehicle_model,
    list_suspension_modes,
    list_vehicle_models,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUTS_DIR = REPO_ROOT / "outputs"
SUITE_11_DIR = OUTPUTS_DIR / "validation_vault" / "11_differentiable_casadi_and_tubes"


def test_orthogonal_vehicle_models_and_suspension_modes() -> None:
    """All 3 vehicle models x all 4 suspension modes must be selectable orthogonally."""
    assert set(VEHICLE_MODELS) == {
        "full_14dof",
        "fast_3dof",
        "differentiable_casadi_14dof",
    }
    assert set(SUSPENSION_MODES) == {
        "analytic_1d_lut",
        "3d_double_wishbone_lut",
        "decoupled_heave_roll",
        "standard_pushrod",
    }
    assert len(list_vehicle_models()) == 3
    assert len(list_suspension_modes()) == 4
    assert abs(COLD_TEST_OCT - 0.47) < 1e-9
    assert abs(FSG_RUBBERED - 0.57) < 1e-9

    for vid in ("ter26_rwd", "ter27_4wd"):
        for mid in VEHICLE_MODELS:
            for smode in SUSPENSION_MODES:
                m = create_vehicle_model(
                    model_id=mid,
                    vehicle_id=vid,
                    suspension_mode=smode,
                )
                assert m.model_id == mid
                assert m.suspension_mode == smode

    # Verify CLI and FastAPI expose all 3 models x 4 suspension modes
    proc = subprocess.run(
        [sys.executable, "-u", str(REPO_ROOT / "ter27.py"), "models"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    for mid in VEHICLE_MODELS:
        assert mid in proc.stdout
    for smode in SUSPENSION_MODES:
        assert smode in proc.stdout

    with TestClient(app) as client:
        r_ov = client.get("/api/vehicle/overview")
        assert r_ov.status_code == 200
        ov = r_ov.json()
        assert {m["id"] for m in ov["available_vehicle_models"]} == set(VEHICLE_MODELS)
        assert {s["id"] for s in ov["available_suspension_modes"]} == set(SUSPENSION_MODES)

        r_sim = client.post(
            "/api/simulation/run",
            json={
                "event": "skidpad",
                "grip_profile": "FSG_RUBBERED",
                "compound": "R25B",
                "vehicle_model": "differentiable_casadi_14dof",
                "suspension_mode": "decoupled_heave_roll",
                "dry_run": True,
            },
        )
        assert r_sim.status_code == 200
        sim = r_sim.json()
        assert sim["vehicle_model"] == "differentiable_casadi_14dof"
        assert sim["suspension_mode"] == "decoupled_heave_roll"

        r_app = client.post(
            "/api/setups/apply",
            json={
                "preset": "optimal_skidpad",
                "vehicle_model": "differentiable_casadi_14dof",
                "suspension_mode": "standard_pushrod",
            },
        )
        assert r_app.status_code == 200
        payload = r_app.json()["command_payload"]
        assert payload["vehicle_model"] == "differentiable_casadi_14dof"
        assert payload["suspension_mode"] == "standard_pushrod"


def test_ad_vs_finite_difference_jacobians_and_adjoint_gradients() -> None:
    """Verify compiled libter_diff_14dof.so AD Jacobians and adjoint gradients (< 1e-6)."""
    model = create_vehicle_model(
        model_id="differentiable_casadi_14dof",
        vehicle_id="ter27_4wd",
        suspension_mode="3d_double_wishbone_lut",
    )
    lib = load_compiled_14dof_library()
    x0 = model.default_state(vx=12.0, vy=0.4, wz=0.35)
    u0 = model.inputs_from_driver(delta_cmd=0.12, throttle=0.55, brake=0.0)
    theta0 = model.default_theta()
    w0 = model.default_disturbance(t_tire_c=75.0, t_motor_c=55.0)

    a_ad, b_ad, s_ad, _ = lib.step_jacobians(x0, u0, theta0, w0)
    a_ad = np.asarray(a_ad, dtype=float)
    b_ad = np.asarray(b_ad, dtype=float)
    s_ad = np.asarray(s_ad, dtype=float)

    # Central FD check on B_u (steer + wheel torques)
    eps = 1e-6
    b_fd = np.zeros_like(b_ad)
    for j in range(u0.size):
        du = np.zeros_like(u0)
        du[j] = eps
        xp, _ = lib.step_dynamics(x0, u0 + du, theta0, w0)
        xm, _ = lib.step_dynamics(x0, u0 - du, theta0, w0)
        b_fd[:, j] = (np.asarray(xp, dtype=float).ravel() - np.asarray(xm, dtype=float).ravel()) / (2.0 * eps)

    rel_err_b = float(np.max(np.abs(b_ad - b_fd)) / max(1.0, float(np.max(np.abs(b_fd)))))
    assert rel_err_b < 1e-6, f"B_u AD vs FD error {rel_err_b:.3e} >= 1e-6"
    assert np.all(np.isfinite(a_ad)) and np.all(np.isfinite(s_ad))

    # Adjoint report FD verification gate across Skidpad, Accel, Autocross
    adj_path = OUTPUTS_DIR / "differentiable_adjoint_report.json"
    assert adj_path.exists(), f"Missing {adj_path}"
    adj = json.loads(adj_path.read_text(encoding="utf-8"))
    fd_ver = adj["adjoint_fd_verification"]
    assert fd_ver["all_passed_1e_6"] is True
    assert float(fd_ver["worst_max_rel_err"]) < 1e-6
    assert float(fd_ver["worst_max_abs_err"]) < 1e-6


def test_ter26_empirical_telemetry_correlation_and_sysid() -> None:
    """TeR26 differentiable 14-DOF must achieve omega_z R >= 0.984 vs real track telemetry."""
    rep_path = OUTPUTS_DIR / "differentiable_ter26_correlation_report.json"
    assert rep_path.exists(), f"Missing {rep_path}"
    rep = json.loads(rep_path.read_text(encoding="utf-8"))

    gates = rep["acceptance_gates"]
    assert bool(gates["wz_pearson_r_ge_0984"]) is True
    assert bool(gates["adjoint_grad_vs_fd_lt_1e6"]) is True
    assert bool(gates["adjoint_sysid_recovered_lt_2pct"]) is True
    assert bool(gates["compiled_so_parity_lt_1pct"]) is True

    wz_r = float(rep["pooled_metrics_diff14_vs_real"]["wz_rads"]["r"])
    assert wz_r >= 0.984, f"TeR26 wz Pearson R={wz_r:.4f} < 0.984"
    assert float(rep["adjoint_gradient_verification"]["max_rel_err"]) < 1e-6
    assert float(rep["adjoint_sysid_recovery"]["max_recovery_err_pct"]) < 2.0


def test_stochastic_tube_propagation_and_stm32_budget() -> None:
    """Stochastic Tube-MPC: 0 cones in grip drop, min_clear >= 0.150 m, STM32F405 CPU < 8.5%."""
    tube_path = OUTPUTS_DIR / "stochastic_tube_report.json"
    assert tube_path.exists(), f"Missing {tube_path}"
    tube = json.loads(tube_path.read_text(encoding="utf-8"))

    gates = tube["acceptance_gates"]
    assert gates["overall_pass"] is True
    assert gates["all_scenarios_zero_cones_and_min_clear_ge_150mm"] is True
    assert gates["c99_sil_parity_pass"] is True
    assert gates["stm32f405_cpu_lt_8_5_pct_pass"] is True

    # Sudden grip drop 0.57 -> 0.47 + SENSOR_TRACK_REALISM (125 ms CAN)
    sc_drop = tube["scenarios"]["sudden_grip_drop_057_to_047_realism"]
    tub_s = sc_drop["stochastic_tube_mpc"]["summary"]
    det_s = sc_drop["deterministic"]["summary"]
    assert int(tub_s["cones_hit"]) == 0
    assert float(tub_s["min_cone_clearance_m"]) >= 0.150
    assert int(tub_s["hv_violations"]) == 0
    assert int(tub_s["bspd_violations"]) == 0
    assert float(tub_s["max_dMz_dbeta"]) < 0.0
    assert int(det_s["cones_hit"]) > 0

    # STM32F405 @ 168 MHz / 200 Hz CPU budget < 8.5% (< 425 us) & 0 B heap
    c99 = tube["c99_sil_and_stm32"]
    stm32 = c99["stm32f405_168mhz_200hz"]
    assert float(stm32["cpu_load_pct"]) < 8.5
    assert float(stm32["total_tick_us"]) < 425.0
    assert int(c99["heap_alloc_bytes"]) == 0
    assert float(c99["parity"]["max_abs_err"]) < 1e-6


def test_validation_vault_suite_11_assets_and_ascii() -> None:
    """Verify Suite 11 300 DPI plots, primary metrics JSON, and pure-ASCII compliance."""
    assert SUITE_11_DIR.is_dir(), f"Missing {SUITE_11_DIR}"
    required_pngs = [
        "fig01_ter26_diff14_vs_telemetry.png",
        "fig02_adjoint_gradient_verification_and_sysid.png",
        "fig03_ter27_adjoint_sensitivity_and_coopt.png",
        "fig04_stochastic_tube_grip_drop_and_risk.png",
        "fig05_three_model_comparison_matrix.png",
    ]
    for name in required_pngs:
        p = SUITE_11_DIR / name
        assert p.exists() and p.stat().st_size > 20 * 1024, f"Missing/small Suite 11 PNG: {p}"
        with Image.open(p) as im:
            dpi_x, dpi_y = im.info.get("dpi", (0, 0))
            assert abs(float(dpi_x) - 300.0) <= 1.0
            assert abs(float(dpi_y) - 300.0) <= 1.0

    m_path = SUITE_11_DIR / "differentiable_casadi_and_tubes_metrics.json"
    assert m_path.exists()
    text = m_path.read_text(encoding="utf-8")
    assert all(ord(ch) <= 127 for ch in text), "Non-ASCII found in Suite 11 metrics JSON"
    data = json.loads(text)
    assert data["overall_pass"] is True
