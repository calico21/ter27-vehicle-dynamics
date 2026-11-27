#!/usr/bin/env python3
"""
Pytest verification suite for TeR27 Sensor Realism Profile (CAN latency + IMU
noise/bias + EKF beta/Fz) and Validation Vault 10 (`10_ter27_dil_and_sensor_realism`).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from vd_correlation_lab.config.paths import (
    MIRENA_SIM_DIR,
    VAULT_10_DIL_SENSOR_REALISM,
)
from vd_correlation_lab.control.sensors_ekf.sensor_realism_profile import (
    SensorMode,
    SensorRealismProfile,
)


TV4_CFG_JSON = (
    MIRENA_SIM_DIR
    / "Scenes"
    / "Vehicle"
    / "Config"
    / "vehicles"
    / "ter27_4wd"
    / "tv_4wd_config.json"
)


def test_sensor_ideal_is_exact_passthrough() -> None:
    prof = SensorRealismProfile.from_tv_config_json(TV4_CFG_JSON, mode=SensorMode.SENSOR_IDEAL)
    fz = np.array([620.0, 580.0, 690.0, 640.0])
    om = np.array([85.0, 86.0, 84.5, 85.5])
    out = prof.step(3.25, -7.80, 0.65, 17.5, -0.45, 0.12, fz, om, 0.005)
    assert abs(out["ax"] - 3.25) < 1e-12
    assert abs(out["ay"] - (-7.80)) < 1e-12
    assert abs(out["wz"] - 0.65) < 1e-12
    assert abs(out["delta"] - 0.12) < 1e-12
    assert np.allclose(out["fz_hat_4"], fz, atol=1e-12)
    assert abs(out["beta_hat"] - np.arctan2(-0.45, 17.5)) < 1e-12
    assert out["mode"] == int(SensorMode.SENSOR_IDEAL)


def test_sensor_track_realism_latency_determinism_and_ekf() -> None:
    p1 = SensorRealismProfile.from_tv_config_json(TV4_CFG_JSON, mode=SensorMode.SENSOR_TRACK_REALISM)
    p2 = SensorRealismProfile.from_tv_config_json(TV4_CFG_JSON, mode=SensorMode.SENSOR_TRACK_REALISM)
    p1.reset(seed=1337)
    p2.reset(seed=1337)

    dt = 0.005
    fz = np.array([600.0, 600.0, 650.0, 650.0])
    om = np.array([90.0, 90.0, 90.0, 90.0])

    ax_trace_1 = []
    ax_trace_2 = []
    beta_hats = []
    fz_hats = []
    for k in range(80):
        ax_in = 8.0 if k >= 10 else 0.0
        o1 = p1.step(ax_in, 4.0, 0.3, 18.0, -0.3, 0.08, fz, om, dt)
        o2 = p2.step(ax_in, 4.0, 0.3, 18.0, -0.3, 0.08, fz, om, dt)
        ax_trace_1.append(o1["ax"])
        ax_trace_2.append(o2["ax"])
        beta_hats.append(o1["beta_hat"])
        fz_hats.append(o1["fz_hat_4"])

    # Exact deterministic seed reproducibility
    assert np.allclose(ax_trace_1, ax_trace_2, atol=1e-12)

    # 125 ms CAN accel latency -> at k=15 (25 ms after step at k=10), ax has NOT stepped yet
    assert ax_trace_1[15] < 2.0
    # By k=40 (150 ms after step at k=10), ax has stepped to ~8.0 m/s^2
    assert ax_trace_1[40] > 6.5

    # EKF beta_hat & fz_hat_4 remain finite and physically bounded
    assert np.all(np.isfinite(beta_hats))
    assert np.max(np.abs(beta_hats)) < 0.5
    fz_arr = np.asarray(fz_hats)
    assert np.all(np.isfinite(fz_arr))
    assert np.all(fz_arr > 0.0) and np.all(fz_arr < 4000.0)


def test_vault_10_metrics_and_four_300dpi_plots() -> None:
    v10 = VAULT_10_DIL_SENSOR_REALISM
    assert v10.is_dir(), f"Missing Vault 10 directory: {v10}"

    metrics_path = v10 / "dil_sensor_realism_metrics.json"
    assert metrics_path.exists(), f"Missing {metrics_path}"
    data = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert abs(float(data["can_accel_latency_ms"]) - 125.0) <= 10.0
    assert data["skidpad"]["track_realism"]["hv_violations"] == 0
    assert data["trail_braking_slalom"]["track_realism"]["hv_violations"] == 0
    assert float(data["braking_ramp_3d"]["peak_elevation_m"]) > 0.25
    assert int(data["braking_ramp_3d"]["nan_count"]) == 0

    expected_pngs = [
        "fig01_can_latency_ekf_step_and_noise.png",
        "fig02_skidpad_and_slalom_sensor_stress.png",
        "fig03_autocross_casadi_sector_deltas.png",
        "fig04_3d_braking_ramp_and_dil_telemetry.png",
    ]
    for name in expected_pngs:
        p = v10 / name
        assert p.exists(), f"Missing Vault 10 figure: {p}"
        assert p.stat().st_size > 20 * 1024, f"{p} smaller than 20 KB"
        with Image.open(p) as im:
            dpi_x, dpi_y = im.info["dpi"]
            assert abs(float(dpi_x) - 300.0) <= 1.0
            assert abs(float(dpi_y) - 300.0) <= 1.0
