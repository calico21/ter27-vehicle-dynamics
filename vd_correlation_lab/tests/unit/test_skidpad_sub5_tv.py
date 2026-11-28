#!/usr/bin/env python3
"""
Unit & Vault Verification Tests for TeR27 Skidpad Dedicated TV (SKIDPAD_MAX_ATTACK_V2),
FSG_RUBBERED calibration from TeR26 RWD (5.50 s anchor), CasADi OCP oracle,
and Validation Vault 09 (12/12 300 DPI MoTeC diagnostic figures).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image

from vd_correlation_lab.config.paths import REPO_ROOT, VAULT_03_TV_MAPS, VAULT_DIR
from vd_correlation_lab.control.skidpad.skidpad_tv_controller import SkidpadMaxAttackController

VAULT_09_DEEP_DIVE: Path = VAULT_DIR / "09_ter27_skidpad_deep_dive"

EXPECTED_VAULT_09_PNGS: List[str] = [
    "01_skidpad_time_waterfall_and_benchmark.png",
    "02_skidpad_trajectory_and_cone_clearance.png",
    "03_vehicle_kinematics_beta_yaw_lataccel.png",
    "04_4wheel_friction_circles_and_utilization.png",
    "05_4wheel_slip_angles_and_long_slip.png",
    "06_4wheel_vertical_loads_and_weight_transfer.png",
    "07_suspension_3d_camber_toe_ackermann.png",
    "08_controller_mz_decomposition_and_tracking.png",
    "09_4wheel_torques_power_and_torque_fight.png",
    "10_figure8_crossover_micro_analysis.png",
    "11_skidpad_sensitivity_setup_and_grip_sweep.png",
    "12_python_vs_godot_skidpad_parity.png",
]

REQUIRED_V2_TELEMETRY_KEYS: List[str] = [
    "mz_req_nm",
    "fx_4",
    "t_motor_4",
    "p_dc_kw",
    "is_crossover",
    "r_ref",
    "mz_ff",
    "mz_fb_p",
    "mz_fb_i",
    "mz_fb_d",
    "mz_beta_damp",
    "mz_crossover",
    "mz_total",
    "mz_sat_flag",
    "torque_fight_kw",
    "wheel_mu_util",
    "T_cmd",
    "T_actual",
    "P_dc_kw",
    "dMz_dbeta",
]


def _extract_time_s(entry: Dict[str, Any]) -> float:
    for k in ("t_mean_s", "t_f8_mean_s", "t_circle_s", "t_skidpad_s", "t_opt_s", "skidpad_time_s"):
        if k in entry and entry[k] is not None:
            return float(entry[k])
    raise KeyError(f"No recognized time key in {sorted(entry.keys())}")


def test_fsg_grip_calibration_from_ter26() -> None:
    cal_path = VAULT_03_TV_MAPS / "fsg_grip_calibration_from_ter26.json"
    assert cal_path.exists(), f"Missing {cal_path}"
    cal = json.loads(cal_path.read_text(encoding="utf-8"))

    t_ter26 = float(cal["ter26_rwd_skidpad_s"])
    assert abs(t_ter26 - 5.50) <= 0.05, f"TeR26 RWD FSG_RUBBERED time {t_ter26:.4f} s outside 5.50 +/- 0.05 s"
    assert float(cal["min_clear_m"]) >= 0.149, f"TeR26 calibration min_clear_m={cal['min_clear_m']} < 0.15 m"
    mu_scale = float(cal["mu_fsg_scale"])
    assert 0.50 < mu_scale < 1.50, f"Unphysical mu_fsg_scale={mu_scale}"
    assert np.isfinite(float(cal["T_fsg_c"]))


def test_skidpad_oracle_report_valid() -> None:
    p = VAULT_03_TV_MAPS / "skidpad_oracle_report.json"
    assert p.exists(), f"Missing {p}"
    data = json.loads(p.read_text(encoding="utf-8"))
    results = data["results"]
    summary = {row["combo"]: row for row in data["summary_table"]}

    for key in ["FSG_RUBBERED__R20", "FSG_RUBBERED__R25B", "TTC_PEAK__R20", "TTC_PEAK__R25B"]:
        assert key in results, f"Missing {key} in oracle results"
        row = summary[key]
        assert float(row["min_clear_m"]) >= 0.149, f"{key} min_clear_m={row['min_clear_m']} < 0.15 m"
        assert int(row["hv_violations"]) == 0, f"{key} hv_violations={row['hv_violations']}"
        assert -25.05 <= float(row["p_dc_stat_kw"]) <= 80.05

    # TeR27 4WD < 5.00 s in calibrated FSG_RUBBERED (R20 and R25B) and <= 4.64 s on R25B
    for key in ["FSG_RUBBERED__R20", "FSG_RUBBERED__R25B"]:
        t_val = _extract_time_s(summary[key])
        assert t_val < 5.00, f"Oracle {key} time {t_val:.4f} s >= 5.00 s"
    assert _extract_time_s(summary["FSG_RUBBERED__R25B"]) <= 4.64, (
        "FSG_RUBBERED__R25B Skidpad OCP exceeded 4.64 s gate"
    )


def test_skidpad_max_attack_v2_controller_hv_crossover_and_telemetry() -> None:
    ctrl = SkidpadMaxAttackController()
    fz = np.array([380.0, 1050.0, 440.0, 1280.0])

    # Steady left turn
    out_ss = ctrl.step(
        fx_req=600.0,
        delta_rad=0.18,
        vx_ms=10.2,
        vy_ms=-0.12,
        wz_rads=1.18,
        ay_meas=11.4,
        fz_4=fz,
        mu_avg=1.30,
        dt=0.005,
    )
    for key in REQUIRED_V2_TELEMETRY_KEYS:
        assert key in out_ss, f"Missing SKIDPAD_MAX_ATTACK_V2 telemetry key: {key}"

    assert -25.0 <= float(out_ss["p_dc_kw"]) <= 80.0
    assert abs(float(out_ss["P_dc_kw"]) - float(out_ss["p_dc_kw"])) < 1e-6
    assert out_ss["is_crossover"] is False
    # Outer > Inner in steady turn
    assert out_ss["t_motor_4"][3] > out_ss["t_motor_4"][2] > 0.0
    assert float(out_ss["torque_fight_kw"]) >= 0.0
    assert float(out_ss["dMz_dbeta"]) < 0.0, f"Unstable dMz_dbeta={out_ss['dMz_dbeta']} (must be < 0)"
    assert len(out_ss["wheel_mu_util"]) == 4
    assert len(out_ss["T_cmd"]) == 4
    assert len(out_ss["T_actual"]) == 4

    # Crossover steering reversal at Skidpad speed (< 14.5 m/s)
    out_cr = ctrl.step(
        fx_req=250.0,
        delta_rad=0.12,
        vx_ms=10.5,
        vy_ms=0.05,
        wz_rads=0.80,
        ay_meas=-8.0,
        fz_4=fz,
        mu_avg=1.30,
        dt=0.005,
    )
    assert out_cr["is_crossover"] is True
    assert -25.0 <= float(out_cr["p_dc_kw"]) <= 80.0
    for key in REQUIRED_V2_TELEMETRY_KEYS:
        assert key in out_cr, f"Missing V2 telemetry key in crossover: {key}"


def test_skidpad_vault_plots_and_catalog() -> None:
    cat_path = VAULT_03_TV_MAPS / "tv_metrics_catalog.json"
    assert cat_path.exists()
    cat = json.loads(cat_path.read_text(encoding="utf-8"))
    bench = cat["skidpad_sub5_benchmark"]["closed_loop_14dof"]

    assert bench["FSG_RUBBERED_R20_MAX_ATTACK"]["t_mean_s"] < bench["FSG_RUBBERED_R20_LEGACY"]["t_mean_s"]
    assert bench["FSG_RUBBERED_R20_MAX_ATTACK"]["t_mean_s"] < 5.00
    assert bench["FSG_RUBBERED_R20_MAX_ATTACK"]["hv_violations"] == 0
    assert bench["FSG_RUBBERED_R20_MAX_ATTACK"]["min_cone_clear_m"] >= 0.15

    # Every closed-loop scenario (including COLD_18C) must have >= 0.15 m clearance and 0 HV violations
    for name, metrics in bench.items():
        if not isinstance(metrics, dict):
            continue
        if "min_cone_clear_m" in metrics:
            assert float(metrics["min_cone_clear_m"]) >= 0.15, (
                f"Vault 03 scenario {name} cone clearance {metrics['min_cone_clear_m']} < 0.15 m"
            )
        if "hv_violations" in metrics:
            assert int(metrics["hv_violations"]) == 0, f"Vault 03 scenario {name} HV violations != 0"

    for fname in [
        "skidpad_sub5_before_vs_after.png",
        "skidpad_4wheel_friction_circles.png",
        "skidpad_figure8_crossover_dynamics.png",
    ]:
        fp = VAULT_03_TV_MAPS / fname
        assert fp.exists() and fp.stat().st_size > 20_000, f"Invalid PNG {fp}"
        with Image.open(fp) as im:
            dpi = im.info.get("dpi", (0, 0))
            assert abs(dpi[0] - 300.0) <= 1.5 and abs(dpi[1] - 300.0) <= 1.5


def test_skidpad_deep_dive_vault_09_assets_and_gates() -> None:
    assert VAULT_09_DEEP_DIVE.is_dir(), f"Missing {VAULT_09_DEEP_DIVE}"

    # Verify all 12/12 PNGs exist, > 20 KB, 300 DPI
    for fname in EXPECTED_VAULT_09_PNGS:
        fp = VAULT_09_DEEP_DIVE / fname
        assert fp.exists(), f"Missing Vault 09 PNG: {fp}"
        assert fp.stat().st_size > 20_000, f"Vault 09 PNG too small ({fp.stat().st_size} B): {fp}"
        with Image.open(fp) as im:
            assert "dpi" in im.info, f"Missing DPI metadata in {fp}"
            dpi_x, dpi_y = im.info["dpi"]
            assert abs(float(dpi_x) - 300.0) <= 1.5 and abs(float(dpi_y) - 300.0) <= 1.5, (
                f"Invalid DPI {dpi_x}x{dpi_y} in {fp}"
            )

    # Verify skidpad_deep_dive_metrics.json
    metrics_path = VAULT_09_DEEP_DIVE / "skidpad_deep_dive_metrics.json"
    assert metrics_path.exists(), f"Missing {metrics_path}"
    raw_text = metrics_path.read_text(encoding="utf-8")
    non_ascii = [(i + 1, ch, ord(ch)) for i, ch in enumerate(raw_text) if ord(ch) > 127]
    assert len(non_ascii) == 0, f"Non-ASCII chars in {metrics_path}: {non_ascii[:10]}"

    data = json.loads(raw_text)
    assert isinstance(data, dict) and len(data) > 0

    # Walk nested scenario dicts to verify universal gates:
    # min_cone_clear_m >= 0.15 m (incl. COLD_18C) and hv_violations == 0
    found_r20_sub5 = False
    found_r25b_sub5 = False
    found_cold_18c = False

    def _walk(obj: Any, path_key: str = "") -> None:
        nonlocal found_r20_sub5, found_r25b_sub5, found_cold_18c
        if isinstance(obj, dict):
            if "min_cone_clear_m" in obj and isinstance(obj["min_cone_clear_m"], (int, float)):
                assert float(obj["min_cone_clear_m"]) >= 0.15, (
                    f"{path_key}: min_cone_clear_m={obj['min_cone_clear_m']} < 0.15 m"
                )
                if "COLD" in path_key.upper():
                    found_cold_18c = True
            if "hv_violations" in obj and isinstance(obj["hv_violations"], (int, float)):
                assert int(obj["hv_violations"]) == 0, (
                    f"{path_key}: hv_violations={obj['hv_violations']} != 0"
                )
            if "dMz_dbeta_max_nm_rad" in obj and isinstance(obj["dMz_dbeta_max_nm_rad"], (int, float)):
                assert float(obj["dMz_dbeta_max_nm_rad"]) < 0.0, (
                    f"{path_key}: dMz_dbeta_max_nm_rad={obj['dMz_dbeta_max_nm_rad']} >= 0"
                )
            up = path_key.upper()
            if "FSG_RUBBERED" in up and ("MAX_ATTACK" in up or "V2" in up) and "LEGACY" not in up:
                for tk in ("t_mean_s", "t_f8_mean_s", "t_circle_s", "t_skidpad_s"):
                    if tk in obj and isinstance(obj[tk], (int, float)):
                        t_val = float(obj[tk])
                        if "R20" in up:
                            assert t_val < 5.00, f"{path_key} R20 time {t_val:.4f} s >= 5.00 s"
                            found_r20_sub5 = True
                        if "R25B" in up:
                            assert t_val <= 4.70, f"{path_key} R25B time {t_val:.4f} s > 4.70 s"
                            found_r25b_sub5 = True
            for k, v in obj.items():
                _walk(v, f"{path_key}.{k}" if path_key else str(k))
        elif isinstance(obj, list):
            for idx, item in enumerate(obj):
                _walk(item, f"{path_key}[{idx}]")

    _walk(data)
    assert found_r20_sub5, "Did not find FSG_RUBBERED R20 MAX_ATTACK_V2 sub-5.00 s entry in skidpad_deep_dive_metrics.json"
    assert found_r25b_sub5, "Did not find FSG_RUBBERED R25B MAX_ATTACK_V2 sub-5.00 s entry in skidpad_deep_dive_metrics.json"
    assert found_cold_18c, "Did not find COLD_18C cone clearance verification in skidpad_deep_dive_metrics.json"
