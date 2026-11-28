#!/usr/bin/env python3
"""
Automated QA Tests for FSG_RUBBERED (0.57) Calibration, Trackside Setup Co-Optimization,
MirenaSim Optimal Setup Presets, and Vault 07 Comparison Artifacts.

Verifies:
  1. COLD_TEST_OCT intact at lambda_muy = 0.47 and FSG_RUBBERED updated to 0.57 (R25B @ 75 C).
  2. Existence, JSON validity, and mechanical bounds of all 4 exported optimal_*.json presets.
  3. Validity of trackside_setup_optimization_report.json and trackside_setup_event_comparison.png (300 DPI).
  4. Strict pure-ASCII (zero emojis, zero non-ASCII bytes) in all new/modified files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image

from vd_correlation_lab.config.paths import (
    MIRENA_SIM_DIR,
    REPO_ROOT,
    TER27_CONFIG_DIR,
    VAULT_07_DESIGN_POINTS,
)
from vd_correlation_lab.models.vehicle_14dof.ter27_physics import GRIP_PROFILES

SETUPS_DIR: Path = TER27_CONFIG_DIR / "setups"
EXPECTED_PRESETS: List[str] = [
    "optimal_skidpad.json",
    "optimal_accel.json",
    "optimal_autocross.json",
    "optimal_endurance.json",
]

SETUP_BOUNDS: Dict[str, Tuple[float, float]] = {
    "lltd": (0.35, 0.62),
    "camber_f_deg": (-2.5, -0.2),
    "camber_r_deg": (-2.2, -0.2),
    "toe_f_deg": (-0.5, 0.5),
    "toe_r_deg": (0.0, 0.6),
    "brake_bias": (0.55, 0.78),
    "rake_mm": (-8.0, 8.0),
}


def _extract_lambda_muy(profile_dict: Dict[str, Any]) -> float:
    for key in ("lambda_muy_target", "lambda_muy", "lmuy", "lambda_muy_r25b"):
        if key in profile_dict and profile_dict[key] is not None:
            return float(profile_dict[key])
    if "env_mu_scale" in profile_dict:
        return 0.47 * float(profile_dict["env_mu_scale"])
    raise KeyError(f"Cannot resolve lambda_muy from grip profile keys: {sorted(profile_dict.keys())}")


def _extract_setup_param(data: Dict[str, Any], canonical: str) -> float | None:
    alias_map: Dict[str, List[str]] = {
        "lltd": ["lltd", "LLTD"],
        "camber_f_deg": ["camber_front_deg", "camber_f_deg"],
        "camber_r_deg": ["camber_rear_deg", "camber_r_deg"],
        "toe_f_deg": ["toe_front_deg", "toe_f_deg"],
        "toe_r_deg": ["toe_rear_deg", "toe_r_deg"],
        "brake_bias": ["brake_bias", "brake_bias_front"],
        "rake_mm": ["rake_mm", "rake"],
    }
    for key in alias_map[canonical]:
        if key in data and data[key] is not None:
            return float(data[key])
    # Fallback: derive rake_mm from ride heights in meters if present
    if canonical == "rake_mm" and "ride_height_rear_m" in data and "ride_height_front_m" in data:
        return (float(data["ride_height_rear_m"]) - float(data["ride_height_front_m"])) * 1000.0
    # Fallback: check nested setup_params / optimal_setup dict if present
    for nested_key in ("optimal_setup", "setup_params", "trackside_setup"):
        sub = data.get(nested_key)
        if isinstance(sub, dict):
            for key in alias_map[canonical]:
                if key in sub and sub[key] is not None:
                    return float(sub[key])
    return None


def test_grip_profiles_cold_oct_intact_and_fsg_rubbered_057() -> None:
    assert "COLD_TEST_OCT" in GRIP_PROFILES, "COLD_TEST_OCT missing from GRIP_PROFILES"
    assert "FSG_RUBBERED" in GRIP_PROFILES, "FSG_RUBBERED missing from GRIP_PROFILES"

    lmuy_cold = _extract_lambda_muy(GRIP_PROFILES["COLD_TEST_OCT"])
    lmuy_fsg = _extract_lambda_muy(GRIP_PROFILES["FSG_RUBBERED"])

    assert abs(lmuy_cold - 0.47) <= 0.005, f"COLD_TEST_OCT lambda_muy={lmuy_cold:.4f} != 0.47"
    assert abs(lmuy_fsg - 0.57) <= 0.005, f"FSG_RUBBERED lambda_muy={lmuy_fsg:.4f} != 0.57"


def test_optimal_setup_presets_exist_and_within_bounds() -> None:
    assert SETUPS_DIR.is_dir(), f"Missing setups directory: {SETUPS_DIR}"

    for fname in EXPECTED_PRESETS:
        p = SETUPS_DIR / fname
        assert p.exists(), f"Missing exported MirenaSim setup preset: {p}"
        raw = p.read_text(encoding="utf-8")
        non_ascii = [(i + 1, ch, ord(ch)) for i, ch in enumerate(raw) if ord(ch) > 127]
        assert len(non_ascii) == 0, f"Non-ASCII chars in {p}: {non_ascii[:10]}"

        data = json.loads(raw)
        assert isinstance(data, dict) and len(data) > 0, f"Invalid JSON dict in {p}"
        assert data.get("vehicle_id", "ter27_4wd") == "ter27_4wd"

        # Check all 7 trackside mechanical setup parameters are within physical bounds
        for param, (lo, hi) in SETUP_BOUNDS.items():
            val = _extract_setup_param(data, param)
            if val is not None:
                assert lo - 1e-4 <= val <= hi + 1e-4, (
                    f"{fname}: {param}={val} outside [{lo}, {hi}]"
                )


def test_trackside_setup_optimization_report_and_plot() -> None:
    report_path = VAULT_07_DESIGN_POINTS / "trackside_setup_optimization_report.json"
    assert report_path.exists(), f"Missing {report_path}"
    raw = report_path.read_text(encoding="utf-8")
    non_ascii = [(i + 1, ch, ord(ch)) for i, ch in enumerate(raw) if ord(ch) > 127]
    assert len(non_ascii) == 0, f"Non-ASCII chars in {report_path}: {non_ascii[:10]}"

    report = json.loads(raw)
    assert isinstance(report, dict) and len(report) > 0

    events_root = report
    for cand in ("events", "event_results", "comparison", "results"):
        if cand in report and isinstance(report[cand], dict):
            events_root = report[cand]
            break

    low_keys = {str(k).lower(): v for k, v in events_root.items()}
    for aliases in (
        ["skidpad"],
        ["accel", "acceleration", "acceleration_75m", "accel_75m"],
        ["autocross"],
        ["endurance", "endurance_22km"],
    ):
        found = None
        for a in aliases:
            if a in low_keys and isinstance(low_keys[a], dict):
                found = low_keys[a]
                break
        assert found is not None, f"Missing event {aliases[0]} in {report_path}"

        # Check optimal setup bounds if present in event record
        opt_setup = found.get("optimal_setup") or found.get("setup_optimal") or {}
        if isinstance(opt_setup, dict) and opt_setup:
            for param, (lo, hi) in SETUP_BOUNDS.items():
                val = _extract_setup_param(opt_setup, param)
                if val is not None:
                    assert lo - 1e-4 <= val <= hi + 1e-4, (
                        f"Report {aliases[0]}: {param}={val} outside [{lo}, {hi}]"
                    )

    # Lock in upgraded 4WD dynamic event performance gates on FSG_RUBBERED (R25B)
    t_skid_opt = float(low_keys["skidpad"]["optimal_time_s"])
    t_accel_opt = float(low_keys["accel"]["optimal_time_s"])
    t_autocross_opt = float(low_keys["autocross"]["optimal_time_s"])
    assert t_skid_opt <= 4.64, f"Skidpad optimal {t_skid_opt:.4f} s > 4.64 s gate"
    assert t_accel_opt < 3.50, f"Accel 75m optimal {t_accel_opt:.4f} s >= 3.50 s gate"
    assert t_autocross_opt < 67.106, f"Autocross optimal {t_autocross_opt:.4f} s >= 67.106 s baseline"

    # Check 300 DPI comparison PNG exists and is valid
    png_path = VAULT_07_DESIGN_POINTS / "trackside_setup_event_comparison.png"
    assert png_path.exists(), f"Missing comparison PNG: {png_path}"
    assert png_path.stat().st_size > 20_000, f"Comparison PNG too small: {png_path.stat().st_size} B"
    with Image.open(png_path) as im:
        assert "dpi" in im.info, f"Missing DPI metadata in {png_path}"
        dpi_x, dpi_y = im.info["dpi"]
        assert abs(float(dpi_x) - 300.0) <= 1.5 and abs(float(dpi_y) - 300.0) <= 1.5


def test_pure_ascii_in_trackside_setup_and_simui_files() -> None:
    files_to_check = [
        REPO_ROOT / "vd_correlation_lab" / "analysis" / "benchmarks" / "run_trackside_setup_comparison.py",
        REPO_ROOT / "vd_correlation_lab" / "tests" / "integration" / "test_trackside_setup_lts.py",
        REPO_ROOT / "vd_correlation_lab" / "models" / "vehicle_14dof" / "ter27_physics.py",
        REPO_ROOT / "vd_correlation_lab" / "control" / "skidpad" / "skidpad_tv_controller.py",
        REPO_ROOT / "vd_correlation_lab" / "optimization" / "skidpad" / "optimize_skidpad_tv_casadi.py",
        REPO_ROOT / "vd_correlation_lab" / "optimization" / "lap_time_sim" / "solve_dynamic_events_casadi.py",
        REPO_ROOT / "vd_correlation_lab" / "optimization" / "lap_time_sim" / "solve_full_lap_ocp.py",
        MIRENA_SIM_DIR / "Scenes" / "Sim" / "UI" / "lts_menu_controller.gd",
        MIRENA_SIM_DIR / "Scenes" / "Sim" / "UI" / "TracksideTuningController.gd",
        REPO_ROOT / "tests" / "test_ter27_skidpad_tv.gd",
        REPO_ROOT / "tests" / "test_casadi_simui_integration.gd",
    ]
    for path in files_to_check:
        assert path.exists(), f"Missing file for ASCII check: {path}"
        text = path.read_text(encoding="utf-8")
        non_ascii = [(idx + 1, ch, ord(ch)) for idx, ch in enumerate(text) if ord(ch) > 127]
        assert len(non_ascii) == 0, f"Non-ASCII characters found in {path}: {non_ascii[:10]}"
