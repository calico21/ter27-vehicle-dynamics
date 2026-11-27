#!/usr/bin/env python3
"""
Phase 5 Quality Gate Pytest Suite:
  - 3D Double-Wishbone Loop Closure & CasADi B-Spline accuracy
  - TeR26 CAN Telemetry Sanity Gates & EKF+RTS kinematic residuals
  - C99 SIL Codegen accuracy & host latency bounds
  - Validation Vault (01..09) PNG existence, > 20 KB size, exact 300 DPI metadata, and JSON metrics
  - Master catalog, INDEX.md, and dashboard_index.html completeness
  - Strict pure-ASCII (zero emojis) enforcement across curated scripts, HTML, and docs
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
LAB_DIR = REPO_ROOT / "vd_correlation_lab"
RESULTS_TER26 = LAB_DIR / "results" / "ter26_extended_correlation"
VAULT_ROOT = REPO_ROOT / "outputs" / "validation_vault"

EXPECTED_VAULT_FOLDERS: List[str] = [
    "01_ter26_sim_to_real",
    "02_ter27_maneuvers_and_handling",
    "03_ter27_torque_vectoring_and_maps",
    "04_ter27_full_lap_and_endurance",
    "05_tires_and_thermal_analysis",
    "06_suspension_kinematics_and_fea",
    "07_design_tradeoffs_and_fs_points",
    "08_codegen_sil_and_godot_parity",
    "09_ter27_skidpad_deep_dive",
    "10_ter27_dil_and_sensor_realism",
]


def test_3d_double_wishbone_closure_and_bspline_accuracy() -> None:
    report_path = RESULTS_TER26 / "ter26_kinematics_3d_report.json"
    assert report_path.exists(), f"Missing {report_path}"
    data = json.loads(report_path.read_text(encoding="utf-8"))

    dom = data["domain_verification"]
    assert dom["closure_pass_lt_1e6_mm"] is True
    assert float(dom["max_residual_overall_mm"]) < 1e-6
    assert float(dom["max_residual_fl_mm"]) < 1e-6
    assert float(dom["max_residual_fr_mm"]) < 1e-6
    assert float(dom["max_residual_rl_mm"]) < 1e-6
    assert float(dom["max_residual_rr_mm"]) < 1e-6

    bsp = data["bspline_validation"]
    assert bsp["pass_angle_lt_1e4_deg"] is True
    assert bsp["pass_disp_lt_1e4_mm"] is True
    assert bsp["c2_hessian_finite"] is True
    assert float(bsp["max_angle_err_deg"]) < 1e-4
    assert float(bsp["max_disp_err_mm"]) < 1e-4


def test_ter26_telemetry_sanity_gates() -> None:
    audit_path = RESULTS_TER26 / "telemetry_audit_report.json"
    assert audit_path.exists(), f"Missing {audit_path}"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))

    sg = audit["sanity_gates"]
    assert sg["all_gates_passed"] is True
    res_ax = float(sg["gates"]["kinematic_residual_ax_rms"]["rms_ms2"])
    res_ay = float(sg["gates"]["kinematic_residual_ay_rms"]["rms_ms2"])
    assert res_ax < 0.35, f"res_ax RMS {res_ax} >= 0.35 m/s^2"
    assert res_ay < 0.35, f"res_ay RMS {res_ay} >= 0.35 m/s^2"
    assert sg["gates"]["no_nan_on_valid_segments"]["passed"] is True
    assert sg["gates"]["iso8855_wz_ay_positive"]["passed"] is True
    assert sg["gates"]["iso8855_delta_wz_positive"]["passed"] is True


def test_sil_c99_accuracy_and_latency() -> None:
    c_src = LAB_DIR / "codegen" / "ter27_casadi_eval.c"
    h_src = LAB_DIR / "codegen" / "ter27_casadi_eval.h"
    assert c_src.exists() and c_src.stat().st_size > 10_000, f"Missing/empty {c_src}"
    assert h_src.exists() and h_src.stat().st_size > 200, f"Missing/empty {h_src}"

    m_path = VAULT_ROOT / "08_codegen_sil_and_godot_parity" / "metrics.json"
    assert m_path.exists(), f"Missing {m_path}"
    data = json.loads(m_path.read_text(encoding="utf-8"))

    sil = data["c99_sil"]
    assert sil["accuracy"]["pass"] is True
    assert float(sil["accuracy"]["torque_max_abs_err"]) < 1e-6
    assert float(sil["accuracy"]["kappa_max_abs_err"]) < 1e-6

    lat_torque = sil["host_latency_us"]["ter27_eval_torque"]
    lat_kappa = sil["host_latency_us"]["ter27_eval_kappa"]
    assert float(lat_torque["mean_us"]) < 50.0
    assert float(lat_torque["p99_us"]) < 150.0
    assert float(lat_kappa["mean_us"]) < 50.0
    assert float(lat_kappa["p99_us"]) < 50.0
    assert data["overall_pass"] is True


@pytest.mark.parametrize("folder_name", EXPECTED_VAULT_FOLDERS)
def test_vault_folder_pngs_and_json_metrics(folder_name: str) -> None:
    folder = VAULT_ROOT / folder_name
    assert folder.is_dir(), f"Missing vault folder: {folder}"

    jsons = sorted(folder.glob("*.json"))
    assert len(jsons) >= 1, f"No JSON metrics file in {folder_name}"
    for jpath in jsons:
        parsed = json.loads(jpath.read_text(encoding="utf-8"))
        assert isinstance(parsed, (dict, list)) and len(parsed) > 0, f"Empty/invalid JSON: {jpath}"

    pngs = sorted(folder.glob("*.png"))
    assert len(pngs) >= 2, f"Expected >= 2 PNGs in {folder_name}, found {len(pngs)}"
    for png in pngs:
        size_bytes = png.stat().st_size
        assert size_bytes > 20 * 1024, f"{png} too small: {size_bytes} bytes (must be > 20 KB)"
        with Image.open(png) as im:
            assert "dpi" in im.info, f"Missing DPI metadata in {png}"
            dpi_x, dpi_y = im.info["dpi"]
            assert abs(float(dpi_x) - 300.0) <= 1.0, f"{png} dpi_x={dpi_x} != 300"
            assert abs(float(dpi_y) - 300.0) <= 1.0, f"{png} dpi_y={dpi_y} != 300"


def test_vault_01_expected_figures_and_metrics() -> None:
    v01 = VAULT_ROOT / "01_ter26_sim_to_real"
    required = [
        "metrics.json",
        "fig01_timeseries_oos.png",
        "fig02_bode_coherence.png",
        "fig03_understeer_gg.png",
        "fig04_regime_heatmap_residuals.png",
        "fig05_can_telemetry_audit.png",
    ]
    for name in required:
        assert (v01 / name).exists(), f"Missing required Vault 01 asset: {name}"


def test_vault_index_catalog_and_dashboard_reference_all_pngs() -> None:
    catalog_path = VAULT_ROOT / "master_metrics_catalog.json"
    index_md_path = VAULT_ROOT / "INDEX.md"
    dashboard_path = VAULT_ROOT / "dashboard_index.html"

    for p in (catalog_path, index_md_path, dashboard_path):
        assert p.exists() and p.stat().st_size > 1000, f"Missing or empty {p}"

    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    index_md = index_md_path.read_text(encoding="utf-8")
    dashboard_html = dashboard_path.read_text(encoding="utf-8")

    # Collect all PNGs across 01..08 on disk
    disk_pngs: List[str] = []
    for folder_name in EXPECTED_VAULT_FOLDERS:
        for png in sorted((VAULT_ROOT / folder_name).glob("*.png")):
            disk_pngs.append(f"{folder_name}/{png.name}")

    catalog_pngs = [
        fig["rel_path"]
        for suite in catalog["suites"]
        for fig in suite["figures"]
    ]
    assert catalog["summary"]["n_suites"] == len(EXPECTED_VAULT_FOLDERS)
    assert catalog["summary"]["total_pngs"] == len(disk_pngs)
    assert set(catalog_pngs) == set(disk_pngs)

    for rel_path in disk_pngs:
        assert rel_path in index_md, f"{rel_path} missing from INDEX.md"
        assert rel_path in dashboard_html, f"{rel_path} missing from dashboard_index.html"

    # Ensure zero external CDN dependencies in dashboard HTML
    assert "http://" not in dashboard_html
    assert "https://" not in dashboard_html


def test_pure_ascii_zero_emojis_in_curated_files() -> None:
    files_to_check = [
        LAB_DIR / "analysis" / "benchmarks" / "run_ter26_extended_correlation.py",
        LAB_DIR / "analysis" / "visualization" / "plot_style.py",
        LAB_DIR / "analysis" / "validation_vault" / "build_validation_vault_index.py",
        LAB_DIR / "tests" / "integration" / "test_validation_vault_and_physics.py",
        LAB_DIR / "tests" / "unit" / "test_skidpad_sub5_tv.py",
        LAB_DIR / "tests" / "unit" / "test_dil_sensor_realism.py",
        LAB_DIR / "control" / "skidpad" / "skidpad_tv_controller.py",
        LAB_DIR / "control" / "sensors_ekf" / "sensor_realism_profile.py",
        LAB_DIR / "analysis" / "benchmarks" / "run_dil_sensor_stress_suite.py",
        REPO_ROOT / "tests" / "test_ter27_skidpad_tv.gd",
        REPO_ROOT / "tests" / "test_ter27_dil_telemetry.gd",
        VAULT_ROOT / "INDEX.md",
        VAULT_ROOT / "dashboard_index.html",
        REPO_ROOT / "README.md",
        REPO_ROOT / "docs" / "architecture" / "architecture.md",
    ]
    for path in files_to_check:
        assert path.exists(), f"Missing file for ASCII check: {path}"
        text = path.read_text(encoding="utf-8")
        non_ascii = [(idx + 1, ch, ord(ch)) for idx, ch in enumerate(text) if ord(ch) > 127]
        assert len(non_ascii) == 0, f"Non-ASCII characters found in {path}: {non_ascii[:10]}"
