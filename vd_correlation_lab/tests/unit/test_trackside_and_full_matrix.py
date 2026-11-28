#!/usr/bin/env python3
"""
Expanded QA Verification Suite for:
  1. Trackside Race Engineering Engine (log_ingestor, <30s fast_adjoint_sysid, <15 min setup_advisor)
  2. FastAPI TestClient endpoints (/api/trackside/analyze, /api/whatif/simulate, /api/models/catalog)
  3. Full Scenario Matrix invariants across all 156 scenarios (0 HV, 0 BSPD, dMz/dbeta < 0, Tube-MPC >= 0.150 m)
  4. Google AI Studio Bundle completeness (exports/ai_studio_bundle/)
  5. Clean Publication Plots (8x 300-DPI pure-white #FFFFFF PNGs)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app
from api.models import (
    ModelsCatalogResponse,
    TracksideAnalyzeResponse,
    WhatIfSimulateResponse,
)
from vd_correlation_lab.trackside import (
    EXCLUDED_HARDWARE_SWAPS,
    advise_pit_box_setup,
    ingest_trace,
    remap_tire_temperatures_rl_fr,
    run_fast_adjoint_sysid,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
RELEASE_DIR = REPO_ROOT / "outputs" / "final_release_validation"
FIGURES_DIR = RELEASE_DIR / "figures"
BUNDLE_DIR = REPO_ROOT / "exports" / "ai_studio_bundle"

EXPECTED_FIGURES: List[str] = [
    "fig01_ter26_empirical_validation.png",
    "fig02_ter27_280kg_lap_times_and_attribution.png",
    "fig03_suspension_4x3_matrix_heatmap.png",
    "fig04_grip_and_weather_envelope.png",
    "fig05_tire_thermal_window_sensitivity.png",
    "fig06_manufacturing_mass_and_aero_sensitivity.png",
    "fig07_tube_mpc_robustness_all_faults.png",
    "fig08_endurance_22km_energy_and_thermal_budget.png",
]


def _assert_pure_ascii(path: Path) -> str:
    assert path.exists(), f"Missing file: {path}"
    text = path.read_text(encoding="utf-8")
    non_ascii = [(i + 1, ch, ord(ch)) for i, ch in enumerate(text) if ord(ch) > 127]
    assert len(non_ascii) == 0, f"Non-ASCII found in {path}: {non_ascii[:5]}"
    return text


def _build_synthetic_trace(n: int = 800, fs_hz: float = 100.0, invert_iso: bool = False) -> Dict[str, Any]:
    t = np.arange(n, dtype=np.float64) / fs_hz
    steer_deg = 12.0 * np.sin(2.0 * np.pi * 0.45 * t)
    vx_ms = 14.0 + 4.0 * np.sin(2.0 * np.pi * 0.25 * t)
    wz_rads = 0.045 * steer_deg
    ay_ms2 = vx_ms * wz_rads
    ax_ms2 = np.gradient(vx_ms, 1.0 / fs_hz)
    throttle = np.clip(0.55 + 0.35 * np.sin(2.0 * np.pi * 0.25 * t), 0.0, 1.0)
    brake = np.clip(-0.3 * np.sin(2.0 * np.pi * 0.25 * t), 0.0, 1.0)

    if invert_iso:
        ay_ms2 = -ay_ms2
        steer_deg = -steer_deg

    return {
        "t_s": t.tolist(),
        "vx_ms": vx_ms.tolist(),
        "ax_ms2": ax_ms2.tolist(),
        "ay_ms2": ay_ms2.tolist(),
        "wz_rads": wz_rads.tolist(),
        "steer_deg": steer_deg.tolist(),
        "throttle": throttle.tolist(),
        "brake": brake.tolist(),
        "t_tire_fl_c": np.full(n, 62.0).tolist(),
        "t_tire_fr_c": np.full(n, 78.0).tolist(),  # raw FR -> real RL
        "t_tire_rl_c": np.full(n, 68.0).tolist(),  # raw RL -> real FR
        "t_tire_rr_c": np.full(n, 71.0).tolist(),
    }


def test_trackside_log_ingestor_remap_and_iso8855() -> None:
    t_fl = np.array([60.0, 61.0])
    t_fr = np.array([80.0, 81.0])
    t_rl = np.array([65.0, 66.0])
    t_rr = np.array([70.0, 71.0])
    remapped = remap_tire_temperatures_rl_fr(t_fl, t_fr, t_rl, t_rr)
    assert np.allclose(remapped["fl"], t_fl)
    assert np.allclose(remapped["fr"], t_rl)
    assert np.allclose(remapped["rl"], t_fr)
    assert np.allclose(remapped["rr"], t_rr)

    # Verify ingest_trace applies RL <-> FR remap and flips inverted ISO 8855 signs
    trace = _build_synthetic_trace(n=600, fs_hz=100.0, invert_iso=True)
    run = ingest_trace(
        trace,
        vehicle_id="ter26_rwd",
        fs_hz=100.0,
        run_id="qa_iso_remap",
        apply_tire_temp_remap=True,
    )
    assert run.n_samples >= 590
    assert run.quality_stats["tire_temps_present"] is True
    assert run.quality_stats["tire_temp_remap_applied"] is True
    assert abs(float(np.mean(run.t_tire_fr_c)) - 68.0) < 1e-6
    assert abs(float(np.mean(run.t_tire_rl_c)) - 78.0) < 1e-6
    assert run.quality_stats["iso8855_ay_flipped"] is True
    assert run.quality_stats["iso8855_delta_flipped"] is True
    assert run.quality_stats["iso8855_passed"] is True
    assert float(run.quality_stats["iso8855_corr_wz_ay"]) > 0.5
    assert float(run.quality_stats["iso8855_corr_delta_wz"]) > 0.5


def test_trackside_fast_adjoint_sysid_and_setup_advisor() -> None:
    trace = _build_synthetic_trace(n=800, fs_hz=100.0, invert_iso=False)
    run = ingest_trace(trace, vehicle_id="ter27_4wd", fs_hz=100.0, run_id="qa_sysid")
    sysid = run_fast_adjoint_sysid(run, max_iters=3)
    assert sysid.within_30s_budget is True
    assert sysid.wall_time_s < 30.0
    assert sysid.lambda_mu_hat > 0.3

    advisor = advise_pit_box_setup(sysid, event="autocross")
    assert len(advisor.top_3_adjustments) == 3
    assert advisor.total_pit_time_min < 15.0
    assert advisor.within_15min_budget is True

    excluded_ids = {item["lever"] for item in EXCLUDED_HARDWARE_SWAPS}
    for adj in advisor.top_3_adjustments:
        assert adj.is_pit_box_legal is True
        assert adj.lever_id not in excluded_ids

    inv = advisor.fsae_ev_invariants
    assert inv["hv_dc_in_bounds"] is True
    assert float(inv["p_dc_min_kw"]) >= -25.0
    assert float(inv["p_dc_max_kw"]) <= 80.0
    assert int(inv["bspd_ev23_violations"]) == 0
    assert float(inv["min_clear_m"]) >= 0.150
    assert float(inv["dMz_dbeta_nm_rad"]) < 0.0
    assert inv["all_invariants_passed"] is True

    # Also verify persisted trackside_demo_report.json
    demo = json.loads(_assert_pure_ascii(RELEASE_DIR / "trackside_demo_report.json"))
    assert demo["all_within_30s_budget"] is True
    assert demo["all_invariants_passed"] is True


def test_new_fastapi_trackside_whatif_and_catalog_endpoints() -> None:
    trace = _build_synthetic_trace(n=600, fs_hz=100.0, invert_iso=False)
    with TestClient(app) as client:
        # 1. POST /api/trackside/analyze
        r_ts = client.post(
            "/api/trackside/analyze",
            json={
                "trace": trace,
                "vehicle_id": "ter27_4wd",
                "event": "autocross",
                "fs_hz": 100.0,
                "max_iters": 3,
            },
        )
        assert r_ts.status_code == 200, r_ts.text
        ts_res = TracksideAnalyzeResponse.model_validate(r_ts.json())
        assert ts_res.within_30s_budget is True
        assert ts_res.within_15min_budget is True
        assert ts_res.total_pit_time_min < 15.0
        assert len(ts_res.top_3_adjustments) == 3
        assert ts_res.fsae_ev_invariants["all_invariants_passed"] is True

        # 2. POST /api/whatif/simulate (cached matrix + CasADi 14-DOF fallback)
        r_wi_cached = client.post(
            "/api/whatif/simulate",
            json={
                "event": "autocross",
                "mass_kg": 280.0,
                "grip_lambda_mu": 0.57,
                "tire_temp_c": 75.0,
                "aero_cla_m2": 4.68,
                "suspension_mode": "3d_double_wishbone_lut",
                "model_id": "differentiable_casadi_14dof",
                "tube_mpc_enabled": True,
                "fault_mode": "nominal",
                "use_cached_matrix": True,
            },
        )
        assert r_wi_cached.status_code == 200, r_wi_cached.text
        wi_cached = WhatIfSimulateResponse.model_validate(r_wi_cached.json())
        assert wi_cached.invariants_passed is True
        assert wi_cached.bspd_violations == 0
        assert wi_cached.min_clear_m >= 0.150
        assert wi_cached.yaw_stable is True

        r_wi_eval = client.post(
            "/api/whatif/simulate",
            json={
                "event": "skidpad",
                "mass_kg": 282.5,
                "grip_lambda_mu": 0.57,
                "tire_temp_c": 65.0,
                "aero_cla_m2": 4.68,
                "suspension_mode": "3d_double_wishbone_lut",
                "model_id": "differentiable_casadi_14dof",
                "tube_mpc_enabled": True,
                "fault_mode": "nominal",
                "use_cached_matrix": False,
            },
        )
        assert r_wi_eval.status_code == 200, r_wi_eval.text
        wi_eval = WhatIfSimulateResponse.model_validate(r_wi_eval.json())
        assert wi_eval.source == "casadi_14dof_adjoint_eval"
        assert wi_eval.invariants_passed is True

        # 3. GET /api/models/catalog
        r_cat = client.get("/api/models/catalog")
        assert r_cat.status_code == 200, r_cat.text
        cat = ModelsCatalogResponse.model_validate(r_cat.json())
        assert cat.total_combinations == 12
        assert len(cat.vehicle_models) == 3
        assert len(cat.suspension_modes) == 4
        assert len(cat.combinations) == 12
        assert abs(cat.constraints.cold_test_oct_lambda_muy - 0.47) < 1e-9
        assert abs(cat.constraints.fsg_rubbered_lambda_muy - 0.57) < 1e-9
        assert abs(cat.constraints.ter27_nominal_mass_kg - 280.0) < 1e-6


def test_full_scenario_matrix_invariants() -> None:
    raw = _assert_pure_ascii(RELEASE_DIR / "all_scenarios_metrics.json")
    data = json.loads(raw)

    assert data["schema_version"] == "ter27_280kg_full_scenario_matrix_v1"
    assert abs(float(data["nominal_mass_kg"]) - 280.0) < 1e-6
    assert abs(float(data["guardrails"]["cold_test_oct_lambda_muy"]) - 0.47) < 1e-9
    assert abs(float(data["guardrails"]["fsg_rubbered_lambda_muy"]) - 0.57) < 1e-9

    summary = data["summary"]
    scenarios = data["scenarios"]
    assert int(summary["total_scenarios"]) == len(scenarios) == 156
    assert int(summary["total_hv_violations"]) == 0
    assert int(summary["total_bspd_violations"]) == 0
    assert int(summary["total_yaw_unstable"]) == 0
    assert int(summary["tube_mpc_unsafe_count"]) == 0
    assert summary["all_global_invariants_passed"] is True

    tube_mpc_count = 0
    for sc in scenarios:
        assert int(sc["hv_violations"]) == 0, f"HV violation in {sc['scenario_id']}"
        assert float(sc["p_dc_min_kw"]) >= -25.0 - 1e-6, f"Regen violation in {sc['scenario_id']}"
        assert float(sc["p_dc_max_kw"]) <= 80.0 + 1e-6, f"Power cap violation in {sc['scenario_id']}"
        assert int(sc["bspd_violations"]) == 0, f"BSPD violation in {sc['scenario_id']}"
        assert float(sc["dMz_dbeta"]) < 0.0, f"Yaw instability in {sc['scenario_id']}"

        if "tube_mpc" in str(sc.get("controller_mode", "")) or "tube_mpc" in str(sc["scenario_id"]):
            tube_mpc_count += 1
            assert int(sc["cones_hit"]) == 0, f"Cone hit under Tube-MPC in {sc['scenario_id']}"
            assert float(sc["min_clear_m"]) >= 0.150, f"Clearance < 0.150 m in {sc['scenario_id']}"
            assert sc["safe_pass"] is True, f"Tube-MPC safe_pass False in {sc['scenario_id']}"

    assert tube_mpc_count == int(summary["tube_mpc_scenarios"]) == 24

    csv_path = RELEASE_DIR / "all_scenarios_summary.csv"
    _assert_pure_ascii(csv_path)
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 156


def test_ai_studio_bundle_completeness() -> None:
    openapi_text = _assert_pure_ascii(BUNDLE_DIR / "openapi_schema.json")
    master_text = _assert_pure_ascii(BUNDLE_DIR / "master_dataset.json")
    spec_text = _assert_pure_ascii(BUNDLE_DIR / "AI_STUDIO_SYSTEM_SPEC.md")

    openapi_disk = json.loads(openapi_text)
    live_openapi = app.openapi()
    assert set(openapi_disk["paths"].keys()) == set(live_openapi["paths"].keys())
    for ep in ("/api/trackside/analyze", "/api/whatif/simulate", "/api/models/catalog"):
        assert ep in openapi_disk["paths"], f"Missing {ep} in bundle openapi_schema.json"

    master = json.loads(master_text)
    for key in (
        "schema_version",
        "vehicle_id",
        "nominal_mass_kg",
        "locked_grip_profiles",
        "models_catalog",
        "scenario_matrix",
        "ter27_280kg_limit_study",
        "ter26_empirical_validation",
        "adjoint_sensitivities",
        "trackside_demo_recommendations",
        "time_series_curves",
    ):
        assert key in master, f"Missing key {key} in master_dataset.json"
    assert master["vehicle_id"] == "ter27_4wd"
    assert abs(float(master["nominal_mass_kg"]) - 280.0) < 1e-6

    assert len(spec_text) > 1000
    for token in ("/api/trackside/analyze", "/api/whatif/simulate", "/api/models/catalog", "280.0 kg"):
        assert token in spec_text, f"Missing '{token}' in AI_STUDIO_SYSTEM_SPEC.md"


def test_clean_publication_plots_300dpi_pure_white() -> None:
    for fname in EXPECTED_FIGURES:
        fpath = FIGURES_DIR / fname
        assert fpath.exists(), f"Missing publication plot: {fpath}"
        assert fpath.stat().st_size > 30_000, f"Plot too small: {fpath}"
        with Image.open(fpath) as im:
            dpi = im.info.get("dpi", (0, 0))
            assert float(dpi[0]) >= 299.0 and float(dpi[1]) >= 299.0, (
                f"Plot {fname} DPI={dpi} < 300"
            )
            rgb = im.convert("RGB")
            w, h = rgb.size
            corners = [
                rgb.getpixel((0, 0)),
                rgb.getpixel((w - 1, 0)),
                rgb.getpixel((0, h - 1)),
                rgb.getpixel((w - 1, h - 1)),
            ]
            for idx, px in enumerate(corners):
                assert px == (255, 255, 255), (
                    f"Plot {fname} corner[{idx}]={px} is not pure white (255, 255, 255)"
                )
