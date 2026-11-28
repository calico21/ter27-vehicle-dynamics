#!/usr/bin/env python3
"""
Automated Verification Suite for TeR27 Vehicle Dynamics & Telemetry API,
Frontend Google AI Studio Workspace, and Repository Integrity.

Covers:
  1. New hierarchical layout: vd_correlation_lab/, api/, data/, outputs/,
     docs/, frontend/ (and removal of _archive_legacy/, tools/, docs/api_contract/).
  2. All FastAPI REST endpoints (200 OK + Pydantic v2 validation) and
     WS /ws/telemetry live/fallback frame delivery.
  3. frontend/contract/ (openapi.json, api_types.ts, endpoints_reference.md)
     and all 5 frontend/mock_data/*.json synchronized 100% with api/.
  4. Zero corny names (TeR-Nexus, Nexus, Oracle, case-insensitive) in api/,
     frontend/, README.md, docs/architecture/REPO_MAP.md, ter27.py, and
     TelemetryUdpStreamer.gd.
  5. Zero modifications in src/ outside src/mirena_sim/ (via git status).
  6. Zero non-ASCII / zero emojis across all curated/new files.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Type

from fastapi.testclient import TestClient
from pydantic import BaseModel

from api.main import app
from api.models import (
    ApplySetupResponse,
    HealthResponse,
    SessionABComparisonResponse,
    SessionsListResponse,
    SetupsMatrixResponse,
    SimulationJobStatusResponse,
    TelemetryFrame,
    TireEvaluateResponse,
    VaultCatalogResponse,
    VehicleOverviewResponse,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
FRONTEND_DIR = REPO_ROOT / "frontend"
CONTRACT_DIR = FRONTEND_DIR / "contract"
MOCK_DATA_DIR = FRONTEND_DIR / "mock_data"
VAULT_ROOT = REPO_ROOT / "outputs" / "validation_vault"
SETUPS_DIR = (
    REPO_ROOT
    / "src"
    / "mirena_sim"
    / "Scenes"
    / "Vehicle"
    / "Config"
    / "vehicles"
    / "ter27_4wd"
    / "setups"
)

EXPECTED_MOCK_FILES: Dict[str, Type[BaseModel]] = {
    "live_telemetry_frame.json": TelemetryFrame,
    "dil_session_comparison.json": SessionABComparisonResponse,
    "tire_evaluation_curves.json": TireEvaluateResponse,
    "trackside_setups_matrix.json": SetupsMatrixResponse,
    "validation_vault_catalog.json": VaultCatalogResponse,
}

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
    "11_differentiable_casadi_and_tubes",
]

EXPECTED_OPTIMAL_PRESETS: List[str] = [
    "optimal_skidpad.json",
    "optimal_accel.json",
    "optimal_autocross.json",
    "optimal_endurance.json",
]

FORBIDDEN_NAME_PATTERN = re.compile(r"(ter-nexus|nexus|oracle)", re.IGNORECASE)


def _assert_pure_ascii(path: Path) -> None:
    assert path.exists(), f"Missing file for ASCII check: {path}"
    text = path.read_text(encoding="utf-8")
    non_ascii = [(idx + 1, ch, ord(ch)) for idx, ch in enumerate(text) if ord(ch) > 127]
    assert len(non_ascii) == 0, f"Non-ASCII found in {path}: {non_ascii[:10]}"


def _assert_no_forbidden_names(path: Path) -> None:
    assert path.exists(), f"Missing file for naming check: {path}"
    text = path.read_text(encoding="utf-8")
    if path.name == "validation_vault_catalog.json":
        # Untouched historical artifact filename inside outputs/validation_vault/03_*
        text = text.replace(
            "03_ter27_torque_vectoring_and_maps/skidpad_oracle_report.json", ""
        )
    matches = FORBIDDEN_NAME_PATTERN.findall(text)
    assert len(matches) == 0, f"Forbidden name(s) {set(matches)} found in {path}"


def test_fastapi_rest_endpoints_and_schemas() -> None:
    with TestClient(app) as client:
        # 1. GET /api/health
        r_health = client.get("/api/health")
        assert r_health.status_code == 200
        health = HealthResponse.model_validate(r_health.json())
        assert health.status == "ok"
        assert abs(health.grip_profiles["COLD_TEST_OCT"] - 0.47) < 1e-6
        assert abs(health.grip_profiles["FSG_RUBBERED"] - 0.57) < 1e-6

        # 2. GET /api/vehicle/overview
        r_veh = client.get("/api/vehicle/overview")
        assert r_veh.status_code == 200
        veh = VehicleOverviewResponse.model_validate(r_veh.json())
        assert veh.vehicle_id == "ter27_4wd"
        assert abs(veh.validation_status.cold_test_oct_lambda_muy - 0.47) < 1e-6
        assert abs(veh.validation_status.fsg_rubbered_lambda_muy - 0.57) < 1e-6

        # 3. GET /api/sessions & GET /api/sessions/{session_id}
        r_sessions = client.get("/api/sessions")
        assert r_sessions.status_code == 200
        sessions = SessionsListResponse.model_validate(r_sessions.json())
        assert sessions.total_count >= 1
        sid = sessions.sessions[0].session_id

        r_ab = client.get(f"/api/sessions/{sid}")
        assert r_ab.status_code == 200
        ab = SessionABComparisonResponse.model_validate(r_ab.json())
        assert ab.session.session_id == sid
        assert len(ab.driver_trace) > 0
        assert len(ab.casadi_reference_trace) > 0

        # 4. POST /api/tires/evaluate
        r_tires = client.post(
            "/api/tires/evaluate",
            json={
                "compound": "R25B",
                "grip_profile": "FSG_RUBBERED",
                "fz_n": 750.0,
                "camber_deg": -1.5,
                "temp_c": 75.0,
                "pressure_kpa": 83.0,
                "n_points": 41,
            },
        )
        assert r_tires.status_code == 200
        tires = TireEvaluateResponse.model_validate(r_tires.json())
        assert tires.peak_fy_n > 500.0
        assert len(tires.fy_vs_alpha) == 41
        assert len(tires.fx_vs_kappa) == 41
        assert len(tires.friction_ellipse) >= 16

        # 5. GET /api/setups & POST /api/setups/apply
        r_setups = client.get("/api/setups")
        assert r_setups.status_code == 200
        setups = SetupsMatrixResponse.model_validate(r_setups.json())
        assert len(setups.presets) >= 4
        assert len(setups.optimization_gains) >= 4

        r_apply = client.post(
            "/api/setups/apply",
            json={"preset": "optimal_skidpad", "sensor_profile": "SENSOR_TRACK_REALISM"},
        )
        assert r_apply.status_code == 200
        applied = ApplySetupResponse.model_validate(r_apply.json())
        assert applied.acknowledged is True
        assert applied.target_port == 9877

        # 6. POST /api/simulation/run (dry_run) & GET /api/simulation/status/{job_id}
        r_sim = client.post(
            "/api/simulation/run",
            json={
                "event": "skidpad",
                "grip_profile": "FSG_RUBBERED",
                "compound": "R25B",
                "optimize_setup": True,
                "dry_run": True,
            },
        )
        assert r_sim.status_code == 200
        sim_job = SimulationJobStatusResponse.model_validate(r_sim.json())
        assert sim_job.status == "completed"

        r_status = client.get(f"/api/simulation/status/{sim_job.job_id}")
        assert r_status.status_code == 200
        sim_status = SimulationJobStatusResponse.model_validate(r_status.json())
        assert sim_status.job_id == sim_job.job_id
        assert sim_status.progress_pct == 100.0

        # 7. GET /api/vault/catalog
        r_vault = client.get("/api/vault/catalog")
        assert r_vault.status_code == 200
        vault = VaultCatalogResponse.model_validate(r_vault.json())
        assert vault.n_suites == 11
        assert vault.total_pngs >= 65


def test_websocket_telemetry_stream() -> None:
    with TestClient(app) as client:
        with client.websocket_connect("/ws/telemetry") as ws:
            payload = ws.receive_json()
            frame = TelemetryFrame.model_validate(payload)
            assert frame.source in ("godot_udp", "dil_session", "synthetic_fallback")
            assert frame.vehicle_id == "ter27_4wd"
            assert frame.wheels.fl.fz_n >= 0.0
            assert 0.0 <= frame.hv.soc_pct <= 100.0


def test_frontend_contract_and_mock_data_sync() -> None:
    ts_path = CONTRACT_DIR / "api_types.ts"
    openapi_path = CONTRACT_DIR / "openapi.json"
    ref_path = CONTRACT_DIR / "endpoints_reference.md"
    instructions_path = FRONTEND_DIR / "AI_STUDIO_INSTRUCTIONS.md"

    for p in (ts_path, openapi_path, ref_path, instructions_path):
        assert p.exists() and p.stat().st_size > 400, f"Missing or empty frontend doc/contract: {p}"
        _assert_pure_ascii(p)

    # OpenAPI path parity with live app.openapi()
    disk_spec = json.loads(openapi_path.read_text(encoding="utf-8"))
    live_spec = app.openapi()
    assert set(disk_spec["paths"].keys()) == set(live_spec["paths"].keys())
    assert disk_spec["info"]["title"] == live_spec["info"]["title"]

    # TypeScript contract must declare all core request/response interfaces
    ts_text = ts_path.read_text(encoding="utf-8")
    for iface in (
        "HealthResponse",
        "TelemetryFrame",
        "VehicleOverviewResponse",
        "SessionsListResponse",
        "SessionABComparisonResponse",
        "TireEvaluateRequest",
        "TireEvaluateResponse",
        "SetupsMatrixResponse",
        "ApplySetupResponse",
        "SimulationRunRequest",
        "SimulationJobStatusResponse",
        "VaultCatalogResponse",
    ):
        assert f"export interface {iface}" in ts_text, f"Missing interface {iface} in {ts_path}"

    # Verify all 5 mock_data JSON files exist, are pure ASCII, and match Pydantic v2 schemas
    for fname, model_cls in EXPECTED_MOCK_FILES.items():
        spath = MOCK_DATA_DIR / fname
        assert spath.exists() and spath.stat().st_size > 100, f"Missing or empty mock file: {spath}"
        _assert_pure_ascii(spath)
        parsed = json.loads(spath.read_text(encoding="utf-8"))
        model_cls.model_validate(parsed)

    # Legacy duplicate docs/api_contract/ must be removed
    assert not (REPO_ROOT / "docs" / "api_contract").exists(), "Legacy docs/api_contract/ still exists"


def test_repository_hierarchy_naming_and_src_integrity() -> None:
    # 1. Verify new hierarchical folders exist and legacy root clutter is gone
    required_dirs = [
        REPO_ROOT / "vd_correlation_lab" / "models" / "vehicle_14dof",
        REPO_ROOT / "vd_correlation_lab" / "models" / "tire_mf62_pinn",
        REPO_ROOT / "vd_correlation_lab" / "control" / "skidpad",
        REPO_ROOT / "vd_correlation_lab" / "optimization" / "lap_time_sim",
        REPO_ROOT / "vd_correlation_lab" / "analysis" / "validation_vault",
        REPO_ROOT / "vd_correlation_lab" / "tests" / "api",
        REPO_ROOT / "api" / "core",
        REPO_ROOT / "api" / "models",
        REPO_ROOT / "api" / "services",
        REPO_ROOT / "api" / "routers",
        REPO_ROOT / "data" / "ttc_tire",
        REPO_ROOT / "data" / "telemetry_ter26",
        REPO_ROOT / "data" / "tracks",
        REPO_ROOT / "outputs" / "validation_vault",
        REPO_ROOT / "docs" / "architecture",
        FRONTEND_DIR / "contract",
        FRONTEND_DIR / "mock_data",
        FRONTEND_DIR / "src",
    ]
    for d in required_dirs:
        assert d.is_dir(), f"Missing required directory: {d}"

    assert not (REPO_ROOT / "_archive_legacy").exists(), "_archive_legacy/ must be removed"
    assert not (REPO_ROOT / "tools").exists(), "tools/ must be consolidated and removed"

    required_files = [
        REPO_ROOT / "ter27.py",
        REPO_ROOT / "README.md",
        REPO_ROOT / "docs" / "architecture" / "REPO_MAP.md",
        REPO_ROOT / "src" / "mirena_sim" / "Scenes" / "Sim" / "TelemetryUdpStreamer.gd",
        REPO_ROOT / "src" / "mirena_sim" / "Scenes" / "Sim" / "sim.gd",
        VAULT_ROOT / "master_metrics_catalog.json",
    ]
    for path in required_files:
        assert path.exists(), f"Missing required repo file: {path}"

    for folder in EXPECTED_VAULT_FOLDERS:
        assert (VAULT_ROOT / folder).is_dir(), f"Missing validation vault folder: {folder}"

    for preset in EXPECTED_OPTIMAL_PRESETS:
        ppath = SETUPS_DIR / preset
        assert ppath.exists(), f"Missing optimal setup preset: {ppath}"
        data = json.loads(ppath.read_text(encoding="utf-8"))
        assert isinstance(data, dict) and len(data) > 0

    # 2. Verify ter27.py api --help works cleanly
    proc = subprocess.run(
        [sys.executable, "-u", str(REPO_ROOT / "ter27.py"), "api", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"ter27.py api --help failed: {proc.stderr}"
    assert "--port" in proc.stdout

    # 3. Zero corny names & pure ASCII across api/, frontend/, README.md, REPO_MAP.md, ter27.py, TelemetryUdpStreamer.gd
    curated_targets: List[Path] = [
        REPO_ROOT / "ter27.py",
        REPO_ROOT / "README.md",
        REPO_ROOT / "docs" / "architecture" / "REPO_MAP.md",
        REPO_ROOT / "src" / "mirena_sim" / "Scenes" / "Sim" / "TelemetryUdpStreamer.gd",
    ]
    curated_targets.extend(sorted((REPO_ROOT / "api").rglob("*.py")))
    curated_targets.extend(
        sorted(
            p
            for p in FRONTEND_DIR.rglob("*")
            if p.is_file()
            and "node_modules" not in p.parts
            and "dist" not in p.parts
            and p.suffix in {".ts", ".tsx", ".css", ".html", ".json", ".md"}
        )
    )

    for path in curated_targets:
        _assert_pure_ascii(path)
        _assert_no_forbidden_names(path)

    # Also enforce pure ASCII on this test file itself
    _assert_pure_ascii(Path(__file__).resolve())

    # 4. Verify zero tracked source modifications in src/ outside src/mirena_sim/
    git_proc = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "diff", "--name-only", "--", "src/"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert git_proc.returncode == 0, f"git diff failed: {git_proc.stderr}"
    for rel_path in git_proc.stdout.splitlines():
        rel_path = rel_path.strip()
        if not rel_path or "__pycache__" in rel_path or rel_path.endswith(".pyc"):
            continue
        assert rel_path.startswith("src/mirena_sim/"), (
            f"Unauthorized tracked modification inside src/ outside src/mirena_sim/: {rel_path}"
        )
