"""Centralized configuration and filesystem paths for the TeR27 API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
PYTHON_BIN: str = "/home/alexr/project_gp_env/bin/python3"

API_TITLE: str = "TeR27 Vehicle Dynamics & Telemetry API"
API_DESCRIPTION: str = (
    "Real-time 60 Hz Godot MirenaSim UDP bridge, CasADi simulation runner, "
    "Pacejka MF 6.2 tire evaluator, and Validation Vault API for Tecnun e-Racing TeR27 4WD."
)

UDP_TELEMETRY_HOST: str = "127.0.0.1"
UDP_TELEMETRY_PORT: int = 9876
UDP_COMMAND_HOST: str = "127.0.0.1"
UDP_COMMAND_PORT: int = 9877
UDP_LIVE_TIMEOUT_S: float = 1.5
WS_STREAM_HZ: float = 60.0

VEHICLE_CONFIG_DIR: Path = (
    REPO_ROOT
    / "src"
    / "mirena_sim"
    / "Scenes"
    / "Vehicle"
    / "Config"
    / "vehicles"
    / "ter27_4wd"
)
SETUPS_DIR: Path = VEHICLE_CONFIG_DIR / "setups"
DIL_SESSIONS_DIR: Path = REPO_ROOT / "outputs" / "dil_sessions"
VALIDATION_VAULT_DIR: Path = REPO_ROOT / "outputs" / "validation_vault"
VAULT_CATALOG_PATH: Path = VALIDATION_VAULT_DIR / "master_metrics_catalog.json"
SKIDPAD_PARITY_PATH: Path = (
    VALIDATION_VAULT_DIR / "09_ter27_skidpad_deep_dive" / "godot_skidpad_parity_trace.json"
)
SETUP_REPORT_PATH: Path = (
    VALIDATION_VAULT_DIR
    / "07_design_tradeoffs_and_fs_points"
    / "trackside_setup_optimization_report.json"
)

# Updated to final vd_correlation_lab/optimization/ paths once @arch-reorg finishes.
SOLVE_DYNAMIC_SCRIPT: Path = (
    REPO_ROOT
    / "vd_correlation_lab"
    / "optimization"
    / "lap_time_sim"
    / "solve_dynamic_events_casadi.py"
)
SOLVE_FULL_LAP_SCRIPT: Path = (
    REPO_ROOT
    / "vd_correlation_lab"
    / "optimization"
    / "lap_time_sim"
    / "solve_full_lap_ocp.py"
)


@dataclass(frozen=True)
class ApiSettings:
    title: str = API_TITLE
    description: str = API_DESCRIPTION
    cors_allow_origins: Tuple[str, ...] = ("*",)
    udp_telemetry_host: str = UDP_TELEMETRY_HOST
    udp_telemetry_port: int = UDP_TELEMETRY_PORT
    udp_command_host: str = UDP_COMMAND_HOST
    udp_command_port: int = UDP_COMMAND_PORT
    ws_stream_hz: float = WS_STREAM_HZ


settings = ApiSettings()
