#!/usr/bin/env python3
"""
Centralized repository path configuration and executable discovery for
Tecnun eRacing Vehicle Dynamics (`vd_correlation_lab`).

All paths are derived dynamically from `Path(__file__).resolve()` so that
every script, benchmark, test, and Godot probe works regardless of working
directory or host machine layout.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Core repository roots
# ---------------------------------------------------------------------------
CONFIG_DIR: Path = Path(__file__).resolve().parent
VD_LAB_DIR: Path = CONFIG_DIR.parent
REPO_ROOT: Path = VD_LAB_DIR.parent

# ---------------------------------------------------------------------------
# Data directories
# ---------------------------------------------------------------------------
DATA_DIR: Path = REPO_ROOT / "data"
TTC_TIRE_DIR: Path = DATA_DIR / "ttc_tire"
TELEMETRY_TER26_DIR: Path = DATA_DIR / "telemetry_ter26"
TRACKS_DATA_DIR: Path = DATA_DIR / "tracks"

RAW_BAGS_DIR: Path = TELEMETRY_TER26_DIR / "raw_bags"
KINEMATICS_SOURCE_DIR: Path = TELEMETRY_TER26_DIR / "kinematics_source"
PROCESSED_DIR: Path = TELEMETRY_TER26_DIR / "processed"

TER26_GROUND_TRUTH_PARQUET: Path = PROCESSED_DIR / "ter26_clean_ground_truth.parquet"
TER26_GROUND_TRUTH_CSV: Path = PROCESSED_DIR / "ter26_clean_ground_truth.csv"

# ---------------------------------------------------------------------------
# MirenaSim (Godot) & vehicle configuration directories
# ---------------------------------------------------------------------------
MIRENA_SIM_DIR: Path = REPO_ROOT / "src" / "mirena_sim"
VEHICLES_CONFIG_DIR: Path = (
    MIRENA_SIM_DIR / "Scenes" / "Vehicle" / "Config" / "vehicles"
)
TER26_CONFIG_DIR: Path = VEHICLES_CONFIG_DIR / "ter26_rwd"
TER27_CONFIG_DIR: Path = VEHICLES_CONFIG_DIR / "ter27_4wd"
MF62_DIR: Path = MIRENA_SIM_DIR / "Scenes" / "Vehicle" / "Dynamics" / "Tires" / "MF62"
TESTS_DIR: Path = REPO_ROOT / "tests"

# ---------------------------------------------------------------------------
# vd_correlation_lab domain packages
# ---------------------------------------------------------------------------
MODELS_DIR: Path = VD_LAB_DIR / "models"
VEHICLE_14DOF_DIR: Path = MODELS_DIR / "vehicle_14dof"
TIRE_MF62_PINN_DIR: Path = MODELS_DIR / "tire_mf62_pinn"
KINEMATICS_DIR: Path = MODELS_DIR / "kinematics"

CONTROL_DIR: Path = VD_LAB_DIR / "control"
TORQUE_VECTORING_DIR: Path = CONTROL_DIR / "torque_vectoring"
SKIDPAD_CONTROL_DIR: Path = CONTROL_DIR / "skidpad"
SENSORS_EKF_DIR: Path = CONTROL_DIR / "sensors_ekf"
HUMAN_CONTROLS_DIR: Path = CONTROL_DIR / "human_controls"

OPTIMIZATION_DIR: Path = VD_LAB_DIR / "optimization"
LAP_TIME_SIM_DIR: Path = OPTIMIZATION_DIR / "lap_time_sim"
SKIDPAD_OPT_DIR: Path = OPTIMIZATION_DIR / "skidpad"
TRACKSIDE_SETUP_DIR: Path = OPTIMIZATION_DIR / "trackside_setup"

ANALYSIS_DIR: Path = VD_LAB_DIR / "analysis"
BENCHMARKS_DIR: Path = ANALYSIS_DIR / "benchmarks"
VALIDATION_VAULT_TOOLS_DIR: Path = ANALYSIS_DIR / "validation_vault"
VISUALIZATION_DIR: Path = ANALYSIS_DIR / "visualization"

CODEGEN_DIR: Path = VD_LAB_DIR / "codegen"
LAB_TESTS_DIR: Path = VD_LAB_DIR / "tests"
DBC_PATH: Path = VD_LAB_DIR / "dbc" / "TER.dbc"
VEHICLES_LAB_DIR: Path = VD_LAB_DIR / "vehicles"

# Aliases for compatibility with internal scripts
CORE_DIR: Path = VEHICLE_14DOF_DIR
TELEMETRY_DIR: Path = SENSORS_EKF_DIR
CORRELATION_DIR: Path = BENCHMARKS_DIR
VIZ_DIR: Path = VISUALIZATION_DIR
LTS_DIR: Path = LAP_TIME_SIM_DIR
YMD_DIR: Path = BENCHMARKS_DIR

RESULTS_DIR: Path = VD_LAB_DIR / "results"
CASADI_SUITE_DIR: Path = RESULTS_DIR / "casadi_suite"
TER26_CORR_DIR: Path = RESULTS_DIR / "ter26_extended_correlation"
LTS_RESULTS_DIR: Path = RESULTS_DIR / "lts"
YMD_RESULTS_DIR: Path = RESULTS_DIR / "ymd"
TV_OPT_RESULTS_DIR: Path = RESULTS_DIR / "tv_ocp"

# ---------------------------------------------------------------------------
# Outputs & Validation Vault (10 structured output folders)
# ---------------------------------------------------------------------------
OUTPUTS_DIR: Path = REPO_ROOT / "outputs"
DIL_SESSIONS_DIR: Path = OUTPUTS_DIR / "dil_sessions"
CASADI_CACHE_DIR: Path = OUTPUTS_DIR / "casadi_cache"
VAULT_DIR: Path = OUTPUTS_DIR / "validation_vault"
VALIDATION_VAULT_ROOT: Path = VAULT_DIR

VAULT_01_TER26: Path = VAULT_DIR / "01_ter26_sim_to_real"
VAULT_02_MANEUVERS: Path = VAULT_DIR / "02_ter27_maneuvers_and_handling"
VAULT_03_TV_MAPS: Path = VAULT_DIR / "03_ter27_torque_vectoring_and_maps"
VAULT_04_FULL_LAP: Path = VAULT_DIR / "04_ter27_full_lap_and_endurance"
VAULT_05_TIRES_THERMAL: Path = VAULT_DIR / "05_tires_and_thermal_analysis"
VAULT_06_KINEMATICS_FEA: Path = VAULT_DIR / "06_suspension_kinematics_and_fea"
VAULT_07_DESIGN_POINTS: Path = VAULT_DIR / "07_design_tradeoffs_and_fs_points"
VAULT_08_CODEGEN_PARITY: Path = VAULT_DIR / "08_codegen_sil_and_godot_parity"
VAULT_09_SKIDPAD_DEEP_DIVE: Path = VAULT_DIR / "09_ter27_skidpad_deep_dive"
VAULT_10_DIL_SENSOR_REALISM: Path = VAULT_DIR / "10_ter27_dil_and_sensor_realism"

VAULT_DIRS: Dict[str, Path] = {
    "01": VAULT_01_TER26,
    "02": VAULT_02_MANEUVERS,
    "03": VAULT_03_TV_MAPS,
    "04": VAULT_04_FULL_LAP,
    "05": VAULT_05_TIRES_THERMAL,
    "06": VAULT_06_KINEMATICS_FEA,
    "07": VAULT_07_DESIGN_POINTS,
    "08": VAULT_08_CODEGEN_PARITY,
    "09": VAULT_09_SKIDPAD_DEEP_DIVE,
    "10": VAULT_10_DIL_SENSOR_REALISM,
}


# ---------------------------------------------------------------------------
# Executable discovery (Godot 4 & Python venv)
# ---------------------------------------------------------------------------
def discover_godot_bin() -> Path:
    """Locate the Godot 4 headless executable."""
    env_bin = os.environ.get("GODOT_BIN", "").strip()
    if env_bin:
        p = Path(env_bin).expanduser()
        if p.exists():
            return p

    which_bin = shutil.which("godot") or shutil.which("godot4")
    if which_bin:
        return Path(which_bin)

    candidates = [
        Path.home() / ".gemini" / "antigravity" / "bin" / "godot",
        Path("/usr/local/bin/godot"),
        Path("/usr/bin/godot"),
        Path("/usr/bin/godot4"),
    ]
    for cand in candidates:
        if cand.exists():
            return cand

    return candidates[0]


def discover_python_bin() -> Path:
    """Locate the project_gp_env Python 3 interpreter (with CasADi/SciPy/Matplotlib)."""
    env_bin = os.environ.get("PROJECT_GP_PYTHON", "").strip()
    if env_bin:
        p = Path(env_bin).expanduser()
        if p.exists():
            return p

    candidates = [
        Path.home() / "project_gp_env" / "bin" / "python3",
        Path(sys.executable),
    ]
    for cand in candidates:
        if cand.exists():
            return cand

    return Path(sys.executable)


GODOT_BIN: Path = discover_godot_bin()
PYTHON_BIN: Path = discover_python_bin()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def ensure_vault_dirs() -> Dict[str, Path]:
    """Ensure all 10 Validation Vault output directories exist and return them."""
    for path in VAULT_DIRS.values():
        path.mkdir(parents=True, exist_ok=True)
    return dict(VAULT_DIRS)


def resolve_raw_bags_dir() -> Path:
    """Return `data/telemetry_ter26/raw_bags` if populated, falling back to `bags/real_bags`."""
    if RAW_BAGS_DIR.exists() and any(RAW_BAGS_DIR.glob("*.log")):
        return RAW_BAGS_DIR
    legacy = REPO_ROOT / "bags" / "real_bags"
    if legacy.exists():
        return legacy
    return RAW_BAGS_DIR


def resolve_ter26_ground_truth(ext: str = "parquet") -> Path:
    """Resolve `ter26_clean_ground_truth.<ext>` from `data/telemetry_ter26/processed/` first."""
    filename = f"ter26_clean_ground_truth.{ext.lstrip('.')}"
    primary = PROCESSED_DIR / filename
    if primary.exists():
        return primary
    legacy = TER26_CORR_DIR / filename
    if legacy.exists():
        return legacy
    return primary


def ensure_repo_in_syspath(extra_dirs: Optional[tuple[Path, ...]] = None) -> None:
    """Insert REPO_ROOT and VD_LAB_DIR into `sys.path` if not already present."""
    dirs = [REPO_ROOT, VD_LAB_DIR]
    if extra_dirs:
        dirs.extend(extra_dirs)
    for d in dirs:
        s = str(d)
        if s not in sys.path:
            sys.path.insert(0, s)
