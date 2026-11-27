# Tecnun eRacing Vehicle Dynamics & Telemetry Workbench

Unified Formula Student Vehicle Dynamics, Sim-to-Real Correlation, CasADi Optimal Control, 4WD Torque Vectoring, Godot 4 14-DOF Digital Twin, FastAPI Telemetry Backend, and Web Frontend Workspace for **TeR26 (RWD)** and **TeR27 (4WD)**.

## Quick Links

- **Repository Map**: [docs/architecture/REPO_MAP.md](docs/architecture/REPO_MAP.md)
- **System Architecture**: [docs/architecture/architecture.md](docs/architecture/architecture.md)
- **Frontend Spec (Google AI Studio)**: [frontend/AI_STUDIO_INSTRUCTIONS.md](frontend/AI_STUDIO_INSTRUCTIONS.md)
- **Validation Vault Index (Markdown)**: [outputs/validation_vault/INDEX.md](outputs/validation_vault/INDEX.md)
- **Interactive Validation Vault Dashboard (HTML, Dark MoTeC i2)**: [outputs/validation_vault/dashboard_index.html](outputs/validation_vault/dashboard_index.html)
- **Master Metrics Catalog (JSON)**: [outputs/validation_vault/master_metrics_catalog.json](outputs/validation_vault/master_metrics_catalog.json)

## Unified CLI Launcher (`ter27.py`)

All primary workflows are accessible from the repository root via `ter27.py` (which invokes `/home/alexr/project_gp_env/bin/python3 -u` and the Godot 4 binary automatically):

```bash
# Launch MirenaSim (Godot 4 14-DOF simulator)
/home/alexr/project_gp_env/bin/python3 -u ter27.py sim

# Start the FastAPI telemetry and analysis backend (REST + WebSocket /ws/telemetry on port 8000)
/home/alexr/project_gp_env/bin/python3 -u ter27.py api --host 0.0.0.0 --port 8000

# Run Skidpad CasADi OCP solver or closed-loop sub-5s 14-DOF benchmark
/home/alexr/project_gp_env/bin/python3 -u ter27.py skidpad
/home/alexr/project_gp_env/bin/python3 -u ter27.py skidpad --benchmark

# Run CasADi LTS (dynamic events or full-lap OCP, with optional 7-param trackside setup co-optimization)
/home/alexr/project_gp_env/bin/python3 -u ter27.py lts --event all --optimize-setup
/home/alexr/project_gp_env/bin/python3 -u ter27.py lts --mode full-lap --optimize-setup

# Rebuild Validation Vault master catalog, INDEX.md, and HTML gallery
/home/alexr/project_gp_env/bin/python3 -u ter27.py vault

# Run Python pytest suite and Godot headless 14-DOF regression suite
/home/alexr/project_gp_env/bin/python3 -u ter27.py test --godot
```

## Telemetry Backend (`api/`) & Web Workbench (`frontend/`)

The `api/` package provides a modular FastAPI backend bridging MirenaSim and the Validation Vault to the standalone `frontend/` workspace (designed for Google AI Studio):

- **Live Telemetry Bridge**: `src/mirena_sim/Scenes/Sim/TelemetryUdpStreamer.gd` streams JSON telemetry packets at 60 Hz over localhost UDP (`127.0.0.1:5005`) to `api/services/udp_bridge.py`, which fans them out to connected clients on `WS /ws/telemetry` (with automatic fallback frames when the simulator is idle).
- **REST Endpoints**:
  - `GET /api/health` and `GET /api/vehicle/overview`
  - `GET /api/sessions` and `GET /api/sessions/{session_id}` (DiL A/B session inspection)
  - `POST /api/tires/evaluate` (MF 6.2 / PINN tire curve sweeps)
  - `GET /api/setups` and `POST /api/setups/apply` (`default.json` and `optimal_{skidpad,accel,autocross,endurance}.json`)
  - `POST /api/lts/run` and `GET /api/lts/jobs/{job_id}` (async CasADi LTS job execution)
  - `GET /api/vault/catalog` (structured Validation Vault `01..10` index)
- **Self-Contained Frontend (`frontend/`)**: Contains `AI_STUDIO_INSTRUCTIONS.md`, `contract/` (`openapi.json`, `api_types.ts`, `endpoints_reference.md`), `mock_data/`, and a typed React + Vite + TypeScript application skeleton under `frontend/src/`.

## Repository Layout

```text
vehicle-dynamics/
|-- ter27.py                        # Unified root CLI launcher
|-- api/                            # Modular FastAPI backend (core, models, services, routers)
|-- frontend/                       # Self-contained React + TypeScript workbench for Google AI Studio
|-- src/mirena_sim/                 # Godot 4.7 14-DOF simulator, 4WD TV ECU & UDP streamer
|-- vd_correlation_lab/             # Python Vehicle Dynamics, CasADi OCP, SysID & Validation Lab
|   |-- models/                     # vehicle_14dof/, tire_mf62_pinn/, kinematics/
|   |-- control/                    # torque_vectoring/, skidpad/, sensors_ekf/, human_controls/
|   |-- optimization/               # lap_time_sim/, skidpad/, trackside_setup/
|   |-- analysis/                   # benchmarks/, validation_vault/, visualization/
|   |-- codegen/                    # CasADi C99 SIL evaluator & FEA load exporter
|   |-- config/                     # Centralized dynamic path resolution
|   |-- dbc/                        # TER.dbc CAN bus specification
|   |-- vehicles/                   # Per-vehicle calibration & reference logs
|   `-- tests/                      # Pytest suites (unit/, integration/, api/)
|-- data/                           # ttc_tire/, telemetry_ter26/, tracks/
|-- outputs/                        # validation_vault/ (01..10), dil_sessions/, casadi_cache/
|-- tests/                          # Godot headless 14-DOF regression suite (run_tests.gd)
|-- docs/                           # architecture/, guides/, reports/
`-- bags/                           # ROS 2 bag recordings
```

See [docs/architecture/REPO_MAP.md](docs/architecture/REPO_MAP.md) for the complete subdirectory breakdown.

## Quality Gates & Regression Suites

- **Grip Calibration**: `COLD_TEST_OCT = 0.47` (TeR26 Oct telemetry) and `FSG_RUBBERED = 0.57` (Hoosier R25B at 75 C, proportional R20).
- **Python Pytest Suite**: `/home/alexr/project_gp_env/bin/python3 -u -m pytest vd_correlation_lab/tests/ -v`
- **Godot Headless 14-DOF Suite**: `/home/alexr/.gemini/antigravity/bin/godot --headless --path src/mirena_sim -s ../../tests/run_tests.gd`

## ROS 2 / DevContainer Setup (Mirena Workspace)

```bash
git clone https://git.eracing.es/Driverless/mirena_ws/ --recurse-submodules
colcon build
source ./install/local_setup.zsh
```

For GPU-accelerated simulation inside the bundled VS Code DevContainer, install `nvidia-container-toolkit` and set `"default-runtime": "nvidia"` in `/etc/docker/daemon.json`, then restart Docker (`sudo systemctl restart docker`).

> [!TIP]
> Si fuera facil, lo haria cualquiera, hazlo. -- **Ozuba.**
