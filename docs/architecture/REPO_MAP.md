# TeR27 Vehicle Dynamics Repository Map (`REPO_MAP.md`)

Canonical directory map for the Tecnun eRacing Vehicle Dynamics, MirenaSim 14-DOF Digital Twin, CasADi Optimal Control, Validation Vault, FastAPI Telemetry Backend, and Web Workbench workspace.

## Top-Level Directory Tree

```text
vehicle-dynamics/
|-- ter27.py                        # Unified CLI launcher (sim, api, skidpad, lts, vault, test)
|-- README.md                       # Workspace overview, quickstart & CLI guide
|-- api/                            # Modular FastAPI backend + UDP-to-WebSocket telemetry bridge
|   |-- main.py                     # Thin entrypoint runnable via uvicorn api.main:app
|   |-- core/                       # config.py, app_factory.py, lifecycle.py
|   |-- models/                     # Pydantic v2 schemas (telemetry, vehicle, tires, setups, simulation, validation)
|   |-- services/                   # udp_bridge, tire_service, setup_service, simulation_runner, session_service, vault_service
|   `-- routers/                    # REST routers & WS /ws/telemetry
|-- frontend/                       # Self-contained React + Vite + TypeScript workbench for Google AI Studio
|   |-- AI_STUDIO_INSTRUCTIONS.md   # Engineering specification & visual palette
|   |-- contract/                   # openapi.json, api_types.ts, endpoints_reference.md
|   |-- mock_data/                  # Offline telemetry, session, tire, setup & vault fixtures
|   `-- src/                        # Typed React component hierarchy, services & state store
|-- src/
|   |-- mirena_sim/                 # Godot 4.7 14-DOF simulator, 4WD TV ECU, DiL, UI & UDP streamer
|   |   |-- Scenes/Sim/             # Simulation scene, sim.gd, TelemetryUdpStreamer.gd, SimUI
|   |   |-- Scenes/Vehicle/Config/vehicles/
|   |   |   |-- ter26_rwd/          # TeR26 RWD parameters & setups
|   |   |   `-- ter27_4wd/          # TeR27 4WD parameters, tv_4wd_config.json & optimal_*.json setups
|   |   |-- Scenes/Vehicle/Dynamics/# Full14DofModel.gd, EcuTorqueVectoring4Wd.gd, MF62/PINN tires
|   |   `-- TrackFiles/             # FSG 2023, Skidpad, Acceleration & custom tracks
|   `-- mirena_*/                   # ROS 2 Driverless stack packages (untouched)
|-- vd_correlation_lab/             # Python Vehicle Dynamics, CasADi OCP, SysID & Validation Lab
|   |-- models/
|   |   |-- vehicle_14dof/          # ter27_physics.py, ter27_casadi_model.py, ter26_high_fidelity_model.py
|   |   |-- tire_mf62_pinn/         # MF 6.2 calibration, PINN residual training & validation framework
|   |   `-- kinematics/             # 3D double-wishbone solver, CasADi B-Spline LUT & Godot bundle exporter
|   |-- control/
|   |   |-- torque_vectoring/       # optimize_tv_maps.py, optimize_tv_closed_loop.py, synthesize_tv_from_ocp.py
|   |   |-- skidpad/                # skidpad_tv_controller.py (SKIDPAD_MAX_ATTACK_V2)
|   |   |-- sensors_ekf/            # sensor_realism_profile.py, clean_ter26_telemetry.py, telemetry_sanity_gates.py
|   |   `-- human_controls/         # GamepadController.gd, WheelFFBController.gd, gamepad.json
|   |-- optimization/
|   |   |-- lap_time_sim/           # solve_full_lap_ocp.py, solve_dynamic_events_casadi.py, QSS/transient LTS, GGV
|   |   |-- skidpad/                # optimize_skidpad_tv_casadi.py, calibrate_fsg_grip_from_ter26.py
|   |   `-- trackside_setup/        # vehicle_design_and_setup_casadi.py, sysid_sim_to_real_casadi.py
|   |-- analysis/
|   |   |-- benchmarks/             # Handling battery, TV maps, Skidpad sub-5s, DiL, YMD & Godot replay scripts
|   |   |-- validation_vault/       # build_validation_vault_index.py
|   |   `-- visualization/          # plot_style.py (300 DPI MoTeC i2) & plotting suites
|   |-- codegen/                    # CasADi C99 SIL evaluator, verification & FEA load exporter
|   |-- config/                     # paths.py centralized dynamic path resolution
|   |-- dbc/                        # TER.dbc CAN bus specification
|   |-- vehicles/                   # Per-vehicle logs, models & sensor_calibration.json
|   |-- results/                    # Solver & correlation intermediate JSON/CSV/PNG outputs
|   `-- tests/
|       |-- unit/                   # test_dil_sensor_realism.py, test_skidpad_sub5_tv.py
|       |-- integration/            # test_trackside_setup_lts.py, test_validation_vault_and_physics.py
|       `-- api/                    # test_api_and_repo_integrity.py
|-- data/
|   |-- ttc_tire/                   # Hoosier 16x7.5-10 & 18x6.0-10 TTC experimental datasets
|   |-- telemetry_ter26/            # processed/ ground truth, raw_bags/ CAN logs, kinematics_source/
|   `-- tracks/                     # Canonical parametric track JSONs (fsg_2023, mirena_track10, fsg_autocross_endurance)
|-- outputs/
|   |-- validation_vault/           # Curated 300 DPI MoTeC PNGs (01..10), JSON KPIs, INDEX.md & dashboard_index.html
|   |-- dil_sessions/               # Recorded Driver-in-the-Loop CSV/JSON sessions
|   `-- casadi_cache/               # Compiled CasADi solver libraries & NLP warm-start cache
|-- tests/                          # Godot headless 14-DOF regression suite (run_tests.gd, >= 882 assertions)
|-- docs/
|   |-- architecture/               # REPO_MAP.md, architecture.md
|   |-- guides/                     # Operational and workflow guides
|   `-- reports/                    # Technical engineering reports
`-- bags/                           # ROS 2 bag recordings
```

## Key Invariants & Protected Assets

- **Grip Calibration**:
  - `COLD_TEST_OCT`: `lambda_muy = 0.47` (TeR26 Oct telemetry correlation).
  - `FSG_RUBBERED`: `lambda_muy = 0.57` (fresh Hoosier R25B at 75 C, proportional R20).
- **Protected Data & Models**: `data/`, `vd_correlation_lab/models/tire_mf62_pinn/`, `outputs/validation_vault/01..10`, `outputs/dil_sessions/`, `src/mirena_sim/Scenes/Vehicle/Config/vehicles/ter27_4wd/setups/{default,optimal_*}.json`, `tests/`, and `vd_correlation_lab/tests/`.
- **SimUI Contract**: All SimUI controls enforce `Control.FOCUS_NONE` and preserve the `F8` shortcut.
- **ASCII Discipline**: Pure ASCII (zero emojis, zero non-ASCII) across active code, JSON, TypeScript, and Markdown.
