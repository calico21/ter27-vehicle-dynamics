# TeR27 Vehicle Dynamics & Telemetry API -- Endpoints Reference

Base URL (local default): `http://localhost:8000`
WebSocket URL: `ws://localhost:8000/ws/telemetry`
OpenAPI Specification: `frontend/contract/openapi.json`
TypeScript Contract: `frontend/contract/api_types.ts`

## REST Endpoints

| Method | Path | Request Model | Response Model | Description |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/api/health` | - | `HealthResponse` | Backend health, UDP link state, active WS client count, grip profile targets (`COLD_TEST_OCT=0.47`, `FSG_RUBBERED=0.57`) |
| `GET` | `/api/vehicle/overview` | - | `VehicleOverviewResponse` | TeR27 4WD physical parameters, 3D double-wishbone hardpoints, grip profiles, setup bounds, validation status summary |
| `GET` | `/api/sessions` | - | `SessionsListResponse` | Lists available DiL CSV recordings and CasADi optimal reference sessions |
| `GET` | `/api/sessions/{session_id}` | - | `SessionABComparisonResponse` | Synchronized Driver vs CasADi reference traces (`x_m`, `y_m`, `vx_mps`, `ay_g`, `ax_g`, `beta_deg`, `steer_deg`, `mz_tv_nm`, `delta_t_s`) and KPIs |
| `POST` | `/api/tires/evaluate` | `TireEvaluateRequest` | `TireEvaluateResponse` | Live Pacejka MF 6.2 evaluation for Hoosier 16.0x7.5-10 (`R20` / `R25B`) returning `fy_vs_alpha`, `fx_vs_kappa`, `mz_vs_alpha`, and `friction_ellipse` |
| `GET` | `/api/setups` | - | `SetupsMatrixResponse` | `default.json` vs `optimal_{skidpad,accel,autocross,endurance}.json` differential matrix, CasADi time gains, and Formula Student points delta |
| `POST` | `/api/setups/apply` | `ApplySetupRequest` | `ApplySetupResponse` | Dispatches live UDP command to MirenaSim (`127.0.0.1:9877`) for preset switch, custom setup override, or sensor profile toggle |
| `POST` | `/api/simulation/run` | `SimulationRunRequest` | `SimulationJobStatusResponse` | Launches async CasADi optimal control solver (`skidpad`, `accel`, `autocross`, `endurance`, optional `optimize_setup` and `dry_run`) |
| `GET` | `/api/simulation/status/{job_id}` | - | `SimulationJobStatusResponse` | Polls async simulation job state (`queued`, `running`, `completed`, `failed`), `progress_pct`, `log_tail`, and `result` |
| `GET` | `/api/vault/catalog` | - | `VaultCatalogResponse` | Structured Validation Vault catalog across all 10 suites (`01..10`) with KPIs and 300-DPI figure metadata |

## Real-Time WebSocket Stream

| Protocol | Path | Payload Model | Rate | Description |
| :--- | :--- | :--- | :--- | :--- |
| `WS` | `/ws/telemetry` | `TelemetryFrame` | 60 Hz | Live 14-DOF state from MirenaSim (`godot_udp` via UDP `9876`), replayed DiL CSV (`dil_session`), or 14-DOF Skidpad synthesis (`synthetic_fallback`) |

## Offline Mock Data Mapping

| Mock File (`frontend/mock_data/`) | Endpoint Source | TypeScript Interface |
| :--- | :--- | :--- |
| `live_telemetry_frame.json` | `WS /ws/telemetry` | `TelemetryFrame` |
| `dil_session_comparison.json` | `GET /api/sessions/{session_id}` | `SessionABComparisonResponse` |
| `tire_evaluation_curves.json` | `POST /api/tires/evaluate` | `TireEvaluateResponse` |
| `trackside_setups_matrix.json` | `GET /api/setups` | `SetupsMatrixResponse` |
| `validation_vault_catalog.json` | `GET /api/vault/catalog` | `VaultCatalogResponse` |
