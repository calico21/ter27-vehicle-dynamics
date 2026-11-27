# TeR27 Vehicle Dynamics & Telemetry Workbench -- Google AI Studio Instructions

This folder (`frontend/`) is a self-contained React + Vite + TypeScript workspace designed to be uploaded directly into Google AI Studio or run locally with Vite against the FastAPI + WebSocket backend (`api/main.py`).

## 1. Visual Design System & Engineering Standards

Use a high-contrast MoTeC i2 / Formula Student pit-wall engineering theme with strict tabular monospace numerals (`font-variant-numeric: tabular-nums`):

| Token | Hex | Role |
| :--- | :--- | :--- |
| Carbon Background | `#090A0D` | Primary viewport background, chart canvas |
| Surface Panel | `#12141A` | Cards, panels, tables, navigation rails |
| Technical Gold | `#C5A059` | Tecnun e-Racing accent, active tab border, optimal reference highlight |
| Telemetry Cyan | `#38BDF8` | Live 60 Hz traces, driver trace, EKF estimates |
| Alert Red | `#EF4444` | BSPD trip, tire saturation >= 0.95, limit breach, offline status |

Typography and engineering conventions:
- Monospace font: `JetBrains Mono` for all numeric KPIs, axis ticks, table cells, timestamps, and telemetry readouts.
- Zero emojis anywhere in UI labels, buttons, badges, or code comments. Use crisp vector icons (`lucide-react`) and explicit SI units (`m/s`, `g`, `rad/s`, `deg`, `N`, `Nm`, `kW`, `C`, `kPa`, `mm`).
- Coordinate standard: ISO 8855 (`+X` forward, `+Y` left, `+Z` up).
- Grip profile reference targets: `COLD_TEST_OCT` (`lambda_muy = 0.47`), `FSG_RUBBERED` (`lambda_muy = 0.57`).

## 2. Backend Connection & Automatic Offline Fallback

Start the local backend from the repository root:

```bash
python3 ter27.py api --host 0.0.0.0 --port 8000
```

- Base REST URL: `http://localhost:8000` (overridable via `VITE_TER27_API_URL`)
- Live WebSocket URL: `ws://localhost:8000/ws/telemetry` (60 Hz `TelemetryFrame` JSON packets)
- CORS is open (`allow_origins=["*"]`) so Google AI Studio preview iframes can connect directly to `http://localhost:8000`.
- Offline Fallback: Whenever `http://localhost:8000` is unreachable (e.g., cloud preview without a local tunnel), `frontend/src/services/apiClient.ts` and `frontend/src/services/telemetrySocket.ts` automatically fall back to the 5 schema-verified JSON payloads in `frontend/mock_data/`:
  - `frontend/mock_data/live_telemetry_frame.json`
  - `frontend/mock_data/dil_session_comparison.json`
  - `frontend/mock_data/tire_evaluation_curves.json`
  - `frontend/mock_data/trackside_setups_matrix.json`
  - `frontend/mock_data/validation_vault_catalog.json`

## 3. Screen Architecture (6 Views)

1. **Live Pit-Wall (`src/components/telemetry/LivePitwallView.tsx`)**
   - Top status bar (`TopStatusBar.tsx`): telemetry source (`godot_udp` / `dil_session` / `synthetic_fallback`), `grip_profile`, `sensor_profile`, HV DC power (`p_dc_kw`), SOC (`soc_pct`), BSPD flag (`bspd_active`), WebSocket state.
   - Multi-channel 60 Hz plot (`MultiChannelPlot.tsx`): synchronized `vx`, `ay_g`, `yaw_rate_rad_s`, `beta_deg`, `steer_deg`, and EKF vs 14-DOF truth overlay.
   - 4-corner tire utilization & thermal matrix (`FrictionCircle2x2.tsx`): `fl`, `fr`, `rl`, `rr` showing `fz_n`, `fx_n`, `fy_n`, `slip_angle_deg`, `slip_ratio`, `tire_util`, `temp_c`, `motor_torque_nm`.
2. **Session A/B Forensics (`src/components/forensics/SessionABComparator.tsx`)**
   - Fetches `/api/sessions` and `/api/sessions/{session_id}`.
   - Renders track map `(x_m, y_m)` colored by speed or `delta_t_s`, synchronized Driver vs CasADi optimal reference traces, KPI cards, and `SectorDeltaBreakdown.tsx`.
3. **3D Digital Twin (`src/components/digital_twin/Vehicle3DViewport.tsx`)**
   - Fetches `/api/vehicle/overview`.
   - Displays TeR27 4WD physical parameters, 3D double-wishbone hardpoints (`SuspensionKinematicsOverlay.tsx`), and live wheel travel from WebSocket frames.
4. **MF 6.2 Tire Lab (`src/components/tire_lab/TireModelExplorer.tsx`)**
   - Calls `POST /api/tires/evaluate` with sliders for `fz_n`, `camber_deg`, `temp_c`, `pressure_kpa`, `compound` (`R20` / `R25B`), and `grip_profile` (`COLD_TEST_OCT` / `FSG_RUBBERED`).
   - Plots `fy_vs_alpha`, `fx_vs_kappa`, `mz_vs_alpha`, and `friction_ellipse`.
5. **Trackside Setup Matrix & Simulation Launcher (`src/components/setup_matrix/`)**
   - `TracksideSetupMatrix.tsx`: fetches `GET /api/setups` (`default` vs `optimal_{skidpad,accel,autocross,endurance}`), shows parameter deltas, laptime gains, Formula Student points delta, and triggers live UDP push via `POST /api/setups/apply`.
   - `SimulationLauncher.tsx`: dispatches async CasADi optimal control jobs via `POST /api/simulation/run` and polls `GET /api/simulation/status/{job_id}`.
6. **Validation Vault (`src/components/validation/ValidationVaultBrowser.tsx`)**
   - Fetches `GET /api/vault/catalog` and renders all 10 verification suites (`01..10`) with KPI pills and 300-DPI figure metadata.

## 4. Contract Files to Keep Synchronized

- `frontend/contract/api_types.ts` -- Strict TypeScript interfaces generated from `api/models/`.
- `frontend/contract/openapi.json` -- OpenAPI 3.1 schema exported from `api.main.app.openapi()`.
- `frontend/contract/endpoints_reference.md` -- REST and WebSocket endpoint reference table.
- `frontend/src/types/index.ts` -- Single re-export barrel from `../../contract/api_types`.
