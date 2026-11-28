# Tecnun eRacing TeR27 (280.0 kg) Vehicle Dynamics & Trackside Engineering System Specification

## 1. Mission & System Overview
This specification defines the complete backend API, data schemas, physical constraints, and UI architecture for the **Tecnun eRacing TeR27 4WD** Vehicle Dynamics, Sim-to-Real Correlation, and Trackside Race Engineering ("15 Minutes Between Runs") platform.

The platform unifies:
1. **3 Vehicle Dynamics Models** x **4 Orthogonal Suspension Architectures** (12 combinations):
   - `full_14dof`: Reference 14-DOF (6-DOF chassis + 4 unsprung masses + 4 spinning wheels, 1000 Hz)
   - `fast_3dof`: Planar 3-DOF + 4-wheel model with 35 ms LLTD filter (200 Hz)
   - `differentiable_casadi_14dof`: C2-smooth 14-DOF CasADi SX model compiled to `libter_diff_14dof.so` with MF 6.2 + PINN tires, exact reverse-mode AD sensitivities (`dT/dtheta`), and C-codegen (200 Hz)
   - Suspension modes: `analytic_1d_lut`, `3d_double_wishbone_lut`, `decoupled_heave_roll`, `standard_pushrod`
2. **Trackside Race Engineering ("15 Minutes Between Runs")**:
   - Ingests `.csv`, `.log`/`.asc`, `.blf`, `.parquet`, or JSON trace payloads
   - Automatically applies `RL <-> FR` tire temperature sensor remap when present and enforces ISO 8855 sign conventions (`+X` forward, `+Y` left, `+Z` up, `+wz` CCW)
   - Runs `<30s` Gauss-Newton / adjoint System Identification (`fast_adjoint_sysid.py`) on `libter_diff_14dof.so`
   - Generates Top 3 pit-box setup adjustments strictly realizable in `< 15 min` (`setup_advisor.py`)
3. **TeR27 280.0 kg Limit Study & Exhaustive Scenario Matrix**:
   - Evaluates Skidpad, Acceleration (75 m), Autocross (1.184 km FSG), and Endurance (22 km) across grip, tire thermal window, mass/aero sensitivity, and Asymmetric `Tube-MPC` fault resilience.

---

## 2. Hard Physical Rules & FSAE EV Invariants (NEVER VIOLATE)
All recommendations, what-if sweeps, and control outputs MUST satisfy:
- **TeR27 Nominal Race Mass**: `280.0 kg` (226.0 kg sprung + 54.0 kg unsprung; reduced from 290.0 kg baseline)
- **Empirical Grip Targets (LOCKED)**:
  - `COLD_TEST_OCT`: `lambda_muy = 0.47` (cold October test track)
  - `FSG_RUBBERED`: `lambda_muy = 0.57` (Formula Student Germany rubbered asphalt)
- **FSAE EV HV DC Power Envelope**: `[-25.0, +80.0] kW` (`P_dc_min >= -25.0 kW`, `P_dc_max <= +80.0 kW`)
- **BSPD EV 2.3**: `0` violations (no simultaneous hard braking + `> 5.0 kW` HV power)
- **Track Corridor Cone Clearance**: `min_clear_m >= 0.150 m` (car half-width `0.720 m`)
- **Static Yaw Stability**: `dMz / dbeta < 0 Nm/rad` (understeer / restoring yaw moment)
- **Front Diffuser Ride-Height Floor**: `min_rh_f >= 10.0 mm` (no bump-stop stall)
- **Pit-Box Time Budget (< 15 min)**:
  - LEGAL: ARB clicks, damper rebound/compression clicks, cold tire pressure in `0.02 bar` steps, heave/preload turns, VCU torque-vectoring/TC map parameters.
  - EXCLUDED (> 15 min): spring swaps (45 min), gearbox ratio swaps (120 min), wishbone/pickup changes (90 min), aero wing swaps (35 min).

---

## 3. REST & WebSocket API Endpoints (`openapi_schema.json`)
Base URL: `http://localhost:8000` | WebSocket: `ws://localhost:8000/ws/telemetry`

### Core Telemetry, Vehicle & Vault Endpoints
- `GET /api/health` -> `HealthResponse`: Backend health, UDP link state (`9876`/`9877`), active WS clients, grip targets (`COLD_TEST_OCT=0.47`, `FSG_RUBBERED=0.57`).
- `GET /api/vehicle/overview` -> `VehicleOverviewResponse`: TeR27 physical params, 3D double-wishbone hardpoints, grip profiles, setup bounds, validation status.
- `GET /api/models/catalog` -> `ModelsCatalogResponse`: Full catalog of 3 vehicle models x 4 suspension modes (12 combinations), vehicle parameters, FSAE EV constraints, and setup bounds.
- `GET /api/sessions` -> `SessionsListResponse`: Lists DiL CSV recordings and CasADi optimal reference sessions.
- `GET /api/sessions/{session_id}` -> `SessionABComparisonResponse`: Synchronized Driver vs CasADi reference traces (`x_m`, `y_m`, `vx_mps`, `ay_g`, `ax_g`, `beta_deg`, `steer_deg`, `mz_tv_nm`, `delta_t_s`).
- `POST /api/tires/evaluate` (`TireEvaluateRequest` -> `TireEvaluateResponse`): Live Pacejka MF 6.2 evaluation for Hoosier 16.0x7.5-10 (`R20` / `R25B`).
- `GET /api/setups` -> `SetupsMatrixResponse`: `default.json` vs `optimal_{skidpad,accel,autocross,endurance}.json` matrix and FS points delta.
- `POST /api/setups/apply` (`ApplySetupRequest` -> `ApplySetupResponse`): Live UDP command dispatch to MirenaSim (`127.0.0.1:9877`).
- `POST /api/simulation/run` & `GET /api/simulation/status/{job_id}`: Async CasADi OCP solver execution and polling.
- `GET /api/vault/catalog` -> `VaultCatalogResponse`: Validation Vault catalog across all 11 suites with KPIs and 300-DPI figures.
- `WS /ws/telemetry` -> `TelemetryFrame` (60 Hz): Live 14-DOF state stream (`godot_udp`, `dil_session`, or `synthetic_fallback`).

### Trackside Race Engineering & What-If Endpoints
- `POST /api/trackside/analyze` (`TracksideAnalyzeRequest` -> `TracksideAnalyzeResponse`):
  - Request: `{ log_path?, trace?, vehicle_id ("ter27_4wd"|"ter26_rwd"), event ("skidpad"|"accel"|"autocross"|"endurance"), fs_hz, apply_tire_temp_remap?, max_iters }`.
  - Response: `total_pipeline_time_s` (`< 30.0 s`), `within_30s_budget`, `within_15min_budget`, `total_pit_time_min` (`< 15.0 min`), `combined_delta_ms`, `ingested_summary`, `sysid` (`lambda_mu_hat`, `sigma_lambda_mu`, `cl_a_hat`, `cd_a_hat`, `k_arb_ratio`, `k_spring_ratio`, `dT_dtheta`), `top_3_adjustments`, `fsae_ev_invariants`, `excluded_hardware_swaps`.
- `POST /api/whatif/simulate` (`WhatIfSimulateRequest` -> `WhatIfSimulateResponse`):
  - Request: `{ event, mass_kg, grip_lambda_mu, tire_temp_c, aero_cla_m2, suspension_mode, model_id, tube_mpc_enabled, fault_mode, use_cached_matrix }`.
  - Response: `predicted_time_s`, `baseline_280kg_time_s`, `delta_vs_baseline_ms`, `max_ay_g`, `max_ax_g`, `max_vx_mps`, `p_dc_min_kw`, `p_dc_max_kw`, `min_clear_m`, `cones_hit`, `bspd_violations`, `dmz_dbeta_nm_rad`, `yaw_stable`, `invariants_passed`.

---

## 4. Unified Dataset (`master_dataset.json`) Structure
`master_dataset.json` bundles all offline/fallback data for zero-backend prototyping in Google AI Studio:
- `metadata`: schema version, vehicle (`ter27_4wd`, 280.0 kg), locked grip profiles.
- `models_catalog`: full `GET /api/models/catalog` payload.
- `ter27_280kg_limit_study`: executive results and physical attribution waterfalls (290 kg -> 280 kg) for Skidpad (`4.838 s`), Accel (`3.708 s`), Autocross (`79.230 s`), Endurance (`1577.95 s`, `333.34 pts`).
- `ter26_empirical_validation`: adjoint SysID results across 5 real TeR26 track runs (`wz_rads` r=0.988, `ay_ms2` r=0.960, `ax_ms2` r=0.927, `vx_ms` r=0.961).
- `scenario_matrix`: complete `all_scenarios_metrics.json` sweep across all 5 axes.
- `trackside_demo_recommendations`: timed `<30s` Trackside SysID + `<15 min` Setup Advisor reports for `ter26_rwd` and `ter27_4wd`.
- `time_series_curves`: synchronized Driver vs CasADi telemetry traces and Pacejka MF 6.2 tire curves.

---

## 5. Google AI Studio UI & UX Guidance
Design a high-contrast, dark-mode motorsport engineering cockpit with 5 focused views:
1. **Trackside Pit-Box ("15 Minutes Between Runs")**:
   - Drag-and-drop `.csv`/`.asc`/`.blf` log uploader or sample selector -> calls `POST /api/trackside/analyze`.
   - Display `lambda_mu_hat +/- sigma`, `wall_time_s` badge (`< 30s` green), and Top 3 Pit-Box Adjustment cards with exact click/bar/turn deltas, `predicted_delta_ms`, and `time_to_adjust_min` (`<= 15 min` progress bar).
   - Display FSAE EV Invariants traffic-light strip (`[-25, +80] kW`, `0 BSPD`, `min_clear >= 0.150 m`, `dMz/dbeta < 0`).
2. **What-If Sensitivity Explorer**:
   - Interactive sliders for `mass_kg` (275-290 kg), `grip_lambda_mu` (0.47-0.57), `tire_temp_c` (25-95 C), `aero_cla_m2`, plus dropdowns for `model_id` (3), `suspension_mode` (4), and `Tube-MPC` toggle -> calls `POST /api/whatif/simulate`.
3. **280 kg Limit Study & Waterfall Attribution**:
   - Interactive waterfall chart showing `290 kg -> 280 kg` gains by physical mechanism (mass, powertrain loss recovery, chassis/aero, free trajectory + 4WD TV, thermal conditioning).
4. **Live Telemetry & DiL A/B Comparison**:
   - 60 Hz WebSocket stream (`WS /ws/telemetry`) + synchronized Driver vs CasADi trace overlay (`GET /api/sessions/{id}`).
5. **Tire & Validation Vault**:
   - Interactive Pacejka MF 6.2 friction ellipse (`POST /api/tires/evaluate`) and 11-suite Validation Vault browser (`GET /api/vault/catalog`).
