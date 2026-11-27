# Vehicle Dynamics & Validation Vault Architecture

## 1. System Overview

This repository integrates four tightly coupled engineering layers under a single ISO 8855 coordinate standard (`+X` forward, `+Y` left, `+Z` up, `+wz` left yaw, `+delta` left road-wheel steer):

1. **Telemetry Ground-Truth Pipeline (`vd_correlation_lab/control/sensors_ekf/`)**
   - Parses raw TeR26 CAN logs (`TER.dbc`), corrects sensor latency (`dt_can_accel = +125 ms`, `dt_can_steer = -50 ms`), calibrates IMU yaw mounting (`-20.68 deg`) and steering ratio (`i_steer = 1.2963`, `R^2 = 0.9721`), and applies an EKF + RTS kinematic smoother at 200 Hz.
   - Enforces 6 automated sanity gates (`telemetry_audit_report.json`), including kinematic consistency residuals `RMS(res_ax) = 0.273 m/s^2 < 0.35 m/s^2` and `RMS(res_ay) = 0.326 m/s^2 < 0.35 m/s^2`, zero NaNs on valid segments, and strict 70%/30% temporal split (`calib_70` / `val_30`).

2. **High-Fidelity Physics, 3D Kinematics & CasADi Optimal Control (`vd_correlation_lab/models/`, `vd_correlation_lab/optimization/`)**
   - **3D Double-Wishbone Kinematics (`vd_correlation_lab/models/kinematics/`)**: Exact Newton-Raphson spatial loop closure with residual `< 1e-11 mm` across bump (`+/-40 mm`) and rack (`+/-45 mm`), exported as C2-smooth CasADi B-Spline lookup tables (`max angle error < 1e-4 deg`, `max displacement error < 1e-4 mm`).
   - **Tire & Thermal Models (`vd_correlation_lab/models/tire_mf62_pinn/`, `data/ttc_tire/`)**: Hoosier 16.0x7.5-10 R20 and R25B Pacejka MF 6.2 with camber, load sensitivity, combined-slip ellipses, lumped thermal state dynamics (optimal window `70 - 85 C`), and optional PINN residual hook.
   - **4WD Torque Vectoring & Full-Lap OCP (`vd_correlation_lab/models/vehicle_14dof/`, `vd_correlation_lab/optimization/lap_time_sim/`)**: Evaluates 5 ECU modes (`AUTOCROSS`, `ACCELERATION`, `SKIDPAD`, `EFFICIENCY`, `TIRE_WARMER`) under Formula Student EV HV limits (`[-25, +80] kW`) and BSPD rules (`FSAE EV 2.3`).

3. **Real-Time Embedded C99 SIL & Godot 14-DOF Digital Twin (`vd_correlation_lab/codegen/`, `src/mirena_sim/`)**
   - **C99 SIL (`ter27_casadi_eval.c`, `verify_c_codegen_sil.py`)**: Auto-generated pure C99 CasADi evaluator compiled with `gcc -std=c99 -O3 -lm`, verified bit-exact (`0.0` max abs error vs Python across 128 operating points), host `p50` latency `1.85 us` (`ter27_eval_kappa`), and `6.12%` projected CPU load on STM32F405 @ 168 MHz (200 Hz control loop).
   - **Godot 14-DOF Simulator (`src/mirena_sim/`)**: Full 14-DOF real-time simulator with 3D kinematics LUT bundle, parity-checked against Python across Step-Steer, Skidpad, Bump-Steer, and FSG 2023 Full Lap (`0.529%` mean error) and guarded by >= 882 headless assertions across 31+ test suites.

4. **Validation Vault & MoTeC i2 Visualization (`vd_correlation_lab/analysis/`, `outputs/validation_vault/`)**
   - Every figure is rendered through `vd_correlation_lab/analysis/visualization/plot_style.py` at exact `300 DPI` in a unified dark-engineering MoTeC i2 palette (`#0D1117` canvas, `#131922` axes, pure ASCII typography).
   - `vd_correlation_lab/analysis/validation_vault/build_validation_vault_index.py` aggregates all 10 suites into `master_metrics_catalog.json`, `INDEX.md`, and the self-contained interactive gallery `dashboard_index.html` (zero external CDN dependencies, sidebar navigation, keyword filtering, and modal lightbox zoom).

## 2. Validation Vault Map (`outputs/validation_vault/`)

| Folder | Scope | Primary Generator Script |
| --- | --- | --- |
| `01_ter26_sim_to_real/` | TeR26 OOS A/B/C correlation (6 dims) + CAN telemetry audit | `vd_correlation_lab/analysis/benchmarks/run_ter26_extended_correlation.py` |
| `02_ter27_maneuvers_and_handling/` | Milliken YMD, ramp/step-steer, Bode, trail-braking, slalom/DLC, split-mu, FS events | `vd_correlation_lab/analysis/benchmarks/run_ter27_handling_battery.py` |
| `03_ter27_torque_vectoring_and_maps/` | TV architectures, 5 ECU modes, 2D maps, 4-wheel friction circles, HV/BSPD, 3WD fault injection | `vd_correlation_lab/analysis/benchmarks/run_tv_maps_exhaustive_suite.py` |
| `04_ter27_full_lap_and_endurance/` | FSG 2023 OCP vs 14-DOF profiles, 22 km voltage/thermal evolution, drivability audit | `vd_correlation_lab/analysis/benchmarks/run_full_subsystems_analysis.py` |
| `05_tires_and_thermal_analysis/` | Pacejka MF 6.2 sweeps, camber/load sensitivity, combined slip, R20 vs R25B, thermal window | `vd_correlation_lab/analysis/benchmarks/run_full_subsystems_analysis.py` |
| `06_suspension_kinematics_and_fea/` | 3D double-wishbone camber/toe/RC/MR, Ackermann torque, 3D FEA loads & rainflow fatigue | `vd_correlation_lab/analysis/benchmarks/run_full_subsystems_analysis.py` |
| `07_design_tradeoffs_and_fs_points/` | Mass vs CL*A/CD*A contours, sensitivity tornado, trackside setup co-optimization, FS points | `vd_correlation_lab/analysis/benchmarks/run_trackside_setup_comparison.py` |
| `08_codegen_sil_and_godot_parity/` | C99 SIL accuracy & STM32F405 latency budget, Godot 3D vs Python parity matrix | `vd_correlation_lab/analysis/benchmarks/run_full_subsystems_analysis.py` |
| `09_ter27_skidpad_deep_dive/` | Skidpad 4.65-4.78 s deep dive, cone clearances, figure-8 crossover & Godot parity | `vd_correlation_lab/analysis/benchmarks/run_skidpad_deep_dive_vault.py` |
| `10_ter27_dil_and_sensor_realism/` | CAN latency, IMU noise/bias, EKF beta/Fz estimation & 3D braking ramp DiL | `vd_correlation_lab/analysis/benchmarks/run_dil_sensor_stress_suite.py` |

## 3. Quality Gate Contracts

All verification gates are automated and enforced by `vd_correlation_lab/tests/` and `tests/run_tests.gd`:

- **3D Kinematics Closure**: `max_residual_overall_mm < 1e-6 mm` (achieved `8.83e-12 mm`).
- **CasADi B-Spline Accuracy**: `max_angle_err_deg < 1e-4 deg` and `max_disp_err_mm < 1e-4 mm`.
- **Telemetry Sanity**: `all_gates_passed == True`, `res_ax < 0.35 m/s^2`, `res_ay < 0.35 m/s^2`.
- **C99 SIL Accuracy & Latency**: `max_abs_err < 1e-6` (achieved `0.0`) and `p99_us < 50 us`.
- **Vault Asset Integrity**: Every PNG in `01..10` exists, `size > 20 KB`, `DPI == (300, 300)` (`+/-1`), every folder contains a valid JSON metrics report, and `master_metrics_catalog.json`, `INDEX.md`, and `dashboard_index.html` reference every PNG.
- **Pure ASCII Discipline**: Zero non-ASCII characters (zero emojis) across curated scripts, HTML dashboard, `README.md`, and `docs/architecture/`.
- **Godot 14-DOF Regression**: 100% pass (>= 882 assertions, `0` failures).
