# Tecnun eRacing Vehicle Dynamics Validation Vault

- **Generated (UTC)**: `2026-11-27T14:32:53Z`
- **Standard**: ISO 8855 (+X fwd, +Y left, +Z up) | MoTeC i2 Dark Engineering | 300 DPI
- **Suites**: `10` | **300 DPI Figures**: `67` | **JSON Metrics Files**: `15` | **Total Plot Payload**: `42.81 MB`
- **Interactive Gallery**: [dashboard_index.html](dashboard_index.html)
- **Master Catalog JSON**: [master_metrics_catalog.json](master_metrics_catalog.json)

## Executive KPI Summary

| Suite | Figures | Key Metrics | Primary JSON |
| --- | ---: | --- | --- |
| `01` TeR26 RWD Sim-to-Real Correlation & CAN Audit | 5 | OOS wz R^2 (C): **0.9539**; OOS ay RMSE (C): **2.329 m/s^2**; EKF res_ax RMS: **0.273 m/s^2** | [metrics.json](01_ter26_sim_to_real/metrics.json) |
| `02` TeR27 4WD Maneuvers & Handling Battery | 13 | YMD Trim ay (90 km/h): **1.69 g**; K_us (65 km/h, 3D): **0.129 deg/g**; Roll Gradient: **0.80 deg/g** | [metrics.json](02_ter27_maneuvers_and_handling/metrics.json) |
| `03` TeR27 Torque Vectoring, 5-Mode Maps & Fault Injection | 12 | TV vs Open Lap Gain: **-0.775 s**; Drivability Gain: **+17.68 pts**; Torque-Fight Reduction: **-29.9%** | [tv_metrics_catalog.json](03_ter27_torque_vectoring_and_maps/tv_metrics_catalog.json) |
| `04` TeR27 Full Lap OCP & 22 km Endurance Simulation | 5 | FSG Rubbered End. Lap: **79.70 s**; OCP vs 14-DOF Gap: **0.01%**; 22 km Min Voltage: **547.8 V** | [metrics.json](04_ter27_full_lap_and_endurance/metrics.json) |
| `05` Hoosier R20 / R25B Tire & Thermal Envelope Analysis | 5 | R20 Peak mu_y (800 N): **2.122**; R25B Peak mu_y (800 N): **2.229 (+5.0%)**; R25B 22 km Delta: **-28.37 s** | [metrics.json](05_tires_and_thermal_analysis/metrics.json) |
| `06` 3D Double-Wishbone Kinematics & FEA Load Envelopes | 5 | 3D Closure Residual: **8.83e-12 mm**; Front Camber Gain: **-38.2 deg/m**; LLTD Mechanical: **40.13%** | [metrics.json](06_suspension_kinematics_and_fea/metrics.json) |
| `07` Design Tradeoffs & Formula Student Points Optimizer | 4 | Mass Sensitivity: **0.388 s / 10 kg**; +10% CL*A Equiv. Mass: **-14.71 kg**; FSG Optimal Points: **282.69 pts** | [metrics.json](07_design_tradeoffs_and_fs_points/metrics.json) |
| `08` C99 SIL Codegen & Godot-Python Parity Verification | 2 | C99 SIL Max Error: **0.0e+00**; Host C99 Latency (p50): **3.63 us**; STM32F405 CPU Load: **6.12% @ 200Hz** | [metrics.json](08_codegen_sil_and_godot_parity/metrics.json) |
| `09` TeR27 4WD Skidpad Sub-5s Diagnostic Deep-Dive | 12 | TeR26 RWD Anchor: **5.500 s**; FSG R20 MAX_ATK V2: **4.703 s**; FSG R25B MAX_ATK V2: **4.683 s** | [skidpad_deep_dive_metrics.json](09_ter27_skidpad_deep_dive/skidpad_deep_dive_metrics.json) |
| `10` TeR27 DiL Telemetry, 3D Braking Line & Sensor Realism | 4 | CAN Accel Latency: **125.0 ms**; Skidpad Real vs Ideal: **-0.068 s**; Autocross Sector Delta: **+0.343 s** | [dil_sensor_realism_metrics.json](10_ter27_dil_and_sensor_realism/dil_sensor_realism_metrics.json) |

## 01 - TeR26 RWD Sim-to-Real Correlation & CAN Audit

Strict 30% OOS val_30 validation, 6-dimension A/B/C ablation, and 200 Hz EKF+RTS CAN telemetry sanity gates.

**Key KPIs:**
- OOS wz R^2 (C): `0.9539`
- OOS ay RMSE (C): `2.329 m/s^2`
- EKF res_ax RMS: `0.273 m/s^2`
- EKF res_ay RMS: `0.326 m/s^2`
- Sanity Gates: `6/6 PASS`

**JSON Reports:** [metrics.json](01_ter26_sim_to_real/metrics.json)

| Figure | Resolution | DPI | Size (KB) | Link |
| --- | --- | --- | ---: | --- |
| Fig01 Timeseries OOS | 4032x2941 | 300x300 | 1104.8 | [fig01_timeseries_oos.png](01_ter26_sim_to_real/fig01_timeseries_oos.png) |
| Fig02 Bode Coherence | 4032x2708 | 300x300 | 828.1 | [fig02_bode_coherence.png](01_ter26_sim_to_real/fig02_bode_coherence.png) |
| Fig03 Understeer G-G | 4471x1600 | 300x300 | 893.4 | [fig03_understeer_gg.png](01_ter26_sim_to_real/fig03_understeer_gg.png) |
| Fig04 Regime Heatmap Residuals | 4647x1658 | 300x300 | 384.7 | [fig04_regime_heatmap_residuals.png](01_ter26_sim_to_real/fig04_regime_heatmap_residuals.png) |
| Fig05 CAN Telemetry Audit | 4325x2766 | 300x300 | 617.4 | [fig05_can_telemetry_audit.png](01_ter26_sim_to_real/fig05_can_telemetry_audit.png) |

## 02 - TeR27 4WD Maneuvers & Handling Battery

Milliken YMD, ISO ramp-steer & step-steer transients, Bode sweeps, trail braking, slalom/DLC, split-mu, and FS dynamic events.

**Key KPIs:**
- YMD Trim ay (90 km/h): `1.69 g`
- K_us (65 km/h, 3D): `0.129 deg/g`
- Roll Gradient: `0.80 deg/g`
- Yaw BW (-3dB, 80 km/h): `5.08 Hz`
- YMD Trim ay (60 km/h): `1.28 g`

**JSON Reports:** [metrics.json](02_ter27_maneuvers_and_handling/metrics.json)

| Figure | Resolution | DPI | Size (KB) | Link |
| --- | --- | --- | ---: | --- |
| Bode Yaw And Lat Accel | 4801x2941 | 300x300 | 775.5 | [bode_yaw_and_lat_accel.png](02_ter27_maneuvers_and_handling/bode_yaw_and_lat_accel.png) |
| FS Acceleration 75 m Telemetry | 4351x2650 | 300x300 | 662.5 | [fs_acceleration_75m_telemetry.png](02_ter27_maneuvers_and_handling/fs_acceleration_75m_telemetry.png) |
| FS Skidpad Steady State | 4350x2650 | 300x300 | 639.7 | [fs_skidpad_steady_state.png](02_ter27_maneuvers_and_handling/fs_skidpad_steady_state.png) |
| Ramp Steer Roll And Load Transfer | 4351x1658 | 300x300 | 437.8 | [ramp_steer_roll_and_load_transfer.png](02_ter27_maneuvers_and_handling/ramp_steer_roll_and_load_transfer.png) |
| Ramp Steer Understeer Gradient | 4351x2766 | 300x300 | 718.8 | [ramp_steer_understeer_gradient.png](02_ter27_maneuvers_and_handling/ramp_steer_understeer_gradient.png) |
| Slalom And DLC Traces | 4501x2650 | 300x300 | 689.5 | [slalom_and_dlc_traces.png](02_ter27_maneuvers_and_handling/slalom_and_dlc_traces.png) |
| Split mu Behavior | 4351x2591 | 300x300 | 595.5 | [split_mu_behavior.png](02_ter27_maneuvers_and_handling/split_mu_behavior.png) |
| Step Steer Metrics Comparison | 4352x2591 | 300x300 | 333.3 | [step_steer_metrics_comparison.png](02_ter27_maneuvers_and_handling/step_steer_metrics_comparison.png) |
| Step Steer Transient Response | 4351x2650 | 300x300 | 559.3 | [step_steer_transient_response.png](02_ter27_maneuvers_and_handling/step_steer_transient_response.png) |
| Trail Braking Stability Envelope | 4348x2650 | 300x300 | 729.8 | [trail_braking_stability_envelope.png](02_ter27_maneuvers_and_handling/trail_braking_stability_envelope.png) |
| YMD Grip Comparison | 4351x1716 | 300x300 | 938.5 | [ymd_grip_comparison.png](02_ter27_maneuvers_and_handling/ymd_grip_comparison.png) |
| YMD TeR27 Vs TeR26 | 4501x1716 | 300x300 | 938.5 | [ymd_ter27_vs_ter26.png](02_ter27_maneuvers_and_handling/ymd_ter27_vs_ter26.png) |
| YMD Velocity Sweep | 4801x1658 | 300x300 | 866.3 | [ymd_velocity_sweep.png](02_ter27_maneuvers_and_handling/ymd_velocity_sweep.png) |

## 03 - TeR27 Torque Vectoring, 5-Mode Maps & Fault Injection

Open-diff vs e-LSD vs TV architectures, 5 ECU modes, 4-wheel friction circles, HV/BSPD compliance, and 3WD fault resilience.

**Key KPIs:**
- TV vs Open Lap Gain: `-0.775 s`
- Drivability Gain: `+17.68 pts`
- Torque-Fight Reduction: `-29.9%`
- HV / BSPD Compliance: `0 viol / PASS`
- Eff. Mode 22 km Energy: `4.62 kWh`

**JSON Reports:** [fsg_grip_calibration_from_ter26.json](03_ter27_torque_vectoring_and_maps/fsg_grip_calibration_from_ter26.json), [skidpad_oracle_report.json](03_ter27_torque_vectoring_and_maps/skidpad_oracle_report.json), [tv_metrics_catalog.json](03_ter27_torque_vectoring_and_maps/tv_metrics_catalog.json)

| Figure | Resolution | DPI | Size (KB) | Link |
| --- | --- | --- | ---: | --- |
| Skidpad 4wheel Friction Circles | 4325x1891 | 300x300 | 232.6 | [skidpad_4wheel_friction_circles.png](03_ter27_torque_vectoring_and_maps/skidpad_4wheel_friction_circles.png) |
| Skidpad Figure8 Crossover Dynamics | 4179x2708 | 300x300 | 478.5 | [skidpad_figure8_crossover_dynamics.png](03_ter27_torque_vectoring_and_maps/skidpad_figure8_crossover_dynamics.png) |
| Skidpad Sub5 Before Vs After | 4617x2766 | 300x300 | 687.9 | [skidpad_sub5_before_vs_after.png](03_ter27_torque_vectoring_and_maps/skidpad_sub5_before_vs_after.png) |
| TV 5 Modes Radar And Bars | 4910x2200 | 300x300 | 878.9 | [tv_5_modes_radar_and_bars.png](03_ter27_torque_vectoring_and_maps/tv_5_modes_radar_and_bars.png) |
| TV Architectures Laptime Drivability | 4943x1775 | 300x300 | 424.6 | [tv_architectures_laptime_drivability.png](03_ter27_torque_vectoring_and_maps/tv_architectures_laptime_drivability.png) |
| TV Control Maps 2D Surfaces | 4651x3145 | 300x300 | 810.7 | [tv_control_maps_2d_surfaces.png](03_ter27_torque_vectoring_and_maps/tv_control_maps_2d_surfaces.png) |
| TV Fault Injection 3WD And Sensor Noise | 4998x2883 | 300x300 | 825.0 | [tv_fault_injection_3wd_and_sensor_noise.png](03_ter27_torque_vectoring_and_maps/tv_fault_injection_3wd_and_sensor_noise.png) |
| TV Friction Circles 4-Wheels | 4032x3437 | 300x300 | 975.0 | [tv_friction_circles_4wheels.png](03_ter27_torque_vectoring_and_maps/tv_friction_circles_4wheels.png) |
| TV Regen Braking Yaw Stability | 5042x1716 | 300x300 | 612.9 | [tv_regen_braking_yaw_stability.png](03_ter27_torque_vectoring_and_maps/tv_regen_braking_yaw_stability.png) |
| TV Torque Distribution 4-Wheels Trace | 4647x2941 | 300x300 | 799.4 | [tv_torque_distribution_4wheels_trace.png](03_ter27_torque_vectoring_and_maps/tv_torque_distribution_4wheels_trace.png) |
| TV Torque Fight And Power Envelope | 4910x2650 | 300x300 | 901.1 | [tv_torque_fight_and_power_envelope.png](03_ter27_torque_vectoring_and_maps/tv_torque_fight_and_power_envelope.png) |
| TV YMD Shift Open Vs CasADi | 5039x1775 | 300x300 | 817.5 | [tv_ymd_shift_open_vs_casadi.png](03_ter27_torque_vectoring_and_maps/tv_ymd_shift_open_vs_casadi.png) |

## 04 - TeR27 Full Lap OCP & 22 km Endurance Simulation

FSG 2023 CasADi OCP vs 14-DOF closed-loop validation across 4 grip profiles, 22 km voltage/thermal evolution, and human drivability audit.

**Key KPIs:**
- FSG Rubbered End. Lap: `79.70 s`
- OCP vs 14-DOF Gap: `0.01%`
- 22 km Min Voltage: `547.8 V`
- 22 km Final SOC: `45.1%`
- Drivability Score: `92.07 / 100`

**JSON Reports:** [metrics.json](04_ter27_full_lap_and_endurance/metrics.json)

| Figure | Resolution | DPI | Size (KB) | Link |
| --- | --- | --- | ---: | --- |
| Endurance 22 km Evolution And Voltage Sag | 4588x2883 | 300x300 | 847.0 | [endurance_22km_evolution_and_voltage_sag.png](04_ter27_full_lap_and_endurance/endurance_22km_evolution_and_voltage_sag.png) |
| Endurance Energy Pareto Curve | 4626x1716 | 300x300 | 504.5 | [endurance_energy_pareto_curve.png](04_ter27_full_lap_and_endurance/endurance_energy_pareto_curve.png) |
| FSG Track Maps Telemetry Overlay | 4530x3349 | 300x300 | 1486.4 | [fsg_track_maps_telemetry_overlay.png](04_ter27_full_lap_and_endurance/fsg_track_maps_telemetry_overlay.png) |
| Human Drivability Inputs Audit | 4471x2766 | 300x300 | 757.2 | [human_drivability_inputs_audit.png](04_ter27_full_lap_and_endurance/human_drivability_inputs_audit.png) |
| Laptime Surface mu Vs Temp | 4588x1775 | 300x300 | 589.2 | [laptime_surface_mu_vs_temp.png](04_ter27_full_lap_and_endurance/laptime_surface_mu_vs_temp.png) |

## 05 - Hoosier R20 / R25B Tire & Thermal Envelope Analysis

Pacejka MF 6.2 load/camber/combined-slip sweeps, R20 vs R25B compound delta, PINN residual hook, and thermal grip window.

**Key KPIs:**
- R20 Peak mu_y (800 N): `2.122`
- R25B Peak mu_y (800 N): `2.229 (+5.0%)`
- R25B 22 km Delta: `-28.37 s`
- Optimal Thermal Window: `70-85 C`
- PINN Residual Hook: `Loaded`

**JSON Reports:** [metrics.json](05_tires_and_thermal_analysis/metrics.json)

| Figure | Resolution | DPI | Size (KB) | Link |
| --- | --- | --- | ---: | --- |
| Tire Camber And Load Sensitivity | 4471x1716 | 300x300 | 560.1 | [tire_camber_and_load_sensitivity.png](05_tires_and_thermal_analysis/tire_camber_and_load_sensitivity.png) |
| Tire Combined Slip Ellipses | 4471x1775 | 300x300 | 486.4 | [tire_combined_slip_ellipses.png](05_tires_and_thermal_analysis/tire_combined_slip_ellipses.png) |
| Tire Fy Fx Mz Load Sweep | 4916x1658 | 300x300 | 594.8 | [tire_fy_fx_mz_load_sweep.png](05_tires_and_thermal_analysis/tire_fy_fx_mz_load_sweep.png) |
| Tire R20 Vs R25B Comparison | 4910x1658 | 300x300 | 483.3 | [tire_r20_vs_r25b_comparison.png](05_tires_and_thermal_analysis/tire_r20_vs_r25b_comparison.png) |
| Tire Thermal Grip Window | 4471x1716 | 300x300 | 461.0 | [tire_thermal_grip_window.png](05_tires_and_thermal_analysis/tire_thermal_grip_window.png) |

## 06 - 3D Double-Wishbone Kinematics & FEA Load Envelopes

Exact 3D loop closure (< 1e-11 mm), camber/toe/roll-center/MR curves, Ackermann steering torque, FEA member loads, and rainflow fatigue.

**Key KPIs:**
- 3D Closure Residual: `8.83e-12 mm`
- Front Camber Gain: `-38.2 deg/m`
- LLTD Mechanical: `40.13%`
- Max FEA Design Load: `9472 N (SF=1.5)`
- 22 km Miner Damage: `1.62e-07`

**JSON Reports:** [fea_hardpoint_loads.json](06_suspension_kinematics_and_fea/fea_hardpoint_loads.json), [metrics.json](06_suspension_kinematics_and_fea/metrics.json)

**CSV Exports:** [fea_hardpoint_loads.csv](06_suspension_kinematics_and_fea/fea_hardpoint_loads.csv)

| Figure | Resolution | DPI | Size (KB) | Link |
| --- | --- | --- | ---: | --- |
| FEA Hardpoint Load Envelopes 3D | 4722x1833 | 300x300 | 341.9 | [fea_hardpoint_load_envelopes_3d.png](06_suspension_kinematics_and_fea/fea_hardpoint_load_envelopes_3d.png) |
| FEA Rainflow Fatigue Spectra | 4588x1716 | 300x300 | 364.5 | [fea_rainflow_fatigue_spectra.png](06_suspension_kinematics_and_fea/fea_rainflow_fatigue_spectra.png) |
| Kinematics Ackermann And Steering Torque | 4471x1716 | 300x300 | 417.6 | [kinematics_ackermann_and_steering_torque.png](06_suspension_kinematics_and_fea/kinematics_ackermann_and_steering_torque.png) |
| Kinematics Camber Toe Bumpsteer | 4471x1716 | 300x300 | 524.2 | [kinematics_camber_toe_bumpsteer.png](06_suspension_kinematics_and_fea/kinematics_camber_toe_bumpsteer.png) |
| Kinematics Roll Center And MR | 4910x1658 | 300x300 | 479.5 | [kinematics_roll_center_and_mr.png](06_suspension_kinematics_and_fea/kinematics_roll_center_and_mr.png) |

## 07 - Design Tradeoffs & Formula Student Points Optimizer

Mass vs aero CL*A/CD*A iso-laptime contours, CG/inertia sensitivity tornado chart, and FSG/FSS Endurance + Efficiency points Pareto.

**Key KPIs:**
- Mass Sensitivity: `0.388 s / 10 kg`
- +10% CL*A Equiv. Mass: `-14.71 kg`
- FSG Optimal Points: `282.69 pts`
- FSS Optimal Points: `335.42 pts`
- Optimal 22 km Energy: `3.12 kWh`

**JSON Reports:** [metrics.json](07_design_tradeoffs_and_fs_points/metrics.json), [trackside_setup_optimization_report.json](07_design_tradeoffs_and_fs_points/trackside_setup_optimization_report.json)

| Figure | Resolution | DPI | Size (KB) | Link |
| --- | --- | --- | ---: | --- |
| Design Sensitivity Tornado Chart | 4120x2066 | 300x300 | 445.6 | [design_sensitivity_tornado_chart.png](07_design_tradeoffs_and_fs_points/design_sensitivity_tornado_chart.png) |
| FS Points Optimizer Endurance Efficiency | 4588x1775 | 300x300 | 437.7 | [fs_points_optimizer_endurance_efficiency.png](07_design_tradeoffs_and_fs_points/fs_points_optimizer_endurance_efficiency.png) |
| Trackside Setup Event Comparison | 5059x3058 | 300x300 | 561.4 | [trackside_setup_event_comparison.png](07_design_tradeoffs_and_fs_points/trackside_setup_event_comparison.png) |
| Tradeoff Mass Vs Aero Contour | 4665x1775 | 300x300 | 673.7 | [tradeoff_mass_vs_aero_contour.png](07_design_tradeoffs_and_fs_points/tradeoff_mass_vs_aero_contour.png) |

## 08 - C99 SIL Codegen & Godot-Python Parity Verification

CasADi C99 bit-exact SIL verification (0.0 abs error), STM32F405 168 MHz RT budget, and Godot 14-DOF vs Python physics parity matrix.

**Key KPIs:**
- C99 SIL Max Error: `0.0e+00`
- Host C99 Latency (p50): `3.63 us`
- STM32F405 CPU Load: `6.12% @ 200Hz`
- Godot 3D vs Py Mean Err: `0.529%`
- SIL & Parity Gate: `PASS`

**JSON Reports:** [metrics.json](08_codegen_sil_and_godot_parity/metrics.json)

| Figure | Resolution | DPI | Size (KB) | Link |
| --- | --- | --- | ---: | --- |
| Godot Vs Python Parity Matrix | 4972x1775 | 300x300 | 515.3 | [godot_vs_python_parity_matrix.png](08_codegen_sil_and_godot_parity/godot_vs_python_parity_matrix.png) |
| SIL C99 Accuracy And Latency | 4981x1658 | 300x300 | 596.8 | [sil_c99_accuracy_and_latency.png](08_codegen_sil_and_godot_parity/sil_c99_accuracy_and_latency.png) |

## 09 - TeR27 4WD Skidpad Sub-5s Diagnostic Deep-Dive

Calibrated FSG_RUBBERED (TeR26 = 5.50 s anchor), SKIDPAD_MAX_ATTACK_V2 200 Hz telemetry, 4-wheel grip/slip/load/3D kinematics, crossover micro-analysis, and Godot 4 14-DOF parity.

**Key KPIs:**
- TeR26 RWD Anchor: `5.500 s`
- FSG R20 MAX_ATK V2: `4.703 s`
- FSG R25B MAX_ATK V2: `4.683 s`
- COLD_18C Min Clear: `0.393 m`
- Godot vs Py Circle Err: `12.48%`

**JSON Reports:** [godot_skidpad_parity_trace.json](09_ter27_skidpad_deep_dive/godot_skidpad_parity_trace.json), [skidpad_deep_dive_metrics.json](09_ter27_skidpad_deep_dive/skidpad_deep_dive_metrics.json)

| Figure | Resolution | DPI | Size (KB) | Link |
| --- | --- | --- | ---: | --- |
| 01 Skidpad Time Waterfall And Benchmark | 4764x2650 | 300x300 | 414.1 | [01_skidpad_time_waterfall_and_benchmark.png](09_ter27_skidpad_deep_dive/01_skidpad_time_waterfall_and_benchmark.png) |
| 02 Skidpad Trajectory And Cone Clearance | 4822x2883 | 300x300 | 1070.3 | [02_skidpad_trajectory_and_cone_clearance.png](09_ter27_skidpad_deep_dive/02_skidpad_trajectory_and_cone_clearance.png) |
| 03 Vehicle Kinematics Beta Yaw Lataccel | 4617x3174 | 300x300 | 587.0 | [03_vehicle_kinematics_beta_yaw_lataccel.png](09_ter27_skidpad_deep_dive/03_vehicle_kinematics_beta_yaw_lataccel.png) |
| 04 4wheel Friction Circles And Utilization | 4764x2883 | 300x300 | 657.5 | [04_4wheel_friction_circles_and_utilization.png](09_ter27_skidpad_deep_dive/04_4wheel_friction_circles_and_utilization.png) |
| 05 4wheel Slip Angles And Long Slip | 4705x2766 | 300x300 | 512.6 | [05_4wheel_slip_angles_and_long_slip.png](09_ter27_skidpad_deep_dive/05_4wheel_slip_angles_and_long_slip.png) |
| 06 4wheel Vertical Loads And Weight Transfer | 4617x2941 | 300x300 | 514.7 | [06_4wheel_vertical_loads_and_weight_transfer.png](09_ter27_skidpad_deep_dive/06_4wheel_vertical_loads_and_weight_transfer.png) |
| 07 Suspension 3D Camber Toe Ackermann | 4705x2825 | 300x300 | 567.9 | [07_suspension_3d_camber_toe_ackermann.png](09_ter27_skidpad_deep_dive/07_suspension_3d_camber_toe_ackermann.png) |
| 08 Controller Mz Decomposition And Tracking | 4617x2941 | 300x300 | 527.5 | [08_controller_mz_decomposition_and_tracking.png](09_ter27_skidpad_deep_dive/08_controller_mz_decomposition_and_tracking.png) |
| 09 4wheel Torques Power And Torque Fight | 4764x2883 | 300x300 | 623.8 | [09_4wheel_torques_power_and_torque_fight.png](09_ter27_skidpad_deep_dive/09_4wheel_torques_power_and_torque_fight.png) |
| 10 Figure8 Crossover Micro Analysis | 4617x3174 | 300x300 | 713.5 | [10_figure8_crossover_micro_analysis.png](09_ter27_skidpad_deep_dive/10_figure8_crossover_micro_analysis.png) |
| 11 Skidpad Sensitivity Setup And Grip Sweep | 4823x2008 | 300x300 | 680.4 | [11_skidpad_sensitivity_setup_and_grip_sweep.png](09_ter27_skidpad_deep_dive/11_skidpad_sensitivity_setup_and_grip_sweep.png) |
| 12 Python Vs Godot Skidpad Parity | 4705x2825 | 300x300 | 750.6 | [12_python_vs_godot_skidpad_parity.png](09_ter27_skidpad_deep_dive/12_python_vs_godot_skidpad_parity.png) |

## 10 - TeR27 DiL Telemetry, 3D Braking Line & Sensor Realism

SENSOR_IDEAL vs SENSOR_TRACK_REALISM (125 ms CAN accel latency, IMU noise/bias, EKF beta/Fz) across Skidpad, Autocross, and Trail-Braking/Slalom, plus CasADi sector deltas and 3D braking ramp elevation.

**Key KPIs:**
- CAN Accel Latency: `125.0 ms`
- Skidpad Real vs Ideal: `-0.068 s`
- Autocross Sector Delta: `+0.343 s`
- Slalom Beta RMS (Real): `1.01 deg`
- 3D Brake Ramp Peak +Y: `0.450 m`

**JSON Reports:** [dil_sensor_realism_metrics.json](10_ter27_dil_and_sensor_realism/dil_sensor_realism_metrics.json)

| Figure | Resolution | DPI | Size (KB) | Link |
| --- | --- | --- | ---: | --- |
| Fig01 CAN Latency EKF Step And Noise | 4051x2475 | 300x300 | 658.2 | [fig01_can_latency_ekf_step_and_noise.png](10_ter27_dil_and_sensor_realism/fig01_can_latency_ekf_step_and_noise.png) |
| Fig02 Skidpad And Slalom Sensor Stress | 4051x2475 | 300x300 | 846.4 | [fig02_skidpad_and_slalom_sensor_stress.png](10_ter27_dil_and_sensor_realism/fig02_skidpad_and_slalom_sensor_stress.png) |
| Fig03 Autocross CasADi Sector Deltas | 4051x2475 | 300x300 | 847.2 | [fig03_autocross_casadi_sector_deltas.png](10_ter27_dil_and_sensor_realism/fig03_autocross_casadi_sector_deltas.png) |
| Fig04 3D Braking Ramp And DiL Telemetry | 4051x1658 | 300x300 | 576.7 | [fig04_3d_braking_ramp_and_dil_telemetry.png](10_ter27_dil_and_sensor_realism/fig04_3d_braking_ramp_and_dil_telemetry.png) |
