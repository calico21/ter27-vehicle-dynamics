# MIRENA Vehicle Dynamics Architecture (TeR27)

## 1. Authority & Frame Conventions

- **Single Physics Authority:** Godot 4 (`physics_ticks_per_second = 400`, `state.step = 0.0025 s`) owns 3D track contact and closed-loop Driverless simulation via [MirenaCarDynamics.gd](Core/MirenaCarDynamics.gd) calling [VehicleDynamicsModel.gd](Core/VehicleDynamicsModel.gd)`::step_forces()`.
- **Headless 14-DOF Integrator:** `VehicleDynamicsModel.step_standalone()` solves the 6-DOF body (Z-Y-X Euler, RK2 Heun) + 4 wheels + 4 suspension corners in ISO 8855 over the **exact same `step_forces()` pipeline** without duplicating tire, suspension, or drivetrain equations.

### Coordinate Systems & Sign Conventions

| Axis / Quantity | Internal Vehicle Dynamics (ISO 8855 / ROS REP-103) | Godot 3D (`RigidBody3D`, Right-Handed) | Mapping (`VehicleConfig`) |
|---|---|---|---|
| Longitudinal | `+X` Forward | `-Z` Forward | `x_ros = -z_godot` |
| Lateral | `+Y` Left | `-X` Left (`+X` Right) | `y_ros = -x_godot` |
| Vertical | `+Z` Up | `+Y` Up | `z_ros = y_godot` |
| Steering (`steer_angle > 0`) | CCW around `+Z` (Left turn) | CCW around `+Y` (`yaw_rate > 0`) | Identical (`+` turns left) |
| Tire Slip Angle (`alpha`) | `alpha = -atan2(v_lat, |u|)` | Derived in ISO wheel frame | `delta > 0 => alpha > 0 => Fy > 0, Mz < 0` |
| Suspension Travel (`delta_z`) | `> 0` compression (jounce), `0` static design | `ray_rest_length - hit_dist` | Positive upward normal force `Fz >= 0` |
| Kinematic Camber Table | Vehicle convention (`< 0` top inward) | N/A | Left: `gamma_L = +tab`; Right: `gamma_R = -tab` |
| Kinematic Toe Table | Vehicle convention (`> 0` toe-in) | N/A | Left: `tau_L = -tab`; Right: `tau_R = +tab` |

---

## 2. Parameter Provenance

All JSON configurations reside under `Core/Config/`:

1. **TeR27 Design / Measured Parameters (`CALCULADORA_FUERZAS_TeR27_ARB_v7.xlsx`):**
   - [chassis.json](Core/Config/chassis.json): `mass = 290.0 kg` (car + driver, incl. `4 x 7.5 = 30 kg` unsprung), `cg_height = 0.290 m` (`SSF = 2.121 g`), `wheelbase = 1.540 m`, `track_width_front = track_width_rear = 1.230 m`, `weight_distribution_rear = 0.55` (`45%` F / `55%` R static split), `wheel_radius = 0.2032 m`, `max_steer_angle = 0.38 rad`, `ackermann_factor = 0.60`.
   - [suspension.json](Core/Config/suspension.json): `spring_stiffness_front = 35000.0 N/m`, `motion_ratio_front = 0.884956` (`1 / 1.13`, $k_{\text{wheel},f} = 27{,}410.1\text{ N/m}$); `spring_stiffness_rear = 48000.0 N/m`, `motion_ratio_rear = 0.884956` (`1 / 1.13`, $k_{\text{wheel},r} = 37{,}591.0\text{ N/m}$).
   - [drivetrain.json](Core/Config/drivetrain.json), [brakes.json](Core/Config/brakes.json), [aero.json](Core/Config/aero.json): `power_limit_watts = 80000.0 W` (Formula Student rule), `gear_ratio = 5.0`, `front_torque_split = 0.35` (35/65 AWD), `brake_bias_front = 0.7659`, `cl_a = 6.0` (`aero_balance_front = 0.48`), `cd_a = 1.5`, `cp_height = 0.144 m`.

2. **TTC Calibrated Tires (`Tires/MF62/` & [tires.json](Core/Config/tires.json)):**
   - Production compounds: `hoosier_16x7.5_10_r25b_mf62.json` (Round 8, Run 6/7) and `hoosier_16x7.5_10_r20_mf62.json` (Round 9, Run 8/9).
   - Default closed-loop setting uses pure MF 6.2 (`TireModel.enable_neural_correction = false`) with `surface_grip_scale = 0.78` smooth road-friction envelope in `WheelDynamics` to scale raw Calspan belt friction ($\mu \approx 2.5$) to track asphalt ($\mu < \text{SSF}$) while preserving low-slip stiffness.
   - Longitudinal $F_x$ and combined slip ($G_{x\alpha}, G_{y\kappa}$) use surrogate calibration from TTC Round 9 `Hoosier 18.0x6.0-10 R20` (`alpha in {-3, -6} deg`).

3. **Estimated / Configurable Placeholders** (tagged in each JSON `_parameter_provenance` block):
   - Damper compression/rebound rates (`2000/3000` F, `2000/3000` R N·s/m), ARB wheel rates (`4500` F / `5500` R N/m for TLLTD 45/55%), bumpstop onset/stiffness, and 1D Optimum Kinematics camber tables (`-1.8°` F / `-1.2°` R static + bump camber gain; `toe_table_*_deg` empty => `0.0`).
   - `motor_peak_torque`, `motor_max_rpm = 12000.0`, `motor_taper_ratio = 0.85`, `drivetrain_efficiency = 0.92`, `brake_mu = 1.30`, `brake_bias_front = 0.60`, `cd_a = 2.75`, `cl_a = 4.554`, `aero_balance_front = 0.45`.

---

## 3. Extension Interfaces & Future Experimental Correlation

### Extension Points
- **`SuspensionSystem` (`Suspension/SuspensionSystem.gd`, `SuspensionCorner.gd`):** Swap 4-corner spring/damper + ARB + 1D lookup tables for a multibody / nonlinear table evaluator behind `evaluate_from_corners(delta_z, delta_z_dot, grounded, config)` without modifying `Tires/`, `Chassis/`, or `MirenaCarDynamics.gd`.
- **`TorqueVectoring` (`Drivetrain/TorqueVectoring.gd`):** Evaluates at `200 Hz` (`tv_rate_hz = 200.0`, $\Delta t_{\text{TV}} = 0.005\text{ s}$) behind an internal accumulator. Supports `MODE_OPEN` (symmetric 50/50 L-R) and `MODE_DYC` ($\Delta M_{z,\text{req}}$ via `VehicleInputs.tv_yaw_moment_req`) plus direct per-wheel override (`override_wheel_torques`) for external integration with `Torque-Vectoring-eRacing`.
- **`TireModel` (`Tires/TireModel.gd`):** Compound switching and optional `enable_neural_correction` / `enable_transient_relaxation`.

### Pending for TeR27 Track Correlation (Next V-Model Stage)
Software unit tests (`run_tests.gd`) and headless maneuvers (`test_acceptance_maneuvers.gd`) verify **software consistency and physical coherence**, *not* track validation. Before claiming experimental correlation:
1. Measure actual TeR27 `h_cog`, rotational inertias (`Ixx, Iyy, Izz`), damper dyno curves, ARB torsional stiffnesses, and Optimum Kinematics camber/toe tables.
2. Verify in-hub motor torque-speed curves, efficiency map, regenerative braking limits, and brake caliper pressure-to-torque gain (`brake_mu`, `brake_bias_front`).
3. Measure track aero coefficients (`cd_a`, `cl_a_front`, `cl_a_rear`) via coast-down and ride-height potentiometers.
4. Correlate simulated vs logged telemetry (`wheel_speeds`, IMU `ax/ay/r`, steer angle, suspension travel) across Skidpad, Acceleration, and Autocross runs.
