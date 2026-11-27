# Milliken Yaw Moment Diagram (YMD) — TeR27 4WD

- **Vehicle:** `ter27_4wd` (`m = 290.0 kg`, `L = 1.540 m`, `Cl*A = 6.00 m^2`, `max_steer = 21.77 deg`, `use_3d_kinematics = True`)

## 1. YMD Summary Table (40 / 65 / 90 km/h)

| Speed [km/h] | Controller | dCN/dbeta @0g | dCN/dbeta @1g | dCN/dbeta @lim | dCN/ddelta | Trim Max Ay [g] | delta_trim [deg] | beta_trim [deg] | Sat Front / Rear [%] | Limit Balance |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 40 | OPEN_4WD | +0.3754 | +0.0911 | +0.0899 | -0.1390 | 1.18 | 10.1 | 1.0 | 77% / 63% | UNDERSTEER (front saturates first) |
| 65 | OPEN_4WD | +0.2386 | +0.2378 | +0.2276 | +0.0212 | 1.28 | 3.5 | -1.0 | 61% / 56% | UNDERSTEER (front saturates first) |
| 90 | OPEN_4WD | +0.1668 | +0.3509 | +0.1231 | +0.0893 | 2.09 | 3.2 | -6.0 | 76% / 72% | UNDERSTEER (front saturates first) |
| 40 | TV4_AUTOCROSS | +0.4540 | +0.1178 | +0.1034 | -0.1968 | 1.27 | 9.6 | 0.0 | 77% / 72% | UNDERSTEER (front saturates first) |
| 65 | TV4_AUTOCROSS | +0.3173 | +0.3304 | +0.1724 | -0.0135 | 1.57 | 5.9 | -2.0 | 76% / 68% | UNDERSTEER (front saturates first) |
| 90 | TV4_AUTOCROSS | +0.2244 | +0.4641 | +0.3251 | +0.0648 | 1.69 | 3.3 | -2.0 | 67% / 55% | UNDERSTEER (front saturates first) |

## 2. Low-Speed Tight Corner Check (25 km/h)

- **R = 6.0 m** (`Ay = 0.82 g`): `delta_trim = 8.0 deg` vs kinematic Ackermann `14.4 deg` (understeer delta `-6.4 deg`, Sat F/R `37% / 77%`).
- **R = 8.0 m** (`Ay = 0.61 g`): `delta_trim = 5.0 deg` vs kinematic Ackermann `10.9 deg` (understeer delta `-5.9 deg`, Sat F/R `17% / 77%`).

## 3. Technical Verdict

- **Physical stability (`dCN/dbeta`):** At 65 km/h, `OPEN_4WD` has `dCN/dbeta = +0.2378/deg` (@1g) and `+0.2276/deg` (@lim), while `TV4_AUTOCROSS` has `+0.3304/deg` (@1g) and `+0.1724/deg` (@lim) with `UNDERSTEER (front saturates first)`.
- **Steer authority mismatch:** At 65 km/h the car reaches its lateral limit at `delta_trim = 5.9 deg`, whereas an unconditioned gamepad stick commands up to `21.8 deg` (`3.7x` the limit steer angle) in ~`0.125 s` (`manual_steer_rate = 8.0 /s`), injecting massive front slip angle transients.
