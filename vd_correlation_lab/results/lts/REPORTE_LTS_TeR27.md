# Reporte Técnico LTS Híbrido — TeR27 4WD (`MirenaSim`)

- **Duración total del benchmark:** 1422.8 s

## 1. Trazados y Línea de Carrera de Mínima Curvatura

- **`fsg_autocross_endurance`**: $L_0 = 1017.2\text{ m}$ ($w = 3.60\text{ m}$, $R_{\min,0} = 4.92\text{ m}$) -> Línea óptima $L_r = 1013.9\text{ m}$ ($-20.5\%$ en $\int \kappa^2 ds$, $R_{\min,r} = 7.50\text{ m}$).

- **`mirena_track10`** (`track10.json` del repo): $L_0 = 449.1\text{ m}$ ($w = 3.00\text{ m}$, $R_{\min,0} = 5.07\text{ m}$) -> Línea óptima $L_r = 448.3\text{ m}$ ($-25.3\%$ en $\int \kappa^2 ds$, $R_{\min,r} = 7.05\text{ m}$).


## 2. Comparativa A/B/C/D en `fsg_autocross_endurance` (14-DOF 3D + Proyección 22 km)

| Controlador | QSS [s] | 14-DOF [s] | Δt vs Open | Ratio 14D/QSS | max|β| [°] | E_net [kJ] | E_reg [kJ] | T_22km [s] | E_22km [kWh] | E·t² rel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Open-Diff 4WD | 49.908 | 58.518 | +0.000 | 1.1725 | 3.59 | 1261.4 | 0.0 | 1269.7 | 7.60 | 1.597 |
| TV V3 | 53.770 | 57.076 | -1.442 | 1.0615 | 7.32 | 1364.2 | 0.0 | 1238.5 | 8.22 | 1.643 |
| TV 4WD Autocross | 55.196 | 56.771 | -1.746 | 1.0285 | 3.77 | 903.3 | 241.8 | 1231.9 | 5.44 | 1.077 |
| TV 4WD Efficiency | 57.909 | 57.970 | -0.548 | 1.0010 | 4.31 | 804.7 | 202.0 | 1257.9 | 4.85 | 1.000 |

## 3. Sensibilidades de Diseño (14-DOF 3D)

| Caso | QSS Δt [ms] | 14-DOF Δt [ms] | ΔE [%] | ay_max @ 22 m/s [m/s²] |
|---|---:|---:|---:|---:|
| Mass -15 kg (275 kg) | -488 | -72 | -4.79 | 18.08 |
| Mass +15 kg (305 kg) | +525 | +465 | +1.98 | 17.15 |
| Aero Cl*A -10% | +666 | +301 | -0.61 | 16.99 |
| Aero Cl*A +10% | -692 | +70 | -1.88 | 18.20 |
| Parametric Susp (no 3D) | -63 | -68 | +0.34 | 17.65 |

## 4. Benchmark Real FSG 2023 (Hockenheim)

- **`fsg_2023`**: $L_0 = 1183.62\text{ m}$ ($w = 3.60\text{ m}$, $R_{\min,0} = 4.23\text{ m}$, clearance mínimo $= 9.45\text{ m}$) -> Línea óptima $L_r = 1180.90\text{ m}$ ($-14.2\%$ en $\int \kappa^2 ds$, $R_{\min,r} = 5.08\text{ m}$).

| Controlador | QSS [s] | 14-DOF [s] | Δt vs Open [s] | v_max [km/h] | v_med [km/h] | E_net [kJ] | E_regen [kJ] | E_22km [kWh] | max|β| [°] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Open-Diff 4WD | 67.529 | 999.000 | +0.000 | 0.0 | 4.3 | 0.0 | 0.0 | 0.00 | 0.00 |
| TV 4WD Autocross | 72.384 | 76.503 | -922.497 | 102.8 | 55.6 | 938.3 | 399.7 | 4.86 | 4.90 |
| TV 4WD Efficiency | 76.261 | 78.880 | -920.120 | 90.2 | 53.9 | 775.6 | 313.8 | 4.01 | 6.60 |
