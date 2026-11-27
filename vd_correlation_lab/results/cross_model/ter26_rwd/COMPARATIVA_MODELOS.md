# Comparativa de Solvers — TeR26 (`ter26_rwd`)

- **Tiempo batch `full_14dof` (1000 Hz):** 170.74 s
- **Tiempo batch `fast_3dof` (200 Hz):** 101.10 s
- **Ganancia de velocidad (Speedup):** **1.69×**

## Métricas en Horizonte 10 s y Lazo Abierto

| Señal | `r` 14-DOF (H10s) | `r` 3-DOF (H10s) | Δr (H10s) | RMSE 14-DOF | RMSE 3-DOF | ΔRMSE | `r` 14-DOF (Open) | `r` 3-DOF (Open) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `vx` (m/s) | 0.9672 | 0.9677 | +0.0005 | 0.816 | 0.810 | -0.006 | 0.8807 | 0.8874 |
| `ax` (m/s²) | 0.9199 | 0.9164 | -0.0035 | 1.143 | 1.168 | +0.025 | 0.9208 | 0.9169 |
| `ay` (m/s²) | 0.9521 | 0.9493 | -0.0028 | 2.053 | 2.136 | +0.083 | 0.9057 | 0.8996 |
| `wz` (rad/s) | 0.9656 | 0.9630 | -0.0026 | 0.213 | 0.223 | +0.011 | 0.9429 | 0.9374 |

## Criterio de Selección Técnica

- **Usar `full_14dof` (1000 Hz)** para Dinámica Vehicular: ajuste de muelles, amortiguadores, anti-dive/anti-squat, alturas de *roll center* y análisis de vibraciones verticales.
- **Usar `fast_3dof` (200 Hz)** para Driverless (HIL / NMPC / ROS 2): mantiene `r > 0.90` en las señales planares principales eliminando el coste de los GDL verticales de suspensión.
