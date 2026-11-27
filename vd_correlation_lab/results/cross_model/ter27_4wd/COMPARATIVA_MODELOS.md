# Comparativa de Solvers — TeR27 (`ter27_4wd`)

- **Tiempo batch `full_14dof` (1000 Hz):** 6.66 s
- **Tiempo batch `fast_3dof` (200 Hz):** 4.43 s
- **Ganancia de velocidad (Speedup):** **1.50×**

## Métricas en Horizonte 10 s y Lazo Abierto

| Señal | `r` 14-DOF (H10s) | `r` 3-DOF (H10s) | Δr (H10s) | RMSE 14-DOF | RMSE 3-DOF | ΔRMSE | `r` 14-DOF (Open) | `r` 3-DOF (Open) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `vx` (m/s) | 0.9969 | 0.9618 | -0.0351 | 0.505 | 1.634 | +1.129 | 0.9988 | 0.8857 |
| `ax` (m/s²) | 0.9967 | 0.9882 | -0.0085 | 0.393 | 0.764 | +0.371 | 0.9962 | 0.9247 |
| `ay` (m/s²) | 0.9922 | 0.9861 | -0.0061 | 1.565 | 2.079 | +0.514 | 0.9924 | 0.9378 |
| `wz` (rad/s) | 0.9950 | 0.9885 | -0.0065 | 0.113 | 0.173 | +0.060 | 0.9959 | 0.7676 |

## Criterio de Selección Técnica

- **Usar `full_14dof` (1000 Hz)** para Dinámica Vehicular: ajuste de muelles, amortiguadores, anti-dive/anti-squat, alturas de *roll center* y análisis de vibraciones verticales.
- **Usar `fast_3dof` (200 Hz)** para Driverless (HIL / NMPC / ROS 2): mantiene `r > 0.90` en las señales planares principales eliminando el coste de los GDL verticales de suspensión.
