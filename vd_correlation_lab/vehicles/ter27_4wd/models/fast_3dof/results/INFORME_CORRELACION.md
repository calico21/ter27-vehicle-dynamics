# Informe de Correlación Sim-to-Real — TeR27 (`ter27_4wd`)

- **Etiqueta de modelo:** `sim_TeR27 (fast_3dof)`
- **Origen de datos:** **Validación Sintética / Pre-pista**
- **Fecha de generación:** 2026-11-17 22:20
- **Tandas analizadas:** 1 (1 de pista, 0 de paddock lento)

> **Nota:** Aún no existen logs CAN de pista para este vehículo; la evaluación se ha ejecutado sobre maniobras sintéticas de referencia con ruido instrumental realista para verificar el pipeline completo (incluida la suite `06_4wd_y_torque_vectoring`).

## 1. Métricas Agrupadas (Tandas de Pista)

| Señal | Horizonte 10 s (`r`) | Horizonte 10 s (`R²`) | Horizonte 10 s (`RMSE`) | Lazo Abierto (`r`) | Lazo Abierto (`R²`) | Lazo Abierto (`RMSE`) | Lag óptimo (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Velocidad `vx` (m/s) | 0.9618 | 0.9103 | 1.634 | 0.8857 | 0.6756 | 3.107 | 30.0 |
| Acel. longitudinal `ax` (m/s²) | 0.9882 | 0.9745 | 0.764 | 0.9247 | 0.8515 | 1.843 | 0.0 |
| Acel. lateral `ay` (m/s²) | 0.9861 | 0.9717 | 2.079 | 0.9378 | 0.8710 | 4.441 | 10.0 |
| Guiñada `wz` (rad/s) | 0.9885 | 0.9765 | 0.173 | 0.7676 | 0.4062 | 0.869 | 0.0 |
| RPM rueda RL | 0.9838 | 0.9550 | 75.668 | 0.7834 | 0.4059 | 275.079 | 0.0 |
| RPM rueda RR | 0.9797 | 0.9478 | 70.287 | 0.9132 | 0.7766 | 145.374 | 0.0 |

## 2. Subviraje y Consistencia por Fase de Curva

- **Gradiente de subviraje `Kus` Real:** -2.03 deg/g
- **Gradiente de subviraje `Kus` sim_TeR27 (fast_3dof):** -2.34 deg/g

| Fase de curva | Muestras | RMSE `ay` (m/s²) | RMSE `ax` (m/s²) | RMSE `wz` (rad/s) |
| :--- | :---: | :---: | :---: | :---: |
| Entrada (Frenada + Turn-in) | 6 | N/A | N/A | N/A |
| Vértice (Apoyo) | 1 | N/A | N/A | N/A |
| Salida (Tracción) | 1771 | 4.951 | 0.844 | 0.967 |

## 3. Desglose por Tanda

| Tanda | Tipo | Duración (s) | EMI filtrados | Offset volante (°) | `r(ay)` Horiz. 10s | `r(wz)` Horiz. 10s | `r(vx)` Lazo Abierto |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ter27_4wd_synthetic_track` | `track` | 40.0 | 0 | 0.00 | 0.9861 | 0.9885 | 0.8857 |

## 4. Batería de Gráficas Generadas (300 DPI)

Las gráficas se encuentran en `plots/` organizadas en las siguientes carpetas:
- `01_dinamica_global/`: series temporales, resumen por tanda y evolución respecto a línea base.
- `02_lateral_y_guiniada/`: ganancia de guiñada, subviraje `Kus`, scatter `±1σ`, diagrama `g-g` y consistencia `ay vs vx·wz`.
- `03_longitudinal_y_ruedas/`: velocidades de rueda vs `vx`, slip ratios `kappa`, respuesta a pedales y ventanas de suelta/trail braking.
- `04_suspension_y_aero/`: gradientes de balanceo/cabeceo, cargas `Fz`, reparto lateral `LLTD`, velocidades de amortiguador y downforce.
- `05_neumaticos_y_envolvente/`: ángulos de deriva `alpha_f/alpha_r`, curvas de eje `Fy(alpha)` y utilización combinada de adherencia.
- `06_4wd_y_torque_vectoring/`: par en las 4 ruedas, momento activo `Mz` vs error de guiñada e intervención de control de tracción `ASR/TC`.
- `07_analisis_error_y_regimenes/`: mapas de calor 2D de error `(vx, |ay|)` y `(ax, ay)`, histogramas de residuos, barrido de lag y error por fase.
- `08_espectro_frecuencia/`: PSD/FFT `0-50 Hz`, función de transferencia y coherencia espectral `γ²(f)`.
