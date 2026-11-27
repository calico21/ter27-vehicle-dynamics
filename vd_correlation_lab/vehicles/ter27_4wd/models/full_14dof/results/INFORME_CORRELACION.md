# Informe de Correlación Sim-to-Real — TeR27 (`ter27_4wd`)

- **Etiqueta de modelo:** `sim_TeR27 (full_14dof)`
- **Origen de datos:** **Validación Sintética / Pre-pista**
- **Fecha de generación:** 2026-11-17 22:20
- **Tandas analizadas:** 1 (1 de pista, 0 de paddock lento)

> **Nota:** Aún no existen logs CAN de pista para este vehículo; la evaluación se ha ejecutado sobre maniobras sintéticas de referencia con ruido instrumental realista para verificar el pipeline completo (incluida la suite `06_4wd_y_torque_vectoring`).

## 1. Métricas Agrupadas (Tandas de Pista)

| Señal | Horizonte 10 s (`r`) | Horizonte 10 s (`R²`) | Horizonte 10 s (`RMSE`) | Lazo Abierto (`r`) | Lazo Abierto (`R²`) | Lazo Abierto (`RMSE`) | Lag óptimo (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Velocidad `vx` (m/s) | 0.9969 | 0.9914 | 0.505 | 0.9988 | 0.9975 | 0.273 | 0.0 |
| Acel. longitudinal `ax` (m/s²) | 0.9967 | 0.9933 | 0.393 | 0.9962 | 0.9923 | 0.419 | 0.0 |
| Acel. lateral `ay` (m/s²) | 0.9922 | 0.9840 | 1.565 | 0.9924 | 0.9848 | 1.525 | -30.0 |
| Guiñada `wz` (rad/s) | 0.9950 | 0.9900 | 0.113 | 0.9959 | 0.9914 | 0.104 | 0.0 |
| RPM rueda RL | 0.9983 | 0.9952 | 24.789 | 0.9995 | 0.9987 | 12.633 | 0.0 |
| RPM rueda RR | 0.9983 | 0.9947 | 22.299 | 0.9994 | 0.9986 | 11.312 | 0.0 |

## 2. Subviraje y Consistencia por Fase de Curva

- **Gradiente de subviraje `Kus` Real:** -2.03 deg/g
- **Gradiente de subviraje `Kus` sim_TeR27 (full_14dof):** -2.03 deg/g

| Fase de curva | Muestras | RMSE `ay` (m/s²) | RMSE `ax` (m/s²) | RMSE `wz` (rad/s) |
| :--- | :---: | :---: | :---: | :---: |
| Entrada (Frenada + Turn-in) | 6 | N/A | N/A | N/A |
| Vértice (Apoyo) | 1 | N/A | N/A | N/A |
| Salida (Tracción) | 1771 | 1.786 | 0.428 | 0.119 |

## 3. Desglose por Tanda

| Tanda | Tipo | Duración (s) | EMI filtrados | Offset volante (°) | `r(ay)` Horiz. 10s | `r(wz)` Horiz. 10s | `r(vx)` Lazo Abierto |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ter27_4wd_synthetic_track` | `track` | 40.0 | 0 | 0.00 | 0.9922 | 0.9950 | 0.9988 |

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
