# Informe de Correlación Sim-to-Real — TeR26 (`ter26_rwd`)

- **Etiqueta de modelo:** `sim_TeR26 (full_14dof)`
- **Origen de datos:** **Validación en Pista (Logs CAN Reales)**
- **Fecha de generación:** 2026-11-17 22:17
- **Tandas analizadas:** 5 (3 de pista, 2 de paddock lento)

## 1. Métricas Agrupadas (Tandas de Pista)

| Señal | Horizonte 10 s (`r`) | Horizonte 10 s (`R²`) | Horizonte 10 s (`RMSE`) | Lazo Abierto (`r`) | Lazo Abierto (`R²`) | Lazo Abierto (`RMSE`) | Lag óptimo (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Velocidad `vx` (m/s) | 0.9672 | 0.9319 | 0.816 | 0.8807 | 0.6880 | 1.746 | -80.0 |
| Acel. longitudinal `ax` (m/s²) | 0.9199 | 0.8312 | 1.143 | 0.9208 | 0.8397 | 1.114 | 0.0 |
| Acel. lateral `ay` (m/s²) | 0.9521 | 0.9021 | 2.053 | 0.9057 | 0.8193 | 2.790 | 30.0 |
| Guiñada `wz` (rad/s) | 0.9656 | 0.9321 | 0.213 | 0.9429 | 0.8875 | 0.274 | -30.0 |
| RPM rueda RL | 0.9255 | 0.8566 | 56.260 | 0.8377 | 0.6239 | 91.118 | -130.0 |
| RPM rueda RR | 0.9756 | 0.9473 | 35.135 | 0.8901 | 0.7551 | 75.762 | 0.0 |

## 2. Subviraje y Consistencia por Fase de Curva

- **Gradiente de subviraje `Kus` Real:** -0.12 deg/g
- **Gradiente de subviraje `Kus` sim_TeR26 (full_14dof):** 1.06 deg/g

| Fase de curva | Muestras | RMSE `ay` (m/s²) | RMSE `ax` (m/s²) | RMSE `wz` (rad/s) |
| :--- | :---: | :---: | :---: | :---: |
| Entrada (Frenada + Turn-in) | 25769 | 2.764 | 1.133 | 0.251 |
| Vértice (Apoyo) | 21535 | 3.632 | 1.172 | 0.368 |
| Salida (Tracción) | 38174 | 3.137 | 1.249 | 0.318 |

## 3. Desglose por Tanda

| Tanda | Tipo | Duración (s) | EMI filtrados | Offset volante (°) | `r(ay)` Horiz. 10s | `r(wz)` Horiz. 10s | `r(vx)` Lazo Abierto |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `20261003T172622.377Z` | `track` | 394.7 | 373 | -2.05 | 0.9605 | 0.9801 | 0.8783 |
| `20261003T173334.167Z` | `paddock_slow` | 32.6 | 0 | -2.26 | N/A | N/A | N/A |
| `20261003T173417.162Z` | `paddock_slow` | 66.9 | 7 | -2.21 | 0.6131 | 0.9878 | 0.7078 |
| `20261004T093600.310Z` | `track` | 722.3 | 0 | -1.88 | 0.9529 | 0.9701 | 0.8875 |
| `20261004T095356.855Z` | `track` | 568.0 | 0 | -1.84 | 0.9502 | 0.9605 | 0.9142 |

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
