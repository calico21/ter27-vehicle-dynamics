# Informe de Correlación Sim-to-Real — TeR26 (`ter26_rwd`)

- **Etiqueta de modelo:** `sim_TeR26 (fast_3dof)`
- **Origen de datos:** **Validación en Pista (Logs CAN Reales)**
- **Fecha de generación:** 2026-11-17 22:49
- **Tandas analizadas:** 5 (3 de pista, 2 de paddock lento)

## 1. Métricas Agrupadas (Tandas de Pista)

| Señal | Horizonte 10 s (`r`) | Horizonte 10 s (`R²`) | Horizonte 10 s (`RMSE`) | Lazo Abierto (`r`) | Lazo Abierto (`R²`) | Lazo Abierto (`RMSE`) | Lag óptimo (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Velocidad `vx` (m/s) | 0.9677 | 0.9328 | 0.810 | 0.8874 | 0.6869 | 1.749 | -90.0 |
| Acel. longitudinal `ax` (m/s²) | 0.9164 | 0.8237 | 1.168 | 0.9169 | 0.8326 | 1.138 | 0.0 |
| Acel. lateral `ay` (m/s²) | 0.9493 | 0.8940 | 2.136 | 0.8996 | 0.8075 | 2.880 | 30.0 |
| Guiñada `wz` (rad/s) | 0.9630 | 0.9250 | 0.223 | 0.9374 | 0.8786 | 0.284 | -20.0 |
| RPM rueda RL | 0.9259 | 0.8573 | 56.133 | 0.8437 | 0.6239 | 91.116 | -120.0 |
| RPM rueda RR | 0.9731 | 0.9424 | 36.740 | 0.8891 | 0.7479 | 76.861 | 0.0 |

## 2. Subviraje y Consistencia por Fase de Curva

- **Gradiente de subviraje `Kus` Real:** -0.12 deg/g
- **Gradiente de subviraje `Kus` sim_TeR26 (fast_3dof):** 0.46 deg/g

| Fase de curva | Muestras | RMSE `ay` (m/s²) | RMSE `ax` (m/s²) | RMSE `wz` (rad/s) |
| :--- | :---: | :---: | :---: | :---: |
| Entrada (Frenada + Turn-in) | 25769 | 2.871 | 1.142 | 0.258 |
| Vértice (Apoyo) | 21535 | 3.801 | 1.226 | 0.391 |
| Salida (Tracción) | 38174 | 3.201 | 1.277 | 0.326 |

## 3. Desglose por Tanda

| Tanda | Tipo | Duración (s) | EMI filtrados | Offset volante (°) | `r(ay)` Horiz. 10s | `r(wz)` Horiz. 10s | `r(vx)` Lazo Abierto |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `20261003T172622.377Z` | `track` | 394.7 | 373 | -2.05 | 0.9584 | 0.9828 | 0.8823 |
| `20261003T173334.167Z` | `paddock_slow` | 32.6 | 0 | -2.26 | N/A | N/A | N/A |
| `20261003T173417.162Z` | `paddock_slow` | 66.9 | 7 | -2.21 | 0.6078 | 0.9880 | 0.7085 |
| `20261004T093600.310Z` | `track` | 722.3 | 0 | -1.88 | 0.9506 | 0.9679 | 0.8905 |
| `20261004T095356.855Z` | `track` | 568.0 | 0 | -1.84 | 0.9472 | 0.9570 | 0.9185 |

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
