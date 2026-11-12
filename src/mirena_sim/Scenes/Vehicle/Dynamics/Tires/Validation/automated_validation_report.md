# Mirena Dynamics — Automated Tire Model Validation Report

**Generated:** 2026-11-12  
**Tire Specifications:** Hoosier 16.0x7.5-10, 8.0-inch rim (TTC Round 8 R25B & Round 9 R20 Cornering) + TTC Round 9 R20 Drive/Brake (18.0x6.0-10, 6-in & 7-in rims)  
**Models Evaluated:**
1. **Model 1 (Baseline MF 6.2):** Historical cornering prior with literature-default longitudinal & combined-slip coefficients.
2. **Model 2 (Unified Calibrated MF 6.2):** Canonical schema v2 MF 6.2 combining TTC Cornering/Mz/Mx calibration with TTC Round 9 Drive/Brake pure longitudinal (`PDX1..3, PKX1..2, PEX1, PPX1/3/4`) and combined-slip (`RBX1..2, RBY1..2, RHY1`) calibration (`R25B` uses `R20` Drive/Brake priors as documented surrogate).
3. **Model 3 (Hybrid MF 6.2 + PINN):** Model 2 augmented with the 16x16 Tanh Physics-Informed Neural Network (`Gyk`-attenuated under combined slip).

---

## 1. Executive Summary & Status

| Compound | Metric | Model 1: Baseline | Model 2: Unified MF | Model 3: Hybrid PINN | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Hoosier R25B** | **$F_y$ Val RMSE (Global Cornering)** | 183.47 N | **120.58 N (+34.3%)** | **109.48 N (+40.3%)** | **BENCHMARK PASS** |
| | **$F_y$ High Camber (4°) RMSE** | 264.58 N | **113.47 N** | **101.38 N** | **RESOLVED** |
| | **$M_z$ Val RMSE (Global Cornering)** | 5.94 Nm | 5.58 Nm | **4.64 Nm** | **PASS** |
| | **$F_x$ Pure Val RMSE *(R20 Surrogate)*** | 172.02 N | **119.87 N (+30.3%)** | *(MF6.2 $F_x$)* | **SURROGATE INTEGRATED** |
| | **Combined $G_{y\kappa} F_y$ RMSE *(R20 Surrogate)*** | 146.89 N | **97.60 N (+33.6%)** | *(PINN $G_{y\kappa}$ scaled)* | **SURROGATE INTEGRATED** |
| **Hoosier R20** | **$F_y$ Val RMSE (Global Cornering)** | 137.68 N | **93.11 N (+32.4%)** | **88.96 N (+35.4%)** | **BENCHMARK PASS** |
| | **$F_y$ High Camber (4°) RMSE** | 203.76 N | **101.04 N** | **97.80 N** | **RESOLVED** |
| | **$M_z$ Val RMSE (Global Cornering)** | 9.48 Nm | 9.34 Nm | **7.04 Nm** | **PASS** |
| | **$F_x$ Pure Val RMSE *(Round 9 D/B)*** | 172.02 N | **119.87 N (+30.3%)** | *(MF6.2 $F_x$)* | **VALIDATED (R20 18x6)** |
| | **Combined $F_x$ / $G_{y\kappa} F_y$ Val RMSE** | 171.9 / 146.9 N | **135.0 / 97.6 N** | *(PINN $G_{y\kappa}$ scaled)* | **VALIDATED ($\alpha\in\{-3,-6\}^\circ$)** |

---

## 2. Sample-Count Reconciliation & Data Provenance

> [!IMPORTANT]
> **Data Integrity Verification:**
> - **R25B Independent Validation (Run 7):** Exactly **52,294 retained rows** across 42 sweeps.
> - **R20 Independent Validation (Run 9):** Exactly **52,154 retained rows** across 42 sweeps.
> - The previously cited numbers **N = 10,088** (R25B) and **N = 9,476** (R20) represented a single nominal operating condition (nominal 40 km/h velocity at 83 kPa/56 kPa), whereas global metrics reported above are calculated across all 52k+ independent validation points.
> - Zero data leakage: Training (R25B Run 6, R20 Run 8) and Validation (R25B Run 7, R20 Run 9) are completely independent physical runs.

---

## 3. Condition-Specific Performance Breakdown (5 Physical Normal Loads)

### Hoosier R25B Normal Load Slices

| Load Level | Nominal Fz | Sample Count N | Baseline RMSE | Optimized MF RMSE | Hybrid PINN RMSE | Improvement |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **L1_Load_220N** | ~222 N | 9,962 | 52.94 N | 41.92 N | **36.30 N** | **+31.4%** |
| **L2_Load_440N** | ~442 N | 9,964 | 93.89 N | 62.46 N | **63.22 N** | **+32.7%** |
| **L3_Load_665N** | ~665 N | 9,962 | 164.35 N | 113.02 N | **96.39 N** | **+41.4%** |
| **L4_Load_885N** | ~887 N | 9,969 | 222.97 N | 146.99 N | **131.28 N** | **+41.1%** |
| **L5_Load_1110N** | ~1111 N | 12,437 | 265.97 N | 170.43 N | **157.74 N** | **+40.7%** |

---

## 4. Model Limitations & Unsupported Operating Domains

1. **Longitudinal ($F_x$) and Combined Slip ($F_x-F_y$) Geometry & Compound Scope:**
   - Pure longitudinal (`PDX1..3, PKX1..2, PEX1, PPX1/3/4`) and combined-slip (`RBX1..2, RBY1..2, RHY1`) parameters are calibrated against Calspan TTC Round 9 Drive/Brake data (`Hoosier 18.0×6.0-10 R20`, 6-in & 7-in rims, Runs `69, 70, 72, 73`, `kappa = SL` free-rolling referenced) after excluding warmup (`Runs 68, 71`), anomalies, and break-in sweeps.
   - Combined-slip experimental data covers only $\alpha \in \{-3^\circ, -6^\circ\}$ (no positive slip angles or $|\alpha| > 6^\circ$); sign/load/camber/`RVY*` terms remain provisional.
   - Because no R25B Drive/Brake TTC dataset exists in the repository, **Hoosier R25B** integrates the R20 Drive/Brake parameters as a documented surrogate; longitudinal parameters reflect the `18×6.0-10` carcass geometry rather than a dedicated `16×7.5-10` Drive/Brake test.
2. **Carcass Relaxation Dynamics:**
   - The canonical MF 6.2 equations are purely steady-state. Carcass lag ($\sigma = 180$ mm) is modeled dynamically in Godot via `TransientRelaxation.gd`.
3. **Airborne and Low-Load Extrapolation:**
   - Physical contact patch forces collapse smoothly to zero when $F_z < 100$ N via the airborne gate $g_z = \text{clamp}(F_z / 100, 0, 1)$.

---

## 5. Structured Figure Catalog

The validation suite generated figures into the organized directory hierarchy:
```text
Tire_Model_Validation/
├── 00_Overview/
├── 01_Experimental_Data/
├── 02_Pure_Lateral_Fy/
├── 03_Camber_Sensitivity/
├── 04_Aligning_Moment_Mz/
├── 05_Longitudinal_Fx/
├── 06_Combined_Slip/
├── 07_Other_Forces_Moments/
├── 08_Pressure_Sensitivity/
├── 09_MF62_Parameters/
├── 10_Residual_Statistics/
├── 11_Hybrid_MLP/
├── 12_Transient_Dynamics/
├── 13_Python_Godot_Parity/
├── 14_Vehicle_Dynamics_Relevance/
└── 15_Real_Telemetry_TeR26/
```
All figures are saved in high resolution (150 DPI) with embedded validation metrics boxes.

---

## 6. Real Track Telemetry Validation — TeR26 (RWD) (`15_Real_Telemetry_TeR26`)

- **Source:** 3 `candump` logs in `bags/real_bags/` (`N = 114,974` samples, `1149.7 s` with $v_x \ge 3\text{ m/s}$).
- **Vehicle Setup:** Explicit RWD (`front_drive_torque_ratio = 0.0`), $m = 290\text{ kg}$, $R_w = 0.2005\text{ m}$, SO(2) IMU yaw alignment.

| Compound | Variant | $F_y$ RMSE | $F_y$ MAE | $F_y$ $R^2$ | $\lambda_{\mu,\text{track}}$ | Lag (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **R25B** | TTC ($\lambda_\mu = 1.0$) | 2140.9 N | 1663.7 N | -0.435 | 1.000 | -300 |
| **R25B** | Asphalt (`0.78`) | 2060.7 N | 1678.9 N | -0.330 | 0.780 | -300 |
| **R25B** | Track Fit ($\lambda_{\mu,\text{opt}}$) | **1636.1 N** | **1397.4 N** | **0.162** | **0.352** | -300 |
| **R20** | TTC ($\lambda_\mu = 1.0$) | 2111.0 N | 1642.0 N | -0.396 | 1.000 | -300 |
| **R20** | Asphalt (`0.78`) | 2048.8 N | 1669.8 N | -0.315 | 0.780 | -300 |
| **R20** | Track Fit ($\lambda_{\mu,\text{opt}}$) | **1632.4 N** | **1396.3 N** | **0.165** | **0.362** | -300 |

- **Level B Open-Loop RWD Replay (`20261004T095356.855Z.log`, 35.0 s):** Yaw rate Pearson $r = 0.238$, RMSE $= 86.9^\circ/\text{s}$.
