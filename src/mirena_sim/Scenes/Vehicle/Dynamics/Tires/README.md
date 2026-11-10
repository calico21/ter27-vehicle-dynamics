# Mirena Dynamics — Tire Simulation Subsystem (MF 6.2 + Hybrid PINN)

This directory contains the production tire modeling framework for **Mirena Dynamics**, developed for high-fidelity Formula Student vehicle dynamics simulation, steady-state sensitivity analysis, lap-time simulation, and real-time interactive driving in Godot 4.

The architecture combines a canonical **Pacejka Magic Formula 6.2 (MF 6.2)** semi-empirical formulation, a **Physics-Informed Neural Network (PINN)** residual correction module, and a **First-Order Transient Carcass Relaxation** engine.

---

## 1. Directory Structure

```
Tires/
├── README.md                           # This architecture and operational manual
├── TireModel.gd                        # Top-level singleton facade orchestrating MF6.2, PINN, and transients
├── WheelDynamics.gd                    # Wheel rotational ODEs, brake/drive torques, contact patch kinematics
│
├── MF62/                               # Canonical Pacejka Magic Formula 6.2 Engine
│   ├── Pacejka62.gd                    # 64-bit float MF 6.2 evaluation engine in pure GDScript
│   ├── hoosier_16x7.5_10_r20_mf62.json # Calibrated MF 6.2 parameters for Hoosier R20 (TTC Round 9)
│   ├── hoosier_16x7.5_10_r25b_mf62.json# Calibrated MF 6.2 parameters for Hoosier R25B (TTC Round 8)
│   ├── baseline_hoosier_*.json         # Historical baseline parameter files (pre-calibration priors)
│   └── hoosier_16x7.5_10_mf62.json     # Default parameter pointer (R25B symlink/fallback)
│
├── NeuralCorrection/                   # Physics-Informed Residual Correction (PINN)
│   ├── NeuralTireCorrection.gd         # Fast GDScript MLP inference engine (Forward pass)
│   ├── neural_residual_weights_r20.json# Quantized/exported MLP weights for R20
│   └── neural_residual_weights_r25b.json# Quantized/exported MLP weights for R25B
│
├── Transient/                          # Carcass Deflection & Dynamic Slip
│   └── TransientRelaxation.gd          # First-order carcass relaxation equations with zero-speed damping
│
├── Training/                           # Calibration & Identification Pipelines (Python)
│   ├── preprocess_ttc.py               # Calspan TTC raw MAT/ASCII ingestion, filtering, and normalization
│   ├── calibrate_mf62.py               # Hierarchical SLSQP/L-BFGS-B MF 6.2 optimizer
│   └── train_residual_mlp.py           # PyTorch residual neural network trainer
│
├── Validation/                         # 15-Module Scientific Engineering Validation Suite
│   ├── run_validation_framework.py     # Orchestrator for all validation modules and automated report generation
│   ├── generate_validation_report_and_plots.py # Automated generation of figures, LaTeX tables, and JSON
│   ├── core/                           # Data loading, evaluation models, and metric aggregators
│   ├── modules/                        # Mod01 through Mod15 specialized analysis modules
│   ├── tests/                          # Parity, regression, and unit test scripts
│   ├── Plots/                          # Generated high-resolution diagnostic plots
│   └── Tire_Model_Validation/          # Complete standalone validation report package (PNGs + Reports)
│
└── Data/                               # Preprocessed experimental TTC datasets (HDF5 / CSV)
```

---

## 2. Coordinate Systems & Sign Conventions

The tire subsystem strictly follows the standard **ISO 8855** vehicle coordinate system.

```
          +Z (Upward)
           |
           |      +X (Forward)
           |     /
           |   /
           | /
           +------------ +Y (Vehicle Left)
```

### Sign Conventions

| Quantity | Symbol | Unit | Positive Direction / Definition |
| :--- | :--- | :--- | :--- |
| **Normal Load** | $F_z$ | $\text{N}$ | Compressive load acting from road onto tire contact patch ($F_z > 0$). |
| **Lateral Force** | $F_y$ | $\text{N}$ | Directed along vehicle $+Y$ (to the left). In ISO 8855, a positive slip angle $\alpha > 0$ generates a **negative lateral force** ($F_y < 0$). |
| **Longitudinal Force** | $F_x$ | $\text{N}$ | Directed along vehicle $+X$ (forward traction $>0$, braking $<0$). |
| **Slip Angle** | $\alpha$ | $\text{rad}$ / $\text{deg}$ | $\alpha = \arctan\left(\frac{v_y}{\lvert v_x \rvert}\right)$. When steering to the left, $v_y > 0 \implies \alpha > 0 \implies F_y < 0$. |
| **Camber / Inclination** | $\gamma$ | $\text{rad}$ / $\text{deg}$ | Wheel plane angle relative to vertical. Positive when top of tire leans toward vehicle $+Y$ (left). |
| **Longitudinal Slip** | $\kappa$ | $-$ | $\kappa = \frac{\omega R_e - v_x}{\lvert v_x \rvert}$ (dimensionless ratio; acceleration $>0$, braking $<0$). |
| **Self-Aligning Torque** | $M_z$ | $\text{N}\cdot\text{m}$ | Moment around $+Z$ axis. Acts to restore the steered wheel back toward zero slip angle. |
| **Overturning Moment** | $M_x$ | $\text{N}\cdot\text{m}$ | Moment around $+X$ axis induced by lateral force and vertical load asymmetry across contact patch. |
| **Rolling Resistance** | $M_y$ | $\text{N}\cdot\text{m}$ | Moment opposing wheel rotation around $+Y$ axis. |

> **Track Asymmetry Flag (`is_right_side`):**  
> Calspan TTC measurements exhibit structural conicity and plysteer offsets ($PVY1, PVY2$). On the vehicle, mounting the tire on the right-hand side mirrors the belt orientation relative to the chassis centerline. When `is_right_side = true`, the signs of structural lateral and aligning torque offsets are inverted to preserve chassis yaw symmetry.

---

## 3. Supported Tire Compounds

Two compounds from the **Formula SAE Tire Test Consortium (FSAE TTC)** are currently supported and calibrated:

### 1. Hoosier 16.0 × 7.5 - 10 R25B (Round 8)
- **Characteristics:** Ultra-soft, fast grip build-up, higher peak lateral friction coefficient ($\mu_{y,\text{peak}} \approx 2.50$ at $F_z = 887\text{ N}$).
- **Target Use:** Autocross, Sprint, low-temperature track conditions, qualifying laps.
- **Sensitivity:** Narrower thermal operating window ($60^\circ\text{C}$ to $70^\circ\text{C}$), steeper grip fall-off past peak slip angle.
- **Parameter File:** `MF62/hoosier_16x7.5_10_r25b_mf62.json`

### 2. Hoosier 16.0 × 7.5 - 10 R20 (Round 9)
- **Characteristics:** Medium compound, robust thermal degradation resistance, progressive breakaway ($\mu_{y,\text{peak}} \approx 2.38$ at $F_z = 887\text{ N}$).
- **Target Use:** Endurance events, high ambient temperature, heavy tire wear circuits.
- **Sensitivity:** Wider plateau at saturation ($\alpha \in [7^\circ, 12^\circ]$), lower thermal sensitivity.
- **Parameter File:** `MF62/hoosier_16x7.5_10_r20_mf62.json`

---

## 4. Mathematical Formulation (MF 6.2 Engine)

The core physics engine [`Pacejka62.gd`](file:///home/alexr/eRacing/vehicle-dynamics/src/mirena_sim/Scenes/Vehicle/Dynamics/Tires/MF62/Pacejka62.gd) implements the canonical Besselink (2010) / Pacejka (2012) Magic Formula 6.2 equations.

### Normalized Variables
$$\text{d}f_z = \frac{F_z - F_{z0}'}{F_{z0}'}, \quad \text{d}p_i = \frac{p_i - p_{i0}}{p_{i0}}$$
where $F_{z0}' = F_{z0} \cdot \lambda_{Fz0}$.

### Pure Lateral Force ($F_{y0}$)
$$F_{y0} = D_y \sin\left( C_y \arctan\left( B_y \alpha_y - E_y (B_y \alpha_y - \arctan(B_y \alpha_y)) \right) \right) + S_{vy}$$

- **Shape Factor:** $C_y = PCY1 \cdot \lambda_{Cy}$ (Fixed to $1.55$ during identification to guarantee mathematical identifiability).
- **Peak Friction Factor:** $D_y = \mu_y \cdot F_z \cdot \zeta_2$, where:
  $$\mu_y = (PDY1 + PDY2 \cdot \text{d}f_z) \cdot (1 - PDY3 \cdot \gamma^2) \cdot (1 + PPY3 \cdot \text{d}p_i + PPY4 \cdot \text{d}p_i^2) \cdot \lambda_{\mu y}$$
- **Cornering Stiffness ($K_{y\alpha0}$):**
  $$K_{y\alpha0} = PKY1 \cdot F_{z0}' \cdot (1 + PPY1 \cdot \text{d}p_i) \cdot \sin\left( 2 \arctan\left( \frac{F_z}{PKY2 \cdot F_{z0}' \cdot (1 + PPY2 \cdot \text{d}p_i)} \right) \right) \cdot \lambda_{Kya}$$
- **Stiffness Factor:** $B_y = \frac{K_{y\alpha0}}{C_y D_y}$
- **Curvature Factor:** $E_y = (PEY1 + PEY2 \cdot \text{d}f_z) \cdot \left(1 - (PEY3 + PEY4 \cdot \gamma) \text{sgn}(\alpha_y)\right) \cdot \lambda_{Ey}$ (Strictly constrained $E_y \le 1.0$ to ensure physical stability).
- **Horizontal & Vertical Shifts:**
  $$S_{hy} = (PHY1 + PHY2 \cdot \text{d}f_z) + \frac{PHY3 \cdot \gamma - K_{y\gamma0} \cdot \gamma}{K_{y\alpha0}} + \zeta_0 - 1$$
  $$S_{vy} = F_z \cdot \left( (PVY1 + PVY2 \cdot \text{d}f_z) + (PVY3 + PVY4 \cdot \text{d}f_z) \cdot \gamma \right) \cdot \lambda_{Vy} \cdot \zeta_2$$

### Self-Aligning Moment ($M_z$)
$$M_z = -t \cdot F_{y0}' + M_{zr} + s \cdot F_x$$
- **Pneumatic Trail ($t$):**
  $$t(\alpha_t) = D_t \cos\left( C_t \arctan\left( B_t \alpha_t - E_t (B_t \alpha_t - \arctan(B_t \alpha_t)) \right) \right) \cdot \cos'(\alpha)$$
- **Residual Torque ($M_{zr}$):**
  $$M_{zr}(\alpha_r) = D_r \cos\left( C_r \arctan(B_r \alpha_r) \right)$$
- **Longitudinal Moment Arm:** $s = \left(SSZ1 + SSZ2 \cdot \frac{F_y}{F_{z0}'} + (SSZ3 + SSZ4 \cdot \text{d}f_z) \cdot \gamma\right) \cdot R_0 \cdot \lambda_{s}$

### Combined Slip & Overturning Moment
- Lateral force combined slip weighting: $F_y = G_{yk} \cdot F_{y0} + S_{vyk}$.
- Longitudinal force combined slip weighting: $F_x = G_{x\alpha} \cdot F_{x0}$.
- Overturning moment: $M_x = R_0 \cdot F_z \cdot \left(QSX1 \cdot \lambda_{VMx} - QSX2 \cdot \gamma + QSX3 \cdot \frac{F_y}{F_{z0}'} + QSX4 \cdot \cos(QSX5 \cdot \arctan(QSX6 \cdot \frac{F_z}{F_{z0}'}))\right)$.

---

## 5. Hybrid Model: PINN Residual Correction

To capture complex intra-sweep thermal transitions and non-linear plysteer characteristics without destabilizing the analytical MF 6.2 model, an optional neural residual module is incorporated:

$$F_{y,\text{total}} = F_{y,\text{MF62}}(\alpha, F_z, \gamma, p) + \Delta F_{y,\text{PINN}}(\alpha, F_z, \gamma, p)$$
$$M_{z,\text{total}} = M_{z,\text{MF62}}(\alpha, F_z, \gamma, p) + \Delta M_{z,\text{PINN}}(\alpha, F_z, \gamma, p)$$

The PINN architecture is a compact Multi-Layer Perceptron (MLP) evaluated directly in GDScript via [`NeuralCorrection/NeuralTireCorrection.gd`](file:///home/alexr/eRacing/vehicle-dynamics/src/mirena_sim/Scenes/Vehicle/Dynamics/Tires/NeuralCorrection/NeuralTireCorrection.gd):
- **Inputs (Normalized):** $\left[\frac{\alpha}{\alpha_{\text{scale}}}, \frac{F_z}{F_{z,\text{scale}}}, \frac{\gamma}{\gamma_{\text{scale}}}, \frac{p}{p_{\text{scale}}}\right]$
- **Hidden Layers:** $2 \times 32$ neurons with LeakyReLU activations.
- **Outputs:** $[\Delta F_y, \Delta M_z]$ (clamped to physical safety envelopes $\pm 120\text{ N}$ and $\pm 5\text{ N}\cdot\text{m}$).
- **Inference Time:** $< 0.8\text{ }\mu\text{s}$ per call in GDScript.

---

## 6. Transient Carcass Relaxation

In dynamic driving maneuvers, the tire force does not respond instantaneously to changes in slip angle or wheel speed. The module [`Transient/TransientRelaxation.gd`](file:///home/alexr/eRacing/vehicle-dynamics/src/mirena_sim/Scenes/Vehicle/Dynamics/Tires/Transient/TransientRelaxation.gd) implements first-order relaxation lengths:

$$\frac{\text{d}u}{\text{d}t} + \frac{\lvert v_x \rvert + v_{\text{low}}}{\sigma_\kappa} u = -v_{sx}$$
$$\frac{\text{d}v}{\text{d}t} + \frac{\lvert v_x \rvert + v_{\text{low}}}{\sigma_\alpha} v = -\lvert v_x \rvert \tan(\alpha)$$

where:
- $\sigma_\alpha = \sigma_{\alpha 0} \cdot \left(1 - p_{\sigma 1} \frac{\Delta F_z}{F_{z0}}\right) \cdot \frac{F_z}{F_{z0}}$ is the lateral relaxation length ($\approx 0.12\text{ m}$ to $0.25\text{ m}$).
- $\sigma_\kappa$ is the longitudinal relaxation length ($\approx 0.10\text{ m}$ to $0.20\text{ m}$).
- $v_{\text{low}}$ provides numerical standstill damping to prevent stiff singularity at zero speed ($v_x \to 0$).

---

## 7. GDScript API Reference & Usage Guide

### Integration in Vehicle Setup

```gdscript
extends Node3D

func _ready() -> void:
    # 1. Select the active compound ("R20" or "R25B")
    var success = TireModel.set_compound("R25B")
    if not success:
        push_error("Failed to load tire compound R25B!")
        
    # 2. Configure model runtime options
    TireModel.enable_neural_correction = true
    TireModel.enable_transient_relaxation = true
```

### Computing Contact Patch Forces

To evaluate tire forces at any simulation physics tick ($60\text{ Hz}$ or $120\text{ Hz}$):

```gdscript
# Input state variables
var kappa: float = 0.05               # 5% longitudinal drive slip
var alpha: float = deg_to_rad(4.5)    # 4.5 degrees slip angle
var Fz: float = 950.0                 # 950 N normal load
var gamma: float = deg_to_rad(-1.5)   # -1.5 degrees negative camber
var pressure_kpa: float = 83.0        # 12 psi inflation pressure
var vx: float = 14.5                  # 14.5 m/s contact patch speed
var is_right_side: bool = false       # Left front wheel

# Compute forces
var forces: Dictionary = TireModel.compute_forces(
    kappa,
    alpha,
    Fz,
    null,           # Optional VehicleConfig reference
    gamma,
    pressure_kpa,
    vx,
    true,           # use_neural
    is_right_side
)

# Extract returned forces and moments
var Fx = forces["Fx"]         # Total longitudinal force (N)
var Fy = forces["Fy"]         # Total lateral force (N)
var Mz = forces["Mz"]         # Total self-aligning moment (N*m)
var Mx = forces["Mx"]         # Overturning moment (N*m)
var My = forces["My"]         # Rolling resistance moment (N*m)

# Optional diagnostic decomposition
var Fx_pure_mf = forces["Fx_mf"]
var Fy_pure_mf = forces["Fy_mf"]
var delta_Fy_nn = forces["delta_Fy"]
```

---

## 8. Calibration & Validation Workflow

### Ingestion & Filtering
To preprocess raw Calspan TTC tests from `.mat` files into filtered training datasets:
```bash
python3 src/mirena_sim/Scenes/Vehicle/Dynamics/Tires/Training/preprocess_ttc.py \
    --compound R25B \
    --round 8
```

### MF 6.2 Parameter Optimization
To calibrate the MF 6.2 parameters against the preprocessed data using multi-start bounded SLSQP:
```bash
python3 src/mirena_sim/Scenes/Vehicle/Dynamics/Tires/Training/calibrate_mf62.py \
    --compound R25B \
    --save-json src/mirena_sim/Scenes/Vehicle/Dynamics/Tires/MF62/hoosier_16x7.5_10_r25b_mf62.json
```

### Full Validation Suite Execution
To generate the complete 15-module engineering audit report and diagnostic figures:
```bash
python3 src/mirena_sim/Scenes/Vehicle/Dynamics/Tires/Validation/run_validation_framework.py
```
Generated reports are stored in:
- High-resolution plots: `Validation/Plots/`
- Aggregated JSON audit: `Validation/automated_validation_report.json`
- Comprehensive package: `Validation/Tire_Model_Validation.zip`

### Python / Godot Numerical Parity Verification
To verify machine-precision parity ($< 10^{-11}\text{ N}$) between Python reference equations and Godot GDScript:
```bash
godot --headless -s src/mirena_sim/tests/test_tire_mf62_suite.gd
```

---

## 9. Simulation Readiness & Operational Envelopes

As established in the authoritative technical audit (`final_mf62_simulation_readiness_audit.md`):

| Maneuver / Use Case | Status | Notes & Safe Envelope |
| :--- | :--- | :--- |
| **Steady-State Skidpad** | **APPROVED** | Validated across $F_z \in [250, 1600]\text{ N}$, $\alpha \in [-12^\circ, +12^\circ]$. Max lateral error $< 4.8\%$. |
| **Suspension Design / K&C** | **APPROVED** | Cornering stiffness $K_{y\alpha}$ validated across camber $\gamma \in [-4^\circ, +4^\circ]$ and pressures $55$–$97\text{ kPa}$. |
| **Yaw Dynamics / Slalom** | **APPROVED** | Aligning moment $M_z$ pneumatic trail $t \le 0.024\text{ m}$ correctly reproduces steering feedback torque. |
| **Combined Slip Cornering** | **APPROVED (WITH LIMITS)** | Validated using standard Pacejka weighting functions ($G_{yk}, G_{xk}$). High slip angles ($\lvert\alpha\rvert > 8^\circ$) with full traction ($\kappa > 0.15$) follow theoretical elliptic decay. |
| **Straight-Line Accel / Brake** | **PROVISIONAL** | Pure longitudinal parameters ($PKX1\text{--}3, PCX1$) are based on generic priors. Dedicated longitudinal TTC runs should be fitted when available. |
| **Thermal / High-Speed Transients** | **PROVISIONAL** | Model represents a mean bulk tire temperature of $62^\circ\text{C}$ to $68^\circ\text{C}$. For long runs ($> 10\text{ laps}$), link to Mirena Thermal Subsystem. |

For detailed mathematical proofs, Jacobian conditioning matrices, and standard error confidence intervals, refer to [final_mf62_simulation_readiness_audit.md](file:///home/alexr/.gemini/antigravity/brain/c5c97b01-25de-4bd1-b6e1-31c6c789a36b/final_mf62_simulation_readiness_audit.md).
