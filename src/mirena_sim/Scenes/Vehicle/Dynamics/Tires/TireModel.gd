class_name TireModel
extends RefCounted

## High-Performance Hybrid Tire Model for Mirena Dynamics
## Integrates:
## 1. Pacejka Magic Formula 6.2 (MF 6.2) calibrated against Calspan TTC Hoosier 16x7.5-10 data.
## 2. Physics-Informed Neural Network (PINN) Residual Correction (learned residuals Delta Fy, Delta Mz).
## 3. Transient Carcass Relaxation Dynamics (Pacejka Ch. 7 relaxation lengths with standstill damping).

const VehicleConfig = preload("../Core/VehicleConfig.gd")
const Pacejka62Script = preload("MF62/Pacejka62.gd")
const NeuralCorrectionScript = preload("NeuralCorrection/NeuralTireCorrection.gd")
const TransientRelaxationScript = preload("Transient/TransientRelaxation.gd")

static var _mf62: RefCounted = null
static var _neural: RefCounted = null

## Flag to enable/disable neural residual correction at runtime (e.g., for benchmarking).
static var enable_neural_correction: bool = true

## Flag to enable/disable transient relaxation integration.
static var enable_transient_relaxation: bool = true

static func get_mf62() -> RefCounted:
	if not _mf62:
		_mf62 = Pacejka62Script.new()
	return _mf62

static func get_neural() -> RefCounted:
	if not _neural:
		_neural = NeuralCorrectionScript.new()
	return _neural

## Safely configures tire compound ("R25B" or "R20"). Fails explicitly if file is missing or mismatch.
static func set_compound(compound_name: String) -> bool:
	return get_mf62().load_compound(compound_name)

## Returns current active tire compound identifier.
static func get_compound() -> String:
	return get_mf62().get_compound()

## Primary tire force evaluation interface.
## Parameters:
##   kappa: longitudinal slip ratio
##   alpha: lateral slip angle (rad)
##   F_z: vertical normal load (N)
##   config: VehicleConfig parameters
##   gamma: inclination / camber angle (rad) (default 0.0)
##   p_kpa: inflation pressure (kPa) (default 83.0 kPa ~ 12 psi)
##   v_x: longitudinal contact speed (m/s) (default 10.0)
##   use_neural: toggle neural residual addition
##   is_right_side: whether wheel is mounted on right vehicle track (mirrors conicity/ply-steer)
## Returns Dictionary with Fx, Fy, Mz, Mx, My, and residual diagnostics.
static func compute_forces(
	kappa: float,
	alpha: float,
	F_z: float,
	config: VehicleConfig = null,
	gamma: float = 0.0,
	p_kpa: float = 83.0,
	v_x: float = 10.0,
	use_neural: bool = true,
	is_right_side: bool = false
) -> Dictionary:
	var normal_load = maxf(0.0, F_z)
	if normal_load <= 1e-3:
		return {
			"Fx": 0.0, "Fy": 0.0, "Mz": 0.0, "Mx": 0.0, "My": 0.0,
			"delta_Fx": 0.0, "delta_Fy": 0.0, "delta_Mz": 0.0,
			"Fx_mf": 0.0, "Fy_mf": 0.0, "Mz_mf": 0.0
		}
	
	var mf = get_mf62()
	
	# 1. Canonical MF 6.2 Evaluation
	var combined = mf.compute_combined_forces(kappa, alpha, normal_load, gamma, p_kpa, is_right_side)
	var Fx_mf: float = combined["Fx"]
	var Fy_mf: float = combined["Fy"]
	var Svy: float = combined.get("Svy", 0.0)
	var Mz_mf: float = mf.compute_mz(alpha, normal_load, Fy_mf, gamma, p_kpa, Fx_mf, Svy, is_right_side)
	var Mx: float = mf.compute_mx(normal_load, gamma, Fy_mf, p_kpa, is_right_side)
	var My: float = mf.compute_my(normal_load, Fx_mf, v_x)
	
	# 2. Physics-Informed Residual PINN Correction
	var delta_Fx: float = 0.0
	var delta_Fy: float = 0.0
	var delta_Mz: float = 0.0
	
	if use_neural and enable_neural_correction:
		var nn = get_neural()
		var corr = nn.predict_correction(alpha, normal_load, gamma, p_kpa)
		delta_Fy = corr["delta_Fy"]
		delta_Mz = corr["delta_Mz"]
		delta_Fx = 0.0
	
	# 3. Hybrid Synthesis: F_total = F_MF + Delta_F
	var Fx_total = Fx_mf + delta_Fx
	var Fy_total = Fy_mf + delta_Fy
	var Mz_total = Mz_mf + delta_Mz
	
	return {
		"Fx": Fx_total,
		"Fy": Fy_total,
		"Mz": Mz_total,
		"Mx": Mx,
		"My": My,
		"delta_Fx": delta_Fx,
		"delta_Fy": delta_Fy,
		"delta_Mz": delta_Mz,
		"Fx_mf": Fx_mf,
		"Fy_mf": Fy_mf,
		"Mz_mf": Mz_mf
	}
