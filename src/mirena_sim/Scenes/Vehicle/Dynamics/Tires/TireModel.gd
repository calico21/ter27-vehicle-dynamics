class_name TireModel
extends RefCounted

## High-Performance Hybrid Tire Model for Mirena Dynamics
## Integrates:
## 1. Pacejka Magic Formula 6.2 (MF 6.2) calibrated against Calspan TTC Hoosier 16x7.5-10 data (schema v2).
## 2. Physics-Informed Neural Network (PINN) Residual Correction (learned residuals Delta Fy, Delta Mz).
## 3. Transient Carcass Relaxation Dynamics (Pacejka Ch. 7 relaxation lengths with standstill damping).
##
## Combined slip attenuation: Delta Fy and Delta Mz are properly attenuated through Gyk
## so neural residuals do not violate friction ellipse limits during wheelspin or lock-up.

const VehicleConfig = preload("../Core/VehicleConfig.gd")
const Pacejka62Script = preload("MF62/Pacejka62.gd")
const NeuralCorrectionScript = preload("NeuralCorrection/NeuralTireCorrection.gd")
const TransientRelaxationScript = preload("Transient/TransientRelaxation.gd")

static var _mf62: Pacejka62 = null
static var _neural: NeuralTireCorrection = null

## Flag to enable/disable neural residual correction at runtime (e.g., for benchmarking).
## Default false: start closed-loop vehicle integration with deterministic pure MF 6.2.
static var enable_neural_correction: bool = false

## Flag to enable/disable transient relaxation integration.
static var enable_transient_relaxation: bool = true

static func get_mf62() -> Pacejka62:
	if not _mf62:
		_mf62 = Pacejka62Script.new()
	return _mf62

static func get_neural() -> NeuralTireCorrection:
	if not _neural:
		_neural = NeuralCorrectionScript.new(get_mf62())
	return _neural

## Safely configures tire compound ("R25B" or "R20"). Fails explicitly if file is missing or mismatch.
static func set_compound(compound_name: String) -> bool:
	var success: bool = get_mf62().load_compound(compound_name)
	if success:
		get_neural().load_for(get_mf62())
	return success

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
##   thermal_mu_scale: tread-temperature friction multiplier (1.0 in optimal 55-72 C window)
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
	is_right_side: bool = false,
	thermal_mu_scale: float = 1.0
) -> Dictionary:
	var normal_load = maxf(0.0, F_z)
	if normal_load <= 1e-3:
		return {
			"Fx": 0.0, "Fy": 0.0, "Mz": 0.0, "Mx": 0.0, "My": 0.0,
			"delta_Fx": 0.0, "delta_Fy": 0.0, "delta_Mz": 0.0,
			"Fx_mf": 0.0, "Fy_mf": 0.0, "Mz_mf": 0.0
		}
	
	var mf = get_mf62()
	var prev_lmux: float = mf.LMUX
	var prev_lmuy: float = mf.LMUY
	var therm_scale: float = clampf(thermal_mu_scale, 0.5, 1.1)
	if config != null:
		mf.LMUX = clampf(config.lambda_mux, 0.1, 1.5) * therm_scale
		mf.LMUY = clampf(config.lambda_muy, 0.1, 1.5) * therm_scale
	else:
		mf.LMUX = therm_scale
		mf.LMUY = therm_scale
	var res_vec: Vector2 = Vector2.ZERO
	if use_neural and enable_neural_correction:
		var nn = get_neural()
		if nn and nn.is_ready:
			res_vec = nn.predict_residual(alpha, normal_load, gamma, p_kpa, is_right_side)
	
	# Canonical evaluation with proper combined slip attenuation on the residual
	var forces = mf.compute_all(kappa, alpha, normal_load, gamma, p_kpa, v_x, is_right_side, res_vec)
	var comb = mf.compute_combined_forces(kappa, alpha, normal_load, gamma, p_kpa, is_right_side)
	mf.LMUX = prev_lmux
	mf.LMUY = prev_lmuy
	var gyk: float = comb["Gyk"]
	var delta_fy_eff: float = gyk * res_vec.x
	var delta_mz_eff: float = gyk * res_vec.y
	
	return {
		"Fx": forces["Fx"],
		"Fy": forces["Fy"],
		"Mz": forces["Mz"],
		"Mx": forces["Mx"],
		"My": forces["My"],
		"delta_Fx": 0.0,
		"delta_Fy": delta_fy_eff,
		"delta_Mz": delta_mz_eff,
		"Fx_mf": comb["Fx"],
		"Fy_mf": comb["Fy"],
		"Mz_mf": forces["Mz"] - delta_mz_eff
	}
