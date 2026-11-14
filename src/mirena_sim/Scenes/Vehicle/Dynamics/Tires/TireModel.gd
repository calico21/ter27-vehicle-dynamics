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
const TeRTireNeuralResidualScript = preload("NeuralCorrection/TeRTireNeuralResidual.gd")
const TransientRelaxationScript = preload("Transient/TransientRelaxation.gd")

static var _mf62: Pacejka62 = null
static var _neural: NeuralTireCorrection = null
static var _ter_pinn: TeRTireNeuralResidualScript = null

## Flag to enable/disable neural residual correction at runtime (e.g., for benchmarking).
## Default false: start closed-loop vehicle integration with deterministic pure MF 6.2.
static var enable_neural_correction: bool = false

## Flag to retain legacy NeuralTireCorrection (4-16-16-2) alongside TeRTireNeuralResidual for TTC parity.
static var enable_legacy_neural_correction: bool = true

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

static func get_ter_pinn() -> TeRTireNeuralResidualScript:
	if not _ter_pinn:
		_ter_pinn = TeRTireNeuralResidualScript.new()
	return _ter_pinn

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
##   t_surf_c: tread surface temperature (C) passed to TeRTireNeuralResidual
##   surface_mu_scale: road-surface friction multiplier (clamped to [0.2, 1.5])
## Returns Dictionary with Fx, Fy, Mz, Mx, My, and residual diagnostics.
static func compute_forces(
	kappa: float,
	alpha: float,
	F_z: float,
	config: VehicleConfig = null,
	gamma: float = 0.0,
	p_kpa: float = 83.0,
	v_x: float = 10.0,
	use_neural: bool = false,
	is_right_side: bool = false,
	thermal_mu_scale: float = 1.0,
	t_surf_c: float = 58.0,
	surface_mu_scale: float = 1.0
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
	var surf_scale: float = clampf(surface_mu_scale, 0.2, 1.5)
	var mu_scale: float = therm_scale * surf_scale
	if config != null:
		mf.LMUX = clampf(config.lambda_mux, 0.1, 1.5) * mu_scale
		mf.LMUY = clampf(config.lambda_muy, 0.1, 1.5) * mu_scale
	else:
		mf.LMUX = mu_scale
		mf.LMUY = mu_scale
	var res_vec: Vector2 = Vector2.ZERO
	var ter_res: Vector3 = Vector3.ZERO
	if use_neural or enable_neural_correction:
		if enable_legacy_neural_correction:
			var nn = get_neural()
			if nn and nn.is_ready:
				res_vec = nn.predict_residual(alpha, normal_load, gamma, p_kpa, is_right_side)
		var ter_pinn := get_ter_pinn()
		if ter_pinn != null and ter_pinn.is_ready:
			ter_res = ter_pinn.predict_residual(kappa, alpha, normal_load, gamma, t_surf_c, is_right_side)
	
	# Canonical evaluation with proper combined slip attenuation on the legacy residual
	var forces = mf.compute_all(kappa, alpha, normal_load, gamma, p_kpa, v_x, is_right_side, res_vec)
	var comb = mf.compute_combined_forces(kappa, alpha, normal_load, gamma, p_kpa, is_right_side)
	mf.LMUX = prev_lmux
	mf.LMUY = prev_lmuy
	var gyk: float = comb["Gyk"]
	var fx_mf: float = float(comb["Fx"])
	var fy_mf: float = float(comb["Fy"])
	var mz_mf: float = float(forces["Mz"]) - gyk * res_vec.y

	var fx_out: float = float(forces["Fx"])
	var fy_out: float = float(forces["Fy"])
	var mz_out: float = float(forces["Mz"])
	var delta_fx_eff: float = 0.0
	var delta_fy_eff: float = gyk * res_vec.x
	var delta_mz_eff: float = gyk * res_vec.y

	if ter_res != Vector3.ZERO:
		var f_env: float = normal_load * 3.2 * maxf(mu_scale, 0.2)
		var mz_env: float = normal_load * mf.r0 * maxf(mu_scale, 0.2)
		fx_out = clampf(fx_out + ter_res.x, -f_env, f_env)
		fy_out = clampf(fy_out + ter_res.y, -f_env, f_env)
		mz_out = clampf(mz_out + ter_res.z, -mz_env, mz_env)
		delta_fx_eff = fx_out - fx_mf
		delta_fy_eff = fy_out - fy_mf
		delta_mz_eff = mz_out - mz_mf
	
	return {
		"Fx": fx_out,
		"Fy": fy_out,
		"Mz": mz_out,
		"Mx": forces["Mx"],
		"My": forces["My"],
		"delta_Fx": delta_fx_eff,
		"delta_Fy": delta_fy_eff,
		"delta_Mz": delta_mz_eff,
		"Fx_mf": fx_mf,
		"Fy_mf": fy_mf,
		"Mz_mf": mz_mf
	}
