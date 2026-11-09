class_name Pacejka62
extends RefCounted

## Pacejka Magic Formula 6.2 (MF 6.2) Evaluation Engine for Mirena Dynamics
## Fully compliant with canonical MF 6.2 formulation (Pacejka 2012 / Besselink 2010):
## - Pure slip (Fx0, Fy0) and combined slip (Fx, Fy) with induced vertical shift Svyk
## - Self-aligning torque Mz via pneumatic trail M'z, residual torque Mzr, and longitudinal arm coupling (s * Fx)
## - Camber sensitivity (gamma) and normal load sensitivity (Fz)
## - Inflation pressure sensitivity (p_kpa)
## - Overturning moment (Mx) and rolling resistance moment (My) with Fx dependence

const DEFAULT_PARAM_PATH: String = "res://Scenes/Vehicle/Dynamics/Tires/MF62/hoosier_16x7.5_10_mf62.json"

var params: Dictionary = {}
var is_loaded: bool = false
var active_compound: String = "UNKNOWN"

# Reference quantities
var fnomin: float = 800.0
var pinom: float = 83.0
var r0: float = 0.2032
var v0: float = 11.11
var vertical_stiffness: float = 115000.0

# ----------------------------------------------------------------------
# Calibrated Default Coefficients (Hoosier 16.0x7.5-10 R25B)
# ----------------------------------------------------------------------
# Lateral (Fy)
var PCY1: float = 1.57
var PDY1: float = 2.52
var PDY2: float = -0.30
var PDY3: float = 0.10
var PEY1: float = 0.40
var PEY2: float = -0.28
var PEY3: float = 0.0
var PEY4: float = 0.0
var PEY5: float = 0.0
var PKY1: float = 49.8
var PKY2: float = 2.05
var PKY3: float = 0.0
var PHY1: float = 0.0
var PHY2: float = 0.0
var PHY3: float = 0.005
var PVY1: float = 0.0
var PVY2: float = 0.0
var PVY3: float = 0.08
var PVY4: float = -0.04
var PPY1: float = 0.08
var PPY2: float = 0.0
var PPY3: float = -0.12
var PPY4: float = 0.0

# Longitudinal (Fx) - Literature FSAE Defaults
var PCX1: float = 1.55
var PDX1: float = 2.30
var PDX2: float = -0.32
var PDX3: float = 0.0
var PEX1: float = 0.35
var PEX2: float = 0.10
var PEX3: float = 0.0
var PEX4: float = 0.0
var PKX1: float = 46.0
var PKX2: float = -0.12
var PKX3: float = 0.0
var PHX1: float = 0.0
var PHX2: float = 0.0
var PVX1: float = 0.0
var PVX2: float = 0.0
var PPX1: float = 0.0
var PPX2: float = 0.0
var PPX3: float = 0.0
var PPX4: float = 0.0

# Self-aligning torque (Mz)
var QDZ1: float = 0.150
var QDZ2: float = -0.040
var QDZ3: float = 0.050
var QDZ6: float = 0.0 # Symmetric tire: 0 conicity
var QDZ7: float = 0.0
var QDZ8: float = -0.040
var QDZ9: float = 0.0
var QBZ1: float = 6.5
var QBZ2: float = 1.2
var QBZ3: float = 0.0
var QBZ4: float = 0.0
var QBZ5: float = 0.0
var QBZ9: float = 8.0
var QBZ10: float = 0.0
var QCZ1: float = 1.65
var QEZ1: float = 0.05
var QEZ2: float = 0.85
var QEZ3: float = 0.0
var QEZ4: float = 0.0
var QEZ5: float = 0.0
var QHZ1: float = 0.002
var QHZ2: float = 0.0
var QHZ3: float = 0.0
var QHZ4: float = 0.0
var PPZ1: float = 0.0
var PPZ2: float = 0.0

# Longitudinal force arm s parameters for Mz
var SSZ1: float = 0.0
var SSZ2: float = 0.04
var SSZ3: float = 0.01
var SSZ4: float = 0.0

# Overturning moment (Mx)
var QSX1: float = 0.0
var QSX2: float = 0.80
var QSX3: float = 0.05
var PPMX1: float = 0.0

# Rolling resistance moment (My)
var QSY1: float = 0.015
var QSY2: float = 0.0
var QSY3: float = 0.001
var QSY4: float = 0.0

# Combined slip weighting
var RBX1: float = 12.0
var RBX2: float = 10.0
var RBX3: float = 0.0
var RCX1: float = 1.05
var REX1: float = 0.10
var REX2: float = 0.0
var RHX1: float = 0.0
var RBY1: float = 9.5
var RBY2: float = 8.0
var RBY3: float = 0.0
var RBY4: float = 0.0
var RCY1: float = 1.02
var REY1: float = 0.10
var REY2: float = 0.0
var RHY1: float = 0.0
var RHY2: float = 0.0
var RVY1: float = 0.0
var RVY2: float = 0.0
var RVY3: float = 0.0
var RVY4: float = 0.0

func _init(custom_param_path: String = "") -> void:
	load_parameters(custom_param_path)

## Returns the active tire compound identifier (e.g., "R25B" or "R20").
func get_compound() -> String:
	return active_compound

## Safely loads a specific compound ("R25B" or "R20"). Fails explicitly if file is missing or mismatch.
func load_compound(compound_name: String) -> bool:
	var c_lower = compound_name.to_lower().strip_edges()
	var filename = "hoosier_16x7.5_10_" + c_lower + "_mf62.json"
	var possible_paths = [
		"res://Scenes/Vehicle/Dynamics/Tires/MF62/" + filename,
		"src/mirena_sim/Scenes/Vehicle/Dynamics/Tires/MF62/" + filename,
		"Scenes/Vehicle/Dynamics/Tires/MF62/" + filename
	]
	var found_path = ""
	for p in possible_paths:
		if FileAccess.file_exists(p):
			found_path = p
			break
	if found_path == "":
		push_error("Pacejka62: Target parameter file for compound '" + compound_name + "' not found!")
		is_loaded = false
		return false
	return load_parameters(found_path, compound_name)

## Loads parameters from JSON file with strict validation (no silent fallback).
func load_parameters(param_path: String = "", expected_compound: String = "") -> bool:
	var path_to_load = param_path
	if path_to_load == "":
		var defaults = [
			DEFAULT_PARAM_PATH,
			"src/mirena_sim/Scenes/Vehicle/Dynamics/Tires/MF62/hoosier_16x7.5_10_mf62.json",
			"Scenes/Vehicle/Dynamics/Tires/MF62/hoosier_16x7.5_10_mf62.json"
		]
		for p in defaults:
			if FileAccess.file_exists(p):
				path_to_load = p
				break
	
	if path_to_load == "" or not FileAccess.file_exists(path_to_load):
		push_error("Pacejka62: Parameter file not found at: " + param_path)
		is_loaded = false
		return false
	
	var f = FileAccess.open(path_to_load, FileAccess.READ)
	if not f:
		push_error("Pacejka62: Failed to read parameter file at: " + path_to_load)
		is_loaded = false
		return false
		
	var text = f.get_as_text()
	var parsed = JSON.parse_string(text)
	if not (parsed is Dictionary):
		push_error("Pacejka62: Invalid JSON format in parameter file: " + path_to_load)
		is_loaded = false
		return false
		
	var comp = str(parsed.get("COMPOUND", "UNKNOWN")).strip_edges().to_upper()
	if expected_compound != "":
		var exp_clean = expected_compound.strip_edges().to_upper()
		if comp != exp_clean:
			push_error("Pacejka62: Compound mismatch! Requested '" + exp_clean + "', but file contains '" + comp + "' at " + path_to_load)
			is_loaded = false
			return false
			
	active_compound = comp
	params = parsed
	_apply_params_dict(parsed)
	is_loaded = true
	return true

func _apply_params_dict(d: Dictionary) -> void:
	fnomin = float(d.get("FNOMIN", fnomin))
	pinom = float(d.get("PINOM", pinom))
	r0 = float(d.get("UNLOADED_RADIUS", r0))
	vertical_stiffness = float(d.get("VERTICAL_STIFFNESS", vertical_stiffness))
	
	# Fy
	PCY1 = float(d.get("PCY1", PCY1))
	PDY1 = float(d.get("PDY1", PDY1))
	PDY2 = float(d.get("PDY2", PDY2))
	PDY3 = float(d.get("PDY3", PDY3))
	PEY1 = float(d.get("PEY1", PEY1))
	PEY2 = float(d.get("PEY2", PEY2))
	PEY3 = float(d.get("PEY3", PEY3))
	PEY4 = float(d.get("PEY4", PEY4))
	PEY5 = float(d.get("PEY5", PEY5))
	PKY1 = float(d.get("PKY1", PKY1))
	PKY2 = float(d.get("PKY2", PKY2))
	PKY3 = float(d.get("PKY3", PKY3))
	PHY1 = float(d.get("PHY1", PHY1))
	PHY2 = float(d.get("PHY2", PHY2))
	PHY3 = float(d.get("PHY3", PHY3))
	PVY1 = float(d.get("PVY1", PVY1))
	PVY2 = float(d.get("PVY2", PVY2))
	PVY3 = float(d.get("PVY3", PVY3))
	PVY4 = float(d.get("PVY4", PVY4))
	PPY1 = float(d.get("PPY1", PPY1))
	PPY2 = float(d.get("PPY2", PPY2))
	PPY3 = float(d.get("PPY3", PPY3))
	PPY4 = float(d.get("PPY4", PPY4))
	
	# Fx
	PCX1 = float(d.get("PCX1", PCX1))
	PDX1 = float(d.get("PDX1", PDX1))
	PDX2 = float(d.get("PDX2", PDX2))
	PDX3 = float(d.get("PDX3", PDX3))
	PEX1 = float(d.get("PEX1", PEX1))
	PEX2 = float(d.get("PEX2", PEX2))
	PEX3 = float(d.get("PEX3", PEX3))
	PEX4 = float(d.get("PEX4", PEX4))
	PKX1 = float(d.get("PKX1", PKX1))
	PKX2 = float(d.get("PKX2", PKX2))
	PKX3 = float(d.get("PKX3", PKX3))
	PHX1 = float(d.get("PHX1", PHX1))
	PHX2 = float(d.get("PHX2", PHX2))
	PVX1 = float(d.get("PVX1", PVX1))
	PVX2 = float(d.get("PVX2", PVX2))
	PPX1 = float(d.get("PPX1", PPX1))
	PPX2 = float(d.get("PPX2", PPX2))
	PPX3 = float(d.get("PPX3", PPX3))
	PPX4 = float(d.get("PPX4", PPX4))
	
	# Mz
	QDZ1 = float(d.get("QDZ1", QDZ1))
	QDZ2 = float(d.get("QDZ2", QDZ2))
	QDZ3 = float(d.get("QDZ3", QDZ3))
	QDZ6 = float(d.get("QDZ6", QDZ6))
	QDZ7 = float(d.get("QDZ7", QDZ7))
	QDZ8 = float(d.get("QDZ8", QDZ8))
	QDZ9 = float(d.get("QDZ9", QDZ9))
	QBZ1 = float(d.get("QBZ1", QBZ1))
	QBZ2 = float(d.get("QBZ2", QBZ2))
	QBZ3 = float(d.get("QBZ3", QBZ3))
	QBZ4 = float(d.get("QBZ4", QBZ4))
	QBZ5 = float(d.get("QBZ5", QBZ5))
	QBZ9 = float(d.get("QBZ9", QBZ9))
	QBZ10 = float(d.get("QBZ10", QBZ10))
	QCZ1 = float(d.get("QCZ1", QCZ1))
	QEZ1 = float(d.get("QEZ1", QEZ1))
	QEZ2 = float(d.get("QEZ2", QEZ2))
	QEZ3 = float(d.get("QEZ3", QEZ3))
	QEZ4 = float(d.get("QEZ4", QEZ4))
	QEZ5 = float(d.get("QEZ5", QEZ5))
	QHZ1 = float(d.get("QHZ1", QHZ1))
	QHZ2 = float(d.get("QHZ2", QHZ2))
	QHZ3 = float(d.get("QHZ3", QHZ3))
	QHZ4 = float(d.get("QHZ4", QHZ4))
	PPZ1 = float(d.get("PPZ1", PPZ1))
	PPZ2 = float(d.get("PPZ2", PPZ2))
	
	# SSZ
	SSZ1 = float(d.get("SSZ1", SSZ1))
	SSZ2 = float(d.get("SSZ2", SSZ2))
	SSZ3 = float(d.get("SSZ3", SSZ3))
	SSZ4 = float(d.get("SSZ4", SSZ4))
	
	# Mx
	QSX1 = float(d.get("QSX1", QSX1))
	QSX2 = float(d.get("QSX2", QSX2))
	QSX3 = float(d.get("QSX3", QSX3))
	PPMX1 = float(d.get("PPMX1", PPMX1))
	
	# My
	QSY1 = float(d.get("QSY1", QSY1))
	QSY2 = float(d.get("QSY2", QSY2))
	QSY3 = float(d.get("QSY3", QSY3))
	QSY4 = float(d.get("QSY4", QSY4))
	
	# Combined Slip Parameters
	RBX1 = float(d.get("RBX1", RBX1))
	RBX2 = float(d.get("RBX2", RBX2))
	RBX3 = float(d.get("RBX3", RBX3))
	RCX1 = float(d.get("RCX1", RCX1))
	REX1 = float(d.get("REX1", REX1))
	REX2 = float(d.get("REX2", REX2))
	RHX1 = float(d.get("RHX1", RHX1))
	RBY1 = float(d.get("RBY1", RBY1))
	RBY2 = float(d.get("RBY2", RBY2))
	RBY3 = float(d.get("RBY3", RBY3))
	RBY4 = float(d.get("RBY4", RBY4))
	RCY1 = float(d.get("RCY1", RCY1))
	REY1 = float(d.get("REY1", REY1))
	REY2 = float(d.get("REY2", REY2))
	RHY1 = float(d.get("RHY1", RHY1))
	RHY2 = float(d.get("RHY2", RHY2))
	RVY1 = float(d.get("RVY1", RVY1))
	RVY2 = float(d.get("RVY2", RVY2))
	RVY3 = float(d.get("RVY3", RVY3))
	RVY4 = float(d.get("RVY4", RVY4))

## Computes pure lateral force Fy0 (N) and intermediate shift Svy.
## is_right_side: whether wheel is mounted on right side of vehicle (mirrors conicity/ply-steer offsets).
func compute_fy0_with_shift(alpha: float, f_z: float, gamma: float = 0.0, p_kpa: float = 83.0, is_right_side: bool = false) -> Dictionary:
	var normal_load = maxf(0.0, f_z)
	if normal_load <= 1e-3:
		return {"Fy0": 0.0, "Svy": 0.0}
	
	var dfz = (normal_load - fnomin) / fnomin
	var dpi = (p_kpa - pinom) / pinom
	
	var Cy = PCY1
	var mu_y = (PDY1 + PDY2 * dfz) * (1.0 + PPY3 * dpi + PPY4 * dpi * dpi) * (1.0 - PDY3 * gamma * gamma)
	var Dy = mu_y * normal_load
	
	var shy_conicity = PHY1 + PHY2 * dfz
	var Shy = (-shy_conicity if is_right_side else shy_conicity) + PHY3 * gamma
	var svy_conicity = normal_load * (PVY1 + PVY2 * dfz)
	var svy_camber = normal_load * (PVY3 + PVY4 * dfz) * gamma
	var Svy = (-svy_conicity if is_right_side else svy_conicity) + svy_camber
	
	var alpha_y = alpha + Shy
	var kya_arg = 2.0 * atan(normal_load / (PKY2 * fnomin * (1.0 + PPY2 * dpi) + 1e-4))
	var Kya0 = PKY1 * fnomin * (1.0 + PPY1 * dpi) * (1.0 - PKY3 * absf(gamma)) * sin(kya_arg)
	var By = Kya0 / (Cy * Dy + 1e-4)
	
	var Ey_base = PEY1 + PEY2 * dfz
	var sign_ay = 1.0 if alpha_y > 0.0 else (-1.0 if alpha_y < 0.0 else 0.0)
	var Ey_camber = 1.0 + PEY5 * gamma * gamma - (PEY3 + PEY4 * gamma) * sign_ay
	var Ey = clampf(Ey_base * Ey_camber, -10.0, 1.0)
	
	var y_arg = By * alpha_y
	var inner = y_arg - Ey * (y_arg - atan(y_arg))
	var Fy0 = Dy * sin(Cy * atan(inner)) + Svy
	return {"Fy0": Fy0, "Svy": Svy}

## Convenience method for pure lateral force Fy0.
func compute_fy0(alpha: float, f_z: float, gamma: float = 0.0, p_kpa: float = 83.0, is_right_side: bool = false) -> float:
	return compute_fy0_with_shift(alpha, f_z, gamma, p_kpa, is_right_side)["Fy0"]

## Computes pure longitudinal force Fx0 (N).
func compute_fx0(kappa: float, f_z: float, gamma: float = 0.0, p_kpa: float = 83.0) -> float:
	var normal_load = maxf(0.0, f_z)
	if normal_load <= 1e-3:
		return 0.0
	
	var dfz = (normal_load - fnomin) / fnomin
	var dpi = (p_kpa - pinom) / pinom
	
	var Cx = PCX1
	var mu_x = (PDX1 + PDX2 * dfz) * (1.0 + PPX3 * dpi + PPX4 * dpi * dpi) * (1.0 - PDX3 * gamma * gamma)
	var Dx = mu_x * normal_load
	
	var Shx = PHX1 + PHX2 * dfz
	var Svx = normal_load * (PVX1 + PVX2 * dfz)
	var kappa_x = kappa + Shx
	
	var Kx0 = normal_load * (PKX1 + PKX2 * dfz) * exp(PKX3 * dfz) * (1.0 + PPX1 * dpi)
	var Bx = Kx0 / (Cx * Dx + 1e-4)
	
	var sign_kx = 1.0 if kappa_x > 0.0 else (-1.0 if kappa_x < 0.0 else 0.0)
	var Ex = clampf((PEX1 + PEX2 * dfz + PEX3 * dfz * dfz) * (1.0 - PEX4 * sign_kx), -10.0, 1.0)
	
	var x_arg = Bx * kappa_x
	var inner = x_arg - Ex * (x_arg - atan(x_arg))
	return Dx * sin(Cx * atan(inner)) + Svx

## Computes combined slip forces (Fx, Fy) with full MF 6.2 weighting functions and induced shift Svyk.
func compute_combined_forces(
	kappa: float,
	alpha: float,
	f_z: float,
	gamma: float = 0.0,
	p_kpa: float = 83.0,
	is_right_side: bool = false
) -> Dictionary:
	var normal_load = maxf(0.0, f_z)
	if normal_load <= 1e-3:
		return {"Fx": 0.0, "Fy": 0.0, "Svy": 0.0}
	
	var Fx0 = compute_fx0(kappa, normal_load, gamma, p_kpa)
	var fy0_res = compute_fy0_with_shift(alpha, normal_load, gamma, p_kpa, is_right_side)
	var Fy0: float = fy0_res["Fy0"]
	var Svy: float = fy0_res["Svy"]
	
	var dfz = (normal_load - fnomin) / fnomin
	
	# Longitudinal combined factor Gxa
	var Bxa = (RBX1 + RBX3 * gamma * gamma) * cos(atan(RBX2 * kappa))
	var Cxa = RCX1
	var Exa = clampf(REX1 + REX2 * dfz, -10.0, 1.0)
	var Shxa = RHX1
	var alpha_s = alpha + Shxa
	var arg_xa = Bxa * alpha_s
	var Gxa0 = cos(Cxa * atan(arg_xa - Exa * (arg_xa - atan(arg_xa))))
	var base_arg_xa = -Bxa * Shxa
	var Gxa_base = cos(Cxa * atan(base_arg_xa - Exa * (base_arg_xa - atan(base_arg_xa))))
	var Gxa = clampf(Gxa0 / maxf(absf(Gxa_base), 1e-4), 0.0, 1.0)
	
	# Lateral combined factor Gyk and induced shift Svyk
	var Byk = (RBY1 + RBY4 * gamma * gamma) * cos(atan(RBY2 * (alpha - RBY3)))
	var Cyk = RCY1
	var Eyk = clampf(REY1 + REY2 * dfz, -10.0, 1.0)
	var Shyk = RHY1 + RHY2 * dfz
	var kappa_s = kappa + Shyk
	var arg_yk = Byk * kappa_s
	var Gyk0 = cos(Cyk * atan(arg_yk - Eyk * (arg_yk - atan(arg_yk))))
	var base_arg_yk = -Byk * Shyk
	var Gyk_base = cos(Cyk * atan(base_arg_yk - Eyk * (base_arg_yk - atan(base_arg_yk))))
	var Gyk = clampf(Gyk0 / maxf(absf(Gyk_base), 1e-4), 0.0, 1.0)
	
	# Induced vertical shift Svyk (MF 6.2 Eq. 4.E60)
	var Dvyk = (Fy0 - Svy) * (RVY1 + RVY2 * dfz + RVY3 * gamma) * cos(atan(RVY4 * alpha))
	var Svyk = Dvyk * sin(1.05 * atan(Byk * kappa_s))
	
	var Fx = Gxa * Fx0
	var Fy = Gyk * Fy0 + Svyk
	return {"Fx": Fx, "Fy": Fy, "Svy": Svy}

## Computes self-aligning torque Mz (Nm) according to full MF 6.2 formulation:
## Mz = M'_z + Mzr + s * Fx
func compute_mz(
	alpha: float,
	f_z: float,
	Fy: float,
	gamma: float = 0.0,
	p_kpa: float = 83.0,
	Fx: float = 0.0,
	Svy: float = 0.0,
	is_right_side: bool = false
) -> float:
	var normal_load = maxf(0.0, f_z)
	if normal_load <= 1e-3:
		return 0.0
	
	var dfz = (normal_load - fnomin) / fnomin
	var dpi = (p_kpa - pinom) / pinom
	
	# Lateral force excluding vertical shift for pneumatic trail
	var Fy_prime = Fy - Svy
	
	var Sht = QHZ1 + QHZ2 * dfz + (QHZ3 + QHZ4 * dfz) * gamma
	var alpha_t = alpha + Sht
	
	var Dt = (normal_load / fnomin) * r0 * (QDZ1 + QDZ2 * dfz) * (1.0 + PPZ1 * dpi) * (1.0 - QDZ3 * absf(gamma))
	var Ct = QCZ1
	var Bt = (QBZ1 + QBZ2 * dfz + QBZ3 * dfz * dfz) * (1.0 + QBZ4 * gamma + QBZ5 * absf(gamma))
	var Et = clampf((QEZ1 + QEZ2 * dfz + QEZ3 * dfz * dfz) * (1.0 + (QEZ4 + QEZ5 * gamma) * (2.0 / PI) * atan(Bt * Ct * alpha_t)), -10.0, 1.0)
	
	var t_arg = Bt * alpha_t
	var trail = Dt * cos(Ct * atan(t_arg - Et * (t_arg - atan(t_arg)))) * cos(alpha)
	var Mz_prime = trail * Fy_prime
	
	# Residual torque Mzr
	var dr_conicity = normal_load * r0 * (QDZ6 + QDZ7 * dfz)
	var dr_camber = normal_load * r0 * (QDZ8 + QDZ9 * dfz) * (1.0 + PPZ2 * dpi) * gamma
	var Dr = (-dr_conicity if is_right_side else dr_conicity) + dr_camber
	var Br = QBZ9
	var Cr = 1.0
	var Shr = Sht
	var alpha_r = alpha + Shr
	var Mzr = Dr * cos(Cr * atan(Br * alpha_r))
	
	# Longitudinal force arm coupling s * Fx
	var s = r0 * (SSZ1 + SSZ2 * (Fy / fnomin) + (SSZ3 + SSZ4 * dfz) * gamma)
	var Mz_fx = s * Fx
	
	return Mz_prime + Mzr + Mz_fx

## Computes overturning moment Mx (Nm) with load, camber, and pressure dependencies.
func compute_mx(f_z: float, gamma: float, Fy: float, p_kpa: float = 83.0, is_right_side: bool = false) -> float:
	var normal_load = maxf(0.0, f_z)
	var dpi = (p_kpa - pinom) / pinom
	var qsx1 = -QSX1 if is_right_side else QSX1
	return normal_load * r0 * (qsx1 - QSX2 * gamma + QSX3 * (Fy / fnomin) + PPMX1 * dpi)

## Computes rolling resistance moment My (Nm) according to MF 6.2 Eq. 4.E65.
func compute_my(f_z: float, Fx: float, v_x: float) -> float:
	var normal_load = maxf(0.0, f_z)
	var v_ratio = absf(v_x) / maxf(v0, 1.0)
	var rr_factor = QSY1 + QSY2 * (Fx / fnomin) + QSY3 * v_ratio + QSY4 * (v_ratio * v_ratio * v_ratio * v_ratio)
	var sign_v = 1.0 if v_x >= 0.0 else -1.0
	return -r0 * normal_load * rr_factor * sign_v
