class_name Pacejka62
extends RefCounted

## pacejka magic formula 6.2 evaluation engine - mirena dynamics (schema v2, paired with calibrate_mf62.py)
## frame: iso 8855 (+x fwd, +y left, +z up). alpha, gamma, kappa in rad / unitless; forces n, moments nm
## implemented: fy0/fx0, combined slip (gxa, gyk, svyk), mz = -t(alpha_t,eq)(fy - svyk) + mzr(alpha_r,eq) + s*fx
## with shf = shy + svy/kya, mx, my
## canonical pacejka 2012 / besselink 2010 structure:
## model constant r0 = 0.2032 m shared with fitter
## svyk and alpha_r = alpha + shf implemented
## pneumatic trail / residual torque coupled with longitudinal slip (alpha_t,eq, alpha_r,eq)
## strict loading schema v2: rejects missing/non-finite parameters, push_error once, zero output if unloaded
## right wheel: conicity/ply-steer terms (phy1, phy2, pvy1, pvy2, qdz6, qdz7, qsx1) flip sign; data are left-as-tested

const SCHEMA_VERSION: int = 2
const PARAM_DIR: String = "res://Scenes/Vehicle/Dynamics/Tires/MF62/"
const DEFAULT_PARAM_FILE: String = "hoosier_16x7.5_10_mf62.json"
const EDITOR_FALLBACK_DIR: String = "src/mirena_sim/Scenes/Vehicle/Dynamics/Tires/MF62/"
const V_EPS_MPS: float = 0.5
const SIGMA_ALPHA_NOMINAL_M: float = 0.28
const SIGMA_KAPPA_NOMINAL_M: float = 0.15
const RELAXATION_V_MIN_MPS: float = 1.5

const HEADER_NUMERIC: PackedStringArray = ["FNOMIN", "PINOM", "UNLOADED_RADIUS", "VERTICAL_STIFFNESS"]
const KNOWN_EXTRA: PackedStringArray = [
	"SCHEMA_VERSION", "COMPOUND", "CALIBRATION_ID", "CALIBRATION_DATE", "MODEL_IDENTIFIER", "FZ_VALID_MAX",
	"R0_SPRING_FIT_M", "VERTICAL_STIFFNESS_ANALYSIS", "PARAMETER_CATEGORIZATION"]

# must equal all_defaults keys in calibrate_mf62.py (checked by the parity test)
const REQUIRED_KEYS: PackedStringArray = [
	"PCY1", "PDY1", "PDY2", "PDY3", "PEY1", "PEY2", "PEY3", "PEY4", "PEY5", "PKY1", "PKY2", "PKY3", "PHY1", "PHY2",
	"PHY3", "PVY1", "PVY2", "PVY3", "PVY4", "PPY1", "PPY2", "PPY3", "PPY4",
	"PCX1", "PDX1", "PDX2", "PDX3", "PEX1", "PEX2", "PEX3", "PEX4", "PKX1", "PKX2", "PKX3", "PHX1", "PHX2", "PVX1",
	"PVX2", "PPX1", "PPX2", "PPX3", "PPX4",
	"RBX1", "RBX2", "RBX3", "RCX1", "REX1", "REX2", "RHX1", "RBY1", "RBY2", "RBY3", "RBY4", "RCY1", "REY1", "REY2",
	"RHY1", "RHY2", "RVY1", "RVY2", "RVY3", "RVY4", "RVY5", "RVY6",
	"QDZ1", "QDZ2", "QDZ3", "QDZ4", "QDZ6", "QDZ7", "QDZ8", "QDZ9", "QBZ1", "QBZ2", "QBZ3", "QBZ4", "QBZ5", "QBZ9",
	"QBZ10", "QCZ1", "QEZ1", "QEZ2", "QEZ3", "QEZ4", "QEZ5", "QHZ1", "QHZ2", "QHZ3", "QHZ4", "PPZ1", "PPZ2",
	"SSZ1", "SSZ2", "SSZ3", "SSZ4",
	"QSX1", "QSX2", "QSX3", "PPMX1",
	"QSY1", "QSY2", "QSY3", "QSY4"]

var is_loaded: bool = false
var active_compound: String = "UNKNOWN"
var calibration_id: String = ""

# reference quantities (from file)
var fnomin: float = 800.0
var pinom: float = 83.0
var r0: float = 0.2032
var vertical_stiffness: float = 0.0
var v0: float = 11.11
var dfz_max: float = INF  # upper clamp of dfz inside shape/mu functions (soft extrapolation guard), from fz_valid_max

# fy
var PCY1: float
var PDY1: float
var PDY2: float
var PDY3: float
var PEY1: float
var PEY2: float
var PEY3: float
var PEY4: float
var PEY5: float
var PKY1: float
var PKY2: float
var PKY3: float
var PHY1: float
var PHY2: float
var PHY3: float
var PVY1: float
var PVY2: float
var PVY3: float
var PVY4: float
var PPY1: float
var PPY2: float
var PPY3: float
var PPY4: float
# fx
var PCX1: float
var PDX1: float
var PDX2: float
var PDX3: float
var PEX1: float
var PEX2: float
var PEX3: float
var PEX4: float
var PKX1: float
var PKX2: float
var PKX3: float
var PHX1: float
var PHX2: float
var PVX1: float
var PVX2: float
var PPX1: float
var PPX2: float
var PPX3: float
var PPX4: float
# combined slip
var RBX1: float
var RBX2: float
var RBX3: float
var RCX1: float
var REX1: float
var REX2: float
var RHX1: float
var RBY1: float
var RBY2: float
var RBY3: float
var RBY4: float
var RCY1: float
var REY1: float
var REY2: float
var RHY1: float
var RHY2: float
var RVY1: float
var RVY2: float
var RVY3: float
var RVY4: float
var RVY5: float
var RVY6: float
# mz
var QDZ1: float
var QDZ2: float
var QDZ3: float
var QDZ4: float
var QDZ6: float
var QDZ7: float
var QDZ8: float
var QDZ9: float
var QBZ1: float
var QBZ2: float
var QBZ3: float
var QBZ4: float
var QBZ5: float
var QBZ9: float
var QBZ10: float
var QCZ1: float
var QEZ1: float
var QEZ2: float
var QEZ3: float
var QEZ4: float
var QEZ5: float
var QHZ1: float
var QHZ2: float
var QHZ3: float
var QHZ4: float
var PPZ1: float
var PPZ2: float
var SSZ1: float
var SSZ2: float
var SSZ3: float
var SSZ4: float
# mx
var QSX1: float
var QSX2: float
var QSX3: float
var PPMX1: float
# my
var QSY1: float
var QSY2: float
var QSY3: float
var QSY4: float

# canonical mf 6.2 friction scaling factors (1.0 = raw ttc belt calibration)
var LMUX: float = 1.0
var LMUY: float = 1.0
var LKYA: float = 1.0
var LCAMBER: float = 1.0

var _unloaded_warned: bool = false


func _init(custom_param_path: String = "") -> void:
	load_parameters(custom_param_path)


func get_compound() -> String:
	return active_compound


## loads "hoosier_16x7.5_10_<compound>_mf62.json" and verifies the file's compound matches
func load_compound(compound_name: String) -> bool:
	var filename: String = "hoosier_16x7.5_10_%s_mf62.json" % compound_name.to_lower().strip_edges()
	var path: String = _first_existing([PARAM_DIR + filename, EDITOR_FALLBACK_DIR + filename])
	if path.is_empty():
		push_error("Pacejka62: parameter file for compound '%s' not found." % compound_name)
		is_loaded = false
		return false
	return load_parameters(path, compound_name)


func load_parameters(param_path: String = "", expected_compound: String = "") -> bool:
	is_loaded = false
	_unloaded_warned = false
	var path: String = param_path
	if path.is_empty():
		path = _first_existing([PARAM_DIR + DEFAULT_PARAM_FILE, EDITOR_FALLBACK_DIR + DEFAULT_PARAM_FILE])
	if path.is_empty() or not FileAccess.file_exists(path):
		push_error("Pacejka62: parameter file not found (requested '%s')." % param_path)
		return false
	var f: FileAccess = FileAccess.open(path, FileAccess.READ)
	if f == null:
		push_error("Pacejka62: cannot open '%s' (error %d)." % [path, FileAccess.get_open_error()])
		return false
	var parsed: Variant = JSON.parse_string(f.get_as_text())
	if not (parsed is Dictionary):
		push_error("Pacejka62: invalid JSON in '%s'." % path)
		return false
	var d: Dictionary = parsed
	var err: String = _validate(d, expected_compound)
	if not err.is_empty():
		push_error("Pacejka62: %s [%s]" % [err, path])
		return false
	err = _commit(d)
	if not err.is_empty():
		push_error("Pacejka62: %s [%s]" % [err, path])
		return false
	is_loaded = true
	return true


func _first_existing(paths: Array) -> String:
	for p in paths:
		if FileAccess.file_exists(p):
			return p
	return ""


func _is_num(v: Variant) -> bool:
	return (typeof(v) == TYPE_FLOAT or typeof(v) == TYPE_INT) and is_finite(float(v))


func _validate(d: Dictionary, expected_compound: String) -> String:
	if int(d.get("SCHEMA_VERSION", 0)) != SCHEMA_VERSION:
		return "SCHEMA_VERSION %s != %d; re-run calibrate_mf62.py" % [str(d.get("SCHEMA_VERSION", "missing")), SCHEMA_VERSION]
	for k in REQUIRED_KEYS:
		if not d.has(k) or not _is_num(d[k]):
			return "missing or non-finite key '%s'" % k
	for k in HEADER_NUMERIC:
		if not d.has(k) or not _is_num(d[k]):
			return "missing or non-finite header key '%s'" % k
	if not (d.get("COMPOUND") is String) or not (d.get("CALIBRATION_ID") is String) or String(d["CALIBRATION_ID"]).is_empty():
		return "COMPOUND / CALIBRATION_ID missing"
	var comp: String = String(d["COMPOUND"]).strip_edges().to_upper()
	if not expected_compound.is_empty() and comp != expected_compound.strip_edges().to_upper():
		return "compound mismatch: requested '%s', file has '%s'" % [expected_compound.to_upper(), comp]
	if float(d["FNOMIN"]) <= 0.0 or float(d["PINOM"]) <= 0.0 or float(d["UNLOADED_RADIUS"]) <= 0.0:
		return "FNOMIN / PINOM / UNLOADED_RADIUS must be > 0"
	if d.has("FZ_VALID_MAX") and (not _is_num(d["FZ_VALID_MAX"]) or float(d["FZ_VALID_MAX"]) <= float(d["FNOMIN"])):
		return "FZ_VALID_MAX invalid"
	for k in d.keys():
		if not (k in REQUIRED_KEYS) and not (k in HEADER_NUMERIC) and not (k in KNOWN_EXTRA):
			push_warning("Pacejka62: unknown key '%s' ignored." % k)
	return ""


func _commit(d: Dictionary) -> String:
	for k in REQUIRED_KEYS:
		var v: float = float(d[k])
		set(k, v)
		if not is_equal_approx(float(get(k)), v):
			return "member '%s' not assignable (REQUIRED_KEYS out of sync with declarations)" % k
	fnomin = float(d["FNOMIN"])
	pinom = float(d["PINOM"])
	r0 = float(d["UNLOADED_RADIUS"])
	vertical_stiffness = float(d["VERTICAL_STIFFNESS"])
	active_compound = String(d["COMPOUND"]).strip_edges().to_upper()
	calibration_id = String(d["CALIBRATION_ID"])
	dfz_max = ((float(d["FZ_VALID_MAX"]) - fnomin) / fnomin) if d.has("FZ_VALID_MAX") else INF
	return ""


func _check_loaded() -> bool:
	if is_loaded:
		return true
	if not _unloaded_warned:
		push_error("Pacejka62: evaluated without a valid parameter set; returning zeros.")
		_unloaded_warned = true
	return false


func _dfz(fz: float) -> float:
	return minf((fz - fnomin) / fnomin, dfz_max)


static func _sgn(x: float) -> float:
	return 1.0 if x >= 0.0 else -1.0


static func _shape_cos(b: float, c: float, e: float, x: float) -> float:
	var bx: float = b * x
	return cos(c * atan(bx - e * (bx - atan(bx))))


# pure slip
func _pure_lateral(alpha: float, fz: float, gamma: float, p_kpa: float, right: bool) -> Dictionary:
	var dfz: float = _dfz(fz)
	var dpi: float = (p_kpa - pinom) / pinom
	var g: float = gamma * LCAMBER
	var cy: float = PCY1
	# runtime-only safety: keep the camber factor positive far outside the +-4 deg training range
	var camber_f: float = maxf(1.0 - PDY3 * g * g, 0.2)
	var mu_y: float = (PDY1 + PDY2 * dfz) * (1.0 + PPY3 * dpi + PPY4 * dpi * dpi) * camber_f * LMUY
	var dy: float = mu_y * fz
	var side: float = -1.0 if right else 1.0
	var shy: float = side * (PHY1 + PHY2 * dfz) + PHY3 * g
	var svy: float = side * fz * (PVY1 + PVY2 * dfz) + fz * (PVY3 + PVY4 * dfz) * g
	var alpha_y: float = alpha + shy
	var kya: float = LKYA * PKY1 * fnomin * (1.0 + PPY1 * dpi) * (1.0 - PKY3 * absf(g)) \
		* sin(2.0 * atan(fz / (PKY2 * fnomin * (1.0 + PPY2 * dpi) + 1e-4)))
	var by: float = kya / (cy * dy + 1e-4)
	var ey: float = clampf((PEY1 + PEY2 * dfz) * (1.0 + PEY5 * g * g - (PEY3 + PEY4 * g) * signf(alpha_y)), -10.0, 1.0)
	var y: float = by * alpha_y
	var fy0: float = dy * sin(cy * atan(y - ey * (y - atan(y)))) + svy
	return {"Fy0": fy0, "Svy": svy, "Shy": shy, "Kya": kya, "By": by, "Cy": cy, "mu_y": mu_y}


func _pure_longitudinal(kappa: float, fz: float, gamma: float, p_kpa: float) -> Dictionary:
	var dfz: float = _dfz(fz)
	var dpi: float = (p_kpa - pinom) / pinom
	var g: float = gamma * LCAMBER
	var cx: float = PCX1
	var mu_x: float = (PDX1 + PDX2 * dfz) * (1.0 + PPX3 * dpi + PPX4 * dpi * dpi) * maxf(1.0 - PDX3 * g * g, 0.2) * LMUX
	var dx: float = mu_x * fz
	var kx: float = kappa + PHX1 + PHX2 * dfz
	var svx: float = fz * (PVX1 + PVX2 * dfz)
	var kxk: float = fz * (PKX1 + PKX2 * dfz) * exp(PKX3 * dfz) * (1.0 + PPX1 * dpi + PPX2 * dpi * dpi)
	var bx: float = kxk / (cx * dx + 1e-4)
	var ex: float = clampf((PEX1 + PEX2 * dfz + PEX3 * dfz * dfz) * (1.0 - PEX4 * signf(kx)), -10.0, 1.0)
	var x: float = bx * kx
	return {"Fx0": dx * sin(cx * atan(x - ex * (x - atan(x)))) + svx, "Kxk": kxk}


## pure lateral force and vertical shift. kept for api compatibility
func compute_fy0_with_shift(alpha: float, f_z: float, gamma: float = 0.0, p_kpa: float = 83.0, is_right_side: bool = false) -> Dictionary:
	var fz: float = maxf(0.0, f_z)
	if not _check_loaded() or fz <= 1e-3:
		return {"Fy0": 0.0, "Svy": 0.0}
	var r: Dictionary = _pure_lateral(alpha, fz, gamma, p_kpa, is_right_side)
	return {"Fy0": r["Fy0"], "Svy": r["Svy"]}


func compute_fy0(alpha: float, f_z: float, gamma: float = 0.0, p_kpa: float = 83.0, is_right_side: bool = false) -> float:
	return compute_fy0_with_shift(alpha, f_z, gamma, p_kpa, is_right_side)["Fy0"]


func compute_fx0(kappa: float, f_z: float, gamma: float = 0.0, p_kpa: float = 83.0) -> float:
	var fz: float = maxf(0.0, f_z)
	if not _check_loaded() or fz <= 1e-3:
		return 0.0
	return _pure_longitudinal(kappa, fz, gamma, p_kpa)["Fx0"]


# combined slip
## returns {fx, fy, fy0, svy, svyk, shf, kya, kxk, by, cy, gyk}. fy is the mf-only value (no nn residual)
func compute_combined_forces(kappa: float, alpha: float, f_z: float, gamma: float = 0.0, p_kpa: float = 83.0, is_right_side: bool = false) -> Dictionary:
	var fz: float = maxf(0.0, f_z)
	if not _check_loaded() or fz <= 1e-3:
		return {"Fx": 0.0, "Fy": 0.0, "Fy0": 0.0, "Svy": 0.0, "Svyk": 0.0, "Shf": 0.0, "Kya": 1.0, "Kxk": 0.0,
				"By": 0.0, "Cy": 1.0, "Gyk": 1.0}
	var fy: Dictionary = _pure_lateral(alpha, fz, gamma, p_kpa, is_right_side)
	var fx: Dictionary = _pure_longitudinal(kappa, fz, gamma, p_kpa)
	var dfz: float = _dfz(fz)
	var g: float = gamma * LCAMBER

	var bxa: float = (RBX1 + RBX3 * g * g) * cos(atan(RBX2 * kappa))
	var exa: float = clampf(REX1 + REX2 * dfz, -10.0, 1.0)
	var gxa: float = maxf(_shape_cos(bxa, RCX1, exa, alpha + RHX1) / maxf(_shape_cos(bxa, RCX1, exa, RHX1), 1e-4), 0.0)

	var byk: float = (RBY1 + RBY4 * g * g) * cos(atan(RBY2 * (alpha - RBY3)))
	var eyk: float = clampf(REY1 + REY2 * dfz, -10.0, 1.0)
	var shyk: float = RHY1 + RHY2 * dfz
	var gyk: float = maxf(_shape_cos(byk, RCY1, eyk, kappa + shyk) / maxf(_shape_cos(byk, RCY1, eyk, shyk), 1e-4), 0.0)

	var dvyk: float = float(fy["mu_y"]) * fz * (RVY1 + RVY2 * dfz + RVY3 * g) * cos(atan(RVY4 * alpha))
	var svyk: float = dvyk * sin(RVY5 * atan(RVY6 * kappa))

	var shf: float = float(fy["Shy"]) + float(fy["Svy"]) / (float(fy["Kya"]) + 1e-3)
	return {
		"Fx": gxa * float(fx["Fx0"]),
		"Fy": gyk * float(fy["Fy0"]) + svyk,
		"Fy0": fy["Fy0"], "Svy": fy["Svy"], "Svyk": svyk, "Shf": shf,
		"Kya": fy["Kya"], "Kxk": fx["Kxk"], "By": fy["By"], "Cy": fy["Cy"], "Gyk": gyk}


# moments
## mz = -t*(fy - svyk) + mzr + s*fx
## if `comb` is empty, it is automatically computed with compute_combined_forces
func compute_mz(alpha: float, f_z: float, Fy: float = 0.0, gamma: float = 0.0, p_kpa: float = 83.0, Fx: float = 0.0,
		comb: Dictionary = {}, kappa: float = 0.0, is_right_side: bool = false) -> float:
	var fz: float = maxf(0.0, f_z)
	if not _check_loaded() or fz <= 1e-3:
		return 0.0
	var c: Dictionary = comb
	if c.is_empty():
		c = compute_combined_forces(kappa, alpha, fz, gamma, p_kpa, is_right_side)
	var fy_val: float = float(c["Fy"]) if is_zero_approx(Fy) else Fy
	var fx_val: float = float(c["Fx"]) if is_zero_approx(Fx) else Fx
	
	var dfz: float = _dfz(fz)
	var dpi: float = (p_kpa - pinom) / pinom
	var g: float = gamma * LCAMBER

	var sht: float = QHZ1 + QHZ2 * dfz + (QHZ3 + QHZ4 * dfz) * g
	var alpha_t: float = alpha + sht
	var alpha_r: float = alpha + float(c["Shf"])
	var kx: float = float(c["Kxk"]) / (float(c["Kya"]) + 1e-3) * kappa
	var alpha_t_eq: float = _sgn(alpha_t) * sqrt(alpha_t * alpha_t + kx * kx)
	var alpha_r_eq: float = _sgn(alpha_r) * sqrt(alpha_r * alpha_r + kx * kx)

	var dt: float = (fz / fnomin) * r0 * (QDZ1 + QDZ2 * dfz) * (1.0 + PPZ1 * dpi) * (1.0 + QDZ3 * absf(g) + QDZ4 * g * g)
	var ct: float = QCZ1
	var bt: float = (QBZ1 + QBZ2 * dfz + QBZ3 * dfz * dfz) * (1.0 + QBZ4 * g + QBZ5 * absf(g))
	var t_arg: float = bt * alpha_t_eq
	var et: float = clampf((QEZ1 + QEZ2 * dfz + QEZ3 * dfz * dfz) * (1.0 + (QEZ4 + QEZ5 * g) * (2.0 / PI) * atan(bt * ct * alpha_t_eq)), -10.0, 1.0)
	var trail: float = dt * cos(ct * atan(t_arg - et * (t_arg - atan(t_arg)))) * cos(alpha)
	var mz_prime: float = -trail * (fy_val - float(c["Svyk"]))

	var side: float = -1.0 if is_right_side else 1.0
	var dr: float = fz * r0 * (side * (QDZ6 + QDZ7 * dfz) + (QDZ8 + QDZ9 * dfz) * (1.0 + PPZ2 * dpi) * g) * cos(alpha)
	var br: float = QBZ9 + QBZ10 * float(c["By"]) * float(c["Cy"])
	var mzr: float = dr * cos(atan(br * alpha_r_eq))

	var s: float = r0 * (SSZ1 + SSZ2 * (fy_val / fnomin) + (SSZ3 + SSZ4 * dfz) * g)
	return mz_prime + mzr + s * fx_val


func compute_mx(f_z: float, gamma: float, Fy: float, p_kpa: float = 83.0, is_right_side: bool = false) -> float:
	if not _check_loaded():
		return 0.0
	var fz: float = maxf(0.0, f_z)
	var dpi: float = (p_kpa - pinom) / pinom
	var qsx1: float = -QSX1 if is_right_side else QSX1
	return fz * r0 * (qsx1 + QSX2 * (gamma * LCAMBER) + QSX3 * (Fy / fnomin) + PPMX1 * dpi)


## rolling-resistance moment; sign is smoothed (tanh) so there is no torque jitter at v_x ~ 0
func compute_my(f_z: float, Fx: float, v_x: float) -> float:
	if not _check_loaded():
		return 0.0
	var fz: float = maxf(0.0, f_z)
	var v_ratio: float = absf(v_x) / maxf(v0, 1.0)
	var rr: float = QSY1 + QSY2 * (Fx / fnomin) + QSY3 * v_ratio + QSY4 * pow(v_ratio, 4.0)
	return -r0 * fz * rr * tanh(v_x / V_EPS_MPS)


# one-call entry point
## full wheel output. `residual` = neuraltirecorrection.predict_residual(...) (dfy, dmz) or zero
## residual enters through the lateral-slip weighting: fy = gyk*(fy0 + dfy) + svyk ; mz += gyk*dmz
## returns {fx, fy, mz, mx, my}
func compute_all(kappa: float, alpha: float, f_z: float, gamma: float = 0.0, p_kpa: float = 83.0, v_x: float = 0.0,
		is_right_side: bool = false, residual: Vector2 = Vector2.ZERO) -> Dictionary:
	var comb: Dictionary = compute_combined_forces(kappa, alpha, f_z, gamma, p_kpa, is_right_side)
	var gyk: float = comb["Gyk"]
	var fx: float = comb["Fx"]
	var fy_mf: float = comb["Fy"]
	var fy: float = gyk * (float(comb["Fy0"]) + residual.x) + float(comb["Svyk"])
	var mz: float = compute_mz(alpha, f_z, fy_mf, gamma, p_kpa, fx, comb, kappa, is_right_side) + gyk * residual.y
	return {"Fx": fx, "Fy": fy, "Mz": mz, "Mx": compute_mx(f_z, gamma, fy, p_kpa, is_right_side), "My": compute_my(f_z, fx, v_x)}


# transient carcass relaxation (sub-stepped exponential ode integrator)
## vertical-load-dependent lateral relaxation length sigma_alpha (m)
## equals sigma_alpha_nominal_m (0.28 m) at nominal f_z = f_z0 (800 n)
static func get_sigma_alpha(f_z: float, f_z0: float = 800.0) -> float:
	var load_ratio: float = clampf(maxf(f_z, 0.0) / maxf(f_z0, 1.0), 0.0, 2.5)
	return clampf(SIGMA_ALPHA_NOMINAL_M * (0.45 + 0.55 * load_ratio), 0.12, 0.45)


## vertical-load-dependent longitudinal relaxation length sigma_kappa (m)
## equals sigma_kappa_nominal_m (0.15 m) at nominal f_z = f_z0 (800 n)
static func get_sigma_kappa(f_z: float, f_z0: float = 800.0) -> float:
	var load_ratio: float = clampf(maxf(f_z, 0.0) / maxf(f_z0, 1.0), 0.0, 2.5)
	return clampf(SIGMA_KAPPA_NOMINAL_M * (0.45 + 0.55 * load_ratio), 0.06, 0.30)


## integrates effective transient slip (alpha_eff, kappa_eff) via unconditionally stable
## sub-stepped exponential relaxation. returns vector2(alpha_eff, kappa_eff)
static func integrate_transient_slip_substep(
	alpha_eff: float,
	alpha_geom: float,
	kappa_eff: float,
	kappa_geom: float,
	v_x: float,
	dt: float,
	f_z: float,
	v_kappa_rel: float = NAN,
	substeps: int = 4,
	sigma_alpha_override: float = -1.0
) -> Vector2:
	if dt <= 0.0:
		return Vector2(alpha_eff, kappa_eff)

	var n_sub: int = maxi(substeps, 1)
	var h: float = dt / float(n_sub)

	var abs_vx: float = absf(v_x)
	var abs_vk: float = abs_vx if is_nan(v_kappa_rel) else absf(v_kappa_rel)

	var v_alpha: float = maxf(abs_vx, RELAXATION_V_MIN_MPS)
	var v_kappa: float = maxf(abs_vk, RELAXATION_V_MIN_MPS)

	var alpha_target: float = alpha_geom * clampf(abs_vx / V_EPS_MPS, 0.0, 1.0)
	var kappa_target: float = kappa_geom * clampf(abs_vk / V_EPS_MPS, 0.0, 1.0)

	var sigma_a: float = clampf(sigma_alpha_override, 0.02, 1.0) if sigma_alpha_override > 0.0 else get_sigma_alpha(f_z)
	var tau_a: float = sigma_a / v_alpha
	var tau_k: float = get_sigma_kappa(f_z) / v_kappa
	var decay_a: float = exp(-h / maxf(tau_a, 1e-6))
	var decay_k: float = exp(-h / maxf(tau_k, 1e-6))

	var a_curr: float = alpha_eff if is_finite(alpha_eff) else 0.0
	var k_curr: float = kappa_eff if is_finite(kappa_eff) else 0.0
	for _s in range(n_sub):
		a_curr = alpha_target + (a_curr - alpha_target) * decay_a
		k_curr = kappa_target + (k_curr - kappa_target) * decay_k

	return Vector2(a_curr, k_curr)
