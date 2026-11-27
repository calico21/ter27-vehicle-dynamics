class_name NeuralTireCorrection
extends RefCounted

## gated residual mlp (4-16-16-2, tanh) for mirena dynamics: [dfy, dmz] = f_exp - f_mf62
## must be loaded against the exact mf62 calibration it was trained on: load_for(mf) checks compound and
## calibration_id and refuses to arm otherwise (is_ready=false => zero correction)
## formulas are identical to train_residual_mlp.py (float64 there, gdscript float is double):
## gate = (1-exp(-(a^2+g^2)/(2 s^2))) * smoothstep(fz/load_trans) * sigmoid(-(sqrt(a^2+eps)-a_lim)/a_slope) * sigmoid(-(fz-fz_lim)/fz_slope)
## out = clip * tanh(gate * denorm(mlp(norm(x))) / clip)
## right wheel: df_r(alpha, gamma) = -df_l(-alpha, -gamma) (training data are left-as-tested)
## hot path predict_residual() performs no heap allocation (flat packedfloat64array buffers, vector2 return)

const NEURAL_DIR: String = "res://Scenes/Vehicle/Dynamics/Tires/NeuralCorrection/"
const EDITOR_FALLBACK_DIR: String = "src/mirena_sim/Scenes/Vehicle/Dynamics/Tires/NeuralCorrection/"
const N_IN: int = 4
const N_H: int = 16
const N_OUT: int = 2
const GATE_KEYS: PackedStringArray = [
	"slip_scale_rad", "load_trans_n", "alpha_limit_rad", "alpha_slope_rad", "fz_limit_n", "fz_slope_n",
	"clip_fy_n", "clip_mz_nm"]

var is_ready: bool = false
var compound: String = "UNKNOWN"
var bound_calibration_id: String = ""

var _x_mean: PackedFloat64Array = PackedFloat64Array()
var _x_std: PackedFloat64Array = PackedFloat64Array()
var _y_mean: PackedFloat64Array = PackedFloat64Array()
var _y_std: PackedFloat64Array = PackedFloat64Array()
var _w1: PackedFloat64Array = PackedFloat64Array()  # n_h x n_in, row-major
var _b1: PackedFloat64Array = PackedFloat64Array()
var _w2: PackedFloat64Array = PackedFloat64Array()  # n_h x n_h
var _b2: PackedFloat64Array = PackedFloat64Array()
var _w3: PackedFloat64Array = PackedFloat64Array()  # n_out x n_h
var _b3: PackedFloat64Array = PackedFloat64Array()

var _slip_scale: float = 0.006
var _load_trans: float = 100.0
var _alpha_limit: float = 0.22
var _alpha_slope: float = 0.02
var _fz_limit: float = 1300.0
var _fz_slope: float = 100.0
var _clip_fy: float = 200.0
var _clip_mz: float = 10.0

var _h1: PackedFloat64Array = PackedFloat64Array()
var _h2: PackedFloat64Array = PackedFloat64Array()


func _init(mf: Pacejka62 = null, weights_path: String = "") -> void:
	_h1.resize(N_H)
	_h2.resize(N_H)
	if mf != null:
		load_for(mf, weights_path)
	else:
		# auto-bind to default mf62 model if none provided
		var default_mf: Pacejka62 = Pacejka62.new()
		if default_mf.is_loaded:
			load_for(default_mf, weights_path)


## loads neural_residual_weights_<compound>.json (or `weights_path`) and binds it to `mf`
func load_for(mf: Pacejka62, weights_path: String = "") -> bool:
	is_ready = false
	if mf == null or not mf.is_loaded:
		push_error("NeuralTireCorrection: MF62 model not loaded; residual disabled.")
		return false
	var path: String = weights_path
	if path.is_empty():
		var fname: String = "neural_residual_weights_%s.json" % mf.active_compound.to_lower()
		for p in [NEURAL_DIR + fname, EDITOR_FALLBACK_DIR + fname]:
			if FileAccess.file_exists(p):
				path = p
				break
	if path.is_empty() or not FileAccess.file_exists(path):
		push_error("NeuralTireCorrection: weights for compound '%s' not found; residual disabled." % mf.active_compound)
		return false
	var err: String = _load_file(path, mf.active_compound, mf.calibration_id)
	if not err.is_empty():
		push_error("NeuralTireCorrection: %s [%s]; residual disabled." % [err, path])
		_clear()
		return false
	is_ready = true
	return true


func _clear() -> void:
	is_ready = false
	bound_calibration_id = ""


func _num_array(v: Variant, n: int) -> PackedFloat64Array:
	var out: PackedFloat64Array = PackedFloat64Array()
	if not (v is Array) or (v as Array).size() != n:
		return out
	for x in v:
		if not (typeof(x) == TYPE_FLOAT or typeof(x) == TYPE_INT) or not is_finite(float(x)):
			return PackedFloat64Array()
		out.append(float(x))
	return out


func _flatten(v: Variant, rows: int, cols: int) -> PackedFloat64Array:
	var out: PackedFloat64Array = PackedFloat64Array()
	if not (v is Array) or (v as Array).size() != rows:
		return out
	for r in v:
		var row: PackedFloat64Array = _num_array(r, cols)
		if row.size() != cols:
			return PackedFloat64Array()
		out.append_array(row)
	return out


func _load_file(path: String, expected_compound: String, expected_calibration_id: String) -> String:
	var f: FileAccess = FileAccess.open(path, FileAccess.READ)
	if f == null:
		return "cannot open (error %d)" % FileAccess.get_open_error()
	var parsed: Variant = JSON.parse_string(f.get_as_text())
	if not (parsed is Dictionary):
		return "invalid JSON"
	var data: Dictionary = parsed

	var arch: Variant = data.get("architecture")
	if not (arch is Dictionary):
		return "missing architecture"
	if int(arch.get("in_dim", -1)) != N_IN or int(arch.get("hidden1", -1)) != N_H \
			or int(arch.get("hidden2", -1)) != N_H or int(arch.get("out_dim", -1)) != N_OUT:
		return "architecture mismatch (expected 4-16-16-2)"
	var comp: String = str(arch.get("compound", "")).strip_edges().to_upper()
	if comp != expected_compound.strip_edges().to_upper():
		return "compound mismatch: weights '%s' vs MF62 '%s'" % [comp, expected_compound]
	var cid: String = str(data.get("mf62_calibration_id", ""))
	if cid.is_empty() or cid != expected_calibration_id:
		return "calibration mismatch: weights trained on '%s', MF62 is '%s' (retrain)" % [cid, expected_calibration_id]

	var norm: Variant = data.get("normalization")
	var w: Variant = data.get("weights")
	var g: Variant = data.get("physics_gates")
	if not (norm is Dictionary) or not (w is Dictionary) or not (g is Dictionary):
		return "missing normalization / weights / physics_gates"

	var xm: PackedFloat64Array = _num_array(norm.get("x_mean"), N_IN)
	var xs: PackedFloat64Array = _num_array(norm.get("x_std"), N_IN)
	var ym: PackedFloat64Array = _num_array(norm.get("y_mean"), N_OUT)
	var ys: PackedFloat64Array = _num_array(norm.get("y_std"), N_OUT)
	if xm.size() != N_IN or xs.size() != N_IN or ym.size() != N_OUT or ys.size() != N_OUT:
		return "bad normalization arrays"
	for v in xs:
		if v <= 0.0:
			return "x_std must be > 0"
	for v in ys:
		if v <= 0.0:
			return "y_std must be > 0"

	var w1: PackedFloat64Array = _flatten(w.get("w1"), N_H, N_IN)
	var w2: PackedFloat64Array = _flatten(w.get("w2"), N_H, N_H)
	var w3: PackedFloat64Array = _flatten(w.get("w3"), N_OUT, N_H)
	var b1: PackedFloat64Array = _num_array(w.get("b1"), N_H)
	var b2: PackedFloat64Array = _num_array(w.get("b2"), N_H)
	var b3: PackedFloat64Array = _num_array(w.get("b3"), N_OUT)
	if w1.is_empty() or w2.is_empty() or w3.is_empty() or b1.is_empty() or b2.is_empty() or b3.is_empty():
		return "bad weight / bias shapes"

	for k in GATE_KEYS:
		var gv: Variant = g.get(k)
		if not (typeof(gv) == TYPE_FLOAT or typeof(gv) == TYPE_INT) or not is_finite(float(gv)) or float(gv) <= 0.0:
			return "bad gate '%s'" % k

	_x_mean = xm
	_x_std = xs
	_y_mean = ym
	_y_std = ys
	_w1 = w1
	_b1 = b1
	_w2 = w2
	_b2 = b2
	_w3 = w3
	_b3 = b3
	_slip_scale = float(g["slip_scale_rad"])
	_load_trans = float(g["load_trans_n"])
	_alpha_limit = float(g["alpha_limit_rad"])
	_alpha_slope = float(g["alpha_slope_rad"])
	_fz_limit = float(g["fz_limit_n"])
	_fz_slope = float(g["fz_slope_n"])
	_clip_fy = float(g["clip_fy_n"])
	_clip_mz = float(g["clip_mz_nm"])
	compound = comp
	bound_calibration_id = cid
	return ""


## residual (dfy [n], dmz [nm]). zero when not ready, airborne, or outside the gates
func predict_residual(alpha: float, f_z: float, gamma: float = 0.0, p_kpa: float = 83.0, is_right_side: bool = false) -> Vector2:
	if not is_ready or f_z <= 1e-3:
		return Vector2.ZERO
	var side: float = -1.0 if is_right_side else 1.0
	var a: float = alpha * side
	var g: float = gamma * side

	var x0: float = (a - _x_mean[0]) / _x_std[0]
	var x1: float = (f_z - _x_mean[1]) / _x_std[1]
	var x2: float = (g - _x_mean[2]) / _x_std[2]
	var x3: float = (p_kpa - _x_mean[3]) / _x_std[3]

	for i in range(N_H):
		var o: int = i * N_IN
		_h1[i] = tanh(_b1[i] + _w1[o] * x0 + _w1[o + 1] * x1 + _w1[o + 2] * x2 + _w1[o + 3] * x3)
	for i in range(N_H):
		var o2: int = i * N_H
		var s2: float = _b2[i]
		for j in range(N_H):
			s2 += _w2[o2 + j] * _h1[j]
		_h2[i] = tanh(s2)
	var y0: float = _b3[0]
	var y1: float = _b3[1]
	for j in range(N_H):
		y0 += _w3[j] * _h2[j]
		y1 += _w3[N_H + j] * _h2[j]
	y0 = y0 * _y_std[0] + _y_mean[0]
	y1 = y1 * _y_std[1] + _y_mean[1]

	var g_slip: float = 1.0 - exp(-(a * a + g * g) / (2.0 * _slip_scale * _slip_scale))
	var t: float = clampf(f_z / _load_trans, 0.0, 1.0)
	var g_load: float = t * t * (3.0 - 2.0 * t)
	var a_abs: float = sqrt(a * a + 1e-6)
	var g_alpha: float = 1.0 / (1.0 + exp((a_abs - _alpha_limit) / _alpha_slope))
	var g_fz: float = 1.0 / (1.0 + exp((f_z - _fz_limit) / _fz_slope))
	var gate: float = g_slip * g_load * g_alpha * g_fz

	return Vector2(_clip_fy * tanh(y0 * gate / _clip_fy), _clip_mz * tanh(y1 * gate / _clip_mz)) * side


## backward-compatible wrapper (allocates a dictionary; use predict_residual() in the physics loop)
func predict_correction(alpha: float, f_z: float, gamma: float = 0.0, p_kpa: float = 83.0, is_right_side: bool = false) -> Dictionary:
	var r: Vector2 = predict_residual(alpha, f_z, gamma, p_kpa, is_right_side)
	return {"delta_Fy": r.x, "delta_Mz": r.y, "delta_Fx": 0.0}
