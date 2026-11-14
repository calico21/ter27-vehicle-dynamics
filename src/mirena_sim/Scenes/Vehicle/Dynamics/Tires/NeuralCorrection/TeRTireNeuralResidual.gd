class_name TeRTireNeuralResidual
extends RefCounted

## Physics-Informed Residual Neural Network (TeR_Tire PINN) over Pacejka MF 6.2.
## Architecture: 5 -> 16 -> 16 -> 3 MLP with tanh activations and pre-allocated
## flat PackedFloat64Array buffers (zero heap allocations on hot path).
##
## Inputs:  [alpha_rad, kappa, fz_n, camber_rad, t_surf_c]
## Outputs: Vector3(delta_Fx_N, delta_Fy_N, delta_Mz_Nm)
##
## Physics-informed structural guarantees:
##   1. Exact algebraic odd symmetry in slip:
##      r(alpha, kappa, ...) = 0.5 * (MLP(alpha, kappa, Fz, gamma, T) - MLP(-alpha, -kappa, Fz, -gamma, T))
##      Strictly satisfies r(-kappa, -alpha, Fz, 0, T) == -r(kappa, alpha, Fz, 0, T)
##      and r(0, 0, Fz, 0, T) == Vector3.ZERO.
##   2. Smooth cubic smoothstep load attenuation as Fz -> 0 (exact zero at Fz <= 0).
##   3. Smooth load-proportional tanh bounding preventing friction-ellipse violations.

const DEFAULT_WEIGHTS_PATH: String = "res://Scenes/Vehicle/Dynamics/Core/Config/ter_tire_pinn_weights.json"
const EDITOR_FALLBACK_PATH: String = "src/mirena_sim/Scenes/Vehicle/Dynamics/Core/Config/ter_tire_pinn_weights.json"
const N_IN: int = 5
const N_H: int = 16
const N_OUT: int = 3

var is_ready: bool = false
var model_name: String = "TeR_Tire_PINN_Residual_v1"

var _x_mean: PackedFloat64Array = PackedFloat64Array()
var _x_std: PackedFloat64Array = PackedFloat64Array()
var _y_mean: PackedFloat64Array = PackedFloat64Array()
var _y_std: PackedFloat64Array = PackedFloat64Array()
var _w1: PackedFloat64Array = PackedFloat64Array()  # N_H x N_IN, row-major
var _b1: PackedFloat64Array = PackedFloat64Array()  # N_H
var _w2: PackedFloat64Array = PackedFloat64Array()  # N_H x N_H, row-major
var _b2: PackedFloat64Array = PackedFloat64Array()  # N_H
var _w3: PackedFloat64Array = PackedFloat64Array()  # N_OUT x N_H, row-major
var _b3: PackedFloat64Array = PackedFloat64Array()  # N_OUT

var _h1: PackedFloat64Array = PackedFloat64Array()
var _h2: PackedFloat64Array = PackedFloat64Array()

var _fz_gate_scale: float = 100.0
var _mu_nom: float = 2.5
var _r0_m: float = 0.2032
var _max_fx_frac: float = 0.20
var _max_fy_frac: float = 0.20
var _max_mz_frac: float = 0.25


func _init(weights_path: String = "") -> void:
	_h1.resize(N_H)
	_h2.resize(N_H)
	load_weights(weights_path)


## Loads TeR_Tire PINN weights JSON from `path` or the default Core/Config path.
func load_weights(path: String = "") -> bool:
	var resolved: String = path
	if resolved.is_empty():
		for candidate in [DEFAULT_WEIGHTS_PATH, EDITOR_FALLBACK_PATH]:
			if FileAccess.file_exists(candidate):
				resolved = candidate
				break
	if resolved.is_empty() or not FileAccess.file_exists(resolved):
		push_error("TeRTireNeuralResidual: weights file not found (requested '%s')." % path)
		_clear()
		return false

	var f: FileAccess = FileAccess.open(resolved, FileAccess.READ)
	if f == null:
		push_error("TeRTireNeuralResidual: cannot open '%s' (error %d)." % [resolved, FileAccess.get_open_error()])
		_clear()
		return false

	var parsed: Variant = JSON.parse_string(f.get_as_text())
	if not (parsed is Dictionary):
		push_error("TeRTireNeuralResidual: invalid JSON in '%s'." % resolved)
		_clear()
		return false

	if not load_from_dict(parsed as Dictionary):
		push_error("TeRTireNeuralResidual: failed validation in '%s'." % resolved)
		return false
	return true


## Loads and validates weights, normalization, and physics bounds from an in-memory Dictionary.
func load_from_dict(data: Dictionary) -> bool:
	_clear()

	var arch: Variant = data.get("architecture")
	if not (arch is Dictionary):
		return false
	var arch_d: Dictionary = arch
	if int(arch_d.get("in_dim", -1)) != N_IN or int(arch_d.get("hidden1", -1)) != N_H \
			or int(arch_d.get("hidden2", -1)) != N_H or int(arch_d.get("out_dim", -1)) != N_OUT:
		return false

	var norm: Variant = data.get("normalization")
	var weights: Variant = data.get("weights")
	if not (norm is Dictionary) or not (weights is Dictionary):
		return false
	var norm_d: Dictionary = norm
	var w_d: Dictionary = weights

	var xm: PackedFloat64Array = _num_array(norm_d.get("x_mean"), N_IN)
	var xs: PackedFloat64Array = _num_array(norm_d.get("x_std"), N_IN)
	var ym: PackedFloat64Array = _num_array(norm_d.get("y_mean"), N_OUT)
	var ys: PackedFloat64Array = _num_array(norm_d.get("y_std"), N_OUT)
	if xm.size() != N_IN or xs.size() != N_IN or ym.size() != N_OUT or ys.size() != N_OUT:
		return false
	for v in xs:
		if v <= 0.0:
			return false
	for v in ys:
		if v <= 0.0:
			return false

	var w1: PackedFloat64Array = _flatten(w_d.get("w1"), N_H, N_IN)
	var w2: PackedFloat64Array = _flatten(w_d.get("w2"), N_H, N_H)
	var w3: PackedFloat64Array = _flatten(w_d.get("w3"), N_OUT, N_H)
	var b1: PackedFloat64Array = _num_array(w_d.get("b1"), N_H)
	var b2: PackedFloat64Array = _num_array(w_d.get("b2"), N_H)
	var b3: PackedFloat64Array = _num_array(w_d.get("b3"), N_OUT)
	if w1.is_empty() or w2.is_empty() or w3.is_empty() or b1.is_empty() or b2.is_empty() or b3.is_empty():
		return false

	var fz_gate_scale: float = 100.0
	var mu_nom: float = 2.5
	var r0_m: float = 0.2032
	var max_fx_frac: float = 0.20
	var max_fy_frac: float = 0.20
	var max_mz_frac: float = 0.25

	if data.has("physics_bounds"):
		var pb: Variant = data.get("physics_bounds")
		if not (pb is Dictionary):
			return false
		var pb_d: Dictionary = pb
		fz_gate_scale = _pos_float(pb_d.get("fz_gate_scale_n", fz_gate_scale))
		mu_nom = _pos_float(pb_d.get("mu_nom", mu_nom))
		r0_m = _pos_float(pb_d.get("r0_m", r0_m))
		max_fx_frac = _pos_float(pb_d.get("max_fx_frac", max_fx_frac))
		max_fy_frac = _pos_float(pb_d.get("max_fy_frac", max_fy_frac))
		max_mz_frac = _pos_float(pb_d.get("max_mz_frac", max_mz_frac))
		if fz_gate_scale <= 0.0 or mu_nom <= 0.0 or r0_m <= 0.0 \
				or max_fx_frac <= 0.0 or max_fy_frac <= 0.0 or max_mz_frac <= 0.0:
			return false

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
	_fz_gate_scale = fz_gate_scale
	_mu_nom = mu_nom
	_r0_m = r0_m
	_max_fx_frac = max_fx_frac
	_max_fy_frac = max_fy_frac
	_max_mz_frac = max_mz_frac
	model_name = str(data.get("model_name", "TeR_Tire_PINN_Residual_v1"))
	is_ready = true
	return true


## Evaluates physics-informed residual Vector3(delta_Fx_N, delta_Fy_N, delta_Mz_Nm).
## Hot path: zero heap allocations, exact odd symmetry in (alpha, kappa), smooth Fz -> 0 gate.
func predict_residual(
	kappa: float,
	alpha: float,
	f_z: float,
	gamma: float = 0.0,
	t_surf_c: float = 58.0,
	is_right_side: bool = false
) -> Vector3:
	if not is_ready or f_z <= 0.0 or not is_finite(f_z):
		return Vector3.ZERO
	if not is_finite(kappa) or not is_finite(alpha) or not is_finite(gamma) or not is_finite(t_surf_c):
		return Vector3.ZERO

	var side: float = -1.0 if is_right_side else 1.0
	var a_in: float = alpha * side
	var k_in: float = kappa
	var g_in: float = gamma * side

	var xm0: float = _x_mean[0]
	var xm1: float = _x_mean[1]
	var xm2: float = _x_mean[2]
	var xm3: float = _x_mean[3]
	var xm4: float = _x_mean[4]
	var xs0: float = _x_std[0]
	var xs1: float = _x_std[1]
	var xs2: float = _x_std[2]
	var xs3: float = _x_std[3]
	var xs4: float = _x_std[4]

	# Shared load/temp normalized channels
	var x2_c: float = (f_z - xm2) / xs2
	var x4_c: float = (t_surf_c - xm4) / xs4

	# Pass 1: (+alpha, +kappa, +gamma)
	var x0_p: float = (a_in - xm0) / xs0
	var x1_p: float = (k_in - xm1) / xs1
	var x3_p: float = (g_in - xm3) / xs3

	for i in N_H:
		var o: int = i * N_IN
		_h1[i] = tanh(_b1[i] + _w1[o] * x0_p + _w1[o + 1] * x1_p + _w1[o + 2] * x2_c + _w1[o + 3] * x3_p + _w1[o + 4] * x4_c)
	for i in N_H:
		var o2: int = i * N_H
		var s2: float = _b2[i]
		for j in N_H:
			s2 += _w2[o2 + j] * _h1[j]
		_h2[i] = tanh(s2)
	var y0_p: float = _b3[0]
	var y1_p: float = _b3[1]
	var y2_p: float = _b3[2]
	for j in N_H:
		var h2j: float = _h2[j]
		y0_p += _w3[j] * h2j
		y1_p += _w3[N_H + j] * h2j
		y2_p += _w3[2 * N_H + j] * h2j

	# Pass 2: (-alpha, -kappa, -gamma) for exact algebraic antisymmetry
	var x0_n: float = (-a_in - xm0) / xs0
	var x1_n: float = (-k_in - xm1) / xs1
	var x3_n: float = (-g_in - xm3) / xs3

	for i in N_H:
		var o: int = i * N_IN
		_h1[i] = tanh(_b1[i] + _w1[o] * x0_n + _w1[o + 1] * x1_n + _w1[o + 2] * x2_c + _w1[o + 3] * x3_n + _w1[o + 4] * x4_c)
	for i in N_H:
		var o2: int = i * N_H
		var s2: float = _b2[i]
		for j in N_H:
			s2 += _w2[o2 + j] * _h1[j]
		_h2[i] = tanh(s2)
	var y0_n: float = _b3[0]
	var y1_n: float = _b3[1]
	var y2_n: float = _b3[2]
	for j in N_H:
		var h2j: float = _h2[j]
		y0_n += _w3[j] * h2j
		y1_n += _w3[N_H + j] * h2j
		y2_n += _w3[2 * N_H + j] * h2j

	var raw_fx: float = 0.5 * (y0_p - y0_n) * _y_std[0]
	var raw_fy: float = 0.5 * (y1_p - y1_n) * _y_std[1] * side
	var raw_mz: float = 0.5 * (y2_p - y2_n) * _y_std[2] * side

	# Smooth Fz -> 0 cubic smoothstep attenuation (O(Fz^2) near zero)
	var u: float = clampf(f_z / _fz_gate_scale, 0.0, 1.0)
	var fz_gate: float = u * u * (3.0 - 2.0 * u)

	# Smooth load-proportional physical bounding
	var max_fx: float = maxf(_max_fx_frac * f_z * _mu_nom, 1e-6)
	var max_fy: float = maxf(_max_fy_frac * f_z * _mu_nom, 1e-6)
	var max_mz: float = maxf(_max_mz_frac * f_z * _mu_nom * _r0_m, 1e-6)

	return Vector3(
		fz_gate * max_fx * tanh(raw_fx / max_fx),
		fz_gate * max_fy * tanh(raw_fy / max_fy),
		fz_gate * max_mz * tanh(raw_mz / max_mz)
	)


func _clear() -> void:
	is_ready = false


func _pos_float(v: Variant) -> float:
	if not (typeof(v) == TYPE_FLOAT or typeof(v) == TYPE_INT):
		return -1.0
	var f: float = float(v)
	return f if (is_finite(f) and f > 0.0) else -1.0


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
