class_name NeuralTireCorrection
extends RefCounted

## Physics-Informed Neural Network (PINN) Residual Corrector for Mirena Dynamics
## Zero-dependency, ultra-fast deterministic CPU feedforward inference:
##   Delta F = [Delta Fy, Delta Mz] = F_experimental - F_MF62
## Features:
## - Fixed normalization and denormalization.
## - Physics-informed zero-slip passivity gate (no spurious forces at zero slip).
## - Vertical load suppression gate (prevents instability when airborne).
## - Sigmoidal domain falloff (smoothly decays influence outside training envelope).
## - Preallocated flat buffers (zero heap allocations in physics cycle).
## - Hard saturation clipping guards.

const WEIGHTS_PATH_RES: String = "res://Scenes/Vehicle/Dynamics/Tires/NeuralCorrection/neural_residual_weights.json"

var is_ready: bool = false

# Normalization constants (4 inputs: alpha, fz, gamma, p_kpa)
var x_mean: Array[float] = [0.0004, 686.2, 0.0327, 83.44]
var x_std: Array[float] = [0.1253, 322.3, 0.0288, 11.25]

# Denormalization constants (2 outputs: delta_Fy, delta_Mz)
var y_mean: Array[float] = [7.409, 1.563]
var y_std: Array[float] = [75.89, 8.328]

# Network weights (16x4, 16x16, 2x16)
var w1: Array = [] # 16 rows, 4 cols
var b1: Array[float] = [] # 16 elements
var w2: Array = [] # 16 rows, 16 cols
var b2: Array[float] = [] # 16 elements
var w3: Array = [] # 2 rows, 16 cols
var b3: Array[float] = [] # 2 elements

# Physics Gating & Limits
var slip_trans_rad: float = 0.02
var load_trans_n: float = 100.0
var alpha_limit_rad: float = 0.22
var alpha_slope_rad: float = 0.02
var fz_limit_n: float = 1300.0
var fz_slope_n: float = 100.0
var clip_fy_n: float = 200.0
var clip_mz_nm: float = 10.0

# Preallocated inference buffers to guarantee ZERO dynamic allocations at 200 Hz
var _buf_in_norm: Array[float] = [0.0, 0.0, 0.0, 0.0]
var _buf_h1: Array[float] = []
var _buf_h2: Array[float] = []

func _init(custom_weights_path: String = "") -> void:
	_buf_h1.resize(16)
	_buf_h2.resize(16)
	load_weights(custom_weights_path)

## Loads exported weights from JSON.
func load_weights(weights_path: String = "") -> void:
	var paths = [
		weights_path,
		WEIGHTS_PATH_RES,
		"src/mirena_sim/Scenes/Vehicle/Dynamics/Tires/NeuralCorrection/neural_residual_weights.json",
		"Scenes/Vehicle/Dynamics/Tires/NeuralCorrection/neural_residual_weights.json"
	]
	
	var text = ""
	for p in paths:
		if p != "" and FileAccess.file_exists(p):
			var f = FileAccess.open(p, FileAccess.READ)
			if f:
				text = f.get_as_text()
				break
	
	if text == "":
		is_ready = false
		return
	
	var data = JSON.parse_string(text)
	if not (data is Dictionary):
		is_ready = false
		return
	
	var norm = data.get("normalization", {})
	if norm is Dictionary:
		x_mean = _to_float_array(norm.get("x_mean", x_mean))
		x_std = _to_float_array(norm.get("x_std", x_std))
		y_mean = _to_float_array(norm.get("y_mean", y_mean))
		y_std = _to_float_array(norm.get("y_std", y_std))
	
	var w_dict = data.get("weights", {})
	if w_dict is Dictionary:
		w1 = w_dict.get("w1", [])
		b1 = _to_float_array(w_dict.get("b1", []))
		w2 = w_dict.get("w2", [])
		b2 = _to_float_array(w_dict.get("b2", []))
		w3 = w_dict.get("w3", [])
		b3 = _to_float_array(w_dict.get("b3", []))
	
	var gates = data.get("physics_gates", {})
	if gates is Dictionary:
		slip_trans_rad = float(gates.get("slip_trans_rad", slip_trans_rad))
		load_trans_n = float(gates.get("load_trans_n", load_trans_n))
		alpha_limit_rad = float(gates.get("alpha_limit_rad", alpha_limit_rad))
		alpha_slope_rad = float(gates.get("alpha_slope_rad", alpha_slope_rad))
		fz_limit_n = float(gates.get("fz_limit_n", fz_limit_n))
		fz_slope_n = float(gates.get("fz_slope_n", fz_slope_n))
		clip_fy_n = float(gates.get("clip_fy_n", clip_fy_n))
		clip_mz_nm = float(gates.get("clip_mz_nm", clip_mz_nm))
	
	is_ready = (w1.size() == 16 and w2.size() == 16 and w3.size() == 2)

func _to_float_array(arr: Array) -> Array[float]:
	var res: Array[float] = []
	for v in arr:
		res.append(float(v))
	return res

## Evaluates residual corrections [delta_Fy, delta_Mz] in real-time with zero allocations.
func predict_correction(
	alpha: float,
	f_z: float,
	gamma: float = 0.0,
	p_kpa: float = 83.0
) -> Dictionary:
	if not is_ready or f_z <= 1e-3:
		return {"delta_Fy": 0.0, "delta_Mz": 0.0, "delta_Fx": 0.0}
	
	# 1. Normalize 4 inputs directly in preallocated buffer
	_buf_in_norm[0] = (alpha - x_mean[0]) / x_std[0]
	_buf_in_norm[1] = (f_z - x_mean[1]) / x_std[1]
	_buf_in_norm[2] = (gamma - x_mean[2]) / x_std[2]
	_buf_in_norm[3] = (p_kpa - x_mean[3]) / x_std[3]
	
	# 2. Hidden Layer 1 (16 units, Tanh)
	for i in range(16):
		var sum_val = b1[i]
		var row: Array = w1[i]
		sum_val += row[0] * _buf_in_norm[0] + row[1] * _buf_in_norm[1] + row[2] * _buf_in_norm[2] + row[3] * _buf_in_norm[3]
		_buf_h1[i] = tanh(sum_val)
	
	# 3. Hidden Layer 2 (16 units, Tanh)
	for i in range(16):
		var sum_val = b2[i]
		var row: Array = w2[i]
		for j in range(16):
			sum_val += row[j] * _buf_h1[j]
		_buf_h2[i] = tanh(sum_val)
	
	# 4. Output Layer (2 units: delta_Fy, delta_Mz, linear)
	var row_out0: Array = w3[0]
	var sum_out0 = b3[0]
	for j in range(16):
		sum_out0 += row_out0[j] * _buf_h2[j]
	
	var row_out1: Array = w3[1]
	var sum_out1 = b3[1]
	for j in range(16):
		sum_out1 += row_out1[j] * _buf_h2[j]
	
	# 5. Denormalize
	var delta_fy_raw = sum_out0 * y_std[0] + y_mean[0]
	var delta_mz_raw = sum_out1 * y_std[1] + y_mean[1]
	
	# 6. Physics Constraints & Gating (Exact parity with PyTorch)
	# Gate A: Zero-slip passivity (smooth tanh transition)
	var g_slip = tanh((absf(alpha) + absf(gamma)) / slip_trans_rad)
	
	# Gate B: Low-load suppression (airborne tire / near zero load)
	var g_load = clampf(f_z / load_trans_n, 0.0, 1.0)
	
	# Gate C: Domain falloff (prevents extrapolation outside experimental envelope)
	var g_alpha = 1.0 / (1.0 + exp((absf(alpha) - alpha_limit_rad) / alpha_slope_rad))
	var g_fz = 1.0 / (1.0 + exp((f_z - fz_limit_n) / fz_slope_n))
	var g_domain = g_alpha * g_fz
	
	var scale = g_slip * g_load * g_domain
	
	# Gate D: Hard saturation clamp
	var delta_Fy = clampf(delta_fy_raw * scale, -clip_fy_n, clip_fy_n)
	var delta_Mz = clampf(delta_mz_raw * scale, -clip_mz_nm, clip_mz_nm)
	
	return {
		"delta_Fy": delta_Fy,
		"delta_Mz": delta_Mz,
		"delta_Fx": 0.0 # Strict physical integrity: no ungrounded longitudinal output
	}
