class_name RoadSurface
extends RefCounted

## Deterministic road surface elevation & velocity synthesizer (ISO 8608 + Pronounced Bumps).
## Zero dynamic heap allocations during sample_world() for real-time 400 Hz simulation.

enum SurfaceMode {
	FLAT = 0,
	ISO_A = 1,
	ISO_B = 2,
	ISO_C = 3,
	ISO_B_CURBS = 4,
}

enum BumpMode {
	OFF = 0,
	SPEED_BUMPS = 1,
	ASYM_POTHOLES = 2,
	APEX_CURBS = 3,
}

const SURFACE_NAMES: Array[String] = [
	"Flat (Baseline)",
	"ISO 8608 Class A",
	"ISO 8608 Class B",
	"ISO 8608 Class C",
	"ISO B + Curbs",
]

const BUMP_NAMES: Array[String] = [
	"Off",
	"Speed Bumps",
	"Asym Potholes",
	"Apex Curbs",
]

const NUM_HARMONICS: int = 24
const N0_CYCLES_M: float = 0.1
const N_MIN_CYCLES_M: float = 0.04
const N_MAX_CYCLES_M: float = 2.5

# ISO 8608 Gd(n0) in m^3 at n0 = 0.1 cycles/m (waviness w = 2.0)
const GD_ISO_A: float = 16.0e-6
const GD_ISO_B: float = 64.0e-6
const GD_ISO_C: float = 256.0e-6

var surface_mode: int = SurfaceMode.FLAT
var roughness_scale: float = 1.0
var bump_mode: int = BumpMode.OFF
var bump_height_m: float = 0.025
var bump_length_m: float = 0.80
var bump_spacing_m: float = 35.0
var pronounced_bumps_enabled: bool = false

# Precomputed ISO spatial harmonics: kx[k], ky[k], amp_unit[k], phase[k]
var _kx: PackedFloat64Array = PackedFloat64Array()
var _ky: PackedFloat64Array = PackedFloat64Array()
var _amp_unit: PackedFloat64Array = PackedFloat64Array()
var _phase: PackedFloat64Array = PackedFloat64Array()

# Precomputed discrete bump markers in ROS XY world frame (up to 64 active bumps)
const MAX_BUMPS: int = 64
var _bump_count: int = 0
var _bx: PackedFloat64Array = PackedFloat64Array()
var _by: PackedFloat64Array = PackedFloat64Array()
var _btx: PackedFloat64Array = PackedFloat64Array()
var _bty: PackedFloat64Array = PackedFloat64Array()
var _b_amp: PackedFloat64Array = PackedFloat64Array()
var _b_len: PackedFloat64Array = PackedFloat64Array()
var _b_lat_off: PackedFloat64Array = PackedFloat64Array()
var _b_half_w: PackedFloat64Array = PackedFloat64Array()

# Optional track polyline in ROS XY
var _trk_x: PackedFloat64Array = PackedFloat64Array()
var _trk_y: PackedFloat64Array = PackedFloat64Array()
var _trk_s: PackedFloat64Array = PackedFloat64Array()
var _trk_k: PackedFloat64Array = PackedFloat64Array()
var _trk_n: int = 0


func _init() -> void:
	_init_harmonics()
	_allocate_bump_buffers()
	rebuild_bumps()


func _init_harmonics() -> void:
	_kx.resize(NUM_HARMONICS)
	_ky.resize(NUM_HARMONICS)
	_amp_unit.resize(NUM_HARMONICS)
	_phase.resize(NUM_HARMONICS)

	var rng := RandomNumberGenerator.new()
	rng.seed = 20270427

	var log_min := log(N_MIN_CYCLES_M)
	var log_max := log(N_MAX_CYCLES_M)
	var d_log := (log_max - log_min) / float(NUM_HARMONICS)
	for i in range(NUM_HARMONICS):
		var n_lo := exp(log_min + float(i) * d_log)
		var n_hi := exp(log_min + float(i + 1) * d_log)
		var n_c := 0.5 * (n_lo + n_hi)
		var dn := n_hi - n_lo
		# Unit spectrum Gd(n) / Gd(n0) = (n / n0)^(-2)
		var psd_norm := pow(n_c / N0_CYCLES_M, -2.0)
		_amp_unit[i] = sqrt(2.0 * psd_norm * dn)
		var theta := rng.randf_range(0.0, TAU)
		var k_mag := TAU * n_c
		_kx[i] = k_mag * cos(theta)
		_ky[i] = k_mag * sin(theta)
		_phase[i] = rng.randf_range(0.0, TAU)


func _allocate_bump_buffers() -> void:
	_bx.resize(MAX_BUMPS)
	_by.resize(MAX_BUMPS)
	_btx.resize(MAX_BUMPS)
	_bty.resize(MAX_BUMPS)
	_b_amp.resize(MAX_BUMPS)
	_b_len.resize(MAX_BUMPS)
	_b_lat_off.resize(MAX_BUMPS)
	_b_half_w.resize(MAX_BUMPS)


func configure(
	p_surface_mode: int,
	p_roughness_scale: float = 1.0,
	p_bump_mode: int = BumpMode.OFF,
	p_bump_height_mm: float = 25.0,
	p_bump_length_m: float = 0.80,
	p_bump_spacing_m: float = 35.0
) -> void:
	surface_mode = clampi(p_surface_mode, 0, SurfaceMode.ISO_B_CURBS)
	roughness_scale = clampf(p_roughness_scale, 0.0, 3.0)
	bump_mode = clampi(p_bump_mode, 0, BumpMode.APEX_CURBS)
	pronounced_bumps_enabled = (bump_mode != BumpMode.OFF) or (surface_mode == SurfaceMode.ISO_B_CURBS)
	bump_height_m = clampf(p_bump_height_mm * 0.001, 0.0, 0.060)
	bump_length_m = clampf(p_bump_length_m, 0.30, 2.50)
	bump_spacing_m = clampf(p_bump_spacing_m, 15.0, 80.0)
	rebuild_bumps()


func set_track_polyline_ros(pts_x: PackedFloat64Array, pts_y: PackedFloat64Array) -> void:
	var n := mini(pts_x.size(), pts_y.size())
	if n < 3:
		_trk_n = 0
		rebuild_bumps()
		return
	_trk_n = n
	_trk_x = pts_x.duplicate()
	_trk_y = pts_y.duplicate()
	_trk_s.resize(n)
	_trk_k.resize(n)
	var s_acc := 0.0
	_trk_s[0] = 0.0
	for i in range(1, n):
		var dx_s := _trk_x[i] - _trk_x[i - 1]
		var dy_s := _trk_y[i] - _trk_y[i - 1]
		s_acc += sqrt(dx_s * dx_s + dy_s * dy_s)
		_trk_s[i] = s_acc
	for i in range(n):
		var im := (i - 1 + n) % n
		var ip := (i + 1) % n
		var dx1 := _trk_x[i] - _trk_x[im]
		var dy1 := _trk_y[i] - _trk_y[im]
		var dx2 := _trk_x[ip] - _trk_x[i]
		var dy2 := _trk_y[ip] - _trk_y[i]
		var ds1 := maxf(sqrt(dx1 * dx1 + dy1 * dy1), 0.1)
		var ds2 := maxf(sqrt(dx2 * dx2 + dy2 * dy2), 0.1)
		var dpsi := wrapf(atan2(dy2, dx2) - atan2(dy1, dx1), -PI, PI)
		_trk_k[i] = dpsi / (0.5 * (ds1 + ds2))
	rebuild_bumps()


func clear_track_polyline() -> void:
	_trk_n = 0
	rebuild_bumps()


func is_active() -> bool:
	return (surface_mode != SurfaceMode.FLAT) or pronounced_bumps_enabled


func _iso_gd() -> float:
	match surface_mode:
		SurfaceMode.ISO_A:
			return GD_ISO_A
		SurfaceMode.ISO_B, SurfaceMode.ISO_B_CURBS:
			return GD_ISO_B
		SurfaceMode.ISO_C:
			return GD_ISO_C
		_:
			return 0.0


func rebuild_bumps() -> void:
	_bump_count = 0
	var eff_bump_mode := bump_mode
	if eff_bump_mode == BumpMode.OFF and surface_mode == SurfaceMode.ISO_B_CURBS:
		eff_bump_mode = BumpMode.APEX_CURBS
	if eff_bump_mode == BumpMode.OFF:
		return

	var h_m := bump_height_m
	var l_m := bump_length_m
	if eff_bump_mode == BumpMode.APEX_CURBS and bump_mode == BumpMode.OFF:
		h_m = 0.028
		l_m = 0.65

	if _trk_n >= 4:
		var total_len := _trk_s[_trk_n - 1]
		if eff_bump_mode == BumpMode.APEX_CURBS:
			var next_s := 10.0
			for i in range(1, _trk_n - 1):
				if _bump_count >= MAX_BUMPS:
					break
				if _trk_s[i] < next_s:
					continue
				if absf(_trk_k[i]) > 0.08:
					var tx := _trk_x[i + 1] - _trk_x[i - 1]
					var ty := _trk_y[i + 1] - _trk_y[i - 1]
					var norm := maxf(sqrt(tx * tx + ty * ty), 1e-4)
					tx /= norm
					ty /= norm
					# Inside apex side: +1 left (+Y) for positive curvature, -1 right for negative curvature
					var sgn_in := 1.0 if _trk_k[i] > 0.0 else -1.0
					_bx[_bump_count] = _trk_x[i]
					_by[_bump_count] = _trk_y[i]
					_btx[_bump_count] = tx
					_bty[_bump_count] = ty
					_b_amp[_bump_count] = maxf(h_m, 0.028) if surface_mode == SurfaceMode.ISO_B_CURBS and bump_mode == BumpMode.OFF else h_m
					_b_len[_bump_count] = maxf(l_m, 0.60)
					_b_lat_off[_bump_count] = sgn_in * 1.35
					_b_half_w[_bump_count] = 0.85
					_bump_count += 1
					next_s = _trk_s[i] + maxf(bump_spacing_m * 0.65, 18.0)
		else:
			var s_target := 18.0
			var idx_b := 0
			for i in range(1, _trk_n - 1):
				if _bump_count >= MAX_BUMPS or s_target >= total_len - 5.0:
					break
				if _trk_s[i] >= s_target:
					var tx := _trk_x[i + 1] - _trk_x[i - 1]
					var ty := _trk_y[i + 1] - _trk_y[i - 1]
					var norm := maxf(sqrt(tx * tx + ty * ty), 1e-4)
					tx /= norm
					ty /= norm
					_bx[_bump_count] = _trk_x[i]
					_by[_bump_count] = _trk_y[i]
					_btx[_bump_count] = tx
					_bty[_bump_count] = ty
					_b_amp[_bump_count] = h_m
					_b_len[_bump_count] = l_m
					if eff_bump_mode == BumpMode.ASYM_POTHOLES:
						_b_lat_off[_bump_count] = 0.62 if (idx_b % 2 == 0) else -0.62
						_b_half_w[_bump_count] = 0.68
					else:
						_b_lat_off[_bump_count] = 0.0
						_b_half_w[_bump_count] = 4.0
					_bump_count += 1
					idx_b += 1
					s_target += bump_spacing_m
	else:
		# Deterministic straight-line / grid bumps along +X ROS axis when no track polyline is set
		var n_straight := mini(12, MAX_BUMPS)
		for k in range(n_straight):
			var x_k := 15.0 + float(k) * bump_spacing_m
			_bx[k] = x_k
			_by[k] = 0.0
			_btx[k] = 1.0
			_bty[k] = 0.0
			_b_amp[k] = h_m
			_b_len[k] = l_m
			if eff_bump_mode == BumpMode.ASYM_POTHOLES:
				_b_lat_off[k] = 0.62 if (k % 2 == 0) else -0.62
				_b_half_w[k] = 0.68
			elif eff_bump_mode == BumpMode.APEX_CURBS:
				_b_lat_off[k] = 1.15 if (k % 2 == 0) else -1.15
				_b_half_w[k] = 0.75
			else:
				_b_lat_off[k] = 0.0
				_b_half_w[k] = 4.0
		_bump_count = n_straight


## Evaluates vertical road profile z_m (upwards positive) and time derivative dz_dt_mps
## at world ROS position (x_ros, y_ros) and wheel ground velocity (vx_ros, vy_ros).
func sample_world(x_ros: float, y_ros: float, vx_ros: float, vy_ros: float) -> Vector2:
	if surface_mode == SurfaceMode.FLAT and not pronounced_bumps_enabled:
		return Vector2.ZERO

	var z_tot := 0.0
	var dz_dt := 0.0

	var gd := _iso_gd()
	if gd > 0.0 and roughness_scale > 0.0:
		var amp_gain := sqrt(gd) * roughness_scale
		for i in range(NUM_HARMONICS):
			var kx_i := _kx[i]
			var ky_i := _ky[i]
			var a_i := amp_gain * _amp_unit[i]
			var arg := kx_i * x_ros + ky_i * y_ros + _phase[i]
			z_tot += a_i * cos(arg)
			dz_dt += -a_i * sin(arg) * (kx_i * vx_ros + ky_i * vy_ros)

	if pronounced_bumps_enabled and _bump_count > 0:
		for b in range(_bump_count):
			var rx := x_ros - _bx[b]
			var ry := y_ros - _by[b]
			var tx := _btx[b]
			var ty := _bty[b]
			var d_long := rx * tx + ry * ty
			var l_b := _b_len[b]
			if absf(d_long) >= l_b:
				continue
			# Lateral normal (-ty, tx) points left (+Y ROS when tx=1, ty=0)
			var d_lat := rx * (-ty) + ry * tx
			var lat_rel := absf(d_lat - _b_lat_off[b])
			var hw := _b_half_w[b]
			if lat_rel >= hw:
				continue
			var lat_w := 1.0
			if hw < 2.0:
				lat_w = 0.5 * (1.0 + cos(PI * lat_rel / hw))
			var amp_b := _b_amp[b] * lat_w
			var u := PI * d_long / l_b
			z_tot += amp_b * 0.5 * (1.0 + cos(u))
			var v_long := vx_ros * tx + vy_ros * ty
			dz_dt += amp_b * (-0.5 * (PI / l_b) * sin(u)) * v_long

	return Vector2(z_tot, dz_dt)


func get_bump_markers() -> Array:
	var out: Array = []
	if not pronounced_bumps_enabled:
		return out
	for b in range(_bump_count):
		out.append({
			"x_ros": _bx[b],
			"y_ros": _by[b],
			"tx": _btx[b],
			"ty": _bty[b],
			"amp_m": _b_amp[b],
			"len_m": _b_len[b],
			"lat_off": _b_lat_off[b],
			"half_w": _b_half_w[b],
		})
	return out
