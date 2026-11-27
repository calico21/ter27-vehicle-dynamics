class_name DoubleWishboneKinematics
extends RefCounted

## 3D double-wishbone + pushrod + rocker + ARB kinematic & elastokinematic solver
## Coordinate convention: ISO 8855 / ROS REP-103 (+X forward, +Y left, +Z up)
## Hardpoints origin: front axle center at ground level (0, 0, 0)
##
## Two-level architecture:
##   1. Offline / initialization: exact 3D Newton-Raphson loop closure (tol < 1e-9 m)
##      and 2D (front) / 1D (rear) LUT precomputation with rack-to-mean-steer calibration.
##   2. Runtime (400 Hz): zero-allocation C1 Catmull-Rom cubic interpolation on precomputed
##      PackedFloat64Array LUTs, mirrored symmetrically for right-side corners.

const LUT_DZ_MIN: float = -0.045
const LUT_DZ_MAX: float = 0.045
const LUT_NZ: int = 37
const LUT_STEER_MIN: float = -0.45
const LUT_STEER_MAX: float = 0.45
const LUT_NS: int = 37

const CH_CAMBER: int = 0
const CH_WHEEL_STEER: int = 1
const CH_TOE: int = 2
const CH_CP_DX: int = 3
const CH_CP_DY: int = 4
const CH_DAMPER_DISP: int = 5
const CH_MR: int = 6
const CH_DMR_DZ: int = 7
const CH_ARB_ANGLE: int = 8
const CH_ARB_MR: int = 9
const CH_FVSA_IC_Y: int = 10
const CH_FVSA_IC_Z: int = 11
const CH_FVSA_SLOPE: int = 12
const CH_RC_Z: int = 13
const CH_SVSA_SLOPE: int = 14
const CH_MECH_TRAIL: int = 15
const CH_SCRUB_RADIUS: int = 16
const STRIDE: int = 17

const LINK_NAMES: Array[String] = [
	"lca_front", "lca_rear", "lbj",
	"uca_front", "uca_rear", "ubj",
	"pushrod_wishbone", "pushrod_rocker",
	"tie_outer", "tie_inner",
	"wheel_center", "contact_patch",
	"damper_chassis", "damper_rocker",
	"rocker_axis_p1", "rocker_pivot",
	"arb_link_rocker", "arb_arm_tip", "arb_chassis_piv"
]

# Shared static cache keyed by canonical hardpoints fingerprint so loading/resetting
# the same vehicle config is instantaneous (< 0.05 ms) after the first build.
static var _shared_lut_cache: Dictionary = {}


class CornerGeometry extends RefCounted:
	var is_front: bool = true
	var is_right: bool = false
	var pushrod_on_lower: bool = false
	var static_camber_deg: float = 0.0
	var static_toe_deg: float = 0.0
	var wheel_radius: float = 0.2032

	var lca_front: Vector3 = Vector3.ZERO
	var lca_rear: Vector3 = Vector3.ZERO
	var lbj: Vector3 = Vector3.ZERO
	var uca_front: Vector3 = Vector3.ZERO
	var uca_rear: Vector3 = Vector3.ZERO
	var ubj: Vector3 = Vector3.ZERO
	var pushrod_wishbone: Vector3 = Vector3.ZERO
	var pushrod_rocker: Vector3 = Vector3.ZERO
	var tie_outer: Vector3 = Vector3.ZERO
	var tie_inner: Vector3 = Vector3.ZERO
	var wheel_center: Vector3 = Vector3.ZERO
	var contact_patch: Vector3 = Vector3.ZERO
	var damper_chassis: Vector3 = Vector3.ZERO
	var damper_rocker: Vector3 = Vector3.ZERO
	var rocker_axis_p1: Vector3 = Vector3.ZERO
	var rocker_pivot: Vector3 = Vector3.ZERO
	var arb_link_rocker: Vector3 = Vector3.ZERO
	var arb_arm_tip: Vector3 = Vector3.ZERO
	var arb_chassis_piv: Vector3 = Vector3.ZERO

	# Precomputed unit axes & rigid link lengths
	var u_lca: Vector3 = Vector3(1.0, 0.0, 0.0)
	var u_uca: Vector3 = Vector3(1.0, 0.0, 0.0)
	var u_rck: Vector3 = Vector3(0.0, 1.0, 0.0)
	var u_arb: Vector3 = Vector3(0.0, 1.0, 0.0)
	var lca_mid: Vector3 = Vector3.ZERO
	var uca_mid: Vector3 = Vector3.ZERO

	var len_kp: float = 0.0
	var len_tie: float = 0.0
	var len_pr: float = 0.0
	var len_dm0: float = 0.0
	var len_al: float = 0.0

	# Static upright basis (columns = ex0, ey0, ez0) and upright-local coordinates
	var r_u0: Basis = Basis.IDENTITY
	var ez0: Vector3 = Vector3.UP
	var wc_loc: Vector3 = Vector3.ZERO
	var to_loc: Vector3 = Vector3.ZERO
	var spin_loc: Vector3 = Vector3(0.0, 1.0, 0.0)
	var arb_jounce_sign: float = 1.0

	func init_from_dict(d: Dictionary, front: bool, right: bool, r_wheel: float) -> bool:
		for k in LINK_NAMES:
			if not d.has(k):
				return false
		is_front = front
		is_right = right
		pushrod_on_lower = bool(d.get("pushrod_on_lower", false))
		static_camber_deg = float(d.get("static_camber_deg", 0.0))
		static_toe_deg = float(d.get("static_toe_deg", 0.0))
		wheel_radius = maxf(r_wheel, 0.05)

		var y_sgn: float = -1.0 if right else 1.0
		lca_front = _to_v3(d["lca_front"], y_sgn)
		lca_rear = _to_v3(d["lca_rear"], y_sgn)
		lbj = _to_v3(d["lbj"], y_sgn)
		uca_front = _to_v3(d["uca_front"], y_sgn)
		uca_rear = _to_v3(d["uca_rear"], y_sgn)
		ubj = _to_v3(d["ubj"], y_sgn)
		pushrod_wishbone = _to_v3(d["pushrod_wishbone"], y_sgn)
		pushrod_rocker = _to_v3(d["pushrod_rocker"], y_sgn)
		tie_outer = _to_v3(d["tie_outer"], y_sgn)
		tie_inner = _to_v3(d["tie_inner"], y_sgn)
		wheel_center = _to_v3(d["wheel_center"], y_sgn)
		contact_patch = _to_v3(d["contact_patch"], y_sgn)
		damper_chassis = _to_v3(d["damper_chassis"], y_sgn)
		damper_rocker = _to_v3(d["damper_rocker"], y_sgn)
		rocker_axis_p1 = _to_v3(d["rocker_axis_p1"], y_sgn)
		rocker_pivot = _to_v3(d["rocker_pivot"], y_sgn)
		arb_link_rocker = _to_v3(d["arb_link_rocker"], y_sgn)
		arb_arm_tip = _to_v3(d["arb_arm_tip"], y_sgn)
		arb_chassis_piv = _to_v3(d["arb_chassis_piv"], y_sgn)

		u_lca = (lca_rear - lca_front).normalized()
		u_uca = (uca_rear - uca_front).normalized()
		u_rck = (rocker_axis_p1 - rocker_pivot).normalized()
		u_arb = Vector3(0.0, 1.0, 0.0)
		lca_mid = 0.5 * (lca_front + lca_rear)
		uca_mid = 0.5 * (uca_front + uca_rear)

		len_kp = (ubj - lbj).length()
		len_tie = (tie_outer - tie_inner).length()
		len_pr = (pushrod_wishbone - pushrod_rocker).length()
		len_dm0 = (damper_chassis - damper_rocker).length()
		len_al = (arb_arm_tip - arb_link_rocker).length()
		if len_kp < 1e-4 or len_tie < 1e-4 or len_pr < 1e-4 or len_dm0 < 1e-4 or len_al < 1e-4:
			return false

		var s_ax_0 := Vector3(0.0, 1.0, 0.0)
		if absf(static_camber_deg) > 1.0e-7 or absf(static_toe_deg) > 1.0e-7:
			var g0: float = deg_to_rad(static_camber_deg)
			var t0: float = deg_to_rad(static_toe_deg)
			var steer0: float = t0 if right else -t0
			s_ax_0 = Vector3(
				-sin(steer0) * cos(g0),
				cos(steer0) * cos(g0),
				sin(g0) if right else -sin(g0)
			).normalized()
			var down := Vector3(0.0, 0.0, -1.0)
			var rad_dn_0: Vector3 = (down - down.dot(s_ax_0) * s_ax_0).normalized()
			wheel_center = contact_patch - wheel_radius * rad_dn_0

		ez0 = (ubj - lbj) / len_kp
		var ey0: Vector3 = ez0.cross(tie_outer - lbj).normalized()
		var ex0: Vector3 = ey0.cross(ez0).normalized()
		r_u0 = Basis(ex0, ey0, ez0)
		var r_u0_t: Basis = r_u0.transposed()
		wc_loc = r_u0_t * (wheel_center - lbj)
		to_loc = r_u0_t * (tie_outer - lbj)
		# Wheel spin axis points car-left (+Y) at static design pose (tilted by static camber/toe if configured)
		spin_loc = r_u0_t * s_ax_0
		arb_jounce_sign = 1.0
		return true

	static func _to_v3(arr: Variant, y_sgn: float) -> Vector3:
		if arr is Vector3:
			var v: Vector3 = arr
			return Vector3(v.x, v.y * y_sgn, v.z)
		if arr is Array and arr.size() >= 3:
			return Vector3(float(arr[0]), float(arr[1]) * y_sgn, float(arr[2]))
		return Vector3.ZERO


var raw_dict: Dictionary = {}
var wheelbase_m: float = 1.540
var track_front_m: float = 1.230
var track_rear_m: float = 1.230
var wheel_radius_m: float = 0.2032
var h_cog_m: float = 0.290

var front_left: CornerGeometry = CornerGeometry.new()
var front_right: CornerGeometry = CornerGeometry.new()
var rear_left: CornerGeometry = CornerGeometry.new()
var rear_right: CornerGeometry = CornerGeometry.new()

var _configured: bool = false
var _ready: bool = false
var _cache_key: String = ""

var front_lut: PackedFloat64Array = PackedFloat64Array()
var rear_lut: PackedFloat64Array = PackedFloat64Array()
var rack_disp_table: PackedFloat64Array = PackedFloat64Array()

var mr_0_front: float = 0.885
var mr_0_rear: float = 0.885
var arb_mr_0_front: float = 1.0
var arb_mr_0_rear: float = 1.0
var static_metrics_front: Dictionary = {}
var static_metrics_rear: Dictionary = {}

var _inv_dz_step: float = float(LUT_NZ - 1) / (LUT_DZ_MAX - LUT_DZ_MIN)
var _inv_steer_step: float = float(LUT_NS - 1) / (LUT_STEER_MAX - LUT_STEER_MIN)


func is_ready() -> bool:
	return _ready


func configure(kin_dict: Dictionary, p_wheel_radius: float = -1.0, p_wheelbase: float = -1.0, p_h_cog: float = -1.0) -> bool:
	_configured = false
	_ready = false
	if kin_dict.is_empty():
		return false
	if not (kin_dict.get("front_left") is Dictionary and kin_dict.get("rear_left") is Dictionary):
		return false

	raw_dict = kin_dict
	wheelbase_m = p_wheelbase if p_wheelbase > 0.0 else float(kin_dict.get("wheelbase_m", 1.540))
	track_front_m = float(kin_dict.get("track_front_m", 1.230))
	track_rear_m = float(kin_dict.get("track_rear_m", 1.230))
	wheel_radius_m = p_wheel_radius if p_wheel_radius > 0.0 else float(kin_dict.get("wheel_radius_m", 0.2032))
	h_cog_m = p_h_cog if p_h_cog > 0.0 else 0.290

	var fl_d: Dictionary = kin_dict["front_left"]
	var rl_d: Dictionary = kin_dict["rear_left"]
	if not front_left.init_from_dict(fl_d, true, false, wheel_radius_m):
		return false
	if not front_right.init_from_dict(fl_d, true, true, wheel_radius_m):
		return false
	if not rear_left.init_from_dict(rl_d, false, false, wheel_radius_m):
		return false
	if not rear_right.init_from_dict(rl_d, false, true, wheel_radius_m):
		return false

	# Determine ARB jounce sign at +5 mm compression so d(arb_angle)/d(delta_z) > 0
	var p_fl0 := _solve_corner_raw(front_left, 0.005, 0.0, Vector3.ZERO)
	if float(p_fl0.get("arb_angle_rad", 0.0)) < 0.0:
		front_left.arb_jounce_sign = -1.0
	var p_fr0 := _solve_corner_raw(front_right, 0.005, 0.0, Vector3.ZERO)
	if float(p_fr0.get("arb_angle_rad", 0.0)) < 0.0:
		front_right.arb_jounce_sign = -1.0
	var p_rl0 := _solve_corner_raw(rear_left, 0.005, 0.0, Vector3.ZERO)
	if float(p_rl0.get("arb_angle_rad", 0.0)) < 0.0:
		rear_left.arb_jounce_sign = -1.0
	var p_rr0 := _solve_corner_raw(rear_right, 0.005, 0.0, Vector3.ZERO)
	if float(p_rr0.get("arb_angle_rad", 0.0)) < 0.0:
		rear_right.arb_jounce_sign = -1.0

	_cache_key = _build_cache_key(fl_d, rl_d, wheel_radius_m, wheelbase_m, h_cog_m)
	_configured = true
	return true


func build_luts(force_rebuild: bool = false) -> bool:
	if not _configured:
		return false

	if not force_rebuild and _shared_lut_cache.has(_cache_key):
		var cached: Dictionary = _shared_lut_cache[_cache_key]
		front_lut = cached["front_lut"]
		rear_lut = cached["rear_lut"]
		rack_disp_table = cached["rack_disp_table"]
		mr_0_front = float(cached["mr_0_front"])
		mr_0_rear = float(cached["mr_0_rear"])
		arb_mr_0_front = float(cached["arb_mr_0_front"])
		arb_mr_0_rear = float(cached["arb_mr_0_rear"])
		static_metrics_front = cached["static_metrics_front"]
		static_metrics_rear = cached["static_metrics_rear"]
		_ready = true
		return true

	static_metrics_front = compute_static_metrics(true)
	static_metrics_rear = compute_static_metrics(false)

	# 1. Calibrate steering rack displacement vs axle-mean steer angle at delta_z = 0
	rack_disp_table = PackedFloat64Array()
	rack_disp_table.resize(LUT_NS)
	var dz_step: float = (LUT_DZ_MAX - LUT_DZ_MIN) / float(LUT_NZ - 1)
	var steer_step: float = (LUT_STEER_MAX - LUT_STEER_MIN) / float(LUT_NS - 1)

	var rack_guess: float = 0.0
	var mid_is: int = (LUT_NS - 1) / 2
	# Sweep upward and downward from center steer = 0 for tight warm starts
	rack_disp_table[mid_is] = 0.0
	for k in range(mid_is + 1, LUT_NS):
		var st_tgt: float = LUT_STEER_MIN + float(k) * steer_step
		rack_guess = _calibrate_rack_for_mean_steer(st_tgt, rack_guess)
		rack_disp_table[k] = rack_guess
	rack_guess = 0.0
	for k in range(mid_is - 1, -1, -1):
		var st_tgt_n: float = LUT_STEER_MIN + float(k) * steer_step
		rack_guess = _calibrate_rack_for_mean_steer(st_tgt_n, rack_guess)
		rack_disp_table[k] = rack_guess

	# 2. Build Front 2D LUT (LUT_NZ x LUT_NS x STRIDE)
	front_lut = PackedFloat64Array()
	front_lut.resize(LUT_NZ * LUT_NS * STRIDE)
	var mid_iz: int = (LUT_NZ - 1) / 2

	for is_idx in range(LUT_NS):
		var st_cmd: float = LUT_STEER_MIN + float(is_idx) * steer_step
		var rack_d: float = rack_disp_table[is_idx]
		# Warm-start from center delta_z outward in both directions
		var guess_up := Vector3(0.0, 0.0, st_cmd)
		for iz in range(mid_iz, LUT_NZ):
			var dz: float = LUT_DZ_MIN + float(iz) * dz_step
			var res_f := solve_pose_with_derivatives(dz, rack_d, true, false, guess_up)
			guess_up = res_f["x_state"]
			_write_lut_entry(front_lut, (iz * LUT_NS + is_idx) * STRIDE, res_f, st_cmd)
		var guess_dn := _solve_corner_raw(front_left, 0.0, rack_d, Vector3(0.0, 0.0, st_cmd))["x_state"] as Vector3
		for iz in range(mid_iz - 1, -1, -1):
			var dz_n: float = LUT_DZ_MIN + float(iz) * dz_step
			var res_fn := solve_pose_with_derivatives(dz_n, rack_d, true, false, guess_dn)
			guess_dn = res_fn["x_state"]
			_write_lut_entry(front_lut, (iz * LUT_NS + is_idx) * STRIDE, res_fn, st_cmd)

	# 3. Build Rear 1D LUT (LUT_NZ x STRIDE)
	rear_lut = PackedFloat64Array()
	rear_lut.resize(LUT_NZ * STRIDE)
	var r_guess_up := Vector3.ZERO
	for iz in range(mid_iz, LUT_NZ):
		var dz: float = LUT_DZ_MIN + float(iz) * dz_step
		var res_r := solve_pose_with_derivatives(dz, 0.0, false, false, r_guess_up)
		r_guess_up = res_r["x_state"]
		_write_lut_entry(rear_lut, iz * STRIDE, res_r, 0.0)
	var r_guess_dn := Vector3.ZERO
	for iz in range(mid_iz - 1, -1, -1):
		var dz_n: float = LUT_DZ_MIN + float(iz) * dz_step
		var res_rn := solve_pose_with_derivatives(dz_n, 0.0, false, false, r_guess_dn)
		r_guess_dn = res_rn["x_state"]
		_write_lut_entry(rear_lut, iz * STRIDE, res_rn, 0.0)

	mr_0_front = front_lut[(mid_iz * LUT_NS + mid_is) * STRIDE + CH_MR]
	arb_mr_0_front = front_lut[(mid_iz * LUT_NS + mid_is) * STRIDE + CH_ARB_MR]
	mr_0_rear = rear_lut[mid_iz * STRIDE + CH_MR]
	arb_mr_0_rear = rear_lut[mid_iz * STRIDE + CH_ARB_MR]

	_shared_lut_cache[_cache_key] = {
		"front_lut": front_lut,
		"rear_lut": rear_lut,
		"rack_disp_table": rack_disp_table,
		"mr_0_front": mr_0_front,
		"mr_0_rear": mr_0_rear,
		"arb_mr_0_front": arb_mr_0_front,
		"arb_mr_0_rear": arb_mr_0_rear,
		"static_metrics_front": static_metrics_front,
		"static_metrics_rear": static_metrics_rear,
	}
	_ready = true
	return true


static func _write_lut_entry(buf: PackedFloat64Array, base: int, res: Dictionary, steer_cmd: float) -> void:
	var w_steer: float = float(res["wheel_steer_rad"])
	var cp_off: Vector2 = res["cp_offset_ros"]
	buf[base + CH_CAMBER] = float(res["camber_rad"])
	buf[base + CH_WHEEL_STEER] = w_steer
	buf[base + CH_TOE] = w_steer - steer_cmd
	buf[base + CH_CP_DX] = cp_off.x
	buf[base + CH_CP_DY] = cp_off.y
	buf[base + CH_DAMPER_DISP] = float(res["damper_disp_m"])
	buf[base + CH_MR] = float(res["motion_ratio"])
	buf[base + CH_DMR_DZ] = float(res["d_mr_dz"])
	buf[base + CH_ARB_ANGLE] = float(res["arb_angle_rad"])
	buf[base + CH_ARB_MR] = float(res["arb_motion_ratio"])
	buf[base + CH_FVSA_IC_Y] = float(res["fvsa_ic_y"])
	buf[base + CH_FVSA_IC_Z] = float(res["fvsa_ic_z"])
	buf[base + CH_FVSA_SLOPE] = float(res["fvsa_slope"])
	buf[base + CH_RC_Z] = float(res["roll_center_z"])
	buf[base + CH_SVSA_SLOPE] = float(res["svsa_slope"])
	buf[base + CH_MECH_TRAIL] = float(res["mechanical_trail_m"])
	buf[base + CH_SCRUB_RADIUS] = float(res["scrub_radius_m"])


## Returns exact static metrics at delta_z = 0, steer = 0 (Excel verification reference)
func compute_static_metrics(is_front: bool) -> Dictionary:
	var g: CornerGeometry = front_left if is_front else rear_left
	var kp: Vector3 = g.ubj - g.lbj
	var kpi_rad: float = atan2(-kp.y, kp.z)
	var caster_rad: float = atan2(-kp.x, kp.z)
	var t_g: float = (g.contact_patch.z - g.lbj.z) / kp.z
	var kp_ground: Vector3 = g.lbj + t_g * kp
	var scrub_m: float = g.contact_patch.y - kp_ground.y
	var trail_m: float = kp_ground.x - g.contact_patch.x

	var fvsa: Vector3 = _compute_fvsa_and_rc(g, g.lbj, g.ubj, g.contact_patch)
	var svsa: Vector3 = _compute_svsa(g, g.lbj, g.ubj, g.contact_patch)

	var f_pr_dir: Vector3 = (g.pushrod_wishbone - g.pushrod_rocker).normalized()
	var arm_pr_m: float = absf((g.pushrod_rocker - g.rocker_pivot).cross(f_pr_dir).dot(g.u_rck))
	var f_dm_dir: Vector3 = (g.damper_chassis - g.damper_rocker).normalized()
	var arm_dm_m: float = absf((g.damper_rocker - g.rocker_pivot).cross(f_dm_dir).dot(g.u_rck))
	var ratio_sp_pr: float = arm_pr_m / maxf(arm_dm_m, 1e-6)

	var svsa_slope: float = svsa.z
	var anti_pitch_100pct: float = svsa_slope * (wheelbase_m / maxf(h_cog_m, 1e-4))

	return {
		"kpi_deg": rad_to_deg(kpi_rad),
		"caster_deg": rad_to_deg(caster_rad),
		"scrub_radius_m": scrub_m,
		"mechanical_trail_m": trail_m,
		"fvsa_ic_y": fvsa.x,
		"fvsa_ic_z": fvsa.y,
		"roll_center_z": fvsa.z,
		"svsa_ic_x": svsa.x,
		"svsa_ic_z": svsa.y,
		"svsa_slope": svsa_slope,
		"anti_pitch_100pct": anti_pitch_100pct,
		"rocker_arm_pushrod_m": arm_pr_m,
		"rocker_arm_damper_m": arm_dm_m,
		"spring_to_pushrod_force_ratio": ratio_sp_pr,
	}


## Public offline/initialization exact 3D solver (delta_z in m, rack_disp in m)
## When is_right == true, solves directly on mirrored right-side 3D hardpoints (Y_R = -Y_L).
func solve_pose_3d(
	delta_z: float,
	rack_disp: float,
	is_front: bool,
	is_right: bool = false,
	init_guess: Vector3 = Vector3.ZERO
) -> Dictionary:
	var g: CornerGeometry
	if is_front:
		g = front_right if is_right else front_left
	else:
		g = rear_right if is_right else rear_left
	return _solve_corner_raw(g, delta_z, rack_disp, init_guess)


## Exact 3D solver + 2nd-order central-difference derivatives for MR, dMR/dz, and ARB MR
func solve_pose_with_derivatives(
	delta_z: float,
	rack_disp: float,
	is_front: bool,
	is_right: bool = false,
	init_guess: Vector3 = Vector3.ZERO
) -> Dictionary:
	var g: CornerGeometry
	if is_front:
		g = front_right if is_right else front_left
	else:
		g = rear_right if is_right else rear_left

	var h: float = 2.0e-3
	var c0: Dictionary = _solve_corner_raw(g, delta_z, rack_disp, init_guess)
	var x0: Vector3 = c0["x_state"]
	var cp: Dictionary = _solve_corner_raw(g, delta_z + h, rack_disp, x0)
	var cm: Dictionary = _solve_corner_raw(g, delta_z - h, rack_disp, x0)

	var s0: float = float(c0["damper_disp_m"])
	var sp: float = float(cp["damper_disp_m"])
	var sm: float = float(cm["damper_disp_m"])
	var arb_p: float = float(cp["arb_angle_rad"])
	var arb_m: float = float(cm["arb_angle_rad"])

	c0["motion_ratio"] = (sp - sm) / (2.0 * h)
	c0["d_mr_dz"] = (sp - 2.0 * s0 + sm) / (h * h)
	c0["arb_motion_ratio"] = (arb_p - arb_m) / (2.0 * h)
	return c0


## Read-only 3D visual pose solver for all corner hardpoints in ISO/ROS coordinates
## relative to chassis CG (+X forward, +Y left, +Z up). Returns empty Dictionary if not ready.
func compute_visual_pose_3d(
	axle_is_front: bool,
	is_left: bool,
	delta_z: float,
	steer_angle: float,
	a_cog_m: float = -1.0
) -> Dictionary:
	if not _ready:
		return {}
	var g: CornerGeometry
	if axle_is_front:
		g = front_left if is_left else front_right
	else:
		g = rear_left if is_left else rear_right
	var dz_c: float = clampf(delta_z, LUT_DZ_MIN, LUT_DZ_MAX)
	var st_c: float = clampf(steer_angle, LUT_STEER_MIN, LUT_STEER_MAX) if axle_is_front else 0.0
	var rack_d: float = get_rack_disp_for_steer(st_c) if axle_is_front else 0.0
	var sol: Dictionary = _solve_corner_raw(g, dz_c, rack_d, Vector3(0.0, 0.0, st_c))
	var a_off: float = a_cog_m if a_cog_m > 0.0 else (0.55 * wheelbase_m)
	var cg_off := Vector3(a_off, 0.0, -h_cog_m)
	return {
		"lca_front": g.lca_front + cg_off,
		"lca_rear": g.lca_rear + cg_off,
		"lbj": (sol["lbj"] as Vector3) + cg_off,
		"uca_front": g.uca_front + cg_off,
		"uca_rear": g.uca_rear + cg_off,
		"ubj": (sol["ubj"] as Vector3) + cg_off,
		"tie_inner": (sol["tie_inner"] as Vector3) + cg_off,
		"tie_outer": (sol["tie_outer"] as Vector3) + cg_off,
		"pushrod_wishbone": (sol["pushrod_wishbone"] as Vector3) + cg_off,
		"pushrod_rocker": (sol["pushrod_rocker"] as Vector3) + cg_off,
		"rocker_pivot": g.rocker_pivot + cg_off,
		"damper_rocker": (sol["damper_rocker"] as Vector3) + cg_off,
		"damper_chassis": g.damper_chassis + cg_off,
		"arb_link_rocker": (sol["arb_link_rocker"] as Vector3) + cg_off,
		"arb_arm_tip": (sol["arb_arm_tip"] as Vector3) + cg_off,
		"arb_chassis_piv": g.arb_chassis_piv + cg_off,
		"wheel_center": (sol["wheel_center"] as Vector3) + cg_off,
		"contact_patch": (sol["contact_patch"] as Vector3) + cg_off,
		"camber_rad": float(sol["camber_rad"]),
		"wheel_steer_rad": float(sol["wheel_steer_rad"]),
	}


## Evaluates calibrated rack displacement (m) from target front-axle mean steer angle (rad)
func get_rack_disp_for_steer(steer_cmd_rad: float) -> float:
	if not _ready or rack_disp_table.is_empty():
		return 0.0
	var f_idx: float = (steer_cmd_rad - LUT_STEER_MIN) * _inv_steer_step
	var is0: int = clampi(int(floor(f_idx)), 0, LUT_NS - 2)
	var u: float = clampf(f_idx - float(is0), 0.0, 1.0)
	var w: Vector4 = _catmull_weights(u)
	var im1: int = maxi(0, is0 - 1)
	var ip1: int = is0 + 1
	var ip2: int = mini(LUT_NS - 1, is0 + 2)
	var p0: float = (2.0 * rack_disp_table[0] - rack_disp_table[1]) if is0 == 0 else rack_disp_table[im1]
	var p1: float = rack_disp_table[is0]
	var p2: float = rack_disp_table[ip1]
	var p3: float = (2.0 * rack_disp_table[LUT_NS - 1] - rack_disp_table[LUT_NS - 2]) if (is0 + 1 == LUT_NS - 1) else rack_disp_table[ip2]
	return w.x * p0 + w.y * p1 + w.z * p2 + w.w * p3


func _calibrate_rack_for_mean_steer(target_steer: float, init_rack: float) -> float:
	if absf(target_steer) < 1e-10:
		return 0.0
	var r: float = init_rack
	var x_l := Vector3(0.0, 0.0, target_steer)
	var x_r := Vector3(0.0, 0.0, -target_steer)
	for _it in range(20):
		var res_l := _solve_corner_raw(front_left, 0.0, r, x_l)
		var res_r := _solve_corner_raw(front_left, 0.0, -r, x_r)
		x_l = res_l["x_state"]
		x_r = res_r["x_state"]
		var mean_st: float = 0.5 * (float(res_l["wheel_steer_rad"]) - float(res_r["wheel_steer_rad"]))
		var err: float = mean_st - target_steer
		if absf(err) < 1e-7:
			break
		var dr: float = 1.0e-4
		var res_lp := _solve_corner_raw(front_left, 0.0, r + dr, x_l)
		var res_rp := _solve_corner_raw(front_left, 0.0, -(r + dr), x_r)
		var mean_stp: float = 0.5 * (float(res_lp["wheel_steer_rad"]) - float(res_rp["wheel_steer_rad"]))
		var deriv: float = (mean_stp - mean_st) / dr
		if absf(deriv) < 1e-6:
			break
		r -= clampf(err / deriv, -0.01, 0.01)
	return r


func _solve_corner_raw(g: CornerGeometry, delta_z: float, rack_disp: float, init_guess: Vector3) -> Dictionary:
	var tie_in_tgt: Vector3 = g.tie_inner + Vector3(0.0, rack_disp, 0.0)
	var x: Vector3 = init_guess
	if x.is_zero_approx() and g.is_front and absf(rack_disp) > 1.0e-4:
		var steer_arm: float = maxf(absf(g.tie_outer.x - g.lbj.x), 0.04)
		var yaw_sgn: float = 1.0 if (g.tie_outer.x >= g.lbj.x) else -1.0
		x.z = clampf(yaw_sgn * rack_disp / steer_arm, -0.85, 0.85)

	var lbj_p: Vector3 = g.lbj
	var ubj_p: Vector3 = g.ubj
	var to_p: Vector3 = g.tie_outer
	var wc_p: Vector3 = g.wheel_center
	var cp_p: Vector3 = g.contact_patch
	var s_ax: Vector3 = Vector3(0.0, 1.0, 0.0)

	# 3D bounded Newton-Raphson with backtracking line search on (theta_lca, theta_uca, psi_kp)
	var r_vec: Vector3 = Vector3.ZERO
	var eval_res := func(xv: Vector3) -> Array:
		var lbj_i: Vector3 = _rot_point(g.lbj, g.lca_front, g.u_lca, xv.x)
		var ubj_i: Vector3 = _rot_point(g.ubj, g.uca_front, g.u_uca, xv.y)
		var kp_vec: Vector3 = ubj_i - lbj_i
		var dist_kp: float = kp_vec.length()
		var ez_u: Vector3 = kp_vec / maxf(dist_kp, 1e-9)
		var r_align: Basis = _basis_align_vec(g.ez0, ez_u)
		var col_x: Vector3 = _rot_vec_axis(r_align * g.r_u0.x, ez_u, xv.z)
		var col_y: Vector3 = _rot_vec_axis(r_align * g.r_u0.y, ez_u, xv.z)
		var r_u: Basis = Basis(col_x, col_y, ez_u)

		var to_i: Vector3 = lbj_i + r_u * g.to_loc
		var wc_i: Vector3 = lbj_i + r_u * g.wc_loc
		var sax_i: Vector3 = (r_u * g.spin_loc).normalized()
		var down := Vector3(0.0, 0.0, -1.0)
		var rad_dn: Vector3 = (down - down.dot(sax_i) * sax_i).normalized()
		var cp_i: Vector3 = wc_i + g.wheel_radius * rad_dn

		var f1: float = dist_kp - g.len_kp
		var f2: float = (to_i - tie_in_tgt).length() - g.len_tie
		var f3: float = (cp_i.z - g.contact_patch.z) - delta_z
		return [Vector3(f1, f2, f3), lbj_i, ubj_i, to_i, wc_i, cp_i, sax_i]

	var state_arr: Array = eval_res.call(x)
	r_vec = state_arr[0]
	var r_norm: float = r_vec.length()
	var eps_j: float = 1.0e-4
	var inv_2eps: float = 0.5 / eps_j

	for _it in range(45):
		if r_norm < 2.0e-7:
			break
		var rp_x: Vector3 = eval_res.call(Vector3(x.x + eps_j, x.y, x.z))[0]
		var rm_x: Vector3 = eval_res.call(Vector3(x.x - eps_j, x.y, x.z))[0]
		var rp_y: Vector3 = eval_res.call(Vector3(x.x, x.y + eps_j, x.z))[0]
		var rm_y: Vector3 = eval_res.call(Vector3(x.x, x.y - eps_j, x.z))[0]
		var rp_z: Vector3 = eval_res.call(Vector3(x.x, x.y, x.z + eps_j))[0]
		var rm_z: Vector3 = eval_res.call(Vector3(x.x, x.y, x.z - eps_j))[0]
		var j_c0: Vector3 = (rp_x - rm_x) * inv_2eps
		var j_c1: Vector3 = (rp_y - rm_y) * inv_2eps
		var j_c2: Vector3 = (rp_z - rm_z) * inv_2eps
		var dx: Vector3 = _solve_linear_3x3(j_c0, j_c1, j_c2, -r_vec)
		var dx_len: float = dx.length()
		if dx_len > 0.30:
			dx *= (0.30 / dx_len)

		var alpha: float = 1.0
		var accepted: bool = false
		for _ls in range(10):
			var x_try: Vector3 = x + dx * alpha
			var try_arr: Array = eval_res.call(x_try)
			var r_try: Vector3 = try_arr[0]
			var try_norm: float = r_try.length()
			if try_norm < r_norm or try_norm < 2.0e-7:
				x = x_try
				state_arr = try_arr
				r_vec = r_try
				r_norm = try_norm
				accepted = true
				break
			alpha *= 0.5
		if not accepted:
			x += dx * alpha
			state_arr = eval_res.call(x)
			r_vec = state_arr[0]
			r_norm = r_vec.length()

	lbj_p = state_arr[1]
	ubj_p = state_arr[2]
	to_p = state_arr[3]
	wc_p = state_arr[4]
	cp_p = state_arr[5]
	s_ax = state_arr[6]

	# Pushrod point belongs rigidly to LowerAArm (pushrod_on_lower=true) or UpperAArm (pushrod_on_lower=false)
	var pr_wb_p: Vector3 = (
		_rot_point(g.pushrod_wishbone, g.lca_front, g.u_lca, x.x)
		if g.pushrod_on_lower
		else _rot_point(g.pushrod_wishbone, g.uca_front, g.u_uca, x.y)
	)

	# Close pushrod onto rocker rotating around rocker_pivot -> rocker_axis_p1
	var th_rck: float = 0.0
	var pr_rck_p: Vector3 = g.pushrod_rocker
	var err_pr: float = 0.0
	for _it_r in range(30):
		pr_rck_p = _rot_point(g.pushrod_rocker, g.rocker_pivot, g.u_rck, th_rck)
		var diff_pr: Vector3 = pr_rck_p - pr_wb_p
		var d_pr: float = diff_pr.length()
		err_pr = d_pr - g.len_pr
		if absf(err_pr) < 1.0e-9:
			break
		var vel_pr: Vector3 = g.u_rck.cross(pr_rck_p - g.rocker_pivot)
		var derr_pr: float = diff_pr.dot(vel_pr) / maxf(d_pr, 1e-9)
		if absf(derr_pr) < 1e-9:
			break
		th_rck -= clampf(err_pr / derr_pr, -0.25, 0.25)

	var dm_rck_p: Vector3 = _rot_point(g.damper_rocker, g.rocker_pivot, g.u_rck, th_rck)
	var damper_disp_m: float = g.len_dm0 - (g.damper_chassis - dm_rck_p).length()
	var arb_rck_p: Vector3 = _rot_point(g.arb_link_rocker, g.rocker_pivot, g.u_rck, th_rck)

	# Close ARB drop-link onto ARB blade rotating around +Y at arb_chassis_piv
	var th_arb: float = 0.0
	var arb_tip_p: Vector3 = g.arb_arm_tip
	var err_arb: float = 0.0
	for _it_a in range(30):
		arb_tip_p = _rot_point(g.arb_arm_tip, g.arb_chassis_piv, g.u_arb, th_arb)
		var diff_a: Vector3 = arb_tip_p - arb_rck_p
		var d_a: float = diff_a.length()
		err_arb = d_a - g.len_al
		if absf(err_arb) < 1.0e-9:
			break
		var vel_a: Vector3 = g.u_arb.cross(arb_tip_p - g.arb_chassis_piv)
		var derr_a: float = diff_a.dot(vel_a) / maxf(d_a, 1e-9)
		if absf(derr_a) < 1e-9:
			break
		th_arb -= clampf(err_arb / derr_a, -0.25, 0.25)

	# ISO 8855 wheel orientation angles:
	# s_ax points car-left (+Y). Top leaning right (-Y) => s_ax.z > 0 => gamma < 0.
	var camber_iso: float = atan2(-s_ax.z, sqrt(s_ax.x * s_ax.x + s_ax.y * s_ax.y))
	var fwd_w: Vector3 = s_ax.cross(Vector3(0.0, 0.0, 1.0)).normalized()
	var steer_iso: float = atan2(fwd_w.y, fwd_w.x)
	var lat_w: Vector3 = Vector3(0.0, 0.0, 1.0).cross(fwd_w).normalized()

	# Dynamic kingpin ground intersection -> mechanical trail & scrub radius in wheel frame
	var kp_now: Vector3 = ubj_p - lbj_p
	var t_g: float = (cp_p.z - lbj_p.z) / kp_now.z
	var kp_g: Vector3 = lbj_p + t_g * kp_now
	var kp_minus_cp: Vector3 = kp_g - cp_p
	var trail_m: float = kp_minus_cp.dot(fwd_w)
	# Scrub radius positive when contact patch is outboard of kingpin ground intersection
	var lat_outboard: Vector3 = -lat_w if g.is_right else lat_w
	var scrub_m: float = (-kp_minus_cp).dot(lat_outboard)

	var fvsa: Vector3 = _compute_fvsa_and_rc(g, lbj_p, ubj_p, cp_p)
	var svsa: Vector3 = _compute_svsa(g, lbj_p, ubj_p, cp_p)

	return {
		"x_state": x,
		"residual_norm": r_norm,
		"err_kp": absf((ubj_p - lbj_p).length() - g.len_kp),
		"err_tie": absf((to_p - tie_in_tgt).length() - g.len_tie),
		"err_pr": absf(err_pr),
		"err_arb": absf(err_arb),
		"err_lca": absf((lbj_p - g.lca_front).length() - (g.lbj - g.lca_front).length()),
		"err_uca": absf((ubj_p - g.uca_front).length() - (g.ubj - g.uca_front).length()),
		"camber_rad": camber_iso,
		"wheel_steer_rad": steer_iso,
		"toe_rad": steer_iso,
		"cp_offset_ros": Vector2(cp_p.x - g.contact_patch.x, cp_p.y - g.contact_patch.y),
		"damper_disp_m": damper_disp_m,
		"arb_angle_rad": g.arb_jounce_sign * th_arb,
		"fvsa_ic_y": fvsa.x,
		"fvsa_ic_z": fvsa.y,
		"fvsa_slope": (fvsa.y - cp_p.z) / maxf(absf(cp_p.y - fvsa.x), 1e-4) if absf(cp_p.y - fvsa.x) > 1e-4 else 0.0,
		"roll_center_z": fvsa.z,
		"svsa_ic_x": svsa.x,
		"svsa_ic_z": svsa.y,
		"svsa_slope": svsa.z,
		"mechanical_trail_m": trail_m,
		"scrub_radius_m": scrub_m,
		"lbj": lbj_p,
		"ubj": ubj_p,
		"tie_outer": to_p,
		"tie_inner": tie_in_tgt,
		"pushrod_wishbone": pr_wb_p,
		"pushrod_rocker": pr_rck_p,
		"damper_rocker": dm_rck_p,
		"arb_link_rocker": arb_rck_p,
		"arb_arm_tip": arb_tip_p,
		"wheel_center": wc_p,
		"contact_patch": cp_p,
	}


static func _compute_fvsa_and_rc(g: CornerGeometry, lbj_p: Vector3, ubj_p: Vector3, cp_p: Vector3) -> Vector3:
	# 2D Front-View (Y-Z) intersection of (lca_mid -> lbj) and (uca_mid -> ubj)
	var x1: float = g.lca_mid.y
	var y1: float = g.lca_mid.z
	var x2: float = lbj_p.y
	var y2: float = lbj_p.z
	var x3: float = g.uca_mid.y
	var y3: float = g.uca_mid.z
	var x4: float = ubj_p.y
	var y4: float = ubj_p.z

	var dx1: float = x1 - x2
	var dy1: float = y1 - y2
	var dx2: float = x3 - x4
	var dy2: float = y3 - y4
	var denom: float = dx1 * dy2 - dy1 * dx2

	var ic_y: float = 0.0
	var ic_z: float = 0.0
	if absf(denom) < 1.0e-7:
		# Parallel wishbones: instant center at large virtual swing radius along wishbone angle
		var slope_par: float = dy1 / (-dx1) if absf(dx1) > 1e-6 else 0.0
		var y_dir: float = 1.0 if g.is_right else -1.0
		ic_y = cp_p.y + y_dir * 1000.0
		ic_z = cp_p.z + slope_par * 1000.0
	else:
		var c1: float = x1 * y2 - y1 * x2
		var c2: float = x3 * y4 - y3 * x4
		ic_y = (c1 * dx2 - dx1 * c2) / denom
		ic_z = (c1 * dy2 - dy1 * c2) / denom

	var dy_arm: float = ic_y - cp_p.y
	var rc_above_road: float = 0.0
	if absf(dy_arm) > 1e-5:
		var t_rc: float = -cp_p.y / dy_arm
		rc_above_road = t_rc * (ic_z - cp_p.z)
	return Vector3(ic_y, ic_z, rc_above_road)


static func _compute_svsa(g: CornerGeometry, lbj_p: Vector3, ubj_p: Vector3, cp_p: Vector3) -> Vector3:
	# 3D intersection of LCA plane and UCA plane with wheel transverse plane Y = cp_p.y
	var n_lca: Vector3 = (g.lca_rear - g.lca_front).cross(lbj_p - g.lca_front)
	var d_lca: float = n_lca.dot(lbj_p)
	var n_uca: Vector3 = (g.uca_rear - g.uca_front).cross(ubj_p - g.uca_front)
	var d_uca: float = n_uca.dot(ubj_p)
	var n_y := Vector3(0.0, 1.0, 0.0)
	var d_y: float = cp_p.y

	# Solve [n_lca^T; n_uca^T; n_y^T] * p = [d_lca; d_uca; d_y] via row cross products
	var c0: Vector3 = n_uca.cross(n_y)
	var c1: Vector3 = n_y.cross(n_lca)
	var c2: Vector3 = n_lca.cross(n_uca)
	var det: float = n_lca.dot(c0)

	var ic_x: float = cp_p.x - (1.0 if g.is_front else -1.0)
	var ic_z: float = cp_p.z
	var slope: float = 0.0
	if absf(det) > 1.0e-9:
		var ic_pt: Vector3 = (c0 * d_lca + c1 * d_uca + c2 * d_y) / det
		ic_x = ic_pt.x
		ic_z = ic_pt.z
		var dx_arm: float = (cp_p.x - ic_x) if g.is_front else (ic_x - cp_p.x)
		if absf(dx_arm) > 1.0e-4:
			slope = (ic_z - cp_p.z) / dx_arm
	return Vector3(ic_x, ic_z, slope)


## Runtime C1 Catmull-Rom 2D lookup for front corner into preallocated out_buf (ZERO heap allocs)
func sample_front_into(
	delta_z: float,
	steer_cmd: float,
	is_right: bool,
	out_buf: PackedFloat64Array,
	out_offset: int
) -> void:
	if not _ready:
		return
	var q_steer: float = -steer_cmd if is_right else steer_cmd
	var fz: float = (clampf(delta_z, LUT_DZ_MIN, LUT_DZ_MAX) - LUT_DZ_MIN) * _inv_dz_step
	var fs: float = (clampf(q_steer, LUT_STEER_MIN, LUT_STEER_MAX) - LUT_STEER_MIN) * _inv_steer_step

	var iz: int = clampi(int(floor(fz)), 0, LUT_NZ - 2)
	var is0: int = clampi(int(floor(fs)), 0, LUT_NS - 2)
	var uz: float = clampf(fz - float(iz), 0.0, 1.0)
	var us: float = clampf(fs - float(is0), 0.0, 1.0)

	var wz: Vector4 = _catmull_weights(uz)
	var ws: Vector4 = _catmull_weights(us)

	var zm1: int = maxi(0, iz - 1)
	var zp1: int = iz + 1
	var zp2: int = mini(LUT_NZ - 1, iz + 2)
	var sm1: int = maxi(0, is0 - 1)
	var sp1: int = is0 + 1
	var sp2: int = mini(LUT_NS - 1, is0 + 2)

	var z_is_lo: bool = (iz == 0)
	var z_is_hi: bool = (iz + 1 == LUT_NZ - 1)
	var s_is_lo: bool = (is0 == 0)
	var s_is_hi: bool = (is0 + 1 == LUT_NS - 1)

	var r_z0: int = (0 if z_is_lo else zm1) * LUT_NS
	var r_z1: int = iz * LUT_NS
	var r_z2: int = zp1 * LUT_NS
	var r_z3: int = ((LUT_NZ - 1) if z_is_hi else zp2) * LUT_NS
	var r_z_lo_n: int = 1 * LUT_NS
	var r_z_hi_p: int = (LUT_NZ - 2) * LUT_NS

	var c_s0: int = 0 if s_is_lo else sm1
	var c_s1: int = is0
	var c_s2: int = sp1
	var c_s3: int = (LUT_NS - 1) if s_is_hi else sp2

	for ch in range(STRIDE):
		# Interpolate along steer for each of the 4 z-rows, handling boundary linear extrapolation
		var v_row0: float = _interp_row_s(r_z0, c_s0, c_s1, c_s2, c_s3, s_is_lo, s_is_hi, ws, ch)
		var v_row1: float = _interp_row_s(r_z1, c_s0, c_s1, c_s2, c_s3, s_is_lo, s_is_hi, ws, ch)
		var v_row2: float = _interp_row_s(r_z2, c_s0, c_s1, c_s2, c_s3, s_is_lo, s_is_hi, ws, ch)
		var v_row3: float = _interp_row_s(r_z3, c_s0, c_s1, c_s2, c_s3, s_is_lo, s_is_hi, ws, ch)

		if z_is_lo:
			var v_next: float = _interp_row_s(r_z_lo_n, c_s0, c_s1, c_s2, c_s3, s_is_lo, s_is_hi, ws, ch)
			v_row0 = 2.0 * v_row1 - v_next
		if z_is_hi:
			var v_prev: float = _interp_row_s(r_z_hi_p, c_s0, c_s1, c_s2, c_s3, s_is_lo, s_is_hi, ws, ch)
			v_row3 = 2.0 * v_row2 - v_prev

		out_buf[out_offset + ch] = wz.x * v_row0 + wz.y * v_row1 + wz.z * v_row2 + wz.w * v_row3

	if is_right:
		out_buf[out_offset + CH_CAMBER] = -out_buf[out_offset + CH_CAMBER]
		out_buf[out_offset + CH_WHEEL_STEER] = -out_buf[out_offset + CH_WHEEL_STEER]
		out_buf[out_offset + CH_TOE] = -out_buf[out_offset + CH_TOE]
		out_buf[out_offset + CH_CP_DY] = -out_buf[out_offset + CH_CP_DY]
		out_buf[out_offset + CH_FVSA_IC_Y] = -out_buf[out_offset + CH_FVSA_IC_Y]


func _interp_row_s(
	row_base: int,
	c0: int,
	c1: int,
	c2: int,
	c3: int,
	s_is_lo: bool,
	s_is_hi: bool,
	ws: Vector4,
	ch: int
) -> float:
	var p1: float = front_lut[(row_base + c1) * STRIDE + ch]
	var p2: float = front_lut[(row_base + c2) * STRIDE + ch]
	var p0: float = (2.0 * p1 - front_lut[(row_base + 1) * STRIDE + ch]) if s_is_lo else front_lut[(row_base + c0) * STRIDE + ch]
	var p3: float = (2.0 * p2 - front_lut[(row_base + LUT_NS - 2) * STRIDE + ch]) if s_is_hi else front_lut[(row_base + c3) * STRIDE + ch]
	return ws.x * p0 + ws.y * p1 + ws.z * p2 + ws.w * p3


## Runtime C1 Catmull-Rom 1D lookup for rear corner into preallocated out_buf (ZERO heap allocs)
func sample_rear_into(
	delta_z: float,
	is_right: bool,
	out_buf: PackedFloat64Array,
	out_offset: int
) -> void:
	if not _ready:
		return
	var fz: float = (clampf(delta_z, LUT_DZ_MIN, LUT_DZ_MAX) - LUT_DZ_MIN) * _inv_dz_step
	var iz: int = clampi(int(floor(fz)), 0, LUT_NZ - 2)
	var uz: float = clampf(fz - float(iz), 0.0, 1.0)
	var wz: Vector4 = _catmull_weights(uz)

	var zm1: int = maxi(0, iz - 1)
	var zp1: int = iz + 1
	var zp2: int = mini(LUT_NZ - 1, iz + 2)
	var z_is_lo: bool = (iz == 0)
	var z_is_hi: bool = (iz + 1 == LUT_NZ - 1)

	var b0: int = zm1 * STRIDE
	var b1: int = iz * STRIDE
	var b2: int = zp1 * STRIDE
	var b3: int = zp2 * STRIDE

	for ch in range(STRIDE):
		var p1: float = rear_lut[b1 + ch]
		var p2: float = rear_lut[b2 + ch]
		var p0: float = (2.0 * p1 - rear_lut[STRIDE + ch]) if z_is_lo else rear_lut[b0 + ch]
		var p3: float = (2.0 * p2 - rear_lut[(LUT_NZ - 2) * STRIDE + ch]) if z_is_hi else rear_lut[b3 + ch]
		out_buf[out_offset + ch] = wz.x * p0 + wz.y * p1 + wz.z * p2 + wz.w * p3

	if is_right:
		out_buf[out_offset + CH_CAMBER] = -out_buf[out_offset + CH_CAMBER]
		out_buf[out_offset + CH_WHEEL_STEER] = -out_buf[out_offset + CH_WHEEL_STEER]
		out_buf[out_offset + CH_TOE] = -out_buf[out_offset + CH_TOE]
		out_buf[out_offset + CH_CP_DY] = -out_buf[out_offset + CH_CP_DY]
		out_buf[out_offset + CH_FVSA_IC_Y] = -out_buf[out_offset + CH_FVSA_IC_Y]


static func _catmull_weights(t: float) -> Vector4:
	var t2: float = t * t
	var t3: float = t2 * t
	return Vector4(
		0.5 * (-t + 2.0 * t2 - t3),
		0.5 * (2.0 - 5.0 * t2 + 3.0 * t3),
		0.5 * (t + 4.0 * t2 - 3.0 * t3),
		0.5 * (-t2 + t3)
	)


static func _rot_vec_axis(v: Vector3, u: Vector3, theta: float) -> Vector3:
	if absf(theta) < 1.0e-15:
		return v
	var c: float = cos(theta)
	var s: float = sin(theta)
	return v * c + u.cross(v) * s + u * (u.dot(v) * (1.0 - c))


static func _rot_point(p: Vector3, origin: Vector3, u: Vector3, theta: float) -> Vector3:
	return origin + _rot_vec_axis(p - origin, u, theta)


static func _basis_align_vec(a: Vector3, b: Vector3) -> Basis:
	var v: Vector3 = a.cross(b)
	var c: float = a.dot(b)
	var s2: float = v.length_squared()
	if s2 < 1.0e-24:
		return Basis.IDENTITY
	var inv_1pc: float = 1.0 / maxf(1.0 + c, 1.0e-12)
	var vx := Basis(
		Vector3(0.0, v.z, -v.y),
		Vector3(-v.z, 0.0, v.x),
		Vector3(v.y, -v.x, 0.0)
	)
	var vx2: Basis = vx * vx
	return Basis(
		Vector3(1.0, 0.0, 0.0) + vx.x + vx2.x * inv_1pc,
		Vector3(0.0, 1.0, 0.0) + vx.y + vx2.y * inv_1pc,
		Vector3(0.0, 0.0, 1.0) + vx.z + vx2.z * inv_1pc
	)


static func _solve_linear_3x3(c0: Vector3, c1: Vector3, c2: Vector3, rhs: Vector3) -> Vector3:
	var cr12: Vector3 = c1.cross(c2)
	var det: float = c0.dot(cr12)
	if absf(det) < 1.0e-18:
		return Vector3.ZERO
	var inv_det: float = 1.0 / det
	return Vector3(
		rhs.dot(cr12) * inv_det,
		c0.dot(rhs.cross(c2)) * inv_det,
		c0.dot(c1.cross(rhs)) * inv_det
	)


static func _build_cache_key(fl_d: Dictionary, rl_d: Dictionary, r_w: float, wb: float, h_cog: float) -> String:
	var parts: PackedStringArray = PackedStringArray()
	parts.append("%.6f|%.6f|%.6f" % [r_w, wb, h_cog])
	for side_d in [fl_d, rl_d]:
		parts.append("%d|%.4f|%.4f" % [
			1 if bool(side_d.get("pushrod_on_lower", false)) else 0,
			float(side_d.get("static_camber_deg", 0.0)),
			float(side_d.get("static_toe_deg", 0.0)),
		])
		for k in LINK_NAMES:
			var v: Variant = side_d[k]
			if v is Vector3:
				parts.append("%.6f,%.6f,%.6f" % [v.x, v.y, v.z])
			elif v is Array and v.size() >= 3:
				parts.append("%.6f,%.6f,%.6f" % [float(v[0]), float(v[1]), float(v[2])])
	return ";".join(parts)


## Returns live suspension & kinematics telemetry summary for UI / HUD / tests.
## Handles both 3D Double Wishbone and Parametric modes cleanly.
func get_telemetry_summary(corners: Array, steer_rad: float, config: Variant) -> Dictionary:
	return compute_telemetry_summary(corners, steer_rad, config, self)


static func compute_telemetry_summary(
	corners: Array,
	steer_rad: float,
	config: Variant,
	kin_3d: DoubleWishboneKinematics = null
) -> Dictionary:
	if config == null or corners.size() < 4:
		return {
			"camber_fl_deg": 0.0,
			"camber_fr_deg": 0.0,
			"toe_fl_deg": 0.0,
			"toe_fr_deg": 0.0,
			"ackermann_pct": 0.0,
			"mr_front": 0.0,
			"mr_rear": 0.0,
			"lltd_pct": 50.0,
		}

	var c_fl = corners[0]
	var c_fr = corners[1]
	var c_rl = corners[2]
	var c_rr = corners[3]

	# Vehicle convention: camber < 0 top inward; toe > 0 toe-in
	var camber_fl_deg: float = rad_to_deg(float(c_fl.camber_rad))
	var camber_fr_deg: float = rad_to_deg(-float(c_fr.camber_rad))
	var toe_fl_deg: float = rad_to_deg(-float(c_fl.toe_rad))
	var toe_fr_deg: float = rad_to_deg(float(c_fr.toe_rad))

	var is_3d: bool = bool(config.use_3d_kinematics) and kin_3d != null and kin_3d.is_ready()
	var mr_front: float = 0.5 * (float(c_fl.motion_ratio) + float(c_fr.motion_ratio)) if is_3d else float(config.motion_ratio_front)
	var mr_rear: float = 0.5 * (float(c_rl.motion_ratio) + float(c_rr.motion_ratio)) if is_3d else float(config.motion_ratio_rear)

	# Ackermann %
	var wb: float = maxf(float(config.wheelbase), 0.2)
	var tf: float = maxf(float(config.track_width_f), 0.2)
	var tr: float = maxf(float(config.track_width_r), 0.2)
	var ackermann_pct: float = float(config.ackermann_factor) * 100.0
	var probe_steer: float = steer_rad
	var d_fl: float = 0.0
	var d_fr: float = 0.0
	if is_3d:
		if absf(steer_rad) > 1.0e-3:
			d_fl = float(c_fl.wheel_steer_rad)
			d_fr = float(c_fr.wheel_steer_rad)
		else:
			probe_steer = deg_to_rad(10.0)
			var probe := PackedFloat64Array()
			probe.resize(STRIDE)
			kin_3d.sample_front_into(0.0, probe_steer, false, probe, 0)
			d_fl = probe[CH_WHEEL_STEER]
			kin_3d.sample_front_into(0.0, probe_steer, true, probe, 0)
			d_fr = probe[CH_WHEEL_STEER]
		var d_in: float = d_fl if probe_steer >= 0.0 else -d_fr
		var d_out: float = d_fr if probe_steer >= 0.0 else -d_fl
		if d_out > 1.0e-4:
			var tan_out: float = tan(d_out)
			var inv_r_in: float = (1.0 / maxf(tan_out, 1.0e-5)) - (tf / wb)
			var d_in_ideal: float = atan(1.0 / maxf(inv_r_in, 1.0e-4))
			var denom_ack: float = d_in_ideal - d_out
			if absf(denom_ack) > 1.0e-6:
				ackermann_pct = ((d_in - d_out) / denom_ack) * 100.0

	# Mechanical LLTD (%) combining elastic roll stiffness + geometric roll center heights
	var kw_f: float = float(config.spring_stiffness_front) * mr_front * mr_front
	var kw_r: float = float(config.spring_stiffness_rear) * mr_rear * mr_rear
	var k_phi_arb_f: float = float(config.arb_stiffness_front) * tf * tf
	var k_phi_arb_r: float = float(config.arb_stiffness_rear) * tr * tr
	if is_3d:
		var mr_arb_f: float = 0.5 * (float(c_fl.arb_motion_ratio) + float(c_fr.arb_motion_ratio))
		var mr_arb_r: float = 0.5 * (float(c_rl.arb_motion_ratio) + float(c_rr.arb_motion_ratio))
		var k_bar_f: float = float(config.arb_stiffness_front) / maxf(kin_3d.arb_mr_0_front * kin_3d.arb_mr_0_front, 1.0e-6)
		var k_bar_r: float = float(config.arb_stiffness_rear) / maxf(kin_3d.arb_mr_0_rear * kin_3d.arb_mr_0_rear, 1.0e-6)
		k_phi_arb_f = k_bar_f * (mr_arb_f * tf) * (mr_arb_f * tf)
		k_phi_arb_r = k_bar_r * (mr_arb_r * tr) * (mr_arb_r * tr)

	var k_phi_f: float = 0.5 * kw_f * tf * tf + k_phi_arb_f
	var k_phi_r: float = 0.5 * kw_r * tr * tr + k_phi_arb_r
	var el_f: float = k_phi_f / maxf(k_phi_f + k_phi_r, 1.0)
	var h_rc_f: float = 0.5 * (float(c_fl.roll_center_z) + float(c_fr.roll_center_z)) if is_3d else float(config.roll_center_height_front)
	var h_rc_r: float = 0.5 * (float(c_rl.roll_center_z) + float(c_rr.roll_center_z)) if is_3d else float(config.roll_center_height_rear)
	var w_f: float = float(config.front_weight_distribution)
	var w_r: float = 1.0 - w_f
	var h_roll: float = maxf(float(config.h_cog) - (w_f * h_rc_f + w_r * h_rc_r), 0.01)
	var dfz_f_u: float = (el_f * h_roll + w_f * h_rc_f) / tf
	var dfz_r_u: float = ((1.0 - el_f) * h_roll + w_r * h_rc_r) / tr
	var lltd_pct: float = (dfz_f_u / maxf(dfz_f_u + dfz_r_u, 1.0e-6)) * 100.0

	return {
		"camber_fl_deg": camber_fl_deg,
		"camber_fr_deg": camber_fr_deg,
		"toe_fl_deg": toe_fl_deg,
		"toe_fr_deg": toe_fr_deg,
		"ackermann_pct": ackermann_pct,
		"mr_front": mr_front,
		"mr_rear": mr_rear,
		"lltd_pct": lltd_pct,
	}
