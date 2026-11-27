class_name SuspensionSystem
extends RefCounted

## decoupled 4-corner susp interface (block b) for mirena dynamics (ter27)
## encapsulates per-corner vertical force generation, axle arb roll coupling,
## asymmetric bump/rebound damping, progressive bumpstops, static preload,
## and 1d kinematic camber/toe lookup tables behind an implementation-agnostic
## interface so future multibody or nonlinear models can swap in cleanly
## conventions:
## delta_z > 0: compression (jounce) relative to static design ride height (delta_z = 0)
## motion ratio mr = d(x_spring) / d(z_wheel) => k_wheel = k_spring * mr^2, c_wheel = c_damper * mr^2
## arb per axle: f_arb_l = +k_arb_wheel * (dz_l - dz_r), f_arb_r = -f_arb_l (f_l + f_r = 0 pure roll moment)
## kinematics (optimum kinematics 1d tables, empty -> 0):
## camber table stores vehicle convention (negative = top inward). mapped to iso 8855 gamma:
## left wheel: gamma_l = +table_val (top right/inward -> -gamma -> negative fy thrust)
## right wheel: gamma_r = -table_val (top left/inward -> +gamma -> positive fy thrust)
## toe table stores vehicle convention (+ = toe-in). mapped to iso 8855 steer offset tau:
## left wheel: tau_l = -table_val (front turned right/cw around +z)
## right wheel: tau_r = +table_val (front turned left/ccw around +z)

const VehicleConfig = preload("../../Config/VehicleConfig.gd")
const SuspensionCorner = preload("SuspensionCorner.gd")
const DoubleWishboneKinematics = preload("DoubleWishboneKinematics.gd")

var corners: Array[SuspensionCorner] = []
var kinematics_3d: DoubleWishboneKinematics = null
var use_3d_kinematics: bool = false
var _lut_buf: PackedFloat64Array = PackedFloat64Array()
var _steer_probe_buf: PackedFloat64Array = PackedFloat64Array()

func _init() -> void:
	_lut_buf.resize(4 * DoubleWishboneKinematics.STRIDE)
	_steer_probe_buf.resize(DoubleWishboneKinematics.STRIDE)
	_init_corners()

func _init_corners() -> void:
	corners.clear()
	corners.append(SuspensionCorner.new(VehicleConfig.WheelIndex.FL, true, false))
	corners.append(SuspensionCorner.new(VehicleConfig.WheelIndex.FR, true, true))
	corners.append(SuspensionCorner.new(VehicleConfig.WheelIndex.RL, false, false))
	corners.append(SuspensionCorner.new(VehicleConfig.WheelIndex.RR, false, true))

func reset() -> void:
	for c in corners:
		c.reset()

func is_3d_active(config: VehicleConfig) -> bool:
	return use_3d_kinematics and config != null and config.use_3d_kinematics and kinematics_3d != null and kinematics_3d.is_ready()

## synchronizes derived geometry, static preload, 3d kinematic luts, and corner alignment
## from vehicleconfig in real time without resetting chassis or corner travel states
func sync_from_config(config: VehicleConfig) -> void:
	if config == null:
		return
	config.update_derived_geometry()
	if corners.size() < 4:
		_init_corners()
	if _lut_buf.size() < 4 * DoubleWishboneKinematics.STRIDE:
		_lut_buf.resize(4 * DoubleWishboneKinematics.STRIDE)
	if _steer_probe_buf.size() < DoubleWishboneKinematics.STRIDE:
		_steer_probe_buf.resize(DoubleWishboneKinematics.STRIDE)

	use_3d_kinematics = false
	if config.use_3d_kinematics and not config.double_wishbone_kinematics.is_empty():
		if kinematics_3d == null:
			kinematics_3d = DoubleWishboneKinematics.new()
		if kinematics_3d.configure(config.double_wishbone_kinematics, config.wheel_radius, config.wheelbase, config.h_cog):
			use_3d_kinematics = kinematics_3d.build_luts()

	for i in range(4):
		var c: SuspensionCorner = corners[i]
		var is_front: bool = (i < 2)
		c.static_preload = compute_static_preload(i, config)
		if is_3d_active(config):
			_sample_corner_lut(i, c.delta_z, 0.0, config)
		else:
			c.camber_rad = compute_corner_camber_iso(c.delta_z, is_front, c.is_right, config)
			c.toe_rad = compute_corner_toe_iso(c.delta_z, is_front, c.is_right, config)
			c.wheel_steer_rad = c.toe_rad
			c.cp_offset_ros = Vector2.ZERO
			c.motion_ratio = config.motion_ratio_front if is_front else config.motion_ratio_rear
			c.d_mr_dz = 0.0
			c.wheel_spring_stiffness = get_wheel_spring_stiffness(is_front, config)

## evaluates all 4 corners given wheel compressions (m), velocities (m/s), and grounded flags
## returns the updated array[suspensioncorner] (fl, fr, rl, rr)
func evaluate_from_corners(
	compressions: Array,
	comp_velocities: Array,
	grounded: Array,
	config: VehicleConfig,
	steer_cmd: float = 0.0
) -> Array[SuspensionCorner]:
	if is_3d_active(config):
		return _evaluate_from_corners_3d(compressions, comp_velocities, grounded, config, steer_cmd)

	for i in range(4):
		var c: SuspensionCorner = corners[i]
		var is_front: bool = (i < 2)
		var dz: float = float(compressions[i]) if i < compressions.size() else 0.0
		var dz_dot: float = float(comp_velocities[i]) if i < comp_velocities.size() else 0.0
		var is_g: bool = bool(grounded[i]) if i < grounded.size() else true

		c.is_grounded = is_g
		c.delta_z = dz
		c.delta_z_dot = dz_dot
		c.static_preload = compute_static_preload(i, config)

		# kinematic alignment evaluated at current corner travel regardless of contact
		c.camber_rad = compute_corner_camber_iso(dz, is_front, c.is_right, config)
		c.toe_rad = compute_corner_toe_iso(dz, is_front, c.is_right, config)

		if not is_g:
			c.spring_force = 0.0
			c.damper_force = 0.0
			c.bumpstop_force = 0.0
			c.arb_force = 0.0
			c.fz = 0.0
			continue

		var k_wheel: float = get_wheel_spring_stiffness(is_front, config)

		c.spring_force = k_wheel * dz
		c.damper_force = compute_damper_force_wheel(is_front, dz_dot, config)
		c.bumpstop_force = compute_bumpstop_force(dz, is_front, config)

	# axle anti-roll bar coupling (pure roll moment: f_arb_l + f_arb_r = 0)
	_apply_axle_arb(VehicleConfig.WheelIndex.FL, VehicleConfig.WheelIndex.FR, true, config)
	_apply_axle_arb(VehicleConfig.WheelIndex.RL, VehicleConfig.WheelIndex.RR, false, config)

	for i in range(4):
		var c: SuspensionCorner = corners[i]
		if c.is_grounded:
			var f_total: float = c.static_preload + c.spring_force + c.damper_force + c.bumpstop_force + c.arb_force
			c.fz = maxf(0.0, f_total)
		else:
			c.fz = 0.0

	return corners


func _sample_corner_lut(corner_idx: int, dz: float, steer_cmd: float, config: VehicleConfig) -> void:
	var c: SuspensionCorner = corners[corner_idx]
	var is_front: bool = (corner_idx < 2)
	var base: int = corner_idx * DoubleWishboneKinematics.STRIDE
	if is_front:
		kinematics_3d.sample_front_into(dz, steer_cmd, c.is_right, _lut_buf, base)
	else:
		kinematics_3d.sample_rear_into(dz, c.is_right, _lut_buf, base)

	var g_corner = (kinematics_3d.front_right if c.is_right else kinematics_3d.front_left) if is_front else (kinematics_3d.rear_right if c.is_right else kinematics_3d.rear_left)
	var static_camber_deg: float = (config.static_camber_front_deg if is_front else config.static_camber_rear_deg) - g_corner.static_camber_deg
	var static_camber_iso: float = -deg_to_rad(static_camber_deg) if c.is_right else deg_to_rad(static_camber_deg)
	var static_toe_deg: float = (config.toe_front_deg if is_front else config.toe_rear_deg) - g_corner.static_toe_deg
	var static_toe_iso: float = deg_to_rad(static_toe_deg) if c.is_right else -deg_to_rad(static_toe_deg)

	c.camber_rad = _lut_buf[base + DoubleWishboneKinematics.CH_CAMBER] + static_camber_iso
	c.wheel_steer_rad = _lut_buf[base + DoubleWishboneKinematics.CH_WHEEL_STEER] + static_toe_iso
	c.toe_rad = _lut_buf[base + DoubleWishboneKinematics.CH_TOE] + static_toe_iso
	c.cp_offset_ros = Vector2(_lut_buf[base + DoubleWishboneKinematics.CH_CP_DX], _lut_buf[base + DoubleWishboneKinematics.CH_CP_DY])
	c.damper_disp_m = _lut_buf[base + DoubleWishboneKinematics.CH_DAMPER_DISP]
	c.motion_ratio = _lut_buf[base + DoubleWishboneKinematics.CH_MR]
	c.d_mr_dz = _lut_buf[base + DoubleWishboneKinematics.CH_DMR_DZ]
	c.arb_angle_rad = _lut_buf[base + DoubleWishboneKinematics.CH_ARB_ANGLE]
	c.arb_motion_ratio = _lut_buf[base + DoubleWishboneKinematics.CH_ARB_MR]
	c.fvsa_ic_y = _lut_buf[base + DoubleWishboneKinematics.CH_FVSA_IC_Y]
	c.fvsa_ic_z = _lut_buf[base + DoubleWishboneKinematics.CH_FVSA_IC_Z]
	c.fvsa_slope = _lut_buf[base + DoubleWishboneKinematics.CH_FVSA_SLOPE]
	c.roll_center_z = _lut_buf[base + DoubleWishboneKinematics.CH_RC_Z]
	c.svsa_slope = _lut_buf[base + DoubleWishboneKinematics.CH_SVSA_SLOPE]
	c.mechanical_trail_m = _lut_buf[base + DoubleWishboneKinematics.CH_MECH_TRAIL]
	c.scrub_radius_m = _lut_buf[base + DoubleWishboneKinematics.CH_SCRUB_RADIUS]


func evaluate_kinematics_only(compressions: Array, config: VehicleConfig, steer_cmd: float = 0.0) -> void:
	if not is_3d_active(config):
		return
	for i in range(4):
		var dz: float = float(compressions[i]) if i < compressions.size() else 0.0
		_sample_corner_lut(i, dz, steer_cmd, config)


func get_ackermann_steer_3d(steer_cmd: float, delta_z: float, is_left: bool, config: VehicleConfig) -> float:
	if not is_3d_active(config):
		return compute_ackermann_steer(steer_cmd, is_left, config)
	kinematics_3d.sample_front_into(delta_z, steer_cmd, not is_left, _steer_probe_buf, 0)
	return _steer_probe_buf[DoubleWishboneKinematics.CH_WHEEL_STEER]


func _evaluate_from_corners_3d(
	compressions: Array,
	comp_velocities: Array,
	grounded: Array,
	config: VehicleConfig,
	steer_cmd: float
) -> Array[SuspensionCorner]:
	for i in range(4):
		var c: SuspensionCorner = corners[i]
		var is_front: bool = (i < 2)
		var dz: float = float(compressions[i]) if i < compressions.size() else 0.0
		var dz_dot: float = float(comp_velocities[i]) if i < comp_velocities.size() else 0.0
		var is_g: bool = bool(grounded[i]) if i < grounded.size() else true

		c.is_grounded = is_g
		c.delta_z = dz
		c.delta_z_dot = dz_dot
		c.static_preload = compute_static_preload(i, config)

		_sample_corner_lut(i, dz, steer_cmd, config)

		var k_spring: float = config.spring_stiffness_front if is_front else config.spring_stiffness_rear
		var mr_0: float = kinematics_3d.mr_0_front if is_front else kinematics_3d.mr_0_rear
		var f_spring_shaft: float = (c.static_preload / maxf(mr_0, 1.0e-4)) + k_spring * c.damper_disp_m
		c.wheel_spring_stiffness = k_spring * c.motion_ratio * c.motion_ratio + f_spring_shaft * c.d_mr_dz

		if not is_g:
			c.spring_force = 0.0
			c.damper_force = 0.0
			c.bumpstop_force = 0.0
			c.arb_force = 0.0
			c.fz = 0.0
			continue

		c.spring_force = f_spring_shaft * c.motion_ratio - c.static_preload
		c.damper_force = _compute_damper_force_with_mr(is_front, dz_dot, c.motion_ratio, config)
		c.bumpstop_force = compute_bumpstop_force(dz, is_front, config)

	_apply_axle_arb_3d(VehicleConfig.WheelIndex.FL, VehicleConfig.WheelIndex.FR, true, config)
	_apply_axle_arb_3d(VehicleConfig.WheelIndex.RL, VehicleConfig.WheelIndex.RR, false, config)

	for i in range(4):
		var c: SuspensionCorner = corners[i]
		if c.is_grounded:
			var f_total: float = c.static_preload + c.spring_force + c.damper_force + c.bumpstop_force + c.arb_force
			c.fz = maxf(0.0, f_total)
		else:
			c.fz = 0.0

	return corners


func _apply_axle_arb_3d(left_idx: int, right_idx: int, is_front: bool, config: VehicleConfig) -> void:
	var c_l: SuspensionCorner = corners[left_idx]
	var c_r: SuspensionCorner = corners[right_idx]
	if not c_l.is_grounded or not c_r.is_grounded:
		c_l.arb_force = 0.0
		c_r.arb_force = 0.0
		return
	var f_pre_l: float = c_l.static_preload + c_l.spring_force + c_l.damper_force + c_l.bumpstop_force
	var f_pre_r: float = c_r.static_preload + c_r.spring_force + c_r.damper_force + c_r.bumpstop_force
	var arb_mr_0: float = kinematics_3d.arb_mr_0_front if is_front else kinematics_3d.arb_mr_0_rear
	var k_arb_wheel: float = config.arb_stiffness_front if is_front else config.arb_stiffness_rear
	var k_arb_bar: float = k_arb_wheel / maxf(arb_mr_0 * arb_mr_0, 1.0e-6)
	var t_arb: float = k_arb_bar * (c_l.arb_angle_rad - c_r.arb_angle_rad)
	var f_arb_l: float = t_arb * c_l.arb_motion_ratio
	var f_arb_r: float = -t_arb * c_r.arb_motion_ratio
	# Scale torsion-bar reaction proportionally if inside corner reaches zero normal load
	var scale: float = 1.0
	if f_pre_l + f_arb_l < 0.0 and f_arb_l < -1.0e-6:
		scale = minf(scale, maxf(0.0, f_pre_l) / (-f_arb_l))
	if f_pre_r + f_arb_r < 0.0 and f_arb_r < -1.0e-6:
		scale = minf(scale, maxf(0.0, f_pre_r) / (-f_arb_r))
	c_l.arb_force = f_arb_l * scale
	c_r.arb_force = f_arb_r * scale

func _apply_axle_arb(left_idx: int, right_idx: int, is_front: bool, config: VehicleConfig) -> void:
	var c_l: SuspensionCorner = corners[left_idx]
	var c_r: SuspensionCorner = corners[right_idx]
	if not c_l.is_grounded or not c_r.is_grounded:
		c_l.arb_force = 0.0
		c_r.arb_force = 0.0
		return
	var f_pre_l: float = c_l.static_preload + c_l.spring_force + c_l.damper_force + c_l.bumpstop_force
	var f_pre_r: float = c_r.static_preload + c_r.spring_force + c_r.damper_force + c_r.bumpstop_force
	var arb_pair: Vector2 = compute_arb_axle_forces(c_l.delta_z, c_r.delta_z, is_front, config)
	var f_arb_l: float = arb_pair.x
	# clamp lateral arb load transfer to available inside-wheel vertical force so liftoff cannot create net vertical jacking
	if f_pre_l + f_arb_l < 0.0:
		f_arb_l = -maxf(0.0, f_pre_l)
	if f_pre_r - f_arb_l < 0.0:
		f_arb_l = maxf(0.0, f_pre_r)
	c_l.arb_force = f_arb_l
	c_r.arb_force = -f_arb_l

## computes static vertical preload f_{0,i} (n) from chassis mass distribution
static func compute_static_preload(index: int, config: VehicleConfig) -> float:
	var loads: Dictionary = config.get_static_axle_loads()
	match index:
		VehicleConfig.WheelIndex.FL:
			return float(loads["wheel_fl"])
		VehicleConfig.WheelIndex.FR:
			return float(loads["wheel_fr"])
		VehicleConfig.WheelIndex.RL:
			return float(loads["wheel_rl"])
		_:
			return float(loads["wheel_rr"])

## wheel-referred vertical spring stiffness: k_wheel = k_spring * mr^2 (n/m)
static func get_wheel_spring_stiffness(is_front: bool, config: VehicleConfig) -> float:
	var k_spring: float = config.spring_stiffness_front if is_front else config.spring_stiffness_rear
	var mr: float = config.motion_ratio_front if is_front else config.motion_ratio_rear
	return k_spring * mr * mr

## series ride rate combining wheel-referred susp stiffness and tire vertical stiffness (n/m)
static func get_ride_rate(is_front: bool, config: VehicleConfig) -> float:
	var k_wheel: float = get_wheel_spring_stiffness(is_front, config)
	var k_z: float = maxf(config.tire_vertical_stiffness, 1.0)
	return (k_wheel * k_z) / maxf(k_wheel + k_z, 1.0)

## effective sprung-mass roll moment arm above the roll axis at cg: h_roll = h_s - h_rc,cg (m)
static func get_roll_moment_arm(config: VehicleConfig) -> float:
	return maxf(0.0, config.h_sprung_cog - config.roll_center_height_cg)

## geometric roll-center + unsprung-mass lateral load transfer per axle (n)
## positive result shifts load from left (+y) to right (-y) when turning left (ay_ros > 0, fy_axle > 0)
static func compute_non_elastic_lateral_load_transfer(
	fy_axle: float,
	ay_ros: float,
	is_front: bool,
	config: VehicleConfig,
	r_loaded: float = -1.0
) -> float:
	var track: float = maxf(config.track_width_f if is_front else config.track_width_r, 0.1)
	var h_rc: float = config.roll_center_height_front if is_front else config.roll_center_height_rear
	var sprung_ratio: float = clampf(config.sprung_mass / maxf(config.mass, 1e-3), 0.0, 1.0)
	var dfz_rc: float = (sprung_ratio * fy_axle * h_rc) / track
	var m_u_axle: float = 0.5 * maxf(config.unsprung_mass_total, 0.0)
	var h_u: float = r_loaded if r_loaded > 0.0 else config.wheel_radius
	var dfz_u: float = (m_u_axle * ay_ros * h_u) / track
	return dfz_rc + dfz_u

## applies non-elastic lateral load transfer (roll-center geometric + unsprung mass)
## across each grounded axle while preserving fz >= 0 and axle vertical force sum
func apply_non_elastic_lateral_load_transfer(
	fy_front: float,
	fy_rear: float,
	ay_ros: float,
	config: VehicleConfig
) -> void:
	for axle_idx in range(2):
		var is_front: bool = (axle_idx == 0)
		var l_idx: int = VehicleConfig.WheelIndex.FL if is_front else VehicleConfig.WheelIndex.RL
		var r_idx: int = VehicleConfig.WheelIndex.FR if is_front else VehicleConfig.WheelIndex.RR
		var c_l: SuspensionCorner = corners[l_idx]
		var c_r: SuspensionCorner = corners[r_idx]
		if not c_l.is_grounded or not c_r.is_grounded:
			continue
		var fy_axle: float = fy_front if is_front else fy_rear
		var dfz: float = compute_non_elastic_lateral_load_transfer(fy_axle, ay_ros, is_front, config)
		# clamp transfer so neither wheel goes below 0 n (conserves c_l.fz + c_r.fz)
		if dfz > 0.0:
			dfz = minf(dfz, c_l.fz)
		elif dfz < 0.0:
			dfz = -minf(-dfz, c_r.fz)
		c_l.fz -= dfz
		c_r.fz += dfz

## 3d dynamic fvsa jacking + unsprung mass lateral load transfer per corner (no double counting)
func apply_non_elastic_lateral_load_transfer_3d(
	fy_corners: Array[float],
	ay_ros: float,
	config: VehicleConfig
) -> void:
	var sprung_ratio: float = clampf(config.sprung_mass / maxf(config.mass, 1.0e-3), 0.0, 1.0)
	var m_u_axle: float = 0.5 * maxf(config.unsprung_mass_total, 0.0)
	var h_u: float = config.wheel_radius
	for axle_idx in range(2):
		var is_front: bool = (axle_idx == 0)
		var l_idx: int = VehicleConfig.WheelIndex.FL if is_front else VehicleConfig.WheelIndex.RL
		var r_idx: int = VehicleConfig.WheelIndex.FR if is_front else VehicleConfig.WheelIndex.RR
		var c_l: SuspensionCorner = corners[l_idx]
		var c_r: SuspensionCorner = corners[r_idx]
		if c_l.is_grounded and c_r.is_grounded:
			var track: float = maxf(config.track_width_f if is_front else config.track_width_r, 0.1)
			var dfz_u: float = (m_u_axle * ay_ros * h_u) / track
			if dfz_u > 0.0:
				dfz_u = minf(dfz_u, c_l.fz)
			elif dfz_u < 0.0:
				dfz_u = -minf(-dfz_u, c_r.fz)
			c_l.fz -= dfz_u
			c_r.fz += dfz_u
		if c_l.is_grounded:
			c_l.fz = maxf(0.0, c_l.fz - c_l.fvsa_slope * sprung_ratio * fy_corners[l_idx])
		if c_r.is_grounded:
			c_r.fz = maxf(0.0, c_r.fz + c_r.fvsa_slope * sprung_ratio * fy_corners[r_idx])

## aplica transferencia longitudinal cinematica instantanea (anti-dive del y anti-squat tras)
## conservando sum(fz) y fz >= 0 entre ejes
func apply_non_elastic_longitudinal_load_transfer(
	fx_front: float,
	fx_rear: float,
	config: VehicleConfig
) -> void:
	var c_fl: SuspensionCorner = corners[VehicleConfig.WheelIndex.FL]
	var c_fr: SuspensionCorner = corners[VehicleConfig.WheelIndex.FR]
	var c_rl: SuspensionCorner = corners[VehicleConfig.WheelIndex.RL]
	var c_rr: SuspensionCorner = corners[VehicleConfig.WheelIndex.RR]
	if not (c_fl.is_grounded and c_fr.is_grounded and c_rl.is_grounded and c_rr.is_grounded):
		return
	var wb: float = maxf(config.wheelbase, 0.2)
	var half_dfz: float = -0.5 * (config.h_cog / wb) * (
		config.anti_dive_ratio_front * fx_front
		+ config.anti_squat_ratio_rear * fx_rear
	)
	if half_dfz > 0.0:
		half_dfz = minf(half_dfz, minf(c_rl.fz, c_rr.fz))
	elif half_dfz < 0.0:
		half_dfz = -minf(-half_dfz, minf(c_fl.fz, c_fr.fz))
	c_fl.fz += half_dfz
	c_fr.fz += half_dfz
	c_rl.fz -= half_dfz
	c_rr.fz -= half_dfz

## 3d dynamic svsa wishbone vertical anti-dive / anti-squat force per corner (no pitch-torque double counting)
func apply_non_elastic_longitudinal_load_transfer_3d(
	fx_corners: Array[float],
	config: VehicleConfig
) -> void:
	for i in range(4):
		var c: SuspensionCorner = corners[i]
		if not c.is_grounded:
			continue
		var dfz_svsa: float = (-c.svsa_slope * fx_corners[i]) if i < 2 else (c.svsa_slope * fx_corners[i])
		c.fz = maxf(0.0, c.fz + dfz_svsa)

## wheel-referred asymmetric vertical damping: c_wheel = c_damper * mr^2 (n*s/m)
## uses digressive low/high-speed slopes when use_digressive_damper is enabled
static func get_wheel_damping(is_front: bool, comp_vel: float, config: VehicleConfig) -> float:
	var mr: float = config.motion_ratio_front if is_front else config.motion_ratio_rear
	return _get_wheel_damping_with_mr(is_front, comp_vel, mr, config)

static func _get_wheel_damping_with_mr(is_front: bool, comp_vel: float, mr: float, config: VehicleConfig) -> float:
	var c_damper: float = 0.0
	if config.use_digressive_damper and config.damper_knee_velocity > 0.0:
		var v_damper: float = absf(comp_vel) * mr
		if comp_vel >= 0.0:
			c_damper = config.damper_comp_low if v_damper <= config.damper_knee_velocity else config.damper_comp_high
		else:
			c_damper = config.damper_reb_low if v_damper <= config.damper_knee_velocity else config.damper_reb_high
	else:
		if comp_vel >= 0.0:
			c_damper = config.damping_compression_front if is_front else config.damping_compression_rear
		else:
			c_damper = config.damping_relaxation_front if is_front else config.damping_relaxation_rear
	return c_damper * mr * mr

## continuous 2-stage digressive damper force referred to wheel (n)
## evaluates shaft force at v_damper = mr * dz_dot and projects to wheel by * mr
static func compute_damper_force_wheel(is_front: bool, comp_vel: float, config: VehicleConfig) -> float:
	var mr: float = config.motion_ratio_front if is_front else config.motion_ratio_rear
	return _compute_damper_force_with_mr(is_front, comp_vel, mr, config)

static func _compute_damper_force_with_mr(is_front: bool, comp_vel: float, mr: float, config: VehicleConfig) -> float:
	if not config.use_digressive_damper or config.damper_knee_velocity <= 0.0:
		return _get_wheel_damping_with_mr(is_front, comp_vel, mr, config) * comp_vel
	var v_d: float = comp_vel * mr
	var v_knee: float = config.damper_knee_velocity
	var abs_vd: float = absf(v_d)
	var f_shaft: float = 0.0
	if v_d >= 0.0:
		if abs_vd <= v_knee:
			f_shaft = config.damper_comp_low * v_d
		else:
			f_shaft = config.damper_comp_low * v_knee + config.damper_comp_high * (abs_vd - v_knee)
	else:
		if abs_vd <= v_knee:
			f_shaft = -config.damper_reb_low * abs_vd
		else:
			f_shaft = -(config.damper_reb_low * v_knee + config.damper_reb_high * (abs_vd - v_knee))
	return f_shaft * mr

## 3d m^-1 wishbone anti-dive / anti-squat geometric reaction forces and net pitch torque (nm)
## sustains the chassis against braking dive (front) and acceleration squat (rear) without altering total sum(fz)
static func compute_anti_pitch_forces(fx_corners: Array, config: VehicleConfig) -> Dictionary:
	var fx0: float = float(fx_corners[0]) if fx_corners.size() > 0 else 0.0
	var fx1: float = float(fx_corners[1]) if fx_corners.size() > 1 else 0.0
	var fx2: float = float(fx_corners[2]) if fx_corners.size() > 2 else 0.0
	var fx3: float = float(fx_corners[3]) if fx_corners.size() > 3 else 0.0
	var dfz0: float = -config.anti_dive_ratio_front * fx0
	var dfz1: float = -config.anti_dive_ratio_front * fx1
	var dfz2: float = config.anti_squat_ratio_rear * fx2
	var dfz3: float = config.anti_squat_ratio_rear * fx3
	var pitch_torque_ros: float = config.h_cog * (
		config.anti_dive_ratio_front * (fx0 + fx1)
		+ config.anti_squat_ratio_rear * (fx2 + fx3)
	)
	return {
		"corner_forces": [dfz0, dfz1, dfz2, dfz3],
		"pitch_torque_ros": pitch_torque_ros
	}

## 3d m^-1 lateral jacking forces per corner [fl, fr, rl, rr] (n), zero-sum per axle
static func compute_jacking_forces(fy_corners: Array, config: VehicleConfig) -> Array[float]:
	var fy_f: float = (float(fy_corners[0]) + float(fy_corners[1])) if fy_corners.size() >= 2 else 0.0
	var fy_r: float = (float(fy_corners[2]) + float(fy_corners[3])) if fy_corners.size() >= 4 else 0.0
	var dfz_f: float = 0.5 * config.jacking_coeff_front * fy_f
	var dfz_r: float = 0.5 * config.jacking_coeff_rear * fy_r
	return [-dfz_f, dfz_f, -dfz_r, dfz_r]

## saca angulo de direccion del (iso +z ccw left) supporting pro-ackermann (> 0)
## and anti-ackermann (< 0, e.g. -0.11 on ter27)
static func compute_ackermann_steer(steer_angle: float, is_left: bool, config: VehicleConfig) -> float:
	var abs_s: float = absf(steer_angle)
	var ack: float = clampf(config.ackermann_factor, -1.0, 1.0)
	if abs_s < 1e-6 or is_zero_approx(ack) or config.wheelbase <= 1e-3:
		return steer_angle
	var half_t: float = 0.5 * config.track_width_f
	var r_turn: float = config.wheelbase / tan(abs_s)
	var is_inner: bool = (steer_angle > 0.0 and is_left) or (steer_angle < 0.0 and not is_left)
	var r_wheel: float = maxf(r_turn - half_t, 0.25 * config.wheelbase) if is_inner else (r_turn + half_t)
	var ack_mag: float = atan2(config.wheelbase, r_wheel)
	var delta_ack: float = ack_mag - abs_s
	return signf(steer_angle) * (abs_s + ack * delta_ack)

## elastic steering compliance deflection (rad) at front kingpin from tire mz and mechanical trail
## in iso 8855, positive fy produces negative mz and negative mechanical-trail torque (-fy * x_trail),
## yielding delta_comp < 0 (elastic understeer)
static func compute_steering_compliance(mz_tire: float, fy_tire: float, config: VehicleConfig, mechanical_trail_override: float = NAN) -> float:
	if config.steering_compliance_coeff_rad_nm <= 0.0:
		return 0.0
	var k_steer: float = maxf(config.steering_compliance_stiffness_nm_rad, 1.0)
	var trail_m: float = config.mechanical_trail_front if is_nan(mechanical_trail_override) else mechanical_trail_override
	var kingpin_torque: float = mz_tire - fy_tire * trail_m
	if absf(config.steering_compliance_coeff_rad_nm - 1.0 / k_steer) > 1e-9:
		return clampf(kingpin_torque * config.steering_compliance_coeff_rad_nm, -0.1, 0.1)
	return clampf(kingpin_torque / k_steer, -0.1, 0.1)

## progressive bump stop force (n, >= 0) engaging when delta_z exceeds bumpstop_start
static func compute_bumpstop_force(delta_z: float, is_front: bool, config: VehicleConfig) -> float:
	var start: float = config.bumpstop_start_front if is_front else config.bumpstop_start_rear
	if delta_z <= start:
		return 0.0
	var excess: float = delta_z - start
	var k1: float = config.bumpstop_stiffness_front if is_front else config.bumpstop_stiffness_rear
	var k2: float = config.bumpstop_quadratic_front if is_front else config.bumpstop_quadratic_rear
	return k1 * excess + k2 * excess * excess

## pure roll-moment arb forces per axle:
## f_arb_l = +k_arb_wheel * (dz_l - dz_r), f_arb_r = -k_arb_wheel * (dz_l - dz_r)
static func compute_arb_axle_forces(delta_z_left: float, delta_z_right: float, is_front: bool, config: VehicleConfig) -> Vector2:
	var k_arb: float = config.arb_stiffness_front if is_front else config.arb_stiffness_rear
	var f_left: float = k_arb * (delta_z_left - delta_z_right)
	return Vector2(f_left, -f_left)

## piecewise-linear 1d interpolation over [[delta_z_m, value], ...], clamped at boundaries
## returns 0.0 if the table is empty
static func lookup_1d_table(table: Array, delta_z: float) -> float:
	var n: int = table.size()
	if n == 0:
		return 0.0
	if n == 1:
		return float(table[0][1])

	var x0: float = float(table[0][0])
	if delta_z <= x0:
		return float(table[0][1])

	var x_last: float = float(table[n - 1][0])
	if delta_z >= x_last:
		return float(table[n - 1][1])

	for i in range(n - 1):
		var xa: float = float(table[i][0])
		var xb: float = float(table[i + 1][0])
		if delta_z >= xa and delta_z <= xb:
			var dx: float = xb - xa
			if absf(dx) < 1e-9:
				return float(table[i][1])
			var t: float = (delta_z - xa) / dx
			var ya: float = float(table[i][1])
			var yb: float = float(table[i + 1][1])
			return lerpf(ya, yb, t)

	return float(table[n - 1][1])

## returns iso 8855 camber gamma (rad) for corner at travel delta_z (m)
static func compute_corner_camber_iso(delta_z: float, is_front: bool, is_right: bool, config: VehicleConfig) -> float:
	var table: Array = config.camber_table_front if is_front else config.camber_table_rear
	var camber_vehicle: float = lookup_1d_table(table, delta_z)
	# negative vehicle camber = top inward: left leans right (-gamma), right leans left (+gamma)
	return -camber_vehicle if is_right else camber_vehicle

## returns effective ground-plane iso 8855 camber gamma_ground (rad) including:
## static camber + kinematic bump camber + chassis roll (-phi) + caster/kpi steer coupling
static func compute_ground_camber_iso(
	delta_z: float,
	steer_rad: float,
	chassis_roll_rad: float,
	is_front: bool,
	is_right: bool,
	config: VehicleConfig
) -> float:
	var gamma_susp: float = compute_corner_camber_iso(delta_z, is_front, is_right, config)
	var gamma_steer: float = 0.0
	if is_front:
		var caster_rad: float = deg_to_rad(config.caster_angle_deg)
		var kpi_rad: float = deg_to_rad(config.kpi_angle_deg)
		var kpi_sign: float = -1.0 if is_right else 1.0
		gamma_steer = caster_rad * sin(steer_rad) + kpi_sign * kpi_rad * (1.0 - cos(steer_rad))
	return gamma_susp - chassis_roll_rad + gamma_steer

## returns iso 8855 toe angle offset tau (rad) for corner at travel delta_z (m)
static func compute_corner_toe_iso(delta_z: float, is_front: bool, is_right: bool, config: VehicleConfig) -> float:
	var table: Array = config.toe_table_front if is_front else config.toe_table_rear
	var static_toe_deg: float = config.toe_front_deg if is_front else config.toe_rear_deg
	var toe_in_vehicle: float = lookup_1d_table(table, delta_z) + deg_to_rad(static_toe_deg)
	# positive toe-in: left points right (cw around +z => -tau), right points left (ccw => +tau)
	return toe_in_vehicle if is_right else -toe_in_vehicle
