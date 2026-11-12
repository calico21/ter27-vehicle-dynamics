class_name SuspensionSystem
extends RefCounted

## Decoupled 4-Corner Suspension Interface (Block B) for Mirena Dynamics (TeR27)
##
## Encapsulates per-corner vertical force generation, axle ARB roll coupling,
## asymmetric bump/rebound damping, progressive bumpstops, static preload,
## and 1D kinematic camber/toe lookup tables behind an implementation-agnostic
## interface so future multibody or nonlinear models can swap in cleanly.
##
## Conventions:
## - delta_z > 0: compression (jounce) relative to static design ride height (delta_z = 0).
## - Motion Ratio MR = d(x_spring) / d(z_wheel) => k_wheel = k_spring * MR^2, c_wheel = c_damper * MR^2.
## - ARB per axle: F_ARB_L = +k_arb_wheel * (dz_L - dz_R), F_ARB_R = -F_ARB_L (F_L + F_R = 0 pure roll moment).
## - Kinematics (Optimum Kinematics 1D tables, empty -> 0):
##   * Camber table stores vehicle convention (negative = top inward). Mapped to ISO 8855 gamma:
##     Left wheel: gamma_L = +table_val (top right/inward -> -gamma -> negative Fy thrust).
##     Right wheel: gamma_R = -table_val (top left/inward -> +gamma -> positive Fy thrust).
##   * Toe table stores vehicle convention (+ = toe-in). Mapped to ISO 8855 steer offset tau:
##     Left wheel: tau_L = -table_val (front turned right/CW around +Z).
##     Right wheel: tau_R = +table_val (front turned left/CCW around +Z).

const VehicleConfig = preload("../Core/VehicleConfig.gd")
const SuspensionCorner = preload("SuspensionCorner.gd")

var corners: Array[SuspensionCorner] = []

func _init() -> void:
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

## Evaluates all 4 corners given wheel compressions (m), velocities (m/s), and grounded flags.
## Returns the updated Array[SuspensionCorner] (FL, FR, RL, RR).
func evaluate_from_corners(
	compressions: Array,
	comp_velocities: Array,
	grounded: Array,
	config: VehicleConfig
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

		# Kinematic alignment evaluated at current corner travel regardless of contact
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
		var c_wheel: float = get_wheel_damping(is_front, dz_dot, config)

		c.spring_force = k_wheel * dz
		c.damper_force = c_wheel * dz_dot
		c.bumpstop_force = compute_bumpstop_force(dz, is_front, config)

	# Axle Anti-Roll Bar coupling (pure roll moment: F_ARB_L + F_ARB_R = 0)
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
	# Clamp lateral ARB load transfer to available inside-wheel vertical force so liftoff cannot create net vertical jacking
	if f_pre_l + f_arb_l < 0.0:
		f_arb_l = -maxf(0.0, f_pre_l)
	if f_pre_r - f_arb_l < 0.0:
		f_arb_l = maxf(0.0, f_pre_r)
	c_l.arb_force = f_arb_l
	c_r.arb_force = -f_arb_l

## Computes static vertical preload F_{0,i} (N) from chassis mass distribution.
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

## Wheel-referred vertical spring stiffness: k_wheel = k_spring * MR^2 (N/m).
static func get_wheel_spring_stiffness(is_front: bool, config: VehicleConfig) -> float:
	var k_spring: float = config.spring_stiffness_front if is_front else config.spring_stiffness_rear
	var mr: float = config.motion_ratio_front if is_front else config.motion_ratio_rear
	return k_spring * mr * mr

## Wheel-referred asymmetric vertical damping: c_wheel = c_damper * MR^2 (N*s/m).
static func get_wheel_damping(is_front: bool, comp_vel: float, config: VehicleConfig) -> float:
	var mr: float = config.motion_ratio_front if is_front else config.motion_ratio_rear
	var c_damper: float = 0.0
	if comp_vel >= 0.0:
		c_damper = config.damping_compression_front if is_front else config.damping_compression_rear
	else:
		c_damper = config.damping_relaxation_front if is_front else config.damping_relaxation_rear
	return c_damper * mr * mr

## Progressive bump stop force (N, >= 0) engaging when delta_z exceeds bumpstop_start.
static func compute_bumpstop_force(delta_z: float, is_front: bool, config: VehicleConfig) -> float:
	var start: float = config.bumpstop_start_front if is_front else config.bumpstop_start_rear
	if delta_z <= start:
		return 0.0
	var excess: float = delta_z - start
	var k1: float = config.bumpstop_stiffness_front if is_front else config.bumpstop_stiffness_rear
	var k2: float = config.bumpstop_quadratic_front if is_front else config.bumpstop_quadratic_rear
	return k1 * excess + k2 * excess * excess

## Pure roll-moment ARB forces per axle:
## F_ARB_L = +k_arb_wheel * (dz_L - dz_R), F_ARB_R = -k_arb_wheel * (dz_L - dz_R).
static func compute_arb_axle_forces(delta_z_left: float, delta_z_right: float, is_front: bool, config: VehicleConfig) -> Vector2:
	var k_arb: float = config.arb_stiffness_front if is_front else config.arb_stiffness_rear
	var f_left: float = k_arb * (delta_z_left - delta_z_right)
	return Vector2(f_left, -f_left)

## Piecewise-linear 1D interpolation over [[delta_z_m, value], ...], clamped at boundaries.
## Returns 0.0 if the table is empty.
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

## Returns ISO 8855 camber gamma (rad) for corner at travel delta_z (m).
static func compute_corner_camber_iso(delta_z: float, is_front: bool, is_right: bool, config: VehicleConfig) -> float:
	var table: Array = config.camber_table_front if is_front else config.camber_table_rear
	var camber_vehicle: float = lookup_1d_table(table, delta_z)
	# Negative vehicle camber = top inward: left leans right (-gamma), right leans left (+gamma)
	return -camber_vehicle if is_right else camber_vehicle

## Returns ISO 8855 toe angle offset tau (rad) for corner at travel delta_z (m).
static func compute_corner_toe_iso(delta_z: float, is_front: bool, is_right: bool, config: VehicleConfig) -> float:
	var table: Array = config.toe_table_front if is_front else config.toe_table_rear
	var toe_in_vehicle: float = lookup_1d_table(table, delta_z)
	# Positive toe-in: left points right (CW around +Z => -tau), right points left (CCW => +tau)
	return toe_in_vehicle if is_right else -toe_in_vehicle
