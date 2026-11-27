class_name SuspensionCorner
extends RefCounted

## decoupled per-corner state & output container for suspensionsystem
## represents a single wheel corner (fl, fr, rl, rr) in iso 8855 / vehicle conventions:
## delta_z > 0: compression (jounce) from static design ride height (delta_z = 0)
## delta_z < 0: extension (droop) below static design ride height
## fz >= 0: total vertical normal force transmitted to road/chassis (n)
## camber_rad, toe_rad: iso 8855 tire orientation offsets (rad)

var index: int = 0
var is_front: bool = true
var is_right: bool = false
var is_grounded: bool = true

# inputs (at wheel contact / hub vertical axis)
var delta_z: float = 0.0
var delta_z_dot: float = 0.0

# component forces at wheel (n, +upward supporting chassis)
var static_preload: float = 0.0
var spring_force: float = 0.0
var damper_force: float = 0.0
var bumpstop_force: float = 0.0
var arb_force: float = 0.0
var fz: float = 0.0

# kinematic alignment outputs in iso 8855 wheel sign convention (rad)
var camber_rad: float = 0.0
var toe_rad: float = 0.0

# 3d double-wishbone kinematic & elastokinematic outputs (populated when use_3d_kinematics == true)
var wheel_steer_rad: float = 0.0
var cp_offset_ros: Vector2 = Vector2.ZERO
var damper_disp_m: float = 0.0
var motion_ratio: float = 1.0
var d_mr_dz: float = 0.0
var arb_angle_rad: float = 0.0
var arb_motion_ratio: float = 0.0
var fvsa_ic_y: float = 0.0
var fvsa_ic_z: float = 0.0
var fvsa_slope: float = 0.0
var roll_center_z: float = 0.0
var svsa_slope: float = 0.0
var mechanical_trail_m: float = 0.0
var scrub_radius_m: float = 0.0
var wheel_spring_stiffness: float = 0.0

func _init(idx: int = 0, front: bool = true, right: bool = false) -> void:
	index = idx
	is_front = front
	is_right = right

func reset() -> void:
	is_grounded = true
	delta_z = 0.0
	delta_z_dot = 0.0
	static_preload = 0.0
	spring_force = 0.0
	damper_force = 0.0
	bumpstop_force = 0.0
	arb_force = 0.0
	fz = 0.0
	camber_rad = 0.0
	toe_rad = 0.0
	wheel_steer_rad = 0.0
	cp_offset_ros = Vector2.ZERO
	damper_disp_m = 0.0
	motion_ratio = 1.0
	d_mr_dz = 0.0
	arb_angle_rad = 0.0
	arb_motion_ratio = 0.0
	fvsa_ic_y = 0.0
	fvsa_ic_z = 0.0
	fvsa_slope = 0.0
	roll_center_z = 0.0
	svsa_slope = 0.0
	mechanical_trail_m = 0.0
	scrub_radius_m = 0.0
	wheel_spring_stiffness = 0.0
