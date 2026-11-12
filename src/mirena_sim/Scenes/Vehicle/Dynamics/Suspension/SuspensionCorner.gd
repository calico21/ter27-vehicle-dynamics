class_name SuspensionCorner
extends RefCounted

## Decoupled per-corner state & output container for SuspensionSystem.
## Represents a single wheel corner (FL, FR, RL, RR) in ISO 8855 / vehicle conventions:
## - delta_z > 0: compression (jounce) from static design ride height (delta_z = 0).
## - delta_z < 0: extension (droop) below static design ride height.
## - fz >= 0: total vertical normal force transmitted to road/chassis (N).
## - camber_rad, toe_rad: ISO 8855 tire orientation offsets (rad).

var index: int = 0
var is_front: bool = true
var is_right: bool = false
var is_grounded: bool = true

# Inputs (at wheel contact / hub vertical axis)
var delta_z: float = 0.0
var delta_z_dot: float = 0.0

# Component forces at wheel (N, +upward supporting chassis)
var static_preload: float = 0.0
var spring_force: float = 0.0
var damper_force: float = 0.0
var bumpstop_force: float = 0.0
var arb_force: float = 0.0
var fz: float = 0.0

# Kinematic alignment outputs in ISO 8855 wheel sign convention (rad)
var camber_rad: float = 0.0
var toe_rad: float = 0.0

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
