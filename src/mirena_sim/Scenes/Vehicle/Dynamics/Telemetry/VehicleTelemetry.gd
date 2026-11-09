class_name VehicleTelemetry
extends RefCounted

## Vehicle Telemetry & Logging Interface for Mirena Dynamics
## Formats physical vehicle states for ROS bags and validation logs.

const VehicleDynamicsModel = preload("../Core/VehicleDynamicsModel.gd")

var timestamp: float = 0.0

# Kinematics (ROS Frame: +X Fwd, +Y Left, +Z Up)
var vx: float = 0.0
var vy: float = 0.0
var vz: float = 0.0
var ax: float = 0.0
var ay: float = 0.0
var az: float = 0.0
var yaw_rate: float = 0.0
var pitch_rate: float = 0.0
var roll_rate: float = 0.0
var steer_angle: float = 0.0

# Wheel speeds (rad/s)
var wheel_omega: Array[float] = [0.0, 0.0, 0.0, 0.0]

# Powertrain & Brakes (Nm)
var drive_torques: Array[float] = [0.0, 0.0, 0.0, 0.0]
var brake_torques: Array[float] = [0.0, 0.0, 0.0, 0.0]

# Tire forces (N)
var tire_fx: Array[float] = [0.0, 0.0, 0.0, 0.0]
var tire_fy: Array[float] = [0.0, 0.0, 0.0, 0.0]
var tire_fz: Array[float] = [0.0, 0.0, 0.0, 0.0]

# Slip quantities
var tire_kappa: Array[float] = [0.0, 0.0, 0.0, 0.0]
var tire_alpha: Array[float] = [0.0, 0.0, 0.0, 0.0]

## Updates telemetry fields directly from the simulated vehicle model.
func update_from_model(model: VehicleDynamicsModel, linear_accel_ros: Vector3, steer_rad: float, last_step_data: Dictionary = {}) -> void:
	vx = model.linear_velocity_ros.x
	vy = model.linear_velocity_ros.y
	vz = model.linear_velocity_ros.z
	
	ax = linear_accel_ros.x
	ay = linear_accel_ros.y
	az = linear_accel_ros.z
	
	roll_rate = model.angular_velocity_ros.x
	pitch_rate = model.angular_velocity_ros.y
	yaw_rate = model.angular_velocity_ros.z
	
	steer_angle = steer_rad
	
	var d_trqs: Array = last_step_data.get("drive_torques", [0.0, 0.0, 0.0, 0.0])
	var b_trqs: Array = last_step_data.get("brake_torques", [0.0, 0.0, 0.0, 0.0])
	
	for i in range(4):
		var w = model.wheels[i]
		wheel_omega[i] = w.omega
		drive_torques[i] = d_trqs[i] if d_trqs.size() > i else 0.0
		brake_torques[i] = b_trqs[i] if b_trqs.size() > i else 0.0
		tire_fx[i] = w.F_x
		tire_fy[i] = w.F_y
		tire_fz[i] = w.F_z
		tire_kappa[i] = w.kappa
		tire_alpha[i] = w.alpha

## Formats telemetry as a Dictionary matching CSV columns for validation scripts.
func to_dict() -> Dictionary:
	return {
		"timestamp": timestamp,
		"vx": vx, "vy": vy, "vz": vz,
		"ax": ax, "ay": ay, "az": az,
		"yaw_rate": yaw_rate, "pitch_rate": pitch_rate, "roll_rate": roll_rate,
		"steer_angle": steer_angle,
		"omega_fl": wheel_omega[0], "omega_fr": wheel_omega[1],
		"omega_rl": wheel_omega[2], "omega_rr": wheel_omega[3],
		"drive_trq_fl": drive_torques[0], "drive_trq_fr": drive_torques[1],
		"drive_trq_rl": drive_torques[2], "drive_trq_rr": drive_torques[3],
		"brake_trq_fl": brake_torques[0], "brake_trq_fr": brake_torques[1],
		"brake_trq_rl": brake_torques[2], "brake_trq_rr": brake_torques[3],
		"fx_fl": tire_fx[0], "fx_fr": tire_fx[1], "fx_rl": tire_fx[2], "fx_rr": tire_fx[3],
		"fy_fl": tire_fy[0], "fy_fr": tire_fy[1], "fy_rl": tire_fy[2], "fy_rr": tire_fy[3],
		"fz_fl": tire_fz[0], "fz_fr": tire_fz[1], "fz_rl": tire_fz[2], "fz_rr": tire_fz[3],
		"kappa_fl": tire_kappa[0], "kappa_fr": tire_kappa[1],
		"kappa_rl": tire_kappa[2], "kappa_rr": tire_kappa[3],
		"alpha_fl": tire_alpha[0], "alpha_fr": tire_alpha[1],
		"alpha_rl": tire_alpha[2], "alpha_rr": tire_alpha[3]
	}
