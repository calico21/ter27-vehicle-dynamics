class_name RewindBuffer
extends RefCounted

## 5.0 s circular buffer at 20 Hz (100 frames) for instant car state rewind.

const SAMPLE_INTERVAL_S: float = 0.05
const MAX_FRAMES: int = 100

var _frames: Array[Dictionary] = []
var _accum_s: float = 0.0


func clear() -> void:
	_frames.clear()
	_accum_s = 0.0


func get_buffer_duration_s() -> float:
	return float(maxi(0, _frames.size() - 1)) * SAMPLE_INTERVAL_S


func record_step(dt: float, car: Node) -> void:
	if car == null or not is_instance_valid(car) or not (car is RigidBody3D):
		return
	_accum_s += maxf(dt, 0.0)
	if not _frames.is_empty() and _accum_s < SAMPLE_INTERVAL_S:
		return
	_accum_s = 0.0

	var rb := car as RigidBody3D
	var w_omegas: Array[float] = [0.0, 0.0, 0.0, 0.0]
	var c_lin := Vector3.ZERO
	var c_ang := Vector3.ZERO
	var has_model := ("model" in car and car.model != null)
	if has_model:
		if "wheels" in car.model and car.model.wheels is Array:
			for i in range(mini(4, car.model.wheels.size())):
				if car.model.wheels[i] != null and "omega" in car.model.wheels[i]:
					w_omegas[i] = float(car.model.wheels[i].omega)
		if "chassis_state" in car.model and car.model.chassis_state != null:
			c_lin = Vector3(car.model.chassis_state.linear_velocity)
			c_ang = Vector3(car.model.chassis_state.angular_velocity)

	_frames.append({
		"transform": rb.global_transform,
		"lin_vel": rb.linear_velocity,
		"ang_vel": rb.angular_velocity,
		"gas": float(car.gas) if "gas" in car else 0.0,
		"steering": float(car.steering) if "steering" in car else 0.0,
		"brake": float(car.brake) if "brake" in car else 0.0,
		"wheel_omegas": w_omegas,
		"chassis_lin": c_lin,
		"chassis_ang": c_ang,
		"has_model": has_model,
	})
	while _frames.size() > MAX_FRAMES:
		_frames.pop_front()


func rewind_seconds(seconds: float, car: Node) -> bool:
	if _frames.is_empty() or car == null or not is_instance_valid(car) or not (car is RigidBody3D):
		return false
	var steps_back: int = clampi(int(roundf(seconds / SAMPLE_INTERVAL_S)), 1, _frames.size() - 1)
	var target_idx: int = maxi(0, _frames.size() - 1 - steps_back)
	var frame: Dictionary = _frames[target_idx]

	var rb := car as RigidBody3D
	rb.global_transform = frame["transform"]
	rb.linear_velocity = frame["lin_vel"]
	rb.angular_velocity = frame["ang_vel"]

	if "gas" in car:
		car.gas = 0.0
	if "steering" in car:
		car.steering = 0.0
	if "brake" in car:
		car.brake = 0.0
	if "input_conditioner" in car and car.input_conditioner != null and car.input_conditioner.has_method("reset"):
		car.input_conditioner.reset()

	if bool(frame.get("has_model", false)) and "model" in car and car.model != null:
		var w_om: Array = frame.get("wheel_omegas", [0.0, 0.0, 0.0, 0.0])
		if "wheels" in car.model and car.model.wheels is Array:
			for i in range(mini(4, car.model.wheels.size())):
				if car.model.wheels[i] != null and "omega" in car.model.wheels[i]:
					car.model.wheels[i].omega = float(w_om[i])
		if "chassis_state" in car.model and car.model.chassis_state != null:
			car.model.chassis_state.linear_velocity = Vector3(frame.get("chassis_lin", Vector3.ZERO))
			car.model.chassis_state.angular_velocity = Vector3(frame.get("chassis_ang", Vector3.ZERO))

	# Trim future frames so repeated rewinds step further back cleanly
	_frames.resize(target_idx + 1)
	_accum_s = 0.0
	return true
