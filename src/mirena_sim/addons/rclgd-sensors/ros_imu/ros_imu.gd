extends Node3D
class_name RosImu

# --- Configuration ---
@export_group("Sensor Settings")
@export var gravity_constant: float = 9.81 
@export var accel_noise_std: float = 0.05
@export var gyro_noise_std: float = 0.005
@export var bias_drift_std: float = 0.0001
@export var lpf_tau: float = 0.15 # Low-pass filter for physics jitter

@export_group("INS Settings")
## The simulated IMU mimics the hardware unit (VectorNav) 1:1: it also runs an
## onboard INS and publishes its solution as nav_msgs/Odometry, so the sensor
## layer is interchangeable between sim and real.
@export var ins_odom_rate: float = 50.0
@export var ins_pos_noise_std: float = 0.05
@export var ins_yaw_noise_std: float = 0.01
@export var ins_vel_noise_std: float = 0.05
@export var ins_omega_noise_std: float = 0.005
## Simulates the INS alignment delay (no odometry before GPS lock on hardware)
@export var ins_startup_delay_sec: float = 0.0

@export_group("ROS 2 Settings")
@export var ros_namespace : String = ""
@export var imu_rate: float = 100.0
@export var frame_id: String = "car/imu"
@export var parent_frame_id: String = "car/cog"
@export var odom_frame_id: String = "odom"



# --- Internal State ---
var is_initialized: bool = false
var _last_velocity: Vector3 = Vector3.ZERO
var _lpf_accel: Vector3 = Vector3.ZERO

var _accel_bias: Vector3 = Vector3.ZERO
var _gyro_bias: Vector3 = Vector3.ZERO

var noisy_accel: Vector3
var noisy_gyro: Vector3
var current_quat: Quaternion
var _ins_elapsed: float = 0.0

# --- ROS Components ---
var _node: RosNode
var _imu_pub: RosPublisher
var _odom_pub: RosPublisher
var _timer: RosTimer
var _odom_timer: RosTimer
var _msg: RosSensorMsgsImu
var _odom_msg: RosNavMsgsOdometry

## Mounts are broadcast centrally by the vehicle's RosTransformManager
func get_static_transforms() -> Array:
	return [{"transform": transform, "frame_id": frame_id, "parent_frame_id": parent_frame_id}]

func _ready() -> void:
	_node = RosNode.new()
	_node.init(name.to_snake_case(),ros_namespace.to_snake_case())
	_imu_pub = _node.create_publisher("/sensors/imu", "sensor_msgs/msg/Imu")
	_timer = _node.create_timer(1.0 / imu_rate, _publish_imu)
	_odom_pub = _node.create_publisher("/sensors/imu/odom", "nav_msgs/msg/Odometry")
	_odom_timer = _node.create_timer(1.0 / ins_odom_rate, _publish_ins_odom)

	_msg = RosSensorMsgsImu.new()
	_fill_static_covariances()
	_odom_msg = RosNavMsgsOdometry.new()
	_fill_odom_covariances()

	is_initialized = true

func _physics_process(delta: float) -> void:
	if not is_initialized: return
	var parent_body = get_parent() as RigidBody3D
	if not parent_body: return

	# 1. Calculate Proper Acceleration (Global Space)
	var current_velocity = parent_body.linear_velocity
	var raw_accel = (current_velocity - _last_velocity) / delta
	_last_velocity = current_velocity
	var proper_accel_g = raw_accel + Vector3(0, gravity_constant, 0)
	
	# 2. Transform to the body (cog) frame — matches the hardware driver, which
	# lever-arms and rotates all readings into the cog frame before publishing.
	var inv_basis = parent_body.global_transform.basis.inverse()
	var local_accel = inv_basis * proper_accel_g
	var local_gyro = inv_basis * parent_body.angular_velocity
	
	# 3. Apply Low-Pass Filter (LPF)
	# Physics engines create "spikes" that break Factor Graphs; LPF smooths them.
	_lpf_accel = _lpf_accel.lerp(local_accel, 1.0 - lpf_tau)

	# 4. Update Random Walk Biases
	# CRITICAL: Drift must scale with sqrt(delta) for mathematical consistency.
	var drift_scale = sqrt(delta)
	_accel_bias += _get_noise_vec(bias_drift_std * drift_scale)
	_gyro_bias += _get_noise_vec(bias_drift_std * drift_scale)

	# 5. Final Noisy Readings
	noisy_accel = _lpf_accel + _accel_bias + _get_noise_vec(accel_noise_std)
	noisy_gyro = local_gyro + _gyro_bias + _get_noise_vec(gyro_noise_std)
	current_quat = Quaternion(parent_body.global_transform.basis)

func _publish_imu() -> void:
	_msg.header.stamp = _node.now()
	# Readings are expressed in the body (cog) frame, like the hardware stream
	_msg.header.frame_id = _node.resolve_frame(parent_frame_id)
	
	# --- Coordinate Mapping: (Godot x,y,z) -> (ROS z,x,y) ---
	# As per your request for the specific mapping:
	_msg.linear_acceleration.x = -noisy_accel.z
	_msg.linear_acceleration.y = -noisy_accel.x
	_msg.linear_acceleration.z = noisy_accel.y
	
	_msg.angular_velocity.x = -noisy_gyro.z
	_msg.angular_velocity.y = -noisy_gyro.x
	_msg.angular_velocity.z = noisy_gyro.y
	
	# --- Orientation Mapping ---
	var ros_quat = Quaternion(
		-current_quat.z,
		-current_quat.x,
		current_quat.y,
		current_quat.w
	)
	#print(ros_basis.get_euler())
	
	_msg.orientation.x = ros_quat.x
	_msg.orientation.y = ros_quat.y
	_msg.orientation.z = ros_quat.z
	_msg.orientation.w = ros_quat.w
	
	_imu_pub.publish(_msg)

## Publishes the simulated onboard-INS solution, mimicking the hardware
## driver's /sensors/imu/odom contract (REP-105): pose in the odom frame,
## twist in the body frame (REP-103: x forward, y left — no sign flips).
func _publish_ins_odom() -> void:
	if not is_initialized: return
	var parent_body = get_parent() as RigidBody3D
	if not parent_body: return

	# Simulated INS alignment delay (hardware publishes nothing before GPS lock)
	_ins_elapsed += 1.0 / ins_odom_rate
	if _ins_elapsed < ins_startup_delay_sec: return

	# Ground-truth pose relative to the car's odom origin (reset on respawn)
	var origin: Transform3D = parent_body.origin if "origin" in parent_body else Transform3D.IDENTITY
	var odom_tf: Transform3D = origin.affine_inverse() * parent_body.global_transform

	_odom_msg.header.stamp = _node.now()
	_odom_msg.header.frame_id = _node.resolve_frame(odom_frame_id)
	_odom_msg.child_frame_id = _node.resolve_frame(parent_frame_id)

	# --- Pose (Godot x,y,z -> ROS: x=-z, y=-x, z=y) + noise ---
	_odom_msg.pose.pose.position.x = -odom_tf.origin.z + randfn(0, ins_pos_noise_std)
	_odom_msg.pose.pose.position.y = -odom_tf.origin.x + randfn(0, ins_pos_noise_std)
	_odom_msg.pose.pose.position.z = odom_tf.origin.y + randfn(0, ins_pos_noise_std)

	# Yaw noise: perturb the basis around the vertical axis before mapping
	var noisy_basis = odom_tf.basis.rotated(Vector3.UP, randfn(0, ins_yaw_noise_std))
	var q = Quaternion(noisy_basis)
	_odom_msg.pose.pose.orientation.x = -q.z
	_odom_msg.pose.pose.orientation.y = -q.x
	_odom_msg.pose.pose.orientation.z = q.y
	_odom_msg.pose.pose.orientation.w = q.w

	# --- Twist (body frame, REP-103) + noise ---
	var local_vel = parent_body.global_transform.basis.inverse() * parent_body.linear_velocity
	_odom_msg.twist.twist.linear.x = -local_vel.z + randfn(0, ins_vel_noise_std)
	_odom_msg.twist.twist.linear.y = -local_vel.x + randfn(0, ins_vel_noise_std)
	_odom_msg.twist.twist.linear.z = local_vel.y + randfn(0, ins_vel_noise_std)
	_odom_msg.twist.twist.angular.z = parent_body.angular_velocity.y + randfn(0, ins_omega_noise_std)

	_odom_pub.publish(_odom_msg)

func _fill_static_covariances() -> void:
	# GTSAM requires non-zero diagonals in the covariance matrix
	var a_var = pow(accel_noise_std, 2)
	var g_var = pow(gyro_noise_std, 2)

	for i in range(3):
		_msg.linear_acceleration_covariance[i * 4] = a_var
		_msg.angular_velocity_covariance[i * 4] = g_var

func _fill_odom_covariances() -> void:
	# Diagonal covariances from the noise model (row-major 6x6 over
	# [x, y, z, roll, pitch, yaw] / [vx, vy, vz, wx, wy, wz])
	var pos_var = pow(ins_pos_noise_std, 2)
	var yaw_var = pow(ins_yaw_noise_std, 2)
	var vel_var = pow(ins_vel_noise_std, 2)
	var omega_var = pow(ins_omega_noise_std, 2)

	var pose_cov = PackedFloat64Array()
	pose_cov.resize(36)
	var twist_cov = PackedFloat64Array()
	twist_cov.resize(36)
	for i in range(3):
		pose_cov[i * 7] = pos_var
		pose_cov[(i + 3) * 7] = yaw_var
		twist_cov[i * 7] = vel_var
		twist_cov[(i + 3) * 7] = omega_var
	_odom_msg.pose.covariance = pose_cov
	_odom_msg.twist.covariance = twist_cov

func _get_noise_vec(std: float) -> Vector3:
	return Vector3(randfn(0, std), randfn(0, std), randfn(0, std))
