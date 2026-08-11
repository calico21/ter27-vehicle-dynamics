extends Node3D
class_name RosIns

## Simulated VectorNav VN-300, and the driver that reads it.
##
## The sim counterpart of mirena_hardware's `mirena_imu` node. On the car there
## is one device: a single measurement carries the raw IMU, the onboard INS
## solution and the GNSS fix, and the driver turns each measurement into four
## messages. This node is that device and that driver together, which is why it
## replaces the separate RosImu and RosGps it was split into -- two nodes with
## two clocks and two independent noise draws could not express the one property
## that matters most about the real thing: /sensors/imu/odom and
## /sensors/gps/fix come from the *same receiver*, and a filter handed both must
## not treat them as independent observations of where the car is. The driver
## reads the fix from the GNSS binary group and the odometry from the INS group,
## so they are no longer literally the same field -- but the INS solution is
## built from that receiver, so they stay drawn from one error process here.
##
## Published, all from one measurement, mirroring ImuNode::publish_loop:
##   /sensors/imu        sensor_msgs/Imu       always, in the cog frame
##   /sensors/lidar/imu  sensor_msgs/Imu       always, in the lidar frame
##   /sensors/imu/odom   nav_msgs/Odometry     only while the INS is tracking
##   /sensors/gps/fix    sensor_msgs/NavSatFix as soon as the receiver has a fix
##
## NOTE on `linear_acceleration`, carried over verbatim from imu_pipeline.hpp
## because it is the single easiest thing to get wrong here: the Imu messages
## carry **gravity-free** acceleration, which sensor_msgs/Imu does not specify --
## that field is defined as the raw accelerometer reading. It is deliberate and
## load bearing: mirena_state's EKF mechanises from gravity-free body
## acceleration. A recorded run confirms it, reading ~0.07 m/s^2 on a level
## stationary car rather than 9.81.

# --- Configuration ---
@export_group("IMU Settings")
@export var accel_noise_std: float = 0.05
@export var gyro_noise_std: float = 0.005
@export var bias_drift_std: float = 0.0001
@export var lpf_tau: float = 0.15 # Low-pass filter for physics jitter
## Covariance published on the Imu messages. Zero matches the hardware, which
## leaves all three matrices at zero -- confirmed over a recorded run, 0 of 26888
## messages carried one. Raise it only if something downstream starts needing a
## number there; nothing in this workspace reads it today.
@export var imu_covariance_scale: float = 0.0

@export_group("INS Settings")
## One rate for the whole device, because the car has one: every message below
## comes from the same measurement. The VN-300 streams at 400 Hz divided down;
## the driver asks for 100.
@export var ins_rate: float = 100.0
## Fraction of measurements the onboard filter has a solution for. The driver
## publishes odometry and a fix only while the INS reports mode 2 (tracking) and
## says nothing otherwise, so this gates rather than throttles -- which is why a
## recorded run shows the IMU at 100 Hz and both INS topics at 76.6 Hz off the
## very same stream (20587 of 26888 measurements).
@export_range(0.0, 1.0) var ins_tracking_fraction: float = 0.766
## Alignment delay: the hardware publishes no odometry at all before the onboard
## filter converges, because mirena_state seeds its EKF from the first odometry
## it sees and an unconverged solution would pin the run's origin on a position
## the sensor does not stand behind.
@export var ins_startup_delay_sec: float = 0.0
## Acquisition delay for the GNSS receiver itself, which is a different and much
## shorter job than the INS alignment above -- the driver publishes /sensors/gps/fix
## as soon as the receiver has a fix, without waiting for the filter. Kept small
## deliberately: the gap between this and ins_startup_delay_sec is the window in
## which the stack can see that GNSS is alive while the INS is still aligning.
@export var gnss_startup_delay_sec: float = 2.0

## Reported position uncertainty wanders inside this band, in metres of standard
## deviation. Defaults are the range measured over a recorded run: posU put
## pose.covariance[0] between 5.36 and 11.07 m^2.
@export var ins_pos_sigma_min: float = 2.32
@export var ins_pos_sigma_max: float = 3.33
## Correlation time of the position error, in seconds. A real INS/GNSS position
## error wanders slowly; it does not resample itself every measurement. An
## estimator tuned against white noise is tuned against the wrong thing, which is
## why this is a random walk and not a per-sample draw.
@export var ins_pos_tau: float = 30.0
## Velocity uncertainty band, likewise measured: velU put twist.covariance[0]
## between 0.018 and 0.059 (m/s)^2.
@export var ins_vel_sigma_min: float = 0.135
@export var ins_vel_sigma_max: float = 0.243
@export var ins_vel_tau: float = 5.0
## How slowly the reported uncertainty itself drifts across its band.
@export var ins_sigma_tau: float = 45.0
## Attitude uncertainty, radians of standard deviation, matching the yprU
## figures the filter reports: roll 1e-6, pitch 2e-6, yaw 9.9e-5 rad^2.
@export var ins_roll_sigma: float = 0.001
@export var ins_pitch_sigma: float = 0.0014
@export var ins_yaw_sigma: float = 0.01
@export var ins_att_tau: float = 20.0
## Published on twist.covariance for the angular rates. The onboard filter does
## not stream a rate uncertainty, so the driver substitutes this constant.
@export var ins_gyro_var: float = 1e-4

@export_group("Geodesy")
## Where the Godot world origin sits on the earth. The INS reports latitude and
## longitude; everything geodetic downstream is relative to the reference the
## driver latches from the first solution, so this only fixes the patch of earth
## the run happens on.
@export var origin_lat: float = 43.29454175
@export var origin_lon: float = -2.01119177
@export var origin_alt: float = 115.962

@export_group("ROS 2 Settings")
@export var ros_namespace: String = ""
## This node *is* the IMU, so this is its own mount frame -- the one the driver
## looks up car/cog <- car/imu and car/lidar <- car/imu against.
@export var frame_id: String = "car/imu"
@export var parent_frame_id: String = "car/cog"
@export var lidar_frame_id: String = "car/lidar"
@export var gnss_frame_id: String = "car/gnss"
@export var odom_frame_id: String = "odom"

# --- WGS84 Constants (mirena::GNSSConversion) ---
const WGS84_A = 6378137.0
const WGS84_E2 = 0.00669437999014

# --- Internal State ---
var is_initialized: bool = false

# Raw sensor state, in this node's own (IMU) axes.
var _last_imu_vel: Vector3 = Vector3.ZERO
var _lpf_accel: Vector3 = Vector3.ZERO
var _accel_bias: Vector3 = Vector3.ZERO
var _gyro_bias: Vector3 = Vector3.ZERO
var noisy_accel: Vector3 = Vector3.ZERO
var noisy_gyro: Vector3 = Vector3.ZERO

# INS solution error. One slow scalar in [0, 1] drives both the reported sigma
# and the size of the error, so the covariance a consumer reads always describes
# the error it is actually being handed -- which is the point of publishing one.
var _ins_quality: float = 0.5
var _ins_pos_sigma: float = 0.0
var _ins_vel_sigma: float = 0.0
var _ins_pos_err: Vector3 = Vector3.ZERO   # metres, NED
var _ins_vel_err: Vector3 = Vector3.ZERO   # m/s, sensor axes
var _att_err: Vector3 = Vector3.ZERO       # rad, (roll, pitch, yaw)
var _ins_elapsed: float = 0.0

## Where the odom frame's origin sits on the earth (mirena::IMU::GeodesicRef),
## latched from the first solution the driver trusts.
var _geodesic_ref: Vector3 = Vector3.ZERO
var _have_ref: bool = false
var _ref_car_origin: Transform3D = Transform3D.IDENTITY

# Mounting geometry, resolved once (ImuNode::ensure_mounting).
var _have_mounting: bool = false
var _cog_from_imu: Transform3D = Transform3D.IDENTITY
var _lidar_from_imu: Transform3D = Transform3D.IDENTITY

# --- ROS Components ---
var _node: RosNode
var _imu_in_cog_pub: RosPublisher
var _imu_in_lidar_pub: RosPublisher
var _odom_pub: RosPublisher
var _fix_pub: RosPublisher
var _timer: RosTimer
var _cog_msg: RosSensorMsgsImu
var _lidar_msg: RosSensorMsgsImu
var _odom_msg: RosNavMsgsOdometry
var _fix_msg: RosSensorMsgsNavSatFix

## Mounts are broadcast centrally by the vehicle's RosTransformManager.
##
## Only car/imu, matching transforms.yaml. car/gnss is deliberately absent: the
## fix is stamped in it but the real car does not publish it either, and a frame
## that resolves in sim and dangles on the car is worse than one that dangles in
## both.
func get_static_transforms() -> Array:
	return [{"transform": transform, "frame_id": frame_id, "parent_frame_id": parent_frame_id}]

func _ready() -> void:
	_node = RosNode.new()
	_node.init(name.to_snake_case(), ros_namespace.to_snake_case())

	# Sensor-layer outputs only, exactly the four the driver owns. State fusion
	# (/state/car) and health reporting are mirena_state's job, not this one's.
	_imu_in_lidar_pub = _node.create_publisher("/sensors/lidar/imu", "sensor_msgs/msg/Imu")
	_imu_in_cog_pub = _node.create_publisher("/sensors/imu", "sensor_msgs/msg/Imu")
	_odom_pub = _node.create_publisher("/sensors/imu/odom", "nav_msgs/msg/Odometry")
	_fix_pub = _node.create_publisher("/sensors/gps/fix", "sensor_msgs/msg/NavSatFix")

	_cog_msg = RosSensorMsgsImu.new()
	_lidar_msg = RosSensorMsgsImu.new()
	_odom_msg = RosNavMsgsOdometry.new()
	_fix_msg = RosSensorMsgsNavSatFix.new()
	_fill_imu_covariances()

	_timer = _node.create_timer(1.0 / ins_rate, _process_measurement)
	is_initialized = true

# --- Raw sensing -------------------------------------------------------------

func _physics_process(delta: float) -> void:
	if not is_initialized: return
	var body := get_parent() as RigidBody3D
	if not body: return

	# 1. Kinematic acceleration of *this point* on the rigid body, in the world.
	# Gravity is never added: the onboard filter removes it before the driver
	# ever sees a sample, so the reading is already gravity-free (see the note at
	# the top of this file).
	var arm_world := global_position - body.global_position
	var imu_vel := body.linear_velocity + body.angular_velocity.cross(arm_world)
	var accel_world := (imu_vel - _last_imu_vel) / delta
	_last_imu_vel = imu_vel

	# 2. Into the sensor's own axes.
	var inv_basis := global_basis.inverse()
	var local_accel := inv_basis * accel_world
	var local_gyro := inv_basis * body.angular_velocity

	# 3. Low-pass filter: physics engines create spikes that break factor graphs.
	_lpf_accel = _lpf_accel.lerp(local_accel, 1.0 - lpf_tau)

	# 4. Random-walk biases. Drift must scale with sqrt(delta) to be consistent.
	var drift_scale := sqrt(delta)
	_accel_bias += _noise_vec(bias_drift_std * drift_scale)
	_gyro_bias += _noise_vec(bias_drift_std * drift_scale)

	# 5. Final noisy readings.
	noisy_accel = _lpf_accel + _accel_bias + _noise_vec(accel_noise_std)
	noisy_gyro = local_gyro + _gyro_bias + _noise_vec(gyro_noise_std)

# --- The driver's publish loop ----------------------------------------------

## One measurement, converted -- the sim's Pipeline::process.
func _process_measurement() -> void:
	if not is_initialized: return
	var body := get_parent() as RigidBody3D
	if not body: return
	_ensure_mounting()

	var dt := 1.0 / ins_rate
	_ins_elapsed += dt
	_advance_ins_error(dt)

	var stamp = _node.now()

	# The attitude solution, shared by every message below exactly as the
	# driver's single `ned_attitude` is. Publishing an exact orientation on the
	# IMU and a perturbed one on the odometry would be two attitudes, which the
	# device does not have.
	var attitude := _noisy_attitude()

	# Always produced, both frames, from the one sample.
	_fill_imu(_cog_msg, stamp, parent_frame_id, _cog_from_imu, attitude)
	_imu_in_cog_pub.publish(_cog_msg)
	_fill_imu(_lidar_msg, stamp, lidar_frame_id, _lidar_from_imu, attitude)
	_imu_in_lidar_pub.publish(_lidar_msg)

	# One position error, two messages -- still. The driver now reads the fix
	# from the GNSS binary group rather than from posLla, so the two are no
	# longer literally the same field, but on the device the INS solution is
	# built from this receiver: they stay drawn from one error process here,
	# because a filter handed both must not treat them as independent.
	var lla := _ins_solution()

	# The receiver has a fix long before the INS converges, and the driver
	# publishes it from that moment. Before this change the whole GNSS side of
	# the stack was silent until the INS came up, which is precisely when
	# someone debugging a car that will not converge needs to see it.
	if _gnss_has_fix():
		_publish_fix(stamp, lla)

	if not _ins_tracking():
		return

	_publish_odom(body, stamp, lla, attitude)

## Whether the GNSS receiver itself has a position fix.
##
## Separate from, and earlier than, _ins_tracking(): acquiring satellites is a
## receiver job that finishes in seconds, while the INS alignment behind it can
## take a minute or more.
func _gnss_has_fix() -> bool:
	return _ins_elapsed >= gnss_startup_delay_sec

## VectorNav INS mode 2 is "tracking"; 0, 1 and 3 are no solution, aligning, and
## dead reckoning after GPS loss. None of them may publish odometry.
func _ins_tracking() -> bool:
	if _ins_elapsed < ins_startup_delay_sec:
		return false
	return randf() < ins_tracking_fraction

## Fill one Imu message for the frame `frame_from_imu` targets.
##
## Mirrors Pipeline::imu_message: rotate the sensor's reading into the target
## frame, then transfer it along the mounting arm. Angular acceleration is
## neglected in the transfer there, so it is neglected here too -- the sim
## carries the driver's approximation rather than papering over it.
func _fill_imu(msg: RosSensorMsgsImu, stamp, frame: String,
		frame_from_imu: Transform3D, attitude: Basis) -> void:
	var rot := frame_from_imu.basis
	var ang := rot * noisy_gyro
	# transfer_acceleration: a - w x (w x arm), arm being where the IMU sits in
	# the target frame.
	var acc := (rot * noisy_accel) - ang.cross(ang.cross(frame_from_imu.origin))
	# orientation_of: world <- frame = (world <- imu)(imu <- frame)
	var orientation := _to_ros_quat(Quaternion(attitude * rot.inverse().orthonormalized()))

	msg.header.stamp = stamp
	msg.header.frame_id = _node.resolve_frame(frame)
	_set_vec(msg.angular_velocity, ang)
	_set_vec(msg.linear_acceleration, acc)
	_set_quat(msg.orientation, orientation)

## The INS position solution, as latitude/longitude/altitude -- what posLla
## carries on the real device, and the one place the solution's position error
## enters. Everything geodetic downstream is derived from this, so the fix and
## the odometry cannot drift apart.
func _ins_solution() -> Vector3:
	# Ground truth of this sensor, in metres north/east/down of the world origin.
	var p := global_position
	var ned := Vector3(-p.z, p.x, -p.y) + _ins_pos_err

	var phi := deg_to_rad(origin_lat)
	var den := sqrt(1.0 - WGS84_E2 * sin(phi) * sin(phi))
	var m_rad := WGS84_A * (1.0 - WGS84_E2) / (den * den * den)
	var n_rad := WGS84_A / den
	return Vector3(
		origin_lat + rad_to_deg(ned.x / m_rad),
		origin_lon + rad_to_deg(ned.y / (n_rad * cos(phi))),
		origin_alt - ned.z)

func _publish_odom(body: RigidBody3D, stamp, lla: Vector3, attitude: Basis) -> void:
	# The odom frame is pinned to wherever the car was when the INS first had a
	# solution worth trusting, latched here so there is exactly one place it can
	# happen. Re-latched when the car is respawned, which the real driver has no
	# equivalent of -- without it a sim reset would teleport the whole stack.
	var car_origin: Transform3D = body.origin if "origin" in body else Transform3D.IDENTITY
	if not _have_ref or not car_origin.is_equal_approx(_ref_car_origin):
		_geodesic_ref = lla
		_ref_car_origin = car_origin
		_have_ref = true

	var ned := _lla_to_ned(lla, _geodesic_ref)
	# flip_x: NED -> the NWU world the stack publishes in.
	var imu_position := Vector3(ned.x, -ned.y, -ned.z)

	var cog_orientation := _to_ros_quat(Quaternion(
		attitude * _cog_from_imu.basis.inverse().orthonormalized()))

	# The IMU is not the car. Shift the reported position along the arm from the
	# IMU to the cog, rotated into the world by the car's own attitude -- the arm
	# is a body-frame quantity, and adding it unrotated is a heading-dependent
	# error the size of the arm.
	var cog_position := imu_position - (cog_orientation * _to_ros_vec(_cog_from_imu.origin))

	# Velocity and rate stay in the body frame, which is what nav_msgs/Odometry
	# asks for and what the EKF consumes as u, v and omega.
	# velBody: the solution's velocity in the sensor's own axes, error included.
	var imu_vel := global_basis.inverse() * _last_imu_vel
	var cog_ang := _cog_from_imu.basis * noisy_gyro
	# transfer_velocity: v - w x arm.
	var cog_vel := (_cog_from_imu.basis * (imu_vel + _ins_vel_err)) \
		- cog_ang.cross(_cog_from_imu.origin)

	_odom_msg.header.stamp = stamp
	_odom_msg.header.frame_id = _node.resolve_frame(odom_frame_id)
	_odom_msg.child_frame_id = _node.resolve_frame(parent_frame_id)

	_odom_msg.pose.pose.position.x = cog_position.x
	_odom_msg.pose.pose.position.y = cog_position.y
	_odom_msg.pose.pose.position.z = cog_position.z
	_set_quat(_odom_msg.pose.pose.orientation, cog_orientation)
	_set_vec(_odom_msg.twist.twist.linear, cog_vel)
	_set_vec(_odom_msg.twist.twist.angular, cog_ang)

	_fill_odom_covariances()
	_odom_pub.publish(_odom_msg)

func _publish_fix(stamp, lla: Vector3) -> void:
	_fix_msg.header.stamp = stamp
	_fix_msg.header.frame_id = _node.resolve_frame(gnss_frame_id)
	_fix_msg.status.status = 0    # STATUS_FIX
	_fix_msg.status.service = 1   # SERVICE_GPS
	_fix_msg.latitude = lla.x
	_fix_msg.longitude = lla.y
	_fix_msg.altitude = lla.z

	# The GNSS group reports a real per-axis uncertainty, unlike the INS's one
	# isotropic sigma, so this is DIAGONAL_KNOWN rather than APPROXIMATED. In
	# ENU order, as NavSatFix specifies -- the driver has to swap north and east
	# coming off the device, which reports NED.
	#
	# Vertical is worse than horizontal on any GNSS receiver, because the
	# satellites are all on one side of the antenna; 1.5x is the usual rule of
	# thumb for VDOP against HDOP.
	var var_h := pow(_ins_pos_sigma, 2)
	var var_v := pow(_ins_pos_sigma * 1.5, 2)
	_fix_msg.position_covariance[0] = var_h   # east
	_fix_msg.position_covariance[4] = var_h   # north
	_fix_msg.position_covariance[8] = var_v   # up
	_fix_msg.position_covariance_type = 2   # COVARIANCE_TYPE_DIAGONAL_KNOWN

	_fix_pub.publish(_fix_msg)

# --- Error model -------------------------------------------------------------

## Walk the INS solution's error forward one measurement.
func _advance_ins_error(dt: float) -> void:
	# Solution quality wanders slowly and bounded; sigma follows it and so does
	# the error, so a degrading fix widens the covariance and drifts further out
	# together rather than one without the other.
	_ins_quality = clampf(_ou_step(_ins_quality - 0.5, 0.25, ins_sigma_tau, dt) + 0.5, 0.0, 1.0)
	_ins_pos_sigma = lerpf(ins_pos_sigma_min, ins_pos_sigma_max, _ins_quality)
	_ins_vel_sigma = lerpf(ins_vel_sigma_min, ins_vel_sigma_max, _ins_quality)

	_ins_pos_err = _ou_vec(_ins_pos_err, _ins_pos_sigma, ins_pos_tau, dt)
	_ins_vel_err = _ou_vec(_ins_vel_err, _ins_vel_sigma, ins_vel_tau, dt)
	_att_err = Vector3(
		_ou_step(_att_err.x, ins_roll_sigma, ins_att_tau, dt),
		_ou_step(_att_err.y, ins_pitch_sigma, ins_att_tau, dt),
		_ou_step(_att_err.z, ins_yaw_sigma, ins_att_tau, dt))

## Ground-truth attitude perturbed by the solution's attitude error.
##
## The ROS axes map onto Godot's as x = -z, y = -x, z = +y, so a ROS roll is a
## rotation about *negative* Godot z and a ROS pitch about negative Godot x --
## only yaw keeps its sign. Zero-mean noise hides the difference statistically,
## which is exactly why it is worth writing down rather than leaving to be
## rediscovered. Applied on the left, so the perturbation is about world axes the
## way an attitude error is.
func _noisy_attitude() -> Basis:
	return Basis.from_euler(Vector3(-_att_err.y, _att_err.z, -_att_err.x)) * global_basis

## One step of an Ornstein-Uhlenbeck process: exponentially correlated noise of
## time constant `tau` whose stationary standard deviation is `sigma`. The exact
## discretisation, so the result does not depend on the rate it is stepped at.
func _ou_step(value: float, sigma: float, tau: float, dt: float) -> float:
	if tau <= 0.0:
		return randfn(0.0, sigma)
	var decay := exp(-dt / tau)
	return value * decay + randfn(0.0, sigma * sqrt(1.0 - decay * decay))

func _ou_vec(value: Vector3, sigma: float, tau: float, dt: float) -> Vector3:
	return Vector3(
		_ou_step(value.x, sigma, tau, dt),
		_ou_step(value.y, sigma, tau, dt),
		_ou_step(value.z, sigma, tau, dt))

func _noise_vec(std: float) -> Vector3:
	return Vector3(randfn(0, std), randfn(0, std), randfn(0, std))

# --- Covariances -------------------------------------------------------------

func _fill_imu_covariances() -> void:
	var a_var := pow(accel_noise_std, 2) * imu_covariance_scale
	var g_var := pow(gyro_noise_std, 2) * imu_covariance_scale
	for i in range(3):
		_cog_msg.linear_acceleration_covariance[i * 4] = a_var
		_cog_msg.angular_velocity_covariance[i * 4] = g_var
		_lidar_msg.linear_acceleration_covariance[i * 4] = a_var
		_lidar_msg.angular_velocity_covariance[i * 4] = g_var

## Refilled every solution, not once at startup: on the car these are the onboard
## filter's own 1-sigma uncertainties and they move as the fix gets better or
## worse. Row-major 6x6 over [x, y, z, roll, pitch, yaw] / [vx, vy, vz, wx, wy,
## wz]. Position is isotropic because the filter reports one posU figure, not a
## horizontal/vertical split.
func _fill_odom_covariances() -> void:
	var pos_var := pow(_ins_pos_sigma, 2)
	var vel_var := pow(_ins_vel_sigma, 2)

	var pose_cov := PackedFloat64Array()
	pose_cov.resize(36)
	var twist_cov := PackedFloat64Array()
	twist_cov.resize(36)
	for i in range(3):
		pose_cov[i * 7] = pos_var
		twist_cov[i * 7] = vel_var
		twist_cov[(i + 3) * 7] = ins_gyro_var
	pose_cov[21] = pow(ins_roll_sigma, 2)
	pose_cov[28] = pow(ins_pitch_sigma, 2)
	pose_cov[35] = pow(ins_yaw_sigma, 2)
	_odom_msg.pose.covariance = pose_cov
	_odom_msg.twist.covariance = twist_cov

# --- Geometry and conversions ------------------------------------------------

## Latch the mounting geometry once (ImuNode::ensure_mounting). The scene is the
## sim's TF: this node's own transform is car/cog <- car/imu, and the lidar's
## gives car/lidar <- car/imu. They are bolts, not motion, so re-deriving them
## per measurement would buy nothing and cost a failure mode.
func _ensure_mounting() -> void:
	if _have_mounting:
		return
	_have_mounting = true
	_cog_from_imu = transform

	var found := false
	var root := get_parent()
	if root:
		for child in root.get_children():
			if child != self and child is Node3D and "frame_id" in child \
					and child.frame_id == lidar_frame_id:
				_lidar_from_imu = (child as Node3D).transform.affine_inverse() * transform
				found = true
				break
	if not found:
		# Left at identity, which would make /sensors/lidar/imu a copy of
		# /sensors/imu -- plausible-looking and wrong, so say so out loud.
		push_warning("RosIns: no sibling mounted at '%s'; the lidar IMU stream will "
			% lidar_frame_id + "duplicate the cog one.")
	print("RosIns: mounting latched, IMU sits at %s in %s." % [_cog_from_imu.origin, parent_frame_id])

## Latitude/longitude/altitude to metres north, east and down of a reference --
## mirena::GNSSConversion::latLonAltToNED, ported so the odometry is derived from
## the fix by the same arithmetic the driver uses.
func _lla_to_ned(lla: Vector3, ref: Vector3) -> Vector3:
	var p := _geodetic_to_ecef(lla)
	var p_ref := _geodetic_to_ecef(ref)
	var d := p - p_ref

	var ref_lat := deg_to_rad(ref.x)
	var ref_lon := deg_to_rad(ref.y)
	var sin_lat := sin(ref_lat)
	var cos_lat := cos(ref_lat)
	var sin_lon := sin(ref_lon)
	var cos_lon := cos(ref_lon)

	return Vector3(
		-sin_lat * cos_lon * d.x - sin_lat * sin_lon * d.y + cos_lat * d.z,
		-sin_lon * d.x + cos_lon * d.y,
		-cos_lat * cos_lon * d.x - cos_lat * sin_lon * d.y - sin_lat * d.z)

func _geodetic_to_ecef(lla: Vector3) -> Vector3:
	var lat := deg_to_rad(lla.x)
	var lon := deg_to_rad(lla.y)
	var sin_lat := sin(lat)
	var cos_lat := cos(lat)
	var n_rad := WGS84_A / sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
	return Vector3(
		(n_rad + lla.z) * cos_lat * cos(lon),
		(n_rad + lla.z) * cos_lat * sin(lon),
		(n_rad * (1.0 - WGS84_E2) + lla.z) * sin_lat)

## Godot -> ROS. Godot is +x right, +y up, -z forward; ROS is +x forward, +y
## left, +z up. Right-handed both ways, so this is a change of basis and nothing
## more -- and applying it to the world gives north = -z, west = -x, up = +y.
func _to_ros_vec(v: Vector3) -> Vector3:
	return Vector3(-v.z, -v.x, v.y)

func _to_ros_quat(q: Quaternion) -> Quaternion:
	return Quaternion(-q.z, -q.x, q.y, q.w)

## Writes a Godot-axis vector into a ROS message field, converting on the way.
func _set_vec(out, v: Vector3) -> void:
	var r := _to_ros_vec(v)
	out.x = r.x
	out.y = r.y
	out.z = r.z

## Writes a quaternion that is *already* in ROS axes. Unlike _set_vec this
## converts nothing -- callers hold ROS orientations because they compose them
## with ROS positions.
func _set_quat(out, q: Quaternion) -> void:
	out.x = q.x
	out.y = q.y
	out.z = q.z
	out.w = q.w
