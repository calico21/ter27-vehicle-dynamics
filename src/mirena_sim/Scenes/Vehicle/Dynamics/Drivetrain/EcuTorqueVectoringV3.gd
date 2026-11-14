class_name EcuTorqueVectoringV3
extends RefCounted

## 4WD (TeR27) port of the ECU Torque Vectoring v3 + Adaptive RLS Traction Control pipeline.
## Faithfully ports gp_math, gp_ekf, gp_vehicle_model, gp_solver, gp_traction_control,
## and gp_torque_vectoring, generalized to 4 independent in-hub motors [FL, FR, RL, RR] at 200 Hz.

const TvSilBridgeScript = preload("res://addons/rclgd/tv_sil_bridge.gd")

const GP_FL: int = 0
const GP_FR: int = 1
const GP_RL: int = 2
const GP_RR: int = 3

enum TVMap {
	OPEN_4WD = 0,
	ENDURANCE_SAFE = 1,
	AUTOCROSS_NEUTRAL = 2,
	SKIDPAD_ATTACK = 3
}

const TV_MAP_PROFILES: Dictionary = {
	TVMap.OPEN_4WD: {
		"name": "OPEN_4WD",
		"front_split": 0.35,
		"rear_split": 0.65,
		"k_us": 0.0018,
		"max_mz": 0.0,
		"tc_kp": 0.0,
		"min_fx_reserve": 0.0
	},
	TVMap.ENDURANCE_SAFE: {
		"name": "ENDURANCE_SAFE",
		"front_split": 0.35,
		"rear_split": 0.65,
		"k_us": 0.0030,
		"max_mz": 550.0,
		"tc_kp": 28.0,
		"min_fx_reserve": 0.08
	},
	TVMap.AUTOCROSS_NEUTRAL: {
		"name": "AUTOCROSS_NEUTRAL",
		"front_split": 0.35,
		"rear_split": 0.65,
		"k_us": 0.0018,
		"max_mz": 750.0,
		"tc_kp": 22.5,
		"min_fx_reserve": 0.06
	},
	TVMap.SKIDPAD_ATTACK: {
		"name": "SKIDPAD_ATTACK",
		"front_split": 0.35,
		"rear_split": 0.65,
		"k_us": 0.0008,
		"max_mz": 850.0,
		"tc_kp": 20.0,
		"min_fx_reserve": 0.05
	}
}

# ── TeR27 Vehicle & Actuator Parameters ─────────────────────────────────────
const GP_MASS: float = 290.0
const GP_IZ: float = 137.2
const GP_WB: float = 1.540
const GP_LF: float = 0.847            # 0.55 * 1.540 (45% Front / 55% Rear static)
const GP_LR: float = 0.693            # 0.45 * 1.540
const GP_TRACK_F: float = 1.230
const GP_TRACK_R: float = 1.230
const GP_R_WHEEL: float = 0.2032
const GP_H_CG: float = 0.290
const GP_GRAVITY: float = 9.81

const GP_AERO_CL_FRONT: float = 2.70
const GP_AERO_CL_REAR: float = 3.30
const GP_AERO_AREA: float = 1.0
const GP_AIR_DENSITY: float = 1.225

const GP_MAX_TRQ_ACT: float = 312.5   # 25 Nm motor * 12.5 gear ratio [Nm at wheel]
const GP_P_MAX_WHL: float = 32000.0   # Per-inverter hardware power cap [W mech per wheel]
const GP_P_MAX_TOTAL_MECH: float = 72000.0 # 80 kW DC bus * 0.90 efficiency [W mech total]
const GP_MOTOR_MAX_RPM: float = 22000.0
const GP_MOTOR_TAPER_RATIO: float = 0.85
const GP_GEAR_RATIO: float = 12.5
const GP_I_WHEEL_EST: float = 0.5075  # [kg*m^2]

# ── Hardware Realism (CAN Latency & Encoder Quantization) ───────────────────
const DEFAULT_CAN_LATENCY_MS: float = 12.0
const DEFAULT_ENCODER_PPR: int = 4096
const RING_BUFFER_CAPACITY: int = 64

# ── Tire & Understeer Reference ─────────────────────────────────────────────
const GP_C_ALPHA_F: float = 35000.0
const GP_C_ALPHA_R: float = 32000.0
const GP_MU_NOM: float = 1.86
const GP_MU_SLIDE: float = 1.62
const GP_MU_DEG_FZ: float = 0.00012
const GP_K_US: float = 0.015
const GP_K_US_FZ: float = 0.0015

# ── Control, QP & Filtering Constants ───────────────────────────────────────
const GP_W_SMOOTH: float = 5.999
const GP_W_REG: float = 0.401
const GP_TV_MAX_MZ: float = 500.0
const GP_TV_WZ_I_MAX: float = 200.0
const GP_TV_RATE_LIMIT: float = 3252.3
const GP_STEER_DEADZONE_RAD: float = 0.0087
const GP_YAW_DEADZONE_RADS: float = 0.0175
const GP_ACCEL_LPF_TAU: float = 0.0200
const GP_UB_LPF_TAU: float = 0.0300
const FIXED_DT: float = 0.005         # 200 Hz ECU rate

# ── Traction Control RLS Constants ──────────────────────────────────────────
const GP_TC_KP: float = 23.113
const GP_TC_KI: float = 12.0
const GP_TC_B0: float = 14.0
const GP_TC_B1: float = -0.0018
const GP_TC_C_PAC: float = 1.65
const GP_TC_ALPHA_PEAK: float = 0.148
const GP_TC_I_MAX: float = 3.5
const GP_TC_V_KP_SCALE: float = 3.0
const GP_TC_ALPHA_KAPPA_LP: float = 0.85
const GP_TC_ALPHA_MU_EMA: float = 0.008
const GP_TC_MU_LO: float = 0.40
const GP_TC_MU_HI: float = 2.20
const GP_TC_MU_NOM: float = GP_MU_NOM
const GP_TC_MU_TAU_RECOVERY: float = 4.0
const GP_TC_V_MIN: float = 1.5
const GP_TC_CLAMP_BETA: float = 50.0

const KP_MAP: Array[float] = [
	100.0, 150.0, 250.0, 300.0,
	120.0, 180.0, 280.0, 350.0,
	150.0, 220.0, 320.0, 400.0,
	180.0, 260.0, 360.0, 450.0
]
const KI_MAP: Array[float] = [
	10.0, 15.0, 20.0, 25.0,
	12.0, 18.0, 22.0, 28.0,
	15.0, 20.0, 25.0, 30.0,
	18.0, 25.0, 30.0, 35.0
]
const KD_MAP: Array[float] = [
	5.0, 10.0, 15.0, 20.0,
	8.0, 12.0, 18.0, 22.0,
	10.0, 15.0, 20.0, 25.0,
	12.0, 18.0, 25.0, 30.0
]

# ── EKF State (2 states: [vy, bw]) ──────────────────────────────────────────
var ekf_x: Array[float] = [0.0, 0.0]
var ekf_p: Array[Array] = [[0.10, 0.0], [0.0, 0.01]]
var ekf_q: Array[float] = [0.0500, 0.0001]
var ekf_r_gps_vy: float = 0.040
var ekf_r_pseudo_vy: float = 0.250
var ekf_beta_est: float = 0.0
var ekf_vy_std: float = sqrt(0.10)
var ekf_wz_corrected: float = 0.0

# ── TV Controller State ─────────────────────────────────────────────────────
var wz_int: float = 0.0
var delta_prev: float = 0.0
var t_qp_prev: Array[float] = [0.0, 0.0, 0.0, 0.0]
var t_out_prev: Array[float] = [0.0, 0.0, 0.0, 0.0]
var t_ub_filt: Array[float] = [0.0, 0.0, 0.0, 0.0]
var t_lb_filt: Array[float] = [0.0, 0.0, 0.0, 0.0]
var _ub_filt_initialized: bool = false
var ax_filt: float = 0.0
var ay_filt: float = 0.0
var mz_sat_ratio: float = 1.0
var qp_residual: float = 0.0

# ── 4WD Traction Control RLS State ──────────────────────────────────────────
var tc_pi_integral: Array[float] = [0.0, 0.0, 0.0, 0.0]
var tc_kappa_filt: Array[float] = [0.0, 0.0, 0.0, 0.0]
var tc_mu_surface: Array[float] = [GP_TC_MU_NOM, GP_TC_MU_NOM, GP_TC_MU_NOM, GP_TC_MU_NOM]
var tc_omega_prev_ema: Array[float] = [0.0, 0.0, 0.0, 0.0]
var tc_rls_p: Array[float] = [1000.0, 1000.0, 1000.0, 1000.0]
var tc_rls_theta: Array[float] = [30000.0, 30000.0, 30000.0, 30000.0]
var tc_theta_prev: Array[float] = [30000.0, 30000.0, 30000.0, 30000.0]
var tc_kappa_prev: Array[float] = [0.0, 0.0, 0.0, 0.0]
var tc_fx_prev: Array[float] = [0.0, 0.0, 0.0, 0.0]
var tc_kappa_opt: Array[float] = [0.11, 0.11, 0.11, 0.11]

# ── Live Map Profile & Active Tunables ──────────────────────────────────────
const TV_MAPS_RES_DIR := "res://Scenes/Vehicle/Dynamics/Core/Config/tv_maps/"
const TV_MAPS_USER_DIR := "user://tv_maps/"
const TV_MAP_PROTECTED_FILE := "00_open_4wd.json"

var active_map_id: int = TVMap.AUTOCROSS_NEUTRAL
var active_map_path: String = ""
var active_map_filename: String = ""
var active_map_display_name: String = ""
var _available_tv_maps: Array[Dictionary] = []
var k_us_active: float:
	get: return live_k_us
	set(v): live_k_us = v
var tv_max_mz_active: float:
	get: return live_max_mz
	set(v): live_max_mz = v
var tc_kp_active: float:
	get: return live_tc_kp
	set(v): live_tc_kp = v
var min_fx_reserve_active: float:
	get: return live_min_fx_reserve
	set(v): live_min_fx_reserve = v
var front_split_active: float = 0.35
var rear_split_active: float = 0.65
var open_4wd_tc_enabled: bool = false
var _custom_params_active: bool = false

# ── Live DiL Tuning Parameters (mutable without resetting EKF) ──────────────
enum ControlLawMode { V3_PID_ALQP = 0, V4_SMC_HYBRID = 1 }
var control_law_mode: int = ControlLawMode.V3_PID_ALQP

enum LaunchState { LC_OFF = 0, LC_ARMED = 1, LC_ACTIVE = 2, LC_DONE = 3 }
const FAST_CORNER_VX_MS: float = 16.67
const LC_PRELOAD_TORQUE_NM: float = 8.0
const LC_EXIT_VX_MS: float = 24.5
const LC_TRANSITION_TIME_S: float = 0.15
const LC_DV_OPT_MIN: float = 0.85
const LC_KAPPA_OPT: float = 0.088
const LC_FRONT_SPLIT: float = 0.32

var active_event_mode: String = "autocross"
var live_eco_power_cap_kw: float = 0.0
var live_regen_boost: float = 1.0
var _skidpad_yaw_integral: float = 0.0
var _skidpad_lock_active: bool = false

var launch_state: int = LaunchState.LC_OFF
var _lc_arm_requested: bool = false
var _lc_done_timer: float = 0.0
var _lc_blend_alpha: float = 0.0
var _lc_pi_integral: Array[float] = [0.0, 0.0, 0.0, 0.0]

var live_k_us: float = 0.011
var live_max_mz: float = 750.0
var live_min_fx_reserve: float = 0.10
var live_tc_kp: float = 21.0
var live_k_beta: float = 4500.0
var live_throttle_mz_cut: float = 0.40
var live_grip_cap: float = 0.94

# ── V4_SMC_HYBRID Parameters & Notch / SMC State ────────────────────────────
var live_smc_k_gain: float = 920.0
var live_smc_k_ffd: float = 95.0
var live_smc_lambda_beta: float = 6.8
var live_smc_phi: float = 0.13
var live_smc_tc_kappa_base: float = 0.052

var _notch_x1: float = 0.0
var _notch_x2: float = 0.0
var _notch_y1: float = 0.0
var _notch_y2: float = 0.0
var _notch_delta_prev: float = 0.0
var _smc_wz_prev: float = 0.0
var _smc_wz_dot_filt: float = 0.0
var smc_s_int: float = 0.0
var last_smc_s: float = 0.0
var last_tc_kappa_lim: Array[float] = [0.040, 0.040, 0.040, 0.040]

# ── Hardware Realism (CAN Latency & Encoder Quantization) ───────────────────
var hardware_realism_enabled: bool = false:
	set(value):
		hardware_realism_enabled = value
		if value:
			if can_latency_ms <= 0.0:
				can_latency_ms = DEFAULT_CAN_LATENCY_MS
			sensor_quantization_enabled = true
		else:
			can_latency_ms = 0.0
			sensor_quantization_enabled = false
var can_latency_ms: float = 0.0
var cmd_latency_ms: float = 0.0
var encoder_ppr: int = DEFAULT_ENCODER_PPR
var sensor_quantization_enabled: bool = false

var _sensor_ring_buffer: Array[Dictionary] = []
var _sensor_rb_head: int = 0
var _sensor_rb_count: int = 0
var _cmd_ring_buffer: Array[Dictionary] = []
var _cmd_rb_head: int = 0
var _cmd_rb_count: int = 0
var _sim_time_s: float = 0.0

# ── C/C++ SIL Bridge ────────────────────────────────────────────────────────
var sil_bridge: TvSilBridgeScript = null
var use_native_sil: bool = false

# ── Fixed-rate 200 Hz Accumulator & Telemetry ───────────────────────────────
var _time_acc: float = 0.0
var last_t_cmd: Array[float] = [0.0, 0.0, 0.0, 0.0]
var last_wz_ref: float = 0.0
var last_mz_req: float = 0.0
var tc_active: bool = false

# ── Internal 8-Stage X-Ray Telemetry (zero-alloc in-place updates) ──────────
var last_vy_ss: float = 0.0
var last_ff_mz: float = 0.0
var last_p_mz: float = 0.0
var last_i_mz: float = 0.0
var last_beta_mz: float = 0.0
var last_mz_raw: float = 0.0
var last_kp: float = 0.0
var last_ki: float = 0.0
var last_kd: float = 0.0
var last_fz_est: Array[float] = [0.0, 0.0, 0.0, 0.0]
var last_fy_est: Array[float] = [0.0, 0.0, 0.0, 0.0]
var last_t_nominal: Array[float] = [0.0, 0.0, 0.0, 0.0]
var last_t_qp: Array[float] = [0.0, 0.0, 0.0, 0.0]
var last_tc_cut_nm: Array[float] = [0.0, 0.0, 0.0, 0.0]

func _init() -> void:
	sil_bridge = TvSilBridgeScript.new(false)
	set_map(TVMap.AUTOCROSS_NEUTRAL)
	reset()

func reset() -> void:
	ekf_x = [0.0, 0.0]
	ekf_p = [[0.10, 0.0], [0.0, 0.01]]
	ekf_beta_est = 0.0
	ekf_vy_std = sqrt(0.10)
	ekf_wz_corrected = 0.0

	wz_int = 0.0
	delta_prev = 0.0
	_notch_x1 = 0.0
	_notch_x2 = 0.0
	_notch_y1 = 0.0
	_notch_y2 = 0.0
	_notch_delta_prev = 0.0
	_smc_wz_prev = 0.0
	_smc_wz_dot_filt = 0.0
	smc_s_int = 0.0
	last_smc_s = 0.0
	for i in range(4):
		last_tc_kappa_lim[i] = live_smc_tc_kappa_base
	t_qp_prev = [0.0, 0.0, 0.0, 0.0]
	t_out_prev = [0.0, 0.0, 0.0, 0.0]
	t_ub_filt = [0.0, 0.0, 0.0, 0.0]
	t_lb_filt = [0.0, 0.0, 0.0, 0.0]
	_ub_filt_initialized = false
	ax_filt = 0.0
	ay_filt = 0.0
	mz_sat_ratio = 1.0
	qp_residual = 0.0

	tc_pi_integral = [0.0, 0.0, 0.0, 0.0]
	tc_kappa_filt = [0.0, 0.0, 0.0, 0.0]
	tc_mu_surface = [GP_TC_MU_NOM, GP_TC_MU_NOM, GP_TC_MU_NOM, GP_TC_MU_NOM]
	tc_omega_prev_ema = [0.0, 0.0, 0.0, 0.0]
	tc_rls_p = [1000.0, 1000.0, 1000.0, 1000.0]
	tc_rls_theta = [30000.0, 30000.0, 30000.0, 30000.0]
	tc_theta_prev = [30000.0, 30000.0, 30000.0, 30000.0]
	tc_kappa_prev = [0.0, 0.0, 0.0, 0.0]
	tc_fx_prev = [0.0, 0.0, 0.0, 0.0]
	tc_kappa_opt = [0.11, 0.11, 0.11, 0.11]

	_time_acc = 0.0
	last_t_cmd = [0.0, 0.0, 0.0, 0.0]
	last_wz_ref = 0.0
	last_mz_req = 0.0
	tc_active = false
	last_vy_ss = 0.0
	last_ff_mz = 0.0
	last_p_mz = 0.0
	last_i_mz = 0.0
	last_beta_mz = 0.0
	last_mz_raw = 0.0
	last_kp = 0.0
	last_ki = 0.0
	last_kd = 0.0
	for i in range(4):
		last_fz_est[i] = 0.0
		last_fy_est[i] = 0.0
		last_t_nominal[i] = 0.0
		last_t_qp[i] = 0.0
		last_tc_cut_nm[i] = 0.0
	_reset_ring_buffers()
	if sil_bridge != null and sil_bridge.is_using_native_sil():
		sil_bridge.gp_tv_init(null)

## Selects one of the 4 live TV calibration maps (OPEN_4WD, ENDURANCE_SAFE, AUTOCROSS_NEUTRAL, SKIDPAD_ATTACK).
func set_map(map_id: int) -> void:
	var clamped_id: int = clampi(map_id, TVMap.OPEN_4WD, TVMap.SKIDPAD_ATTACK)
	active_map_id = clamped_id
	control_law_mode = ControlLawMode.V3_PID_ALQP
	var profile: Dictionary = TV_MAP_PROFILES[clamped_id]
	front_split_active = float(profile["front_split"])
	rear_split_active = float(profile["rear_split"])
	k_us_active = float(profile["k_us"])
	tv_max_mz_active = float(profile["max_mz"])
	tc_kp_active = float(profile["tc_kp"])
	min_fx_reserve_active = float(profile["min_fx_reserve"])
	live_k_us = k_us_active
	live_max_mz = tv_max_mz_active
	live_tc_kp = tc_kp_active
	live_min_fx_reserve = min_fx_reserve_active
	live_k_beta = 4500.0
	live_throttle_mz_cut = 0.40
	live_grip_cap = 0.87 if clamped_id == TVMap.ENDURANCE_SAFE else (0.97 if clamped_id == TVMap.SKIDPAD_ATTACK else 0.94)
	_custom_params_active = false

## Cycles to the next live TV map across all JSON files in tv_maps/ and returns the new index.
func cycle_map() -> int:
	var maps: Array[Dictionary] = list_available_tv_maps()
	if maps.is_empty():
		var next_id: int = (active_map_id + 1) % 4
		set_map(next_id)
		return active_map_id
	var cur_idx: int = -1
	for i in range(maps.size()):
		if String(maps[i]["path"]) == active_map_path:
			cur_idx = i
			break
	if cur_idx < 0:
		cur_idx = clampi(active_map_id, 0, maps.size() - 1)
	var next_idx: int = (cur_idx + 1) % maps.size()
	load_tv_map_from_file(String(maps[next_idx]["path"]))
	return next_idx

## Returns the human-readable name of the currently active TV map profile.
func get_map_name() -> String:
	if _custom_params_active:
		return "CUSTOM"
	if active_map_display_name != "":
		return active_map_display_name
	if TV_MAP_PROFILES.has(active_map_id):
		return String(TV_MAP_PROFILES[active_map_id]["name"])
	return "UNKNOWN"

## Applies custom live calibration parameters without altering the static reference constants.
func set_custom_parameters(k_us: float, max_mz: float, tc_kp: float, min_fx_reserve: float) -> void:
	k_us_active = k_us
	tv_max_mz_active = maxf(max_mz, 0.0)
	tc_kp_active = maxf(tc_kp, 0.0)
	min_fx_reserve_active = clampf(min_fx_reserve, 0.0, 0.95)
	live_k_us = k_us_active
	live_max_mz = tv_max_mz_active
	live_tc_kp = tc_kp_active
	live_min_fx_reserve = min_fx_reserve_active
	_custom_params_active = true

## Returns a Dictionary snapshot of all live-tunable DiL TV parameters.
func get_live_params() -> Dictionary:
	return {
		"control_law_mode": float(control_law_mode),
		"live_k_us": live_k_us,
		"live_max_mz": live_max_mz,
		"live_min_fx_reserve": live_min_fx_reserve,
		"live_tc_kp": live_tc_kp,
		"live_k_beta": live_k_beta,
		"live_throttle_mz_cut": live_throttle_mz_cut,
		"live_grip_cap": live_grip_cap,
		"live_smc_k_gain": live_smc_k_gain,
		"live_smc_k_ffd": live_smc_k_ffd,
		"live_smc_lambda_beta": live_smc_lambda_beta,
		"live_smc_phi": live_smc_phi,
		"live_smc_tc_kappa_base": live_smc_tc_kappa_base,
		"front_drive_bias": front_split_active,
		"can_latency_ms": can_latency_ms
	}

## Updates a single live TV parameter in-place without resetting EKF or thermal states.
func set_live_param(key: String, val: float) -> void:
	match key:
		"control_law_mode":
			control_law_mode = ControlLawMode.V4_SMC_HYBRID if int(round(val)) == 1 else ControlLawMode.V3_PID_ALQP
		"live_k_us", "k_us":
			live_k_us = clampf(val, 0.0, 0.05)
			k_us_active = live_k_us
		"live_max_mz", "max_mz":
			live_max_mz = clampf(val, 0.0, 1500.0)
			tv_max_mz_active = live_max_mz
		"live_min_fx_reserve", "min_fx_reserve":
			live_min_fx_reserve = clampf(val, 0.0, 0.95)
			min_fx_reserve_active = live_min_fx_reserve
		"live_tc_kp", "tc_kp":
			live_tc_kp = clampf(val, 0.0, 60.0)
			tc_kp_active = live_tc_kp
		"live_k_beta", "k_beta":
			live_k_beta = clampf(val, 0.0, 15000.0)
		"live_throttle_mz_cut", "throttle_mz_cut":
			live_throttle_mz_cut = clampf(val, 0.0, 1.0)
		"live_grip_cap", "grip_cap":
			live_grip_cap = clampf(val, 0.50, 1.15)
		"live_smc_k_gain", "smc_k_gain":
			live_smc_k_gain = clampf(val, 100.0, 2500.0)
		"live_smc_k_ffd", "smc_k_ffd":
			live_smc_k_ffd = clampf(val, 0.0, 600.0)
		"live_smc_lambda_beta", "smc_lambda_beta":
			live_smc_lambda_beta = clampf(val, 0.0, 15.0)
		"live_smc_phi", "smc_phi":
			live_smc_phi = clampf(val, 0.02, 1.0)
		"live_smc_tc_kappa_base", "smc_tc_kappa_base":
			live_smc_tc_kappa_base = clampf(val, 0.01, 0.15)
		"front_drive_bias":
			front_split_active = clampf(val, 0.0, 1.0)
			rear_split_active = 1.0 - front_split_active
		"can_latency_ms":
			can_latency_ms = clampf(val, 0.0, 100.0)
			hardware_realism_enabled = can_latency_ms > 0.0
	_custom_params_active = true

## Arms Launch Control when stationary (vx < 0.5 m/s) with brake held (> 0.25).
func request_launch_control_arm(vx: float = 0.0, brake_norm: float = 1.0) -> bool:
	if absf(vx) < 0.5 and brake_norm > 0.25:
		launch_state = LaunchState.LC_ARMED
		_lc_arm_requested = false
		_lc_done_timer = 0.0
		_lc_blend_alpha = 1.0
		_lc_pi_integral = [0.0, 0.0, 0.0, 0.0]
		return true
	_lc_arm_requested = false
	return false

## Cancels Launch Control and returns smoothly to normal TV control.
func disarm_launch_control() -> void:
	launch_state = LaunchState.LC_OFF
	_lc_arm_requested = false
	_lc_done_timer = 0.0
	_lc_blend_alpha = 0.0
	_lc_pi_integral = [0.0, 0.0, 0.0, 0.0]

func get_launch_state_name() -> String:
	match launch_state:
		LaunchState.LC_ARMED:
			return "LC_ARMED"
		LaunchState.LC_ACTIVE:
			return "LC_ACTIVE"
		LaunchState.LC_DONE:
			return "LC_DONE"
		_:
			return "LC_OFF"

func get_event_mode_display() -> String:
	match active_event_mode:
		"endurance":
			return "Endurance Eco"
		"skidpad":
			return "Skidpad Attack"
		"wet":
			return "Wet"
		"acceleration":
			return "Acceleration"
		_:
			return "Autocross Pivot"

## Selects the first TV map matching the requested Formula Student event mode.
func set_event_mode(mode_key: String) -> bool:
	var target: String = mode_key.to_lower().strip_edges()
	var maps: Array[Dictionary] = list_available_tv_maps()
	for entry in maps:
		var path: String = String(entry.get("path", ""))
		var meta: Dictionary = _read_tv_map_metadata(path)
		var ev: String = String(meta.get("event_mode", "")).to_lower().strip_edges()
		if ev == target:
			return load_tv_map_from_file(path)
	active_event_mode = target
	return false

## Exports the current live TV tuning parameters as a C/C++ #define header snippet for TeR_ECU.
func export_c_header_snippet() -> String:
	var dt := Time.get_datetime_string_from_system(false, true)
	var setup_label := active_map_display_name if active_map_display_name != "" else get_map_name()
	return (
		"// TeR27 ECU Torque Vectoring Live Calibration: %s (%s)\n" % [setup_label, dt]
		+ "#define GP_K_US              %.5ff\n" % live_k_us
		+ "#define GP_TV_MAX_MZ         %.1ff\n" % live_max_mz
		+ "#define GP_MIN_FX_RESERVE    %.4ff\n" % live_min_fx_reserve
		+ "#define GP_TC_KP             %.3ff\n" % live_tc_kp
		+ "#define GP_K_BETA            %.1ff\n" % live_k_beta
		+ "#define GP_THROTTLE_MZ_CUT   %.3ff\n" % live_throttle_mz_cut
		+ "#define GP_FRONT_DRIVE_BIAS  %.3ff\n" % front_split_active
		+ "#define GP_CAN_LATENCY_MS    %.1ff\n" % can_latency_ms
	)

## Scans res://.../tv_maps/ and user://tv_maps/ for .json TV setups sorted alphabetically.
func list_available_tv_maps() -> Array[Dictionary]:
	var entries: Array[Dictionary] = []
	var seen_names: Dictionary = {}
	for dir_path in [TV_MAPS_RES_DIR, TV_MAPS_USER_DIR]:
		var d := DirAccess.open(dir_path)
		if d == null:
			continue
		d.list_dir_begin()
		var fname := d.get_next()
		while fname != "":
			if not d.current_is_dir() and fname.to_lower().ends_with(".json"):
				var full_path: String = dir_path + fname
				var meta := _read_tv_map_metadata(full_path)
				var base_name := fname.get_basename()
				if not seen_names.has(full_path):
					seen_names[full_path] = true
					entries.append({
						"id": entries.size(),
						"name": String(meta.get("name", base_name)),
						"filename": base_name,
						"path": full_path,
						"description": String(meta.get("description", ""))
					})
			fname = d.get_next()
		d.list_dir_end()
	entries.sort_custom(func(a: Dictionary, b: Dictionary) -> bool:
		return String(a["filename"]) < String(b["filename"])
	)
	for i in range(entries.size()):
		entries[i]["id"] = i
	_available_tv_maps = entries
	return entries

func _read_tv_map_metadata(file_path: String) -> Dictionary:
	var f := FileAccess.open(file_path, FileAccess.READ)
	if f == null:
		return {}
	var parsed = JSON.parse_string(f.get_as_text())
	f.close()
	if parsed is Dictionary:
		return parsed
	return {}

## Loads a TV setup JSON file and applies its parameters in-place without resetting EKF or thermal states.
func load_tv_map_from_file(file_path: String) -> bool:
	var f := FileAccess.open(file_path, FileAccess.READ)
	if f == null:
		return false
	var parsed = JSON.parse_string(f.get_as_text())
	f.close()
	if not (parsed is Dictionary):
		return false
	var data: Dictionary = parsed
	var params: Dictionary = data.get("params", {})
	var law_str: String = String(data.get("control_law", params.get("control_law", "v3"))).to_lower().strip_edges()
	control_law_mode = ControlLawMode.V4_SMC_HYBRID if (law_str == "smc" or law_str == "v4" or law_str == "v4_smc_hybrid") else ControlLawMode.V3_PID_ALQP
	var ev_str: String = String(data.get("event_mode", params.get("event_mode", ""))).to_lower().strip_edges()
	if ev_str != "":
		active_event_mode = ev_str
	else:
		var fn_low: String = file_path.get_file().get_basename().to_lower()
		if fn_low.contains("endurance"):
			active_event_mode = "endurance"
		elif fn_low.contains("skidpad"):
			active_event_mode = "skidpad"
		elif fn_low.contains("wet"):
			active_event_mode = "wet"
		elif fn_low.contains("accel") or fn_low.contains("launch"):
			active_event_mode = "acceleration"
		else:
			active_event_mode = "autocross"
	live_eco_power_cap_kw = clampf(float(params.get("eco_power_cap_kw", data.get("eco_power_cap_kw", 0.0))), 0.0, 120.0)
	live_regen_boost = clampf(float(params.get("regen_boost", data.get("regen_boost", 1.0))), 0.5, 1.5)
	_skidpad_yaw_integral = 0.0
	_skidpad_lock_active = false
	if params.has("k_us"):
		live_k_us = clampf(float(params["k_us"]), 0.0, 0.05)
		k_us_active = live_k_us
	if params.has("max_mz"):
		live_max_mz = clampf(float(params["max_mz"]), 0.0, 1500.0)
		tv_max_mz_active = live_max_mz
	if params.has("min_fx_reserve"):
		live_min_fx_reserve = clampf(float(params["min_fx_reserve"]), 0.0, 0.95)
		min_fx_reserve_active = live_min_fx_reserve
	if params.has("tc_kp"):
		live_tc_kp = clampf(float(params["tc_kp"]), 0.0, 60.0)
		tc_kp_active = live_tc_kp
	if params.has("k_beta"):
		live_k_beta = clampf(float(params["k_beta"]), 0.0, 15000.0)
	if params.has("throttle_mz_cut"):
		live_throttle_mz_cut = clampf(float(params["throttle_mz_cut"]), 0.0, 1.0)
	if params.has("grip_cap"):
		live_grip_cap = clampf(float(params["grip_cap"]), 0.50, 1.15)
	elif data.has("grip_cap"):
		live_grip_cap = clampf(float(data["grip_cap"]), 0.50, 1.15)
	else:
		live_grip_cap = 0.94
	if params.has("smc_k_gain"):
		live_smc_k_gain = clampf(float(params["smc_k_gain"]), 100.0, 2500.0)
	if params.has("smc_k_ffd"):
		live_smc_k_ffd = clampf(float(params["smc_k_ffd"]), 0.0, 600.0)
	if params.has("smc_lambda_beta"):
		live_smc_lambda_beta = clampf(float(params["smc_lambda_beta"]), 0.0, 15.0)
	if params.has("smc_phi"):
		live_smc_phi = clampf(float(params["smc_phi"]), 0.02, 1.0)
	if params.has("smc_tc_kappa_base"):
		live_smc_tc_kappa_base = clampf(float(params["smc_tc_kappa_base"]), 0.01, 0.15)
	if params.has("front_drive_bias"):
		front_split_active = clampf(float(params["front_drive_bias"]), 0.0, 1.0)
		rear_split_active = 1.0 - front_split_active
	if params.has("can_latency_ms"):
		can_latency_ms = clampf(float(params["can_latency_ms"]), 0.0, 100.0)
		hardware_realism_enabled = can_latency_ms > 0.0
	active_map_path = file_path
	active_map_filename = file_path.get_file().get_basename()
	active_map_display_name = String(data.get("name", active_map_filename))
	_custom_params_active = false
	var maps: Array[Dictionary] = list_available_tv_maps()
	for i in range(maps.size()):
		if String(maps[i]["path"]) == file_path:
			active_map_id = i
			break
	return true

## Saves current live TV parameters to a JSON file in tv_maps/ and refreshes the map list.
func save_current_tv_map_to_file(map_name: String, description: String = "", overwrite_path: String = "") -> String:
	var target_path := overwrite_path
	if target_path == "":
		var sanitized := _sanitize_map_filename(map_name)
		if sanitized == "":
			sanitized = "custom_tv_map"
		DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path(TV_MAPS_RES_DIR))
		target_path = TV_MAPS_RES_DIR + sanitized + ".json"
	var payload := {
		"name": map_name if map_name.strip_edges() != "" else target_path.get_file().get_basename(),
		"description": description,
		"author": "TeR27 Vehicle Dynamics",
		"control_law": "smc" if control_law_mode == ControlLawMode.V4_SMC_HYBRID else "v3",
		"event_mode": active_event_mode,
		"params": {
			"k_us": live_k_us,
			"max_mz": live_max_mz,
			"min_fx_reserve": live_min_fx_reserve,
			"tc_kp": live_tc_kp,
			"k_beta": live_k_beta,
			"throttle_mz_cut": live_throttle_mz_cut,
			"grip_cap": live_grip_cap,
			"smc_k_gain": live_smc_k_gain,
			"smc_k_ffd": live_smc_k_ffd,
			"smc_lambda_beta": live_smc_lambda_beta,
			"smc_phi": live_smc_phi,
			"smc_tc_kappa_base": live_smc_tc_kappa_base,
			"front_drive_bias": front_split_active,
			"can_latency_ms": can_latency_ms
		}
	}
	var f := FileAccess.open(target_path, FileAccess.WRITE)
	if f == null:
		return ""
	f.store_string(JSON.stringify(payload, "\t") + "\n")
	f.close()
	active_map_path = target_path
	active_map_filename = target_path.get_file().get_basename()
	active_map_display_name = String(payload["name"])
	_custom_params_active = false
	list_available_tv_maps()
	return target_path

## Deletes a TV map JSON file, protecting 00_open_4wd.json.
func delete_tv_map_file(file_path: String) -> bool:
	if file_path == "" or file_path.get_file() == TV_MAP_PROTECTED_FILE:
		return false
	var abs_path := ProjectSettings.globalize_path(file_path)
	var err := DirAccess.remove_absolute(abs_path)
	if err != OK:
		return false
	list_available_tv_maps()
	return true

func _sanitize_map_filename(raw_name: String) -> String:
	var s := raw_name.strip_edges().to_lower()
	var out := ""
	for i in range(s.length()):
		var ch := s[i]
		if (ch >= "a" and ch <= "z") or (ch >= "0" and ch <= "9"):
			out += ch
		elif ch == " " or ch == "-" or ch == "_":
			if not out.ends_with("_"):
				out += "_"
	return out.strip_edges().trim_prefix("_").trim_suffix("_")

## Enables or disables CAN bus latency + 4096 PPR encoder quantization simulation.
func set_hardware_realism(
	enabled: bool,
	latency_ms: float = DEFAULT_CAN_LATENCY_MS,
	ppr: int = DEFAULT_ENCODER_PPR
) -> void:
	hardware_realism_enabled = enabled
	if enabled:
		can_latency_ms = maxf(latency_ms, 0.0)
		encoder_ppr = maxi(ppr, 1)
		sensor_quantization_enabled = true
	else:
		can_latency_ms = 0.0
		encoder_ppr = maxi(ppr, 1)
		sensor_quantization_enabled = false
	_reset_ring_buffers()

## Requests switching between GDScript and native C/C++ GDExtension SIL backend.
## Falls back safely to GDScript when native `GpTorqueVectoringSIL` is not registered.
func set_use_native_sil(enabled: bool) -> bool:
	if sil_bridge == null:
		sil_bridge = TvSilBridgeScript.new(false)
	var target_mode: int = TvSilBridgeScript.IMPLEMENTATION_C_SIL if enabled else TvSilBridgeScript.IMPLEMENTATION_GDSCRIPT
	sil_bridge.set_implementation(target_mode)
	use_native_sil = sil_bridge.is_using_native_sil()
	return use_native_sil

func get_sil_status() -> Dictionary:
	if sil_bridge == null:
		sil_bridge = TvSilBridgeScript.new(false)
	return sil_bridge.sil_backend_status()

func _reset_ring_buffers() -> void:
	_sensor_ring_buffer.clear()
	_sensor_ring_buffer.resize(RING_BUFFER_CAPACITY)
	for i in range(RING_BUFFER_CAPACITY):
		_sensor_ring_buffer[i] = {}
	_sensor_rb_head = 0
	_sensor_rb_count = 0

	_cmd_ring_buffer.clear()
	_cmd_ring_buffer.resize(RING_BUFFER_CAPACITY)
	for i in range(RING_BUFFER_CAPACITY):
		_cmd_ring_buffer[i] = {}
	_cmd_rb_head = 0
	_cmd_rb_count = 0
	_sim_time_s = 0.0

func _push_sensor_sample(entry: Dictionary) -> void:
	if _sensor_ring_buffer.size() != RING_BUFFER_CAPACITY:
		_reset_ring_buffers()
	_sensor_ring_buffer[_sensor_rb_head] = entry
	_sensor_rb_head = (_sensor_rb_head + 1) % RING_BUFFER_CAPACITY
	_sensor_rb_count = mini(_sensor_rb_count + 1, RING_BUFFER_CAPACITY)

func _push_cmd_sample(t_s: float, cmd: Array[float]) -> void:
	if _cmd_ring_buffer.size() != RING_BUFFER_CAPACITY:
		_reset_ring_buffers()
	_cmd_ring_buffer[_cmd_rb_head] = {"t": t_s, "cmd": cmd.duplicate()}
	_cmd_rb_head = (_cmd_rb_head + 1) % RING_BUFFER_CAPACITY
	_cmd_rb_count = mini(_cmd_rb_count + 1, RING_BUFFER_CAPACITY)

static func _sample_ring_buffer(buf: Array[Dictionary], head: int, count: int, capacity: int, target_t: float) -> Dictionary:
	if count <= 0:
		return {}
	var oldest_idx: int = (head - count + capacity) % capacity
	for offset in range(count):
		var idx: int = (head - 1 - offset + capacity) % capacity
		var entry: Dictionary = buf[idx]
		if not entry.is_empty() and float(entry["t"]) <= target_t + 1e-9:
			return entry
	return buf[oldest_idx]

## Quantizes wheel angular speed using integer encoder quadrature counts over window `dt`.
static func quantize_wheel_omega(omega_rad_s: float, dt: float, ppr: int = DEFAULT_ENCODER_PPR) -> float:
	var window_dt: float = dt if dt > 1e-6 else FIXED_DT
	var ticks_per_rev: float = float(maxi(ppr, 1)) * 4.0
	var counts: float = roundf((omega_rad_s * window_dt) * (ticks_per_rev / TAU))
	return (counts * TAU) / (ticks_per_rev * window_dt)

static func _compute_motor_speed_scale(wheel_omega: float) -> float:
	var omega_max_wheel: float = (GP_MOTOR_MAX_RPM * (TAU / 60.0)) / GP_GEAR_RATIO
	var omega_taper_start: float = omega_max_wheel * GP_MOTOR_TAPER_RATIO
	var w_abs: float = absf(wheel_omega)
	if w_abs <= omega_taper_start:
		return 1.0
	if w_abs >= omega_max_wheel - 1e-6:
		return 0.0
	return clampf(1.0 - (w_abs - omega_taper_start) / maxf(omega_max_wheel - omega_taper_start, 1e-5), 0.0, 1.0)

# ── gp_math ──────────────────────────────────────────────────────────────────
static func gp_softplus(x: float) -> float:
	if x > 20.0:
		return x
	return log(1.0 + exp(x))

static func gp_sigmoid(x: float) -> float:
	var xc: float = clampf(x, -20.0, 20.0)
	return 1.0 / (1.0 + exp(-xc))

static func gp_softplus_sqrt(x: float) -> float:
	return sqrt(gp_softplus(x) + 1e-6)

static func gp_bilinear_interp_4x4(table: Array[float], x_norm: float, y_norm: float) -> float:
	var gx: float = clampf(x_norm * 3.0, 0.0, 2.99999)
	var gy: float = clampf(y_norm * 3.0, 0.0, 2.99999)
	var x0: int = int(gx)
	var y0: int = int(gy)
	var x1: int = clampi(x0 + 1, 0, 3)
	var y1: int = clampi(y0 + 1, 0, 3)
	var fx: float = gx - float(x0)
	var fy: float = gy - float(y0)
	var c00: float = table[x0 * 4 + y0]
	var c10: float = table[x1 * 4 + y0]
	var c01: float = table[x0 * 4 + y1]
	var c11: float = table[x1 * 4 + y1]
	return (
		c00 * (1.0 - fx) * (1.0 - fy)
		+ c10 * fx * (1.0 - fy)
		+ c01 * (1.0 - fx) * fy
		+ c11 * fx * fy
	)

# ── gp_ekf ───────────────────────────────────────────────────────────────────
func gp_ekf_predict(delta: float, ax_f: float, ay_f: float, wz_raw: float, vx: float, dt: float) -> void:
	var _unused = ax_f
	var _unused_d = delta
	var vx_safe: float = maxf(absf(vx), 0.5)
	ekf_wz_corrected = wz_raw - ekf_x[1]

	# Heun RK2 integration
	var vy_dot_k: float = ay_f - (vx_safe * ekf_wz_corrected)
	var _vy_pred: float = clampf(ekf_x[0] + vy_dot_k * dt, -6.0, 6.0)
	var vy_dot_corr: float = ay_f - (vx_safe * ekf_wz_corrected)
	var vy_dot_eff: float = 0.5 * (vy_dot_k + vy_dot_corr)

	ekf_x[0] = clampf(ekf_x[0] + vy_dot_eff * dt, -6.0, 6.0)

	var f01: float = dt * vx_safe
	var p00: float = float(ekf_p[0][0])
	var p01: float = float(ekf_p[0][1])
	var p10: float = float(ekf_p[1][0])
	var p11: float = float(ekf_p[1][1])

	ekf_p[0][0] = p00 + f01 * (p10 + p01) + (f01 * f01) * p11 + ekf_q[0]
	ekf_p[0][1] = p01 + f01 * p11
	ekf_p[1][0] = ekf_p[0][1]
	ekf_p[1][1] = p11 + ekf_q[1]

	for i in range(2):
		ekf_p[i][i] = clampf(float(ekf_p[i][i]), 1e-6, 2.0)

	ekf_beta_est = clampf(atan2(ekf_x[0], vx_safe), -0.523, 0.523)
	ekf_vy_std = sqrt(float(ekf_p[0][0]))

func _ekf_scalar_update(state_idx: int, z: float, r_var: float) -> void:
	var y: float = z - ekf_x[state_idx]
	var s_val: float = float(ekf_p[state_idx][state_idx]) + r_var
	if s_val < 1e-6:
		return
	var std_dev: float = sqrt(s_val)
	if absf(y) > 3.0 * std_dev:
		return

	var inv_s: float = 1.0 / s_val
	var k_gain: Array[float] = [
		float(ekf_p[0][state_idx]) * inv_s,
		float(ekf_p[1][state_idx]) * inv_s
	]
	for i in range(2):
		ekf_x[i] += k_gain[i] * y

	var p_tmp: Array[Array] = [[0.0, 0.0], [0.0, 0.0]]
	for i in range(2):
		for j in range(2):
			p_tmp[i][j] = float(ekf_p[i][j]) - k_gain[i] * float(ekf_p[state_idx][j])

	for i in range(2):
		for j in range(2):
			ekf_p[i][j] = 0.5 * (float(p_tmp[i][j]) + float(p_tmp[j][i]))
		ekf_p[i][i] = maxf(float(ekf_p[i][i]), 1e-6)

func gp_ekf_update_gps(vy_gps: float, gps_valid: bool) -> void:
	if not gps_valid:
		return
	_ekf_scalar_update(0, vy_gps, ekf_r_gps_vy)

func gp_ekf_update_kinematic_ss(ax_f: float, ay_f: float, wz_raw: float, vx: float) -> void:
	var vx_safe: float = maxf(absf(vx), 0.5)
	var wz_corr: float = wz_raw - ekf_x[1]
	var vy_ss: float = clampf((GP_LR * wz_corr) - ((GP_MASS * ay_f * GP_LF * vx_safe) / (GP_WB * GP_C_ALPHA_R)), -3.0, 3.0)
	last_vy_ss = vy_ss

	var a_total_g: float = sqrt(ax_f * ax_f + ay_f * ay_f) / GP_GRAVITY
	var excess_g: float = clampf(a_total_g - 0.85, 0.0, 2.5)
	var sat_penalty: float = 1.0 + 150.0 * (excess_g * excess_g)
	var r_eff: float = ekf_r_pseudo_vy * sat_penalty

	_ekf_scalar_update(0, vy_ss, r_eff)
	ekf_x[0] = clampf(ekf_x[0], -6.0, 6.0)
	ekf_beta_est = clampf(atan2(ekf_x[0], vx_safe), -0.523, 0.523)
	ekf_vy_std = sqrt(float(ekf_p[0][0]))

func gp_ekf_update_friction() -> void:
	ekf_x[1] = clampf(ekf_x[1], -0.10, 0.10)

# ── gp_vehicle_model (4WD TeR27) ────────────────────────────────────────────
static func gp_estimate_fz(vx: float, ax: float, ay: float) -> Array[float]:
	var total_weight: float = GP_MASS * GP_GRAVITY
	var inv_wb: float = 1.0 / (GP_WB + 1e-3)

	var fz_static_f: float = 0.5 * total_weight * (GP_LR * inv_wb)
	var fz_static_r: float = 0.5 * total_weight * (GP_LF * inv_wb)

	var dfz_lon_wheel: float = 0.5 * (GP_MASS * ax * GP_H_CG * inv_wb)
	var dfz_lat_f: float = 0.5 * (total_weight * (GP_LR * inv_wb) * (ay / GP_GRAVITY) * GP_H_CG) / GP_TRACK_F
	var dfz_lat_r: float = 0.5 * (total_weight * (GP_LF * inv_wb) * (ay / GP_GRAVITY) * GP_H_CG) / GP_TRACK_R

	var q_dyn: float = 0.25 * GP_AIR_DENSITY * (vx * vx) * GP_AERO_AREA
	var downforce_f: float = q_dyn * GP_AERO_CL_FRONT
	var downforce_r: float = q_dyn * GP_AERO_CL_REAR

	var fz_raw: Array[float] = [
		fz_static_f - dfz_lon_wheel - dfz_lat_f + downforce_f,
		fz_static_f - dfz_lon_wheel + dfz_lat_f + downforce_f,
		fz_static_r + dfz_lon_wheel - dfz_lat_r + downforce_r,
		fz_static_r + dfz_lon_wheel + dfz_lat_r + downforce_r
	]
	var fz_out: Array[float] = [0.0, 0.0, 0.0, 0.0]
	for i in range(4):
		fz_out[i] = 50.0 + gp_softplus(fz_raw[i] - 50.0)
	return fz_out

static func gp_estimate_fy(vx: float, vy: float, wz: float, delta: float, fz: Array[float]) -> Array[float]:
	var vx_safe: float = absf(vx) + 0.5
	var alpha_f: float = delta - atan2(vy + wz * GP_LF, vx_safe)
	var alpha_r: float = -atan2(vy - wz * GP_LR, vx_safe)
	var alpha_axle: Array[float] = [alpha_f, alpha_r]
	var c_alpha: Array[float] = [GP_C_ALPHA_F, GP_C_ALPHA_R]
	var wheel_indices: Array = [[GP_FL, GP_FR], [GP_RL, GP_RR]]

	var fy_out: Array[float] = [0.0, 0.0, 0.0, 0.0]
	for a in range(2):
		var alpha: float = alpha_axle[a]
		var sign_a: float = 1.0 if alpha >= 0.0 else -1.0
		var abs_alpha: float = absf(alpha)
		for w_idx in range(2):
			var idx: int = int(wheel_indices[a][w_idx])
			var fz_w: float = fz[idx]
			var mu_peak: float = clampf(GP_MU_NOM * (1.0 - GP_MU_DEG_FZ * (fz_w - 750.0)), 0.8, 2.1)
			var mu_slide: float = GP_MU_SLIDE * (mu_peak / GP_MU_NOM)
			var theta: float = (0.5 * c_alpha[a] * abs_alpha) / (3.0 * mu_peak * fz_w + 1e-3)
			var fy_w: float = 0.0
			if theta < 1.0:
				fy_w = sign_a * (3.0 * mu_peak * fz_w * theta * (1.0 - theta + (theta * theta) / 3.0))
			else:
				var decay: float = 1.0 / (1.0 + 0.5 * (theta - 1.0))
				var mu_eff: float = mu_slide + (mu_peak - mu_slide) * decay
				fy_w = sign_a * (mu_eff * fz_w)
			fy_out[idx] = fy_w
	return fy_out

static func gp_friction_ellipse_t_ub_param(
	fz: Array[float],
	fy_est: Array[float],
	mu_est: float,
	min_fx_reserve_ratio: float = 0.08
) -> Array[float]:
	var mu_safe: float = clampf(mu_est, 1.35, 2.20)
	var reserve_ratio: float = clampf(min_fx_reserve_ratio, 0.0, 0.95)
	var t_ub: Array[float] = [0.0, 0.0, 0.0, 0.0]
	for i in range(4):
		var max_cap: float = mu_safe * fz[i]
		var fx_reserve: float = reserve_ratio * max_cap
		var fx_sq_max: float = maxf((max_cap * max_cap) - (fy_est[i] * fy_est[i]), fx_reserve * fx_reserve)
		var fx_max: float = gp_softplus_sqrt(fx_sq_max * 4.0) * 0.5
		t_ub[i] = fx_max * GP_R_WHEEL
	return t_ub

static func gp_friction_ellipse_t_ub(fz: Array[float], fy_est: Array[float], mu_est: float) -> Array[float]:
	return gp_friction_ellipse_t_ub_param(fz, fy_est, mu_est, 0.08)

static func gp_power_limited_t_ub(omega_wheel: Array[float]) -> Array[float]:
	var t_ub: Array[float] = [0.0, 0.0, 0.0, 0.0]
	for i in range(4):
		var omega_safe: float = gp_softplus(omega_wheel[i])
		var t_power: float = GP_P_MAX_WHL / (omega_safe + 1e-3)
		var speed_scale: float = _compute_motor_speed_scale(omega_wheel[i])
		t_ub[i] = clampf(t_power, 0.0, GP_MAX_TRQ_ACT) * speed_scale
	return t_ub

static func gp_power_limited_t_lb(omega_wheel: Array[float], p_max_charge_w: float) -> Array[float]:
	var t_lb: Array[float] = [0.0, 0.0, 0.0, 0.0]
	for i in range(4):
		var omega_safe: float = gp_softplus(absf(omega_wheel[i]))
		var t_power: float = p_max_charge_w / (omega_safe + 1e-3)
		t_lb[i] = clampf(t_power, 0.0, GP_MAX_TRQ_ACT)
	return t_lb

static func gp_adaptive_k_us_param(fz: Array[float], k_us_base: float = GP_K_US) -> float:
	var fz_f: float = 0.5 * (fz[GP_FL] + fz[GP_FR])
	var fz_r: float = 0.5 * (fz[GP_RL] + fz[GP_RR])
	return k_us_base * (1.0 + GP_K_US_FZ * (fz_r - fz_f))

static func gp_adaptive_k_us(fz: Array[float]) -> float:
	return gp_adaptive_k_us_param(fz, GP_K_US)

static func gp_moment_arms() -> Array[float]:
	var hw_f: float = 0.5 * GP_TRACK_F / GP_R_WHEEL
	var hw_r: float = 0.5 * GP_TRACK_R / GP_R_WHEEL
	return [-hw_f, hw_f, -hw_r, hw_r]

# ── gp_solver (4WD) ──────────────────────────────────────────────────────────
static func gp_nominal_allocation_4wd(
	fx_driver: float,
	mz_target: float,
	fz_est: Array[float],
	front_split: float = -1.0
) -> Array[float]:
	var arms: Array[float] = gp_moment_arms()
	var t_fx_total: float = fx_driver * GP_R_WHEEL

	var fz_f_sum: float = maxf(fz_est[GP_FL], 1.0) + maxf(fz_est[GP_FR], 1.0)
	var fz_r_sum: float = maxf(fz_est[GP_RL], 1.0) + maxf(fz_est[GP_RR], 1.0)
	var fz_sum: float = maxf(fz_f_sum + fz_r_sum, 1e-3)

	var arm_sq_sum: float = 0.0
	for i in range(4):
		arm_sq_sum += arms[i] * arms[i]
	arm_sq_sum = maxf(arm_sq_sum, 1e-3)

	var use_axle_split: bool = front_split >= 0.0
	var sf: float = clampf(front_split, 0.0, 1.0) if use_axle_split else (fz_f_sum / fz_sum)
	var sr: float = 1.0 - sf

	var t_nom: Array[float] = [0.0, 0.0, 0.0, 0.0]
	for i in range(4):
		var t_fx_i: float = (0.5 * sf * t_fx_total) if i < 2 else (0.5 * sr * t_fx_total)
		var t_mz_i: float = (arms[i] * mz_target) / arm_sq_sum
		t_nom[i] = t_fx_i + t_mz_i
	return t_nom

## Exact closed-form KKT active-set water-filling solver for 4WD:
## min_{T} 0.5 * sum((T_i - T_blend_i)^2) s.t. sum(T_i) = T_target, T_lb_i <= T_i <= T_ub_i.
static func gp_qp_solve_4wd_closedform(
	t_warmstart: Array[float],
	t_prev: Array[float],
	fx_driver: float,
	t_lb: Array[float],
	t_ub: Array[float],
	w_reg_scale: float = 1.0,
	w_smooth_scale: float = 1.0
) -> Dictionary:
	var w_reg: float = GP_W_REG * maxf(w_reg_scale, 0.1)
	var w_smooth: float = GP_W_SMOOTH * maxf(w_smooth_scale, 0.1)
	var h: float = w_reg + w_smooth
	var a_eq: float = 1.0 / GP_R_WHEEL
	var t_blend: Array[float] = [0.0, 0.0, 0.0, 0.0]
	var t_total_min: float = 0.0
	var t_total_max: float = 0.0
	for i in range(4):
		t_blend[i] = (w_reg * t_warmstart[i] + w_smooth * t_prev[i]) / h
		t_total_min += t_lb[i]
		t_total_max += t_ub[i]

	var t_target: float = clampf(fx_driver * GP_R_WHEEL, t_total_min, t_total_max)
	var t_sol: Array[float] = [0.0, 0.0, 0.0, 0.0]
	var active: Array[bool] = [true, true, true, true]

	for _iter in range(4):
		var rem_target: float = t_target
		var free_sum: float = 0.0
		var n_free: int = 0
		for i in range(4):
			if active[i]:
				free_sum += t_blend[i]
				n_free += 1
			else:
				rem_target -= t_sol[i]
		if n_free == 0:
			break

		var nu: float = (rem_target - free_sum) / float(n_free)
		var worst_idx: int = -1
		var worst_viol: float = 0.0
		var worst_val: float = 0.0

		for i in range(4):
			if not active[i]:
				continue
			var cand: float = t_blend[i] + nu
			t_sol[i] = cand
			if cand > t_ub[i]:
				var v: float = cand - t_ub[i]
				if v > worst_viol:
					worst_viol = v
					worst_idx = i
					worst_val = t_ub[i]
			elif cand < t_lb[i]:
				var v: float = t_lb[i] - cand
				if v > worst_viol:
					worst_viol = v
					worst_idx = i
					worst_val = t_lb[i]

		if worst_idx == -1:
			break
		t_sol[worst_idx] = worst_val
		active[worst_idx] = false

	var t_sum: float = t_sol[0] + t_sol[1] + t_sol[2] + t_sol[3]
	var residual: float = absf(a_eq * t_sum - fx_driver)
	return {"torques": t_sol, "residual": residual}

# ── gp_traction_control (4WD RLS) ───────────────────────────────────────────
static func _tc_kappa_star(fz: float, mu_rt: float) -> float:
	var b_eff: float = maxf(GP_TC_B0 + GP_TC_B1 * fz, 2.0)
	var kappa_nom: float = tan(PI / (2.0 * GP_TC_C_PAC)) / b_eff
	var mu_corr: float = 1.0 + 0.15 * (1.5 - clampf(mu_rt, 0.5, 2.0)) / 1.5
	return clampf(kappa_nom * mu_corr, 0.04, 0.22)

static func _tc_combined_slip_factor(vx: float, vy: float, wz: float) -> float:
	var vx_safe: float = absf(vx) + 0.5
	var alpha_f: float = absf((vy + wz * GP_LF) / vx_safe)
	var alpha_r: float = absf((vy - wz * GP_LR) / vx_safe)
	var alpha_avg: float = 0.5 * (alpha_f + alpha_r)
	var ratio: float = clampf(alpha_avg / GP_TC_ALPHA_PEAK, 0.0, 0.95)
	return 1.0 / sqrt(1.0 + ratio * ratio)

static func _tc_compute_kappa(omega: float, vx_wheel: float) -> float:
	var denom: float = GP_TC_V_MIN + gp_softplus(vx_wheel - GP_TC_V_MIN)
	return clampf((omega * GP_R_WHEEL - vx_wheel) / denom, -0.80, 0.80)

func gp_tc_step_4wd(
	t_req_out: Array[float],
	omega: Array[float],
	vx: float,
	vy: float,
	wz: float,
	fz: Array[float],
	dt: float,
	wheel_steers: Array[float] = [0.0, 0.0, 0.0, 0.0],
	brake_norm: float = 0.0,
	tc_kp_override: float = -1.0
) -> void:
	var t_mean_abs: float = 0.25 * (
		absf(t_req_out[0]) + absf(t_req_out[1]) + absf(t_req_out[2]) + absf(t_req_out[3])
	)
	var omega_ema: Array[float] = [0.0, 0.0, 0.0, 0.0]
	var omega_dot: Array[float] = [0.0, 0.0, 0.0, 0.0]
	var fx_wheels: Array[float] = [0.0, 0.0, 0.0, 0.0]
	var mu_meas: Array[float] = [0.0, 0.0, 0.0, 0.0]

	for i in range(4):
		omega_ema[i] = 0.15 * omega[i] + 0.85 * tc_omega_prev_ema[i]
		var raw_dot: float = (omega_ema[i] - tc_omega_prev_ema[i]) / dt
		omega_dot[i] = clampf(raw_dot, -2000.0, 2000.0)
		fx_wheels[i] = (t_req_out[i] - GP_I_WHEEL_EST * omega_dot[i]) / GP_R_WHEEL
		mu_meas[i] = clampf(absf(fx_wheels[i]) / maxf(fz[i], 100.0), GP_TC_MU_LO, GP_TC_MU_HI)

	var gate_mu: float = gp_sigmoid(t_mean_abs - 20.0)
	var alpha_gated: float = GP_TC_ALPHA_MU_EMA * gate_mu
	var cs_factor: float = _tc_combined_slip_factor(vx, vy, wz)
	var speed_gate: float = gp_sigmoid((vx - GP_TC_V_MIN) * 3.0)
	var kp_gain: float = tc_kp_override if tc_kp_override >= 0.0 else tc_kp_active
	var kp_eff: float = (kp_gain * 20.0) * (1.0 + GP_TC_V_KP_SCALE / (absf(vx) + 0.5))
	var ki_eff: float = GP_TC_KI * 20.0

	var any_tc_cut: bool = false
	for i in range(4):
		var sign_i: float = -1.0 if t_req_out[i] < 0.0 else 1.0
		var mag_in: float = absf(t_req_out[i])

		# Project velocity onto steered wheel plane for front wheels so steering does not fake slip
		var vx_wheel: float = vx
		if i < 2:
			var delta_i: float = wheel_steers[i] if i < wheel_steers.size() else 0.0
			vx_wheel = vx * cos(delta_i) + (vy + wz * GP_LF) * sin(delta_i)

		var kappa_raw: float = _tc_compute_kappa(omega[i], vx_wheel)
		tc_kappa_filt[i] = GP_TC_ALPHA_KAPPA_LP * tc_kappa_filt[i] + (1.0 - GP_TC_ALPHA_KAPPA_LP) * kappa_raw

		# mu_surface: only evidence of real saturation (|kappa| > 0.08) may lower it
		var slipping: bool = absf(tc_kappa_filt[i]) > 0.08
		var mu_new: float = tc_mu_surface[i]
		if slipping:
			mu_new = (1.0 - alpha_gated) * mu_new + alpha_gated * mu_meas[i]
		else:
			if mu_meas[i] > mu_new:
				mu_new = (1.0 - alpha_gated) * mu_new + alpha_gated * mu_meas[i]
			mu_new += (GP_TC_MU_NOM - mu_new) * clampf(dt / GP_TC_MU_TAU_RECOVERY, 0.0, 1.0)
		tc_mu_surface[i] = clampf(mu_new, GP_TC_MU_LO, GP_TC_MU_HI)

		var traction_demand: bool = t_req_out[i] > 15.0 and brake_norm < 0.05

		var prev_k: float = tc_kappa_prev[i]
		var d_kappa: float = tc_kappa_filt[i] - prev_k
		var d_fx: float = fx_wheels[i] - tc_fx_prev[i]
		tc_kappa_prev[i] = tc_kappa_filt[i]
		tc_fx_prev[i] = fx_wheels[i]

		var max_phys_dk: float = 0.30 * dt / 0.005
		if traction_demand and absf(d_kappa) > 0.0001 and absf(d_kappa) < max_phys_dk and vx > 2.0:
			var lam_rls: float = 0.985
			var p_val: float = tc_rls_p[i]
			var phi: float = d_kappa
			var denom_rls: float = lam_rls + p_val * phi * phi
			var k_rls: float = (p_val * phi) / denom_rls
			var theta_hat: float = tc_rls_theta[i]
			var err_rls: float = d_fx - phi * theta_hat
			tc_rls_theta[i] = clampf(theta_hat + k_rls * err_rls, -50000.0, 150000.0)
			tc_rls_p[i] = clampf((p_val - k_rls * phi * p_val) / lam_rls, 10.0, 10000.0)

		var dtheta: float = tc_rls_theta[i] - tc_theta_prev[i]
		var dtheta_sign: float = 1.0 if dtheta >= 0.0 else -1.0
		var kappa_secant: float = (prev_k * tc_rls_theta[i] - tc_kappa_filt[i] * tc_theta_prev[i]) / (dtheta + dtheta_sign * 10.0)
		var secant_ok: float = gp_sigmoid(absf(dtheta) / 500.0 - 2.0)

		var theta_mag: float = absf(tc_rls_theta[i])
		var grad_dir: float = tc_rls_theta[i] / (theta_mag + 5000.0)
		var kappa_grad: float = tc_kappa_opt[i] + 0.03 * grad_dir * dt

		if traction_demand:
			tc_kappa_opt[i] = (
				secant_ok * clampf(kappa_secant, 0.05, 0.22)
				+ (1.0 - secant_ok) * clampf(kappa_grad, 0.05, 0.22)
			)
		tc_theta_prev[i] = tc_rls_theta[i]

		var kappa_analytical: float = _tc_kappa_star(fz[i], tc_mu_surface[i]) * cs_factor
		var kappa_target_mag: float = 0.5 * kappa_analytical + 0.5 * tc_kappa_opt[i]
		var kappa_meas_mag: float = sign_i * tc_kappa_filt[i]
		var error: float = kappa_meas_mag - kappa_target_mag

		if traction_demand:
			var raw_integral: float = tc_pi_integral[i] + error * dt * speed_gate
			tc_pi_integral[i] = GP_TC_I_MAX * tanh(raw_integral / GP_TC_I_MAX)
			if error < 0.0:
				tc_pi_integral[i] *= (1.0 - 0.08)
			else:
				tc_pi_integral[i] *= (1.0 - 0.002)
		else:
			tc_pi_integral[i] *= 0.5
			if absf(tc_pi_integral[i]) < 1e-6:
				tc_pi_integral[i] = 0.0

		var pi_out: float = kp_eff * error + ki_eff * tc_pi_integral[i]
		tc_omega_prev_ema[i] = omega_ema[i]

		var error_gate: float = gp_sigmoid(error * 40.0)
		var deriv_kick_pos: float = 12.0 * gp_softplus((omega_dot[i] - 250.0) * 0.05)
		var deriv_kick_neg: float = 12.0 * gp_softplus((-omega_dot[i] - 250.0) * 0.05)
		var deriv_kick: float = clampf(2.0 * (deriv_kick_pos if sign_i > 0.0 else deriv_kick_neg), 0.0, 40.0)
		pi_out += deriv_kick * error_gate

		var reduction: float = speed_gate * gp_softplus(pi_out * GP_TC_CLAMP_BETA) / GP_TC_CLAMP_BETA
		var mag_cmd: float = mag_in - reduction
		var mag_out: float = minf(gp_softplus(mag_cmd * GP_TC_CLAMP_BETA) / GP_TC_CLAMP_BETA, mag_in)
		if (mag_in - mag_out) > 1.0:
			any_tc_cut = true
		t_req_out[i] = sign_i * mag_out
	tc_active = any_tc_cut

## 2nd-order IIR biquad notch filter (default f0=25 Hz, Q=3.5) for clean steering feedforward.
func gp_biquad_notch_step(x: float, dt: float, f0: float = 25.0, q_val: float = 3.5) -> float:
	var dt_safe: float = maxf(dt, 1e-4)
	if f0 * dt_safe >= 0.49:
		_notch_x2 = _notch_x1
		_notch_x1 = x
		_notch_y2 = _notch_y1
		_notch_y1 = x
		return x
	var omega0: float = TAU * f0 * dt_safe
	var cos_w0: float = cos(omega0)
	var alpha: float = sin(omega0) / (2.0 * maxf(q_val, 0.1))
	var a0: float = 1.0 + alpha
	var b0: float = 1.0 / a0
	var b1: float = (-2.0 * cos_w0) / a0
	var b2: float = 1.0 / a0
	var a1: float = (-2.0 * cos_w0) / a0
	var a2: float = (1.0 - alpha) / a0
	var y: float = b0 * x + b1 * _notch_x1 + b2 * _notch_x2 - a1 * _notch_y1 - a2 * _notch_y2
	_notch_x2 = _notch_x1
	_notch_x1 = x
	_notch_y2 = _notch_y1
	_notch_y1 = y
	return y

## Deterministic, highly restrictive combined-ellipse Traction Control for V4_SMC_HYBRID.
func _gp_tc_step_4wd_smc(
	t_req_out: Array[float],
	omega: Array[float],
	vx: float,
	vy: float,
	wz: float,
	ay_mps2: float,
	dt: float,
	wheel_steers: Array[float] = [0.0, 0.0, 0.0, 0.0],
	brake_norm: float = 0.0
) -> void:
	var ay_g: float = ay_mps2 / GP_GRAVITY
	var ay_factor: float = 1.0 / (1.0 + 0.45 * ay_g * ay_g)
	var alpha_peak: float = deg_to_rad(8.0)
	var speed_gate: float = gp_sigmoid((vx - GP_TC_V_MIN) * 3.0)
	var kp_nm: float = 900.0
	var ki_nm: float = 1100.0

	var dt_before: float = (t_req_out[1] - t_req_out[0]) + (t_req_out[3] - t_req_out[2])
	var any_tc_cut: bool = false

	for i in range(4):
		var half_track: float = 0.5 * (GP_TRACK_F if i < 2 else GP_TRACK_R)
		var x_arm: float = GP_LF if i < 2 else -GP_LR
		var y_arm: float = half_track if (i == 0 or i == 2) else -half_track
		var vx_i: float = vx - wz * y_arm
		var vy_i: float = vy + wz * x_arm
		var delta_i: float = wheel_steers[i] if (i < 2 and i < wheel_steers.size()) else 0.0
		var alpha_i: float = atan2(vy_i, maxf(absf(vx_i), 1.0)) - delta_i
		var alpha_ratio: float = alpha_i / alpha_peak
		var alpha_factor: float = sqrt(maxf(0.10, 1.0 - alpha_ratio * alpha_ratio))

		var kappa_ellipse: float = live_smc_tc_kappa_base * ay_factor * alpha_factor
		var kappa_low_speed_floor: float = 0.65 / maxf(vx, 2.0)
		var kappa_lim_i: float = maxf(kappa_ellipse, kappa_low_speed_floor)
		last_tc_kappa_lim[i] = kappa_lim_i
		tc_kappa_opt[i] = kappa_lim_i

		var vx_wheel: float = vx
		if i < 2:
			vx_wheel = vx * cos(delta_i) + (vy + wz * GP_LF) * sin(delta_i)
		var kappa_raw: float = _tc_compute_kappa(omega[i], vx_wheel)
		tc_kappa_filt[i] = 0.45 * tc_kappa_filt[i] + 0.55 * kappa_raw
		tc_omega_prev_ema[i] = omega[i]

		var t_cmd_pos: float = maxf(t_req_out[i], 0.0)
		var traction_demand: bool = t_cmd_pos > 5.0 and brake_norm < 0.05
		var excess: float = maxf(0.0, absf(tc_kappa_filt[i]) - kappa_lim_i)
		if traction_demand and excess > 0.0:
			var max_cut: float = t_cmd_pos * 0.70
			tc_pi_integral[i] = clampf(tc_pi_integral[i] + ki_nm * excess * dt * speed_gate, 0.0, max_cut)
			var tc_cut: float = speed_gate * (kp_nm * excess + tc_pi_integral[i])
			tc_cut = clampf(tc_cut, 0.0, max_cut)
			if tc_cut > 1.0:
				any_tc_cut = true
			t_req_out[i] = maxf(0.0, t_cmd_pos - tc_cut)
		else:
			tc_pi_integral[i] *= 0.82
			if absf(tc_pi_integral[i]) < 0.05:
				tc_pi_integral[i] = 0.0

	# Preserve yaw moment direction/magnitude via soft_cap if TC cut creates an unwanted Mz spike
	if any_tc_cut:
		var dt_after: float = (t_req_out[1] - t_req_out[0]) + (t_req_out[3] - t_req_out[2])
		if absf(dt_after) > maxf(absf(dt_before), 20.0):
			var excess: float = absf(dt_after) - absf(dt_before)
			if dt_after > 0.0:
				var sum_r: float = maxf(t_req_out[1] + t_req_out[3], 1e-3)
				var sc_r: float = clampf((sum_r - excess) / sum_r, 0.0, 1.0)
				t_req_out[1] *= sc_r
				t_req_out[3] *= sc_r
			else:
				var sum_l: float = maxf(t_req_out[0] + t_req_out[2], 1e-3)
				var sc_l: float = clampf((sum_l - excess) / sum_l, 0.0, 1.0)
				t_req_out[0] *= sc_l
				t_req_out[2] *= sc_l
	tc_active = any_tc_cut

# ── gp_torque_vectoring (200 Hz 8-stage step) ───────────────────────────────
func step(
	fx_driver: float,
	delta: float,
	vx: float,
	vy: float,
	wz: float,
	ay: float,
	ax: float,
	omega: Array[float],
	brake_norm: float,
	dt: float,
	wheel_steers: Array[float] = [0.0, 0.0, 0.0, 0.0],
	vy_gps: float = 0.0,
	gps_valid: bool = false
) -> Array[float]:
	if dt <= 0.0:
		return last_t_cmd.duplicate()

	var use_realism: bool = (
		hardware_realism_enabled
		or can_latency_ms > 0.0
		or cmd_latency_ms > 0.0
		or sensor_quantization_enabled
	)

	if not use_realism:
		_time_acc += dt
		var steps_ideal: int = 0
		while _time_acc >= FIXED_DT and steps_ideal < 8:
			_time_acc -= FIXED_DT
			steps_ideal += 1
			_execute_tv_step(fx_driver, delta, vx, vy, wz, ay, ax, omega, brake_norm, FIXED_DT, wheel_steers, vy_gps, gps_valid)
		return last_t_cmd.duplicate()

	var omega_in: Array[float] = [0.0, 0.0, 0.0, 0.0]
	for i in range(4):
		var w_raw: float = omega[i] if i < omega.size() else 0.0
		omega_in[i] = quantize_wheel_omega(w_raw, dt, encoder_ppr) if sensor_quantization_enabled else w_raw

	if _sensor_rb_count == 0:
		var prefill_sample: Dictionary = {
			"t": 0.0,
			"fx_driver": 0.0,
			"delta": 0.0,
			"vx": vx,
			"vy": 0.0,
			"wz": 0.0,
			"ay": 0.0,
			"ax": 0.0,
			"omega": omega_in.duplicate(),
			"brake_norm": 0.0,
			"wheel_steers": [0.0, 0.0, 0.0, 0.0],
			"vy_gps": 0.0,
			"gps_valid": false
		}
		_push_sensor_sample(prefill_sample)
		_push_cmd_sample(0.0, last_t_cmd)

	_sim_time_s += dt
	var current_sample: Dictionary = {
		"t": _sim_time_s,
		"fx_driver": fx_driver,
		"delta": delta,
		"vx": vx,
		"vy": vy,
		"wz": wz,
		"ay": ay,
		"ax": ax,
		"omega": omega_in,
		"brake_norm": brake_norm,
		"wheel_steers": wheel_steers.duplicate(),
		"vy_gps": vy_gps,
		"gps_valid": gps_valid
	}
	_push_sensor_sample(current_sample)

	var sensor_delay_s: float = maxf(can_latency_ms, 0.0) * 0.001
	var target_sensor_t: float = _sim_time_s - sensor_delay_s
	var delayed: Dictionary = _sample_ring_buffer(
		_sensor_ring_buffer, _sensor_rb_head, _sensor_rb_count, RING_BUFFER_CAPACITY, target_sensor_t
	)

	var d_omega: Array[float] = [0.0, 0.0, 0.0, 0.0]
	var raw_d_omega: Array = delayed.get("omega", omega_in)
	for i in range(4):
		d_omega[i] = float(raw_d_omega[i]) if i < raw_d_omega.size() else 0.0

	var d_steers: Array[float] = [0.0, 0.0, 0.0, 0.0]
	var raw_d_steers: Array = delayed.get("wheel_steers", wheel_steers)
	for i in range(4):
		d_steers[i] = float(raw_d_steers[i]) if i < raw_d_steers.size() else 0.0

	_time_acc += dt
	var steps: int = 0
	while _time_acc >= FIXED_DT and steps < 8:
		_time_acc -= FIXED_DT
		steps += 1
		_execute_tv_step(
			float(delayed.get("fx_driver", 0.0)),
			float(delayed.get("delta", 0.0)),
			float(delayed.get("vx", vx)),
			float(delayed.get("vy", 0.0)),
			float(delayed.get("wz", 0.0)),
			float(delayed.get("ay", 0.0)),
			float(delayed.get("ax", 0.0)),
			d_omega,
			float(delayed.get("brake_norm", 0.0)),
			FIXED_DT,
			d_steers,
			float(delayed.get("vy_gps", 0.0)),
			bool(delayed.get("gps_valid", false))
		)

	_push_cmd_sample(_sim_time_s, last_t_cmd)
	var cmd_delay_s: float = maxf(cmd_latency_ms, 0.0) * 0.001
	var cmd_entry: Dictionary = _sample_ring_buffer(
		_cmd_ring_buffer, _cmd_rb_head, _cmd_rb_count, RING_BUFFER_CAPACITY, _sim_time_s - cmd_delay_s
	)
	var raw_cmd: Array = cmd_entry.get("cmd", last_t_cmd)
	if raw_cmd.size() >= 4:
		return [float(raw_cmd[0]), float(raw_cmd[1]), float(raw_cmd[2]), float(raw_cmd[3])]
	return last_t_cmd.duplicate()

func _execute_tv_step(
	fx_driver: float,
	delta: float,
	vx: float,
	vy_in: float,
	wz: float,
	ay: float,
	ax: float,
	omega: Array[float],
	brake_norm: float,
	dt: float,
	wheel_steers: Array[float],
	vy_gps: float,
	gps_valid: bool
) -> void:
	if use_native_sil and sil_bridge != null and sil_bridge.is_using_native_sil():
		var sil_trqs: Array[float] = sil_bridge.gp_tv_step(
			fx_driver, delta, vx, vy_in, wz, ay, ax, omega, brake_norm, dt, wheel_steers, vy_gps, gps_valid
		)
		for i in range(4):
			t_out_prev[i] = sil_trqs[i]
			last_t_cmd[i] = sil_trqs[i]
		return

	_gp_tv_step(fx_driver, delta, vx, vy_in, wz, ay, ax, omega, brake_norm, dt, wheel_steers, vy_gps, gps_valid)

func _update_launch_control_state(vx: float, fx_driver: float, brake_norm: float, dt: float) -> void:
	var max_fx_est: float = (4.0 * GP_MAX_TRQ_ACT) / GP_R_WHEEL
	var throttle_norm: float = clampf(fx_driver / maxf(max_fx_est, 1.0), 0.0, 1.0)
	match launch_state:
		LaunchState.LC_ARMED:
			_lc_blend_alpha = 1.0
			if absf(vx) > 2.0:
				launch_state = LaunchState.LC_OFF
				_lc_blend_alpha = 0.0
			elif brake_norm < 0.08 and throttle_norm > 0.70:
				launch_state = LaunchState.LC_ACTIVE
				_lc_blend_alpha = 1.0
				_lc_pi_integral = [0.0, 0.0, 0.0, 0.0]
		LaunchState.LC_ACTIVE:
			_lc_blend_alpha = 1.0
			if vx > LC_EXIT_VX_MS or throttle_norm < 0.40 or brake_norm > 0.15:
				launch_state = LaunchState.LC_DONE
				_lc_done_timer = 0.0
		LaunchState.LC_DONE:
			_lc_done_timer += dt
			_lc_blend_alpha = clampf(1.0 - (_lc_done_timer / LC_TRANSITION_TIME_S), 0.0, 1.0)
			if _lc_done_timer >= LC_TRANSITION_TIME_S:
				launch_state = LaunchState.LC_OFF
				_lc_blend_alpha = 0.0
		_:
			_lc_blend_alpha = 0.0

func _apply_launch_control_step(
	delta_dz: float,
	vx: float,
	wz_corr: float,
	ay_filt: float,
	omega: Array[float],
	t_ub: Array[float],
	dt: float
) -> Dictionary:
	var blend: float = _lc_blend_alpha
	var straight_lock: bool = absf(delta_dz) < 0.05
	var mz_lc: float = (-950.0 * wz_corr - 320.0 * ay_filt) if straight_lock else 0.0
	var lc_tc_active: bool = false
	var vx_pos: float = maxf(vx, 0.0)
	var dv_opt: float = maxf(LC_DV_OPT_MIN, LC_KAPPA_OPT * vx_pos)
	var v_target: float = vx_pos + dv_opt
	for i in range(4):
		var v_wheel: float = omega[i] * GP_R_WHEEL
		var dv_err: float = v_wheel - v_target
		if dv_err > 0.0:
			_lc_pi_integral[i] = clampf(_lc_pi_integral[i] + dv_err * dt, 0.0, 2.0)
			var cut_nm: float = 180.0 * dv_err + 320.0 * _lc_pi_integral[i]
			var t_cap: float = maxf(t_ub[i] - cut_nm * blend, 15.0)
			if t_cap < t_ub[i] - 0.5:
				lc_tc_active = true
			t_ub[i] = minf(t_ub[i], t_cap)
		else:
			_lc_pi_integral[i] *= 0.85
	return {
		"front_split": lerpf(front_split_active, LC_FRONT_SPLIT, blend),
		"straight_lock": straight_lock,
		"mz_lc": mz_lc,
		"blend": blend,
		"tc_active": lc_tc_active
	}

func _apply_endurance_efficiency_policy(
	fx_driver: float,
	t_lb: Array[float],
	omega: Array[float],
	t_cmd: Array[float],
	is_post_qp: bool
) -> Dictionary:
	if not is_post_qp:
		if fx_driver > 0.0:
			for i in range(4):
				t_lb[i] = maxf(t_lb[i], 0.0)
		return {"w_reg_scale": 2.2, "w_smooth_scale": 1.8}
	if live_eco_power_cap_kw > 0.0:
		var p_cap_w: float = live_eco_power_cap_kw * 1000.0
		var p_elec: float = 0.0
		for i in range(4):
			p_elec += maxf(0.0, t_cmd[i] * absf(omega[i])) / 0.90
		if p_elec > p_cap_w and p_elec > 1e-3:
			var p_scale: float = p_cap_w / p_elec
			for i in range(4):
				if t_cmd[i] > 0.0:
					t_cmd[i] *= p_scale
	return {}

func _apply_skidpad_radius_lock(
	fz_est: Array[float],
	delta_dot_est: float,
	ay_filt: float,
	wz_ref: float,
	wz_corr: float,
	fx_driver: float,
	mz_req: float,
	dt: float
) -> Dictionary:
	var steady_state: bool = absf(delta_dot_est) < 0.35 and absf(ay_filt) > 6.5
	if not steady_state:
		_skidpad_yaw_integral *= 0.90
		_skidpad_lock_active = false
		return {"active": false, "mz_req": mz_req}
	_skidpad_lock_active = true
	_skidpad_yaw_integral = clampf(_skidpad_yaw_integral + (wz_ref - wz_corr) * dt, -0.35, 0.35)
	var mz_adj: float = mz_req + 180.0 * _skidpad_yaw_integral
	var arms: Array[float] = gp_moment_arms()
	var arm_sq_sum: float = 0.0
	var fz_sum: float = 0.0
	for i in range(4):
		arm_sq_sum += arms[i] * arms[i]
		fz_sum += maxf(fz_est[i], 1.0)
	arm_sq_sum = maxf(arm_sq_sum, 1e-3)
	fz_sum = maxf(fz_sum, 1.0)
	var t_fx_total: float = fx_driver * GP_R_WHEEL
	var t_nom_fz: Array[float] = [0.0, 0.0, 0.0, 0.0]
	for i in range(4):
		var t_fx_i: float = t_fx_total * (maxf(fz_est[i], 1.0) / fz_sum)
		var t_mz_i: float = (arms[i] * mz_adj) / arm_sq_sum
		t_nom_fz[i] = t_fx_i + t_mz_i
	return {"active": true, "mz_req": mz_adj, "t_nominal": t_nom_fz}

func _gp_tv_step(
	fx_driver: float,
	delta: float,
	vx: float,
	_vy_in: float,
	wz: float,
	ay: float,
	ax: float,
	omega: Array[float],
	brake_norm: float,
	dt: float,
	wheel_steers: Array[float],
	vy_gps: float,
	gps_valid: bool
) -> void:
	_update_launch_control_state(vx, fx_driver, brake_norm, dt)
	var lc_engaged: bool = launch_state == LaunchState.LC_ACTIVE or (launch_state == LaunchState.LC_DONE and _lc_blend_alpha > 0.0)

	if launch_state == LaunchState.LC_ARMED:
		for i in range(4):
			last_t_cmd[i] = LC_PRELOAD_TORQUE_NM
			t_out_prev[i] = LC_PRELOAD_TORQUE_NM
			t_qp_prev[i] = LC_PRELOAD_TORQUE_NM
			last_tc_cut_nm[i] = 0.0
		tc_active = false
		last_wz_ref = 0.0
		last_mz_req = 0.0
		return

	# Stage 1: Standstill hold (bypassed when Launch Control is active)
	if vx < 1.0 and brake_norm > 0.5 and fx_driver > 500.0 and not lc_engaged:
		for i in range(4):
			last_t_cmd[i] = 7.5
			tc_pi_integral[i] = 0.0
			t_out_prev[i] = 7.5
			t_qp_prev[i] = 7.5
			t_lb_filt[i] = 0.0
		wz_int = 0.0
		last_wz_ref = 0.0
		last_mz_req = 0.0
		return

	# Stage 2: Deadzones & Acceleration LPF
	var delta_dz: float = 0.0 if absf(delta) < GP_STEER_DEADZONE_RAD else delta
	var wz_dz: float = 0.0 if absf(wz) < GP_YAW_DEADZONE_RADS else wz

	var alpha_lpf: float = clampf(dt / (GP_ACCEL_LPF_TAU + dt), 0.0, 1.0)
	ax_filt += alpha_lpf * (ax - ax_filt)
	ay_filt += alpha_lpf * (ay - ay_filt)

	var vx_safe: float = maxf(absf(vx), 0.5)

	# Stage 3: EKF 2-state [vy, bw]
	gp_ekf_predict(delta_dz, ax_filt, ay_filt, wz_dz, vx, dt)
	gp_ekf_update_gps(vy_gps, gps_valid)
	gp_ekf_update_kinematic_ss(ax_filt, ay_filt, wz_dz, vx)

	var vy_est: float = ekf_x[0]
	var wz_corr: float = ekf_wz_corrected
	var beta: float = ekf_beta_est

	# Stage 4: 4WD Tire Load & Lateral Force Estimation
	var fz_est: Array[float] = gp_estimate_fz(vx, ax_filt, ay_filt)
	var fy_est: Array[float] = gp_estimate_fy(vx, vy_est, wz_corr, delta_dz, fz_est)

	# Map 0 (OPEN_4WD): symmetric 35/65% axle split with Mz = 0, respecting power & max motor speed limits
	if active_map_id == TVMap.OPEN_4WD and not _custom_params_active:
		var k_us_open: float = gp_adaptive_k_us_param(fz_est, k_us_active)
		last_wz_ref = (vx_safe * delta_dz) / (GP_WB + k_us_open * vx_safe * vx_safe)
		last_mz_req = 0.0
		mz_sat_ratio = 1.0
		qp_residual = 0.0
		delta_prev = delta_dz

		var w_chassis_open: float = vx_safe / GP_R_WHEEL
		var req_total_trq: float = maxf(fx_driver * GP_R_WHEEL, 0.0)
		var sf: float = clampf(front_split_active, 0.0, 1.0)
		var sr: float = 1.0 - sf
		var t_open: Array[float] = [
			0.5 * sf * req_total_trq,
			0.5 * sf * req_total_trq,
			0.5 * sr * req_total_trq,
			0.5 * sr * req_total_trq
		]

		var p_total_mech: float = 0.0
		for i in range(4):
			var w_i: float = omega[i] if i < omega.size() else w_chassis_open
			var w_eff: float = maxf(absf(w_i), w_chassis_open)
			var t_pwr_i: float = GP_P_MAX_WHL / (w_eff + 1e-3)
			var t_cap_i: float = minf(GP_MAX_TRQ_ACT, t_pwr_i) * _compute_motor_speed_scale(w_eff)
			t_open[i] = clampf(t_open[i], 0.0, t_cap_i)
			p_total_mech += maxf(0.0, t_open[i] * w_eff)

		if p_total_mech > GP_P_MAX_TOTAL_MECH and p_total_mech > 1e-3:
			var p_scale: float = GP_P_MAX_TOTAL_MECH / p_total_mech
			for i in range(4):
				t_open[i] *= p_scale

		if open_4wd_tc_enabled:
			gp_tc_step_4wd(t_open, omega, vx, vy_est, wz_corr, fz_est, dt, wheel_steers, brake_norm, tc_kp_active)
		else:
			tc_active = false
		gp_ekf_update_friction()

		last_ff_mz = 0.0
		last_p_mz = 0.0
		last_i_mz = 0.0
		last_beta_mz = 0.0
		last_mz_raw = 0.0
		last_mz_req = 0.0
		last_kp = 0.0
		last_ki = 0.0
		last_kd = 0.0
		for i in range(4):
			last_fz_est[i] = fz_est[i]
			last_fy_est[i] = fy_est[i]
			last_t_nominal[i] = t_open[i]
			last_t_qp[i] = t_open[i]
			last_tc_cut_nm[i] = 0.0
			t_qp_prev[i] = t_open[i]
			t_out_prev[i] = t_open[i]
			last_t_cmd[i] = t_open[i]
		return

	# Stage 5: Physical Grip-Capped Yaw-Rate Reference & Moment Law (V3_PID_ALQP or V4_SMC_HYBRID)
	var mu_avg: float = 0.25 * (tc_mu_surface[0] + tc_mu_surface[1] + tc_mu_surface[2] + tc_mu_surface[3])
	var k_us: float = gp_adaptive_k_us_param(fz_est, k_us_active)
	var wz_kin: float = (vx_safe * delta_dz) / (GP_WB + k_us * vx_safe * vx_safe)
	var ay_cap: float = live_grip_cap * mu_avg * GP_GRAVITY
	var wz_max_grip: float = ay_cap / maxf(vx_safe, 2.5)
	var wz_ref: float = wz_max_grip * tanh(wz_kin / maxf(wz_max_grip, 0.05))
	last_wz_ref = wz_ref

	var wz_err: float = wz_ref - wz_corr
	var ff_mz: float = 0.0
	var p_mz: float = 0.0
	var i_mz: float = 0.0
	var beta_term: float = 0.0
	var mz_raw: float = 0.0

	if control_law_mode == ControlLawMode.V4_SMC_HYBRID:
		var delta_notch: float = gp_biquad_notch_step(delta_dz, dt, 25.0, 3.5)
		var delta_dot_notch: float = (delta_notch - _notch_delta_prev) / maxf(dt, 1e-4)
		_notch_delta_prev = delta_notch
		delta_prev = delta_dz

		var raw_wz_dot: float = (wz_corr - _smc_wz_prev) / maxf(dt, 1e-4)
		_smc_wz_prev = wz_corr
		_smc_wz_dot_filt += clampf(dt / (0.040 + dt), 0.0, 1.0) * (raw_wz_dot - _smc_wz_dot_filt)

		# Instantaneous IMU sideslip rate & velocity-growing beta shield
		var beta_dot_imu: float = (ay_filt / maxf(vx_safe, 3.0)) - wz_corr
		var lambda_beta_eff: float = live_smc_lambda_beta * (1.0 + 0.048 * vx_safe)

		# ISO/ROS (+Y left): left oversteer => vy < 0 => beta < 0 => needs Mz < 0, so +lambda_beta * beta brakes drift
		var beta_excess: float = maxf(0.0, absf(beta) - 0.022)
		var beta_emg: float = (1.0 if beta >= 0.0 else -1.0) * 1200.0 * beta_excess * beta_excess
		var s_instant: float = wz_err + lambda_beta_eff * beta + 0.45 * beta_dot_imu + beta_emg
		var lambda_int: float = 2.0
		if mz_sat_ratio > 0.85:
			smc_s_int = clampf(smc_s_int + s_instant * dt, -0.35, 0.35)
		else:
			smc_s_int *= 0.92
		var s_surface: float = s_instant + lambda_int * smc_s_int
		last_smc_s = s_surface

		var phi_eff: float = maxf(live_smc_phi, 0.02) * (1.0 + 0.03 * vx_safe)
		var hs_authority_scale: float = clampf(1.32 - 0.026 * vx_safe, 0.48, 1.0)
		var smc_mz: float = live_smc_k_gain * hs_authority_scale * tanh(s_surface / phi_eff)

		# Pivot feedforward: only in slow/medium corners (< 60 km/h) before 80% target yaw rate, zero if drifting
		var ff_allowed: bool = (vx_safe < FAST_CORNER_VX_MS) and (absf(wz_corr) < 0.80 * absf(wz_ref)) and (absf(beta) < 0.015)
		ff_mz = (live_smc_k_ffd * delta_dot_notch) if ff_allowed else 0.0

		var wz_damp_mz: float = -35.0 * _smc_wz_dot_filt * hs_authority_scale
		p_mz = live_smc_k_gain * hs_authority_scale * tanh((wz_err + lambda_int * smc_s_int) / phi_eff) + wz_damp_mz
		beta_term = smc_mz - (p_mz - wz_damp_mz)
		i_mz = 0.0
		mz_raw = smc_mz + ff_mz + wz_damp_mz
		last_kp = (live_smc_k_gain * hs_authority_scale) / phi_eff
		last_ki = lambda_int
		last_kd = live_smc_k_ffd if vx_safe < FAST_CORNER_VX_MS else 0.0
	else:
		last_smc_s = 0.0
		var v_norm: float = clampf(vx_safe / 30.0, 0.0, 1.0)
		var ay_norm: float = clampf(absf(ay_filt) / 15.0, 0.0, 1.0)
		var kp: float = gp_bilinear_interp_4x4(KP_MAP, v_norm, ay_norm)
		var ki: float = gp_bilinear_interp_4x4(KI_MAP, v_norm, ay_norm)
		var kd: float = gp_bilinear_interp_4x4(KD_MAP, v_norm, ay_norm)

		var mu_scale: float = clampf(mu_avg / GP_MU_NOM, 0.4, 1.0)
		kp *= mu_scale
		ki *= mu_scale
		last_kp = kp
		last_ki = ki
		last_kd = kd

		var delta_dot: float = (delta_dz - delta_prev) / dt
		delta_prev = delta_dz

		var vy_std_clamped: float = clampf(ekf_vy_std, 0.0, 0.3)
		var k_beta_dynamic: float = live_k_beta * (1.0 + 2.0 * vy_std_clamped)
		# ISO/ROS (+X fwd, +Y left): left oversteer => vy < 0 => beta < 0 => needs Mz < 0 (+k_beta * beta)
		beta_term = clampf(k_beta_dynamic * beta, -1200.0, 1200.0)

		var is_oversteering: bool = absf(wz_corr) > absf(wz_ref) + 0.03 and (wz_corr * wz_ref) > 0.0
		if is_oversteering:
			wz_int *= 0.90
		else:
			var raw_int: float = wz_int + wz_err * dt * mz_sat_ratio
			wz_int = GP_TV_WZ_I_MAX * tanh(raw_int / GP_TV_WZ_I_MAX)

		var os_gate: float = 1.0 - gp_sigmoid((absf(wz_corr) - absf(wz_ref) - 0.04) * 15.0)
		var counter_steer_factor: float = 1.0 - gp_sigmoid(-(delta_dz * wz_corr + 0.05) * 40.0)

		# Gate only pro-turn feedforward; keep stabilizing feedback active during oversteer/countersteer
		ff_mz = kd * delta_dot * (vx_safe / 10.0) * os_gate * counter_steer_factor
		p_mz = kp * wz_err
		i_mz = ki * wz_int
		var fb_mz: float = p_mz + i_mz + beta_term
		mz_raw = ff_mz + fb_mz

	last_ff_mz = ff_mz
	last_p_mz = p_mz
	last_i_mz = i_mz
	last_beta_mz = beta_term
	last_mz_raw = mz_raw

	var spin_sign: float = 1.0 if wz_corr > 1e-4 else (-1.0 if wz_corr < -1e-4 else 0.0)
	var stab_limit: float = maxf(tv_max_mz_active, 1000.0)
	var mz_req: float = 0.0
	if tv_max_mz_active <= 0.0:
		mz_req = 0.0
	elif mz_raw * spin_sign > 0.0:
		mz_req = clampf(mz_raw, -tv_max_mz_active, tv_max_mz_active)
	else:
		mz_req = clampf(mz_raw, -stab_limit, stab_limit)
	last_mz_req = mz_req

	# Stage 6: 4WD Friction Ellipse + Power Bounds (filtered tau = 30 ms)
	var w_chassis: float = vx_safe / GP_R_WHEEL
	var omega_power: Array[float] = [w_chassis, w_chassis, w_chassis, w_chassis]
	var t_ub_friction: Array[float] = gp_friction_ellipse_t_ub_param(fz_est, fy_est, mu_avg, min_fx_reserve_active)
	var t_ub_power: Array[float] = gp_power_limited_t_ub(omega_power)

	var alpha_ub: float = clampf(dt / (GP_UB_LPF_TAU + dt), 0.0, 1.0)
	var t_lb: Array[float] = [0.0, 0.0, 0.0, 0.0]
	var t_ub: Array[float] = [0.0, 0.0, 0.0, 0.0]
	var max_sum: float = 0.0
	var allow_regen_tv: bool = vx > 2.0 and brake_norm < 0.05
	for i in range(4):
		var raw_ub_i: float = minf(t_ub_friction[i], t_ub_power[i])
		if not _ub_filt_initialized:
			t_ub_filt[i] = raw_ub_i
		else:
			t_ub_filt[i] += alpha_ub * (raw_ub_i - t_ub_filt[i])
		t_ub[i] = t_ub_filt[i]
		var lb_raw: float = -minf(80.0, 0.25 * t_ub[i]) if (allow_regen_tv and fx_driver <= 50.0) else 0.0
		if control_law_mode == ControlLawMode.V4_SMC_HYBRID and i >= 2 and fx_driver <= 20.0 and brake_norm < 0.05:
			lb_raw = maxf(lb_raw, -25.0)
		t_lb_filt[i] = lb_raw
		t_lb[i] = lb_raw
		max_sum += t_ub[i]
	_ub_filt_initialized = true

	var fx_clamped: float = fx_driver
	var max_sum_power: float = GP_P_MAX_TOTAL_MECH / maxf(w_chassis, 1e-3)
	var eff_max_sum: float = minf(max_sum, max_sum_power)
	if delta_dz > 0.02 and mz_req < -25.0:
		var t_in_cap_l: float = t_ub[GP_FL] + t_ub[GP_RL]
		var max_sum_stab_l: float = maxf(2.0 * t_in_cap_l + (mz_req * GP_R_WHEEL) / (0.5 * GP_TRACK_F), 0.0)
		var blend_l: float = clampf((absf(mz_req) - 25.0) / 100.0, 0.0, 1.0)
		eff_max_sum = minf(eff_max_sum, lerpf(eff_max_sum, max_sum_stab_l, blend_l))
	elif delta_dz < -0.02 and mz_req > 25.0:
		var t_in_cap_r: float = t_ub[GP_FR] + t_ub[GP_RR]
		var max_sum_stab_r: float = maxf(2.0 * t_in_cap_r - (mz_req * GP_R_WHEEL) / (0.5 * GP_TRACK_F), 0.0)
		var blend_r: float = clampf((absf(mz_req) - 25.0) / 100.0, 0.0, 1.0)
		eff_max_sum = minf(eff_max_sum, lerpf(eff_max_sum, max_sum_stab_r, blend_r))
	var req_sum: float = fx_clamped * GP_R_WHEEL
	if req_sum > eff_max_sum:
		fx_clamped = eff_max_sum / GP_R_WHEEL
	if req_sum < 0.0:
		fx_clamped = 0.0

	# Corner-exit throttle Mz cut: attenuate pro-turn yaw moment under longitudinal acceleration
	if mz_req * delta_dz > 0.0:
		mz_req *= (1.0 - live_throttle_mz_cut * clampf(fx_clamped / 2500.0, 0.0, 1.0))

	var eff_front_split: float = front_split_active
	var w_reg_scale: float = 1.0
	var w_smooth_scale: float = 1.0
	var lc_res: Dictionary = {}
	if lc_engaged:
		lc_res = _apply_launch_control_step(delta_dz, vx, wz_corr, ay_filt, omega, t_ub, dt)
		eff_front_split = float(lc_res.get("front_split", LC_FRONT_SPLIT))
		if bool(lc_res.get("straight_lock", false)):
			var blend_lc: float = float(lc_res.get("blend", 1.0))
			mz_req = lerpf(mz_req, float(lc_res.get("mz_lc", 0.0)), blend_lc)
			last_wz_ref = lerpf(last_wz_ref, 0.0, blend_lc)
	elif active_event_mode == "endurance":
		var end_pre: Dictionary = _apply_endurance_efficiency_policy(fx_driver, t_lb, omega, [], false)
		w_reg_scale = float(end_pre.get("w_reg_scale", 1.0))
		w_smooth_scale = float(end_pre.get("w_smooth_scale", 1.0))
	last_mz_req = mz_req

	# Stage 7: 4WD Nominal Allocation + Closed-Form KKT QP Solver + Rate Limiter
	var t_nominal: Array[float] = gp_nominal_allocation_4wd(fx_clamped, mz_req, fz_est, eff_front_split)
	if not lc_engaged and active_event_mode == "skidpad":
		var delta_dot_skid: float = (_notch_delta_prev - delta_prev) / maxf(dt, 1e-4)
		var skid_res: Dictionary = _apply_skidpad_radius_lock(fz_est, delta_dot_skid, ay_filt, wz_ref, wz_corr, fx_clamped, mz_req, dt)
		if bool(skid_res.get("active", false)):
			t_nominal = skid_res["t_nominal"]
			last_mz_req = float(skid_res.get("mz_req", mz_req))
	var qp_res: Dictionary = gp_qp_solve_4wd_closedform(t_nominal, t_qp_prev, fx_clamped, t_lb, t_ub, w_reg_scale, w_smooth_scale)
	var qp_result: Array[float] = qp_res["torques"]
	qp_residual = float(qp_res["residual"])

	var dt_req: float = (t_nominal[GP_FR] - t_nominal[GP_FL]) + (t_nominal[GP_RR] - t_nominal[GP_RL])
	var dt_ach: float = (qp_result[GP_FR] - qp_result[GP_FL]) + (qp_result[GP_RR] - qp_result[GP_RL])
	if absf(dt_req) > 1.0:
		mz_sat_ratio = clampf(dt_ach / dt_req, 0.0, 1.0)
	else:
		mz_sat_ratio = 1.0

	var max_delta_t: float = GP_TV_RATE_LIMIT * dt
	var t_cmd: Array[float] = [0.0, 0.0, 0.0, 0.0]
	for i in range(4):
		last_fz_est[i] = fz_est[i]
		last_fy_est[i] = fy_est[i]
		last_t_nominal[i] = t_nominal[i]
		last_t_qp[i] = qp_result[i]
		var d_t: float = clampf(qp_result[i] - t_qp_prev[i], -max_delta_t, max_delta_t)
		var tv_final: float = t_qp_prev[i] + d_t
		t_qp_prev[i] = tv_final
		t_cmd[i] = tv_final

	var t_pre_tc_0: float = t_cmd[0]
	var t_pre_tc_1: float = t_cmd[1]
	var t_pre_tc_2: float = t_cmd[2]
	var t_pre_tc_3: float = t_cmd[3]

	# Stage 8: 4WD Traction Control (Launch Control 0 km/h PI, V4 combined-ellipse, or V3 RLS)
	if launch_state == LaunchState.LC_ACTIVE:
		var lc_tc_on: bool = false
		var vx_pos: float = maxf(vx, 0.0)
		var dv_opt: float = maxf(LC_DV_OPT_MIN, LC_KAPPA_OPT * vx_pos)
		var v_target: float = vx_pos + dv_opt
		for i in range(4):
			var v_wheel: float = omega[i] * GP_R_WHEEL
			var dv_err: float = v_wheel - v_target
			if dv_err > 0.0 and t_cmd[i] > 0.0:
				_lc_pi_integral[i] = clampf(_lc_pi_integral[i] + dv_err * dt, 0.0, 2.0)
				var cut_nm: float = 260.0 * dv_err + 480.0 * _lc_pi_integral[i]
				t_cmd[i] = maxf(t_cmd[i] - cut_nm, 15.0)
				lc_tc_on = true
			else:
				_lc_pi_integral[i] *= 0.82
		tc_active = lc_tc_on
	elif control_law_mode == ControlLawMode.V4_SMC_HYBRID:
		_gp_tc_step_4wd_smc(t_cmd, omega, vx, vy_est, wz_corr, ay_filt, dt, wheel_steers, brake_norm)
	else:
		gp_tc_step_4wd(t_cmd, omega, vx, vy_est, wz_corr, fz_est, dt, wheel_steers, brake_norm, tc_kp_active)
		gp_ekf_update_friction()

	if active_event_mode == "endurance" and not lc_engaged:
		_apply_endurance_efficiency_policy(fx_driver, t_lb, omega, t_cmd, true)

	last_tc_cut_nm[0] = maxf(0.0, t_pre_tc_0 - t_cmd[0])
	last_tc_cut_nm[1] = maxf(0.0, t_pre_tc_1 - t_cmd[1])
	last_tc_cut_nm[2] = maxf(0.0, t_pre_tc_2 - t_cmd[2])
	last_tc_cut_nm[3] = maxf(0.0, t_pre_tc_3 - t_cmd[3])

	for i in range(4):
		t_out_prev[i] = t_cmd[i]
		last_t_cmd[i] = t_cmd[i]
