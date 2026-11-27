class_name EcuTorqueVectoring4Wd
extends RefCounted

## controlador tv 4wd en cascada pa ter27 (observador ekf + lqr + qp fista + tc rls + modos)
## 100 % autocontenido en mirena_sim, sin asignaciones dinamicas en el lazo rapido

const VehicleConfig = preload("../../Config/VehicleConfig.gd")
const DEFAULT_CFG_PATH := "res://Scenes/Vehicle/Config/vehicles/ter27_4wd/tv_4wd_config.json"

enum TestMode {
	AUTOCROSS = 0,
	SKIDPAD = 1,
	EFFICIENCY = 2,
	ACCELERATION = 3,
	TIRE_WARMER = 4
}

# params fisicos y electricos (se cargan de tv_4wd_config.json o vehicleconfig)
var mass: float = 290.0
var izz: float = 83.1
var wb: float = 1.540
var a_lf: float = 0.847
var b_lr: float = 0.693
var tf: float = 1.230
var tr: float = 1.230
var r_eff: float = 0.2032
var h_cog: float = 0.290
var h_sprung_cog: float = 0.30986
var h_rc_f: float = 0.01959
var h_rc_r: float = 0.04003
var front_w_dist: float = 0.45
var anti_dive_f: float = 0.56
var anti_squat_r: float = 0.40
var lltd_f: float = 0.455
var cl_a: float = 6.0
var aero_bal_f: float = 0.48
var tau_wt: float = 0.035

var p_bms_max: float = 80000.0
var t_motor_peak: float = 25.0
var gear_ratio: float = 12.5
var motor_max_rpm: float = 18600.0
var base_speed_rpm: float = 9500.0
var motor_taper_ratio: float = 0.945
var eta_drive: float = 0.90
var eta_regen: float = 0.88
var t_regen_max: float = 120.0
# params inercia rueda, frenos y neuma vertical desde vehicleconfig
var i_wheel_eff: float = 0.5075
var brake_mu_cfg: float = 1.30
var brake_bias_f_cfg: float = 0.58
var p_max_charge_w: float = 25000.0
var mech_brake_start: float = 0.35
var motor_temps_c: Array[float] = [45.0, 45.0, 45.0, 45.0]
var thermal_derate_start_c: float = 85.0
var thermal_derate_max_c: float = 105.0
var tire_kz: float = 115000.0

# tablas lqr y qp
var vx_bp: Array[float] = [4.0, 10.0, 16.0, 24.0, 32.0]
var kr_tbl: Array[float] = [380.0, 620.0, 820.0, 980.0, 1100.0]
var kb_tbl: Array[float] = [120.0, 240.0, 380.0, 520.0, 640.0]
var kbd_tbl: Array[float] = [35.0, 65.0, 95.0, 125.0, 150.0]

var fista_iters: int = 12
var w_wheels: Array[float] = [1.0, 1.0, 1.0, 1.0]
var lambda_pitch: float = 0.08
var lambda_slew: float = 0.04
var lambda_slack: float = 50.0
var pitch_bias_gain: float = 18.0

# modo activo y config de modos
var active_mode: int = TestMode.AUTOCROSS
var modes_cfg: Dictionary = {}
var _empty_dict: Dictionary = {}

# cache de modo y coefs lpf por dt pa cero dict.get / exp() en step()
var _mc_mode: int = -1
var _mc_cold_floor: float = 0.82
var _mc_e_ref: float = 18000.0
var _mc_kus_rad_g: float = 0.35 * PI / (180.0 * 9.81)
var _mc_kus_low_rad_g: float = 0.35 * PI / (180.0 * 9.81)
var _mc_kus_vx_low: float = 11.0
var _mc_kus_vx_high: float = 20.0
var _mc_r_lim_scale: float = 0.90
var _mc_r_lim_low_scale: float = 0.90
var _mc_k_ff: float = 0.58
var _mc_k_ff_low_mult: float = 1.0
var _mc_rear_bias_gain: float = 0.0
var _mc_tb_turn_gain: float = 0.0
var _mc_p_max_low_w: float = -1.0
var _mc_p_max_high_w: float = 80000.0
var _mc_p_max_v_low: float = 14.0
var _mc_p_max_v_high: float = 23.0
var _mc_th_corner_exp: float = 1.0
var _mc_th_corner_delta: float = 0.22
var _mc_beta_soft_rad: float = 0.05585
var _mc_beta_hi_spd_rad: float = 0.04189
var _mc_beta_soft_gain: float = 0.0
var _mc_hairpin_regen_max_nm: float = 0.0
var _mc_hairpin_pivot_gain: float = 1.0
var _mc_yaw_agility_gain: float = 1.0
var _mc_launch_rear_bias_pct: float = 70.0
var _mc_slalom_damp_gain: float = 0.0
var _mc_rear_slide_gain: float = 0.0
var _mc_lift_regen_frac: float = 0.0
var _mc_lift_regen_v_min: float = 5.0
var _lift_regen_fx_filt: float = 0.0
var _mc_mz_scale: float = 1.0
var _mc_alpha_eq_gain: float = 480.0
var _mc_gyro_lock_gain: float = 420.0
var _mc_k_warmer_f: float = 0.140
var _mc_k_warmer_r: float = 0.275
var _mc_allow_regen: bool = true
var _mc_coast_gain: float = 0.0
var _mc_ti_gain: float = 280.0
var _mc_pitch_tau: float = 0.18
var _mc_regen_ay_th: float = 6.0
var _mc_kappa_lim: float = 0.115
var _mc_forbid_antag: bool = false
var _mc_forbid_antag_drv: bool = false
var _mc_part_th: float = 220.0
var _mc_p_eff_max: float = 56000.0
var _mc_skidpad_neutral_trim: float = 0.0
# SKIDPAD_MAX_ATTACK_V2 (active only in TestMode.SKIDPAD; all default neutral 0.0)
var _mc_sk_mz_ff_gain: float = 0.0      # Nm per (m/s2) steady-state understeer-neutralizing feedforward
var _mc_sk_beta_damp: float = 0.0       # Nm per (rad/s) beta_dot damping (dMz/dbeta < 0)
var _mc_sk_fz_shed_gain: float = 0.0    # 0..1 outer-front torque shedding at high friction utilization
var _mc_sk_cross_boost_nm: float = 0.0  # Nm Mz flip on figure-8 steering reversal
var _mc_sk_cross_regen_frac: float = 0.28 # 0..1 cap on crossover asymmetric regen to avoid over-braking
var _mc_sk_kd_yaw: float = 0.0          # Nm per (rad/s2) yaw-rate error derivative damping
enum LaunchState { IDLE = 0, ARMED = 1, LAUNCHING = 2, NORMAL = 3 }

var launch_state: int = LaunchState.IDLE
var _mc_friction_ellipse_margin: float = 0.94
var _mc_yaw_accel_ff_gain: float = 1.0
var _mc_delta_dot_lead_gain: float = 35.0
var _mc_exit_traction_blend: float = 0.75
# Winning surgical ablation parameters (Candidate A: Fx_f*sin(delta), Candidate B: Delta_alpha + curvature)
var _mc_centripetal_fx_gain: float = 20.0
var _mc_delta_alpha_eq_gain: float = 2600.0
var _qp_bfy_f: Array[float] = [0.0, 0.0, 0.0, 0.0]

var delta_norm_raw: float = 0.0
var _has_ext_delta_norm_raw: bool = false
var max_steer_rad: float = 0.45
var _was_moving_in_run: bool = false
var _was_braking_prev: bool = false
var _pre_brake_mu: Array[float] = [1.50, 1.50, 1.50, 1.50]
var _fz_unclamped: Array[float] = [640.0, 640.0, 780.0, 780.0]
var _airborne_ramp: Array[float] = [1.0, 1.0, 1.0, 1.0]

var _diag_active_regime: String = "BALANCED"
var _diag_mz_tv_nm: float = 0.0
var _diag_t_out_in_ratio: float = 1.0
var _diag_pivot_blend: float = 0.0
var _diag_balance_state: String = "NEUTRAL"
var _diag_mu_hat_avg: float = 1.50
var _diag_beta_deg: float = 0.0
var _diag_beta_dot_dps: float = 0.0

var _dt_cache: float = -1.0
var _a_bd_10: float = 0.0
var _a_d_8: float = 0.0
var _a_ff_20: float = 0.0
var _a_wt: float = 0.0
var _a_ay_12: float = 0.0
var _a_om_15: float = 0.0
var _a_rlim_45: float = 0.0
var _e_mz_45: float = 1.0
var _a_eff_150: float = 0.0

const _RAD_1_2: float = 1.2 * PI / 180.0
const _RAD_1_4: float = 1.4 * PI / 180.0
const _RAD_2_0: float = 2.0 * PI / 180.0
const _RAD_2_2: float = 2.2 * PI / 180.0
const _RAD_2_3: float = 2.3 * PI / 180.0
const _RAD_2_5: float = 2.5 * PI / 180.0
const _RAD_2_8: float = 2.8 * PI / 180.0
const _RAD_3_0: float = 3.0 * PI / 180.0
const _RAD_3_2: float = 3.2 * PI / 180.0
const _RAD_3_5: float = 3.5 * PI / 180.0
const _RAD_3_8: float = 3.8 * PI / 180.0
const _RAD_4_0: float = 4.0 * PI / 180.0
const _RAD_4_2: float = 4.2 * PI / 180.0
const _RAD_4_5: float = 4.5 * PI / 180.0
const _RAD_5_0: float = 5.0 * PI / 180.0
const _RAD_5_5: float = 5.5 * PI / 180.0
const _RAD_6_0: float = 6.0 * PI / 180.0
const _RAD_7_0: float = 7.0 * PI / 180.0
const _RAD_8_0: float = 8.0 * PI / 180.0
const _RAD_10_0: float = 10.0 * PI / 180.0

# estado capa 1: ekf 3 estados [vx, vy, bz] + covarianza diagonalizada
var _ekf_init: bool = false
var ekf_vx: float = 0.0
var ekf_vy: float = 0.0
var ekf_bz: float = 0.0
var _ekf_p: Array[float] = [0.2, 0.2, 0.01]
var _beta_prev: float = 0.0
var _beta_dot: float = 0.0
var _beta_dot_filt: float = 0.0

# buffers lpf y ff pa direccion
var _delta_lpf1: float = 0.0
var _delta_lpf2: float = 0.0
var _delta_ff1: float = 0.0
var _delta_ff2: float = 0.0
var _delta_filt_prev: float = 0.0
var _r_ref_prev: float = 0.0

# estimador fz/fy, freno hidraulico y radio cargado
var _ax_filt: float = 0.0
var _ax_long_filt: float = 0.0
var _ay_filt: float = 0.0
var fz_est: Array[float] = [640.0, 640.0, 780.0, 780.0]
var fy_est: Array[float] = [0.0, 0.0, 0.0, 0.0]
var fy_cons_est: Array[float] = [0.0, 0.0, 0.0, 0.0]
var r_load_est: Array[float] = [0.200, 0.200, 0.200, 0.200]
var t_hyd_est: Array[float] = [0.0, 0.0, 0.0, 0.0]
var _w_eff: Array[float] = [0.0, 0.0, 0.0, 0.0]
var _med_buf: Array[float] = [0.0, 0.0, 0.0, 0.0]

# capa 3: warm-start fista y pares previos (sin allocs)
var t_prev: Array[float] = [0.0, 0.0, 0.0, 0.0]
var _qp_x: Array[float] = [0.0, 0.0, 0.0, 0.0]
var _qp_y: Array[float] = [0.0, 0.0, 0.0, 0.0]
var _qp_z: Array[float] = [0.0, 0.0, 0.0, 0.0]
var _qp_lb: Array[float] = [0.0, 0.0, 0.0, 0.0]
var _qp_ub: Array[float] = [0.0, 0.0, 0.0, 0.0]
var _qp_bx: Array[float] = [0.0, 0.0, 0.0, 0.0]
var _qp_bmz: Array[float] = [0.0, 0.0, 0.0, 0.0]
var _qp_diag_w: Array[float] = [0.0, 0.0, 0.0, 0.0]
var _qp_tmp: Array[float] = [0.0, 0.0, 0.0, 0.0]
var _qp_v: Array[float] = [0.0, 0.0, 0.0, 0.0]
var _qp_t_abs: Array[float] = [0.0, 0.0, 0.0, 0.0]
var _qp_w_abs: Array[float] = [0.0, 0.0, 0.0, 0.0]

# capa 4: tc predictivo rls
var mu_init: float = 1.50
var mu_min: float = 0.55
var mu_max: float = 1.52
var rls_lambda: float = 0.985
var kappa_opt: float = 0.115
var mu_hat: Array[float] = [1.50, 1.50, 1.50, 1.50]
var rls_theta: Array[float] = [30000.0, 30000.0, 30000.0, 30000.0]
var rls_p: Array[float] = [1000.0, 1000.0, 1000.0, 1000.0]
var _omega_prev: Array[float] = [0.0, 0.0, 0.0, 0.0]
var _om_dot_filt: Array[float] = [0.0, 0.0, 0.0, 0.0]
var _kappa_prev: Array[float] = [0.0, 0.0, 0.0, 0.0]
var _kappa_rel: Array[float] = [0.0, 0.0, 0.0, 0.0]
var _ki_buf: Array[float] = [0.0, 0.0, 0.0, 0.0]
var _split_spin_cnt: Array[int] = [0, 0, 0, 0]
var _fx_est_prev: Array[float] = [0.0, 0.0, 0.0, 0.0]

# estados tacticos y guardias de seguridad
var slip_energy_j: float = 0.0
var _accel_elapsed_s: float = 0.0
var _launch_timer_s: float = 0.0
var _live_overrides: Dictionary = {}
var _curb_timer_s: float = 0.0
var _curb_wheel_timer: Array[float] = [0.0, 0.0, 0.0, 0.0]
var _az_prev: float = 9.81
var _ay_prev: float = 0.0
var _alpha_f_rel: float = 0.0
var _alpha_r_rel: float = 0.0
var _r_lim_filt: float = 0.0
var _warmer_timer_s: float = 0.0
var _warmer_phase_t: float = 0.0
var _warmer_state: int = 0
var _warmer_sgn: float = 1.0
var _warmer_v_int: float = 0.0
var _warmer_k_int_f: float = 0.0
var _warmer_k_int_r: float = 0.0
var _warmer_trq_des: Array[float] = [0.0, 0.0, 0.0, 0.0]
var _warmer_cap_des: Array[float] = [0.0, 0.0, 0.0, 0.0]
var _eff_front_disc: bool = false
var _eff_disc_timer_s: float = 0.0
var _eff_t_filt: float = 0.0
var _turnin_timer_s: float = 0.0
var _wz_abs_prev: float = 0.0
var _rls_theta_prev: Array[float] = [30000.0, 30000.0, 30000.0, 30000.0]
var _fault_auto: Array[bool] = [false, false, false, false]
var _fault_cnt: Array[int] = [0, 0, 0, 0]
var _fault_rec_cnt: Array[int] = [0, 0, 0, 0]
var limp_home_active: bool = false
var fault_wheel_mask: Array[bool] = [false, false, false, false]
var fault_steer_sensor: bool = false

# salidas de telemetria y diagnostico
var last_t_cmd: Array[float] = [0.0, 0.0, 0.0, 0.0]
var last_wz_ref: float = 0.0
var last_mz_req: float = 0.0
var last_mz_ff: float = 0.0
var last_mz_fb: float = 0.0
var last_pitch_bias_nm: float = 0.0
var last_power_w: float = 0.0
var mz_sat_ratio: float = 0.0
var tc_active: bool = false
var _mz_int: float = 0.0
var _r_err_prev: float = 0.0
var _r_err_dot_filt: float = 0.0
var last_mz_crossover: float = 0.0
var last_mz_beta_damp: float = 0.0
var last_torque_fight_kw: float = 0.0
var last_is_crossover: bool = false


func _init() -> void:
	load_config_json(DEFAULT_CFG_PATH)


func load_config_json(path: String = DEFAULT_CFG_PATH) -> bool:
	if not FileAccess.file_exists(path):
		return false
	var f := FileAccess.open(path, FileAccess.READ)
	if f == null:
		return false
	var parsed = JSON.parse_string(f.get_as_text())
	f.close()
	if not (parsed is Dictionary):
		return false
	var d: Dictionary = parsed
	if d.has("physical") and d["physical"] is Dictionary:
		var ph: Dictionary = d["physical"]
		mass = float(ph.get("mass_kg", mass))
		izz = float(ph.get("izz_kgm2", izz))
		wb = float(ph.get("wheelbase_m", wb))
		tf = float(ph.get("track_front_m", tf))
		tr = float(ph.get("track_rear_m", tr))
		r_eff = float(ph.get("wheel_radius_eff_m", r_eff))
		h_cog = float(ph.get("h_cog_m", h_cog))
		h_sprung_cog = float(ph.get("h_sprung_cog_m", h_sprung_cog))
		h_rc_f = float(ph.get("h_rc_front_m", h_rc_f))
		h_rc_r = float(ph.get("h_rc_rear_m", h_rc_r))
		front_w_dist = float(ph.get("front_weight_dist", front_w_dist))
		anti_dive_f = float(ph.get("anti_dive_front", anti_dive_f))
		anti_squat_r = float(ph.get("anti_squat_rear", anti_squat_r))
		lltd_f = float(ph.get("lltd_front", lltd_f))
		cl_a = float(ph.get("cl_a_m2", cl_a))
		aero_bal_f = float(ph.get("aero_balance_front", aero_bal_f))
		tau_wt = float(ph.get("tau_wt_s", tau_wt))
		a_lf = wb * (1.0 - front_w_dist)
		b_lr = wb * front_w_dist
	if d.has("electrical") and d["electrical"] is Dictionary:
		var el: Dictionary = d["electrical"]
		p_bms_max = float(el.get("p_bms_max_w", p_bms_max))
		t_motor_peak = float(el.get("t_motor_peak_nm", t_motor_peak))
		gear_ratio = float(el.get("gear_ratio", gear_ratio))
		motor_max_rpm = float(el.get("motor_max_rpm", motor_max_rpm))
		base_speed_rpm = float(el.get("base_speed_rpm", base_speed_rpm))
		eta_drive = float(el.get("eta_drive", eta_drive))
		eta_regen = float(el.get("eta_regen", eta_regen))
		t_regen_max = float(el.get("t_regen_max_wheel_nm", t_regen_max))
		p_max_charge_w = maxf(float(el.get("p_max_charge_w", p_max_charge_w)), 1000.0)
		mech_brake_start = clampf(float(el.get("mech_brake_start", mech_brake_start)), 0.0, 0.95)
		thermal_derate_start_c = float(el.get("thermal_derate_start_c", thermal_derate_start_c))
		thermal_derate_max_c = maxf(float(el.get("thermal_derate_max_c", thermal_derate_max_c)), thermal_derate_start_c + 5.0)
	if d.has("observer") and d["observer"] is Dictionary:
		var ob: Dictionary = d["observer"]
		mu_init = float(ob.get("mu_init", mu_init))
		mu_min = float(ob.get("mu_min", mu_min))
		# techo fisico realista de mu_hat pa evitar sobreestimacion
		mu_max = minf(float(ob.get("mu_max", 1.52)), 1.55)
		rls_lambda = float(ob.get("rls_lambda", rls_lambda))
		kappa_opt = float(ob.get("kappa_opt_init", kappa_opt))
		if not _ekf_init:
			for i in range(4):
				mu_hat[i] = mu_init
	if d.has("lqr_gain_schedule") and d["lqr_gain_schedule"] is Dictionary:
		var lq: Dictionary = d["lqr_gain_schedule"]
		_copy_float_arr(lq.get("vx_breakpoints_ms", []), vx_bp)
		_copy_float_arr(lq.get("k_r", []), kr_tbl)
		_copy_float_arr(lq.get("k_beta", []), kb_tbl)
		_copy_float_arr(lq.get("k_beta_dot", []), kbd_tbl)
	if d.has("qp_allocation") and d["qp_allocation"] is Dictionary:
		var qp: Dictionary = d["qp_allocation"]
		fista_iters = clampi(int(qp.get("fista_iters", fista_iters)), 8, 24)
		_copy_float_arr(qp.get("w_wheels", []), w_wheels)
		lambda_pitch = float(qp.get("lambda_pitch", lambda_pitch))
		lambda_slew = float(qp.get("lambda_slew", lambda_slew))
		lambda_slack = float(qp.get("lambda_slack", lambda_slack))
		pitch_bias_gain = float(qp.get("pitch_bias_gain_nm_per_ms2", pitch_bias_gain))
	if d.has("modes") and d["modes"] is Dictionary:
		modes_cfg = d["modes"]
	_mc_mode = -1
	_dt_cache = -1.0
	return true


static func _copy_float_arr(src: Variant, dst: Array[float]) -> void:
	if src is Array and not src.is_empty():
		dst.clear()
		for v in src:
			dst.append(float(v))


func configure_from_vehicle_config(cfg: VehicleConfig) -> void:
	if cfg == null:
		return
	mass = maxf(cfg.mass, 100.0)
	izz = maxf(cfg.get_total_izz(), 20.0)
	wb = maxf(cfg.wheelbase, 0.5)
	tf = maxf(cfg.track_width_f, 0.5)
	tr = maxf(cfg.track_width_r, 0.5)
	r_eff = maxf(cfg.wheel_radius, 0.1)
	h_cog = cfg.h_cog
	h_sprung_cog = cfg.h_sprung_cog
	h_rc_f = cfg.roll_center_height_front
	h_rc_r = cfg.roll_center_height_rear
	front_w_dist = cfg.front_weight_distribution
	a_lf = cfg.a_cog
	b_lr = cfg.b_cog
	anti_dive_f = clampf(cfg.anti_dive_ratio_front, 0.0, 1.0)
	anti_squat_r = clampf(cfg.anti_squat_ratio_rear, 0.0, 1.0)
	cl_a = cfg.cl_a
	aero_bal_f = cfg.aero_balance_front
	p_bms_max = maxf(cfg.max_dc_power_w, 1000.0)
	t_motor_peak = cfg.motor_peak_torque
	gear_ratio = cfg.gear_ratio
	motor_max_rpm = cfg.motor_max_rpm
	base_speed_rpm = cfg.base_speed_rpm
	motor_taper_ratio = clampf(cfg.motor_taper_ratio, 0.5, 0.999)
	eta_drive = maxf(cfg.drivetrain_efficiency, 0.5)
	i_wheel_eff = cfg.get_effective_wheel_inertia()
	brake_mu_cfg = cfg.brake_mu
	brake_bias_f_cfg = cfg.brake_bias_front
	p_max_charge_w = maxf(cfg.p_max_charge_w, 1000.0)
	tire_kz = maxf(cfg.tire_vertical_stiffness, 20000.0)
	var kw_f: float = cfg.spring_stiffness_front * cfg.motion_ratio_front * cfg.motion_ratio_front
	var kw_r: float = cfg.spring_stiffness_rear * cfg.motion_ratio_rear * cfg.motion_ratio_rear
	var k_phi_f: float = 0.5 * kw_f * tf * tf + cfg.arb_stiffness_front
	var k_phi_r: float = 0.5 * kw_r * tr * tr + cfg.arb_stiffness_rear
	lltd_f = clampf(k_phi_f / maxf(k_phi_f + k_phi_r, 1.0), 0.15, 0.85)
	max_steer_rad = maxf(cfg.max_steer_angle, 0.1)
	_mc_mode = -1
	_dt_cache = -1.0
	if not _ekf_init:
		for i in range(4):
			fz_est[i] = 0.5 * mass * 9.81 * (front_w_dist if i < 2 else (1.0 - front_w_dist))
			r_load_est[i] = clampf(r_eff - fz_est[i] / tire_kz, 0.175, r_eff)


func reset() -> void:
	_ekf_init = false
	_mc_mode = -1
	_dt_cache = -1.0
	ekf_vx = 0.0
	ekf_vy = 0.0
	ekf_bz = 0.0
	_ekf_p[0] = 0.2
	_ekf_p[1] = 0.2
	_ekf_p[2] = 0.01
	_beta_prev = 0.0
	_beta_dot = 0.0
	_beta_dot_filt = 0.0
	_delta_lpf1 = 0.0
	_delta_lpf2 = 0.0
	_delta_ff1 = 0.0
	_delta_ff2 = 0.0
	_delta_filt_prev = 0.0
	_r_ref_prev = 0.0
	_ax_filt = 0.0
	_ax_long_filt = 0.0
	_ay_filt = 0.0
	_mz_int = 0.0
	_r_err_prev = 0.0
	_r_err_dot_filt = 0.0
	last_mz_crossover = 0.0
	last_mz_beta_damp = 0.0
	last_torque_fight_kw = 0.0
	last_is_crossover = false
	mz_sat_ratio = 0.0
	slip_energy_j = 0.0
	_accel_elapsed_s = 0.0
	_launch_timer_s = 0.0
	launch_state = LaunchState.IDLE
	_curb_timer_s = 0.0
	_az_prev = 9.81
	_ay_prev = 0.0
	_alpha_f_rel = 0.0
	_alpha_r_rel = 0.0
	_r_lim_filt = 0.0
	_warmer_timer_s = 0.0
	_warmer_phase_t = 0.0
	_warmer_state = 0
	_warmer_sgn = 1.0
	_warmer_v_int = 0.0
	_warmer_k_int_f = 0.0
	_warmer_k_int_r = 0.0
	_eff_front_disc = false
	_eff_disc_timer_s = 0.0
	_eff_t_filt = 0.0
	_turnin_timer_s = 0.0
	_wz_abs_prev = 0.0
	_was_moving_in_run = false
	_was_braking_prev = false
	delta_norm_raw = 0.0
	_has_ext_delta_norm_raw = false
	_diag_active_regime = "BALANCED"
	_diag_mz_tv_nm = 0.0
	_diag_t_out_in_ratio = 1.0
	_diag_pivot_blend = 0.0
	_diag_balance_state = "NEUTRAL"
	_diag_mu_hat_avg = mu_init
	_diag_beta_deg = 0.0
	_diag_beta_dot_dps = 0.0
	limp_home_active = false
	tc_active = false
	var kz_safe := maxf(tire_kz, 20000.0)
	for i in range(4):
		motor_temps_c[i] = 45.0
		_warmer_trq_des[i] = 0.0
		_warmer_cap_des[i] = 0.0
		_ki_buf[i] = 0.0
		_split_spin_cnt[i] = 0
		_curb_wheel_timer[i] = 0.0
		_rls_theta_prev[i] = 30000.0
		_pre_brake_mu[i] = mu_init
		_airborne_ramp[i] = 1.0
		_fault_auto[i] = false
		_fault_cnt[i] = 0
		_fault_rec_cnt[i] = 0
		fault_wheel_mask[i] = false
		t_prev[i] = 0.0
		_qp_x[i] = 0.0
		_qp_y[i] = 0.0
		_qp_z[i] = 0.0
		_qp_lb[i] = 0.0
		_qp_ub[i] = 0.0
		_qp_bx[i] = 0.0
		_qp_bmz[i] = 0.0
		_qp_diag_w[i] = 0.0
		_qp_tmp[i] = 0.0
		_qp_v[i] = 0.0
		_qp_t_abs[i] = 0.0
		_qp_w_abs[i] = 0.0
		last_t_cmd[i] = 0.0
		mu_hat[i] = mu_init
		rls_theta[i] = 30000.0
		rls_p[i] = 1000.0
		_omega_prev[i] = 0.0
		_om_dot_filt[i] = 0.0
		_kappa_prev[i] = 0.0
		_kappa_rel[i] = 0.0
		_fx_est_prev[i] = 0.0
		fy_est[i] = 0.0
		fy_cons_est[i] = 0.0
		t_hyd_est[i] = 0.0
		_w_eff[i] = 0.0
		_med_buf[i] = 0.0
		fz_est[i] = 0.5 * mass * 9.81 * (front_w_dist if i < 2 else (1.0 - front_w_dist))
		r_load_est[i] = clampf(r_eff - fz_est[i] / kz_safe, 0.175, r_eff)
	_lift_regen_fx_filt = 0.0


func set_motor_temperatures(t_fl: float, t_fr: float, t_rl: float, t_rr: float) -> void:
	motor_temps_c[0] = t_fl
	motor_temps_c[1] = t_fr
	motor_temps_c[2] = t_rl
	motor_temps_c[3] = t_rr


# mediana o(1) in-place sobre buffer fijo de hasta 4 elems (cero malloc)
static func _median_buf(buf: Array[float], n: int) -> float:
	if n <= 0:
		return 0.0
	if n == 1:
		return buf[0]
	if n == 2:
		return 0.5 * (buf[0] + buf[1])
	if buf[0] > buf[1]:
		var t := buf[0]; buf[0] = buf[1]; buf[1] = t
	if n == 3:
		if buf[1] > buf[2]:
			var t := buf[1]; buf[1] = buf[2]; buf[2] = t
		if buf[0] > buf[1]:
			var t := buf[0]; buf[0] = buf[1]; buf[1] = t
		return buf[1]
	if buf[2] > buf[3]:
		var t := buf[2]; buf[2] = buf[3]; buf[3] = t
	if buf[0] > buf[2]:
		var t := buf[0]; buf[0] = buf[2]; buf[2] = t
	if buf[1] > buf[3]:
		var t := buf[1]; buf[1] = buf[3]; buf[3] = t
	if buf[1] > buf[2]:
		var t := buf[1]; buf[1] = buf[2]; buf[2] = t
	return 0.5 * (buf[1] + buf[2])


# fuerza lateral hoosier sensible a carga (mu_y/mu_x ~0.89 a 300n -> 0.73 a 2100n; calibrado contra mf62)
static func _calc_hoosier_fy(alpha_rad: float, mu_i: float, fz_i: float) -> float:
	var load_u := clampf((fz_i - 300.0) / 1800.0, 0.0, 1.0)
	var mu_y_ratio := lerpf(0.89, 0.73, load_u)
	var c_norm := lerpf(32.5, 13.8, load_u)
	var fy_max_i := mu_y_ratio * mu_i * fz_i
	return fy_max_i * tanh((c_norm * fz_i * alpha_rad) / maxf(fy_max_i, 100.0))


static func _interp_1d(x: float, xs: Array[float], ys: Array[float]) -> float:
	var n: int = mini(xs.size(), ys.size())
	if n == 0:
		return 0.0
	if x <= xs[0]:
		return ys[0]
	if x >= xs[n - 1]:
		return ys[n - 1]
	for i in range(n - 1):
		if x >= xs[i] and x <= xs[i + 1]:
			var u: float = (x - xs[i]) / maxf(xs[i + 1] - xs[i], 1e-6)
			return lerpf(ys[i], ys[i + 1], u)
	return ys[n - 1]


# proyeccion exacta sobre caja [lb, ub] e hiperplano bmz*x = mz_tgt (o(1), cero malloc)
static func _project_box_eq(
	v: Array[float],
	lb: Array[float],
	ub: Array[float],
	bmz: Array[float],
	mz_tgt: float,
	out: Array[float]
) -> void:
	var mz_min := 0.0
	var mz_max := 0.0
	var nu_lo := 1e12
	var nu_hi := -1e12
	var has_act := false
	for i in range(4):
		var bi := bmz[i]
		if bi >= 0.0:
			mz_min += bi * lb[i]
			mz_max += bi * ub[i]
		else:
			mz_min += bi * ub[i]
			mz_max += bi * lb[i]
		if absf(bi) > 1e-6:
			has_act = true
			var n1 := (lb[i] - v[i]) / bi
			var n2 := (ub[i] - v[i]) / bi
			nu_lo = minf(nu_lo, minf(n1, n2))
			nu_hi = maxf(nu_hi, maxf(n1, n2))
	var tgt := clampf(mz_tgt, mz_min, mz_max)
	if not has_act or (mz_max - mz_min) <= 1e-7:
		for i in range(4):
			if bmz[i] > 1e-6:
				out[i] = lb[i]
			elif bmz[i] < -1e-6:
				out[i] = ub[i]
			else:
				out[i] = clampf(v[i], lb[i], ub[i])
		return
	if tgt <= mz_min + 1e-6:
		for i in range(4):
			out[i] = lb[i] if bmz[i] > 1e-6 else (ub[i] if bmz[i] < -1e-6 else clampf(v[i], lb[i], ub[i]))
		return
	if tgt >= mz_max - 1e-6:
		for i in range(4):
			out[i] = ub[i] if bmz[i] > 1e-6 else (lb[i] if bmz[i] < -1e-6 else clampf(v[i], lb[i], ub[i]))
		return
	nu_lo -= 1e-4
	nu_hi += 1e-4
	var nu := 0.0 if (nu_lo < 0.0 and 0.0 < nu_hi) else 0.5 * (nu_lo + nu_hi)
	for _it in range(14):
		var g := -tgt
		var dg := 0.0
		for i in range(4):
			var bi := bmz[i]
			var xi_unc := v[i] + nu * bi
			if xi_unc <= lb[i]:
				g += bi * lb[i]
			elif xi_unc >= ub[i]:
				g += bi * ub[i]
			else:
				g += bi * xi_unc
				dg += bi * bi
		if absf(g) <= 1e-4:
			break
		if g < 0.0:
			nu_lo = nu
		else:
			nu_hi = nu
		var nu_new := 0.5 * (nu_lo + nu_hi)
		if dg > 1e-9:
			var nu_nt := nu - g / dg
			if nu_nt > nu_lo and nu_nt < nu_hi:
				nu_new = nu_nt
		else:
			# si todas saturadas en nu, salta a la siguiente esquina activa pa no estancar
			if g < 0.0:
				var c1 := nu_hi
				var c2 := nu_hi
				for i in range(4):
					var bi := bmz[i]
					if absf(bi) > 1e-6:
						var na := (lb[i] - v[i]) / bi
						var nb := (ub[i] - v[i]) / bi
						if na > nu + 1e-12 and na < c1:
							c2 = c1; c1 = na
						elif na > c1 + 1e-12 and na < c2:
							c2 = na
						if nb > nu + 1e-12 and nb < c1:
							c2 = c1; c1 = nb
						elif nb > c1 + 1e-12 and nb < c2:
							c2 = nb
				if c1 < nu_hi:
					if c1 > nu + 1e-6:
						nu_lo = c1
					nu_new = 0.5 * (c1 + c2)
			else:
				var c1 := nu_lo
				var c2 := nu_lo
				for i in range(4):
					var bi := bmz[i]
					if absf(bi) > 1e-6:
						var na := (lb[i] - v[i]) / bi
						var nb := (ub[i] - v[i]) / bi
						if na < nu - 1e-12 and na > c1:
							c2 = c1; c1 = na
						elif na < c1 - 1e-12 and na > c2:
							c2 = na
						if nb < nu - 1e-12 and nb > c1:
							c2 = c1; c1 = nb
						elif nb < c1 - 1e-12 and nb > c2:
							c2 = nb
				if c1 > nu_lo:
					if c1 < nu - 1e-6:
						nu_hi = c1
					nu_new = 0.5 * (c1 + c2)
		nu = nu_new
	for i in range(4):
		out[i] = clampf(v[i] + nu * bmz[i], lb[i], ub[i])


static func _calc_punch_mz_bounds(
	lb: Array[float],
	ub: Array[float],
	bx: Array[float],
	bmz: Array[float],
	fx_drop_max: float,
	mz_at_ub: float
) -> Vector2:
	# Exact greedy knapsack over the 4 wheels for [mz_lo, mz_hi] subject to Fx >= Fx(ub) - fx_drop_max
	var hi_gain := mz_at_ub
	var rem_hi := fx_drop_max
	var used_hi := 0
	for _pass in range(4):
		var best_i := -1
		var best_r := -1.0
		for i in range(4):
			if (used_hi & (1 << i)) != 0:
				continue
			if bmz[i] < -1e-6 and bx[i] > 1e-4 and ub[i] > lb[i]:
				var r := (-bmz[i]) / bx[i]
				if r > best_r:
					best_r = r
					best_i = i
		if best_i < 0 or rem_hi <= 1e-5:
			break
		used_hi |= (1 << best_i)
		var dt_max := ub[best_i] - lb[best_i]
		var dt_take := minf(dt_max, rem_hi / bx[best_i])
		hi_gain += (-bmz[best_i]) * dt_take
		rem_hi -= bx[best_i] * dt_take

	var lo_gain := mz_at_ub
	var rem_lo := fx_drop_max
	var used_lo := 0
	for _pass in range(4):
		var best_i := -1
		var best_r := -1.0
		for i in range(4):
			if (used_lo & (1 << i)) != 0:
				continue
			if bmz[i] > 1e-6 and bx[i] > 1e-4 and ub[i] > lb[i]:
				var r := bmz[i] / bx[i]
				if r > best_r:
					best_r = r
					best_i = i
		if best_i < 0 or rem_lo <= 1e-5:
			break
		used_lo |= (1 << best_i)
		var dt_max := ub[best_i] - lb[best_i]
		var dt_take := minf(dt_max, rem_lo / bx[best_i])
		lo_gain -= bmz[best_i] * dt_take
		rem_lo -= bx[best_i] * dt_take

	return Vector2(lo_gain, hi_gain)


# pot electrica dc neta (w) usando eta_drive/eta_regen sin malloc
static func _calc_p_elec(
	trqs: Array[float],
	w_omegas: Array[float],
	w_eff: Array[float],
	eta_drv: float,
	eta_reg: float
) -> float:
	var p := 0.0
	var ed := maxf(eta_drv, 0.1)
	for i in range(4):
		var wi := w_eff[i] if is_nan(w_omegas[i]) else w_omegas[i]
		var pm := trqs[i] * wi
		p += (pm / ed) if pm >= 0.0 else (pm * eta_reg)
	return p


# evalua pot dc tras restar gamma * w+ y proyectar conservando mz_tgt (cero malloc)
static func _eval_p_gamma(
	x_base: Array[float],
	gamma: float,
	w_omegas: Array[float],
	w_eff: Array[float],
	lb: Array[float],
	ub: Array[float],
	bmz: Array[float],
	mz_tgt: float,
	eta_drv: float,
	eta_reg: float,
	v_buf: Array[float],
	out_buf: Array[float]
) -> float:
	for i in range(4):
		var wi := maxf(0.0 if is_nan(w_omegas[i]) else w_omegas[i], 0.0)
		v_buf[i] = x_base[i] - gamma * wi
	_project_box_eq(v_buf, lb, ub, bmz, mz_tgt, out_buf)
	return _calc_p_elec(out_buf, w_omegas, w_eff, eta_drv, eta_reg)


func _get_mode_dict() -> Dictionary:
	var key := "autocross"
	match active_mode:
		TestMode.SKIDPAD: key = "skidpad"
		TestMode.EFFICIENCY: key = "efficiency"
		TestMode.ACCELERATION: key = "acceleration"
		TestMode.TIRE_WARMER: key = "tire_warmer"
	var v = modes_cfg.get(key, _empty_dict)
	return v if v is Dictionary else _empty_dict


# cachea config del modo activo pa evitar dict.get en el lazo rapido
func _refresh_mode_cache(m_cfg: Dictionary) -> void:
	_mc_mode = active_mode
	_mc_cold_floor = float(m_cfg.get("cold_tire_mu_floor", 0.82))
	_mc_e_ref = maxf(float(m_cfg.get("thermal_slip_energy_ref_j", 18000.0)), 1000.0)
	var kus_hi_deg: float = float(m_cfg.get("k_us_high_deg_g", m_cfg.get("k_us_deg_g", 0.35)))
	_mc_kus_rad_g = deg_to_rad(kus_hi_deg) / 9.81
	_mc_kus_low_rad_g = deg_to_rad(float(m_cfg.get("k_us_low_deg_g", kus_hi_deg))) / 9.81
	_mc_kus_vx_low = float(m_cfg.get("k_us_vx_low_ms", 11.0))
	_mc_kus_vx_high = maxf(float(m_cfg.get("k_us_vx_high_ms", 20.0)), _mc_kus_vx_low + 0.5)
	_mc_r_lim_scale = float(m_cfg.get("r_lim_scale", 0.90))
	_mc_r_lim_low_scale = float(m_cfg.get("r_lim_low_scale", _mc_r_lim_scale))
	_mc_k_ff = float(m_cfg.get("k_ff", 0.58))
	_mc_k_ff_low_mult = float(m_cfg.get("k_ff_low_mult", 1.0))
	_mc_rear_bias_gain = float(m_cfg.get("dynamic_rear_bias_gain", 0.0))
	_mc_tb_turn_gain = clampf(float(m_cfg.get("trail_brake_yaw_gain", m_cfg.get("trail_brake_pro_turn_gain", 0.0))), 0.0, 2.5)
	_mc_skidpad_neutral_trim = clampf(float(m_cfg.get("skidpad_neutral_trim", 0.0)), -25.0, 25.0)
	_mc_delta_dot_lead_gain = clampf(float(m_cfg.get("delta_dot_lead_gain", 0.0)), 0.0, 120.0)
	_mc_exit_traction_blend = clampf(float(m_cfg.get("exit_traction_blend", 0.0)), 0.0, 1.0)
	_mc_centripetal_fx_gain = clampf(float(m_cfg.get("centripetal_fx_gain", 0.0)), 0.0, 50.0)
	_mc_delta_alpha_eq_gain = clampf(float(m_cfg.get("delta_alpha_eq_gain", 0.0)), 0.0, 4000.0)
	_mc_sk_mz_ff_gain = clampf(float(m_cfg.get("skidpad_mz_ff_gain", 0.0)), 0.0, 60.0)
	_mc_sk_beta_damp = clampf(float(m_cfg.get("skidpad_beta_damp", 0.0)), 0.0, 2000.0)
	_mc_sk_fz_shed_gain = clampf(float(m_cfg.get("skidpad_fz_shed_gain", 0.0)), 0.0, 1.0)
	_mc_sk_cross_boost_nm = clampf(float(m_cfg.get("skidpad_cross_boost_nm", 0.0)), 0.0, 500.0)
	_mc_sk_cross_regen_frac = clampf(float(m_cfg.get("skidpad_cross_regen_frac", 0.28)), 0.0, 1.0)
	_mc_sk_kd_yaw = clampf(float(m_cfg.get("skidpad_kd_yaw", 0.0)), 0.0, 120.0)
	var p_lo_kw: float = float(m_cfg.get("p_max_low_kw", -1.0))
	_mc_p_max_low_w = p_lo_kw * 1000.0 if p_lo_kw > 0.0 else -1.0
	_mc_p_max_high_w = float(m_cfg.get("p_max_high_kw", p_lo_kw if p_lo_kw > 0.0 else 80.0)) * 1000.0
	_mc_p_max_v_low = float(m_cfg.get("p_max_v_low_mps", 14.0))
	_mc_p_max_v_high = maxf(float(m_cfg.get("p_max_v_high_mps", 23.0)), _mc_p_max_v_low + 0.5)
	_mc_th_corner_exp = maxf(float(m_cfg.get("throttle_corner_exp", 1.0)), 1.0)
	_mc_th_corner_delta = maxf(float(m_cfg.get("throttle_corner_delta_rad", 0.22)), 0.05)
	_mc_beta_soft_rad = deg_to_rad(clampf(float(m_cfg.get("beta_soft_deg", 3.2)), 1.0, 10.0))
	_mc_beta_hi_spd_rad = deg_to_rad(clampf(float(m_cfg.get("beta_soft_high_spd_deg", m_cfg.get("beta_soft_deg", 2.4))), 1.0, 6.0))
	_mc_beta_soft_gain = maxf(float(m_cfg.get("beta_soft_damp_gain", 0.0)), 0.0)
	_mc_hairpin_regen_max_nm = clampf(float(m_cfg.get("hairpin_regen_max_nm", 0.0)), 0.0, 120.0)
	_mc_hairpin_pivot_gain = clampf(float(m_cfg.get("hairpin_pivot_gain", 1.0)), 0.5, 2.0)
	_mc_yaw_agility_gain = clampf(float(m_cfg.get("yaw_agility_gain", 1.0)), 0.5, 2.0)
	_mc_friction_ellipse_margin = clampf(float(m_cfg.get("friction_ellipse_margin", 0.94)), 0.70, 1.05)
	_mc_yaw_accel_ff_gain = clampf(float(m_cfg.get("yaw_accel_ff_gain", 1.0)), 0.30, 2.00)
	_mc_launch_rear_bias_pct = clampf(float(m_cfg.get("launch_rear_bias_pct", 70.0)), 50.0, 80.0)
	_mc_slalom_damp_gain = maxf(float(m_cfg.get("slalom_yaw_damp_gain", 0.0)), 0.0)
	_mc_rear_slide_gain = maxf(float(m_cfg.get("rear_slide_damp_gain", 0.0)), 0.0)
	_mc_lift_regen_frac = clampf(float(m_cfg.get("lift_off_regen_frac", 0.0)), 0.0, 0.50)
	_mc_lift_regen_v_min = maxf(float(m_cfg.get("lift_off_regen_v_min_mps", 5.0)), 1.5)
	_mc_mz_scale = float(m_cfg.get("mz_scale", 1.0))
	_mc_alpha_eq_gain = float(m_cfg.get("alpha_eq_gain", 480.0))
	_mc_gyro_lock_gain = float(m_cfg.get("gyro_lock_gain", 420.0))
	_mc_k_warmer_f = float(m_cfg.get("target_kappa_warmer", 0.140))
	_mc_k_warmer_r = float(m_cfg.get("target_kappa_warmer_rear", 0.275))
	_mc_allow_regen = bool(m_cfg.get("allow_regen", true))
	_mc_coast_gain = float(m_cfg.get("coast_regen_gain_nm_per_ms", 0.0))
	_mc_ti_gain = float(m_cfg.get("turnin_pitch_gain_nm", 280.0))
	_mc_pitch_tau = maxf(float(m_cfg.get("pitch_ramp_tau_s", 0.18)), 0.05)
	_mc_regen_ay_th = maxf(float(m_cfg.get("regen_ay_atten_ms2", 6.0)), 1.0)
	_mc_kappa_lim = float(m_cfg.get("kappa_opt", kappa_opt))
	_mc_forbid_antag = bool(m_cfg.get("forbid_antagonistic", false))
	_mc_forbid_antag_drv = bool(m_cfg.get("forbid_antagonistic_drive", false))
	_mc_part_th = float(m_cfg.get("part_load_axle_disconnect_nm", 220.0))
	_mc_p_eff_max = float(m_cfg.get("p_eff_max_w", 56000.0))
	if not _live_overrides.is_empty():
		_apply_live_overrides_internal(_live_overrides)


func _apply_live_overrides_internal(params: Dictionary) -> void:
	if params.has("hairpin_pivot_gain"):
		_mc_hairpin_pivot_gain = clampf(float(params["hairpin_pivot_gain"]), 0.5, 2.0)
	if params.has("inner_regen_max_nm"):
		_mc_hairpin_regen_max_nm = clampf(float(params["inner_regen_max_nm"]), 0.0, 120.0)
	if params.has("yaw_agility_gain"):
		_mc_yaw_agility_gain = clampf(float(params["yaw_agility_gain"]), 0.5, 2.0)
	if params.has("slalom_beta_damp"):
		_mc_slalom_damp_gain = clampf(float(params["slalom_beta_damp"]), 0.0, 1500.0)
	if params.has("beta_soft_deg"):
		_mc_beta_soft_rad = deg_to_rad(clampf(float(params["beta_soft_deg"]), 1.0, 10.0))
	if params.has("launch_rear_bias_pct"):
		_mc_launch_rear_bias_pct = clampf(float(params["launch_rear_bias_pct"]), 50.0, 80.0)
	if params.has("tc_slip_target"):
		_mc_kappa_lim = clampf(float(params["tc_slip_target"]), 0.04, 0.25)
	if params.has("power_limit_kw"):
		var p_kw := clampf(float(params["power_limit_kw"]), 10.0, 80.0)
		p_bms_max = p_kw * 1000.0
		if _mc_p_max_low_w > 0.0:
			_mc_p_max_low_w = minf(_mc_p_max_low_w, p_bms_max)
		_mc_p_max_high_w = minf(_mc_p_max_high_w, p_bms_max)
	if params.has("regen_limit_kw"):
		var r_raw := float(params["regen_limit_kw"])
		var r_kw := clampf(r_raw if r_raw <= 0.0 else -r_raw, -25.0, 0.0)
		p_max_charge_w = absf(r_kw) * 1000.0
	if params.has("throttle_exp"):
		_mc_th_corner_exp = clampf(float(params["throttle_exp"]), 0.5, 2.5)
	if params.has("trail_brake_yaw_gain"):
		_mc_tb_turn_gain = clampf(float(params["trail_brake_yaw_gain"]), 0.0, 2.5)
	if params.has("skidpad_neutral_trim"):
		_mc_skidpad_neutral_trim = clampf(float(params["skidpad_neutral_trim"]), -25.0, 25.0)
	if params.has("delta_dot_lead_gain"):
		_mc_delta_dot_lead_gain = clampf(float(params["delta_dot_lead_gain"]), 0.0, 120.0)
	if params.has("exit_traction_blend"):
		_mc_exit_traction_blend = clampf(float(params["exit_traction_blend"]), 0.0, 1.0)
	if params.has("friction_ellipse_margin"):
		_mc_friction_ellipse_margin = clampf(float(params["friction_ellipse_margin"]), 0.70, 1.05)
	if params.has("yaw_accel_ff_gain"):
		_mc_yaw_accel_ff_gain = clampf(float(params["yaw_accel_ff_gain"]), 0.30, 2.00)
	if params.has("centripetal_fx_gain"):
		_mc_centripetal_fx_gain = clampf(float(params["centripetal_fx_gain"]), 0.0, 50.0)
	if params.has("delta_alpha_eq_gain"):
		_mc_delta_alpha_eq_gain = clampf(float(params["delta_alpha_eq_gain"]), 0.0, 4000.0)


func apply_live_tuning(params: Dictionary) -> void:
	for k in params.keys():
		_live_overrides[k] = params[k]
	if active_mode != _mc_mode:
		_refresh_mode_cache(_get_mode_dict())
	else:
		_apply_live_overrides_internal(params)


func reset_live_tuning() -> void:
	_live_overrides.clear()
	_refresh_mode_cache(_get_mode_dict())


func get_default_live_tuning() -> Dictionary:
	var m_cfg := _get_mode_dict()
	var p_lim_kw := clampf(float(m_cfg.get("p_max_low_kw", 80.0)), 10.0, 80.0)
	if active_mode == TestMode.EFFICIENCY:
		p_lim_kw = clampf(float(m_cfg.get("p_max_low_kw", 76.0)), 10.0, 80.0)
	return {
		"hairpin_pivot_gain": clampf(float(m_cfg.get("hairpin_pivot_gain", 1.0)), 0.5, 2.0),
		"inner_regen_max_nm": clampf(float(m_cfg.get("hairpin_regen_max_nm", 0.0)), 0.0, 120.0),
		"yaw_agility_gain": clampf(float(m_cfg.get("yaw_agility_gain", 1.0)), 0.5, 2.0),
		"slalom_beta_damp": clampf(float(m_cfg.get("slalom_yaw_damp_gain", 0.0)), 0.0, 1500.0),
		"beta_soft_deg": clampf(float(m_cfg.get("beta_soft_deg", 3.2)), 1.0, 10.0),
		"launch_rear_bias_pct": clampf(float(m_cfg.get("launch_rear_bias_pct", 70.0)), 50.0, 80.0),
		"tc_slip_target": clampf(float(m_cfg.get("kappa_opt", kappa_opt)), 0.04, 0.25),
		"power_limit_kw": p_lim_kw,
		"regen_limit_kw": -clampf(p_max_charge_w * 0.001, 0.0, 25.0) if bool(m_cfg.get("allow_regen", true)) else 0.0,
		"throttle_exp": clampf(float(m_cfg.get("throttle_corner_exp", 1.0)), 0.5, 2.5),
		"trail_brake_yaw_gain": clampf(float(m_cfg.get("trail_brake_yaw_gain", m_cfg.get("trail_brake_pro_turn_gain", 0.0))), 0.0, 2.5),
		"skidpad_neutral_trim": clampf(float(m_cfg.get("skidpad_neutral_trim", 0.0)), -25.0, 25.0),
		"delta_dot_lead_gain": clampf(float(m_cfg.get("delta_dot_lead_gain", 0.0)), 0.0, 120.0),
		"exit_traction_blend": clampf(float(m_cfg.get("exit_traction_blend", 0.0)), 0.0, 1.0),
		"friction_ellipse_margin": clampf(float(m_cfg.get("friction_ellipse_margin", 0.94)), 0.70, 1.05),
		"yaw_accel_ff_gain": clampf(float(m_cfg.get("yaw_accel_ff_gain", 1.0)), 0.30, 2.00),
		"centripetal_fx_gain": clampf(float(m_cfg.get("centripetal_fx_gain", 0.0)), 0.0, 50.0),
		"delta_alpha_eq_gain": clampf(float(m_cfg.get("delta_alpha_eq_gain", 0.0)), 0.0, 4000.0),
	}


func get_live_tuning() -> Dictionary:
	if active_mode != _mc_mode:
		_refresh_mode_cache(_get_mode_dict())
	return {
		"hairpin_pivot_gain": _mc_hairpin_pivot_gain,
		"inner_regen_max_nm": _mc_hairpin_regen_max_nm,
		"yaw_agility_gain": _mc_yaw_agility_gain,
		"slalom_beta_damp": _mc_slalom_damp_gain,
		"beta_soft_deg": rad_to_deg(_mc_beta_soft_rad),
		"launch_rear_bias_pct": _mc_launch_rear_bias_pct,
		"tc_slip_target": _mc_kappa_lim,
		"power_limit_kw": clampf(p_bms_max * 0.001, 10.0, 80.0),
		"regen_limit_kw": -clampf(p_max_charge_w * 0.001, 0.0, 25.0),
		"throttle_exp": _mc_th_corner_exp,
		"trail_brake_yaw_gain": _mc_tb_turn_gain,
		"skidpad_neutral_trim": _mc_skidpad_neutral_trim,
		"delta_dot_lead_gain": _mc_delta_dot_lead_gain,
		"exit_traction_blend": _mc_exit_traction_blend,
		"friction_ellipse_margin": _mc_friction_ellipse_margin,
		"yaw_accel_ff_gain": _mc_yaw_accel_ff_gain,
		"centripetal_fx_gain": _mc_centripetal_fx_gain,
		"delta_alpha_eq_gain": _mc_delta_alpha_eq_gain,
	}


func set_raw_steer_norm(val: float) -> void:
	delta_norm_raw = clampf(val, -1.0, 1.0)
	_has_ext_delta_norm_raw = true


func launch_state_name() -> String:
	match launch_state:
		LaunchState.ARMED:
			return "ARMED"
		LaunchState.LAUNCHING:
			return "LAUNCHING"
		LaunchState.NORMAL:
			return "NORMAL"
		_:
			return "IDLE"


func get_tv_live_diagnostics() -> Dictionary:
	return {
		"active_regime": _diag_active_regime,
		"mz_tv_nm": _diag_mz_tv_nm,
		"t_out_in_ratio": _diag_t_out_in_ratio,
		"pivot_blend": _diag_pivot_blend,
		"beta_deg": _diag_beta_deg,
		"beta_dot_dps": _diag_beta_dot_dps,
		"balance_state": _diag_balance_state,
		"p_dc_kw": last_power_w * 0.001,
		"mu_hat_avg": _diag_mu_hat_avg,
		"launch_state": launch_state_name(),
		"mz_crossover_nm": last_mz_crossover,
		"mz_beta_damp_nm": last_mz_beta_damp,
		"torque_fight_kw": last_torque_fight_kw,
		"is_crossover": last_is_crossover,
	}


static func _smoothstep01(x: float) -> float:
	var u := clampf(x, 0.0, 1.0)
	return u * u * (3.0 - 2.0 * u)


func get_effective_power_cap_w(vx_mps: float) -> float:
	var p_lim_w: float = p_bms_max
	if _mc_p_max_low_w > 0.0:
		var u_p := clampf((vx_mps - _mc_p_max_v_low) / maxf(_mc_p_max_v_high - _mc_p_max_v_low, 0.5), 0.0, 1.0)
		p_lim_w = lerpf(_mc_p_max_low_w, _mc_p_max_high_w, u_p)
	elif active_mode == TestMode.EFFICIENCY:
		p_lim_w = _mc_p_eff_max
	return minf(p_bms_max, clampf(p_lim_w, 1000.0, 80000.0))


func shape_corner_throttle(u_th: float, abs_delta_rad: float) -> float:
	var u := clampf(u_th, 0.0, 1.0)
	if _mc_th_corner_exp <= 1.001:
		return u
	var c_turn := clampf((abs_delta_rad - 0.08) / maxf(_mc_th_corner_delta, 0.05), 0.0, 1.0)
	var exp_eff := 1.0 + (_mc_th_corner_exp - 1.0) * c_turn
	return pow(u, exp_eff)


func compute_soft_beta_barrier_nm(
	beta_rad: float,
	beta_dot_rad_s: float,
	wz_rad_s: float = 0.0,
	vx_mps: float = 10.0
) -> float:
	var beta_kin := atan2(b_lr * wz_rad_s, maxf(vx_mps, 3.0))
	var beta_dyn := beta_rad - beta_kin
	var u_hi := clampf((vx_mps - 10.0) / 3.0, 0.0, 1.0)
	var b_thr := lerpf(_mc_beta_soft_rad, _mc_beta_hi_spd_rad, u_hi)
	# Regime 4: decouple steady |beta| from transient |beta_dot| (< 12 deg/s)
	if absf(beta_dot_rad_s) < 0.20943951:
		var steady_env := deg_to_rad(4.8) if active_mode == TestMode.SKIDPAD else lerpf(deg_to_rad(3.8), _mc_beta_hi_spd_rad, u_hi)
		b_thr = maxf(b_thr, steady_env)
	if _mc_beta_soft_gain <= 0.0 or absf(beta_dyn) <= b_thr:
		return 0.0
	var b_exc := absf(beta_dyn) - b_thr
	var b_hi := maxf(absf(beta_dyn) - lerpf(_RAD_5_0, _RAD_3_5, u_hi), 0.0)
	var b_eff := b_exc + lerpf(1.8, 2.4, u_hi) * b_hi
	var damp_scale := b_eff / maxf(b_thr, 0.02)
	return _mc_beta_soft_gain * (b_eff * signf(beta_dyn) + lerpf(0.22, 0.34, u_hi) * damp_scale * beta_dot_rad_s)


# precalcula coefs lpf cuando cambia dt pa ahorrar exp() en step()
func _refresh_dt_cache(dt: float) -> void:
	_dt_cache = dt
	_a_bd_10 = 1.0 - exp(-dt / 0.010)
	_a_d_8 = 1.0 - exp(-dt / 0.008)
	_a_ff_20 = 1.0 - exp(-dt / 0.020)
	_a_wt = 1.0 - exp(-dt / maxf(tau_wt, 0.005))
	_a_ay_12 = 1.0 - exp(-dt / 0.012)
	_a_om_15 = 1.0 - exp(-dt / 0.015)
	_e_mz_45 = exp(-dt / 0.045)
	_a_rlim_45 = 1.0 - _e_mz_45
	_a_eff_150 = 1.0 - exp(-dt / 0.15)


## paso principal del controlador 4wd (200 hz): 4 capas + modos + guardias
func step(
	fx_req: float,
	steer_rad: float,
	vx_meas: float,
	wz_meas: float,
	ax_meas: float,
	ay_meas: float,
	az_meas: float,
	w_omegas: Array[float],
	brake_in: float,
	dt: float,
	w_steers: Array[float]
) -> Array[float]:
	if dt <= 0.0:
		return last_t_cmd

	var delta_raw := 0.0 if fault_steer_sensor else steer_rad
	var d_norm_raw := 0.0 if fault_steer_sensor else (delta_norm_raw if _has_ext_delta_norm_raw else clampf(delta_raw / maxf(max_steer_rad, 0.1), -1.0, 1.0))
	var is_first_tick := not _ekf_init
	if is_first_tick or active_mode != _mc_mode:
		_refresh_mode_cache(_get_mode_dict())
	if absf(dt - _dt_cache) > 1e-7:
		_refresh_dt_cache(dt)

	# init ekf en primer tick pa evitar error t=0 en derivadas, curb y rls
	if is_first_tick:
		var cnt_init := 0
		var kz_safe_i := maxf(tire_kz, 20000.0)
		for i in range(4):
			if not fault_wheel_mask[i] and not is_nan(w_omegas[i]):
				var r_eff_init := clampf(r_eff - (fz_est[i] / kz_safe_i) / 3.0, 0.185, r_eff)
				_med_buf[cnt_init] = maxf(w_omegas[i] * r_eff_init, 0.0)
				cnt_init += 1
		var v_init := _median_buf(_med_buf, cnt_init) if cnt_init > 0 else maxf(vx_meas, 0.0)
		ekf_vx = maxf(v_init, 0.0)
		ekf_vy = 0.0
		ekf_bz = 0.0
		_ax_filt = ax_meas
		_ax_long_filt = ax_meas
		_ay_filt = ay_meas
		_az_prev = az_meas
		_delta_lpf1 = delta_raw
		_delta_lpf2 = delta_raw
		_delta_ff1 = delta_raw
		_delta_ff2 = delta_raw
		_delta_filt_prev = delta_raw
		_r_lim_filt = 0.0
		_beta_prev = 0.0
		_beta_dot = 0.0
		_beta_dot_filt = 0.0
		for i in range(4):
			_omega_prev[i] = w_omegas[i] if not is_nan(w_omegas[i]) else (ekf_vx / r_eff)
			_om_dot_filt[i] = 0.0
			_kappa_prev[i] = 0.0
			_kappa_rel[i] = 0.0
			_ki_buf[i] = 0.0
			_split_spin_cnt[i] = 0
		_turnin_timer_s = 0.16
		_wz_abs_prev = absf(wz_meas)
		_ay_prev = ay_meas
		_alpha_f_rel = 0.0
		_alpha_r_rel = 0.0
		_eff_t_filt = 0.0
		_lift_regen_fx_filt = 0.0
		_ekf_init = true

	# guardias seguridad: deteccion fallo rueda <=10 ms + recuperacion histeresis + curb-strike
	var wz_corr := wz_meas - ekf_bz
	var faulty_wheels := 0
	var az_old := _az_prev
	var dz_signed := az_meas - az_old
	var daz := absf(dz_signed)
	_az_prev = az_meas
	for i in range(4):
		var trk_i := tf if i < 2 else tr
		var side_i := -0.5 if (i % 2 == 0) else 0.5
		var nan_w := is_nan(w_omegas[i])
		var om_dot_i := (w_omegas[i] - _omega_prev[i]) / dt if not nan_w else 0.0
		var om_dot_chk := lerpf(_om_dot_filt[i], om_dot_i, _a_om_15)
		var v_exp_i := maxf(ekf_vx + side_i * trk_i * wz_corr, 0.5)
		if nan_w:
			_fault_auto[i] = true
			_fault_cnt[i] = 2
			_fault_rec_cnt[i] = 0
		elif ekf_vx > 3.0 and brake_in < 0.05:
			var w_vel_i := w_omegas[i] * r_eff
			var curb_rebound := (_curb_wheel_timer[i] > 0.0 and w_vel_i > 0.70 * v_exp_i)
			if (w_vel_i < 0.35 * v_exp_i) or (active_mode != TestMode.TIRE_WARMER and om_dot_chk < -450.0 and not curb_rebound):
				_fault_cnt[i] += 1
				_fault_rec_cnt[i] = 0
				if _fault_cnt[i] >= 2:
					_fault_auto[i] = true
			elif w_vel_i > 0.55 * v_exp_i:
				_fault_cnt[i] = 0
				if _fault_auto[i]:
					_fault_rec_cnt[i] += 1
					if _fault_rec_cnt[i] >= 4:
						_fault_auto[i] = false
						_fault_rec_cnt[i] = 0
				else:
					_fault_rec_cnt[i] = 0
			else:
				_fault_cnt[i] = 0
				_fault_rec_cnt[i] = 0
		else:
			_fault_cnt[i] = 0
			if ekf_vx < 2.0:
				_fault_auto[i] = false
				_fault_rec_cnt[i] = 0
			elif _fault_auto[i] and (w_omegas[i] * r_eff) > 0.55 * v_exp_i:
				_fault_rec_cnt[i] += 1
				if _fault_rec_cnt[i] >= 4:
					_fault_auto[i] = false
					_fault_rec_cnt[i] = 0
			else:
				_fault_rec_cnt[i] = 0

		var wheel_bad := fault_wheel_mask[i] or _fault_auto[i] or nan_w
		if wheel_bad:
			faulty_wheels += 1
		elif _fault_cnt[i] == 0:
			# detecta rueda en aire por aceleracion libre diferencial y caida de soporte vertical az
			var j_cb := (1 - i) if i < 2 else (5 - i)
			var om_dot_p := (w_omegas[j_cb] - _omega_prev[j_cb]) / dt if not is_nan(w_omegas[j_cb]) else 0.0
			var free_spin_i := (om_dot_i > 125.0) and ((om_dot_i - om_dot_p) > 40.0) and (om_dot_i > 0.45 * (t_prev[i] / maxf(i_wheel_eff, 0.25)))
			var curb_hit := (active_mode != TestMode.TIRE_WARMER) and ((ekf_vx > 5.0 and az_meas > 21.0 and dz_signed > 6.0 and om_dot_i > 500.0 and (om_dot_i - om_dot_p) > 200.0) or ((not is_first_tick) and active_mode != TestMode.ACCELERATION and ekf_vx > 8.0 and mu_hat[i] > 1.15 and t_prev[i] > 70.0 and free_spin_i and az_old > 17.2 and az_meas < 18.25 and (dz_signed < -1.10 or az_meas < 17.85) and (w_omegas[i] * r_eff - v_exp_i) > 0.25))
			if curb_hit:
				_curb_wheel_timer[i] = 0.072
				_curb_timer_s = maxf(_curb_timer_s, 0.072)
		_rls_theta_prev[i] = rls_theta[i]
	limp_home_active = fault_steer_sensor or (faulty_wheels >= 1)

	# reconstruye vel de rueda fallida con cinematica ekf (cero malloc)
	var w_eff: Array[float] = _w_eff
	for i in range(4):
		if fault_wheel_mask[i] or _fault_auto[i] or is_nan(w_omegas[i]):
			var trk_f := tf if i < 2 else tr
			var side_f := -0.5 if (i % 2 == 0) else 0.5
			w_eff[i] = maxf(ekf_vx + side_f * trk_f * wz_corr, 0.0) / r_eff
		else:
			w_eff[i] = w_omegas[i]

	# capa 1a: ekf drift-free [vx, vy, bz] con rueda virtual robusta a split-mu/bloqueo/curb
	var vx_pred := ekf_vx + (ax_meas + wz_corr * ekf_vy) * dt
	var cnt_v := 0
	var v_min := 1e9
	var v_max := -1e9
	for i in range(4):
		if fault_wheel_mask[i] or _fault_auto[i] or is_nan(w_omegas[i]) or _curb_wheel_timer[i] > 0.0 or _fault_cnt[i] > 0:
			continue
		var trk_i := tf if i < 2 else tr
		var side_i := -0.5 if (i % 2 == 0) else 0.5
		# usa radio efectivo cargado de wheeldynamics y compensa slip pa quitar sesgo +vx
		var r_eff_i := clampf(r_eff - (fz_est[i] / maxf(tire_kz, 20000.0)) / 3.0, 0.185, r_eff)
		var fx_p := (t_prev[i] - t_hyd_est[i]) / maxf(r_load_est[i], 0.15)
		var c_kx_i := maxf(38.0 * fz_est[i], 15000.0)
		var k_est_i := clampf(fx_p / c_kx_i, -0.03, 0.03)
		var v_w_i := (w_eff[i] * r_eff_i) / (1.0 + k_est_i)
		var v_corr_i := maxf(v_w_i - side_i * trk_i * wz_corr, 0.0)
		_med_buf[cnt_v] = v_corr_i
		cnt_v += 1
		if v_corr_i < v_min:
			v_min = v_corr_i
		if v_corr_i > v_max:
			v_max = v_corr_i

	var vx_wheel_ref := maxf(vx_pred, 0.0)
	if cnt_v > 0:
		if active_mode == TestMode.TIRE_WARMER:
			# en warmer usa como ref ekf el eje en rama ascendente segun fase
			var use_front_ref := (_warmer_sgn < 0.0)
			var w_ref_ax := 0.5 * (w_eff[0] + w_eff[1]) if use_front_ref else 0.5 * (w_eff[2] + w_eff[3])
			var fz_ref_ax := maxf((fz_est[0] + fz_est[1]) if use_front_ref else (fz_est[2] + fz_est[3]), 400.0)
			var t_ref_ax := (t_prev[0] + t_prev[1]) if use_front_ref else (t_prev[2] + t_prev[3])
			var r_eff_ax := clampf(r_eff - (0.5 * fz_ref_ax / maxf(tire_kz, 20000.0)) / 3.0, 0.185, r_eff)
			var k_ref_est := clampf(t_ref_ax / (36.5 * r_eff_ax * fz_ref_ax), -0.07, 0.07)
			vx_wheel_ref = maxf((w_ref_ax * r_eff_ax) / (1.0 + k_ref_est), 0.0)
		elif ax_meas > 0.5:
			vx_wheel_ref = v_min
		elif ax_meas < -0.5 or brake_in > 0.05:
			vx_wheel_ref = v_max
		else:
			vx_wheel_ref = _median_buf(_med_buf, cnt_v)
	elif faulty_wheels >= 4:
		vx_wheel_ref = maxf(vx_meas, 0.0)

	_ekf_p[0] += 0.05 * dt
	var r_ekf_vx := 0.30 if limp_home_active else 0.04
	var k0 := _ekf_p[0] / (_ekf_p[0] + r_ekf_vx)
	ekf_vx = maxf(vx_pred + k0 * (vx_wheel_ref - vx_pred), 0.0)
	_ekf_p[0] *= (1.0 - k0)

	# Deterministic Launch State Machine (IDLE -> ARMED -> LAUNCHING -> NORMAL)
	var fx_drv_cap := 4.0 * t_motor_peak * gear_ratio / maxf(r_eff, 0.05)
	var throttle_intent := clampf(fx_req / maxf(fx_drv_cap, 100.0), 0.0, 1.0)
	if ekf_vx > 1.5:
		_was_moving_in_run = true
	if brake_in < 0.05 and ekf_vx > 2.0:
		for i in range(4):
			_pre_brake_mu[i] = mu_hat[i]

	if launch_state == LaunchState.IDLE or launch_state == LaunchState.NORMAL:
		if ekf_vx < 1.35:
			launch_state = LaunchState.ARMED
		elif launch_state == LaunchState.IDLE:
			launch_state = LaunchState.NORMAL

	if launch_state == LaunchState.ARMED:
		if throttle_intent > 0.70 and brake_in < 0.20:
			launch_state = LaunchState.LAUNCHING
			_launch_timer_s = 0.0
			_mz_int = 0.0
			_accel_elapsed_s = 0.0
			_r_lim_filt = 0.0
			_curb_timer_s = 0.0
			_beta_prev = 0.0
			_beta_dot = 0.0
			_beta_dot_filt = 0.0
			limp_home_active = false
			for i in range(4):
				mu_hat[i] = mu_init
				_pre_brake_mu[i] = mu_init
				_ki_buf[i] = 0.0
				_kappa_prev[i] = 0.0
				_kappa_rel[i] = 0.0
				_split_spin_cnt[i] = 0
				_om_dot_filt[i] = 0.0
				_fx_est_prev[i] = 0.0
				t_prev[i] = 0.0
				_qp_x[i] = 0.0
				_qp_y[i] = 0.0
				_qp_z[i] = 0.0
				_curb_wheel_timer[i] = 0.0
				_fault_auto[i] = false
				_fault_cnt[i] = 0
				_fault_rec_cnt[i] = 0
				rls_theta[i] = 30000.0
				rls_p[i] = 1000.0
				_rls_theta_prev[i] = 30000.0
				_omega_prev[i] = w_eff[i]
			_was_moving_in_run = false
		elif ekf_vx >= 1.35:
			launch_state = LaunchState.NORMAL
		elif _was_moving_in_run and ekf_vx < 0.5 and brake_in < 0.05:
			_mz_int = 0.0
			_accel_elapsed_s = 0.0
			_launch_timer_s = 0.0
			_r_lim_filt = 0.0
			for i in range(4):
				_ki_buf[i] = 0.0
				_kappa_prev[i] = 0.0
				_kappa_rel[i] = 0.0
				_split_spin_cnt[i] = 0
				_om_dot_filt[i] = 0.0
				_fx_est_prev[i] = 0.0
				t_prev[i] = 0.0
				_qp_x[i] = 0.0
				_qp_y[i] = 0.0
				_qp_z[i] = 0.0
				rls_theta[i] = 30000.0
				rls_p[i] = 1000.0
				_rls_theta_prev[i] = 30000.0
				_omega_prev[i] = w_eff[i]
				if _was_braking_prev or mu_hat[i] < _pre_brake_mu[i]:
					mu_hat[i] = maxf(mu_hat[i], _pre_brake_mu[i])
			_was_moving_in_run = false
	elif launch_state == LaunchState.LAUNCHING:
		if ekf_vx > 11.0:
			launch_state = LaunchState.NORMAL
		elif brake_in >= 0.25 or throttle_intent <= 0.25:
			launch_state = LaunchState.ARMED if ekf_vx < 1.35 else LaunchState.NORMAL
	_was_braking_prev = (brake_in >= 0.05)

	# ekf_vy: cinematica + ref fisica eje tras con tau adaptativa pa capturar deriva dlc/mu-jump
	var vy_pred := ekf_vy + (ay_meas - wz_corr * ekf_vx) * dt
	_ekf_p[1] += 0.08 * dt
	var fy_r_est := (a_lf / wb) * mass * ay_meas
	var fz_r_tot := maxf(fz_est[2] + fz_est[3], 500.0)
	var load_u_r := clampf((0.5 * fz_r_tot - 300.0) / 1800.0, 0.0, 1.0)
	var mu_y_r := lerpf(0.89, 0.73, load_u_r)
	var c_norm_r := lerpf(32.5, 13.8, load_u_r)
	var mu_r_avg := 0.5 * (mu_hat[2] + mu_hat[3])
	var fy_r_max := maxf(mu_y_r * mu_r_avg * fz_r_tot, 500.0)
	var fy_r_rat := clampf(absf(fy_r_est) / fy_r_max, 0.0, 0.88)
	var atanh_r := 0.5 * log((1.0 + fy_r_rat) / (1.0 - fy_r_rat))
	var alpha_r_est := signf(fy_r_est) * clampf(((mu_y_r * mu_r_avg) / c_norm_r) * atanh_r, 0.0, 0.14)
	var vy_model := b_lr * wz_corr - maxf(ekf_vx, 0.5) * alpha_r_est
	var mu_avg_prev := 0.25 * (mu_hat[0] + mu_hat[1] + mu_hat[2] + mu_hat[3])
	var tau_vy := 2.2
	var vy_tgt := vy_model
	if ekf_vx < 0.8:
		tau_vy = 0.05
		vy_tgt = 0.0
	elif absf(wz_corr) < 0.05 and absf(ay_meas) < 1.0:
		tau_vy = 0.12
		vy_tgt = 0.0
	elif absf(ay_meas) < 0.70 * mu_avg_prev * 9.81 and absf(ay_meas - wz_corr * ekf_vx) < 3.0:
		tau_vy = 0.45
	var k_vy := dt / (tau_vy + dt)
	ekf_vy = clampf(vy_pred + k_vy * (vy_tgt - vy_pred), -15.0, 15.0)
	_ekf_p[1] *= (1.0 - k_vy)

	var bz_pred := ekf_bz
	_ekf_p[2] += 1e-5 * dt
	if ekf_vx < 0.25 and absf(wz_meas) < 0.05:
		var k2 := _ekf_p[2] / (_ekf_p[2] + 0.02)
		ekf_bz = bz_pred + k2 * (wz_meas - bz_pred)
		_ekf_p[2] *= (1.0 - k2)

	var vx_eff := maxf(ekf_vx, 0.5)
	var beta := atan2(ekf_vy, vx_eff) if ekf_vx > 1.5 else 0.0
	_beta_prev = beta
	# beta_dot analitico exacto sin deriva numerica ruidosa + lpf 10 ms
	var v2 := maxf(ekf_vx * ekf_vx + ekf_vy * ekf_vy, 1.0)
	var beta_dot_raw := ((ay_meas * ekf_vx - ax_meas * ekf_vy) / v2 - wz_corr) if ekf_vx > 1.0 else 0.0
	_beta_dot_filt = lerpf(_beta_dot_filt, beta_dot_raw, _a_bd_10)
	_beta_dot = _beta_dot_filt

	# capa 1b/2a: cascada 2x lpf critica pa delta y derivada unipolar band-limited pa ff
	var alpha_d := _a_d_8
	_delta_lpf1 = lerpf(_delta_lpf1, delta_raw, alpha_d)
	_delta_lpf2 = lerpf(_delta_lpf2, _delta_lpf1, alpha_d)
	var delta_filt := _delta_lpf2

	var tau_ff := 0.020
	var alpha_ff := _a_ff_20
	_delta_ff1 = lerpf(_delta_ff1, delta_raw, alpha_ff)
	_delta_ff2 = lerpf(_delta_ff2, _delta_ff1, alpha_ff)
	var delta_dot := clampf((_delta_ff1 - _delta_ff2) / tau_ff, -12.0, 12.0)
	var d_delta_step := delta_filt - _delta_filt_prev
	_delta_filt_prev = delta_filt
	# ventana transitoria de giro pa no disparar lat_sat ni falsos positivos en dlc/skidpad
	if absf(delta_dot) > 0.8 or absf(d_delta_step) > 1e-4:
		_turnin_timer_s = 0.16
	elif _turnin_timer_s > 0.0:
		_turnin_timer_s = maxf(_turnin_timer_s - dt, 0.0)

	# Launch detection (< 11.0 m/s in LAUNCHING FSM) for < 15 ms filter bypass
	var is_launch_req := (launch_state == LaunchState.LAUNCHING) or ((ekf_vx < 4.5) and (fx_req > 0.80 * fx_drv_cap) and (brake_in < 0.20))
	if is_launch_req:
		_launch_timer_s += dt
	elif ekf_vx >= 11.0 or brake_in >= 0.25 or fx_req <= 50.0:
		_launch_timer_s = 0.0
	var launch_bypass_filt := is_launch_req and (_launch_timer_s <= 0.015)

	# capa 1c: freno hidraulico + fz_i acoplada a anti-dive/squat sin doble conteo + radio cargado
	if launch_bypass_filt or (launch_state == LaunchState.LAUNCHING and _ax_long_filt < 0.0):
		var ax_launch_est := maxf(ax_meas, fx_req / maxf(mass, 100.0))
		_ax_filt = ax_launch_est
		_ax_long_filt = ax_launch_est
	else:
		_ax_filt = lerpf(_ax_filt, ax_meas, _a_wt)
		_ax_long_filt = lerpf(_ax_long_filt, ax_meas, _a_ff_20)
	_ay_filt = lerpf(_ay_filt, ay_meas, _a_ay_12)

	if brake_in > 0.05 and fx_req <= 0.0 and ekf_vx > 0.25:
		if mech_brake_start > 0.05:
			var hyd_frac := clampf((brake_in - mech_brake_start) / maxf(1.0 - mech_brake_start, 0.05), 0.0, 1.0)
			var low_spd_hyd := clampf((2.0 - ekf_vx) / 1.2, 0.0, 1.0)
			var hyd_scale := maxf(hyd_frac, low_spd_hyd * brake_in)
			var f_hyd_tot := hyd_scale * brake_mu_cfg * mass * 9.81
			for i in range(4):
				var bias_i := brake_bias_f_cfg if i < 2 else (1.0 - brake_bias_f_cfg)
				t_hyd_est[i] = f_hyd_tot * 0.5 * bias_i * r_eff
		else:
			var sum_regen_prev := 0.0
			for j in range(4):
				sum_regen_prev += -minf(0.0, t_prev[j])
			var f_brk_tot := brake_in * brake_mu_cfg * mass * 9.81
			for i in range(4):
				var bias_i := brake_bias_f_cfg if i < 2 else (1.0 - brake_bias_f_cfg)
				var t_brk_req_i := f_brk_tot * 0.5 * bias_i * r_eff
				var cut_regen_i := 0.5 * sum_regen_prev * bias_i
				t_hyd_est[i] = maxf(0.0, t_brk_req_i - cut_regen_i)
	else:
		for i in range(4):
			t_hyd_est[i] = 0.0

	var f_down := 0.5 * 1.225 * cl_a * ekf_vx * ekf_vx
	var fz_s_f := 0.5 * mass * 9.81 * front_w_dist + 0.5 * f_down * aero_bal_f
	var fz_s_r := 0.5 * mass * 9.81 * (1.0 - front_w_dist) + 0.5 * f_down * (1.0 - aero_bal_f)

	var r_load_avg_f := maxf(0.5 * (r_load_est[0] + r_load_est[1]), 0.15)
	var r_load_avg_r := maxf(0.5 * (r_load_est[2] + r_load_est[3]), 0.15)
	var fx_f_est := (t_prev[0] + t_prev[1] - t_hyd_est[0] - t_hyd_est[1]) / r_load_avg_f
	var fx_r_est := (t_prev[2] + t_prev[3] - t_hyd_est[2] - t_hyd_est[3]) / r_load_avg_r
	var dfz_kin_front := -(h_cog / wb) * (anti_dive_f * fx_f_est + anti_squat_r * fx_r_est)
	var anti_w := clampf(absf(dfz_kin_front) / maxf(absf(mass * ax_meas * h_cog / wb), 50.0), 0.0, 0.85)
	var elast_scale := 1.0 - anti_w
	var dfz_elast_front := -elast_scale * (mass * _ax_long_filt * h_cog) / wb
	var dfz_long_front := dfz_kin_front + dfz_elast_front

	var h_rc_cg := (b_lr / wb) * h_rc_f + (a_lf / wb) * h_rc_r
	var h_arm := maxf(h_sprung_cog - h_rc_cg, 0.05)
	var dfz_lat_f := (mass * _ay_filt / tf) * ((b_lr / wb) * h_rc_f + lltd_f * h_arm)
	var dfz_lat_r := (mass * _ay_filt / tr) * ((a_lf / wb) * h_rc_r + (1.0 - lltd_f) * h_arm)

	_fz_unclamped[0] = fz_s_f + 0.5 * dfz_long_front - dfz_lat_f
	_fz_unclamped[1] = fz_s_f + 0.5 * dfz_long_front + dfz_lat_f
	_fz_unclamped[2] = fz_s_r - 0.5 * dfz_long_front - dfz_lat_r
	_fz_unclamped[3] = fz_s_r - 0.5 * dfz_long_front + dfz_lat_r
	fz_est[0] = maxf(_fz_unclamped[0], 50.0)
	fz_est[1] = maxf(_fz_unclamped[1], 50.0)
	fz_est[2] = maxf(_fz_unclamped[2], 50.0)
	fz_est[3] = maxf(_fz_unclamped[3], 50.0)
	for i in range(4):
		r_load_est[i] = clampf(r_eff - fz_est[i] / tire_kz, 0.175, r_eff)
		if _fz_unclamped[i] < 40.0 or _curb_wheel_timer[i] > 0.0:
			_airborne_ramp[i] = 0.0
		else:
			_airborne_ramp[i] = minf(_airborne_ramp[i] + 25.0 * dt, 1.0)

	# derivas x eje + filtro relajacion de carcasa pa observador lateral y modelo fy
	var d_fl := w_steers[0] if w_steers.size() >= 2 else delta_filt
	var d_fr := w_steers[1] if w_steers.size() >= 2 else delta_filt
	var alpha_f := delta_filt - atan2(ekf_vy + a_lf * wz_corr, vx_eff) if ekf_vx > 2.0 else 0.0
	var alpha_r := -atan2(ekf_vy - b_lr * wz_corr, vx_eff) if ekf_vx > 2.0 else 0.0
	var tau_rel := clampf(0.26 / maxf(ekf_vx, 3.0), 0.010, 0.065)
	var a_rel := 1.0 - exp(-dt / tau_rel)
	_alpha_f_rel = lerpf(_alpha_f_rel, alpha_f, a_rel)
	_alpha_r_rel = lerpf(_alpha_r_rel, alpha_r, a_rel)

	# capa 4 / id online: rls con fx_dyn + relajacion kappa_rel + mu_hat combinado y lateral
	var mu_avg := 0.0
	var slip_power_step := 0.0
	var kz_safe_rls := maxf(tire_kz, 20000.0)
	var tau_kx := clampf(0.16 / maxf(ekf_vx, 3.0), 0.008, 0.045)
	var a_kx := 1.0 - exp(-dt / tau_kx)
	for i in range(4):
		var trk_k := tf if i < 2 else tr
		var side_k := -0.5 if (i % 2 == 0) else 0.5
		var vc_k := maxf(ekf_vx + side_k * trk_k * wz_corr, 0.5)
		_ki_buf[i] = (w_eff[i] * r_eff - vc_k) / maxf(vc_k, 1.0)
	for i in range(4):
		var trk_i := tf if i < 2 else tr
		var side_i := -0.5 if (i % 2 == 0) else 0.5
		var v_corner_x := maxf(ekf_vx + side_i * trk_i * wz_corr, 0.5)
		var ki := _ki_buf[i]
		var dk := ki - _kappa_prev[i]
		var wheel_bad := fault_wheel_mask[i] or _fault_auto[i] or is_nan(w_omegas[i])
		if wheel_bad:
			_om_dot_filt[i] = 0.0
			_omega_prev[i] = w_eff[i]
		else:
			var om_dot_raw := (w_eff[i] - _omega_prev[i]) / dt
			_om_dot_filt[i] = lerpf(_om_dot_filt[i], om_dot_raw, _a_om_15)
			_omega_prev[i] = w_eff[i]
		var fx_dyn_i := (t_prev[i] - t_hyd_est[i] - i_wheel_eff * _om_dot_filt[i]) / r_load_est[i]
		var dfx := fx_dyn_i - _fx_est_prev[i]
		if not is_first_tick and absf(dk) > 0.002 and absf(ki) < 0.25 and not wheel_bad and _curb_wheel_timer[i] <= 0.0 and brake_in <= maxf(mech_brake_start, 0.05) and t_hyd_est[i] < 1.0:
			var p_i := rls_p[i]
			var gain := (p_i * dk) / (rls_lambda + dk * p_i * dk)
			rls_theta[i] = clampf(rls_theta[i] + gain * (dfx - dk * rls_theta[i]), 2000.0, 90000.0)
			rls_p[i] = clampf((1.0 - gain * dk) * p_i / rls_lambda, 10.0, 5000.0)
		var r_eff_rls := clampf(r_eff - (fz_est[i] / kz_safe_rls) / 3.0, 0.185, r_eff)
		var ki_true := (w_eff[i] * r_eff_rls - v_corner_x) / maxf(v_corner_x, 1.0)
		if is_first_tick or wheel_bad:
			_kappa_rel[i] = 0.0
		else:
			_kappa_rel[i] = lerpf(_kappa_rel[i], ki_true, a_kx)
		# nunca actualiza mu_hat con rueda bloqueada o en transitorio de relajacion de carcasa
		if not is_first_tick and not wheel_bad and _curb_wheel_timer[i] <= 0.0 and ekf_vx > 2.0 and active_mode != TestMode.TIRE_WARMER and absf(w_eff[i]) >= 2.0 and absf(ki) < 0.28:
			var f_comb := sqrt(fx_dyn_i * fx_dyn_i + fy_est[i] * fy_est[i])
			if fz_est[i] > 150.0:
				var mu_obs := clampf(f_comb / fz_est[i], mu_min, mu_max)
				var recovering := (_om_dot_filt[i] * signf(ki_true)) < -40.0
				var j_ax := (1 - i) if i < 2 else (5 - i)
				var dw_ax := w_eff[i] - w_eff[j_ax]
				var dk_ax := ki - _ki_buf[j_ax]
				if absf(delta_filt) < _RAD_6_0 and (dw_ax > 0.8 and dk_ax > 0.040) and absf(ki_true) > 0.040:
					_split_spin_cnt[i] = mini(_split_spin_cnt[i] + 1, 10)
				else:
					_split_spin_cnt[i] = 0
				var split_spin := (_split_spin_cnt[i] >= 2) and mu_hat[j_ax] > 1.10
				var allow_ax_drop := absf(delta_filt) >= _RAD_1_2 or (dk_ax >= 0.015 and dw_ax >= 0.35 and mu_hat[j_ax] > 1.10) or split_spin
				var loaded_or_spin := (fz_est[i] > 550.0) or (dk_ax > 0.035) or (absf(ki_true) > 0.070) or (absf(ay_meas) < 5.5)
				var corner_unsat := (absf(ay_meas) > 5.0) and (not split_spin) and (absf(ki_true) < 0.090)
				var skip_launch_drop := (launch_state == LaunchState.LAUNCHING) and (absf(delta_filt) < _RAD_4_0) and (ekf_vx < 8.0) and (mu_obs < 0.85 * mu_init)
				if (not skip_launch_drop) and (not corner_unsat) and (not recovering) and loaded_or_spin and allow_ax_drop and absf(_kappa_rel[i]) > 0.055 and absf(ki_true) > 0.048 and absf(fx_dyn_i) > 250.0:
					mu_hat[i] = lerpf(mu_hat[i], mu_obs, 0.24)
				elif (not corner_unsat) and active_mode != TestMode.ACCELERATION and launch_state != LaunchState.LAUNCHING and (not recovering) and loaded_or_spin and absf(fx_dyn_i) > 200.0 and mu_obs < 0.78 * mu_hat[i] and ((allow_ax_drop and absf(_kappa_rel[i]) > 0.036 and absf(ki_true) > 0.048) or split_spin):
					mu_hat[i] = lerpf(mu_hat[i], mu_obs, 0.78 if split_spin else 0.36)
				elif absf(_kappa_rel[i]) > 0.025 and mu_obs > mu_hat[i] + 0.04:
					mu_hat[i] = lerpf(mu_hat[i], mu_obs, 0.14)
		slip_power_step += absf(fx_dyn_i * (w_eff[i] * r_eff - v_corner_x))
		_kappa_prev[i] = ki
		_fx_est_prev[i] = fx_dyn_i
		mu_avg += 0.25 * mu_hat[i]

	# ajuste mu_hat lateral: salto brusco, caida vs modelo con relajacion y saturacion
	var g_eff := maxf((fz_est[0] + fz_est[1] + fz_est[2] + fz_est[3]) / mass, 9.81)
	var mu_x_lat := 1.22 * (absf(ay_meas) / g_eff)
	var fy_mod_sum := 0.0
	var cos_df := cos(delta_filt)
	for i in range(4):
		var ai_rel := _alpha_f_rel if i < 2 else _alpha_r_rel
		var fy_i_mod := _calc_hoosier_fy(ai_rel, mu_hat[i], fz_est[i])
		fy_mod_sum += (fy_i_mod * cos_df) if i < 2 else fy_i_mod
	var ay_mod_rel := absf(fy_mod_sum) / mass
	var model_wet_drop := (not is_first_tick) and absf(delta_dot) < 0.85 and absf(alpha_f) > _RAD_2_5 and (absf(alpha_f) >= 0.70 * absf(_alpha_f_rel) or absf(_alpha_f_rel) <= _RAD_2_5) and ay_mod_rel > 3.8 and absf(ay_meas) > 2.2 and (absf(_alpha_f_rel) > _RAD_3_5 or absf(_alpha_r_rel) > _RAD_2_5) and (absf(ay_meas) < 0.77 * ay_mod_rel) and (_alpha_f_rel * _alpha_r_rel >= -0.0005)
	var sudden_mu_drop := (not is_first_tick) and absf(_ay_filt) > 7.5 and absf(ay_meas) < 0.72 * absf(_ay_filt) and absf(delta_filt) > _RAD_3_0 and absf(delta_dot) < 1.2
	var lat_sat := _turnin_timer_s <= 0.0 and absf(delta_dot) < 1.2 and absf(ay_meas - _ay_prev) / dt < 30.0 and absf(alpha_f) > _RAD_5_5 and absf(alpha_r) > _RAD_3_5
	if (sudden_mu_drop or model_wet_drop or lat_sat or absf(ay_meas) > 4.0) and not (launch_state == LaunchState.LAUNCHING and ekf_vx < 5.0):
		var mu_avg_pre := mu_avg
		var mu_down_tgt := clampf(mu_x_lat * 1.02, mu_min, mu_max)
		var mu_up_tgt := clampf(mu_x_lat * 1.08, mu_min, mu_max)
		var mod_ratio := absf(ay_meas) / maxf(ay_mod_rel, 1.0)
		mu_avg = 0.0
		for i in range(4):
			if sudden_mu_drop and mu_down_tgt < 0.85 * mu_hat[i]:
				mu_hat[i] = lerpf(mu_hat[i], mu_down_tgt, 0.45)
			elif model_wet_drop:
				var mu_mod_tgt := clampf(mu_hat[i] * mod_ratio * 1.03, mu_min, mu_max)
				if mu_mod_tgt < mu_hat[i]:
					mu_hat[i] = lerpf(mu_hat[i], mu_mod_tgt, 0.45)
			elif lat_sat and mu_x_lat < 0.82 * mu_avg_pre:
				mu_hat[i] = lerpf(mu_hat[i], mu_down_tgt, 0.22)
			elif mu_up_tgt > mu_hat[i] and absf(ay_meas) > 4.0 and mu_x_lat > 0.85 * mu_hat[i]:
				mu_hat[i] = lerpf(mu_hat[i], mu_up_tgt, 0.18)
			mu_avg += 0.25 * mu_hat[i]
	_ay_prev = ay_meas

	slip_energy_j += slip_power_step * dt
	if active_mode == TestMode.AUTOCROSS:
		var warm_frac := clampf(slip_energy_j / _mc_e_ref, 0.0, 1.0)
		var mu_cap := lerpf(_mc_cold_floor * mu_init, mu_max, warm_frac)
		for i in range(4):
			mu_hat[i] = minf(mu_hat[i], mu_cap)
		mu_avg = minf(mu_avg, mu_cap)

	# estimacion fy_i y fy_cons_est_i sensible a carga hoosier (sin duplicar ley)
	var fz_ax_f := maxf(fz_est[0] + fz_est[1], 100.0)
	var fz_ax_r := maxf(fz_est[2] + fz_est[3], 100.0)
	var fy_ax_f := (b_lr / wb) * mass * absf(ay_meas)
	var fy_ax_r := (a_lf / wb) * mass * absf(ay_meas)
	for i in range(4):
		var ai := alpha_f if i < 2 else alpha_r
		fy_est[i] = _calc_hoosier_fy(ai, mu_hat[i], fz_est[i])
		var load_u := clampf((fz_est[i] - 300.0) / 1100.0, 0.0, 1.0)
		var fy_lin_i := (fy_ax_f * (fz_est[i] / fz_ax_f)) if i < 2 else (fy_ax_r * (fz_est[i] / fz_ax_r))
		var ls_boost := lerpf(1.12, 0.94, load_u)
		var fy_newton_i := 0.92 * ls_boost * fy_lin_i
		fy_cons_est[i] = maxf(absf(fy_est[i]), fy_newton_i)

	# capa 2b: ref guinada r_ref con k_us(vx), ff inercial unipolar, lqr con barrera c1 y anti-windup
	# mezcla C1 de regimenes de velocidad: Regimen 1 (vx < 9.5) -> Regimen 2 (vx >= 11.5)
	var beta_kin := atan2(b_lr * wz_corr, maxf(ekf_vx, 3.0))
	var beta_dyn := beta - beta_kin
	var u_spd_raw := clampf((ekf_vx - 9.5) / 2.0, 0.0, 1.0)
	var u_hi_spd := u_spd_raw * u_spd_raw * (3.0 - 2.0 * u_spd_raw)
	var u_lo_spd := 1.0 - u_hi_spd
	var u_tight_hp := u_lo_spd * clampf((absf(delta_filt) - _RAD_8_0) / _RAD_5_0, 0.0, 1.0) if (active_mode == TestMode.AUTOCROSS and launch_state != LaunchState.LAUNCHING and ekf_vx > 1.2 and mu_avg >= 0.90) else 0.0
	# Regime 2: pre-attenuation hairpin pivot trigger (|delta_norm_raw| > 0.42 & vx < 11.0 m/s, without active oversteer)
	var ov_pivot_gate := clampf((absf(beta_dyn) - _RAD_3_5) / _RAD_2_0, 0.0, 1.0)
	var u_hp_raw := _smoothstep01((absf(d_norm_raw) - 0.42) / 0.18) * (1.0 - _smoothstep01((ekf_vx - 8.5) / 2.5))
	var pivot_blend := u_hp_raw * (1.0 - ov_pivot_gate) if (active_mode == TestMode.AUTOCROSS and launch_state != LaunchState.LAUNCHING and ekf_vx > 1.2 and mu_avg >= 0.85 and not limp_home_active and brake_in <= 0.05) else 0.0
	u_tight_hp = maxf(u_tight_hp, pivot_blend if active_mode == TestMode.AUTOCROSS else 0.0)
	var u_kus := clampf((ekf_vx - _mc_kus_vx_low) / maxf(_mc_kus_vx_high - _mc_kus_vx_low, 0.5), 0.0, 1.0)
	var kus_eff := lerpf(_mc_kus_low_rad_g, _mc_kus_rad_g, u_kus)
	var r_lim_scale_eff := lerpf(_mc_r_lim_low_scale, _mc_r_lim_scale, u_kus)
	var delta_alpha_rel := absf(_alpha_f_rel) - absf(_alpha_r_rel)
	# Candidate B (kept): steady front-limited understeer detection (beta_dyn = -alpha_r in steady cornering;
	# true stability check is low |beta_dot|, low |delta_dot| non-slalom, and |alpha_f| > |alpha_r|).
	var stable_understeer := (_mc_delta_alpha_eq_gain > 1.0) and (delta_alpha_rel > 0.004) and (absf(beta_dyn) < _RAD_7_0) and (absf(_beta_dot) < 0.14) and (not limp_home_active) and (absf(delta_dot) < 0.35) and (_turnin_timer_s <= 0.02) and (ekf_vx > 9.0)
	if active_mode == TestMode.AUTOCROSS and stable_understeer:
		var b_us_u := clampf((delta_alpha_rel - 0.004) / 0.025, 0.0, 1.0) * clampf((0.14 - absf(_beta_dot)) / 0.08, 0.0, 1.0)
		kus_eff = lerpf(kus_eff, 0.10 * kus_eff, b_us_u)
		r_lim_scale_eff = lerpf(r_lim_scale_eff, 1.15, b_us_u)
	var r_ref_std := (ekf_vx * delta_filt) / (wb + kus_eff * ekf_vx * ekf_vx) if ekf_vx > 1.0 else 0.0
	var tan_d_hp := tan(clampf(delta_filt, -0.55, 0.55))
	var r_ack_hp := (ekf_vx * tan_d_hp) / (wb + 0.03 * _mc_kus_low_rad_g * ekf_vx * ekf_vx) if ekf_vx > 1.0 else 0.0
	var hp_gain_eff := lerpf(1.0, _mc_hairpin_pivot_gain, u_tight_hp)
	var r_ref_ss := lerpf(r_ref_std, r_ack_hp * hp_gain_eff, u_tight_hp)
	var g_lat_lim := 9.81 + (0.95 if (active_mode == TestMode.AUTOCROSS and stable_understeer) else 0.35) * maxf(g_eff - 9.81, 0.0)
	var r_lim_raw := (r_lim_scale_eff * mu_avg * g_lat_lim) / maxf(ekf_vx, 3.0)
	if u_tight_hp > 0.0:
		var r_hp_cap := (1.18 * _mc_hairpin_pivot_gain * maxf(mu_avg, 1.35) * g_lat_lim) / maxf(ekf_vx, 3.0)
		var r_lim_hp := minf(maxf(r_lim_raw, absf(r_ack_hp) * 1.10 * _mc_hairpin_pivot_gain), r_hp_cap)
		r_lim_raw = lerpf(r_lim_raw, r_lim_hp, u_tight_hp)
	if _r_lim_filt <= 0.0:
		_r_lim_filt = r_lim_raw
	else:
		_r_lim_filt = lerpf(_r_lim_filt, r_lim_raw, _a_rlim_45)
	var r_ref := clampf(r_ref_ss, -_r_lim_filt, _r_lim_filt)
	# atenua r_ref si |alpha_f| > 10 deg y bloquea x deriva dinamica pa no pelear contra kb_eff*beta
	var low_spd_hairpin_dry: bool = (active_mode == TestMode.AUTOCROSS) and (ekf_vx < 11.5) and (mu_avg >= 0.95)
	if absf(alpha_f) > _RAD_10_0:
		var af_exc := (absf(alpha_f) - _RAD_10_0) / _RAD_7_0
		var af_cut_max := 0.10 if low_spd_hairpin_dry else 0.35
		r_ref *= clampf(1.0 - af_cut_max * af_exc, 1.0 - af_cut_max, 1.0)
	var b_ref_chk: float = absf(beta_dyn) if low_spd_hairpin_dry else absf(beta)
	var b_gate_start: float = lerpf(_RAD_2_3, _RAD_4_5, u_tight_hp)
	if b_ref_chk > b_gate_start:
		var b_gate := clampf(1.0 - (b_ref_chk - b_gate_start) / _RAD_3_8, 0.62, 1.0)
		r_ref *= b_gate
	_r_ref_prev = r_ref

	# feedforward unipolar desde delta_dot (0 en t=0 y giro estacionario) + boost en baja velocidad
	var r_gain_ss := ekf_vx / (wb + kus_eff * ekf_vx * ekf_vx) if ekf_vx > 1.0 else 0.0
	var r_ref_dot := clampf(r_gain_ss * delta_dot, -12.0, 12.0)
	var ff_spd_scale := lerpf(_mc_k_ff_low_mult, 1.0, u_kus) * clampf((ekf_vx - 1.5) / 2.5, 0.35, 1.0)
	var mz_ff := 0.0 if is_first_tick else clampf(_mc_yaw_accel_ff_gain * _mc_k_ff * _mc_yaw_agility_gain * izz * r_ref_dot * ff_spd_scale, -450.0, 450.0)
	# Regime 5: phase-lead compensation delta_dot_lead_gain * delta_dot for fast chicanes & slaloms (vx >= 11.5 m/s)
	if not is_first_tick and u_hi_spd > 0.01 and _mc_delta_dot_lead_gain > 0.0:
		mz_ff += clampf(u_hi_spd * _mc_delta_dot_lead_gain * delta_dot, -240.0, 240.0)
	# Regime 2: direct feedforward Mz_pivot_ff scaled by hairpin_pivot_gain
	var mz_pivot_ff := signf(d_norm_raw) * pivot_blend * _mc_hairpin_pivot_gain * 340.0
	mz_ff += mz_pivot_ff
	var kr := _interp_1d(ekf_vx, vx_bp, kr_tbl) * _mc_yaw_agility_gain
	var kb := _interp_1d(ekf_vx, vx_bp, kb_tbl)
	var kbd := _interp_1d(ekf_vx, vx_bp, kbd_tbl)
	# barrera c1 continua de deriva (en iso/ros +y izq: sobreviraje izq => beta < 0 => +kb*beta frena deriva)
	var beta_exc := maxf(absf(beta) - _RAD_1_4, 0.0)
	var kb_eff := kb * (1.0 + 55.0 * beta_exc)
	var kbd_eff := kbd * (1.0 + 28.0 * beta_exc)
	var err_r := r_ref - wz_corr
	if mz_sat_ratio > 0.10 or absf(beta) > _RAD_2_2 or absf(delta_dot) > 1.0 or (_mz_int * err_r < -1.0) or (delta_filt * wz_corr < -0.005):
		_mz_int *= _e_mz_45
	else:
		_mz_int = clampf(_mz_int + 85.0 * err_r * dt, -180.0, 180.0)
	var mz_soft_b := compute_soft_beta_barrier_nm(beta, _beta_dot, wz_corr, ekf_vx)
	var mz_fb := -kr * (wz_corr - r_ref) + kb_eff * beta + kbd_eff * _beta_dot + _mz_int + mz_soft_b
	if _mc_delta_alpha_eq_gain > 1.0 and active_mode == TestMode.AUTOCROSS and (not limp_home_active) and ekf_vx > 9.0 and absf(delta_filt) > _RAD_2_0 and brake_in < 0.05:
		# Candidate B (kept): Curvature tracking (rho = wz / vx) + Delta_alpha = |alpha_f| - |alpha_r| regulation
		var mz_curv := 0.0
		if fx_req > 50.0 and absf(beta_dyn) < _RAD_5_0 and absf(_beta_dot) < 0.14 and _turnin_timer_s <= 0.02:
			var rho_meas := wz_corr / maxf(ekf_vx, 2.5)
			var rho_ref := r_ref / maxf(ekf_vx, 2.5)
			var rho_err := rho_ref - rho_meas
			var ax_curv_u := clampf(absf(_ax_long_filt) / 6.0, 0.0, 1.0)
			mz_curv = clampf(100.0 * ax_curv_u * ekf_vx * rho_err, -120.0, 120.0)
		var sgn_turn_b := signf(delta_filt)
		var da_dd_ok := clampf((0.35 - absf(delta_dot)) / 0.20, 0.0, 1.0) if (_turnin_timer_s <= 0.02) else 0.0
		var da_bd_ok := clampf((0.14 - absf(_beta_dot)) / 0.08, 0.0, 1.0)
		var da_b_ok := clampf((_RAD_7_0 - absf(beta_dyn)) / _RAD_2_0, 0.0, 1.0)
		var mz_dalpha := 0.0
		if delta_alpha_rel > 0.004 and da_dd_ok > 0.01 and da_bd_ok > 0.01 and da_b_ok > 0.01:
			mz_dalpha = clampf(_mc_delta_alpha_eq_gain * (delta_alpha_rel - 0.004) * sgn_turn_b, -340.0, 340.0) * da_dd_ok * da_bd_ok * da_b_ok
		elif delta_alpha_rel < -0.012 and absf(beta_dyn) > _RAD_2_0:
			mz_dalpha = clampf(1200.0 * (delta_alpha_rel + 0.012) * signf(wz_corr), -240.0, 240.0)
		mz_fb += mz_curv + mz_dalpha
	var mz_req := (mz_ff + mz_fb) * _mc_mz_scale

	# Regimen 2: proteccion activa contra sobreviraje y deslizamiento trasero en slalom / alta velocidad
	var sgn_wz := signf(wz_corr)
	var over_wz := 0.0
	if absf(wz_corr) > 0.12 and (wz_corr * r_ref >= -0.05 or absf(r_ref) < 0.06):
		over_wz = maxf(absf(wz_corr) - absf(r_ref), 0.0)
	var rear_slide_ang := maxf(-beta_dyn * sgn_wz, 0.0) if absf(wz_corr) > 0.12 else 0.0
	var rear_slide_rate := maxf(-_beta_dot * sgn_wz, 0.0) if absf(wz_corr) > 0.12 else 0.0
	var rear_sliding_hi := (u_hi_spd > 0.01) and (over_wz > 0.03 or rear_slide_ang > _RAD_1_4 or rear_slide_rate > 0.12)
	if stable_understeer and active_mode == TestMode.AUTOCROSS and rear_slide_rate <= 0.12 and absf(beta) < _RAD_2_5:
		# Do not misclassify front-limited steady understeer (|alpha_f| > |alpha_r|, low beta_dot) as rear sliding
		rear_sliding_hi = false
	if active_mode == TestMode.AUTOCROSS and (not limp_home_active) and rear_sliding_hi:
		if mz_req * sgn_wz > 0.0:
			mz_req *= (1.0 - u_hi_spd)
		_mz_int *= _e_mz_45
		var mz_stab_mag := _mc_slalom_damp_gain * over_wz \
			+ _mc_rear_slide_gain * maxf(rear_slide_ang - _RAD_1_4, 0.0) \
			+ (0.35 * _mc_rear_slide_gain) * maxf(rear_slide_rate - 0.10, 0.0)
		mz_req -= sgn_wz * u_hi_spd * minf(mz_stab_mag, 750.0)

	# ajustes x modo de prueba sobre fx_req y mz_req
	var fx_cmd := fx_req
	var fx_max_drv := minf(4.0 * t_motor_peak * gear_ratio / maxf(r_eff, 0.05), (p_bms_max * eta_drive) / maxf(absf(ekf_vx), 1.0))
	var u_th_raw := clampf(fx_req / maxf(fx_max_drv, 100.0), 0.0, 1.0)
	if fx_cmd > 0.0 and _mc_th_corner_exp > 1.001 and absf(delta_filt) > 0.08 and not (launch_state == LaunchState.LAUNCHING and absf(d_norm_raw) < 0.10):
		var u_th := clampf(fx_cmd / maxf(fx_max_drv, 100.0), 0.0, 1.0)
		fx_cmd = shape_corner_throttle(u_th, absf(delta_filt)) * fx_max_drv
	var warmer_qp_active := false
	var warmer_pitch_des := 0.0
	if active_mode == TestMode.SKIDPAD:
		# SKIDPAD_MAX_ATTACK_V2: neutral steer K_us -> 0, nonlinear ay FF, PI+D yaw, beta* damping, ballistic Figure-8 crossover
		var r_lim_skid := 1.00 * (mu_avg * g_eff) / maxf(ekf_vx, 4.0)
		var r_lat_skid := 1.00 * r_lim_skid
		var sgn_turn := signf(delta_filt)
		var eq_dd_ok := clampf(1.0 - (absf(delta_dot) - 0.040) / 0.060, 0.0, 1.0)
		var eq_wz_ok := clampf((1.02 * r_lat_skid - absf(wz_corr)) / 0.035, 0.0, 1.0)
		var eq_blend := eq_dd_ok * eq_wz_ok
		var alpha_eq := clampf(_mc_alpha_eq_gain * (absf(alpha_f) - absf(alpha_r)) * sgn_turn, -280.0, 280.0)
		mz_req += alpha_eq * eq_blend
		# Nonlinear steady-state yaw moment feedforward coupled to lateral acceleration ay
		var ay_util_sk := clampf(absf(ay_meas) / maxf(mu_avg * g_eff, 2.0), 0.0, 1.25)
		if absf(_mc_skidpad_neutral_trim) > 1e-3 and eq_dd_ok > 0.01:
			mz_req += clampf(_mc_skidpad_neutral_trim * ay_meas * eq_dd_ok, -320.0, 320.0)
		if _mc_sk_mz_ff_gain > 0.0 and eq_dd_ok > 0.01 and ekf_vx > 6.0:
			var ff_nl_sk := _mc_sk_mz_ff_gain * (1.0 + 0.55 * ay_util_sk * ay_util_sk)
			mz_req += clampf(ff_nl_sk * ay_meas * eq_dd_ok, -360.0, 360.0)
		# Yaw-rate error derivative damping (D-term of PI+D)
		var r_err_dot_sk := clampf((err_r - _r_err_prev) / maxf(dt, 1e-4), -25.0, 25.0)
		_r_err_dot_filt = lerpf(_r_err_dot_filt, r_err_dot_sk, _a_om_15)
		_r_err_prev = err_r
		if _mc_sk_kd_yaw > 0.0 and not is_first_tick:
			mz_req += clampf(_mc_sk_kd_yaw * _r_err_dot_filt, -180.0, 180.0)
		# Active sideslip damping centered on optimal beta* (matching simultaneous 4-wheel alpha_peak(Fz_i))
		var fz_r_avg_sk := 0.5 * (fz_est[2] + fz_est[3])
		var alpha_r_pk_rad := deg_to_rad(4.6 + 1.4 * clampf((fz_r_avg_sk - 500.0) / 800.0, 0.0, 1.0))
		var beta_star_sk := clampf(beta_kin - (alpha_r_pk_rad * minf(ay_util_sk, 1.0) * signf(wz_corr) if absf(wz_corr) > 0.08 else 0.0), -0.05236, 0.05236)
		var beta_err_sk := beta - beta_star_sk
		last_mz_beta_damp = 0.0
		if _mc_sk_beta_damp > 0.0 and ekf_vx > 6.0:
			var mz_bd_sk := clampf(0.35 * _mc_sk_beta_damp * beta_err_sk + _mc_sk_beta_damp * _beta_dot, -280.0, 280.0)
			mz_req += mz_bd_sk
			last_mz_beta_damp = mz_bd_sk
		# Grip-adaptive Ballistic Figure-8 Crossover (gated to vx < 14.5 m/s)
		var is_crossover := (ekf_vx < 14.5) and (absf(delta_dot) > 0.35) and (delta_filt * delta_dot < -0.015 or delta_filt * ay_meas < -0.4)
		last_is_crossover = is_crossover
		last_mz_crossover = 0.0
		if is_crossover:
			var mu_adapt := clampf(mu_avg / 1.25, 0.45, 1.10)
			var cr_lim := (_mc_sk_cross_boost_nm if _mc_sk_cross_boost_nm > 1.0 else 220.0) * mu_adapt
			var cross_boost := clampf(mu_adapt * (95.0 * delta_dot + 145.0 * (r_ref - wz_corr)), -cr_lim, cr_lim)
			mz_req += cross_boost
			last_mz_crossover = cross_boost
		else:
			var ov_th := 1.04 * r_lat_skid
			var ov_wz := clampf((absf(wz_corr) - ov_th) / (0.04 * r_lat_skid), 0.0, 1.0)
			if ov_wz > 0.0:
				var sgn_w := signf(wz_corr)
				if mz_req * sgn_w > 0.0:
					mz_req *= (1.0 - 0.5 * ov_wz)
				mz_req -= 260.0 * ov_wz * sgn_w
		if absf(ay_meas) > mu_avg * 9.81 * 1.18 and fx_cmd > 0.0:
			fx_cmd *= 0.94
		elif fx_cmd > 0.0 and absf(delta_dot) < 0.25:
			fx_cmd *= 1.35
		_accel_elapsed_s += dt
	elif active_mode == TestMode.ACCELERATION:
		if fx_cmd > 50.0:
			_accel_elapsed_s += dt
		else:
			_accel_elapsed_s = 0.0
		# compensa r_load/r_eff en tramo de potencia dc (techo 80 kw exacto en capa 3)
		if fx_cmd > 500.0 and ekf_vx > 12.0:
			fx_cmd *= 1.055 if ekf_vx > 22.0 else 1.038
		# gyro-lock en recta
		if absf(delta_filt) < _RAD_2_0 or (launch_state == LaunchState.LAUNCHING and absf(d_norm_raw) < 0.10):
			mz_req = -_mc_gyro_lock_gain * wz_corr
	elif active_mode == TestMode.AUTOCROSS and is_launch_req and (absf(delta_filt) < _RAD_3_0 or (launch_state == LaunchState.LAUNCHING and absf(d_norm_raw) < 0.10)):
		# Regime 1: Gyro-Lock directional stabilization on AUTOCROSS standstill launch
		var gl_u := 1.0 if (launch_state == LaunchState.LAUNCHING and absf(d_norm_raw) < 0.10) else clampf((_RAD_3_0 - absf(delta_filt)) / _RAD_1_4, 0.0, 1.0)
		mz_req = lerpf(mz_req, -_mc_gyro_lock_gain * wz_corr, gl_u)
	elif active_mode == TestMode.EFFICIENCY:
		# en efficiency el techo p_eff_max_w + desconexion del + regen hacen el ahorro sin frenar ritmo
		pass
	elif active_mode == TestMode.TIRE_WARMER:
		# warmer integrado en qp fista con rampa c1 de 0.4 s y equilibrio termico del/tras
		_warmer_timer_s += dt
		_warmer_phase_t += dt
		if _warmer_timer_s >= 10.0:
			_warmer_timer_s -= 10.0
			_warmer_state = 0
			_warmer_phase_t = 0.0
		var k_tgt_f := _mc_k_warmer_f
		var k_tgt_r := _mc_k_warmer_r
		var v_ref_w := maxf(ekf_vx, 1.5)
		var fz_f := 0.5 * (fz_est[0] + fz_est[1])
		var fz_r := 0.5 * (fz_est[2] + fz_est[3])
		var r_f_load := maxf(0.5 * (r_load_est[0] + r_load_est[1]), 0.15)
		var r_r_load := maxf(0.5 * (r_load_est[2] + r_load_est[3]), 0.15)
		var kz_safe_w := maxf(tire_kz, 20000.0)
		var r_e_f := clampf(r_eff - (fz_f / kz_safe_w) / 3.0, 0.185, r_eff)
		var r_e_r := clampf(r_eff - (fz_r / kz_safe_w) / 3.0, 0.185, r_eff)
		var k_fl := (w_eff[0] * r_e_f - ekf_vx) / v_ref_w
		var k_fr := (w_eff[1] * r_e_f - ekf_vx) / v_ref_w
		var k_rl := (w_eff[2] * r_e_r - ekf_vx) / v_ref_w
		var k_rr := (w_eff[3] * r_e_r - ekf_vx) / v_ref_w
		var k_f_m := 0.5 * (absf(k_fl) + absf(k_fr))
		var k_r_m := 0.5 * (absf(k_rl) + absf(k_rr))
		if ekf_vx > 3.5 and brake_in < 0.05 and not limp_home_active:
			warmer_qp_active = true
			if absf(delta_filt) < _RAD_2_0:
				_mz_int = 0.0
				mz_req = -120.0 * wz_corr
			var e_v_w := 9.722 - ekf_vx
			if (_warmer_state == 2 and _warmer_phase_t >= 0.20) or (_warmer_state == 7 and e_v_w > 0.0):
				_warmer_v_int = clampf(_warmer_v_int + e_v_w * dt, -0.5, 0.5)
			var t_cruise := clampf(500.0 * e_v_w + 280.0 * _warmer_v_int, -160.0, 160.0)
			var df_eq := clampf(1400.0 * (absf(k_fl) - k_f_m), -50.0, 50.0)
			var dr_eq := clampf(1400.0 * (absf(k_rl) - k_r_m), -50.0, 50.0)
			var k_f_neg := -0.5 * (k_fl + k_fr)
			var k_r_neg := -0.5 * (k_rl + k_rr)
			# avanza estado warmer con rampa c1 limpia de 0.4 s pasando por 0 nm (sin latigazo de vel)
			if _warmer_timer_s >= 7.50 and _warmer_state < 3:
				_warmer_state = 3
				_warmer_phase_t = 0.0
			elif _warmer_state == 0 and ((fz_f <= 343.0 and _warmer_phase_t >= 0.020) or _warmer_phase_t >= 0.034):
				_warmer_state = 1
				_warmer_phase_t = 0.0
			elif _warmer_state == 1 and (k_f_neg >= 0.158 or _warmer_phase_t >= 0.060):
				_warmer_state = 2
				_warmer_phase_t = 0.0
			elif _warmer_state == 3 and _warmer_phase_t >= 0.40:
				_warmer_state = 7
				_warmer_phase_t = 0.0

			if _warmer_state == 0:
				_warmer_sgn = 1.0
				var tf_pre := -60.0 if _warmer_phase_t < 0.006 else 285.0
				var tr_pre := minf(312.5, 0.96 * mu_init * fz_r * r_r_load) if _warmer_phase_t < 0.006 else 312.5
				_warmer_trq_des[0] = tf_pre; _warmer_trq_des[1] = tf_pre
				_warmer_trq_des[2] = tr_pre; _warmer_trq_des[3] = tr_pre
				for i in range(4): _warmer_cap_des[i] = 312.5
			elif _warmer_state == 1:
				_warmer_sgn = 1.0
				_warmer_trq_des[0] = -308.0; _warmer_trq_des[1] = -308.0
				_warmer_trq_des[2] = 312.5; _warmer_trq_des[3] = 312.5
				for i in range(4): _warmer_cap_des[i] = 312.5
			elif _warmer_state == 2:
				_warmer_sgn = 1.0
				var u_c1 := clampf(_warmer_phase_t / 0.40, 0.0, 1.0)
				var c1 := u_c1 * u_c1 * (3.0 - 2.0 * u_c1)
				var t_slip_cap_f := 0.960 * mu_init * fz_f * r_f_load
				var t_rise_cap_r := 0.935 * mu_init * fz_r * r_r_load
				var k_f_eff := maxf(k_tgt_f, 0.155)
				_warmer_k_int_f = clampf(_warmer_k_int_f + (k_f_eff - k_f_m) * dt, -0.05, 0.10)
				var w_act := clampf(210.0 + 1300.0 * (k_f_eff - k_f_m) + 900.0 * _warmer_k_int_f, 105.0, maxf(t_slip_cap_f, 155.0))
				_warmer_trq_des[0] = -clampf(w_act - df_eq, 85.0, t_slip_cap_f)
				_warmer_trq_des[1] = -clampf(w_act + df_eq, 85.0, t_slip_cap_f)
				var tc_a2 := clampf(t_cruise, -45.0 * c1, 160.0)
				var r_drv := (0.5 * absf(_warmer_trq_des[0] + _warmer_trq_des[1])) * (r_r_load / r_f_load) + tc_a2
				_warmer_trq_des[2] = clampf(r_drv + dr_eq, -t_rise_cap_r, t_rise_cap_r)
				_warmer_trq_des[3] = clampf(r_drv - dr_eq, -t_rise_cap_r, t_rise_cap_r)
				_warmer_cap_des[0] = t_slip_cap_f; _warmer_cap_des[1] = t_slip_cap_f
				_warmer_cap_des[2] = t_rise_cap_r; _warmer_cap_des[3] = t_rise_cap_r
			elif _warmer_state == 3:
				# rampa c1 simetrica 0.4 s pasando por 0 nm manteniendo crucero 35 km/h
				var u_sw := clampf(_warmer_phase_t / 0.40, 0.0, 1.0)
				var s_sw := 1.0 - 2.0 * (u_sw * u_sw * (3.0 - 2.0 * u_sw))
				_warmer_sgn = 1.0 if s_sw >= 0.0 else -1.0
				var t_cap_f_sw := 0.945 * mu_init * fz_f * r_e_f
				var t_cap_r_sw := 0.945 * mu_init * fz_r * r_e_r
				var t_antag := 190.0 * s_sw
				var tf_sw := clampf(-t_antag + 0.5 * t_cruise, -t_cap_f_sw, t_cap_f_sw)
				var tr_sw := clampf(t_antag * (r_r_load / r_f_load) + 0.5 * t_cruise, -t_cap_r_sw, t_cap_r_sw)
				_warmer_trq_des[0] = tf_sw; _warmer_trq_des[1] = tf_sw
				_warmer_trq_des[2] = tr_sw; _warmer_trq_des[3] = tr_sw
				_warmer_cap_des[0] = t_cap_f_sw; _warmer_cap_des[1] = t_cap_f_sw
				_warmer_cap_des[2] = t_cap_r_sw; _warmer_cap_des[3] = t_cap_r_sw
			else:
				_warmer_sgn = -1.0
				var k_r_tgt_eff := minf(k_tgt_r, 0.165)
				var t_rise_cap_f := 0.960 * mu_init * fz_f * r_e_f
				var t_r_bal := maxf((t_rise_cap_f - maxf(t_cruise, 12.0)) * (r_r_load / r_f_load), 80.0)
				var t_slip_cap_r := minf(0.955 * mu_init * fz_r * r_e_r, t_r_bal)
				_warmer_k_int_r = clampf(_warmer_k_int_r + (k_r_tgt_eff - k_r_m) * dt, -0.05, 0.10)
				var w_act_r := clampf(175.0 + 1200.0 * (k_r_tgt_eff - k_r_m) + 800.0 * _warmer_k_int_r, 75.0, maxf(t_slip_cap_r, 85.0))
				_warmer_trq_des[2] = -clampf(w_act_r - dr_eq, 60.0, t_slip_cap_r)
				_warmer_trq_des[3] = -clampf(w_act_r + dr_eq, 60.0, t_slip_cap_r)
				var f_base := (0.5 * absf(_warmer_trq_des[2] + _warmer_trq_des[3])) * (r_f_load / r_r_load) + t_cruise
				_warmer_trq_des[0] = clampf(f_base + df_eq, -t_rise_cap_f, t_rise_cap_f)
				_warmer_trq_des[1] = clampf(f_base - df_eq, -t_rise_cap_f, t_rise_cap_f)
				_warmer_cap_des[0] = t_rise_cap_f; _warmer_cap_des[1] = t_rise_cap_f
				_warmer_cap_des[2] = t_slip_cap_r; _warmer_cap_des[3] = t_slip_cap_r
			fx_cmd = (_warmer_trq_des[0] + _warmer_trq_des[1]) / r_f_load + (_warmer_trq_des[2] + _warmer_trq_des[3]) / r_r_load
			warmer_pitch_des = (_warmer_trq_des[0] + _warmer_trq_des[1]) - (_warmer_trq_des[2] + _warmer_trq_des[3])

	# estabilidad en recta sobre split-mu: prioriza amortiguacion de guinada y deriva si delta~0
	var is_straight_split := false
	if absf(delta_filt) < _RAD_1_2 and not limp_home_active and not warmer_qp_active:
		if absf(mu_hat[1] - mu_hat[0]) > 0.12 or absf(wz_corr) > 0.03:
			is_straight_split = true
			_mz_int *= _e_mz_45
			mz_req = -950.0 * wz_corr + kb_eff * beta + kbd_eff * _beta_dot + _mz_int

	# limp-home: desactiva tv y recorta par en apoyo lateral alto si falla sensor
	if limp_home_active:
		mz_req = 0.0
		if fx_cmd > 0.0 and absf(ay_meas) > 7.5:
			fx_cmd *= clampf(1.0 - (absf(ay_meas) - 7.5) / 7.0, 0.18, 1.0)

	# regen en coasting proporcional a vx + lift-off regen suave rampado (cortado si |beta| > 3.5 deg)
	var allow_regen := _mc_allow_regen
	var coast_gain := _mc_coast_gain
	var lift_tgt_fx := 0.0
	if allow_regen and _mc_lift_regen_frac > 0.0 and fx_req <= 5.0 and brake_in < 0.02 and ekf_vx > _mc_lift_regen_v_min and not warmer_qp_active and absf(beta) <= _RAD_3_5:
		var f_reg_max_v := minf(4.0 * t_regen_max / maxf(r_eff, 0.05), (p_max_charge_w / maxf(eta_regen, 0.5)) / maxf(ekf_vx, 2.0))
		lift_tgt_fx = -_mc_lift_regen_frac * f_reg_max_v
	if absf(beta) > _RAD_3_5 or fx_req > 5.0 or brake_in >= 0.02:
		_lift_regen_fx_filt = 0.0
	else:
		_lift_regen_fx_filt = lerpf(_lift_regen_fx_filt, lift_tgt_fx, _a_eff_150)
	var is_coasting := (absf(fx_cmd) < 5.0 and brake_in <= 0.02 and ekf_vx > 2.0 and not warmer_qp_active)
	if allow_regen and is_coasting:
		if coast_gain > 0.0:
			fx_cmd = -(coast_gain * ekf_vx * 4.0) / r_eff
		if _lift_regen_fx_filt < -1.0:
			fx_cmd = minf(fx_cmd, _lift_regen_fx_filt)

	# control activo de cabeceo: anti-dive desde t=0 ms en frenada y anti-squat en accel
	if absf(r_ref_dot) > 1.5:
		_turnin_timer_s = maxf(_turnin_timer_s, 0.16)
	var delta_t_pitch := warmer_pitch_des if warmer_qp_active else (pitch_bias_gain * _ax_filt)
	var rear_bias_u := 0.0
	var is_braking := (brake_in > 0.05 and fx_req <= 0.0)
	if is_braking:
		if lambda_pitch > 0.0:
			var ti_gain := _mc_ti_gain
			delta_t_pitch = -ti_gain * clampf(brake_in / 0.45, 0.4, 1.3)
			if not limp_home_active:
				if absf(ay_meas) < 4.5:
					mz_req = clampf(mz_req, -110.0, 110.0)
				elif brake_in < 0.50 and absf(ay_meas) > 10.5 and (mz_req * signf(delta_filt) > 0.0):
					mz_req = 0.0
		else:
			delta_t_pitch = 0.0
	elif active_mode == TestMode.ACCELERATION:
		var tau_r := _mc_pitch_tau
		var ramp := 1.0 - exp(-_accel_elapsed_s / tau_r)
		var fade_hi := clampf(1.0 - maxf(ekf_vx - 15.0, 0.0) / 6.0, 0.0, 1.0)
		if fx_cmd > 200.0:
			var r_init_bias := clampf(2.0 * (_mc_launch_rear_bias_pct * 0.01) - 1.0, 0.0, 0.6)
			var r_mid_bias := 0.30 # 35% F / 65% R
			var acc_bias := lerpf(r_init_bias, r_mid_bias, ramp) * fade_hi
			delta_t_pitch = -acc_bias * fx_cmd * r_eff
	elif fx_cmd > 0.0 and not warmer_qp_active:
		var launch_spd_u := clampf((14.0 - ekf_vx) / 4.0, 0.0, 1.0)
		if launch_spd_u > 0.0:
			var launch_init_u := clampf((4.5 - ekf_vx) / 2.5, 0.0, 1.0) if (launch_state == LaunchState.LAUNCHING or u_th_raw > 0.70) else 0.0
			var r_launch_target := clampf(2.0 * (_mc_launch_rear_bias_pct * 0.01) - 1.0, 0.0, 0.6)
			var launch_rear_bias := lerpf(lerpf(0.20, 0.34, clampf(_ax_long_filt / 10.0, 0.0, 1.0)), r_launch_target, launch_init_u)
			delta_t_pitch = lerpf(delta_t_pitch, -launch_rear_bias * fx_cmd * r_eff, launch_spd_u)
		if _mc_rear_bias_gain > 0.0:
			rear_bias_u = clampf((absf(ay_meas) - 2.5) / 8.5, 0.0, 1.0) * clampf((16.5 - ekf_vx) / 3.0, 0.0, 1.0)
			if rear_bias_u > 0.0:
				delta_t_pitch = lerpf(delta_t_pitch, -_mc_rear_bias_gain * fx_cmd * r_eff, rear_bias_u)
		# Candidate A (kept): relax rear bias at high steering angle to exploit front Fx_f * sin(delta) during steady/apex (not WOT exit)
		if _mc_centripetal_fx_gain > 0.1 and (active_mode == TestMode.AUTOCROSS or active_mode == TestMode.SKIDPAD) and absf(delta_filt) > 0.12 and delta_alpha_rel > 0.0 and absf(beta_dyn) < _RAD_5_0 and absf(_beta_dot) < 0.18 and absf(_ax_long_filt) < 2.5 and u_th_raw < 0.65 and (not rear_sliding_hi):
			var cent_u := clampf((absf(delta_filt) - 0.12) / 0.16, 0.0, 1.0) * clampf((_RAD_5_0 - absf(beta_dyn)) / _RAD_1_4, 0.0, 1.0)
			var t_front_cent := clampf(140.0 * sin(absf(delta_filt)), 0.0, 90.0)
			delta_t_pitch = lerpf(delta_t_pitch, maxf(delta_t_pitch, t_front_cent), cent_u)

	# atenuacion de regen en apoyo lateral fuerte pa evitar lift-off oversteer
	var regen_atten := 1.0
	if fx_cmd < 0.0 and not warmer_qp_active:
		var ay_th := _mc_regen_ay_th
		regen_atten = clampf(1.0 - maxf(absf(ay_meas) - ay_th, 0.0) / 5.0, 0.08, 1.0)
		if absf(ay_meas) > 0.65 * 9.81 and ay_th < 50.0:
			regen_atten = minf(regen_atten, clampf(1.0 - (absf(ay_meas) - 0.65 * 9.81) / 3.2, 0.05, 1.0))
		if is_coasting and regen_atten < 0.999:
			fx_cmd *= regen_atten
			delta_t_pitch *= clampf(regen_atten, 0.25, 1.0)
		if not allow_regen:
			fx_cmd = 0.0

	# transicion fisica de cruce de direccion (contragiro con desfase lpf) sin umbral magico
	var d_lag := absf(delta_raw - delta_filt)
	var cross_rev := (delta_raw * wz_corr < -0.05) and (d_lag > 0.18) and (absf(wz_corr) > 0.65)
	var cross_init := cross_rev and (d_lag > 0.27)
	if cross_rev:
		mz_req = 0.0
	last_wz_ref = r_ref
	last_mz_ff = mz_ff
	last_mz_fb = mz_fb
	last_mz_req = mz_req
	last_pitch_bias_nm = delta_t_pitch

	# capa 3: asignador jerarquico exacto (mz > fx) con fista proyectado (12 iters) + techo 80 kw
	# matriz geometrica ackermann b(d_fl, d_fr) usando radio cargado r_load_est
	var cos_fl := cos(d_fl)
	var sin_fl := sin(d_fl)
	var cos_fr := cos(d_fr)
	var sin_fr := sin(d_fr)
	_qp_bx[0] = cos_fl / r_load_est[0]
	_qp_bx[1] = cos_fr / r_load_est[1]
	_qp_bx[2] = 1.0 / r_load_est[2]
	_qp_bx[3] = 1.0 / r_load_est[3]

	_qp_bmz[0] = (-0.5 * tf * cos_fl + a_lf * sin_fl) / r_load_est[0]
	_qp_bmz[1] = (0.5 * tf * cos_fr + a_lf * sin_fr) / r_load_est[1]
	_qp_bmz[2] = (-0.5 * tr) / r_load_est[2]
	_qp_bmz[3] = (0.5 * tr) / r_load_est[3]
	var sgn_turn_qp := signf(delta_filt)
	_qp_bfy_f[0] = (sgn_turn_qp * sin_fl) / r_load_est[0]
	_qp_bfy_f[1] = (sgn_turn_qp * sin_fr) / r_load_est[1]
	_qp_bfy_f[2] = 0.0
	_qp_bfy_f[3] = 0.0

	# cotas por motor (incl. taper rpm), elipse acoplada con fy_cons_est y tc en ub_i
	var kappa_lim := _mc_kappa_lim
	var omega_base := maxf(base_speed_rpm, 100.0) * (TAU / 60.0)
	var omega_max := maxf(motor_max_rpm, 200.0) * (TAU / 60.0)
	var p_m_peak := t_motor_peak * omega_base
	var eta_acc := lerpf(0.930, 0.994, clampf((ekf_vx - 0.5) / 1.2, 0.0, 1.0)) if (mu_avg >= 0.96 and absf(delta_filt) < _RAD_4_0) else lerpf(0.890, 0.994, clampf((ekf_vx - 0.8) / 1.5, 0.0, 1.0))
	var allow_low_b := (not is_braking) and (not is_straight_split) and (absf(delta_filt) > _RAD_5_0 or mu_avg >= 0.96)
	var low_spd_b := clampf((14.0 - ekf_vx) / 3.0, 0.0, 1.0) if allow_low_b else 0.0
	var dry_cor_stable: bool = (active_mode == TestMode.AUTOCROSS) and (not is_braking) and (not is_straight_split) and (mu_avg >= 0.96) and (absf(beta_dyn) < _mc_beta_soft_rad)
	var eta_base := lerpf(0.880 if dry_cor_stable else 0.850, 0.925 if dry_cor_stable else 0.915, low_spd_b)
	var hi_ay_cut := clampf((absf(ay_meas) - 9.0) / 4.5, 0.0, 1.0) * (1.0 - low_spd_b)
	var sk_init_tr := (active_mode == TestMode.SKIDPAD and _accel_elapsed_s < 0.25)
	var sk_trans := 0.035 if sk_init_tr else 0.0
	var straight_spd_u := clampf((22.0 - ekf_vx) / 4.0, 0.0, 1.0) if dry_cor_stable else low_spd_b
	var dry_launch_u := clampf((3.5 - absf(ay_meas)) / 2.0, 0.0, 1.0) * clampf((mu_avg - 0.96) / 0.15, 0.0, 1.0) * straight_spd_u if (active_mode == TestMode.AUTOCROSS and not is_braking and not is_straight_split) else 0.0
	var eta_auto := lerpf(lerpf(eta_base, 0.875 if dry_cor_stable else 0.785, hi_ay_cut), eta_acc, dry_launch_u)
	var eta_cap := eta_acc if ((active_mode == TestMode.ACCELERATION and absf(ay_meas) < 2.0) or (launch_state == LaunchState.LAUNCHING and not is_straight_split)) else (
		(0.970 - sk_trans) if active_mode == TestMode.SKIDPAD else eta_auto
	)
	var true_split_mu := is_straight_split and (absf(mu_hat[1] - mu_hat[0]) > 0.12 or absf(mu_hat[3] - mu_hat[2]) > 0.12)
	var stab_b := clampf((absf(beta_dyn) - _RAD_2_8) / _RAD_1_4, 0.0, 1.0) if (beta_dyn * wz_corr < 0.0) else 0.0
	var stab_w := clampf((absf(wz_corr - r_ref) - 0.55) / 0.45, 0.0, 1.0) if (absf(beta_dyn) > _RAD_2_8) else 0.0
	var stab_hi := u_hi_spd * clampf(maxf(over_wz / 0.25, maxf((rear_slide_ang - _RAD_1_4) / _RAD_1_4, (rear_slide_rate - 0.12) / 0.35)), 0.0, 1.0)
	var mz_emg := clampf((absf(mz_req) - 100.0) / 140.0, 0.0, 1.0)
	var emg_frac := maxf(maxf(stab_b, stab_w), stab_hi) * mz_emg
	var forbid_antag := _mc_forbid_antag
	var forbid_antag_drv := _mc_forbid_antag_drv
	var us_err_apex := signf(delta_filt) * (r_ref - wz_corr)
	var is_apex_understeer: bool = (active_mode == TestMode.AUTOCROSS or active_mode == TestMode.SKIDPAD) and (not warmer_qp_active) and (not limp_home_active) and (absf(fx_req) < 350.0) and (brake_in <= mech_brake_start) and (ekf_vx > 4.5) and (us_err_apex > 0.04) and (absf(beta_dyn) < _RAD_3_5)
	var apex_u := clampf((us_err_apex - 0.04) / 0.16, 0.25, 1.0) * clampf(1.0 - absf(beta_dyn) / _RAD_3_5, 0.0, 1.0) if is_apex_understeer else 0.0
	# Regimen 2 horquilla cerrada (vx < 11.0 m/s, |delta_norm_raw| > 0.42): entrada/apex y salida
	var hp_asym_active: bool = (u_tight_hp > 0.05) and (_mc_hairpin_regen_max_nm > 1.0) and (not is_braking) and (not warmer_qp_active) and (not limp_home_active) and allow_regen and (ekf_vx > 1.2) and (u_th_raw < 0.45) and (absf(beta_dyn) < _RAD_4_5) and (mz_req * signf(d_norm_raw if absf(d_norm_raw) > 0.1 else delta_filt) > 8.0)
	var hp_asym_u: float = u_tight_hp * clampf((absf(mz_req) - 8.0) / 80.0, 0.50, 1.0) * clampf((0.45 - u_th_raw) / 0.15, 0.0, 1.0) if hp_asym_active else 0.0
	var hp_unwinding: bool = (d_norm_raw * delta_dot < -0.002) or (delta_filt * delta_dot < -0.002)
	var hp_exit_active: bool = (u_tight_hp > 0.05 or pivot_blend > 0.05) and (u_th_raw >= 0.45) and hp_unwinding and (not is_braking) and (not warmer_qp_active) and (not limp_home_active) and (_mc_exit_traction_blend > 0.01)
	# Regimen 3 trail-braking yaw assist (brake > 0.15 & |delta_norm_raw| > 0.22)
	var tb_blend: float = _smoothstep01((brake_in - 0.15) / 0.15) * _smoothstep01((absf(d_norm_raw) - 0.22) / 0.15) * clampf((ekf_vx - 2.5) / 2.0, 0.0, 1.0)
	var pro_turn_tb: bool = (not warmer_qp_active) and is_braking and (fx_cmd <= 5.0) and (brake_in <= mech_brake_start or tb_blend > 0.10) and (_mc_tb_turn_gain > 0.0) and allow_regen and ((us_err_apex > 0.05) or tb_blend > 0.10) and (absf(beta_dyn) < _RAD_4_5) and (ekf_vx > 2.5) and (not limp_home_active)
	tc_active = false
	var any_abs := false

	for i in range(4):
		_qp_t_abs[i] = 0.0
		_qp_w_abs[i] = 0.0
		var om_m := absf(w_eff[i]) * gear_ratio
		var t_env := t_motor_peak
		if om_m > omega_base:
			t_env = minf(t_motor_peak, p_m_peak / om_m)
		var th_u := clampf((motor_temps_c[i] - thermal_derate_start_c) / maxf(thermal_derate_max_c - thermal_derate_start_c, 5.0), 0.0, 1.0)
		var th_scale := lerpf(1.0, 0.35, th_u * th_u)
		var t_max_drv := t_env * gear_ratio * th_scale
		if om_m > motor_taper_ratio * omega_max:
			t_max_drv *= clampf((omega_max - om_m) / maxf((1.0 - motor_taper_ratio) * omega_max, 1.0), 0.0, 1.0)
		var j_ax_e := (1 - i) if i < 2 else (5 - i)
		var mu_ell := minf(minf(mu_hat[i], mu_hat[j_ax_e]) if absf(delta_filt) < _RAD_4_0 else mu_hat[i], mu_max)
		# en bajo mu en recta a baja vel recorta elipse y slip pa no saturar durante relajacion
		var split_low_i := (mu_ell < 1.02) and (absf(delta_filt) < _RAD_4_0) and (fx_cmd > 0.0 and not is_braking) and (launch_state != LaunchState.LAUNCHING)
		var eta_split_i := lerpf(0.48, 0.86, clampf((ekf_vx - 6.0) / 10.0, 0.0, 1.0)) if split_low_i else 1.0
		var f_cap_i := (eta_cap / 0.94) * _mc_friction_ellipse_margin * eta_split_i * mu_ell * fz_est[i]
		var fy_cap_ratio := 0.86 if (dry_cor_stable and fx_cmd > 50.0 and not true_split_mu) else 0.945
		var fy_eff_i := minf(fy_cons_est[i], fy_cap_ratio * f_cap_i)
		var fx_max_i := sqrt(maxf(0.0, f_cap_i * f_cap_i - fy_eff_i * fy_eff_i))
		var t_ellipse := fx_max_i * r_load_est[i]
		# suma inercia rotacional al lanzar recto pa clavar fuerza util de contacto
		var inert_scale := 0.91 if (active_mode == TestMode.ACCELERATION or (launch_state == LaunchState.LAUNCHING and not is_straight_split and absf(delta_filt) < _RAD_8_0)) else (0.90 * clampf((3.5 - absf(ay_meas)) / 2.0, 0.0, 1.0) * straight_spd_u * clampf((mu_avg - 0.96) / 0.15, 0.0, 1.0) if (active_mode == TestMode.AUTOCROSS and not is_straight_split) else 0.0)
		var t_inert_ff := (inert_scale * i_wheel_eff * maxf(_ax_long_filt, 0.0) / r_load_est[i]) if (fx_cmd > 0.0 and not is_braking) else 0.0
		var ub_i := minf(t_max_drv, t_ellipse + t_inert_ff)
		var lb_i := -minf(t_regen_max * regen_atten, t_ellipse) if allow_regen else 0.0

		# proyeccion vel esquina a lo largo de rueda guiada pa abs y tc precisos
		var trk_u := tf if i < 2 else tr
		var side_u := -0.5 if (i % 2 == 0) else 0.5
		var v_w_long_i := ekf_vx + side_u * trk_u * wz_corr
		if i < 2:
			var cos_w := cos_fl if i == 0 else cos_fr
			var sin_w := sin_fl if i == 0 else sin_fr
			v_w_long_i = v_w_long_i * cos_w + (ekf_vy + a_lf * wz_corr) * sin_w
		var v_cu_x := maxf(v_w_long_i, 0.8)
		var ki_ub := (w_eff[i] * r_eff - v_cu_x) / maxf(v_cu_x, 1.2)
		var wheel_bad_i := fault_wheel_mask[i] or _fault_auto[i] or is_nan(w_omegas[i])

		if warmer_qp_active:
			var t_cap_w := minf(t_max_drv, _warmer_cap_des[i])
			ub_i = t_cap_w
			lb_i = -t_cap_w
			_qp_t_abs[i] = clampf(_warmer_trq_des[i], lb_i, ub_i)
			_qp_w_abs[i] = 45.0
		elif is_braking:
			# abs electronico calibrado al pico hoosier + regen-first sin cap de 95 nm en efficiency
			var t_reg_th := t_regen_max * th_scale
			var reg_floor := -minf(t_reg_th, t_ellipse) if allow_regen else 0.0
			var hyd_k := 0.76 if is_straight_split else 0.88
			var hyd_exc := t_hyd_est[i] - hyd_k * t_ellipse
			var slip_abs := maxf(0.0, (-ki_ub - 0.065) * 2800.0) if not wheel_bad_i else 0.0
			var t_abs_req := clampf(maxf(hyd_exc, slip_abs), 0.0, maxf(0.85 * t_hyd_est[i], 55.0 if slip_abs > 10.0 else 0.0))
			if t_abs_req > 2.0 and (t_hyd_est[i] > 1.0 or slip_abs > 10.0):
				ub_i = t_abs_req
				lb_i = 0.85 * t_abs_req
				_qp_t_abs[i] = t_abs_req
				_qp_w_abs[i] = 12.0
				any_abs = true
			elif active_mode == TestMode.EFFICIENCY or forbid_antag or (active_mode == TestMode.AUTOCROSS and not true_split_mu):
				var atten_brk := clampf(1.0 - maxf(absf(ay_meas) - 8.5, 0.0) / 8.5, 0.45, 1.0) if (active_mode == TestMode.EFFICIENCY or active_mode == TestMode.AUTOCROSS) else clampf(1.0 - maxf(absf(ay_meas) - 4.0, 0.0) / 5.5, 0.04, 1.0)
				ub_i = 0.0
				var reg_eff := -minf(t_reg_th * atten_brk, t_ellipse) if allow_regen else 0.0
				lb_i = reg_eff if allow_regen else 0.0
				if active_mode == TestMode.EFFICIENCY and allow_regen:
					_qp_t_abs[i] = clampf(0.85 * reg_eff, lb_i, 0.0)
					_qp_w_abs[i] = 10.0
			else:
				if true_split_mu:
					ub_i = minf(0.65 * t_hyd_est[i], 140.0)
				else:
					ub_i = 0.0
				lb_i = maxf(reg_floor, t_hyd_est[i] - 0.85 * t_ellipse)
				lb_i = minf(lb_i, -minf(18.0, 0.30 * t_ellipse)) if (allow_regen and t_hyd_est[i] < 0.95 * t_ellipse) else minf(lb_i, 0.0)
		elif active_mode == TestMode.EFFICIENCY or forbid_antag:
			if fx_cmd >= 0.0:
				lb_i = 0.0
			else:
				ub_i = 0.0
		elif forbid_antag_drv:
			if fx_cmd >= 0.0:
				if hp_asym_active and (_qp_bmz[i] * mz_req < 0.0):
					var t_hp_reg := clampf(_mc_hairpin_regen_max_nm * _mc_hairpin_pivot_gain * hp_asym_u, 0.0, minf(t_regen_max, 0.85 * t_ellipse))
					lb_i = -t_hp_reg
					if u_th_raw < 0.45:
						var ub_cut_u := clampf(hp_asym_u * clampf((0.45 - u_th_raw) / 0.12, 0.0, 1.0), 0.0, 1.0)
						ub_i = minf(ub_i, lerpf(ub_i, -0.45 * t_hp_reg, ub_cut_u))
						_qp_t_abs[i] = -t_hp_reg
						_qp_w_abs[i] = 2.5 * ub_cut_u
				elif hp_asym_active and (_qp_bmz[i] * mz_req > 0.0) and u_th_raw < 0.45 and ub_i > 50.0:
					var t_out_target := clampf(lerpf(70.0, 160.0, hp_asym_u), 70.0, ub_i)
					lb_i = maxf(lb_i, minf(t_out_target * 0.65, ub_i - 1.0))
					_qp_t_abs[i] = t_out_target
					_qp_w_abs[i] = 1.5 * hp_asym_u
				elif hp_exit_active and (_qp_bmz[i] * signf(d_norm_raw if absf(d_norm_raw) > 0.1 else delta_filt) > 0.0):
					lb_i = maxf(lb_i, minf(ub_i * 0.65 * _mc_exit_traction_blend, maxf(ub_i - 1.0, 0.0)))
					_qp_t_abs[i] = ub_i
					_qp_w_abs[i] = 1.2 * _mc_exit_traction_blend
				elif hp_exit_active and (_qp_bmz[i] * signf(d_norm_raw if absf(d_norm_raw) > 0.1 else delta_filt) < 0.0):
					lb_i = maxf(lb_i, 0.0)
				elif is_apex_understeer and allow_regen and i >= 2 and (_qp_bmz[i] * mz_req < 0.0):
					lb_i = -minf(28.0 * apex_u, minf(t_regen_max, 0.75 * t_ellipse))
				elif (active_mode == TestMode.AUTOCROSS or active_mode == TestMode.SKIDPAD) and emg_frac > 0.0 and (_qp_bmz[i] * mz_req < 0.0):
					lb_i = -minf(95.0 * emg_frac, t_ellipse)
				else:
					lb_i = 0.0
			elif not is_apex_understeer and not hp_asym_active:
				ub_i = 0.0

		if pro_turn_tb and not any_abs:
			ub_i = 0.0
			var rear_antilock_trip := (i >= 2) and ((-ki_ub > 0.085) or (w_eff[i] * r_eff < 0.86 * v_cu_x))
			if rear_antilock_trip:
				lb_i = 0.0
				ub_i = 0.0
				_qp_t_abs[i] = 0.0
				_qp_w_abs[i] = 0.0
			elif _qp_bmz[i] * mz_req < 0.0:
				var front_ok: bool = (i >= 2) or (fy_cons_est[i] < 0.80 * mu_hat[i] * fz_est[i])
				if front_ok:
					var tb_reg := minf(t_regen_max * _mc_tb_turn_gain * maxf(tb_blend, 0.35), t_ellipse)
					lb_i = minf(lb_i, -tb_reg)
					if tb_blend > 0.15:
						_qp_t_abs[i] = -tb_reg
						_qp_w_abs[i] = 2.0 * tb_blend
			elif not is_braking:
				lb_i = 0.0
		elif is_apex_understeer and (not any_abs) and (not is_braking) and i >= 2:
			if allow_regen and (_qp_bmz[i] * mz_req < 0.0):
				lb_i = minf(lb_i, -minf(28.0 * apex_u, minf(t_regen_max, 0.75 * t_ellipse)))
			elif _qp_bmz[i] * mz_req > 0.0:
				ub_i = minf(maxf(ub_i, minf(85.0 * apex_u, 0.85 * t_ellipse)), minf(t_max_drv, t_ellipse))

		if active_mode == TestMode.ACCELERATION or (launch_state == LaunchState.LAUNCHING and not warmer_qp_active):
			# Closed-loop optimal slip kappa_opt(mu_hat[i]) across all 4 wheels during LAUNCHING / ACCELERATION
			if fx_cmd > 0.0 and not wheel_bad_i:
				var v_acc := maxf(v_w_long_i, 0.0)
				var ki_acc := (w_eff[i] * r_eff - v_acc) / maxf(v_acc, 1.2 if launch_state == LaunchState.LAUNCHING else 0.8)
				var dv_spin_launch := w_eff[i] * r_eff - v_acc
				var skip_standstill_tc := (launch_state == LaunchState.LAUNCHING) and (ekf_vx < 3.5) and (dv_spin_launch < 2.2)
				var kappa_opt_i := _mc_kappa_lim * clampf(mu_hat[i] / maxf(mu_init, 1.0), 0.55, 1.15)
				if active_mode == TestMode.ACCELERATION and v_acc < 0.55:
					var w_lim_0 := (v_acc + kappa_opt_i * maxf(v_acc, 0.92)) / r_eff
					var t_arr := maxf(_fx_est_prev[i], 0.0) * r_load_est[i] + i_wheel_eff * maxf(w_lim_0 - w_eff[i], 0.0) / maxf(dt, 0.005)
					var blend_arr := clampf(v_acc / 0.55, 0.0, 1.0)
					ub_i = minf(ub_i, lerpf(maxf(t_arr, minf(t_ellipse, 140.0)), ub_i, blend_arr * blend_arr))
				var k_lim_launch := lerpf(maxf(kappa_opt_i, 0.22), kappa_opt_i, clampf((ekf_vx - 3.5) / 2.5, 0.0, 1.0)) if launch_state == LaunchState.LAUNCHING else kappa_opt_i
				if (not skip_standstill_tc) and ki_acc > k_lim_launch:
					var tc_min := 0.65 if launch_state == LaunchState.LAUNCHING else 0.12
					var tc_scale := maxf(1.0 - 4.4 * (ki_acc - k_lim_launch), tc_min)
					ub_i = minf(ub_i, tc_scale * t_ellipse)
					tc_active = true
		elif (not warmer_qp_active) and fx_cmd > 0.0 and not wheel_bad_i:
			# techo progresivo suave + cero cortes espurios en arranque seco cerca de 0 m/s + proteccion split-mu
			var v_tc_den := maxf(v_w_long_i, 1.4)
			var ki_tc := (w_eff[i] * r_eff - maxf(v_w_long_i, 0.8)) / v_tc_den
			var dry_launch_tc := (ekf_vx < 6.0) and (not split_low_i) and (mu_avg >= 0.96) and (absf(delta_filt) < _RAD_4_0)
			var dv_spin_i := w_eff[i] * r_eff - maxf(v_w_long_i, 0.0)
			var skip_zero_spd_tc := dry_launch_tc and (ekf_vx < 3.5) and (dv_spin_i < 2.5)
			var k_mu_scale_i := clampf(mu_hat[i] / maxf(mu_init, 1.0), 0.50, 1.10)
			var kappa_lim_i := kappa_lim * k_mu_scale_i
			var k_lim_i := (0.65 * kappa_lim_i) if split_low_i else (lerpf(0.24, kappa_lim_i, clampf((ekf_vx - 3.5) / 2.5, 0.0, 1.0)) if dry_launch_tc else kappa_lim_i)
			var k_trig_i := k_lim_i if dry_launch_tc else (0.95 * k_lim_i)
			if (not skip_zero_spd_tc) and ki_tc > k_trig_i:
				var frac_slip := clampf((ki_tc - k_trig_i) / maxf(0.55 * k_lim_i, 0.035), 0.0, 1.0)
				var t_tc_ub := maxf(0.65 * t_ellipse, 40.0)
				var ub_tc := minf(ub_i, lerpf(ub_i, t_tc_ub, frac_slip))
				if ub_tc < ub_i - 1e-3:
					tc_active = true
				ub_i = ub_tc
			var dw_ax_tc := w_eff[i] - w_eff[j_ax_e]
			if ekf_vx > 2.0 and absf(delta_filt) < _RAD_6_0 and dw_ax_tc > 1.2 and ki_tc > 0.045 and _om_dot_filt[i] > 45.0:
				var t_road_i := clampf(0.88 * maxf(_fx_est_prev[i], 110.0) * r_load_est[i], 45.0, ub_i)
				var sp_u := clampf((dw_ax_tc - 1.2) / 1.5, 0.0, 1.0)
				ub_i = minf(ub_i, lerpf(ub_i, t_road_i, sp_u))
				tc_active = true

		# Regime 4: SKIDPAD_MAX_ATTACK_V2 steady-state positive inner split (T_out > T_in > 0), crossover regen allowed
		var sk_ay_th := 6.5 if sk_init_tr else 9.0
		if active_mode == TestMode.SKIDPAD and (not is_braking) and fx_cmd >= 0.0:
			var sk_cross_rev := (ekf_vx < 14.5) and (absf(delta_dot) >= 0.35) and (delta_filt * delta_dot < -0.015 or delta_filt * ay_meas < -0.4)
			if not sk_cross_rev:
				lb_i = maxf(lb_i, 0.0)
			else:
				var mu_adapt_cr := clampf(mu_avg / 1.25, 0.45, 1.10)
				var t_cross_reg := _mc_sk_cross_regen_frac * t_regen_max * mu_adapt_cr
				lb_i = maxf(lb_i, -minf(t_cross_reg, 0.45 * t_ellipse))
			var is_sk_inner := (ay_meas > sk_ay_th and (i % 2 == 0)) or (ay_meas < -sk_ay_th and (i % 2 == 1))
			if absf(ay_meas) > sk_ay_th and is_sk_inner and (not sk_cross_rev):
				var sk_steady_u := clampf((0.25 - absf(delta_dot)) / 0.15, 0.0, 1.0)
				if i < 2 or sk_steady_u > 0.05:
					var sk_in := clampf((absf(ay_meas) - 6.5) / 2.5, 0.0, 1.0) if sk_init_tr else 1.0
					var target_in_ub := lerpf(0.05 * t_ellipse, 0.28 * t_ellipse, sk_steady_u)
					ub_i = minf(ub_i, lerpf(ub_i, target_in_ub, sk_in))
		elif active_mode == TestMode.AUTOCROSS and (not is_braking) and fx_cmd > 0.0 and i < 2 and absf(delta_filt) > _RAD_7_0 and absf(delta_dot) < 0.35 and absf(beta) < _RAD_2_5 and absf(ay_meas) > 11.80:
			if (i == 0 and ay_meas > 11.80) or (i == 1 and ay_meas < -11.80):
				var in_frac := clampf((absf(ay_meas) - 11.80) / 0.60, 0.0, 1.0)
				ub_i = minf(ub_i, lerpf(ub_i, 0.0, in_frac))

		# sincroniza vel rueda del + amortiguacion anti-squat tras durante contragiro de cruce
		if cross_rev:
			if i < 2:
				if (not (active_mode == TestMode.EFFICIENCY or forbid_antag)) and ekf_vx > 3.0 and ki_ub > 0.015:
					var t_inert_match := clampf(ki_ub * 5800.0, 75.0, minf(t_max_drv, 312.5))
					ub_i = -t_inert_match
					lb_i = minf(lb_i, ub_i)
				else:
					ub_i = minf(ub_i, 0.0)
			else:
				if cross_init and (not (active_mode == TestMode.EFFICIENCY or forbid_antag)) and ekf_vx > 3.0:
					var t_as_snap := minf(t_max_drv, maxf(1.80 * t_ellipse, 180.0))
					ub_i = -t_as_snap
					lb_i = minf(lb_i, ub_i)
				else:
					ub_i = minf(ub_i, 0.0)

		if _fz_unclamped[i] < 40.0:
			ub_i = 0.0
			lb_i = 0.0
		elif _airborne_ramp[i] < 0.999:
			ub_i *= _airborne_ramp[i]
		if _curb_wheel_timer[i] > 0.0:
			# suelta antes solo si rueda decelera fuerte al recontactar asfalto
			if _curb_wheel_timer[i] < 0.030 and _om_dot_filt[i] < -80.0:
				_curb_wheel_timer[i] = minf(_curb_wheel_timer[i], 0.015)
			var curb_frac := clampf(_curb_wheel_timer[i] / 0.015, 0.0, 1.0)
			ub_i = minf(ub_i, lerpf(ub_i, 0.0, curb_frac))
			_curb_wheel_timer[i] = maxf(_curb_wheel_timer[i] - dt, 0.0)
		elif _curb_timer_s > 0.0:
			ub_i *= 0.65

		if ekf_vx < (3.0 / 3.6):
			lb_i = maxf(lb_i, 0.0)
			ub_i = maxf(ub_i, 0.0)
		lb_i = minf(lb_i, ub_i)

		_qp_ub[i] = ub_i
		_qp_lb[i] = lb_i
		# peso elipse normalizado o(1) pa balancear eta_i real entre ruedas descargadas y cargadas
		var t_avail_i := maxf(t_ellipse, 35.0)
		var wi_cfg := w_wheels[i] if i < w_wheels.size() else 1.0
		var inv_t := 145.0 / t_avail_i
		var th_w := 1.0 + 6.0 * th_u * th_u
		_qp_diag_w[i] = 0.005 if warmer_qp_active else (wi_cfg * th_w * clampf(inv_t * inv_t, 0.08, 10.0))

	# evita lucha de pares positivos vs regen cuando abs actua (mantiene split-mu y efficiency)
	if is_braking and any_abs:
		for i in range(4):
			if _qp_w_abs[i] <= 0.0 or (active_mode == TestMode.EFFICIENCY and _qp_t_abs[i] <= 0.0):
				var j_p := (1 - i) if i < 2 else (5 - i)
				if active_mode == TestMode.EFFICIENCY:
					_qp_ub[i] = maxf(_qp_lb[i], minf(-18.0, -0.85 * _qp_t_abs[j_p]))
				elif true_split_mu and _qp_w_abs[j_p] > 0.0:
					_qp_ub[i] = maxf(_qp_ub[i], _qp_ub[j_p])
					_qp_lb[i] = 0.75 * _qp_lb[j_p]
					_qp_t_abs[i] = _qp_t_abs[j_p]
					_qp_w_abs[i] = 8.0
				else:
					_qp_ub[i] = minf(_qp_ub[i], 0.0)
					_qp_lb[i] = minf(_qp_ub[i], maxf(_qp_lb[i], -1.5))

	# en curva seca bajo traccion, limita la rueda interior descargada segun su carga vertical real
	if (active_mode == TestMode.AUTOCROSS or active_mode == TestMode.SKIDPAD) and fx_cmd > 50.0 and (not is_braking) and (not true_split_mu) and (not warmer_qp_active) and absf(delta_filt) > _RAD_2_5 and absf(ay_meas) > 3.0:
		var idx_in_f := 0 if ay_meas > 0.0 else 1
		var idx_out_f := 1 - idx_in_f
		if fz_est[idx_out_f] > fz_est[idx_in_f] + 40.0 and _qp_ub[idx_out_f] > 1.0:
			var fz_ratio_f := clampf(fz_est[idx_in_f] / maxf(fz_est[idx_out_f], 100.0), 0.12, 0.85)
			_qp_ub[idx_in_f] = minf(_qp_ub[idx_in_f], _qp_ub[idx_out_f] * fz_ratio_f * 0.80)
			_qp_lb[idx_in_f] = minf(_qp_lb[idx_in_f], _qp_ub[idx_in_f])
			# Solo en Regimen 1 (baja velocidad) y sin sobreviraje: evita descargar la rueda delantera exterior en salida de horquilla
			var f_dem_u := clampf(fx_cmd / 1500.0, 0.0, 1.0)
			var b_pro_f_lim := lerpf(_RAD_2_8, _RAD_3_8, u_tight_hp)
			var turn_pro_f := (u_lo_spd > 0.05) and (not limp_home_active) and (not cross_rev) and (_curb_timer_s <= 0.0) and (delta_filt * wz_corr >= -0.05) and (mz_req * signf(delta_filt) > 0.0) and (absf(wz_corr) <= absf(r_ref) + 0.08) and (absf(beta_dyn) < b_pro_f_lim) and (emg_frac < 0.15)
			if turn_pro_f and f_dem_u > 0.15 and (delta_filt * _qp_bmz[idx_out_f] > 0.0):
				var t_out_floor := minf(_qp_ub[idx_out_f] * 0.42 * f_dem_u * u_lo_spd, maxf(_qp_ub[idx_out_f] - 1.0, 0.0))
				_qp_lb[idx_out_f] = maxf(_qp_lb[idx_out_f], t_out_floor)
				if not hp_asym_active:
					_qp_ub[idx_in_f] = minf(_qp_ub[idx_in_f], maxf(0.75 * t_out_floor, 0.0))
					_qp_lb[idx_in_f] = minf(_qp_lb[idx_in_f], _qp_ub[idx_in_f])
		var idx_in_r := 2 if ay_meas > 0.0 else 3
		var idx_out_r := 5 - idx_in_r
		if fz_est[idx_out_r] > fz_est[idx_in_r] + 40.0 and _qp_ub[idx_out_r] > 1.0:
			var fz_ratio_r := clampf(fz_est[idx_in_r] / maxf(fz_est[idx_out_r], 100.0), 0.12, 0.88)
			_qp_ub[idx_in_r] = minf(_qp_ub[idx_in_r], _qp_ub[idx_out_r] * fz_ratio_r * 0.90)
			_qp_lb[idx_in_r] = minf(_qp_lb[idx_in_r], _qp_ub[idx_in_r])

	# SKIDPAD_MAX_ATTACK_V2 friction-circle shedding: unload outer front drive when its mu utilization > 80 %, transfer to loaded outer rear
	if active_mode == TestMode.SKIDPAD and _mc_sk_fz_shed_gain > 0.0 and (not is_braking) and (not warmer_qp_active) and absf(ay_meas) > 6.0:
		var idx_o := 1 if ay_meas > 0.0 else 0
		var idx_or := 3 if ay_meas > 0.0 else 2
		var util_o := fy_cons_est[idx_o] / maxf(mu_hat[idx_o] * fz_est[idx_o], 100.0)
		var shed_u := clampf((util_o - 0.80) / 0.18, 0.0, 1.0) * _mc_sk_fz_shed_gain
		if shed_u > 0.0 and _qp_ub[idx_o] > 0.0:
			_qp_ub[idx_o] *= (1.0 - 0.90 * shed_u)
			_qp_lb[idx_o] = minf(_qp_lb[idx_o], _qp_ub[idx_o])
			_qp_diag_w[idx_o] *= (1.0 + 3.0 * shed_u)
			_qp_diag_w[idx_or] *= maxf(1.0 - 0.40 * shed_u, 0.45)

	if _curb_timer_s > 0.0:
		_curb_timer_s = maxf(_curb_timer_s - dt, 0.0)

	# modo efficiency: desconexion eje del con filtro unico _eff_t_filt + histeresis beta/par 0.50 s
	if active_mode == TestMode.EFFICIENCY:
		if _eff_disc_timer_s > 0.0:
			_eff_disc_timer_s = maxf(_eff_disc_timer_s - dt, 0.0)
		var part_th := _mc_part_th
		var t_act_sum := absf(t_prev[0] + t_prev[1] + t_prev[2] + t_prev[3])
		_eff_t_filt = lerpf(_eff_t_filt, t_act_sum, _a_eff_150)
		var safe_disc := fx_cmd > 0.0 and not is_braking and not limp_home_active
		if not _eff_front_disc:
			if safe_disc and absf(beta) < _RAD_2_2 and _eff_disc_timer_s <= 0.0 and t_act_sum > 1.0 and _eff_t_filt < part_th and t_act_sum < part_th * 1.25 and absf(mz_req) < 220.0 and absf(ay_meas) < 13.5:
				_eff_front_disc = true
				_eff_disc_timer_s = 0.50
		else:
			if (not safe_disc) or absf(beta) > _RAD_3_2 or absf(ay_meas) > 14.5 or (_eff_disc_timer_s <= 0.0 and (_eff_t_filt > part_th * 1.35 or absf(mz_req) > 240.0)):
				_eff_front_disc = false
				_eff_disc_timer_s = 0.50
		if _eff_front_disc:
			_qp_ub[0] = 0.0; _qp_lb[0] = 0.0
			_qp_ub[1] = 0.0; _qp_lb[1] = 0.0
	else:
		_eff_front_disc = false
		_eff_disc_timer_s = 0.0
		_eff_t_filt = 0.0

	# rango factible exacto [mz_min, mz_max] y objetivo estricto mz_tgt
	var mz_min := 0.0
	var mz_max := 0.0
	for i in range(4):
		var bi := _qp_bmz[i]
		if bi >= 0.0:
			mz_min += bi * _qp_lb[i]
			mz_max += bi * _qp_ub[i]
		else:
			mz_min += bi * _qp_ub[i]
			mz_max += bi * _qp_lb[i]
	# guarda signo mz: siempre activa en todas las velocidades (si lb>0 o ub<0 fuerza signo opuesto a mz_req, abre a 0)
	if (not limp_home_active) and ((mz_req > 25.0 and mz_max <= 1.0) or (mz_req < -25.0 and mz_min >= -1.0)):
		mz_min = 0.0
		mz_max = 0.0
		for i in range(4):
			if _qp_lb[i] > 0.0 and (_qp_bmz[i] * mz_req < 0.0):
				_qp_lb[i] = 0.0
			if _qp_ub[i] < 0.0 and (_qp_bmz[i] * mz_req > 0.0):
				_qp_ub[i] = 0.0
			var bi := _qp_bmz[i]
			if bi >= 0.0:
				mz_min += bi * _qp_lb[i]
				mz_max += bi * _qp_ub[i]
			else:
				mz_min += bi * _qp_ub[i]
				mz_max += bi * _qp_lb[i]
	var mz_tgt := 0.0 if limp_home_active else mz_req
	mz_tgt = clampf(mz_tgt, mz_min, mz_max)

	var fx_box_max := 0.0
	var mz_at_ub := 0.0
	for i in range(4):
		fx_box_max += _qp_bx[i] * _qp_ub[i]
		mz_at_ub += _qp_bmz[i] * _qp_ub[i]
	var fx_want := clampf(fx_cmd, 0.0, maxf(fx_box_max, 1.0))
	var load_scale := fx_want / maxf(fx_box_max, 1.0)
	var mz_load := mz_at_ub * load_scale

	# ancla minima pro-curva SOLO en Regimen 1 (baja velocidad, sin sobreviraje ni deslizamiento trasero)
	var u_pro_turn := 0.0
	if (u_lo_spd > 0.01) and (active_mode == TestMode.AUTOCROSS or active_mode == TestMode.SKIDPAD) and (not limp_home_active) and (not is_braking) and (not warmer_qp_active) and (not is_straight_split) and (not true_split_mu) and (not cross_rev) and (not rear_sliding_hi) and (fx_cmd > 50.0) and (absf(delta_filt) > _RAD_2_0) and (_curb_timer_s <= 0.0):
		if delta_filt * wz_corr >= -0.05 and absf(wz_corr) <= absf(r_ref) + 0.08 and (mz_req * signf(delta_filt) > -20.0):
			var ov_dyn := maxf(-beta_dyn * signf(delta_filt), 0.0)
			var u_beta := clampf(1.0 - maxf(absf(beta_dyn) - _RAD_2_8, 0.0) / _RAD_1_4, 0.0, 1.0)
			var u_ov := clampf(1.0 - maxf(ov_dyn - _RAD_2_5, 0.0) / _RAD_1_4, 0.0, 1.0)
			var u_mu_pro := clampf((mu_avg - 0.85) / 0.25, 0.0, 1.0)
			u_pro_turn = u_lo_spd * u_beta * u_ov * u_mu_pro * (1.0 - emg_frac)
			if u_pro_turn > 0.01:
				var turn_frac := lerpf(0.45, 0.65, u_lo_spd) * u_pro_turn
				if delta_filt > 0.0 and mz_load > 0.0:
					mz_tgt = maxf(mz_tgt, clampf(turn_frac * mz_load, mz_min, mz_max))
				elif delta_filt < 0.0 and mz_load < 0.0:
					mz_tgt = minf(mz_tgt, clampf(turn_frac * mz_load, mz_min, mz_max))

	# reserva >=82-85% del presupuesto longitudinal en salida de curva estable (desactivada ante sobreviraje/Regimen 2)
	var wz_abs_now := absf(wz_corr)
	var wz_decreasing := (not is_first_tick) and (wz_abs_now <= _wz_abs_prev + 0.005)
	var turning_in := (delta_filt * delta_dot > 0.05) or (_turnin_timer_s > 0.05 and not wz_decreasing and absf(delta_filt) > _RAD_3_0 and delta_filt * delta_dot >= -0.004)
	var b_exit_lim := lerpf(_RAD_3_5, _RAD_2_2, u_hi_spd)
	var u_unw := clampf((-delta_filt * delta_dot - 0.003) / 0.010, 0.0, 1.0) * clampf((b_exit_lim - absf(beta_dyn)) / _RAD_1_2, 0.0, 1.0)
	var stable_exit := (wz_decreasing or absf(beta_dyn) < b_exit_lim or u_unw > 0.05) and (not turning_in) and (not rear_sliding_hi) and (over_wz <= 0.06)
	var counter_mz_act := (absf(wz_corr) > 0.10) and (mz_req * signf(wz_corr) < -30.0)
	if active_mode == TestMode.AUTOCROSS and (not limp_home_active) and (not is_braking) and (not warmer_qp_active) and (not is_straight_split) and (not true_split_mu) and (not counter_mz_act) and (emg_frac <= 0.02) and (u_th_raw > 0.25) and (fx_cmd > 50.0) and stable_exit:
		var u_exit := 1.0 if (wz_decreasing or u_unw > 0.5) else clampf((b_exit_lim - absf(beta_dyn)) / _RAD_1_4, 0.0, 1.0)
		var u_mu := clampf((mu_avg - 0.88) / 0.22, 0.0, 1.0)
		var mz_along := mz_req * signf(wz_corr) if absf(wz_corr) > 0.05 else absf(mz_req)
		var u_yaw := maxf(maxf(u_unw, u_pro_turn), clampf(1.0 + mz_along / 25.0, 0.0, 1.0))
		var eff_punch := 0.83 * u_exit * u_mu * u_yaw * (1.0 - emg_frac)
		if eff_punch > 0.01:
			var fx_drop_max := maxf(fx_box_max - eff_punch * fx_want, 0.0)
			var p_bounds := _calc_punch_mz_bounds(_qp_lb, _qp_ub, _qp_bx, _qp_bmz, fx_drop_max, mz_at_ub)
			var mz_p_lo := p_bounds.x
			var mz_p_hi := p_bounds.y
			if mz_req > 25.0 and mz_p_hi < 0.0:
				mz_p_hi = 0.0
			elif mz_req < -25.0 and mz_p_lo > 0.0 and u_unw < 0.5:
				mz_p_lo = minf(mz_p_lo, 0.0)
			mz_tgt = clampf(mz_tgt, maxf(mz_min, mz_p_lo), minf(mz_max, mz_p_hi))
	mz_tgt = clampf(mz_tgt, mz_min, mz_max)

	# pesos qp, cte weyl lipschitz y fista proyectado (12 iters warm-started)
	var fast_launch_qp := (active_mode == TestMode.AUTOCROSS or active_mode == TestMode.ACCELERATION) and (fx_cmd > 200.0) and (not is_braking) and (not is_straight_split) and (not warmer_qp_active) and (emg_frac <= 0.05) and (ekf_vx < 14.0)
	var rear_sat_launch: bool = fast_launch_qp and absf(delta_filt) < _RAD_2_5 and fx_cmd >= 0.80 * fx_box_max
	var delta_t_pitch_qp := (_qp_ub[0] + _qp_ub[1] - _qp_ub[2] - _qp_ub[3]) * load_scale if (u_pro_turn > 0.2 and fx_cmd >= 0.70 * fx_box_max) else delta_t_pitch
	var launch_slew_bypass := (_launch_timer_s > 0.0 and _launch_timer_s <= 0.020)
	var slew_eff := 0.0 if launch_slew_bypass else ((lambda_slew * 0.02) if warmer_qp_active else ((lambda_slew * 0.15) if is_braking else ((lambda_slew * 0.25) if (fast_launch_qp and mz_sat_ratio <= 0.75) else (lambda_slew * (3.0 if (mz_sat_ratio > 0.75 or ekf_vx < 10.0) else 1.2)))))
	var w_pitch := 0.0 if (warmer_qp_active or rear_sat_launch) else ((lambda_pitch * 35.0) if (is_braking and lambda_pitch > 0.0) else (lambda_pitch * lerpf(1.0, 3.5, rear_bias_u)))
	var w_fx := 0.0 if warmer_qp_active else ((0.18 if (hp_asym_active and u_th_raw < 0.40) else 1.20) if (is_braking or (hp_asym_active and u_th_raw < 0.40)) else (0.30 if is_straight_split else (8.0 if (fast_launch_qp or u_pro_turn > 0.1) else 0.90)))
	if active_mode == TestMode.ACCELERATION and not is_braking:
		w_fx = 12.0
	var d_max := 0.0
	var bx2_sum := 0.0
	for i in range(4):
		d_max = maxf(d_max, _qp_diag_w[i] + _qp_w_abs[i])
		bx2_sum += _qp_bx[i] * _qp_bx[i]
	var l_est := d_max + slew_eff + 4.0 * w_pitch + w_fx * bx2_sum
	var step_alpha := 1.0 / maxf(l_est, 1.0)

	_project_box_eq(t_prev, _qp_lb, _qp_ub, _qp_bmz, mz_tgt, _qp_x)
	for i in range(4):
		_qp_y[i] = _qp_x[i]
	var t_fista := 1.0
	var n_iters := mini(fista_iters, 12)

	# Candidate A (kept): centripetal projection gradient weight (-w_cent * Fx_f * sin(delta) in QP objective)
	var w_cent_qp := 0.0
	if _mc_centripetal_fx_gain > 0.1 and (active_mode == TestMode.AUTOCROSS or active_mode == TestMode.SKIDPAD) and (not is_braking) and (not warmer_qp_active) and (not is_straight_split) and (not rear_sliding_hi) and fx_cmd > 0.0 and absf(delta_filt) > 0.10 and delta_alpha_rel > 0.0 and absf(beta_dyn) < _RAD_5_0 and absf(_beta_dot) < 0.18 and absf(_ax_long_filt) < 2.5:
		w_cent_qp = _mc_centripetal_fx_gain * sin(absf(delta_filt)) * clampf((_RAD_5_0 - absf(beta_dyn)) / _RAD_1_4, 0.0, 1.0)

	for _it in range(n_iters):
		var fx_y := 0.0
		for i in range(4):
			fx_y += _qp_bx[i] * _qp_y[i]
		var err_fx := fx_y - fx_cmd
		var err_p := (_qp_y[0] + _qp_y[1]) - (_qp_y[2] + _qp_y[3]) - delta_t_pitch_qp

		for i in range(4):
			var p_sgn := 1.0 if i < 2 else -1.0
			# quita slew si sale de frenada regen hacia gas pa cero torque fighting
			var slew_i := 0.0 if (fx_cmd > 0.0 and not is_braking and t_prev[i] < -1.0) else slew_eff
			var grad := _qp_diag_w[i] * _qp_y[i] + slew_i * (_qp_y[i] - t_prev[i]) \
				+ _qp_w_abs[i] * (_qp_y[i] - _qp_t_abs[i]) + w_pitch * p_sgn * err_p + w_fx * _qp_bx[i] * err_fx \
				- w_cent_qp * _qp_bfy_f[i]
			_qp_v[i] = _qp_y[i] - step_alpha * grad
		_project_box_eq(_qp_v, _qp_lb, _qp_ub, _qp_bmz, mz_tgt, _qp_z)

		# reinicio adaptativo de nesterov pa evitar sobrepico en restricciones activas
		var r_dot := 0.0
		for i in range(4):
			r_dot += (_qp_y[i] - _qp_z[i]) * (_qp_z[i] - _qp_x[i])
		if r_dot > 0.0:
			t_fista = 1.0
			for i in range(4):
				_qp_y[i] = _qp_z[i]
				_qp_x[i] = _qp_z[i]
		else:
			var t_next := 0.5 * (1.0 + sqrt(1.0 + 4.0 * t_fista * t_fista))
			var beta_f := (t_fista - 1.0) / t_next
			for i in range(4):
				_qp_y[i] = _qp_z[i] + beta_f * (_qp_z[i] - _qp_x[i])
				_qp_x[i] = _qp_z[i]
			t_fista = t_next

	# techo P_max(vx) conservando mz_tgt via bisecc proyectada (siempre <= p_bms_max 80 kW)
	var p_cap := get_effective_power_cap_w(ekf_vx)
	var p_elec := _calc_p_elec(_qp_x, w_omegas, w_eff, eta_drive, eta_regen)
	if p_elec > p_cap and p_elec > 1e-3:
		var w2_sum := 0.0
		var w_ref_max := 0.0
		var span_max := 0.0
		for i in range(4):
			var wi := maxf(0.0 if is_nan(w_omegas[i]) else w_omegas[i], 0.0)
			w2_sum += wi * wi
			w_ref_max = maxf(w_ref_max, wi)
			span_max = maxf(span_max, _qp_x[i] - _qp_lb[i])
		var gamma_est := ((p_elec - p_cap) * eta_drive) / maxf(w2_sum, 1.0)
		var gamma_lo := 0.0
		var gamma_hi := clampf(1.05 * gamma_est, 0.05, 50.0)
		var p_hi := _eval_p_gamma(_qp_x, gamma_hi, w_omegas, w_eff, _qp_lb, _qp_ub, _qp_bmz, mz_tgt, eta_drive, eta_regen, _qp_v, _qp_tmp)
		if p_hi > p_cap:
			gamma_lo = gamma_hi
			gamma_hi = maxf(4.0 * gamma_est, maxf((2.0 * span_max) / maxf(w_ref_max, 1.0), 1.0))
			p_hi = _eval_p_gamma(_qp_x, gamma_hi, w_omegas, w_eff, _qp_lb, _qp_ub, _qp_bmz, mz_tgt, eta_drive, eta_regen, _qp_v, _qp_tmp)
			if p_hi > p_cap:
				gamma_lo = gamma_hi
				gamma_hi *= 4.0
				p_hi = _eval_p_gamma(_qp_x, gamma_hi, w_omegas, w_eff, _qp_lb, _qp_ub, _qp_bmz, mz_tgt, eta_drive, eta_regen, _qp_v, _qp_tmp)
		else:
			var g_low_t := 0.90 * gamma_est
			if _eval_p_gamma(_qp_x, g_low_t, w_omegas, w_eff, _qp_lb, _qp_ub, _qp_bmz, mz_tgt, eta_drive, eta_regen, _qp_v, _qp_tmp) > p_cap:
				gamma_lo = g_low_t

		if p_hi <= p_cap:
			for _b in range(6):
				var gamma_mid := 0.5 * (gamma_lo + gamma_hi)
				if _eval_p_gamma(_qp_x, gamma_mid, w_omegas, w_eff, _qp_lb, _qp_ub, _qp_bmz, mz_tgt, eta_drive, eta_regen, _qp_v, _qp_tmp) <= p_cap:
					gamma_hi = gamma_mid
				else:
					gamma_lo = gamma_mid
			_eval_p_gamma(_qp_x, gamma_hi, w_omegas, w_eff, _qp_lb, _qp_ub, _qp_bmz, mz_tgt, eta_drive, eta_regen, _qp_v, _qp_tmp)
			for i in range(4):
				_qp_x[i] = _qp_tmp[i]
		elif p_hi < p_elec:
			for i in range(4):
				_qp_x[i] = _qp_tmp[i]

	# simetria izq-der con caja compartida en limp-home + reescalado estricto 80 kw si aun supera p_bms_max
	if limp_home_active:
		var lb_f := maxf(_qp_lb[0], _qp_lb[1])
		var ub_f := minf(_qp_ub[0], _qp_ub[1])
		var lb_r := maxf(_qp_lb[2], _qp_lb[3])
		var ub_r := minf(_qp_ub[2], _qp_ub[3])
		var tf_sym := clampf(0.5 * (_qp_x[0] + _qp_x[1]), minf(lb_f, ub_f), ub_f)
		var tr_sym := clampf(0.5 * (_qp_x[2] + _qp_x[3]), minf(lb_r, ub_r), ub_r)
		_qp_x[0] = tf_sym; _qp_x[1] = tf_sym
		_qp_x[2] = tr_sym; _qp_x[3] = tr_sym

	p_elec = _calc_p_elec(_qp_x, w_omegas, w_eff, eta_drive, eta_regen)
	var p_bms_hard := minf(p_bms_max, 80000.0)
	var p_chg_hard := minf(p_max_charge_w, 25000.0)
	var p_chg_lim := 0.98 * p_chg_hard * (eta_regen / maxf(eta_regen, eta_drive))
	if p_elec > p_bms_hard and p_elec > 1e-3:
		var p_pos := 0.0
		var p_neg_abs := 0.0
		var ed := maxf(eta_drive, 0.1)
		for i in range(4):
			var wi := w_eff[i] if is_nan(w_omegas[i]) else w_omegas[i]
			var pm := _qp_x[i] * wi
			if pm >= 0.0:
				p_pos += pm / ed
			else:
				p_neg_abs += -pm * eta_regen
		if p_pos > 1e-3:
			var sc_pos := clampf((p_bms_hard + p_neg_abs) / p_pos, 0.0, 1.0)
			for i in range(4):
				if _qp_x[i] > 0.0:
					_qp_x[i] *= sc_pos
		else:
			var sc_all := p_bms_hard / p_elec
			for i in range(4):
				_qp_x[i] *= sc_all
	elif p_elec < -p_chg_lim and p_elec < -1e-3:
		var p_pos := 0.0
		var p_neg_abs := 0.0
		var ed := maxf(eta_drive, 0.1)
		for i in range(4):
			var wi := w_eff[i] if is_nan(w_omegas[i]) else w_omegas[i]
			var pm := _qp_x[i] * wi
			if pm >= 0.0:
				p_pos += pm / ed
			else:
				p_neg_abs += -pm * eta_regen
		if p_neg_abs > 1e-3:
			var sc_chg := clampf((p_pos + p_chg_lim) / p_neg_abs, 0.0, 1.0)
			for i in range(4):
				if _qp_x[i] < 0.0:
					_qp_x[i] *= sc_chg

	if ekf_vx < (3.0 / 3.6):
		for i in range(4):
			if _qp_x[i] < 0.0:
				_qp_x[i] = 0.0

	# calcula saturacion de mz y telemetria final
	var mz_out := 0.0
	var p_pos_mech_tf := 0.0
	var p_neg_mech_tf := 0.0
	p_elec = _calc_p_elec(_qp_x, w_omegas, w_eff, eta_drive, eta_regen)
	for i in range(4):
		mz_out += _qp_bmz[i] * _qp_x[i]
		t_prev[i] = _qp_x[i]
		last_t_cmd[i] = _qp_x[i]
		var wi_tf := w_eff[i] if is_nan(w_omegas[i]) else w_omegas[i]
		var pm_tf := _qp_x[i] * wi_tf
		if pm_tf >= 0.0:
			p_pos_mech_tf += pm_tf
		else:
			p_neg_mech_tf += -pm_tf
	last_torque_fight_kw = minf(p_pos_mech_tf, p_neg_mech_tf) * 0.001
	last_power_w = p_elec
	mz_sat_ratio = clampf(1.0 - absf(mz_out) / maxf(absf(mz_req), 1.0), 0.0, 1.0) if absf(mz_req) > 25.0 else 0.0
	_wz_abs_prev = wz_abs_now

	# Regime 7: live diagnostics update
	_diag_mz_tv_nm = mz_out
	_diag_pivot_blend = pivot_blend
	_diag_beta_deg = rad_to_deg(beta)
	_diag_beta_dot_dps = rad_to_deg(_beta_dot)
	_diag_mu_hat_avg = mu_avg
	var turn_sgn := signf(d_norm_raw) if absf(d_norm_raw) > 0.10 else (signf(delta_filt) if absf(delta_filt) > 0.04 else 0.0)
	if absf(turn_sgn) > 0.5:
		var t_in_avg := 0.5 * (_qp_x[0] + _qp_x[2]) if turn_sgn > 0.0 else 0.5 * (_qp_x[1] + _qp_x[3])
		var t_out_avg := 0.5 * (_qp_x[1] + _qp_x[3]) if turn_sgn > 0.0 else 0.5 * (_qp_x[0] + _qp_x[2])
		_diag_t_out_in_ratio = clampf(t_out_avg / maxf(absf(t_in_avg), 1.0), -20.0, 20.0)
	else:
		_diag_t_out_in_ratio = 1.0
	if rear_sliding_hi or (absf(beta_dyn) > _RAD_2_8 and beta_dyn * wz_corr < 0.0):
		_diag_balance_state = "OVERSTEER"
	elif us_err_apex > 0.06 and absf(alpha_f) > absf(alpha_r) + _RAD_1_4:
		_diag_balance_state = "UNDERSTEER"
	else:
		_diag_balance_state = "NEUTRAL"
	if is_launch_req:
		_diag_active_regime = "LAUNCH_BITE"
	elif is_braking and tb_blend > 0.15:
		_diag_active_regime = "TRAIL_BRAKE"
	elif pivot_blend > 0.15:
		_diag_active_regime = "HAIRPIN_PIVOT"
	elif active_mode == TestMode.SKIDPAD and absf(ay_meas) > 6.0:
		_diag_active_regime = "SKIDPAD_STEADY"
	elif u_hi_spd > 0.25 and (absf(delta_dot) > 0.5 or rear_sliding_hi or absf(beta) > _RAD_1_4):
		_diag_active_regime = "SLALOM_DAMP"
	else:
		_diag_active_regime = "BALANCED"

	return last_t_cmd
