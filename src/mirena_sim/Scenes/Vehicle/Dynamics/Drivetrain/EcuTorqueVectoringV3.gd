class_name EcuTorqueVectoringV3
extends RefCounted

## 4WD (TeR27) port of the ECU Torque Vectoring v3 + Adaptive RLS Traction Control pipeline.
## Faithfully ports gp_math, gp_ekf, gp_vehicle_model, gp_solver, gp_traction_control,
## and gp_torque_vectoring, generalized to 4 independent in-hub motors [FL, FR, RL, RR] at 200 Hz.

const GP_FL: int = 0
const GP_FR: int = 1
const GP_RL: int = 2
const GP_RR: int = 3

# ── TeR27 Vehicle & Actuator Parameters ─────────────────────────────────────
const GP_MASS: float = 290.0
const GP_IZ: float = 125.0
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
const GP_I_WHEEL_EST: float = 0.5075  # [kg*m^2]

# ── Tire & Understeer Reference ─────────────────────────────────────────────
const GP_C_ALPHA_F: float = 35000.0
const GP_C_ALPHA_R: float = 32000.0
const GP_MU_NOM: float = 1.90
const GP_MU_SLIDE: float = 1.55
const GP_MU_DEG_FZ: float = 0.00012
const GP_K_US: float = 0.006
const GP_K_US_FZ: float = 0.0015

# ── Control, QP & Filtering Constants ───────────────────────────────────────
const GP_W_SMOOTH: float = 5.999
const GP_W_REG: float = 0.401
const GP_TV_MAX_MZ: float = 1500.0
const GP_TV_WZ_I_MAX: float = 200.0
const GP_TV_RATE_LIMIT: float = 3252.3
const GP_STEER_DEADZONE_RAD: float = 0.0087
const GP_YAW_DEADZONE_RADS: float = 0.0175
const GP_ACCEL_LPF_TAU: float = 0.0200
const GP_UB_LPF_TAU: float = 0.0300
const FIXED_DT: float = 0.005         # 200 Hz ECU rate

# ── Traction Control RLS Constants ──────────────────────────────────────────
const GP_TC_KP: float = 12.0
const GP_TC_KI: float = 12.0
const GP_TC_B0: float = 14.0
const GP_TC_B1: float = -0.0018
const GP_TC_C_PAC: float = 1.65
const GP_TC_ALPHA_PEAK: float = 0.20
const GP_TC_I_MAX: float = 3.5
const GP_TC_V_KP_SCALE: float = 3.0
const GP_TC_ALPHA_KAPPA_LP: float = 0.85
const GP_TC_ALPHA_MU_EMA: float = 0.008
const GP_TC_MU_LO: float = 0.40
const GP_TC_MU_HI: float = 2.20
const GP_TC_MU_NOM: float = 1.90
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
var tc_kappa_opt: Array[float] = [0.15, 0.15, 0.15, 0.15]

# ── Fixed-rate 200 Hz Accumulator & Telemetry ───────────────────────────────
var _time_acc: float = 0.0
var last_t_cmd: Array[float] = [0.0, 0.0, 0.0, 0.0]
var last_wz_ref: float = 0.0
var last_mz_req: float = 0.0
var tc_active: bool = false

func _init() -> void:
	reset()

func reset() -> void:
	ekf_x = [0.0, 0.0]
	ekf_p = [[0.10, 0.0], [0.0, 0.01]]
	ekf_beta_est = 0.0
	ekf_vy_std = sqrt(0.10)
	ekf_wz_corrected = 0.0

	wz_int = 0.0
	delta_prev = 0.0
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
	tc_kappa_opt = [0.15, 0.15, 0.15, 0.15]

	_time_acc = 0.0
	last_t_cmd = [0.0, 0.0, 0.0, 0.0]
	last_wz_ref = 0.0
	last_mz_req = 0.0
	tc_active = false

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

	var a_total_g: float = sqrt(ax_f * ax_f + ay_f * ay_f) / GP_GRAVITY
	var excess_g: float = clampf(a_total_g - 0.45, 0.0, 2.0)
	var sat_penalty: float = 1.0 + 10.0 * (excess_g * excess_g)
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

static func gp_friction_ellipse_t_ub(fz: Array[float], fy_est: Array[float], mu_est: float) -> Array[float]:
	var mu_safe: float = clampf(mu_est, 1.35, 2.20)
	var t_ub: Array[float] = [0.0, 0.0, 0.0, 0.0]
	for i in range(4):
		var max_cap: float = mu_safe * fz[i]
		var min_fx_reserve_ratio: float = 0.35
		var fx_reserve: float = min_fx_reserve_ratio * max_cap
		var fx_sq_max: float = maxf((max_cap * max_cap) - (fy_est[i] * fy_est[i]), fx_reserve * fx_reserve)
		var fx_max: float = gp_softplus_sqrt(fx_sq_max * 4.0) * 0.5
		t_ub[i] = fx_max * GP_R_WHEEL
	return t_ub

static func gp_power_limited_t_ub(omega_wheel: Array[float]) -> Array[float]:
	var t_ub: Array[float] = [0.0, 0.0, 0.0, 0.0]
	for i in range(4):
		var omega_safe: float = gp_softplus(omega_wheel[i])
		var t_power: float = GP_P_MAX_WHL / (omega_safe + 1e-3)
		t_ub[i] = clampf(t_power, 0.0, GP_MAX_TRQ_ACT)
	return t_ub

static func gp_power_limited_t_lb(omega_wheel: Array[float], p_max_charge_w: float) -> Array[float]:
	var t_lb: Array[float] = [0.0, 0.0, 0.0, 0.0]
	for i in range(4):
		var omega_safe: float = gp_softplus(absf(omega_wheel[i]))
		var t_power: float = p_max_charge_w / (omega_safe + 1e-3)
		t_lb[i] = clampf(t_power, 0.0, GP_MAX_TRQ_ACT)
	return t_lb

static func gp_adaptive_k_us(fz: Array[float]) -> float:
	var fz_f: float = 0.5 * (fz[GP_FL] + fz[GP_FR])
	var fz_r: float = 0.5 * (fz[GP_RL] + fz[GP_RR])
	return GP_K_US * (1.0 + GP_K_US_FZ * (fz_r - fz_f))

static func gp_moment_arms() -> Array[float]:
	var hw_f: float = 0.5 * GP_TRACK_F / GP_R_WHEEL
	var hw_r: float = 0.5 * GP_TRACK_R / GP_R_WHEEL
	return [-hw_f, hw_f, -hw_r, hw_r]

# ── gp_solver (4WD) ──────────────────────────────────────────────────────────
static func gp_nominal_allocation_4wd(fx_driver: float, mz_target: float, fz_est: Array[float]) -> Array[float]:
	var arms: Array[float] = gp_moment_arms()
	var t_fx_total: float = fx_driver * GP_R_WHEEL

	var fz_sum: float = 0.0
	var denom: float = 0.0
	for i in range(4):
		var fz_i: float = maxf(fz_est[i], 1.0)
		fz_sum += fz_i
		denom += arms[i] * arms[i] * fz_i
	fz_sum = maxf(fz_sum, 1e-3)
	denom = maxf(denom, 1e-3)

	var t_nom: Array[float] = [0.0, 0.0, 0.0, 0.0]
	for i in range(4):
		var fz_i: float = maxf(fz_est[i], 1.0)
		var t_fx_i: float = t_fx_total * (fz_i / fz_sum)
		var t_mz_i: float = (arms[i] * fz_i * mz_target) / denom
		t_nom[i] = t_fx_i + t_mz_i
	return t_nom

## Exact closed-form KKT active-set water-filling solver for 4WD:
## min_{T} 0.5 * sum((T_i - T_blend_i)^2) s.t. sum(T_i) = T_target, T_lb_i <= T_i <= T_ub_i.
static func gp_qp_solve_4wd_closedform(
	t_warmstart: Array[float],
	t_prev: Array[float],
	fx_driver: float,
	t_lb: Array[float],
	t_ub: Array[float]
) -> Dictionary:
	var h: float = GP_W_REG + GP_W_SMOOTH
	var a_eq: float = 1.0 / GP_R_WHEEL
	var t_blend: Array[float] = [0.0, 0.0, 0.0, 0.0]
	var t_total_min: float = 0.0
	var t_total_max: float = 0.0
	for i in range(4):
		t_blend[i] = (GP_W_REG * t_warmstart[i] + GP_W_SMOOTH * t_prev[i]) / h
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
	brake_norm: float = 0.0
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
	var cs_factor: float = maxf(_tc_combined_slip_factor(vx, vy, wz), 0.72)
	var speed_gate: float = gp_sigmoid((vx - GP_TC_V_MIN) * 3.0)
	var kp_eff: float = (GP_TC_KP * 20.0) * (1.0 + GP_TC_V_KP_SCALE / (absf(vx) + 0.5))
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

	_time_acc += dt
	var steps: int = 0
	while _time_acc >= FIXED_DT and steps < 8:
		_time_acc -= FIXED_DT
		steps += 1
		_gp_tv_step(fx_driver, delta, vx, vy, wz, ay, ax, omega, brake_norm, FIXED_DT, wheel_steers, vy_gps, gps_valid)

	return last_t_cmd.duplicate()

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
	# Stage 1: Standstill hold
	if vx < 1.0 and brake_norm > 0.5 and fx_driver > 500.0:
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

	# Stage 5: Bilinear 4x4 Yaw-Rate Reference + Beta Damping + OS/Countersteer Gates
	var k_us: float = gp_adaptive_k_us(fz_est)
	var wz_ref: float = (vx_safe * delta_dz) / (GP_WB + k_us * vx_safe * vx_safe)
	last_wz_ref = wz_ref

	var v_norm: float = clampf(vx_safe / 30.0, 0.0, 1.0)
	var ay_norm: float = clampf(absf(ay_filt) / 15.0, 0.0, 1.0)
	var kp: float = gp_bilinear_interp_4x4(KP_MAP, v_norm, ay_norm)
	var ki: float = gp_bilinear_interp_4x4(KI_MAP, v_norm, ay_norm)
	var kd: float = gp_bilinear_interp_4x4(KD_MAP, v_norm, ay_norm)

	var mu_avg: float = 0.25 * (tc_mu_surface[0] + tc_mu_surface[1] + tc_mu_surface[2] + tc_mu_surface[3])
	var mu_scale: float = clampf(mu_avg / GP_MU_NOM, 0.4, 1.0)
	kp *= mu_scale
	ki *= mu_scale

	var wz_err: float = wz_ref - wz_corr
	var delta_dot: float = (delta_dz - delta_prev) / dt
	delta_prev = delta_dz

	var vy_std_clamped: float = clampf(ekf_vy_std, 0.0, 0.3)
	var k_beta_dynamic: float = 4000.0 * (1.0 + 2.0 * vy_std_clamped)
	var beta_term: float = clampf(-k_beta_dynamic * beta, -600.0, 600.0)

	var raw_int: float = wz_int + wz_err * dt * mz_sat_ratio
	wz_int = GP_TV_WZ_I_MAX * tanh(raw_int / GP_TV_WZ_I_MAX)

	var os_gate: float = 1.0 - gp_sigmoid((absf(wz_corr) - absf(wz_ref) - 0.2) * 10.0)
	var counter_steer_factor: float = 1.0 - gp_sigmoid(-(delta_dz * wz_corr + 0.05) * 40.0)

	var ff_mz: float = kd * delta_dot * (vx_safe / 10.0)
	var fb_mz: float = kp * wz_err + ki * wz_int + beta_term
	var mz_req: float = clampf((ff_mz + fb_mz) * os_gate * counter_steer_factor, -GP_TV_MAX_MZ, GP_TV_MAX_MZ)
	last_mz_req = mz_req

	# Stage 6: 4WD Friction Ellipse + Power Bounds (filtered tau = 30 ms)
	var w_chassis: float = vx_safe / GP_R_WHEEL
	var omega_power: Array[float] = [w_chassis, w_chassis, w_chassis, w_chassis]
	var t_ub_friction: Array[float] = gp_friction_ellipse_t_ub(fz_est, fy_est, mu_avg)
	var t_ub_power: Array[float] = gp_power_limited_t_ub(omega_power)

	var alpha_ub: float = clampf(dt / (GP_UB_LPF_TAU + dt), 0.0, 1.0)
	var t_lb: Array[float] = [0.0, 0.0, 0.0, 0.0]
	var t_ub: Array[float] = [0.0, 0.0, 0.0, 0.0]
	var max_sum: float = 0.0
	for i in range(4):
		var raw_ub_i: float = minf(t_ub_friction[i], t_ub_power[i])
		if not _ub_filt_initialized:
			t_ub_filt[i] = raw_ub_i
		else:
			t_ub_filt[i] += alpha_ub * (raw_ub_i - t_ub_filt[i])
		t_ub[i] = t_ub_filt[i]
		t_lb_filt[i] = 0.0
		max_sum += t_ub[i]
	_ub_filt_initialized = true

	var fx_clamped: float = fx_driver
	var max_sum_power: float = GP_P_MAX_TOTAL_MECH / maxf(w_chassis, 1e-3)
	var eff_max_sum: float = minf(max_sum, max_sum_power)
	var req_sum: float = fx_clamped * GP_R_WHEEL
	if req_sum > eff_max_sum:
		fx_clamped = eff_max_sum / GP_R_WHEEL
	if req_sum < 0.0:
		fx_clamped = 0.0

	# Stage 7: 4WD Nominal Allocation + Closed-Form KKT QP Solver + Rate Limiter
	var t_nominal: Array[float] = gp_nominal_allocation_4wd(fx_clamped, mz_req, fz_est)
	var qp_res: Dictionary = gp_qp_solve_4wd_closedform(t_nominal, t_qp_prev, fx_clamped, t_lb, t_ub)
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
		var d_t: float = clampf(qp_result[i] - t_qp_prev[i], -max_delta_t, max_delta_t)
		var tv_final: float = t_qp_prev[i] + d_t
		t_qp_prev[i] = tv_final
		t_cmd[i] = tv_final

	# Stage 8: 4WD Adaptive RLS Traction Control
	gp_tc_step_4wd(t_cmd, omega, vx, vy_est, wz_corr, fz_est, dt, wheel_steers, brake_norm)
	gp_ekf_update_friction()

	for i in range(4):
		t_out_prev[i] = t_cmd[i]
		last_t_cmd[i] = t_cmd[i]
