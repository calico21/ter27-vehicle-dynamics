class_name TvFrictionYmdWidget
extends Control

## Compact custom-drawn widget inside the Setup / TV tab:
##   - Top section: Live 4-wheel Pacejka Friction Circles (FL/FR/RL/RR),
##     per-wheel torque split bars, QP yaw moment Mz_TV, and HV DC power (-25..+80 kW).
##   - Bottom section: Instant Yaw Moment Diagram (Mz vs ay) preview updated immediately
##     when centripetal_fx_gain, delta_alpha_eq_gain, brake_bias_front_pct, or TV sliders change.

const CORNER_NAMES: Array[String] = ["FL", "FR", "RL", "RR"]

# Live telemetry state
var tire_sat_pct: Array[float] = [0.0, 0.0, 0.0, 0.0]
var tire_fx_norm: Array[float] = [0.0, 0.0, 0.0, 0.0]
var tire_fy_norm: Array[float] = [0.0, 0.0, 0.0, 0.0]
var wheel_torques_nm: Array[float] = [0.0, 0.0, 0.0, 0.0]
var mz_tv_nm: float = 0.0
var p_dc_kw: float = 0.0

# Instant YMD preview state (baseline vs current)
var _ymd_base_pts: PackedVector2Array = PackedVector2Array() # (ay_g, mz_nm)
var _ymd_cur_pts: PackedVector2Array = PackedVector2Array()
var ymd_ay_max_g: float = 0.0
var ymd_mz_ctrl_nm: float = 0.0
var ymd_stability_nm_deg: float = 0.0
var ymd_base_ay_max_g: float = 0.0
var ymd_base_stability_nm_deg: float = 0.0
var _last_params_sig: String = ""


func _init() -> void:
	custom_minimum_size = Vector2(360, 232)
	size_flags_horizontal = Control.SIZE_EXPAND_FILL
	focus_mode = Control.FOCUS_NONE
	mouse_filter = Control.MOUSE_FILTER_IGNORE


func _ready() -> void:
	focus_mode = Control.FOCUS_NONE
	if _ymd_cur_pts.is_empty():
		update_ymd_preview({}, {})


func update_live_telemetry(car: Node, snap: Dictionary, tv_diag: Dictionary = {}) -> void:
	var sat_arr: Array = snap.get("tire_sat_pct", [0.0, 0.0, 0.0, 0.0])
	var ax_g := float(snap.get("ax_g", 0.0))
	var ay_g := float(snap.get("ay_g", 0.0))
	var t_cmd_arr: Array = snap.get("t_cmd", [])

	for i in range(4):
		var sp := clampf(float(sat_arr[i]) if i < sat_arr.size() else 0.0, 0.0, 125.0)
		tire_sat_pct[i] = sp
		var r_sat := clampf(sp / 100.0, 0.0, 1.15)
		var got_exact := false
		if car != null and "model" in car and car.model != null and "wheels" in car.model and car.model.wheels is Array and i < car.model.wheels.size():
			var wh: Variant = car.model.wheels[i]
			if wh != null:
				if "t_drive" in wh:
					wheel_torques_nm[i] = float(wh.t_drive)
				elif "torque_cmd" in wh:
					wheel_torques_nm[i] = float(wh.torque_cmd)
				elif i < t_cmd_arr.size():
					wheel_torques_nm[i] = float(t_cmd_arr[i])
				if "fx" in wh and "fy" in wh and "fz" in wh:
					var fz_w := maxf(float(wh.fz), 120.0)
					var mu_ref := 1.45
					var denom := maxf(mu_ref * fz_w, 120.0)
					tire_fx_norm[i] = clampf(float(wh.fx) / denom, -1.15, 1.15)
					tire_fy_norm[i] = clampf(float(wh.fy) / denom, -1.15, 1.15)
					got_exact = true
		if not got_exact:
			if i < t_cmd_arr.size():
				wheel_torques_nm[i] = float(t_cmd_arr[i])
			var g_mag := maxf(sqrt(ax_g * ax_g + ay_g * ay_g), 0.15)
			tire_fx_norm[i] = clampf((ax_g / g_mag) * r_sat, -1.15, 1.15)
			tire_fy_norm[i] = clampf((ay_g / g_mag) * r_sat, -1.15, 1.15)

	mz_tv_nm = float(tv_diag.get("mz_tv_nm", 0.0))
	p_dc_kw = float(tv_diag.get("p_dc_kw", snap.get("p_dc_kw", 0.0)))
	queue_redraw()


func update_ymd_preview(current_params: Dictionary, default_params: Dictionary = {}) -> void:
	var sig := "%.2f|%.1f|%.1f|%.1f|%.2f|%.2f|%.1f" % [
		float(current_params.get("centripetal_fx_gain", 15.0)),
		float(current_params.get("delta_alpha_eq_gain", 2620.0)),
		float(current_params.get("brake_bias_front_pct", 62.0)),
		float(current_params.get("arb_front_balance_pct", 50.0)),
		float(current_params.get("yaw_agility_gain", 1.0)),
		float(current_params.get("friction_ellipse_margin", 0.94)),
		float(current_params.get("beta_soft_deg", 3.0)),
	]
	if sig == _last_params_sig and not _ymd_cur_pts.is_empty():
		return
	_last_params_sig = sig

	var base_res := compute_ymd_envelope(default_params)
	var cur_res := compute_ymd_envelope(current_params)
	_ymd_base_pts = base_res["hull"]
	ymd_base_ay_max_g = float(base_res["ay_max_g"])
	ymd_base_stability_nm_deg = float(base_res["stability_nm_deg"])

	_ymd_cur_pts = cur_res["hull"]
	ymd_ay_max_g = float(cur_res["ay_max_g"])
	ymd_mz_ctrl_nm = float(cur_res["mz_ctrl_nm"])
	ymd_stability_nm_deg = float(cur_res["stability_nm_deg"])
	queue_redraw()


static func _eval_ymd_point(beta_deg: float, delta_deg: float, params: Dictionary) -> Vector2:
	# Analytical TeR27 4WD YMD model at v = 55 km/h (15.28 m/s)
	var m := 235.0
	var g := 9.81
	var v := 15.28
	var a := 0.783
	var b := 0.752
	var wb := a + b
	var fz_static := m * g
	var fz_aero := 0.5 * 1.225 * 4.2 * v * v
	var fz_tot := fz_static + fz_aero

	var arb_f := clampf(float(params.get("arb_front_balance_pct", 50.0)) * 0.01, 0.20, 0.80)
	var brake_bias_f := clampf(float(params.get("brake_bias_front_pct", 62.0)) * 0.01, 0.40, 0.75)
	var cent_fx := clampf(float(params.get("centripetal_fx_gain", 15.0)), 0.0, 50.0)
	var d_alpha_eq := clampf(float(params.get("delta_alpha_eq_gain", 2620.0)), 0.0, 4000.0)
	var yaw_agil := clampf(float(params.get("yaw_agility_gain", 1.0)), 0.50, 2.00)
	var ell_margin := clampf(float(params.get("friction_ellipse_margin", 0.94)), 0.70, 1.05)
	var beta_soft := clampf(float(params.get("beta_soft_deg", 3.0)), 1.0, 10.0)

	var fz_f := fz_tot * (b / wb)
	var fz_r := fz_tot * (a / wb)
	var beta := deg_to_rad(beta_deg)
	var delta := deg_to_rad(delta_deg)

	# Slip angles at zero yaw-rate (standard Milliken YMD)
	var alpha_f := delta - beta
	var alpha_r := -beta

	# Tire load sensitivity & ARB roll distribution
	var mu_base := 1.48 * (0.85 + 0.15 * clampf(ell_margin, 0.70, 1.05))
	var mu_f := mu_base * (1.0 - 0.12 * (arb_f - 0.50))
	var mu_r := mu_base * (1.0 + 0.12 * (arb_f - 0.50))

	# Simplified MF 6.2 lateral force
	var fy_f := mu_f * fz_f * sin(1.38 * atan(10.5 * alpha_f))
	var fy_r := mu_r * fz_r * sin(1.38 * atan(10.5 * alpha_r))

	# Centripetal Fx*sin(delta) contribution from front motors
	var fx_f_cent := cent_fx * 18.0 * sin(absf(delta)) * signf(delta)
	var fy_cent := fx_f_cent * sin(absf(delta))

	# TV yaw moment: agility FF + delta-alpha equalization + beta stability damping + brake bias balance
	var mz_ff := (yaw_agil - 0.5) * 420.0 * sin(1.25 * atan(6.5 * delta))
	var mz_eq := (d_alpha_eq / 2620.0) * 190.0 * clampf((alpha_f - alpha_r) * 3.5, -1.0, 1.0)
	var mz_beta_stab := -(beta_deg / maxf(beta_soft, 1.0)) * 135.0
	var mz_bb := -(brake_bias_f - 0.60) * 480.0 * sin(atan(5.0 * delta))
	var mz_tv := clampf(mz_ff + mz_eq + mz_beta_stab + mz_bb, -950.0, 950.0)

	var ay_g := (fy_f * cos(delta) + fy_r + fy_cent) / (m * g)
	var mz_tot := a * fy_f * cos(delta) - b * fy_r + mz_tv
	return Vector2(ay_g, mz_tot)


static func compute_ymd_envelope(params: Dictionary) -> Dictionary:
	var betas := [-8.0, -5.0, -3.0, -1.5, 0.0, 1.5, 3.0, 5.0, 8.0]
	var deltas := [-14.0, -9.0, -5.0, -2.0, 0.0, 2.0, 5.0, 9.0, 14.0]

	var ay_max := 0.0
	var mz_ctrl := 0.0
	# Build outer boundary polygon from perimeter of (beta, delta) grid
	var hull := PackedVector2Array()
	# Top edge: beta = -8, delta from -14..+14
	for d_deg in deltas:
		var pt := _eval_ymd_point(-8.0, float(d_deg), params)
		hull.append(pt)
		ay_max = maxf(ay_max, absf(pt.x))
	# Right edge: delta = +14, beta from -8..+8
	for b_deg in betas:
		var pt := _eval_ymd_point(float(b_deg), 14.0, params)
		hull.append(pt)
		ay_max = maxf(ay_max, absf(pt.x))
	# Bottom edge: beta = +8, delta from +14..-14
	for di in range(deltas.size() - 1, -1, -1):
		var pt := _eval_ymd_point(8.0, float(deltas[di]), params)
		hull.append(pt)
		ay_max = maxf(ay_max, absf(pt.x))
	# Left edge: delta = -14, beta from +8..-8
	for bi in range(betas.size() - 1, -1, -1):
		var pt := _eval_ymd_point(float(betas[bi]), -14.0, params)
		hull.append(pt)
		ay_max = maxf(ay_max, absf(pt.x))

	# Control reserve Mz at beta = 0, delta = +14 deg
	var pt_ctrl := _eval_ymd_point(0.0, 14.0, params)
	mz_ctrl = absf(pt_ctrl.y)
	# Stability margin -dMz/dbeta [Nm/deg] at delta = 0 around beta = 0
	var pt_bp := _eval_ymd_point(2.0, 0.0, params)
	var pt_bm := _eval_ymd_point(-2.0, 0.0, params)
	var stab_nm_deg := -(pt_bp.y - pt_bm.y) / 4.0

	return {
		"hull": hull,
		"ay_max_g": ay_max,
		"mz_ctrl_nm": mz_ctrl,
		"stability_nm_deg": stab_nm_deg,
	}


func _draw() -> void:
	var w := size.x
	var h := size.y
	if w < 120.0 or h < 120.0:
		return

	var font := ThemeDB.fallback_font
	draw_rect(Rect2(0, 0, w, h), Color(0.08, 0.10, 0.14, 0.95), true)
	draw_rect(Rect2(0, 0, w, h), Color(0.24, 0.30, 0.38, 1.0), false, 1.0)

	var split_y := h * 0.49
	_draw_friction_and_power(Rect2(6.0, 4.0, w - 12.0, split_y - 6.0), font)
	draw_line(Vector2(6.0, split_y), Vector2(w - 6.0, split_y), Color(0.24, 0.30, 0.38, 0.9), 1.0)
	_draw_ymd_preview(Rect2(6.0, split_y + 4.0, w - 12.0, h - split_y - 8.0), font)


func _draw_friction_and_power(rect: Rect2, font: Font) -> void:
	if font != null:
		draw_string(font, Vector2(rect.position.x + 2.0, rect.position.y + 11.0),
			"FRICTION CIRCLES (FL/FR/RL/RR) | Mz_TV & HV POWER",
			HORIZONTAL_ALIGNMENT_LEFT, -1, 10, Color(0.74, 0.90, 1.0))

	# 4 friction circles across top row
	var circ_y := rect.position.y + 42.0
	var slot_w := rect.size.x * 0.25
	var rad := minf(slot_w * 0.28, 22.0)

	for i in range(4):
		var cx := rect.position.x + (float(i) + 0.5) * slot_w
		var center := Vector2(cx, circ_y)
		var sat := tire_sat_pct[i]
		var col := Color(0.16, 0.84, 0.46) if sat < 70.0 else (Color(0.98, 0.68, 0.10) if sat <= 90.0 else Color(0.96, 0.24, 0.24))

		# Outer circle (100% grip) & inner 70% ring
		draw_arc(center, rad, 0.0, TAU, 28, Color(0.35, 0.42, 0.52, 0.9), 1.2)
		draw_arc(center, rad * 0.70, 0.0, TAU, 20, Color(0.22, 0.28, 0.36, 0.7), 1.0)
		draw_line(Vector2(cx - rad, circ_y), Vector2(cx + rad, circ_y), Color(0.22, 0.28, 0.36, 0.6), 1.0)
		draw_line(Vector2(cx, circ_y - rad), Vector2(cx, circ_y + rad), Color(0.22, 0.28, 0.36, 0.6), 1.0)

		# Live force vector (Fy horizontal, Fx vertical)
		var vx := clampf(tire_fy_norm[i], -1.1, 1.1) * rad
		var vy := -clampf(tire_fx_norm[i], -1.1, 1.1) * rad
		var tip := center + Vector2(vx, vy)
		draw_line(center, tip, col, 1.8)
		draw_circle(tip, 3.0, col)

		if font != null:
			draw_string(font, Vector2(cx - rad, circ_y - rad - 3.0), "%s %2.0f%%" % [CORNER_NAMES[i], sat], HORIZONTAL_ALIGNMENT_LEFT, -1, 9, col)
			draw_string(font, Vector2(cx - rad, circ_y + rad + 10.0), "%+.0fNm" % wheel_torques_nm[i], HORIZONTAL_ALIGNMENT_LEFT, -1, 9, Color(0.78, 0.84, 0.92))

	# Bottom dual bars: Mz_TV (-900..+900 Nm) and P_dc (-25..+80 kW)
	var bar_y := rect.position.y + rect.size.y - 26.0
	var half_w := (rect.size.x - 10.0) * 0.5

	# Mz_TV bar
	var mz_rect := Rect2(rect.position.x, bar_y, half_w, 10.0)
	draw_rect(mz_rect, Color(0.14, 0.17, 0.22, 1.0), true)
	var mz_zero := mz_rect.position.x + mz_rect.size.x * 0.5
	var mz_frac := clampf(mz_tv_nm / 900.0, -1.0, 1.0)
	var mz_w := absf(mz_frac) * (mz_rect.size.x * 0.5)
	var mz_x := mz_zero if mz_frac >= 0.0 else (mz_zero - mz_w)
	draw_rect(Rect2(mz_x, bar_y + 1.0, mz_w, 8.0), Color(0.22, 0.84, 0.96), true)
	draw_line(Vector2(mz_zero, bar_y), Vector2(mz_zero, bar_y + 10.0), Color(0.88, 0.92, 0.96), 1.0)

	# HV Power bar (-25..+80 kW)
	var p_rect := Rect2(rect.position.x + half_w + 10.0, bar_y, half_w, 10.0)
	draw_rect(p_rect, Color(0.14, 0.17, 0.22, 1.0), true)
	var p_zero := p_rect.position.x + p_rect.size.x * (25.0 / 105.0)
	var p_clamped := clampf(p_dc_kw, -25.0, 80.0)
	if p_clamped >= 0.0:
		var pw := (p_clamped / 80.0) * (p_rect.size.x * (80.0 / 105.0))
		var pcol := Color(0.96, 0.24, 0.22) if p_clamped > 74.0 else Color(0.96, 0.64, 0.14)
		draw_rect(Rect2(p_zero, bar_y + 1.0, pw, 8.0), pcol, true)
	else:
		var pw := (-p_clamped / 25.0) * (p_rect.size.x * (25.0 / 105.0))
		draw_rect(Rect2(p_zero - pw, bar_y + 1.0, pw, 8.0), Color(0.18, 0.82, 0.94), true)
	draw_line(Vector2(p_zero, bar_y), Vector2(p_zero, bar_y + 10.0), Color(0.88, 0.92, 0.96), 1.0)

	if font != null:
		draw_string(font, Vector2(mz_rect.position.x, bar_y + 21.0), "Mz_TV: %+6.1f Nm" % mz_tv_nm, HORIZONTAL_ALIGNMENT_LEFT, -1, 9, Color(0.84, 0.90, 0.98))
		draw_string(font, Vector2(p_rect.position.x, bar_y + 21.0), "HV P_dc: %+5.1f kW [-25..+80]" % p_dc_kw, HORIZONTAL_ALIGNMENT_LEFT, -1, 9, Color(0.84, 0.90, 0.98))


func _draw_ymd_preview(rect: Rect2, font: Font) -> void:
	if font != null:
		draw_string(font, Vector2(rect.position.x + 2.0, rect.position.y + 10.0),
			"INSTANT YMD PREVIEW (Mz vs ay @ 55 km/h)",
			HORIZONTAL_ALIGNMENT_LEFT, -1, 10, Color(0.74, 0.90, 1.0))

	var px0 := rect.position.x + 34.0
	var py0 := rect.position.y + 16.0
	var pw := maxf(rect.size.x - 42.0, 40.0)
	var ph := maxf(rect.size.y - 34.0, 30.0)
	var cx := px0 + pw * 0.5
	var cy := py0 + ph * 0.5

	# Axes
	var ax_c := Color(0.36, 0.44, 0.54, 0.85)
	draw_line(Vector2(px0, cy), Vector2(px0 + pw, cy), ax_c, 1.0)
	draw_line(Vector2(cx, py0), Vector2(cx, py0 + ph), ax_c, 1.0)

	var ay_lim := 2.4
	var mz_lim := 2200.0

	# Baseline envelope (grey)
	if _ymd_base_pts.size() >= 3:
		var scr_b := PackedVector2Array()
		for pt in _ymd_base_pts:
			var sx := cx + clampf(pt.x / ay_lim, -1.0, 1.0) * (pw * 0.48)
			var sy := cy - clampf(pt.y / mz_lim, -1.0, 1.0) * (ph * 0.46)
			scr_b.append(Vector2(sx, sy))
		scr_b.append(scr_b[0])
		draw_polyline(scr_b, Color(0.55, 0.62, 0.70, 0.55), 1.2)

	# Current envelope (cyan)
	if _ymd_cur_pts.size() >= 3:
		var scr_c := PackedVector2Array()
		for pt in _ymd_cur_pts:
			var sx := cx + clampf(pt.x / ay_lim, -1.0, 1.0) * (pw * 0.48)
			var sy := cy - clampf(pt.y / mz_lim, -1.0, 1.0) * (ph * 0.46)
			scr_c.append(Vector2(sx, sy))
		draw_colored_polygon(scr_c, Color(0.16, 0.80, 0.96, 0.14))
		scr_c.append(scr_c[0])
		draw_polyline(scr_c, Color(0.22, 0.88, 0.98, 0.95), 1.8)

	if font != null:
		draw_string(font, Vector2(rect.position.x + 2.0, py0 + 8.0), "+Mz", HORIZONTAL_ALIGNMENT_LEFT, -1, 9, Color(0.68, 0.76, 0.84))
		draw_string(font, Vector2(px0 + pw - 22.0, cy - 3.0), "+ay", HORIZONTAL_ALIGNMENT_LEFT, -1, 9, Color(0.68, 0.76, 0.84))
		draw_string(font, Vector2(rect.position.x + 2.0, rect.position.y + rect.size.y - 2.0),
			"ay_max: %.2fg (base %.2fg) | Mz_ctrl: %.0fNm | Stab: %+.0f Nm/deg" % [
				ymd_ay_max_g, ymd_base_ay_max_g, ymd_mz_ctrl_nm, ymd_stability_nm_deg
			],
			HORIZONTAL_ALIGNMENT_LEFT, -1, 9, Color(0.86, 0.92, 0.98))
