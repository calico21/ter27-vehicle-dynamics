extends CanvasLayer
class_name TelemetryInspectorOverlay

## Real-Time Engineering Telemetry Inspector & ECU Torque Vectoring V3 X-Ray (F11 / F12).
## Zero-cost when hidden (updates only when `visible == true`).

const BUFFER_LEN: int = 300 # 5 seconds @ 60 Hz

const CHANNEL_GROUPS: Array[Dictionary] = [
	{
		"id": "yaw_beta",
		"label": "Guiñada & Deriva",
		"traces": [
			{"key": "wz_ref", "name": "wz_ref [rad/s]", "color": Color(0.25, 0.90, 1.00)},
			{"key": "wz_real", "name": "wz_real [rad/s]", "color": Color(0.30, 1.00, 0.55)},
			{"key": "beta_real", "name": "beta_real [deg]", "color": Color(1.00, 0.78, 0.25)},
			{"key": "beta_ekf", "name": "beta_ekf [deg]", "color": Color(1.00, 0.42, 0.45)}
		]
	},
	{
		"id": "tv_torques",
		"label": "Momentos & Pares TV",
		"traces": [
			{"key": "mz_req_scaled", "name": "Mz_req/10 [Nm]", "color": Color(0.25, 0.90, 1.00)},
			{"key": "t_fl", "name": "T_FL [Nm]", "color": Color(0.30, 1.00, 0.55)},
			{"key": "t_fr", "name": "T_FR [Nm]", "color": Color(1.00, 0.80, 0.25)},
			{"key": "t_rl", "name": "T_RL [Nm]", "color": Color(0.70, 0.55, 1.00)},
			{"key": "t_rr", "name": "T_RR [Nm]", "color": Color(1.00, 0.45, 0.55)}
		]
	},
	{
		"id": "slips",
		"label": "Deslizamientos",
		"traces": [
			{"key": "k_fl", "name": "kappa_FL [%]", "color": Color(0.25, 0.90, 1.00)},
			{"key": "k_fr", "name": "kappa_FR [%]", "color": Color(0.30, 1.00, 0.55)},
			{"key": "k_rl", "name": "kappa_RL [%]", "color": Color(1.00, 0.80, 0.25)},
			{"key": "k_rr", "name": "kappa_RR [%]", "color": Color(1.00, 0.45, 0.55)},
			{"key": "a_fl", "name": "alpha_FL [deg]", "color": Color(0.60, 0.80, 1.00)},
			{"key": "a_rl", "name": "alpha_RL [deg]", "color": Color(0.85, 0.65, 1.00)}
		]
	},
	{
		"id": "accel_susp",
		"label": "Aceleraciones & Suspensión",
		"traces": [
			{"key": "ax_g", "name": "ax [g]", "color": Color(0.30, 1.00, 0.55)},
			{"key": "ay_g", "name": "ay [g]", "color": Color(0.25, 0.90, 1.00)},
			{"key": "roll_deg", "name": "roll [deg]", "color": Color(1.00, 0.80, 0.25)},
			{"key": "pitch_deg", "name": "pitch [deg]", "color": Color(1.00, 0.45, 0.55)},
			{"key": "dz_fl_mm", "name": "dz_FL [mm]", "color": Color(0.70, 0.85, 1.00)},
			{"key": "dz_rl_mm", "name": "dz_RL [mm]", "color": Color(0.85, 0.65, 1.00)}
		]
	},
	{
		"id": "power_brake",
		"label": "Potencia & Frenada",
		"traces": [
			{"key": "p_elec_kw", "name": "P_elec [kW]", "color": Color(0.25, 0.90, 1.00)},
			{"key": "p_regen_kw", "name": "P_regen [kW]", "color": Color(0.30, 1.00, 0.55)},
			{"key": "v_dc_scaled", "name": "V_dc/10 [V]", "color": Color(1.00, 0.80, 0.25)},
			{"key": "t_brk_hyd", "name": "T_brk_hyd [Nm]", "color": Color(1.00, 0.45, 0.55)},
			{"key": "t_brk_reg", "name": "T_brk_reg [Nm]", "color": Color(0.70, 0.55, 1.00)}
		]
	}
]


class MzBreakdownBar extends Control:
	var ff_mz: float = 0.0
	var p_mz: float = 0.0
	var i_mz: float = 0.0
	var beta_mz: float = 0.0
	var mz_raw: float = 0.0
	var mz_req: float = 0.0

	func _draw() -> void:
		var sz := size
		draw_rect(Rect2(Vector2.ZERO, sz), Color(0.08, 0.10, 0.14, 0.95), true)
		draw_rect(Rect2(Vector2.ZERO, sz), Color(0.22, 0.30, 0.40, 0.85), false, 1.0)
		var mid_x := sz.x * 0.5
		draw_line(Vector2(mid_x, 2.0), Vector2(mid_x, sz.y - 2.0), Color(0.65, 0.72, 0.82, 0.5), 1.0)

		var scale_nm: float = maxf(1200.0, maxf(absf(mz_raw), absf(mz_req)) * 1.1)
		var half_w := (sz.x - 16.0) * 0.5
		var pos_cursor := mid_x
		var neg_cursor := mid_x
		var terms := [
			[ff_mz, Color(0.25, 0.90, 1.00, 0.90)],
			[p_mz, Color(0.30, 1.00, 0.55, 0.90)],
			[i_mz, Color(1.00, 0.80, 0.25, 0.90)],
			[beta_mz, Color(1.00, 0.42, 0.48, 0.90)]
		]
		var bar_h := sz.y * 0.46
		for t in terms:
			var val := float(t[0])
			var col: Color = t[1]
			var w := (absf(val) / scale_nm) * half_w
			if w < 0.5:
				continue
			if val >= 0.0:
				draw_rect(Rect2(pos_cursor, 4.0, w, bar_h), col, true)
				pos_cursor += w
			else:
				draw_rect(Rect2(neg_cursor - w, 4.0, w, bar_h), col, true)
				neg_cursor -= w

		# Final saturated Mz_req bar on lower track
		var req_w := clampf(absf(mz_req) / scale_nm, 0.0, 1.0) * half_w
		var req_x := mid_x if mz_req >= 0.0 else mid_x - req_w
		draw_rect(Rect2(req_x, 6.0 + bar_h, req_w, sz.y - bar_h - 10.0), Color(0.95, 0.97, 1.00, 0.90), true)


class FrictionCirclesView extends Control:
	var wheel_fx: Array[float] = [0.0, 0.0, 0.0, 0.0]
	var wheel_fy: Array[float] = [0.0, 0.0, 0.0, 0.0]
	var wheel_fz: Array[float] = [700.0, 700.0, 700.0, 700.0]
	var wheel_mu: Array[float] = [1.4, 1.4, 1.4, 1.4]

	func _draw() -> void:
		var sz := size
		var cell_w := sz.x * 0.5
		var cell_h := sz.y * 0.5
		var names := ["FL", "FR", "RL", "RR"]
		var font := ThemeDB.fallback_font
		for idx in range(4):
			var col_idx := idx % 2
			var row_idx := idx / 2
			var center := Vector2((float(col_idx) + 0.5) * cell_w, (float(row_idx) + 0.54) * cell_h)
			var r := minf(cell_w, cell_h) * 0.36
			var f_max := maxf(wheel_fz[idx] * wheel_mu[idx], 100.0)
			var fx := wheel_fx[idx]
			var fy := wheel_fy[idx]
			var f_res := sqrt(fx * fx + fy * fy)
			var sat := clampf(f_res / f_max, 0.0, 1.5)
			var ring_col := Color(0.28, 0.85, 0.95, 0.75)
			if sat > 0.95:
				ring_col = Color(1.00, 0.38, 0.42, 0.90)
			elif sat > 0.80:
				ring_col = Color(1.00, 0.78, 0.25, 0.85)

			draw_circle(center, r, Color(0.06, 0.08, 0.12, 0.92))
			draw_arc(center, r, 0.0, TAU, 36, ring_col, 1.6)
			draw_line(center - Vector2(r, 0.0), center + Vector2(r, 0.0), Color(0.30, 0.38, 0.50, 0.45), 1.0)
			draw_line(center - Vector2(0.0, r), center + Vector2(0.0, r), Color(0.30, 0.38, 0.50, 0.45), 1.0)

			# ISO +X fwd (up), +Y left (left on screen)
			var vec := Vector2(-fy / f_max, -fx / f_max)
			if vec.length() > 1.25:
				vec = vec.normalized() * 1.25
			var tip := center + vec * r
			draw_line(center, tip, Color(0.35, 1.00, 0.58), 2.2)
			draw_circle(tip, 3.2, Color(1.0, 0.95, 0.45))

			var lbl := "%s %.0f%%" % [names[idx], sat * 100.0]
			draw_string(font, Vector2(float(col_idx) * cell_w + 6.0, float(row_idx) * cell_h + 14.0), lbl, HORIZONTAL_ALIGNMENT_LEFT, -1, 11, Color(0.88, 0.94, 1.0))


class OscilloscopeView extends Control:
	var owner_overlay: TelemetryInspectorOverlay = null
	var hover_pos: Vector2 = Vector2(-1.0, -1.0)

	func _gui_input(event: InputEvent) -> void:
		if event is InputEventMouseMotion:
			hover_pos = event.position
			queue_redraw()

	func _notification(what: int) -> void:
		if what == NOTIFICATION_MOUSE_EXIT:
			hover_pos = Vector2(-1.0, -1.0)
			queue_redraw()

	func _draw() -> void:
		var sz := size
		draw_rect(Rect2(Vector2.ZERO, sz), Color(0.05, 0.07, 0.10, 0.95), true)
		draw_rect(Rect2(Vector2.ZERO, sz), Color(0.20, 0.30, 0.42, 0.80), false, 1.0)
		for g in range(1, 4):
			var gy := sz.y * float(g) / 4.0
			draw_line(Vector2(0.0, gy), Vector2(sz.x, gy), Color(0.22, 0.28, 0.36, 0.35), 1.0)
		if owner_overlay == null:
			return
		var samples: Array[Dictionary] = owner_overlay.get_ordered_samples()
		if samples.size() < 2:
			return
		var group: Dictionary = CHANNEL_GROUPS[owner_overlay.active_group_idx]
		var traces: Array = group["traces"]

		# Compute shared min/max across active traces
		var y_min := 1e9
		var y_max := -1e9
		for s in samples:
			for tr in traces:
				var v := float(s.get(tr["key"], 0.0))
				y_min = minf(y_min, v)
				y_max = maxf(y_max, v)
		if y_max - y_min < 0.2:
			var mid := 0.5 * (y_max + y_min)
			y_min = mid - 0.5
			y_max = mid + 0.5
		else:
			var pad := 0.08 * (y_max - y_min)
			y_min -= pad
			y_max += pad

		var n := samples.size()
		var font := ThemeDB.fallback_font
		for tr in traces:
			var pts := PackedVector2Array()
			pts.resize(n)
			var k: String = tr["key"]
			for idx in range(n):
				var px := (float(idx) / float(maxi(n - 1, 1))) * sz.x
				var v := float(samples[idx].get(k, 0.0))
				var py := sz.y - ((v - y_min) / (y_max - y_min)) * (sz.y - 8.0) - 4.0
				pts[idx] = Vector2(px, py)
			draw_polyline(pts, tr["color"], 1.6, true)

		# Legend top-left
		var lx := 8.0
		for tr in traces:
			draw_string(font, Vector2(lx, 14.0), str(tr["name"]), HORIZONTAL_ALIGNMENT_LEFT, -1, 10, tr["color"])
			lx += 118.0

		# Interactive hover cursor
		if hover_pos.x >= 0.0 and hover_pos.x <= sz.x:
			var s_idx := clampi(int(round((hover_pos.x / maxf(sz.x, 1.0)) * float(n - 1))), 0, n - 1)
			var cx := (float(s_idx) / float(maxi(n - 1, 1))) * sz.x
			draw_line(Vector2(cx, 0.0), Vector2(cx, sz.y), Color(0.95, 0.98, 1.0, 0.75), 1.0)
			var samp: Dictionary = samples[s_idx]
			var readout := "t=%.2fs" % float(samp.get("t", 0.0))
			for tr in traces:
				readout += " | %s=%.2f" % [str(tr["name"]).split(" ")[0], float(samp.get(tr["key"], 0.0))]
			draw_rect(Rect2(6.0, sz.y - 20.0, sz.x - 12.0, 18.0), Color(0.03, 0.05, 0.08, 0.88), true)
			draw_string(font, Vector2(10.0, sz.y - 6.0), readout, HORIZONTAL_ALIGNMENT_LEFT, -1, 11, Color(0.95, 0.98, 1.0))


var frozen: bool = false
var active_group_idx: int = 0
var _sim_time: float = 0.0
var _sample_acc: float = 0.0
var _ring: Array[Dictionary] = []
var _ring_head: int = 0
var _ring_count: int = 0

var _mono_font: SystemFont = null
var _freeze_btn: Button = null
var _vars_toggle_btn: Button = null
var _vars_drawer: PanelContainer = null
var _vars_filter_edit: LineEdit = null
var _vars_label: RichTextLabel = null

var _yaw_label: Label = null
var _mz_label: Label = null
var _mz_bar: MzBreakdownBar = null
var _cascade_label: Label = null
var _friction_view: FrictionCirclesView = null
var _wheels_label: Label = null
var _scope_view: OscilloscopeView = null
var _group_buttons: Array[Button] = []

var _latest_vars: Array[Array] = []


func _ready() -> void:
	layer = 12
	_mono_font = SystemFont.new()
	_mono_font.font_names = PackedStringArray(["JetBrains Mono", "Fira Code", "DejaVu Sans Mono", "Monospace"])
	_ring.resize(BUFFER_LEN)
	for i in range(BUFFER_LEN):
		_ring[i] = {}
	_build_ui()
	visible = false


func toggle_visibility() -> void:
	visible = not visible


func toggle_freeze() -> void:
	frozen = not frozen
	if _freeze_btn != null:
		_freeze_btn.text = "[F12] REANUDAR" if frozen else "[F12] CONGELAR"
		_freeze_btn.modulate = Color(1.0, 0.72, 0.35) if frozen else Color(0.85, 0.96, 1.0)


func is_text_input_focused() -> bool:
	return _vars_filter_edit != null and _vars_filter_edit.has_focus()


func get_ordered_samples() -> Array[Dictionary]:
	var out: Array[Dictionary] = []
	if _ring_count == 0:
		return out
	var start := (_ring_head - _ring_count + BUFFER_LEN) % BUFFER_LEN
	for i in range(_ring_count):
		out.append(_ring[(start + i) % BUFFER_LEN])
	return out


func _make_panel_style(bg: Color = Color(0.05, 0.07, 0.11, 0.86)) -> StyleBoxFlat:
	var st := StyleBoxFlat.new()
	st.bg_color = bg
	st.border_color = Color(0.18, 0.45, 0.62, 0.72)
	st.set_border_width_all(1)
	st.set_corner_radius_all(6)
	st.content_margin_left = 8
	st.content_margin_right = 8
	st.content_margin_top = 6
	st.content_margin_bottom = 6
	return st


func _apply_mono(ctrl: Control, sz: int = 11) -> void:
	if _mono_font != null:
		ctrl.add_theme_font_override("font", _mono_font)
	ctrl.add_theme_font_size_override("font_size", sz)


func _build_ui() -> void:
	var root := Control.new()
	root.set_anchors_preset(Control.PRESET_FULL_RECT)
	root.mouse_filter = Control.MOUSE_FILTER_IGNORE
	add_child(root)

	# Left Panel — ECU_TV_V3 X-Ray (top-left below minimal HUD, leaving center clear)
	var left_panel := PanelContainer.new()
	left_panel.anchor_left = 0.0
	left_panel.anchor_right = 0.31
	left_panel.anchor_top = 0.09
	left_panel.anchor_bottom = 0.68
	left_panel.offset_left = 12
	left_panel.offset_right = 0
	left_panel.add_theme_stylebox_override("panel", _make_panel_style())
	root.add_child(left_panel)

	var left_vb := VBoxContainer.new()
	left_vb.add_theme_constant_override("separation", 4)
	left_panel.add_child(left_vb)

	var hdr := HBoxContainer.new()
	left_vb.add_child(hdr)
	var title := Label.new()
	title.text = "RAYOS X ECU_TV_V3 [F11]"
	title.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	title.add_theme_color_override("font_color", Color(0.28, 0.90, 1.0))
	_apply_mono(title, 12)
	hdr.add_child(title)

	_freeze_btn = Button.new()
	_freeze_btn.text = "[F12] CONGELAR"
	_freeze_btn.focus_mode = Control.FOCUS_NONE
	_apply_mono(_freeze_btn, 10)
	_freeze_btn.pressed.connect(toggle_freeze)
	hdr.add_child(_freeze_btn)

	_vars_toggle_btn = Button.new()
	_vars_toggle_btn.text = "Todas las Vars"
	_vars_toggle_btn.focus_mode = Control.FOCUS_NONE
	_apply_mono(_vars_toggle_btn, 10)
	_vars_toggle_btn.pressed.connect(func():
		if _vars_drawer != null:
			_vars_drawer.visible = not _vars_drawer.visible
	)
	hdr.add_child(_vars_toggle_btn)

	_yaw_label = Label.new()
	_apply_mono(_yaw_label, 11)
	_yaw_label.add_theme_color_override("font_color", Color(0.88, 0.95, 1.0))
	left_vb.add_child(_yaw_label)

	_mz_label = Label.new()
	_apply_mono(_mz_label, 11)
	_mz_label.add_theme_color_override("font_color", Color(1.0, 0.85, 0.45))
	left_vb.add_child(_mz_label)

	_mz_bar = MzBreakdownBar.new()
	_mz_bar.custom_minimum_size = Vector2(0, 26)
	left_vb.add_child(_mz_bar)

	_cascade_label = Label.new()
	_apply_mono(_cascade_label, 10)
	_cascade_label.add_theme_color_override("font_color", Color(0.75, 0.98, 0.82))
	left_vb.add_child(_cascade_label)

	# Right Panel — 4 Friction Circles + Wheel Data
	var right_panel := PanelContainer.new()
	right_panel.anchor_left = 0.69
	right_panel.anchor_right = 1.0
	right_panel.anchor_top = 0.02
	right_panel.anchor_bottom = 0.68
	right_panel.offset_right = -12
	right_panel.add_theme_stylebox_override("panel", _make_panel_style())
	root.add_child(right_panel)

	var right_vb := VBoxContainer.new()
	right_vb.add_theme_constant_override("separation", 4)
	right_panel.add_child(right_vb)

	var r_title := Label.new()
	r_title.text = "CÍRCULOS DE FRICCIÓN & RUEDAS"
	r_title.add_theme_color_override("font_color", Color(0.28, 0.90, 1.0))
	_apply_mono(r_title, 12)
	right_vb.add_child(r_title)

	_friction_view = FrictionCirclesView.new()
	_friction_view.custom_minimum_size = Vector2(0, 220)
	_friction_view.size_flags_vertical = Control.SIZE_EXPAND_FILL
	right_vb.add_child(_friction_view)

	_wheels_label = Label.new()
	_apply_mono(_wheels_label, 10)
	_wheels_label.add_theme_color_override("font_color", Color(0.88, 0.94, 0.98))
	right_vb.add_child(_wheels_label)

	# Bottom Panel — Multi-Channel Oscilloscope (5 s @ 60 Hz)
	var bot_panel := PanelContainer.new()
	bot_panel.anchor_left = 0.0
	bot_panel.anchor_right = 1.0
	bot_panel.anchor_top = 0.69
	bot_panel.anchor_bottom = 1.0
	bot_panel.offset_left = 12
	bot_panel.offset_right = -12
	bot_panel.offset_bottom = -10
	bot_panel.add_theme_stylebox_override("panel", _make_panel_style())
	root.add_child(bot_panel)

	var bot_vb := VBoxContainer.new()
	bot_vb.add_theme_constant_override("separation", 4)
	bot_panel.add_child(bot_vb)

	var grp_bar := HBoxContainer.new()
	grp_bar.add_theme_constant_override("separation", 6)
	bot_vb.add_child(grp_bar)

	for i in range(CHANNEL_GROUPS.size()):
		var b := Button.new()
		b.text = CHANNEL_GROUPS[i]["label"]
		b.focus_mode = Control.FOCUS_NONE
		_apply_mono(b, 10)
		var g_idx := i
		b.pressed.connect(func():
			active_group_idx = g_idx
			_refresh_group_buttons()
			if _scope_view != null:
				_scope_view.queue_redraw()
		)
		grp_bar.add_child(b)
		_group_buttons.append(b)
	_refresh_group_buttons()

	_scope_view = OscilloscopeView.new()
	_scope_view.owner_overlay = self
	_scope_view.size_flags_vertical = Control.SIZE_EXPAND_FILL
	bot_vb.add_child(_scope_view)

	# Floating "Todas las Variables" Drawer (center-top when toggled)
	_vars_drawer = PanelContainer.new()
	_vars_drawer.anchor_left = 0.32
	_vars_drawer.anchor_right = 0.68
	_vars_drawer.anchor_top = 0.06
	_vars_drawer.anchor_bottom = 0.67
	_vars_drawer.add_theme_stylebox_override("panel", _make_panel_style(Color(0.04, 0.06, 0.09, 0.94)))
	_vars_drawer.visible = false
	root.add_child(_vars_drawer)

	var dv := VBoxContainer.new()
	dv.add_theme_constant_override("separation", 4)
	_vars_drawer.add_child(dv)

	_vars_filter_edit = LineEdit.new()
	_vars_filter_edit.placeholder_text = "Filtrar variables SI (ej: kappa, fz, ekf, disc, mz)..."
	_apply_mono(_vars_filter_edit, 11)
	_vars_filter_edit.text_changed.connect(func(_t): _refresh_vars_drawer())
	dv.add_child(_vars_filter_edit)

	_vars_label = RichTextLabel.new()
	_vars_label.size_flags_vertical = Control.SIZE_EXPAND_FILL
	_vars_label.bbcode_enabled = false
	_apply_mono(_vars_label, 10)
	dv.add_child(_vars_label)


func _refresh_group_buttons() -> void:
	for i in range(_group_buttons.size()):
		_group_buttons[i].modulate = Color(0.35, 1.0, 0.85) if i == active_group_idx else Color(0.78, 0.84, 0.92)


func update_from_car(car: Node, dt: float) -> void:
	if not visible or frozen or car == null:
		return
	var model = car.get("model")
	if model == null:
		return

	_sim_time += maxf(dt, 0.0)
	var ecu = model.get("ecu_tv_v3")
	var wheels: Array = model.get("wheels") if model.get("wheels") is Array else []
	var hv = model.get("hv_state")
	var b_state = model.get("brake_state")
	var last_res: Dictionary = car.get("_last_step_res") if car.get("_last_step_res") is Dictionary else {}
	var v_ros: Vector3 = model.linear_velocity_ros
	var w_ros: Vector3 = model.angular_velocity_ros
	var beta_real := VehicleTelemetry.compute_beta_deg(v_ros.x, v_ros.y)
	var beta_ekf := rad_to_deg(float(ecu.ekf_beta_est)) if ecu != null else 0.0
	var wz_ref := float(ecu.last_wz_ref) if ecu != null else 0.0
	var wz_ekf := float(ecu.ekf_wz_corrected) if ecu != null else 0.0
	var vy_std := float(ecu.ekf_vy_std) if ecu != null else 0.0
	var beta_std_deg := rad_to_deg(atan2(vy_std, maxf(absf(v_ros.x), 1.0)))

	if _yaw_label != null and ecu != null:
		_yaw_label.text = (
			"wz ref/real/ekf: %+5.2f / %+5.2f / %+5.2f rad/s\n"
			+ "beta real / ekf: %+5.2f / %+5.2f +-%.2f deg\n"
			+ "vy_ekf / vy_ss : %+5.2f / %+5.2f m/s (std=%.3f)\n"
			+ "Gains Kp/Ki/Kd : %.0f / %.0f / %.1f"
		) % [
			wz_ref, w_ros.z, wz_ekf,
			beta_real, beta_ekf, beta_std_deg,
			float(ecu.ekf_x[0]), float(ecu.last_vy_ss), vy_std,
			float(ecu.last_kp), float(ecu.last_ki), float(ecu.last_kd)
		]

	if _mz_label != null and ecu != null:
		_mz_label.text = (
			"Mz FF=%+5.0f P=%+5.0f I=%+5.0f Beta=%+5.0f\n"
			+ "Mz_raw=%+5.0f -> Mz_req=%+5.0f Nm | Sat=%3.0f%% QP=%s"
		) % [
			float(ecu.last_ff_mz), float(ecu.last_p_mz), float(ecu.last_i_mz), float(ecu.last_beta_mz),
			float(ecu.last_mz_raw), float(ecu.last_mz_req),
			float(ecu.mz_sat_ratio) * 100.0, String.num_scientific(float(ecu.qp_residual))
		]

	if _mz_bar != null and ecu != null:
		_mz_bar.ff_mz = float(ecu.last_ff_mz)
		_mz_bar.p_mz = float(ecu.last_p_mz)
		_mz_bar.i_mz = float(ecu.last_i_mz)
		_mz_bar.beta_mz = float(ecu.last_beta_mz)
		_mz_bar.mz_raw = float(ecu.last_mz_raw)
		_mz_bar.mz_req = float(ecu.last_mz_req)
		_mz_bar.queue_redraw()

	if _cascade_label != null and ecu != null:
		var c_lines := "Whl |  T_nom ->  T_QP -> T_cmd | [lb, ub] | TC_cut | mu/k_opt\n"
		var w_names := ["FL ", "FR ", "RL ", "RR "]
		for i in range(4):
			c_lines += "%s | %6.1f -> %6.1f -> %5.1f | [%3.0f,%3.0f] | -%5.1f | %.2f/%.2f\n" % [
				w_names[i],
				float(ecu.last_t_nominal[i]), float(ecu.last_t_qp[i]), float(ecu.last_t_cmd[i]),
				float(ecu.t_lb_filt[i]), float(ecu.t_ub_filt[i]),
				float(ecu.last_tc_cut_nm[i]),
				float(ecu.tc_mu_surface[i]), float(ecu.tc_kappa_opt[i])
			]
		_cascade_label.text = c_lines.strip_edges()

	if wheels.size() >= 4:
		var disc_temps: Array = last_res.get("disc_temps", b_state.disc_temp if b_state != null else [40.0, 40.0, 40.0, 40.0])
		if _friction_view != null:
			for i in range(4):
				_friction_view.wheel_fx[i] = float(wheels[i].F_x)
				_friction_view.wheel_fy[i] = float(wheels[i].F_y)
				_friction_view.wheel_fz[i] = float(wheels[i].F_z)
				_friction_view.wheel_mu[i] = 1.5 * float(wheels[i].surface_mu_scale)
			_friction_view.queue_redraw()
		if _wheels_label != null:
			var wl := "Whl |  Fz  | Fx/Fy   | k[%] | a[°] | RPM  | I/M/O°C | Tdisc | mu\n"
			var wn := ["FL ", "FR ", "RL ", "RR "]
			for i in range(4):
				var w = wheels[i]
				var rpm := float(w.omega) * 60.0 / TAU
				wl += "%s | %4.0f | %4.0f/%4.0f | %4.1f | %4.1f | %4.0f | %2.0f/%2.0f/%2.0f | %5.1f | %.2f\n" % [
					wn[i], float(w.F_z), float(w.F_x), float(w.F_y),
					float(w.kappa) * 100.0, rad_to_deg(float(w.alpha)), rpm,
					float(w.t_zone_inner_c), float(w.t_zone_mid_c), float(w.t_zone_outer_c),
					float(disc_temps[i]) if i < disc_temps.size() else 40.0,
					float(w.surface_mu_scale)
				]
			_wheels_label.text = wl.strip_edges()

	# Sample ring buffer at ~60 Hz
	_sample_acc += maxf(dt, 1.0 / 60.0)
	if _sample_acc >= (1.0 / 60.0) - 1e-5:
		_sample_acc = 0.0
		var drv_trqs: Array = last_res.get("drive_torques", ecu.last_t_cmd if ecu != null else [0.0, 0.0, 0.0, 0.0])
		var dz_arr: Array = [0.0, 0.0, 0.0, 0.0]
		if model.suspension != null and model.suspension.corners.size() >= 4:
			for i in range(4):
				dz_arr[i] = float(model.suspension.corners[i].delta_z) * 1000.0
		var hyd_sum := 0.0
		var reg_sum := 0.0
		if b_state != null:
			for i in range(4):
				hyd_sum += float(b_state.hydraulic_torques[i])
				reg_sum += float(b_state.regen_torques[i])

		_ring[_ring_head] = {
			"t": _sim_time,
			"wz_ref": wz_ref,
			"wz_real": w_ros.z,
			"beta_real": beta_real,
			"beta_ekf": beta_ekf,
			"mz_req_scaled": (float(ecu.last_mz_req) * 0.1) if ecu != null else 0.0,
			"t_fl": float(drv_trqs[0]) if drv_trqs.size() > 0 else 0.0,
			"t_fr": float(drv_trqs[1]) if drv_trqs.size() > 1 else 0.0,
			"t_rl": float(drv_trqs[2]) if drv_trqs.size() > 2 else 0.0,
			"t_rr": float(drv_trqs[3]) if drv_trqs.size() > 3 else 0.0,
			"k_fl": float(wheels[0].kappa) * 100.0 if wheels.size() > 0 else 0.0,
			"k_fr": float(wheels[1].kappa) * 100.0 if wheels.size() > 1 else 0.0,
			"k_rl": float(wheels[2].kappa) * 100.0 if wheels.size() > 2 else 0.0,
			"k_rr": float(wheels[3].kappa) * 100.0 if wheels.size() > 3 else 0.0,
			"a_fl": rad_to_deg(float(wheels[0].alpha)) if wheels.size() > 0 else 0.0,
			"a_rl": rad_to_deg(float(wheels[2].alpha)) if wheels.size() > 2 else 0.0,
			"ax_g": float(model._prev_ax_ros) / 9.81,
			"ay_g": float(model._prev_ay_ros) / 9.81,
			"roll_deg": rad_to_deg(float(model.chassis_state.roll)),
			"pitch_deg": rad_to_deg(float(model.chassis_state.pitch)),
			"dz_fl_mm": dz_arr[0],
			"dz_rl_mm": dz_arr[2],
			"p_elec_kw": float(hv.power_kw) if hv != null else 0.0,
			"p_regen_kw": (float(b_state.p_regen_w) / 1000.0) if b_state != null else 0.0,
			"v_dc_scaled": (float(hv.v_dc) * 0.1) if hv != null else 58.8,
			"t_brk_hyd": hyd_sum,
			"t_brk_reg": reg_sum
		}
		_ring_head = (_ring_head + 1) % BUFFER_LEN
		_ring_count = mini(_ring_count + 1, BUFFER_LEN)
		if _scope_view != null:
			_scope_view.queue_redraw()

	if _vars_drawer != null and _vars_drawer.visible:
		_collect_all_vars(model, ecu, wheels, hv, b_state, v_ros, w_ros, beta_real)
		_refresh_vars_drawer()


func _collect_all_vars(model, ecu, wheels: Array, hv, b_state, v_ros: Vector3, w_ros: Vector3, beta_deg: float) -> void:
	_latest_vars = [
		["vx", "%.3f m/s" % v_ros.x],
		["vy", "%.3f m/s" % v_ros.y],
		["vz", "%.3f m/s" % v_ros.z],
		["ax", "%.3f m/s^2" % float(model._prev_ax_ros)],
		["ay", "%.3f m/s^2" % float(model._prev_ay_ros)],
		["wz_real", "%.4f rad/s" % w_ros.z],
		["beta_real", "%.3f deg" % beta_deg],
		["roll", "%.3f deg" % rad_to_deg(float(model.chassis_state.roll))],
		["pitch", "%.3f deg" % rad_to_deg(float(model.chassis_state.pitch))],
		["pinn_tire", str(TireModel.enable_neural_correction)]
	]
	if hv != null:
		_latest_vars.append(["v_dc", "%.2f V" % float(hv.v_dc)])
		_latest_vars.append(["i_dc", "%.2f A" % float(hv.i_dc)])
		_latest_vars.append(["power_kw", "%.2f kW" % float(hv.power_kw)])
		_latest_vars.append(["soc", "%.2f %%" % (float(hv.soc) * 100.0)])
		_latest_vars.append(["energy_kwh", "%.4f kWh" % float(hv.energy_kwh)])
	if b_state != null:
		_latest_vars.append(["p_regen_kw", "%.2f kW" % (float(b_state.p_regen_w) / 1000.0)])
		_latest_vars.append(["disc_temp_fl..rr", "%.1f/%.1f/%.1f/%.1f C" % [b_state.disc_temp[0], b_state.disc_temp[1], b_state.disc_temp[2], b_state.disc_temp[3]]])
	if ecu != null:
		_latest_vars.append(["ekf_vy", "%.4f m/s" % float(ecu.ekf_x[0])])
		_latest_vars.append(["ekf_vy_ss", "%.4f m/s" % float(ecu.last_vy_ss)])
		_latest_vars.append(["ekf_wz_corr", "%.4f rad/s" % float(ecu.ekf_wz_corrected)])
		_latest_vars.append(["ekf_beta", "%.3f deg" % rad_to_deg(float(ecu.ekf_beta_est))])
		_latest_vars.append(["mz_ff/p/i/beta", "%.1f / %.1f / %.1f / %.1f Nm" % [ecu.last_ff_mz, ecu.last_p_mz, ecu.last_i_mz, ecu.last_beta_mz]])
		_latest_vars.append(["mz_raw/req", "%.1f / %.1f Nm" % [ecu.last_mz_raw, ecu.last_mz_req]])
		_latest_vars.append(["tc_cut_fl..rr", "%.1f/%.1f/%.1f/%.1f Nm" % [ecu.last_tc_cut_nm[0], ecu.last_tc_cut_nm[1], ecu.last_tc_cut_nm[2], ecu.last_tc_cut_nm[3]]])
	for i in range(mini(4, wheels.size())):
		var w = wheels[i]
		_latest_vars.append(["wheel_%d_fz/fx/fy" % i, "%.0f / %.0f / %.0f N" % [w.F_z, w.F_x, w.F_y]])
		_latest_vars.append(["wheel_%d_kappa/alpha" % i, "%.2f%% / %.2f deg" % [w.kappa * 100.0, rad_to_deg(w.alpha)]])
		_latest_vars.append(["wheel_%d_surface_mu" % i, "%.3f" % float(w.surface_mu_scale)])


func _refresh_vars_drawer() -> void:
	if _vars_label == null:
		return
	var filt := _vars_filter_edit.text.strip_edges().to_lower() if _vars_filter_edit != null else ""
	var txt := ""
	for pair in _latest_vars:
		var k := str(pair[0])
		if filt != "" and not k.to_lower().contains(filt):
			continue
		txt += "%-22s : %s\n" % [k, str(pair[1])]
	_vars_label.text = txt
