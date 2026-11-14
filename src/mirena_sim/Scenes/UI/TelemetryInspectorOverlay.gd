extends CanvasLayer
class_name TelemetryInspectorOverlay

## Modular Real-Time Engineering Telemetry Inspector & ECU TV V3 X-Ray (F11 / F12 / F8).
## Zero-cost when hidden (updates only when `visible == true`).

const BUFFER_LEN: int = 300 # 5 seconds @ 60 Hz
const LAYOUT_PATH: String = "user://telemetry_layout.json"

const CHANNEL_CATALOG: Array[Dictionary] = [
	{"key": "wz_ref", "name": "wz_ref [rad/s]", "color": Color(0.25, 0.90, 1.00)},
	{"key": "wz_real", "name": "wz_real [rad/s]", "color": Color(0.30, 1.00, 0.55)},
	{"key": "wz_ekf", "name": "wz_ekf [rad/s]", "color": Color(0.55, 0.85, 1.00)},
	{"key": "beta_real", "name": "beta_real [deg]", "color": Color(1.00, 0.78, 0.25)},
	{"key": "beta_ekf", "name": "beta_ekf [deg]", "color": Color(1.00, 0.42, 0.45)},
	{"key": "ax_g", "name": "ax_g [g]", "color": Color(0.30, 1.00, 0.55)},
	{"key": "ay_g", "name": "ay_g [g]", "color": Color(0.25, 0.90, 1.00)},
	{"key": "roll_deg", "name": "roll_deg [deg]", "color": Color(1.00, 0.80, 0.25)},
	{"key": "pitch_deg", "name": "pitch_deg [deg]", "color": Color(1.00, 0.45, 0.55)},
	{"key": "mz_req", "name": "mz_req [Nm]", "color": Color(0.25, 0.90, 1.00)},
	{"key": "mz_raw", "name": "mz_raw [Nm]", "color": Color(0.75, 0.85, 1.00)},
	{"key": "ff_mz", "name": "ff_mz [Nm]", "color": Color(0.30, 1.00, 0.55)},
	{"key": "p_mz", "name": "p_mz [Nm]", "color": Color(1.00, 0.80, 0.25)},
	{"key": "i_mz", "name": "i_mz [Nm]", "color": Color(0.75, 0.55, 1.00)},
	{"key": "beta_mz", "name": "beta_mz [Nm]", "color": Color(1.00, 0.42, 0.48)},
	{"key": "smc_s", "name": "smc_s [rad/s]", "color": Color(0.95, 0.45, 1.00)},
	{"key": "kappa_lim_FL", "name": "kappa_lim_FL [-]", "color": Color(0.45, 0.95, 0.75)},
	{"key": "kappa_lim_RL", "name": "kappa_lim_RL [-]", "color": Color(0.75, 0.95, 0.45)},
	{"key": "T_cmd_FL", "name": "T_cmd_FL [Nm]", "color": Color(0.30, 1.00, 0.55)},
	{"key": "T_cmd_FR", "name": "T_cmd_FR [Nm]", "color": Color(1.00, 0.80, 0.25)},
	{"key": "T_cmd_RL", "name": "T_cmd_RL [Nm]", "color": Color(0.70, 0.55, 1.00)},
	{"key": "T_cmd_RR", "name": "T_cmd_RR [Nm]", "color": Color(1.00, 0.45, 0.55)},
	{"key": "T_nom_FL", "name": "T_nom_FL [Nm]", "color": Color(0.45, 0.90, 0.70)},
	{"key": "T_nom_FR", "name": "T_nom_FR [Nm]", "color": Color(0.95, 0.85, 0.45)},
	{"key": "T_nom_RL", "name": "T_nom_RL [Nm]", "color": Color(0.78, 0.68, 1.00)},
	{"key": "T_nom_RR", "name": "T_nom_RR [Nm]", "color": Color(0.98, 0.62, 0.68)},
	{"key": "T_qp_FL", "name": "T_qp_FL [Nm]", "color": Color(0.20, 0.85, 0.95)},
	{"key": "T_qp_FR", "name": "T_qp_FR [Nm]", "color": Color(0.95, 0.70, 0.20)},
	{"key": "T_qp_RL", "name": "T_qp_RL [Nm]", "color": Color(0.60, 0.50, 0.95)},
	{"key": "T_qp_RR", "name": "T_qp_RR [Nm]", "color": Color(0.95, 0.35, 0.45)},
	{"key": "TC_cut_FL", "name": "TC_cut_FL [Nm]", "color": Color(1.00, 0.50, 0.30)},
	{"key": "TC_cut_FR", "name": "TC_cut_FR [Nm]", "color": Color(1.00, 0.65, 0.30)},
	{"key": "TC_cut_RL", "name": "TC_cut_RL [Nm]", "color": Color(1.00, 0.40, 0.50)},
	{"key": "TC_cut_RR", "name": "TC_cut_RR [Nm]", "color": Color(1.00, 0.30, 0.35)},
	{"key": "Fz_FL", "name": "Fz_FL [N]", "color": Color(0.35, 0.90, 1.00)},
	{"key": "Fz_FR", "name": "Fz_FR [N]", "color": Color(0.40, 1.00, 0.60)},
	{"key": "Fz_RL", "name": "Fz_RL [N]", "color": Color(1.00, 0.82, 0.30)},
	{"key": "Fz_RR", "name": "Fz_RR [N]", "color": Color(1.00, 0.45, 0.55)},
	{"key": "Fy_FL", "name": "Fy_FL [N]", "color": Color(0.30, 0.85, 1.00)},
	{"key": "Fy_FR", "name": "Fy_FR [N]", "color": Color(0.35, 0.95, 0.55)},
	{"key": "Fy_RL", "name": "Fy_RL [N]", "color": Color(0.95, 0.75, 0.25)},
	{"key": "Fy_RR", "name": "Fy_RR [N]", "color": Color(0.95, 0.40, 0.50)},
	{"key": "Fx_FL", "name": "Fx_FL [N]", "color": Color(0.55, 0.95, 0.85)},
	{"key": "Fx_FR", "name": "Fx_FR [N]", "color": Color(0.65, 1.00, 0.45)},
	{"key": "Fx_RL", "name": "Fx_RL [N]", "color": Color(1.00, 0.68, 0.40)},
	{"key": "Fx_RR", "name": "Fx_RR [N]", "color": Color(0.85, 0.55, 1.00)},
	{"key": "kappa_FL", "name": "kappa_FL [%]", "color": Color(0.25, 0.90, 1.00)},
	{"key": "kappa_FR", "name": "kappa_FR [%]", "color": Color(0.30, 1.00, 0.55)},
	{"key": "kappa_RL", "name": "kappa_RL [%]", "color": Color(1.00, 0.80, 0.25)},
	{"key": "kappa_RR", "name": "kappa_RR [%]", "color": Color(1.00, 0.45, 0.55)},
	{"key": "alpha_FL", "name": "alpha_FL [deg]", "color": Color(0.60, 0.80, 1.00)},
	{"key": "alpha_FR", "name": "alpha_FR [deg]", "color": Color(0.60, 1.00, 0.75)},
	{"key": "alpha_RL", "name": "alpha_RL [deg]", "color": Color(0.85, 0.65, 1.00)},
	{"key": "alpha_RR", "name": "alpha_RR [deg]", "color": Color(1.00, 0.60, 0.75)},
	{"key": "grip_pct_FL", "name": "grip_pct_FL [%]", "color": Color(0.10, 0.85, 0.55)},
	{"key": "grip_pct_FR", "name": "grip_pct_FR [%]", "color": Color(0.95, 0.75, 0.15)},
	{"key": "grip_pct_RL", "name": "grip_pct_RL [%]", "color": Color(0.35, 0.85, 1.00)},
	{"key": "grip_pct_RR", "name": "grip_pct_RR [%]", "color": Color(0.95, 0.35, 0.35)},
	{"key": "dz_FL", "name": "dz_FL [mm]", "color": Color(0.70, 0.85, 1.00)},
	{"key": "dz_FR", "name": "dz_FR [mm]", "color": Color(0.55, 1.00, 0.75)},
	{"key": "dz_RL", "name": "dz_RL [mm]", "color": Color(0.85, 0.65, 1.00)},
	{"key": "dz_RR", "name": "dz_RR [mm]", "color": Color(1.00, 0.65, 0.55)},
	{"key": "T_tire_mid_FL", "name": "T_tire_mid_FL [C]", "color": Color(0.95, 0.55, 0.35)},
	{"key": "T_tire_mid_FR", "name": "T_tire_mid_FR [C]", "color": Color(0.95, 0.75, 0.35)},
	{"key": "T_tire_mid_RL", "name": "T_tire_mid_RL [C]", "color": Color(0.80, 0.45, 0.95)},
	{"key": "T_tire_mid_RR", "name": "T_tire_mid_RR [C]", "color": Color(0.95, 0.35, 0.55)},
	{"key": "T_disc_FL", "name": "T_disc_FL [C]", "color": Color(1.00, 0.45, 0.25)},
	{"key": "T_disc_FR", "name": "T_disc_FR [C]", "color": Color(1.00, 0.65, 0.25)},
	{"key": "T_disc_RL", "name": "T_disc_RL [C]", "color": Color(0.95, 0.35, 0.65)},
	{"key": "T_disc_RR", "name": "T_disc_RR [C]", "color": Color(0.85, 0.35, 0.85)},
	{"key": "P_elec_kw", "name": "P_elec_kw [kW]", "color": Color(0.25, 0.90, 1.00)},
	{"key": "P_regen_kw", "name": "P_regen_kw [kW]", "color": Color(0.30, 1.00, 0.55)},
	{"key": "V_dc", "name": "V_dc [V]", "color": Color(1.00, 0.80, 0.25)},
	{"key": "T_brk_hyd", "name": "T_brk_hyd [Nm]", "color": Color(1.00, 0.45, 0.55)},
	{"key": "T_brk_reg", "name": "T_brk_reg [Nm]", "color": Color(0.70, 0.55, 1.00)}
]


static func _make_style(bg: Color = Color(0.05, 0.07, 0.11, 0.86), border: Color = Color(0.18, 0.45, 0.62, 0.72)) -> StyleBoxFlat:
	var st := StyleBoxFlat.new()
	st.bg_color = bg
	st.border_color = border
	st.set_border_width_all(1)
	st.set_corner_radius_all(6)
	st.content_margin_left = 7
	st.content_margin_right = 7
	st.content_margin_top = 5
	st.content_margin_bottom = 5
	return st


static func _catalog_entry(key: String) -> Dictionary:
	for entry in CHANNEL_CATALOG:
		if entry["key"] == key:
			return entry
	return {"key": key, "name": key, "color": Color(0.85, 0.92, 1.0)}


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
				draw_rect(Rect2(pos_cursor, 3.0, w, bar_h), col, true)
				pos_cursor += w
			else:
				draw_rect(Rect2(neg_cursor - w, 3.0, w, bar_h), col, true)
				neg_cursor -= w

		var req_w := clampf(absf(mz_req) / scale_nm, 0.0, 1.0) * half_w
		var req_x := mid_x if mz_req >= 0.0 else mid_x - req_w
		draw_rect(Rect2(req_x, 5.0 + bar_h, req_w, sz.y - bar_h - 8.0), Color(0.95, 0.97, 1.00, 0.90), true)


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
			var center := Vector2((float(col_idx) + 0.5) * cell_w, (float(row_idx) + 0.55) * cell_h)
			var r := minf(cell_w, cell_h) * 0.34
			var f_max := maxf(wheel_fz[idx] * wheel_mu[idx], 100.0)
			var fx := wheel_fx[idx]
			var fy := wheel_fy[idx]
			var f_res := sqrt(fx * fx + fy * fy)
			var sat := clampf(f_res / f_max, 0.0, 1.5)
			var ring_col := Color(0.06, 0.73, 0.51, 0.85)
			if sat > 0.95:
				ring_col = Color(0.94, 0.27, 0.27, 0.95)
			elif sat > 0.78:
				ring_col = Color(0.96, 0.62, 0.04, 0.90)

			draw_circle(center, r, Color(0.06, 0.08, 0.12, 0.92))
			draw_arc(center, r, 0.0, TAU, 32, ring_col, 1.6)
			draw_line(center - Vector2(r, 0.0), center + Vector2(r, 0.0), Color(0.30, 0.38, 0.50, 0.45), 1.0)
			draw_line(center - Vector2(0.0, r), center + Vector2(0.0, r), Color(0.30, 0.38, 0.50, 0.45), 1.0)

			var vec := Vector2(-fy / f_max, -fx / f_max)
			if vec.length() > 1.25:
				vec = vec.normalized() * 1.25
			var tip := center + vec * r
			draw_line(center, tip, Color(0.35, 1.00, 0.58), 2.0)
			draw_circle(tip, 2.8, Color(1.0, 0.95, 0.45))

			var lbl := "%s %.0f%%" % [names[idx], sat * 100.0]
			draw_string(font, Vector2(float(col_idx) * cell_w + 5.0, float(row_idx) * cell_h + 13.0), lbl, HORIZONTAL_ALIGNMENT_LEFT, -1, 10, Color(0.88, 0.94, 1.0))


class DraggableSidePanel extends PanelContainer:
	var header_bar: HBoxContainer = null
	var title_label: Label = null
	var collapse_btn: Button = null
	var body_container: VBoxContainer = null
	var collapsed: bool = false
	var _dragging: bool = false
	var _drag_offset: Vector2 = Vector2.ZERO
	var _expanded_size: Vector2 = Vector2(380, 260)

	func setup_panel(title_text: String, init_pos: Vector2, init_width: float, font: Font) -> void:
		position = init_pos
		custom_minimum_size = Vector2(init_width, 0)
		size = Vector2(init_width, 0)
		add_theme_stylebox_override("panel", TelemetryInspectorOverlay._make_style())

		var root_vb := VBoxContainer.new()
		root_vb.add_theme_constant_override("separation", 4)
		add_child(root_vb)

		header_bar = HBoxContainer.new()
		header_bar.mouse_default_cursor_shape = Control.CURSOR_MOVE
		header_bar.gui_input.connect(_on_header_gui_input)
		root_vb.add_child(header_bar)

		title_label = Label.new()
		title_label.text = title_text
		title_label.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		title_label.add_theme_color_override("font_color", Color(0.28, 0.90, 1.0))
		if font != null:
			title_label.add_theme_font_override("font", font)
		title_label.add_theme_font_size_override("font_size", 11)
		header_bar.add_child(title_label)

		collapse_btn = Button.new()
		collapse_btn.text = "–"
		collapse_btn.focus_mode = Control.FOCUS_NONE
		if font != null:
			collapse_btn.add_theme_font_override("font", font)
		collapse_btn.add_theme_font_size_override("font_size", 11)
		collapse_btn.pressed.connect(toggle_collapsed)
		header_bar.add_child(collapse_btn)

		body_container = VBoxContainer.new()
		body_container.add_theme_constant_override("separation", 4)
		root_vb.add_child(body_container)

	func toggle_collapsed() -> void:
		set_collapsed(not collapsed)

	func set_collapsed(is_col: bool) -> void:
		if not collapsed and is_col:
			_expanded_size = size
		collapsed = is_col
		if body_container != null:
			body_container.visible = not collapsed
		if collapse_btn != null:
			collapse_btn.text = "+" if collapsed else "–"
		reset_size()

	func _on_header_gui_input(event: InputEvent) -> void:
		if event is InputEventMouseButton and event.button_index == MOUSE_BUTTON_LEFT:
			if event.pressed:
				_dragging = true
				_drag_offset = get_global_mouse_position() - position
				move_to_front()
			else:
				_dragging = false
		elif event is InputEventMouseMotion and _dragging:
			var target := get_global_mouse_position() - _drag_offset
			if is_inside_tree():
				var vp := get_viewport_rect().size
				position = Vector2(
					clampf(target.x, 0.0, maxf(0.0, vp.x - size.x)),
					clampf(target.y, 0.0, maxf(0.0, vp.y - size.y))
				)
			else:
				position = Vector2(maxf(0.0, target.x), maxf(0.0, target.y))


class ChartPlotCanvas extends Control:
	var owner_overlay: TelemetryInspectorOverlay = null
	var chart_window: CustomChartWindow = null
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
		draw_rect(Rect2(Vector2.ZERO, sz), Color(0.04, 0.06, 0.09, 0.96), true)
		draw_rect(Rect2(Vector2.ZERO, sz), Color(0.20, 0.30, 0.42, 0.80), false, 1.0)
		for g in range(1, 4):
			var gy := sz.y * float(g) / 4.0
			draw_line(Vector2(0.0, gy), Vector2(sz.x, gy), Color(0.22, 0.28, 0.36, 0.35), 1.0)

		var font := ThemeDB.fallback_font
		if owner_overlay == null or chart_window == null:
			return
		var keys: Array[String] = chart_window.selected_channels
		if keys.is_empty():
			draw_string(font, Vector2(10.0, sz.y * 0.5), "Pulsa [⚙ Canales] para seleccionar variables", HORIZONTAL_ALIGNMENT_LEFT, -1, 11, Color(0.62, 0.72, 0.84))
			return
		var samples: Array[Dictionary] = owner_overlay.get_ordered_samples()
		if samples.size() < 2:
			draw_string(font, Vector2(10.0, sz.y * 0.5), "Esperando muestras...", HORIZONTAL_ALIGNMENT_LEFT, -1, 11, Color(0.62, 0.72, 0.84))
			return

		# Per-channel min/max to decide shared vs normalized scale
		var mins: Array[float] = []
		var maxs: Array[float] = []
		var spans: Array[float] = []
		var g_min := 1e9
		var g_max := -1e9
		var min_span := 1e9
		var max_span := 0.0
		for k in keys:
			var c_min := 1e9
			var c_max := -1e9
			for s in samples:
				var v := float(s.get(k, 0.0))
				c_min = minf(c_min, v)
				c_max = maxf(c_max, v)
			if c_max - c_min < 0.1:
				var mid := 0.5 * (c_min + c_max)
				c_min = mid - 0.5
				c_max = mid + 0.5
			else:
				var pad := 0.08 * (c_max - c_min)
				c_min -= pad
				c_max += pad
			var sp := c_max - c_min
			mins.append(c_min)
			maxs.append(c_max)
			spans.append(sp)
			g_min = minf(g_min, c_min)
			g_max = maxf(g_max, c_max)
			min_span = minf(min_span, sp)
			max_span = maxf(max_span, sp)

		var use_norm := keys.size() > 1 and (max_span / maxf(min_span, 1e-4) > 4.5)
		var n := samples.size()
		var lx := 8.0
		for c_idx in range(keys.size()):
			var k := keys[c_idx]
			var entry := TelemetryInspectorOverlay._catalog_entry(k)
			var col: Color = entry["color"]
			var y_lo := mins[c_idx] if use_norm else g_min
			var y_hi := maxs[c_idx] if use_norm else g_max
			var y_sp := maxf(y_hi - y_lo, 1e-5)

			var pts := PackedVector2Array()
			pts.resize(n)
			for idx in range(n):
				var px := (float(idx) / float(maxi(n - 1, 1))) * sz.x
				var v := float(samples[idx].get(k, 0.0))
				var py := sz.y - ((v - y_lo) / y_sp) * (sz.y - 24.0) - 10.0
				pts[idx] = Vector2(px, py)
			draw_polyline(pts, col, 1.6, true)

			var leg := k
			if lx + 82.0 < sz.x:
				draw_string(font, Vector2(lx, 13.0), leg, HORIZONTAL_ALIGNMENT_LEFT, -1, 10, col)
				lx += 82.0

		if use_norm:
			draw_string(font, Vector2(sz.x - 76.0, 13.0), "[AUTO-NORM]", HORIZONTAL_ALIGNMENT_LEFT, -1, 9, Color(0.68, 0.78, 0.90))

		# Interactive hover cursor
		if hover_pos.x >= 0.0 and hover_pos.x <= sz.x:
			var s_idx := clampi(int(round((hover_pos.x / maxf(sz.x, 1.0)) * float(n - 1))), 0, n - 1)
			var cx := (float(s_idx) / float(maxi(n - 1, 1))) * sz.x
			draw_line(Vector2(cx, 0.0), Vector2(cx, sz.y), Color(0.95, 0.98, 1.0, 0.75), 1.0)
			var samp: Dictionary = samples[s_idx]
			var readout := "t=%.2fs" % float(samp.get("t", 0.0))
			for k in keys:
				readout += " | %s=%.2f" % [k, float(samp.get(k, 0.0))]
			draw_rect(Rect2(4.0, sz.y - 19.0, sz.x - 8.0, 17.0), Color(0.03, 0.05, 0.08, 0.90), true)
			draw_string(font, Vector2(8.0, sz.y - 6.0), readout, HORIZONTAL_ALIGNMENT_LEFT, -1, 10, Color(0.95, 0.98, 1.0))


class CustomChartWindow extends PanelContainer:
	const MIN_SIZE: Vector2 = Vector2(220.0, 120.0)

	var owner_overlay: TelemetryInspectorOverlay = null
	var selected_channels: Array[String] = []
	var plot_canvas: ChartPlotCanvas = null
	var title_label: Label = null
	var channels_btn: Button = null
	var picker_panel: PanelContainer = null
	var picker_search: LineEdit = null
	var picker_list: VBoxContainer = null
	var _checkboxes: Dictionary = {}
	var _dragging: bool = false
	var _drag_offset: Vector2 = Vector2.ZERO
	var _resizing: bool = false
	var _resize_start_mouse: Vector2 = Vector2.ZERO
	var _resize_start_size: Vector2 = Vector2.ZERO

	func setup_window(overlay: TelemetryInspectorOverlay, chart_id: int, init_channels: Array[String], init_pos: Vector2, init_size: Vector2, font: Font) -> void:
		owner_overlay = overlay
		position = init_pos
		custom_minimum_size = MIN_SIZE
		size = Vector2(maxf(init_size.x, MIN_SIZE.x), maxf(init_size.y, MIN_SIZE.y))
		add_theme_stylebox_override("panel", TelemetryInspectorOverlay._make_style(Color(0.05, 0.07, 0.11, 0.90), Color(0.24, 0.56, 0.78, 0.80)))

		var outer := Control.new()
		outer.set_anchors_preset(Control.PRESET_FULL_RECT)
		outer.mouse_filter = Control.MOUSE_FILTER_PASS
		add_child(outer)

		var vb := VBoxContainer.new()
		vb.set_anchors_preset(Control.PRESET_FULL_RECT)
		vb.add_theme_constant_override("separation", 3)
		outer.add_child(vb)

		var hdr := HBoxContainer.new()
		hdr.mouse_default_cursor_shape = Control.CURSOR_MOVE
		hdr.gui_input.connect(_on_header_gui_input)
		vb.add_child(hdr)

		title_label = Label.new()
		title_label.text = "Gráfica %d" % chart_id
		title_label.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		title_label.add_theme_color_override("font_color", Color(0.32, 0.92, 1.0))
		if font != null:
			title_label.add_theme_font_override("font", font)
		title_label.add_theme_font_size_override("font_size", 11)
		hdr.add_child(title_label)

		channels_btn = Button.new()
		channels_btn.text = "⚙ Canales"
		channels_btn.focus_mode = Control.FOCUS_NONE
		if font != null:
			channels_btn.add_theme_font_override("font", font)
		channels_btn.add_theme_font_size_override("font_size", 10)
		channels_btn.pressed.connect(toggle_picker)
		hdr.add_child(channels_btn)

		var close_btn := Button.new()
		close_btn.text = "✕"
		close_btn.focus_mode = Control.FOCUS_NONE
		if font != null:
			close_btn.add_theme_font_override("font", font)
		close_btn.add_theme_font_size_override("font_size", 10)
		close_btn.pressed.connect(func():
			if owner_overlay != null:
				owner_overlay.close_chart_window(self)
		)
		hdr.add_child(close_btn)

		plot_canvas = ChartPlotCanvas.new()
		plot_canvas.owner_overlay = overlay
		plot_canvas.chart_window = self
		plot_canvas.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		plot_canvas.size_flags_vertical = Control.SIZE_EXPAND_FILL
		vb.add_child(plot_canvas)

		# Channel Picker Popover inside window
		picker_panel = PanelContainer.new()
		picker_panel.visible = false
		picker_panel.set_anchors_preset(Control.PRESET_FULL_RECT)
		picker_panel.offset_top = 26
		picker_panel.offset_bottom = -14
		picker_panel.offset_left = 6
		picker_panel.offset_right = -6
		picker_panel.add_theme_stylebox_override("panel", TelemetryInspectorOverlay._make_style(Color(0.04, 0.06, 0.10, 0.97), Color(0.30, 0.72, 0.95, 0.90)))
		outer.add_child(picker_panel)

		var p_vb := VBoxContainer.new()
		p_vb.add_theme_constant_override("separation", 3)
		picker_panel.add_child(p_vb)

		var p_top := HBoxContainer.new()
		p_vb.add_child(p_top)
		picker_search = LineEdit.new()
		picker_search.placeholder_text = "Buscar canal (ej: wz, Fz, T_cmd, kappa)..."
		picker_search.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		if font != null:
			picker_search.add_theme_font_override("font", font)
		picker_search.add_theme_font_size_override("font_size", 10)
		picker_search.text_changed.connect(_on_search_changed)
		p_top.add_child(picker_search)

		var done_btn := Button.new()
		done_btn.text = "OK"
		done_btn.focus_mode = Control.FOCUS_NONE
		if font != null:
			done_btn.add_theme_font_override("font", font)
		done_btn.add_theme_font_size_override("font_size", 10)
		done_btn.pressed.connect(func(): picker_panel.visible = false)
		p_top.add_child(done_btn)

		var sc := ScrollContainer.new()
		sc.size_flags_vertical = Control.SIZE_EXPAND_FILL
		p_vb.add_child(sc)

		picker_list = VBoxContainer.new()
		picker_list.add_theme_constant_override("separation", 1)
		sc.add_child(picker_list)

		for entry in CHANNEL_CATALOG:
			var key: String = entry["key"]
			var cb := CheckBox.new()
			cb.text = str(entry["name"])
			cb.focus_mode = Control.FOCUS_NONE
			cb.add_theme_color_override("font_color", entry["color"])
			if font != null:
				cb.add_theme_font_override("font", font)
			cb.add_theme_font_size_override("font_size", 10)
			cb.toggled.connect(func(pressed: bool): toggle_channel(key, pressed))
			picker_list.add_child(cb)
			_checkboxes[key] = cb

		# Bottom-right resize handle ◢
		var resize_handle := Label.new()
		resize_handle.text = "◢"
		resize_handle.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
		resize_handle.vertical_alignment = VERTICAL_ALIGNMENT_BOTTOM
		resize_handle.anchor_left = 1.0
		resize_handle.anchor_right = 1.0
		resize_handle.anchor_top = 1.0
		resize_handle.anchor_bottom = 1.0
		resize_handle.offset_left = -18
		resize_handle.offset_top = -18
		resize_handle.offset_right = 0
		resize_handle.offset_bottom = 0
		resize_handle.mouse_filter = Control.MOUSE_FILTER_STOP
		resize_handle.mouse_default_cursor_shape = Control.CURSOR_FDIAGSIZE
		resize_handle.add_theme_color_override("font_color", Color(0.55, 0.85, 1.0, 0.90))
		resize_handle.gui_input.connect(_on_resize_gui_input)
		outer.add_child(resize_handle)

		set_selected_channels(init_channels)

	func toggle_picker() -> void:
		if picker_panel != null:
			picker_panel.visible = not picker_panel.visible

	func is_search_focused() -> bool:
		return picker_search != null and picker_search.has_focus()

	func _on_search_changed(txt: String) -> void:
		var q := txt.strip_edges().to_lower()
		for k in _checkboxes:
			var cb: CheckBox = _checkboxes[k]
			cb.visible = (q == "" or String(k).to_lower().contains(q) or cb.text.to_lower().contains(q))

	func set_selected_channels(keys: Array[String]) -> void:
		selected_channels.clear()
		for k in keys:
			if not selected_channels.has(k):
				selected_channels.append(k)
		for k in _checkboxes:
			var cb: CheckBox = _checkboxes[k]
			cb.set_pressed_no_signal(selected_channels.has(k))
		if plot_canvas != null:
			plot_canvas.queue_redraw()

	func toggle_channel(key: String, enabled: bool) -> void:
		if enabled:
			if not selected_channels.has(key):
				selected_channels.append(key)
		else:
			selected_channels.erase(key)
		if _checkboxes.has(key):
			var cb: CheckBox = _checkboxes[key]
			cb.set_pressed_no_signal(enabled)
		if plot_canvas != null:
			plot_canvas.queue_redraw()

	func get_selected_channels() -> Array[String]:
		return selected_channels.duplicate()

	func move_window(new_pos: Vector2) -> void:
		if is_inside_tree():
			var vp := get_viewport_rect().size
			if vp.x > 10.0 and vp.y > 10.0:
				position = Vector2(
					clampf(new_pos.x, 0.0, maxf(0.0, vp.x - size.x)),
					clampf(new_pos.y, 0.0, maxf(0.0, vp.y - size.y))
				)
				return
		position = Vector2(maxf(0.0, new_pos.x), maxf(0.0, new_pos.y))

	func resize_window(new_size: Vector2) -> void:
		size = Vector2(maxf(new_size.x, MIN_SIZE.x), maxf(new_size.y, MIN_SIZE.y))
		if plot_canvas != null:
			plot_canvas.queue_redraw()

	func _on_header_gui_input(event: InputEvent) -> void:
		if event is InputEventMouseButton and event.button_index == MOUSE_BUTTON_LEFT:
			if event.pressed:
				_dragging = true
				_drag_offset = get_global_mouse_position() - position
				move_to_front()
			else:
				_dragging = false
		elif event is InputEventMouseMotion and _dragging:
			move_window(get_global_mouse_position() - _drag_offset)

	func _on_resize_gui_input(event: InputEvent) -> void:
		if event is InputEventMouseButton and event.button_index == MOUSE_BUTTON_LEFT:
			if event.pressed:
				_resizing = true
				_resize_start_mouse = get_global_mouse_position()
				_resize_start_size = size
				move_to_front()
			else:
				_resizing = false
		elif event is InputEventMouseMotion and _resizing:
			var delta := get_global_mouse_position() - _resize_start_mouse
			resize_window(_resize_start_size + delta)

	func to_dict() -> Dictionary:
		var ch: Array[String] = []
		for k in selected_channels:
			ch.append(k)
		return {
			"position": [position.x, position.y],
			"size": [size.x, size.y],
			"channels": ch
		}

	func from_dict(d: Dictionary) -> void:
		if d.has("size") and d["size"] is Array and d["size"].size() >= 2:
			resize_window(Vector2(float(d["size"][0]), float(d["size"][1])))
		if d.has("position") and d["position"] is Array and d["position"].size() >= 2:
			move_window(Vector2(float(d["position"][0]), float(d["position"][1])))
		if d.has("channels") and d["channels"] is Array:
			var ch: Array[String] = []
			for item in d["channels"]:
				ch.append(str(item))
			set_selected_channels(ch)


var frozen: bool = false
var _car: Node = null
var _sim_time: float = 0.0
var _sample_acc: float = 0.0
var _ring: Array[Dictionary] = []
var _ring_head: int = 0
var _ring_count: int = 0
var _next_chart_id: int = 1

var _mono_font: SystemFont = null
var _root_ctrl: Control = null
var _left_panel: DraggableSidePanel = null
var _right_panel: DraggableSidePanel = null
var _charts_layer: Control = null
var _chart_windows: Array[CustomChartWindow] = []

var _freeze_btn: Button = null
var _arrows_btn: Button = null
var _vars_toggle_btn: Button = null
var _vars_drawer: PanelContainer = null
var _vars_filter_edit: LineEdit = null
var _vars_label: RichTextLabel = null
var _status_toast: Label = null

var _yaw_label: Label = null
var _mz_label: Label = null
var _mz_bar: MzBreakdownBar = null
var _cascade_label: Label = null
var _friction_view: FrictionCirclesView = null
var _wheels_label: Label = null

var _latest_vars: Array[Array] = []


func _ready() -> void:
	layer = 12
	_mono_font = SystemFont.new()
	_mono_font.font_names = PackedStringArray(["JetBrains Mono", "Fira Code", "DejaVu Sans Mono", "Monospace"])
	_ring.resize(BUFFER_LEN)
	for i in range(BUFFER_LEN):
		_ring[i] = {}
	_build_ui()
	if FileAccess.file_exists(LAYOUT_PATH):
		load_layout(LAYOUT_PATH)
	visible = false


func bind_car(car: Node) -> void:
	_car = car


func toggle_visibility() -> void:
	visible = not visible


func toggle_freeze() -> void:
	frozen = not frozen
	if _freeze_btn != null:
		_freeze_btn.text = "[F12] REANUDAR" if frozen else "[F12] CONGELAR"
		_freeze_btn.modulate = Color(1.0, 0.72, 0.35) if frozen else Color(0.85, 0.96, 1.0)


func sync_wheel_arrows_button(enabled: bool) -> void:
	if _arrows_btn != null:
		_arrows_btn.set_pressed_no_signal(enabled)
		_arrows_btn.modulate = Color(0.30, 1.0, 0.65) if enabled else Color(0.85, 0.94, 1.0)


func is_text_input_focused() -> bool:
	if _vars_filter_edit != null and _vars_filter_edit.has_focus():
		return true
	for cw in _chart_windows:
		if is_instance_valid(cw) and cw.is_search_focused():
			return true
	return false


func get_ordered_samples() -> Array[Dictionary]:
	var out: Array[Dictionary] = []
	if _ring_count == 0:
		return out
	var start := (_ring_head - _ring_count + BUFFER_LEN) % BUFFER_LEN
	for i in range(_ring_count):
		out.append(_ring[(start + i) % BUFFER_LEN])
	return out


func _apply_mono(ctrl: Control, sz: int = 11) -> void:
	if _mono_font != null:
		ctrl.add_theme_font_override("font", _mono_font)
	ctrl.add_theme_font_size_override("font_size", sz)


func _build_ui() -> void:
	_root_ctrl = Control.new()
	_root_ctrl.set_anchors_preset(Control.PRESET_FULL_RECT)
	_root_ctrl.mouse_filter = Control.MOUSE_FILTER_IGNORE
	add_child(_root_ctrl)

	# Floating Charts Layer (behind top toolbar/drawer, above track)
	_charts_layer = Control.new()
	_charts_layer.set_anchors_preset(Control.PRESET_FULL_RECT)
	_charts_layer.mouse_filter = Control.MOUSE_FILTER_IGNORE
	_root_ctrl.add_child(_charts_layer)

	# Compact Top Toolbar
	var top_bar := PanelContainer.new()
	top_bar.anchor_left = 0.5
	top_bar.anchor_right = 0.5
	top_bar.anchor_top = 0.0
	top_bar.anchor_bottom = 0.0
	top_bar.offset_left = -380
	top_bar.offset_right = 380
	top_bar.offset_top = 8
	top_bar.offset_bottom = 42
	top_bar.add_theme_stylebox_override("panel", _make_style(Color(0.04, 0.06, 0.10, 0.90), Color(0.24, 0.58, 0.80, 0.82)))
	_root_ctrl.add_child(top_bar)

	var tb := HBoxContainer.new()
	tb.add_theme_constant_override("separation", 6)
	top_bar.add_child(tb)

	var badge := Label.new()
	badge.text = "TELEMETRÍA [F11]"
	badge.add_theme_color_override("font_color", Color(0.28, 0.90, 1.0))
	_apply_mono(badge, 11)
	tb.add_child(badge)

	var new_chart_btn := Button.new()
	new_chart_btn.text = "+ Nueva Gráfica"
	new_chart_btn.focus_mode = Control.FOCUS_NONE
	_apply_mono(new_chart_btn, 10)
	new_chart_btn.pressed.connect(func(): create_chart_window(["wz_ref", "wz_real"]))
	tb.add_child(new_chart_btn)

	var save_btn := Button.new()
	save_btn.text = "Guardar Layout"
	save_btn.focus_mode = Control.FOCUS_NONE
	_apply_mono(save_btn, 10)
	save_btn.pressed.connect(func():
		if save_layout(LAYOUT_PATH):
			_show_toast("Layout guardado")
	)
	tb.add_child(save_btn)

	var load_btn := Button.new()
	load_btn.text = "Cargar Layout"
	load_btn.focus_mode = Control.FOCUS_NONE
	_apply_mono(load_btn, 10)
	load_btn.pressed.connect(func():
		if load_layout(LAYOUT_PATH):
			_show_toast("Layout cargado")
	)
	tb.add_child(load_btn)

	_arrows_btn = Button.new()
	_arrows_btn.text = "Flechas 3D Ruedas [F8]"
	_arrows_btn.toggle_mode = true
	_arrows_btn.focus_mode = Control.FOCUS_NONE
	_apply_mono(_arrows_btn, 10)
	_arrows_btn.toggled.connect(func(pressed: bool):
		if _car != null and _car.has_method("set_wheel_arrows_enabled"):
			_car.set_wheel_arrows_enabled(pressed)
		sync_wheel_arrows_button(pressed)
	)
	tb.add_child(_arrows_btn)

	_freeze_btn = Button.new()
	_freeze_btn.text = "[F12] CONGELAR"
	_freeze_btn.focus_mode = Control.FOCUS_NONE
	_apply_mono(_freeze_btn, 10)
	_freeze_btn.pressed.connect(toggle_freeze)
	tb.add_child(_freeze_btn)

	_vars_toggle_btn = Button.new()
	_vars_toggle_btn.text = "Todas las Vars"
	_vars_toggle_btn.focus_mode = Control.FOCUS_NONE
	_apply_mono(_vars_toggle_btn, 10)
	_vars_toggle_btn.pressed.connect(func():
		if _vars_drawer != null:
			_vars_drawer.visible = not _vars_drawer.visible
	)
	tb.add_child(_vars_toggle_btn)

	_status_toast = Label.new()
	_status_toast.visible = false
	_apply_mono(_status_toast, 10)
	_status_toast.add_theme_color_override("font_color", Color(0.35, 1.0, 0.65))
	tb.add_child(_status_toast)

	# Left Compact Draggable/Collapsible Panel — ECU_TV_V3 X-Ray
	_left_panel = DraggableSidePanel.new()
	_left_panel.setup_panel("ECU_TV_V3 X-RAY", Vector2(12, 84), 382.0, _mono_font)
	_root_ctrl.add_child(_left_panel)

	_yaw_label = Label.new()
	_apply_mono(_yaw_label, 10)
	_yaw_label.add_theme_color_override("font_color", Color(0.88, 0.95, 1.0))
	_left_panel.body_container.add_child(_yaw_label)

	_mz_label = Label.new()
	_apply_mono(_mz_label, 10)
	_mz_label.add_theme_color_override("font_color", Color(1.0, 0.85, 0.45))
	_left_panel.body_container.add_child(_mz_label)

	_mz_bar = MzBreakdownBar.new()
	_mz_bar.custom_minimum_size = Vector2(0, 22)
	_left_panel.body_container.add_child(_mz_bar)

	_cascade_label = Label.new()
	_apply_mono(_cascade_label, 9)
	_cascade_label.add_theme_color_override("font_color", Color(0.75, 0.98, 0.82))
	_left_panel.body_container.add_child(_cascade_label)

	# Right Compact Draggable/Collapsible Panel — 4 Friction Circles + Wheel Data
	var vp_w := 1280.0
	if get_viewport() != null:
		vp_w = maxf(get_viewport().get_visible_rect().size.x, 900.0)
	_right_panel = DraggableSidePanel.new()
	_right_panel.setup_panel("CÍRCULOS DE FRICCIÓN & RUEDAS", Vector2(vp_w - 410.0, 50.0), 398.0, _mono_font)
	_root_ctrl.add_child(_right_panel)

	_friction_view = FrictionCirclesView.new()
	_friction_view.custom_minimum_size = Vector2(0, 178)
	_right_panel.body_container.add_child(_friction_view)

	_wheels_label = Label.new()
	_apply_mono(_wheels_label, 9)
	_wheels_label.add_theme_color_override("font_color", Color(0.88, 0.94, 0.98))
	_right_panel.body_container.add_child(_wheels_label)

	# Floating "Todas las Variables" Drawer
	_vars_drawer = PanelContainer.new()
	_vars_drawer.anchor_left = 0.34
	_vars_drawer.anchor_right = 0.66
	_vars_drawer.anchor_top = 0.08
	_vars_drawer.anchor_bottom = 0.58
	_vars_drawer.add_theme_stylebox_override("panel", _make_style(Color(0.04, 0.06, 0.09, 0.94)))
	_vars_drawer.visible = false
	_root_ctrl.add_child(_vars_drawer)

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


func _show_toast(msg: String) -> void:
	if _status_toast != null:
		_status_toast.text = msg
		_status_toast.visible = true


func create_chart_window(channels: Array[String] = [], pos: Vector2 = Vector2(-1, -1), win_size: Vector2 = Vector2(360, 200)) -> CustomChartWindow:
	var cw := CustomChartWindow.new()
	var offset_idx := _chart_windows.size() % 5
	var init_pos := pos if pos.x >= 0.0 else Vector2(410.0 + float(offset_idx) * 26.0, 56.0 + float(offset_idx) * 26.0)
	var init_ch: Array[String] = channels.duplicate()
	if init_ch.is_empty():
		init_ch = ["wz_ref", "wz_real"]
	_charts_layer.add_child(cw)
	cw.setup_window(self, _next_chart_id, init_ch, init_pos, win_size, _mono_font)
	_next_chart_id += 1
	_chart_windows.append(cw)
	return cw


func close_chart_window(cw: CustomChartWindow) -> void:
	_chart_windows.erase(cw)
	if is_instance_valid(cw):
		cw.queue_free()


func get_chart_windows() -> Array[CustomChartWindow]:
	return _chart_windows


func save_layout(path: String = LAYOUT_PATH) -> bool:
	var charts_data: Array[Dictionary] = []
	for cw in _chart_windows:
		if is_instance_valid(cw):
			charts_data.append(cw.to_dict())
	var payload := {
		"charts": charts_data,
		"left_panel": {
			"position": [_left_panel.position.x, _left_panel.position.y],
			"collapsed": _left_panel.collapsed
		},
		"right_panel": {
			"position": [_right_panel.position.x, _right_panel.position.y],
			"collapsed": _right_panel.collapsed
		},
		"wheel_arrows_3d": _arrows_btn.button_pressed if _arrows_btn != null else false
	}
	var f := FileAccess.open(path, FileAccess.WRITE)
	if f == null:
		return false
	f.store_string(JSON.stringify(payload, "  "))
	f.close()
	return true


func load_layout(path: String = LAYOUT_PATH) -> bool:
	if not FileAccess.file_exists(path):
		return false
	var f := FileAccess.open(path, FileAccess.READ)
	if f == null:
		return false
	var parsed = JSON.parse_string(f.get_as_text())
	f.close()
	if not (parsed is Dictionary):
		return false

	# Clear existing chart windows
	for cw in _chart_windows.duplicate():
		close_chart_window(cw)
	_next_chart_id = 1

	if parsed.has("charts") and parsed["charts"] is Array:
		for c_dict in parsed["charts"]:
			if c_dict is Dictionary:
				var cw := create_chart_window([])
				cw.from_dict(c_dict)

	if parsed.has("left_panel") and parsed["left_panel"] is Dictionary and _left_panel != null:
		var lp: Dictionary = parsed["left_panel"]
		if lp.has("position") and lp["position"] is Array and lp["position"].size() >= 2:
			_left_panel.position = Vector2(float(lp["position"][0]), float(lp["position"][1]))
		if lp.has("collapsed"):
			_left_panel.set_collapsed(bool(lp["collapsed"]))

	if parsed.has("right_panel") and parsed["right_panel"] is Dictionary and _right_panel != null:
		var rp: Dictionary = parsed["right_panel"]
		if rp.has("position") and rp["position"] is Array and rp["position"].size() >= 2:
			_right_panel.position = Vector2(float(rp["position"][0]), float(rp["position"][1]))
		if rp.has("collapsed"):
			_right_panel.set_collapsed(bool(rp["collapsed"]))

	if parsed.has("wheel_arrows_3d"):
		var en := bool(parsed["wheel_arrows_3d"])
		if _car != null and _car.has_method("set_wheel_arrows_enabled"):
			_car.set_wheel_arrows_enabled(en)
		sync_wheel_arrows_button(en)

	return true


func update_from_car(car: Node, dt: float) -> void:
	if not visible or frozen or car == null:
		return
	_car = car
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
		var law_tag: String = "V4_SMC" if int(ecu.control_law_mode) == 1 else "V3_PID"
		var ev_tag: String = ecu.get_event_mode_display() if ecu.has_method("get_event_mode_display") else "Autocross"
		var lc_tag: String = ecu.get_launch_state_name() if ecu.has_method("get_launch_state_name") else "LC_OFF"
		_yaw_label.text = (
			"[%s | %s | %s] S=%+5.3f rad/s | k_lim=%.3f\n"
			+ "wz ref/real/ekf: %+5.2f / %+5.2f / %+5.2f rad/s\n"
			+ "beta real / ekf: %+5.2f / %+5.2f +-%.2f deg\n"
			+ "vy_ekf / vy_ss : %+5.2f / %+5.2f m/s (std=%.3f)\n"
			+ "Gains Kp/Ki/Kd : %.0f / %.0f / %.1f"
		) % [
			law_tag, ev_tag, lc_tag, float(ecu.last_smc_s), float(ecu.last_tc_kappa_lim[0]),
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

	var disc_temps: Array = last_res.get("disc_temps", b_state.disc_temp if b_state != null else [40.0, 40.0, 40.0, 40.0])
	var grip_pcts: Array[float] = [0.0, 0.0, 0.0, 0.0]
	if wheels.size() >= 4:
		for i in range(4):
			var f_max := maxf(float(wheels[i].F_z) * 1.5 * float(wheels[i].surface_mu_scale), 1.0)
			var f_res := sqrt(float(wheels[i].F_x) * float(wheels[i].F_x) + float(wheels[i].F_y) * float(wheels[i].F_y))
			grip_pcts[i] = clampf(f_res / f_max, 0.0, 1.2) * 100.0
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
			"wz_ekf": wz_ekf,
			"beta_real": beta_real,
			"beta_ekf": beta_ekf,
			"ax_g": float(model._prev_ax_ros) / 9.81,
			"ay_g": float(model._prev_ay_ros) / 9.81,
			"roll_deg": rad_to_deg(float(model.chassis_state.roll)),
			"pitch_deg": rad_to_deg(float(model.chassis_state.pitch)),
			"mz_req": float(ecu.last_mz_req) if ecu != null else 0.0,
			"mz_raw": float(ecu.last_mz_raw) if ecu != null else 0.0,
			"ff_mz": float(ecu.last_ff_mz) if ecu != null else 0.0,
			"p_mz": float(ecu.last_p_mz) if ecu != null else 0.0,
			"i_mz": float(ecu.last_i_mz) if ecu != null else 0.0,
			"beta_mz": float(ecu.last_beta_mz) if ecu != null else 0.0,
			"smc_s": float(ecu.last_smc_s) if ecu != null else 0.0,
			"kappa_lim_FL": float(ecu.last_tc_kappa_lim[0]) if ecu != null else 0.045,
			"kappa_lim_RL": float(ecu.last_tc_kappa_lim[2]) if ecu != null else 0.045,
			"T_cmd_FL": float(drv_trqs[0]) if drv_trqs.size() > 0 else 0.0,
			"T_cmd_FR": float(drv_trqs[1]) if drv_trqs.size() > 1 else 0.0,
			"T_cmd_RL": float(drv_trqs[2]) if drv_trqs.size() > 2 else 0.0,
			"T_cmd_RR": float(drv_trqs[3]) if drv_trqs.size() > 3 else 0.0,
			"T_nom_FL": float(ecu.last_t_nominal[0]) if ecu != null else 0.0,
			"T_nom_FR": float(ecu.last_t_nominal[1]) if ecu != null else 0.0,
			"T_nom_RL": float(ecu.last_t_nominal[2]) if ecu != null else 0.0,
			"T_nom_RR": float(ecu.last_t_nominal[3]) if ecu != null else 0.0,
			"T_qp_FL": float(ecu.last_t_qp[0]) if ecu != null else 0.0,
			"T_qp_FR": float(ecu.last_t_qp[1]) if ecu != null else 0.0,
			"T_qp_RL": float(ecu.last_t_qp[2]) if ecu != null else 0.0,
			"T_qp_RR": float(ecu.last_t_qp[3]) if ecu != null else 0.0,
			"TC_cut_FL": float(ecu.last_tc_cut_nm[0]) if ecu != null else 0.0,
			"TC_cut_FR": float(ecu.last_tc_cut_nm[1]) if ecu != null else 0.0,
			"TC_cut_RL": float(ecu.last_tc_cut_nm[2]) if ecu != null else 0.0,
			"TC_cut_RR": float(ecu.last_tc_cut_nm[3]) if ecu != null else 0.0,
			"Fz_FL": float(wheels[0].F_z) if wheels.size() > 0 else 0.0,
			"Fz_FR": float(wheels[1].F_z) if wheels.size() > 1 else 0.0,
			"Fz_RL": float(wheels[2].F_z) if wheels.size() > 2 else 0.0,
			"Fz_RR": float(wheels[3].F_z) if wheels.size() > 3 else 0.0,
			"Fy_FL": float(wheels[0].F_y) if wheels.size() > 0 else 0.0,
			"Fy_FR": float(wheels[1].F_y) if wheels.size() > 1 else 0.0,
			"Fy_RL": float(wheels[2].F_y) if wheels.size() > 2 else 0.0,
			"Fy_RR": float(wheels[3].F_y) if wheels.size() > 3 else 0.0,
			"Fx_FL": float(wheels[0].F_x) if wheels.size() > 0 else 0.0,
			"Fx_FR": float(wheels[1].F_x) if wheels.size() > 1 else 0.0,
			"Fx_RL": float(wheels[2].F_x) if wheels.size() > 2 else 0.0,
			"Fx_RR": float(wheels[3].F_x) if wheels.size() > 3 else 0.0,
			"kappa_FL": float(wheels[0].kappa) * 100.0 if wheels.size() > 0 else 0.0,
			"kappa_FR": float(wheels[1].kappa) * 100.0 if wheels.size() > 1 else 0.0,
			"kappa_RL": float(wheels[2].kappa) * 100.0 if wheels.size() > 2 else 0.0,
			"kappa_RR": float(wheels[3].kappa) * 100.0 if wheels.size() > 3 else 0.0,
			"alpha_FL": rad_to_deg(float(wheels[0].alpha)) if wheels.size() > 0 else 0.0,
			"alpha_FR": rad_to_deg(float(wheels[1].alpha)) if wheels.size() > 1 else 0.0,
			"alpha_RL": rad_to_deg(float(wheels[2].alpha)) if wheels.size() > 2 else 0.0,
			"alpha_RR": rad_to_deg(float(wheels[3].alpha)) if wheels.size() > 3 else 0.0,
			"grip_pct_FL": grip_pcts[0],
			"grip_pct_FR": grip_pcts[1],
			"grip_pct_RL": grip_pcts[2],
			"grip_pct_RR": grip_pcts[3],
			"dz_FL": dz_arr[0],
			"dz_FR": dz_arr[1],
			"dz_RL": dz_arr[2],
			"dz_RR": dz_arr[3],
			"T_tire_mid_FL": float(wheels[0].t_zone_mid_c) if wheels.size() > 0 else 25.0,
			"T_tire_mid_FR": float(wheels[1].t_zone_mid_c) if wheels.size() > 1 else 25.0,
			"T_tire_mid_RL": float(wheels[2].t_zone_mid_c) if wheels.size() > 2 else 25.0,
			"T_tire_mid_RR": float(wheels[3].t_zone_mid_c) if wheels.size() > 3 else 25.0,
			"T_disc_FL": float(disc_temps[0]) if disc_temps.size() > 0 else 40.0,
			"T_disc_FR": float(disc_temps[1]) if disc_temps.size() > 1 else 40.0,
			"T_disc_RL": float(disc_temps[2]) if disc_temps.size() > 2 else 40.0,
			"T_disc_RR": float(disc_temps[3]) if disc_temps.size() > 3 else 40.0,
			"P_elec_kw": float(hv.power_kw) if hv != null else 0.0,
			"P_regen_kw": (float(b_state.p_regen_w) / 1000.0) if b_state != null else 0.0,
			"V_dc": float(hv.v_dc) if hv != null else 588.0,
			"T_brk_hyd": hyd_sum,
			"T_brk_reg": reg_sum
		}
		_ring_head = (_ring_head + 1) % BUFFER_LEN
		_ring_count = mini(_ring_count + 1, BUFFER_LEN)
		for cw in _chart_windows:
			if is_instance_valid(cw) and cw.plot_canvas != null:
				cw.plot_canvas.queue_redraw()

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
		_latest_vars.append(["smc_s", "%.4f rad/s" % float(ecu.last_smc_s)])
		_latest_vars.append(["tc_kappa_lim_fl..rr", "%.3f/%.3f/%.3f/%.3f" % [ecu.last_tc_kappa_lim[0], ecu.last_tc_kappa_lim[1], ecu.last_tc_kappa_lim[2], ecu.last_tc_kappa_lim[3]]])
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
