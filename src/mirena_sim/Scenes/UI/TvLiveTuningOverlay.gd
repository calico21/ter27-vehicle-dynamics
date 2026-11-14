extends CanvasLayer
class_name TvLiveTuningOverlay

const TV_SLIDER_SPECS := [
	{"key": "live_k_us", "label": "k_us (Understeer)", "min": 0.0, "max": 0.025, "step": 0.0005, "fmt": "%.4f"},
	{"key": "live_max_mz", "label": "max_mz (Nm)", "min": 200.0, "max": 2500.0, "step": 25.0, "fmt": "%.0f"},
	{"key": "live_min_fx_reserve", "label": "min_fx_reserve (N)", "min": 0.0, "max": 2500.0, "step": 50.0, "fmt": "%.0f"},
	{"key": "live_tc_kp", "label": "tc_kp", "min": 0.0, "max": 1000.0, "step": 25.0, "fmt": "%.0f"},
	{"key": "live_k_beta", "label": "k_beta (Nm/rad)", "min": 0.0, "max": 12000.0, "step": 100.0, "fmt": "%.0f"},
	{"key": "live_throttle_mz_cut", "label": "throttle_mz_cut", "min": 0.0, "max": 1.0, "step": 0.05, "fmt": "%.2f"}
]

const GP_SLIDER_SPECS := [
	{"key": "steering_deadzone", "label": "steering_deadzone", "min": 0.00, "max": 0.15, "step": 0.01, "fmt": "%.2f"},
	{"key": "steering_gamma", "label": "steering_gamma", "min": 1.00, "max": 3.50, "step": 0.05, "fmt": "%.2f"},
	{"key": "steering_speed", "label": "steering_speed", "min": 0.50, "max": 4.00, "step": 0.10, "fmt": "%.1f"},
	{"key": "steering_filter", "label": "steering_filter", "min": 0.00, "max": 0.40, "step": 0.01, "fmt": "%.2f"},
	{"key": "speed_sensitivity", "label": "speed_sensitivity", "min": 0.00, "max": 0.90, "step": 0.02, "fmt": "%.2f"},
	{"key": "v_ref_mps", "label": "v_ref_mps (m/s)", "min": 12.0, "max": 35.0, "step": 0.5, "fmt": "%.1f"},
	{"key": "throttle_gamma", "label": "throttle_gamma", "min": 1.00, "max": 2.50, "step": 0.05, "fmt": "%.2f"},
	{"key": "brake_gamma", "label": "brake_gamma", "min": 1.00, "max": 2.50, "step": 0.05, "fmt": "%.2f"},
	{"key": "rumble_effects", "label": "rumble_effects", "min": 0.00, "max": 1.00, "step": 0.05, "fmt": "%.2f"}
]

var _ecu = null
var _gamepad = null
var _car = null
var _root: Control = null
var _panel: PanelContainer = null
var _tabs: TabContainer = null

var _tv_sliders: Dictionary = {}
var _tv_val_labels: Dictionary = {}
var _gp_sliders: Dictionary = {}
var _gp_val_labels: Dictionary = {}

var _telem_label: Label = null
var _status_label: Label = null

var _steer_raw_bar: ProgressBar = null
var _steer_filt_bar: ProgressBar = null
var _th_raw_bar: ProgressBar = null
var _th_filt_bar: ProgressBar = null
var _brk_raw_bar: ProgressBar = null
var _brk_filt_bar: ProgressBar = null
var _gp_telem_label: Label = null
var _gp_status_label: Label = null

var _syncing: bool = false


func _ready() -> void:
	layer = 50
	_build_ui()
	visible = false


func bind_ecu(ecu) -> void:
	_ecu = ecu
	sync_from_ecu()


func bind_car(ecu, gamepad, car = null) -> void:
	_ecu = ecu
	_gamepad = gamepad
	_car = car
	sync_from_ecu()
	sync_from_gamepad()


func toggle_visibility() -> void:
	visible = not visible
	if visible:
		sync_from_ecu()
		sync_from_gamepad()


func sync_from_ecu() -> void:
	if _ecu == null or not _ecu.has_method("get_live_params"):
		return
	_syncing = true
	var params: Dictionary = _ecu.get_live_params()
	for spec in TV_SLIDER_SPECS:
		var k: String = spec["key"]
		if params.has(k) and _tv_sliders.has(k):
			var v := float(params[k])
			(_tv_sliders[k] as HSlider).value = v
			(_tv_val_labels[k] as Label).text = spec["fmt"] % v
	_syncing = false


func sync_from_gamepad() -> void:
	var params: Dictionary = {}
	if _car != null and _car.has_method("get_live_gamepad_params"):
		params = _car.get_live_gamepad_params()
	elif _gamepad != null and _gamepad.has_method("get_live_gamepad_params"):
		params = _gamepad.get_live_gamepad_params()
	if params.is_empty():
		return
	_syncing = true
	for spec in GP_SLIDER_SPECS:
		var k: String = spec["key"]
		if params.has(k) and _gp_sliders.has(k):
			var v := float(params[k])
			(_gp_sliders[k] as HSlider).value = v
			(_gp_val_labels[k] as Label).text = spec["fmt"] % v
	_syncing = false


func update_telemetry(wz_ref: float, wz: float, beta_deg: float, mz_req: float, tc_active: bool) -> void:
	if not visible or _telem_label == null:
		return
	var tc_txt := "ON" if tc_active else "off"
	_telem_label.text = "wz_ref: %+5.2f | wz: %+5.2f rad/s | beta: %+4.1f deg\nmz_req: %+6.0f Nm | TC: %s" % [
		wz_ref, wz, beta_deg, mz_req, tc_txt
	]


func update_input_telemetry(raw_steer: float, filt_steer: float, raw_th: float, filt_th: float, raw_brk: float, filt_brk: float) -> void:
	if not visible or _gp_telem_label == null:
		return
	if _steer_raw_bar != null:
		_steer_raw_bar.value = (raw_steer + 1.0) * 50.0
	if _steer_filt_bar != null:
		_steer_filt_bar.value = (filt_steer + 1.0) * 50.0
	if _th_raw_bar != null:
		_th_raw_bar.value = raw_th * 100.0
	if _th_filt_bar != null:
		_th_filt_bar.value = filt_th * 100.0
	if _brk_raw_bar != null:
		_brk_raw_bar.value = raw_brk * 100.0
	if _brk_filt_bar != null:
		_brk_filt_bar.value = filt_brk * 100.0
	_gp_telem_label.text = "Steer %+5.2f -> %+5.2f | Gas %4.2f -> %4.2f | Brk %4.2f -> %4.2f" % [
		raw_steer, filt_steer, raw_th, filt_th, raw_brk, filt_brk
	]


func _build_ui() -> void:
	_root = Control.new()
	_root.set_anchors_preset(Control.PRESET_FULL_RECT)
	_root.mouse_filter = Control.MOUSE_FILTER_IGNORE
	add_child(_root)

	_panel = PanelContainer.new()
	_panel.position = Vector2(20, 20)
	_panel.custom_minimum_size = Vector2(430, 0)
	_panel.mouse_filter = Control.MOUSE_FILTER_STOP
	var style := StyleBoxFlat.new()
	style.bg_color = Color(0.06, 0.08, 0.11, 0.86)
	style.border_color = Color(0.25, 0.72, 0.95, 0.75)
	style.set_border_width_all(1)
	style.set_corner_radius_all(6)
	style.set_content_margin_all(10)
	_panel.add_theme_stylebox_override("panel", style)
	_root.add_child(_panel)

	var main_vbox := VBoxContainer.new()
	main_vbox.add_theme_constant_override("separation", 6)
	_panel.add_child(main_vbox)

	var title := Label.new()
	title.text = "LIVE TUNING [F10] (no bloquea conduccion)"
	title.add_theme_font_size_override("font_size", 14)
	title.add_theme_color_override("font_color", Color(0.4, 0.88, 1.0))
	main_vbox.add_child(title)

	_tabs = TabContainer.new()
	_tabs.focus_mode = Control.FOCUS_NONE
	main_vbox.add_child(_tabs)

	_build_tv_tab()
	_build_gamepad_tab()


func _build_tv_tab() -> void:
	var vbox := VBoxContainer.new()
	vbox.name = "Torque Vectoring"
	vbox.add_theme_constant_override("separation", 6)
	_tabs.add_child(vbox)

	_telem_label = Label.new()
	_telem_label.text = "wz_ref: 0.00 | wz: 0.00 | beta: 0.0 deg"
	_telem_label.add_theme_font_size_override("font_size", 12)
	vbox.add_child(_telem_label)

	for spec in TV_SLIDER_SPECS:
		var k: String = spec["key"]
		var row := HBoxContainer.new()
		row.add_theme_constant_override("separation", 8)
		vbox.add_child(row)

		var lbl := Label.new()
		lbl.text = spec["label"]
		lbl.custom_minimum_size = Vector2(150, 0)
		lbl.add_theme_font_size_override("font_size", 12)
		row.add_child(lbl)

		var slider := HSlider.new()
		slider.min_value = spec["min"]
		slider.max_value = spec["max"]
		slider.step = spec["step"]
		slider.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		slider.focus_mode = Control.FOCUS_NONE
		slider.value_changed.connect(_on_tv_slider_changed.bind(k, spec["fmt"]))
		row.add_child(slider)
		_tv_sliders[k] = slider

		var val_lbl := Label.new()
		val_lbl.custom_minimum_size = Vector2(58, 0)
		val_lbl.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
		val_lbl.add_theme_font_size_override("font_size", 12)
		row.add_child(val_lbl)
		_tv_val_labels[k] = val_lbl

	var btn_row := HBoxContainer.new()
	btn_row.add_theme_constant_override("separation", 6)
	vbox.add_child(btn_row)

	var btn_reset := Button.new()
	btn_reset.text = "Reset al Mapa F6"
	btn_reset.focus_mode = Control.FOCUS_NONE
	btn_reset.pressed.connect(_on_reset_pressed)
	btn_row.add_child(btn_reset)

	var btn_save := Button.new()
	btn_save.text = "Guardar Preset TV JSON"
	btn_save.focus_mode = Control.FOCUS_NONE
	btn_save.pressed.connect(_on_save_tv_json_pressed)
	btn_row.add_child(btn_save)

	var btn_copy := Button.new()
	btn_copy.text = "Copiar C Config"
	btn_copy.focus_mode = Control.FOCUS_NONE
	btn_copy.pressed.connect(_on_copy_c_pressed)
	btn_row.add_child(btn_copy)

	_status_label = Label.new()
	_status_label.text = ""
	_status_label.add_theme_font_size_override("font_size", 11)
	_status_label.add_theme_color_override("font_color", Color(0.65, 0.95, 0.65))
	vbox.add_child(_status_label)


func _build_gamepad_tab() -> void:
	var vbox := VBoxContainer.new()
	vbox.name = "Mando / Volante"
	vbox.add_theme_constant_override("separation", 5)
	_tabs.add_child(vbox)

	var preset_row := HBoxContainer.new()
	preset_row.add_theme_constant_override("separation", 6)
	vbox.add_child(preset_row)

	var btn_ps4 := Button.new()
	btn_ps4.text = "Preset Mando PS4"
	btn_ps4.focus_mode = Control.FOCUS_NONE
	btn_ps4.pressed.connect(_on_preset_ps4_pressed)
	preset_row.add_child(btn_ps4)

	var btn_dd := Button.new()
	btn_dd.text = "Preset Volante Direct Drive"
	btn_dd.focus_mode = Control.FOCUS_NONE
	btn_dd.pressed.connect(_on_preset_dd_pressed)
	preset_row.add_child(btn_dd)

	_gp_telem_label = Label.new()
	_gp_telem_label.text = "Steer +0.00 -> +0.00 | Gas 0.00 -> 0.00 | Brk 0.00 -> 0.00"
	_gp_telem_label.add_theme_font_size_override("font_size", 11)
	vbox.add_child(_gp_telem_label)

	var bars_grid := GridContainer.new()
	bars_grid.columns = 3
	bars_grid.add_theme_constant_override("h_separation", 6)
	bars_grid.add_theme_constant_override("v_separation", 2)
	vbox.add_child(bars_grid)

	_steer_raw_bar = _add_bar_pair(bars_grid, "Steer Raw/Filt")
	_steer_filt_bar = _add_single_bar(bars_grid)
	_th_raw_bar = _add_bar_pair(bars_grid, "Gas Raw/Filt")
	_th_filt_bar = _add_single_bar(bars_grid)
	_brk_raw_bar = _add_bar_pair(bars_grid, "Brk Raw/Filt")
	_brk_filt_bar = _add_single_bar(bars_grid)

	for spec in GP_SLIDER_SPECS:
		var k: String = spec["key"]
		var row := HBoxContainer.new()
		row.add_theme_constant_override("separation", 8)
		vbox.add_child(row)

		var lbl := Label.new()
		lbl.text = spec["label"]
		lbl.custom_minimum_size = Vector2(150, 0)
		lbl.add_theme_font_size_override("font_size", 12)
		row.add_child(lbl)

		var slider := HSlider.new()
		slider.min_value = spec["min"]
		slider.max_value = spec["max"]
		slider.step = spec["step"]
		slider.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		slider.focus_mode = Control.FOCUS_NONE
		slider.value_changed.connect(_on_gp_slider_changed.bind(k, spec["fmt"]))
		row.add_child(slider)
		_gp_sliders[k] = slider

		var val_lbl := Label.new()
		val_lbl.custom_minimum_size = Vector2(58, 0)
		val_lbl.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
		val_lbl.add_theme_font_size_override("font_size", 12)
		row.add_child(val_lbl)
		_gp_val_labels[k] = val_lbl

	var gp_btn_row := HBoxContainer.new()
	gp_btn_row.add_theme_constant_override("separation", 6)
	vbox.add_child(gp_btn_row)

	var btn_reload := Button.new()
	btn_reload.text = "Recargar gamepad.json"
	btn_reload.focus_mode = Control.FOCUS_NONE
	btn_reload.pressed.connect(_on_reload_gp_json_pressed)
	gp_btn_row.add_child(btn_reload)

	var btn_save_gp := Button.new()
	btn_save_gp.text = "Guardar en gamepad.json"
	btn_save_gp.focus_mode = Control.FOCUS_NONE
	btn_save_gp.pressed.connect(_on_save_gp_json_pressed)
	gp_btn_row.add_child(btn_save_gp)

	_gp_status_label = Label.new()
	_gp_status_label.text = ""
	_gp_status_label.add_theme_font_size_override("font_size", 11)
	_gp_status_label.add_theme_color_override("font_color", Color(0.65, 0.95, 0.65))
	vbox.add_child(_gp_status_label)


func _add_bar_pair(grid: GridContainer, title_txt: String) -> ProgressBar:
	var lbl := Label.new()
	lbl.text = title_txt
	lbl.custom_minimum_size = Vector2(92, 0)
	lbl.add_theme_font_size_override("font_size", 11)
	grid.add_child(lbl)
	return _add_single_bar(grid)


func _add_single_bar(grid: GridContainer) -> ProgressBar:
	var bar := ProgressBar.new()
	bar.min_value = 0.0
	bar.max_value = 100.0
	bar.show_percentage = false
	bar.custom_minimum_size = Vector2(140, 12)
	bar.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	grid.add_child(bar)
	return bar


func _on_tv_slider_changed(val: float, key: String, fmt: String) -> void:
	if _syncing:
		return
	if _tv_val_labels.has(key):
		(_tv_val_labels[key] as Label).text = fmt % val
	if _ecu != null and _ecu.has_method("set_live_param"):
		_ecu.set_live_param(key, val)


func _on_gp_slider_changed(val: float, key: String, fmt: String) -> void:
	if _syncing:
		return
	if _gp_val_labels.has(key):
		(_gp_val_labels[key] as Label).text = fmt % val
	if _car != null and _car.has_method("set_live_gamepad_param"):
		_car.set_live_gamepad_param(key, val)
	elif _gamepad != null and _gamepad.has_method("set_live_gamepad_param"):
		_gamepad.set_live_gamepad_param(key, val)


func _on_reset_pressed() -> void:
	if _ecu != null and _ecu.has_method("set_map"):
		_ecu.set_map(_ecu.active_map)
		sync_from_ecu()
		_status_label.text = "Mapa %s restaurado." % _ecu.active_map


func _on_save_tv_json_pressed() -> void:
	if _ecu == null or not _ecu.has_method("get_live_params"):
		return
	DirAccess.make_dir_recursive_absolute("user://tv_presets")
	var ts := int(Time.get_unix_time_from_system())
	var path := "user://tv_presets/custom_%d.json" % ts
	var f := FileAccess.open(path, FileAccess.WRITE)
	if f != null:
		f.store_string(JSON.stringify(_ecu.get_live_params(), "\t"))
		f.close()
		_status_label.text = "Guardado: " + path


func _on_copy_c_pressed() -> void:
	if _ecu != null and _ecu.has_method("export_c_header_snippet"):
		DisplayServer.clipboard_set(_ecu.export_c_header_snippet())
		_status_label.text = "Snippet C copiado al portapapeles."


func _on_preset_ps4_pressed() -> void:
	var preset := {
		"steering_deadzone": 0.05,
		"steering_gamma": 2.2,
		"steering_speed": 2.2,
		"steering_filter": 0.14,
		"speed_sensitivity": 0.55,
		"v_ref_mps": 28.0,
		"throttle_gamma": 1.6,
		"brake_gamma": 1.4,
		"rumble_effects": 0.8
	}
	_apply_gamepad_preset(preset, "Preset Mando PS4 aplicado.")


func _on_preset_dd_pressed() -> void:
	var preset := {
		"steering_deadzone": 0.0,
		"steering_gamma": 1.0,
		"steering_speed": 4.0,
		"steering_filter": 0.0,
		"speed_sensitivity": 0.0,
		"v_ref_mps": 28.0,
		"throttle_gamma": 1.0,
		"brake_gamma": 1.0,
		"rumble_effects": 0.5
	}
	_apply_gamepad_preset(preset, "Preset Volante Direct Drive aplicado.")


func _apply_gamepad_preset(preset: Dictionary, msg: String) -> void:
	for k in preset.keys():
		if _car != null and _car.has_method("set_live_gamepad_param"):
			_car.set_live_gamepad_param(k, float(preset[k]))
		elif _gamepad != null and _gamepad.has_method("set_live_gamepad_param"):
			_gamepad.set_live_gamepad_param(k, float(preset[k]))
	sync_from_gamepad()
	if _gp_status_label != null:
		_gp_status_label.text = msg


func _on_reload_gp_json_pressed() -> void:
	var ok := false
	if _gamepad != null and _gamepad.has_method("load_from_json"):
		ok = _gamepad.load_from_json()
	if ok and _car != null and _car.has_method("get_live_gamepad_params") and _gamepad.has_method("get_live_gamepad_params"):
		var loaded: Dictionary = _gamepad.get_live_gamepad_params()
		for k in loaded.keys():
			_car.set_live_gamepad_param(k, float(loaded[k]))
	sync_from_gamepad()
	if _gp_status_label != null:
		_gp_status_label.text = "gamepad.json recargado." if ok else "Error al recargar gamepad.json."


func _on_save_gp_json_pressed() -> void:
	var ok := false
	if _car != null and _car.has_method("save_gamepad_config_to_json"):
		ok = _car.save_gamepad_config_to_json()
	elif _gamepad != null and _gamepad.has_method("save_gamepad_config_to_json"):
		ok = _gamepad.save_gamepad_config_to_json()
	if _gp_status_label != null:
		_gp_status_label.text = "Guardado en gamepad.json." if ok else "Error al guardar gamepad.json."
