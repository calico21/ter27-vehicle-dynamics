class_name TvLiveTuningOverlay
extends CanvasLayer

## Real-Time DiL Torque Vectoring Live Tuning Overlay (toggled with F8).
## Hidden by default; uses FOCUS_NONE on all controls so driving inputs are never blocked.

const PARAM_SPECS: Array[Dictionary] = [
	{"key": "live_k_us", "label": "k_us (Understeer)", "min": 0.0, "max": 0.030, "step": 0.0005, "fmt": "%.4f"},
	{"key": "live_max_mz", "label": "max_mz (Nm)", "min": 0.0, "max": 1400.0, "step": 10.0, "fmt": "%.0f"},
	{"key": "live_min_fx_reserve", "label": "min_fx_reserve", "min": 0.0, "max": 0.30, "step": 0.005, "fmt": "%.3f"},
	{"key": "live_tc_kp", "label": "tc_kp (TC Gain)", "min": 5.0, "max": 40.0, "step": 0.5, "fmt": "%.1f"},
	{"key": "live_k_beta", "label": "k_beta (Sideslip)", "min": 0.0, "max": 10000.0, "step": 100.0, "fmt": "%.0f"},
	{"key": "live_throttle_mz_cut", "label": "throttle_mz_cut", "min": 0.0, "max": 1.0, "step": 0.02, "fmt": "%.2f"}
]

var _ecu: EcuTorqueVectoringV3 = null
var _sliders: Dictionary = {}
var _val_labels: Dictionary = {}
var _telem_label: Label = null
var _status_label: Label = null
var _syncing_ui: bool = false

func _init() -> void:
	layer = 20
	visible = false

func bind_ecu(ecu: EcuTorqueVectoringV3) -> void:
	_ecu = ecu
	sync_from_ecu()

func _ready() -> void:
	var anchor := Control.new()
	anchor.set_anchors_preset(Control.PRESET_TOP_RIGHT)
	anchor.mouse_filter = Control.MOUSE_FILTER_IGNORE
	add_child(anchor)

	var panel := PanelContainer.new()
	panel.position = Vector2(-410, 16)
	panel.custom_minimum_size = Vector2(394, 0)
	var style := StyleBoxFlat.new()
	style.bg_color = Color(0.06, 0.07, 0.10, 0.84)
	style.border_color = Color(0.25, 0.55, 0.95, 0.65)
	style.set_border_width_all(1)
	style.set_corner_radius_all(8)
	style.content_margin_left = 14
	style.content_margin_right = 14
	style.content_margin_top = 10
	style.content_margin_bottom = 10
	panel.add_theme_stylebox_override("panel", style)
	anchor.add_child(panel)

	var vbox := VBoxContainer.new()
	vbox.add_theme_constant_override("separation", 6)
	panel.add_child(vbox)

	var title := Label.new()
	title.text = "TV LIVE TUNING [F8]"
	title.add_theme_font_size_override("font_size", 15)
	title.add_theme_color_override("font_color", Color(0.45, 0.80, 1.0))
	vbox.add_child(title)

	_telem_label = Label.new()
	_telem_label.text = "wz_ref: 0.00 | wz: 0.00 rad/s | beta: 0.0°\nmz_req: 0 Nm | TC: OFF"
	_telem_label.add_theme_font_size_override("font_size", 12)
	_telem_label.add_theme_color_override("font_color", Color(0.88, 0.92, 0.96))
	vbox.add_child(_telem_label)

	vbox.add_child(HSeparator.new())

	for spec in PARAM_SPECS:
		var key: String = String(spec["key"])
		var row := HBoxContainer.new()
		row.add_theme_constant_override("separation", 8)
		vbox.add_child(row)

		var lbl := Label.new()
		lbl.text = String(spec["label"])
		lbl.custom_minimum_size = Vector2(138, 0)
		lbl.add_theme_font_size_override("font_size", 12)
		row.add_child(lbl)

		var slider := HSlider.new()
		slider.min_value = float(spec["min"])
		slider.max_value = float(spec["max"])
		slider.step = float(spec["step"])
		slider.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		slider.focus_mode = Control.FOCUS_NONE
		slider.value_changed.connect(_on_slider_changed.bind(key))
		row.add_child(slider)

		var v_lbl := Label.new()
		v_lbl.custom_minimum_size = Vector2(58, 0)
		v_lbl.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
		v_lbl.add_theme_font_size_override("font_size", 12)
		row.add_child(v_lbl)

		_sliders[key] = slider
		_val_labels[key] = v_lbl

	vbox.add_child(HSeparator.new())

	var btn_row := HBoxContainer.new()
	btn_row.add_theme_constant_override("separation", 6)
	vbox.add_child(btn_row)

	var btn_reset := Button.new()
	btn_reset.text = "Reset al Mapa F6"
	btn_reset.focus_mode = Control.FOCUS_NONE
	btn_reset.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	btn_reset.pressed.connect(_on_reset_pressed)
	btn_row.add_child(btn_reset)

	var btn_save := Button.new()
	btn_save.text = "Guardar JSON"
	btn_save.focus_mode = Control.FOCUS_NONE
	btn_save.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	btn_save.pressed.connect(_on_save_json_pressed)
	btn_row.add_child(btn_save)

	var btn_copy := Button.new()
	btn_copy.text = "Copiar C Config"
	btn_copy.focus_mode = Control.FOCUS_NONE
	btn_copy.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	btn_copy.pressed.connect(_on_copy_c_pressed)
	btn_row.add_child(btn_copy)

	_status_label = Label.new()
	_status_label.text = ""
	_status_label.add_theme_font_size_override("font_size", 11)
	_status_label.add_theme_color_override("font_color", Color(0.55, 0.95, 0.65))
	vbox.add_child(_status_label)

	sync_from_ecu()

func toggle_visibility() -> bool:
	visible = not visible
	if visible:
		sync_from_ecu()
	return visible

func sync_from_ecu() -> void:
	if _ecu == null or _sliders.is_empty():
		return
	_syncing_ui = true
	var params: Dictionary = _ecu.get_live_params()
	for spec in PARAM_SPECS:
		var key: String = String(spec["key"])
		var val: float = float(params.get(key, 0.0))
		var slider: HSlider = _sliders[key]
		slider.value = val
		(_val_labels[key] as Label).text = String(spec["fmt"]) % val
	_syncing_ui = false

func update_telemetry(wz_ref: float, wz: float, beta_deg: float, mz_req: float, tc_active: bool) -> void:
	if not visible or _telem_label == null:
		return
	_telem_label.text = "wz_ref: %+.2f | wz: %+.2f rad/s | beta: %+.1f°\nmz_req: %+.0f Nm | TC: %s" % [
		wz_ref, wz, beta_deg, mz_req, "ACTIVE" if tc_active else "OFF"
	]

func _on_slider_changed(val: float, key: String) -> void:
	if _syncing_ui or _ecu == null:
		return
	_ecu.set_live_param(key, val)
	for spec in PARAM_SPECS:
		if String(spec["key"]) == key:
			(_val_labels[key] as Label).text = String(spec["fmt"]) % val
			break

func _on_reset_pressed() -> void:
	if _ecu == null:
		return
	_ecu.set_map(_ecu.active_map_id)
	sync_from_ecu()
	if _status_label != null:
		_status_label.text = "Reset a mapa %d (%s)" % [_ecu.active_map_id, _ecu.get_map_name()]

func _on_save_json_pressed() -> void:
	if _ecu == null:
		return
	var dir_path: String = "user://tv_presets/"
	DirAccess.make_dir_recursive_absolute(dir_path)
	var ts: String = Time.get_datetime_string_from_system(false, true).replace(":", "").replace("-", "").replace(" ", "_")
	var file_path: String = "%scustom_%s.json" % [dir_path, ts]
	var payload: Dictionary = _ecu.get_live_params()
	payload["base_map_id"] = _ecu.active_map_id
	payload["base_map_name"] = _ecu.get_map_name()
	var f: FileAccess = FileAccess.open(file_path, FileAccess.WRITE)
	if f != null:
		f.store_string(JSON.stringify(payload, "  "))
		f.close()
		if _status_label != null:
			_status_label.text = "Guardado: %s" % file_path

func _on_copy_c_pressed() -> void:
	if _ecu == null:
		return
	DisplayServer.clipboard_set(_ecu.export_c_header_snippet())
	if _status_label != null:
		_status_label.text = "C Config copiado al portapapeles"
