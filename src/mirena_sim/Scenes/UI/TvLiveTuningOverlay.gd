extends CanvasLayer
class_name TvLiveTuningOverlay

const VehicleConfigScript = preload("../Vehicle/Config/VehicleConfig.gd")
const SetupProfileScript = preload("../Vehicle/Config/SetupProfile.gd")
const TireModelScript = preload("../Vehicle/Dynamics/Tires/TireModel.gd")

const TV_SLIDER_SPECS := [
	{"key": "live_k_us", "label": "k_us (Understeer)", "min": 0.0, "max": 0.025, "step": 0.0005, "fmt": "%.4f"},
	{"key": "live_max_mz", "label": "max_mz (Nm)", "min": 200.0, "max": 2500.0, "step": 25.0, "fmt": "%.0f"},
	{"key": "live_min_fx_reserve", "label": "min_fx_reserve (N)", "min": 0.0, "max": 2500.0, "step": 50.0, "fmt": "%.0f"},
	{"key": "live_tc_kp", "label": "tc_kp", "min": 0.0, "max": 1000.0, "step": 25.0, "fmt": "%.0f"},
	{"key": "live_k_beta", "label": "k_beta (Nm/rad)", "min": 0.0, "max": 12000.0, "step": 100.0, "fmt": "%.0f"},
	{"key": "live_throttle_mz_cut", "label": "throttle_mz_cut", "min": 0.0, "max": 1.0, "step": 0.05, "fmt": "%.2f"},
	{"key": "front_drive_bias", "label": "front_drive_bias", "min": 0.0, "max": 1.0, "step": 0.05, "fmt": "%.2f"},
	{"key": "can_latency_ms", "label": "can_latency_ms", "min": 0.0, "max": 50.0, "step": 1.0, "fmt": "%.1f"},
	{"key": "control_law_mode", "label": "Law (0=V3, 1=SMC)", "min": 0.0, "max": 1.0, "step": 1.0, "fmt": "%.0f"},
	{"key": "live_smc_k_gain", "label": "smc_k_gain (Nm)", "min": 200.0, "max": 2000.0, "step": 25.0, "fmt": "%.0f"},
	{"key": "live_smc_k_ffd", "label": "smc_k_ffd (Nms/r)", "min": 0.0, "max": 500.0, "step": 10.0, "fmt": "%.0f"},
	{"key": "live_smc_lambda_beta", "label": "smc_lambda_beta", "min": 0.0, "max": 12.0, "step": 0.2, "fmt": "%.1f"},
	{"key": "live_smc_phi", "label": "smc_phi (rad/s)", "min": 0.03, "max": 0.50, "step": 0.01, "fmt": "%.2f"},
	{"key": "live_smc_tc_kappa_base", "label": "smc_tc_kappa_base", "min": 0.015, "max": 0.12, "step": 0.005, "fmt": "%.3f"}
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

const CHASSIS_SLIDER_SPECS := [
	{"key": "spring_stiffness_front", "label": "k_spring_F (N/m)", "min": 20000.0, "max": 80000.0, "step": 500.0, "fmt": "%.0f"},
	{"key": "spring_stiffness_rear", "label": "k_spring_R (N/m)", "min": 25000.0, "max": 90000.0, "step": 500.0, "fmt": "%.0f"},
	{"key": "arb_stiffness_front", "label": "k_arb_F (Nm/rad)", "min": 1000.0, "max": 15000.0, "step": 100.0, "fmt": "%.0f"},
	{"key": "arb_stiffness_rear", "label": "k_arb_R (Nm/rad)", "min": 1000.0, "max": 15000.0, "step": 100.0, "fmt": "%.0f"},
	{"key": "static_camber_front_deg", "label": "Camber F (deg)", "min": -4.5, "max": 0.0, "step": 0.1, "fmt": "%+.1f"},
	{"key": "static_camber_rear_deg", "label": "Camber R (deg)", "min": -4.0, "max": 0.0, "step": 0.1, "fmt": "%+.1f"},
	{"key": "toe_front_deg", "label": "Toe F (deg)", "min": -1.0, "max": 1.0, "step": 0.05, "fmt": "%+.2f"},
	{"key": "toe_rear_deg", "label": "Toe R (deg)", "min": -0.5, "max": 1.5, "step": 0.05, "fmt": "%+.2f"},
	{"key": "brake_bias_front", "label": "Brake Bias F", "min": 0.45, "max": 0.80, "step": 0.01, "fmt": "%.2f"},
	{"key": "aero_balance_front", "label": "Aero Balance F", "min": 0.35, "max": 0.65, "step": 0.01, "fmt": "%.2f"}
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
var _map_dropdown: OptionButton = null
var _event_mode_dropdown: OptionButton = null
var _btn_launch_arm: Button = null
var _map_name_edit: LineEdit = null
var _tv_maps_cache: Array[Dictionary] = []

const EVENT_MODE_CHOICES: Array = [
	["open", "OPEN 4WD"],
	["autocross", "AUTOCROSS"],
	["endurance", "ENDURANCE"],
	["skidpad", "SKIDPAD"],
	["acceleration", "ACCELERATION"],
	["tire_warmer", "TIRE WARMER"]
]

var _steer_raw_bar: ProgressBar = null
var _steer_filt_bar: ProgressBar = null
var _th_raw_bar: ProgressBar = null
var _th_filt_bar: ProgressBar = null
var _brk_raw_bar: ProgressBar = null
var _brk_filt_bar: ProgressBar = null
var _gp_telem_label: Label = null
var _gp_status_label: Label = null

var _setup_sliders: Dictionary = {}
var _setup_val_labels: Dictionary = {}
var _setup_dropdown: OptionButton = null
var _setup_name_edit: LineEdit = null
var _setup_presets_cache: Array[Dictionary] = []
var _active_setup_path: String = ""
var _pinn_tire_check: CheckBox = null
var _regen_blend_check: CheckBox = null
var _chassis_telem_label: Label = null
var _setup_status_label: Label = null

var _vehicle_dropdown: OptionButton = null
var _vehicles_cache: Array[Dictionary] = []
var _profile_setup_dropdown: OptionButton = null
var _profile_setups_cache: Array[Dictionary] = []
var _profile_name_edit: LineEdit = null
var _vehicle_status_label: Label = null

var _syncing: bool = false


func _ready() -> void:
	layer = 50
	_build_ui()
	visible = false


func _input(event: InputEvent) -> void:
	if visible and (event is InputEventJoypadMotion or event is InputEventJoypadButton):
		get_viewport().set_input_as_handled()


func bind_ecu(ecu) -> void:
	_ecu = ecu
	sync_from_ecu()


func bind_car(ecu, gamepad, car = null) -> void:
	_ecu = ecu
	_gamepad = gamepad
	_car = car
	sync_from_ecu()
	sync_from_gamepad()
	sync_from_setup()


func toggle_visibility() -> void:
	visible = not visible
	if visible:
		sync_from_ecu()
		sync_from_gamepad()
		sync_from_setup()


func is_text_input_focused() -> bool:
	var tv_focused := _map_name_edit != null and _map_name_edit.has_focus()
	var setup_focused := _setup_name_edit != null and _setup_name_edit.has_focus()
	var profile_focused := _profile_name_edit != null and _profile_name_edit.has_focus()
	return tv_focused or setup_focused or profile_focused


func sync_from_ecu() -> void:
	if _ecu == null or not _ecu.has_method("get_live_params"):
		return
	_refresh_tv_map_dropdown()
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


func sync_from_setup() -> void:
	_refresh_vehicle_dropdown()
	_refresh_vehicle_setup_dropdown()
	_refresh_setup_dropdown()
	var params: Dictionary = {}
	if _car != null and _car.has_method("get_setup_params"):
		params = _car.get_setup_params()
	elif _car != null and _car.get("config") != null and _car.config.has_method("get_setup_params"):
		params = _car.config.get_setup_params()
	_syncing = true
	for spec in CHASSIS_SLIDER_SPECS:
		var k: String = spec["key"]
		if params.has(k) and _setup_sliders.has(k):
			var v := float(params[k])
			(_setup_sliders[k] as HSlider).value = v
			(_setup_val_labels[k] as Label).text = spec["fmt"] % v
	if _pinn_tire_check != null:
		_pinn_tire_check.button_pressed = TireModelScript.enable_neural_correction
	if _regen_blend_check != null and params.has("regen_blending_enabled"):
		_regen_blend_check.button_pressed = bool(params["regen_blending_enabled"])
	_syncing = false


func update_telemetry(wz_ref: float, wz: float, beta_deg: float, mz_req: float, tc_active: bool) -> void:
	if not visible or _telem_label == null:
		return
	_sync_launch_button_state()
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


func update_chassis_telemetry(imo_temps: Array, disc_temps: Array, surface_mu: float) -> void:
	if not visible or _chassis_telem_label == null:
		return
	var fl_i := _format_imo(imo_temps, 0)
	var fr_i := _format_imo(imo_temps, 1)
	var rl_i := _format_imo(imo_temps, 2)
	var rr_i := _format_imo(imo_temps, 3)
	var d_fl := float(disc_temps[0]) if disc_temps.size() > 0 else 40.0
	var d_fr := float(disc_temps[1]) if disc_temps.size() > 1 else 40.0
	var d_rl := float(disc_temps[2]) if disc_temps.size() > 2 else 40.0
	var d_rr := float(disc_temps[3]) if disc_temps.size() > 3 else 40.0
	_chassis_telem_label.text = "Neumaticos I/M/O (C): FL %s | FR %s\n                      RL %s | RR %s\nDiscos (C): FL %3.0f | FR %3.0f | RL %3.0f | RR %3.0f | Surface Mu: %.2f" % [
		fl_i, fr_i, rl_i, rr_i, d_fl, d_fr, d_rl, d_rr, surface_mu
	]


func _format_imo(imo_temps: Array, idx: int) -> String:
	if idx >= imo_temps.size():
		return "--/--/--"
	var corner = imo_temps[idx]
	if corner is Array and corner.size() >= 3:
		return "%2.0f/%2.0f/%2.0f" % [float(corner[0]), float(corner[1]), float(corner[2])]
	return "--/--/--"


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

	var veh_row := HBoxContainer.new()
	veh_row.add_theme_constant_override("separation", 6)
	main_vbox.add_child(veh_row)

	var veh_lbl := Label.new()
	veh_lbl.text = "Vehiculo:"
	veh_lbl.custom_minimum_size = Vector2(68, 0)
	veh_lbl.add_theme_font_size_override("font_size", 12)
	veh_row.add_child(veh_lbl)

	_vehicle_dropdown = OptionButton.new()
	_vehicle_dropdown.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_vehicle_dropdown.focus_mode = Control.FOCUS_NONE
	_vehicle_dropdown.item_selected.connect(_on_vehicle_selected)
	veh_row.add_child(_vehicle_dropdown)

	var prof_row := HBoxContainer.new()
	prof_row.add_theme_constant_override("separation", 6)
	main_vbox.add_child(prof_row)

	var prof_lbl := Label.new()
	prof_lbl.text = "Set-up:"
	prof_lbl.custom_minimum_size = Vector2(68, 0)
	prof_lbl.add_theme_font_size_override("font_size", 12)
	prof_row.add_child(prof_lbl)

	_profile_setup_dropdown = OptionButton.new()
	_profile_setup_dropdown.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_profile_setup_dropdown.focus_mode = Control.FOCUS_NONE
	_profile_setup_dropdown.item_selected.connect(_on_vehicle_setup_selected)
	prof_row.add_child(_profile_setup_dropdown)

	_profile_name_edit = LineEdit.new()
	_profile_name_edit.placeholder_text = "nombre_setup"
	_profile_name_edit.custom_minimum_size = Vector2(115, 0)
	prof_row.add_child(_profile_name_edit)

	var btn_save_prof := Button.new()
	btn_save_prof.text = "Guardar Set-up Como..."
	btn_save_prof.focus_mode = Control.FOCUS_NONE
	btn_save_prof.pressed.connect(_on_save_vehicle_setup_as_pressed)
	prof_row.add_child(btn_save_prof)

	_vehicle_status_label = Label.new()
	_vehicle_status_label.text = ""
	_vehicle_status_label.add_theme_font_size_override("font_size", 11)
	_vehicle_status_label.add_theme_color_override("font_color", Color(0.65, 0.95, 0.65))
	main_vbox.add_child(_vehicle_status_label)

	_tabs = TabContainer.new()
	_tabs.focus_mode = Control.FOCUS_NONE
	_tabs.get_tab_bar().focus_mode = Control.FOCUS_NONE
	main_vbox.add_child(_tabs)

	_build_tv_tab()
	_build_gamepad_tab()
	_build_chassis_tab()


func _build_tv_tab() -> void:
	var vbox := VBoxContainer.new()
	vbox.name = "Torque Vectoring"
	vbox.add_theme_constant_override("separation", 6)
	_tabs.add_child(vbox)

	var map_row := HBoxContainer.new()
	map_row.add_theme_constant_override("separation", 6)
	vbox.add_child(map_row)

	_map_dropdown = OptionButton.new()
	_map_dropdown.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_map_dropdown.focus_mode = Control.FOCUS_NONE
	_map_dropdown.item_selected.connect(_on_map_selected)
	map_row.add_child(_map_dropdown)

	var btn_refresh := Button.new()
	btn_refresh.text = "Refresh"
	btn_refresh.tooltip_text = "Refrescar carpeta tv_maps/"
	btn_refresh.focus_mode = Control.FOCUS_NONE
	btn_refresh.pressed.connect(_refresh_tv_map_dropdown)
	map_row.add_child(btn_refresh)

	var mode_lc_row := HBoxContainer.new()
	mode_lc_row.add_theme_constant_override("separation", 6)
	vbox.add_child(mode_lc_row)

	_event_mode_dropdown = OptionButton.new()
	_event_mode_dropdown.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_event_mode_dropdown.focus_mode = Control.FOCUS_NONE
	for item in EVENT_MODE_CHOICES:
		_event_mode_dropdown.add_item(String(item[1]))
	_event_mode_dropdown.item_selected.connect(_on_event_mode_selected)
	mode_lc_row.add_child(_event_mode_dropdown)

	_btn_launch_arm = Button.new()
	_btn_launch_arm.text = "Armar Launch Control [L]"
	_btn_launch_arm.focus_mode = Control.FOCUS_NONE
	_btn_launch_arm.pressed.connect(_on_launch_arm_pressed)
	mode_lc_row.add_child(_btn_launch_arm)

	var save_row := HBoxContainer.new()
	save_row.add_theme_constant_override("separation", 6)
	vbox.add_child(save_row)

	_map_name_edit = LineEdit.new()
	_map_name_edit.placeholder_text = "Nombre del set-up..."
	_map_name_edit.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_map_name_edit.text_submitted.connect(func(_t: String): _map_name_edit.release_focus())
	save_row.add_child(_map_name_edit)

	var btn_save_map := Button.new()
	btn_save_map.text = "Save"
	btn_save_map.focus_mode = Control.FOCUS_NONE
	btn_save_map.pressed.connect(_on_save_overwrite_pressed)
	save_row.add_child(btn_save_map)

	var btn_save_as := Button.new()
	btn_save_as.text = "Save As"
	btn_save_as.focus_mode = Control.FOCUS_NONE
	btn_save_as.pressed.connect(_on_save_as_pressed)
	save_row.add_child(btn_save_as)

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
	btn_reset.text = "Reset Map"
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


func _refresh_tv_map_dropdown() -> void:
	if _map_dropdown == null or _ecu == null or not _ecu.has_method("list_available_tv_maps"):
		return
	_tv_maps_cache = _ecu.list_available_tv_maps()
	_map_dropdown.clear()
	var sel_idx := 0
	for i in range(_tv_maps_cache.size()):
		var entry: Dictionary = _tv_maps_cache[i]
		_map_dropdown.add_item("%s (%s)" % [String(entry["name"]), String(entry["filename"])], i)
		if String(entry["path"]) == _ecu.active_map_path:
			sel_idx = i
	if not _tv_maps_cache.is_empty():
		_map_dropdown.select(sel_idx)
		if _map_name_edit != null and not _map_name_edit.has_focus():
			_map_name_edit.text = String(_tv_maps_cache[sel_idx]["name"])
	if _event_mode_dropdown != null:
		if _car != null and ("sim_drive_mode" in _car):
			_event_mode_dropdown.select(clampi(int(_car.sim_drive_mode), 0, EVENT_MODE_CHOICES.size() - 1))
		elif _ecu != null:
			var ev: String = String(_ecu.get("active_event_mode"))
			for i in range(EVENT_MODE_CHOICES.size()):
				if String(EVENT_MODE_CHOICES[i][0]) == ev:
					_event_mode_dropdown.select(i)
					break


func _on_event_mode_selected(idx: int) -> void:
	if idx < 0 or idx >= EVENT_MODE_CHOICES.size():
		return
	if _car != null and _car.has_method("set_sim_drive_mode"):
		_car.set_sim_drive_mode(idx)
	elif _ecu != null and _ecu.has_method("set_event_mode"):
		_ecu.set_event_mode(String(EVENT_MODE_CHOICES[idx][0]))
	sync_from_ecu()
	if _status_label != null:
		_status_label.text = "Drive Mode: %s" % String(EVENT_MODE_CHOICES[idx][1])


func _on_launch_arm_pressed() -> void:
	if _car != null and _car.has_method("launch_control_arm"):
		var ok: bool = _car.launch_control_arm()
		if _status_label != null:
			_status_label.text = "LC ARMED (8 Nm)" if ok else "LC requiere vx < 0.5 y freno > 25%"
	elif _ecu != null and _ecu.has_method("request_launch_control_arm"):
		_ecu.request_launch_control_arm(0.0, 1.0)
	_sync_launch_button_state()


func _sync_launch_button_state() -> void:
	if _btn_launch_arm == null or _ecu == null or not _ecu.has_method("get_launch_state_name"):
		return
	var st: String = _ecu.get_launch_state_name()
	if st == "LC_ARMED":
		_btn_launch_arm.text = "LC ARMED [L]"
	elif st == "LC_ACTIVE":
		_btn_launch_arm.text = "LC ACTIVE"
	else:
		_btn_launch_arm.text = "Armar Launch Control [L]"


func _on_map_selected(idx: int) -> void:
	if idx < 0 or idx >= _tv_maps_cache.size() or _ecu == null:
		return
	var entry: Dictionary = _tv_maps_cache[idx]
	if _ecu.load_tv_map_from_file(String(entry["path"])):
		sync_from_ecu()
		if _status_label != null:
			_status_label.text = "Cargado: %s" % String(entry["filename"])


func _on_save_overwrite_pressed() -> void:
	if _ecu == null or not _ecu.has_method("save_current_tv_map_to_file"):
		return
	var map_name := _map_name_edit.text.strip_edges() if _map_name_edit != null else ""
	if map_name == "":
		map_name = _ecu.active_map_display_name if _ecu.active_map_display_name != "" else "custom_tv_map"
	var saved: String = _ecu.save_current_tv_map_to_file(map_name, "", _ecu.active_map_path)
	if saved != "":
		_refresh_tv_map_dropdown()
		_status_label.text = "Guardado: " + saved.get_file()


func _on_save_as_pressed() -> void:
	if _ecu == null or not _ecu.has_method("save_current_tv_map_to_file"):
		return
	var map_name := _map_name_edit.text.strip_edges() if _map_name_edit != null else ""
	if map_name == "":
		map_name = "custom_%d" % int(Time.get_unix_time_from_system())
	var saved: String = _ecu.save_current_tv_map_to_file(map_name, "", "")
	if saved != "":
		_refresh_tv_map_dropdown()
		_status_label.text = "Nuevo set-up: " + saved.get_file()


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


func _build_chassis_tab() -> void:
	var vbox := VBoxContainer.new()
	vbox.name = "Chasis / Set-up"
	vbox.add_theme_constant_override("separation", 5)
	_tabs.add_child(vbox)

	var preset_row := HBoxContainer.new()
	preset_row.add_theme_constant_override("separation", 6)
	vbox.add_child(preset_row)

	_setup_dropdown = OptionButton.new()
	_setup_dropdown.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_setup_dropdown.focus_mode = Control.FOCUS_NONE
	_setup_dropdown.item_selected.connect(_on_setup_selected)
	preset_row.add_child(_setup_dropdown)

	var btn_refresh := Button.new()
	btn_refresh.text = "Refresh"
	btn_refresh.tooltip_text = "Refrescar carpeta setup_presets/"
	btn_refresh.focus_mode = Control.FOCUS_NONE
	btn_refresh.pressed.connect(_refresh_setup_dropdown)
	preset_row.add_child(btn_refresh)

	var save_row := HBoxContainer.new()
	save_row.add_theme_constant_override("separation", 6)
	vbox.add_child(save_row)

	_setup_name_edit = LineEdit.new()
	_setup_name_edit.placeholder_text = "Nombre del preset..."
	_setup_name_edit.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_setup_name_edit.text_submitted.connect(func(_t: String): _setup_name_edit.release_focus())
	save_row.add_child(_setup_name_edit)

	var btn_save := Button.new()
	btn_save.text = "Guardar"
	btn_save.focus_mode = Control.FOCUS_NONE
	btn_save.pressed.connect(_on_setup_save_pressed)
	save_row.add_child(btn_save)

	var btn_save_as := Button.new()
	btn_save_as.text = "Guardar Como"
	btn_save_as.focus_mode = Control.FOCUS_NONE
	btn_save_as.pressed.connect(_on_setup_save_as_pressed)
	save_row.add_child(btn_save_as)

	var btn_reset_base := Button.new()
	btn_reset_base.text = "Reset a Baseline"
	btn_reset_base.focus_mode = Control.FOCUS_NONE
	btn_reset_base.pressed.connect(_on_setup_reset_baseline_pressed)
	save_row.add_child(btn_reset_base)

	for spec in CHASSIS_SLIDER_SPECS:
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
		slider.value_changed.connect(_on_chassis_slider_changed.bind(k, spec["fmt"]))
		row.add_child(slider)
		_setup_sliders[k] = slider

		var val_lbl := Label.new()
		val_lbl.custom_minimum_size = Vector2(58, 0)
		val_lbl.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
		val_lbl.add_theme_font_size_override("font_size", 12)
		row.add_child(val_lbl)
		_setup_val_labels[k] = val_lbl

	var toggle_row := HBoxContainer.new()
	toggle_row.add_theme_constant_override("separation", 12)
	vbox.add_child(toggle_row)

	_pinn_tire_check = CheckBox.new()
	_pinn_tire_check.text = "PINN Tire (TeR_Tire)"
	_pinn_tire_check.focus_mode = Control.FOCUS_NONE
	_pinn_tire_check.toggled.connect(_on_pinn_tire_toggled)
	toggle_row.add_child(_pinn_tire_check)

	_regen_blend_check = CheckBox.new()
	_regen_blend_check.text = "Brake Blending Regen"
	_regen_blend_check.focus_mode = Control.FOCUS_NONE
	_regen_blend_check.toggled.connect(_on_regen_blend_toggled)
	toggle_row.add_child(_regen_blend_check)

	_chassis_telem_label = Label.new()
	_chassis_telem_label.text = "Neumaticos I/M/O (C): FL 58/58/58 | FR 58/58/58\n                      RL 58/58/58 | RR 58/58/58\nDiscos (C): FL  40 | FR  40 | RL  40 | RR  40 | Surface Mu: 1.50"
	_chassis_telem_label.add_theme_font_size_override("font_size", 11)
	vbox.add_child(_chassis_telem_label)

	_setup_status_label = Label.new()
	_setup_status_label.text = ""
	_setup_status_label.add_theme_font_size_override("font_size", 11)
	_setup_status_label.add_theme_color_override("font_color", Color(0.65, 0.95, 0.65))
	vbox.add_child(_setup_status_label)


func _get_car_config():
	if _car != null:
		return _car.get("config")
	return null


func _refresh_vehicle_dropdown() -> void:
	if _vehicle_dropdown == null:
		return
	_vehicles_cache = VehicleConfigScript.list_available_vehicles()
	_vehicle_dropdown.clear()
	var cfg = _get_car_config()
	var active_id: String = String(cfg.vehicle_id) if cfg != null else "ter27_4wd"
	var sel_idx := 0
	for i in range(_vehicles_cache.size()):
		var entry: Dictionary = _vehicles_cache[i]
		_vehicle_dropdown.add_item(String(entry["display_name"]), i)
		if String(entry["vehicle_id"]) == active_id:
			sel_idx = i
	if not _vehicles_cache.is_empty():
		_vehicle_dropdown.select(sel_idx)


func _refresh_vehicle_setup_dropdown() -> void:
	if _profile_setup_dropdown == null:
		return
	var cfg = _get_car_config()
	var active_vid: String = String(cfg.vehicle_id) if cfg != null else "ter27_4wd"
	var active_setup: String = String(cfg.active_setup_name).to_lower() if cfg != null else "default"
	_profile_setups_cache = SetupProfileScript.list_setups_for_vehicle(active_vid)
	_profile_setup_dropdown.clear()
	var sel_idx := 0
	for i in range(_profile_setups_cache.size()):
		var entry: Dictionary = _profile_setups_cache[i]
		_profile_setup_dropdown.add_item("%s (%s)" % [String(entry["name"]), String(entry["stem"])], i)
		if String(entry["stem"]).to_lower() == active_setup or String(entry["name"]).to_lower() == active_setup:
			sel_idx = i
	if not _profile_setups_cache.is_empty():
		_profile_setup_dropdown.select(sel_idx)
		if _profile_name_edit != null and not _profile_name_edit.has_focus():
			_profile_name_edit.text = String(_profile_setups_cache[sel_idx]["name"])


func _on_vehicle_selected(idx: int) -> void:
	if _syncing or idx < 0 or idx >= _vehicles_cache.size():
		return
	var entry: Dictionary = _vehicles_cache[idx]
	var vid: String = String(entry["vehicle_id"])
	if _car != null and _car.has_method("switch_vehicle"):
		if _car.switch_vehicle(vid, "default"):
			if _car.get("model") != null and _car.model.get("ecu_tv_v3") != null:
				_ecu = _car.model.ecu_tv_v3
			sync_from_ecu()
			sync_from_setup()
			if _vehicle_status_label != null:
				_vehicle_status_label.text = "Vehiculo activo: %s" % String(entry["display_name"])


func _on_vehicle_setup_selected(idx: int) -> void:
	if _syncing or idx < 0 or idx >= _profile_setups_cache.size():
		return
	var entry: Dictionary = _profile_setups_cache[idx]
	var stem: String = String(entry["stem"])
	if _car != null and _car.has_method("apply_vehicle_setup"):
		if _car.apply_vehicle_setup(stem):
			sync_from_ecu()
			sync_from_setup()
			if _vehicle_status_label != null:
				_vehicle_status_label.text = "Set-up aplicado: %s" % String(entry["name"])


func _on_save_vehicle_setup_as_pressed() -> void:
	if _car == null or not _car.has_method("save_vehicle_setup"):
		return
	var raw_name := _profile_name_edit.text.strip_edges() if _profile_name_edit != null else ""
	if raw_name == "":
		raw_name = "custom_%d" % int(Time.get_unix_time_from_system())
	var saved_path: String = _car.save_vehicle_setup(raw_name)
	if saved_path != "":
		sync_from_setup()
		if _vehicle_status_label != null:
			_vehicle_status_label.text = "Set-up guardado: %s" % saved_path.get_file()


func _refresh_setup_dropdown() -> void:
	if _setup_dropdown == null:
		return
	_setup_presets_cache = VehicleConfigScript.list_setup_presets()
	_setup_dropdown.clear()
	var sel_idx := 0
	for i in range(_setup_presets_cache.size()):
		var entry: Dictionary = _setup_presets_cache[i]
		_setup_dropdown.add_item("%s (%s)" % [String(entry["name"]), String(entry["filename"])], i)
		if String(entry["path"]) == _active_setup_path:
			sel_idx = i
	if not _setup_presets_cache.is_empty():
		_setup_dropdown.select(sel_idx)
		if _active_setup_path == "":
			_active_setup_path = String(_setup_presets_cache[sel_idx]["path"])
		if _setup_name_edit != null and not _setup_name_edit.has_focus():
			_setup_name_edit.text = String(_setup_presets_cache[sel_idx]["name"])


func _on_chassis_slider_changed(val: float, key: String, fmt: String) -> void:
	if _syncing:
		return
	if _setup_val_labels.has(key):
		(_setup_val_labels[key] as Label).text = fmt % val
	if _car != null and _car.has_method("apply_live_setup_param"):
		_car.apply_live_setup_param(key, val)
	else:
		var cfg = _get_car_config()
		if cfg != null and cfg.has_method("apply_setup_params"):
			cfg.apply_setup_params({key: val})


func _on_pinn_tire_toggled(toggled_on: bool) -> void:
	if _syncing:
		return
	TireModelScript.enable_neural_correction = toggled_on
	if _car != null and _car.has_method("apply_live_setup_param"):
		_car.apply_live_setup_param("pinn_tire_enabled", toggled_on)
	if _setup_status_label != null:
		_setup_status_label.text = "PINN Tire (TeR_Tire): %s" % ("ON" if toggled_on else "OFF")


func _on_regen_blend_toggled(toggled_on: bool) -> void:
	if _syncing:
		return
	if _car != null and _car.has_method("apply_live_setup_param"):
		_car.apply_live_setup_param("regen_blending_enabled", toggled_on)
	else:
		var cfg = _get_car_config()
		if cfg != null and cfg.has_method("apply_setup_params"):
			cfg.apply_setup_params({"regen_blending_enabled": toggled_on})
	if _setup_status_label != null:
		_setup_status_label.text = "Brake Blending Regen: %s" % ("ON" if toggled_on else "OFF")


func _on_setup_selected(idx: int) -> void:
	if idx < 0 or idx >= _setup_presets_cache.size():
		return
	var entry: Dictionary = _setup_presets_cache[idx]
	var path: String = String(entry["path"])
	var data: Dictionary = VehicleConfigScript.load_setup_preset(path)
	var params: Dictionary = data.get("params", entry.get("params", {}))
	if params.is_empty():
		return
	if _car != null and _car.has_method("apply_setup_preset"):
		_car.apply_setup_preset(params)
	else:
		var cfg = _get_car_config()
		if cfg != null and cfg.has_method("apply_setup_params"):
			cfg.apply_setup_params(params)
	_active_setup_path = path
	sync_from_setup()
	if _setup_status_label != null:
		_setup_status_label.text = "Cargado: %s" % String(entry["filename"])


func _on_setup_save_pressed() -> void:
	var cfg = _get_car_config()
	if cfg == null or not cfg.has_method("save_setup_preset"):
		return
	var preset_name := _setup_name_edit.text.strip_edges() if _setup_name_edit != null else ""
	var stem := _active_setup_path.get_file().get_basename() if _active_setup_path != "" else ""
	if preset_name == "":
		preset_name = stem if stem != "" else "custom_setup"
	var saved: String = cfg.save_setup_preset(preset_name, "", stem)
	if saved != "":
		_active_setup_path = saved
		_refresh_setup_dropdown()
		if _setup_status_label != null:
			_setup_status_label.text = "Guardado: " + saved.get_file()


func _on_setup_save_as_pressed() -> void:
	var cfg = _get_car_config()
	if cfg == null or not cfg.has_method("save_setup_preset"):
		return
	var preset_name := _setup_name_edit.text.strip_edges() if _setup_name_edit != null else ""
	if preset_name == "":
		preset_name = "custom_setup_%d" % int(Time.get_unix_time_from_system())
	var saved: String = cfg.save_setup_preset(preset_name, "", "")
	if saved != "":
		_active_setup_path = saved
		_refresh_setup_dropdown()
		if _setup_status_label != null:
			_setup_status_label.text = "Nuevo set-up: " + saved.get_file()


func _on_setup_reset_baseline_pressed() -> void:
	var baseline_path := ""
	for entry in _setup_presets_cache:
		if String(entry["filename"]) == "00_baseline_ter27":
			baseline_path = String(entry["path"])
			break
	if baseline_path == "":
		baseline_path = VehicleConfigScript.get_setup_presets_dir().rstrip("/") + "/00_baseline_ter27.json"
	var data: Dictionary = VehicleConfigScript.load_setup_preset(baseline_path)
	var params: Dictionary = data.get("params", {})
	if params.is_empty():
		params = {
			"spring_stiffness_front": 35000.0,
			"spring_stiffness_rear": 48000.0,
			"arb_stiffness_front": 4500.0,
			"arb_stiffness_rear": 5500.0,
			"static_camber_front_deg": -2.0,
			"static_camber_rear_deg": -1.5,
			"toe_front_deg": 0.0,
			"toe_rear_deg": 0.0,
			"brake_bias_front": 0.58,
			"aero_balance_front": 0.48,
			"regen_blending_enabled": true,
			"static_ride_height_front": 0.035,
			"static_ride_height_rear": 0.035
		}
	if _car != null and _car.has_method("apply_setup_preset"):
		_car.apply_setup_preset(params)
	else:
		var cfg = _get_car_config()
		if cfg != null and cfg.has_method("apply_setup_params"):
			cfg.apply_setup_params(params)
	_active_setup_path = baseline_path
	sync_from_setup()
	if _setup_status_label != null:
		_setup_status_label.text = "Baseline TeR27 restaurado."
