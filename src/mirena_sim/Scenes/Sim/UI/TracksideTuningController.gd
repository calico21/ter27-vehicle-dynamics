class_name TracksideTuningController
extends VBoxContainer

## Setup / TV Tab Controller inside SimUI MainMenu.
## Live 5-map TV calibration, Undo/Redo, A/B comparator, user://tv_presets JSON,
## STM32 C header export, and automatic Last Launch telemetry block.

const PRESETS_DIR := "user://tv_presets"
const STM32_EXPORT_PATH := "user://tv_presets/exported_tv_params.h"
const TvFrictionYmdWidgetScript = preload("res://Scenes/Sim/UI/TvFrictionYmdWidget.gd")

const TV_MAP_NAMES: Array[String] = [
	"Autocross",
	"Acceleration",
	"Skidpad",
	"Endurance",
	"Tire Warmer",
]

# Maps display index -> car.set_sim_drive_mode() index
const TV_MAP_TO_SIM_MODE: Array[int] = [1, 4, 3, 2, 5]

const EVENT_SETUP_PRESET_LABELS: Array[String] = [
	"Base",
	"Optimal Skidpad",
	"Optimal Accel",
	"Optimal Autocross",
	"Optimal Endurance",
]
const EVENT_SETUP_PRESET_STEMS: Array[String] = [
	"default",
	"optimal_skidpad",
	"optimal_accel",
	"optimal_autocross",
	"optimal_endurance",
]

const PARAM_DEFS: Array[Dictionary] = [
	{"section": "Hairpin Pivot & Trail-Brake", "key": "hairpin_pivot_gain", "label": "Hairpin Pivot", "min": 0.50, "max": 2.00, "step": 0.05, "def": 1.00, "fmt": "%.2fx"},
	{"section": "Hairpin Pivot & Trail-Brake", "key": "inner_regen_max_nm", "label": "Inner Regen Max", "min": 0.0, "max": 120.0, "step": 5.0, "def": 55.0, "fmt": "%.0f Nm"},
	{"section": "Hairpin Pivot & Trail-Brake", "key": "trail_brake_yaw_gain", "label": "Trail-Brake Yaw", "min": 0.00, "max": 2.50, "step": 0.05, "def": 1.00, "fmt": "%.2fx"},
	{"section": "Hairpin Pivot & Trail-Brake", "key": "exit_traction_blend", "label": "Exit Traction Blend", "min": 0.00, "max": 1.00, "step": 0.05, "def": 0.50, "fmt": "%.2f"},

	{"section": "Launch & TC", "key": "launch_rear_bias_pct", "label": "Launch Rear Bias", "min": 50.0, "max": 80.0, "step": 1.0, "def": 70.0, "fmt": "%.0f%%"},
	{"section": "Launch & TC", "key": "tc_slip_target", "label": "TC Slip Target", "min": 0.04, "max": 0.25, "step": 0.005, "def": 0.13, "fmt": "%.3f"},
	{"section": "Launch & TC", "key": "friction_ellipse_margin", "label": "Friction Ellipse Margin [%]", "min": 0.70, "max": 1.05, "step": 0.01, "def": 0.94, "fmt": "%.0f%%", "disp_scale": 100.0},

	{"section": "Skidpad & Sweeper", "key": "skidpad_neutral_trim", "label": "Skidpad Neutral Trim", "min": -25.0, "max": 25.0, "step": 1.0, "def": 0.0, "fmt": "%+.0f Nm"},
	{"section": "Skidpad & Sweeper", "key": "yaw_agility_gain", "label": "Yaw Agility", "min": 0.50, "max": 2.00, "step": 0.05, "def": 1.00, "fmt": "%.2fx"},
	{"section": "Skidpad & Sweeper", "key": "yaw_accel_ff_gain", "label": "Yaw Accel FF Gain", "min": 0.30, "max": 2.00, "step": 0.05, "def": 1.00, "fmt": "%.2fx"},
	{"section": "Skidpad & Sweeper", "key": "centripetal_fx_gain", "label": "Centripetal Fx Gain", "min": 0.0, "max": 50.0, "step": 1.0, "def": 15.0, "fmt": "%.0f"},
	{"section": "Skidpad & Sweeper", "key": "delta_alpha_eq_gain", "label": "Delta-Alpha Eq Gain", "min": 0.0, "max": 4000.0, "step": 20.0, "def": 2620.0, "fmt": "%.0f"},

	{"section": "Slalom & High-Speed", "key": "delta_dot_lead_gain", "label": "Delta-Dot Lead", "min": 0.0, "max": 120.0, "step": 2.0, "def": 30.0, "fmt": "%.0f"},
	{"section": "Slalom & High-Speed", "key": "slalom_beta_damp", "label": "Slalom Beta Damp", "min": 0.0, "max": 1500.0, "step": 25.0, "def": 400.0, "fmt": "%.0f"},
	{"section": "Slalom & High-Speed", "key": "beta_soft_deg", "label": "Beta Soft Limit", "min": 1.0, "max": 10.0, "step": 0.1, "def": 3.0, "fmt": "%.1f deg"},

	{"section": "Power, Regen & Chassis", "key": "power_limit_kw", "label": "Power Limit", "min": 10.0, "max": 80.0, "step": 1.0, "def": 80.0, "fmt": "%.0f kW"},
	{"section": "Power, Regen & Chassis", "key": "regen_limit_kw", "label": "Regen Limit", "min": -25.0, "max": 0.0, "step": 1.0, "def": -25.0, "fmt": "%.0f kW"},
	{"section": "Power, Regen & Chassis", "key": "lift_regen_nm", "label": "Lift Regen", "min": 0.0, "max": 100.0, "step": 1.0, "def": 12.0, "fmt": "%.0f Nm"},
	{"section": "Power, Regen & Chassis", "key": "brake_bias_front_pct", "label": "Brake Bias F", "min": 40.0, "max": 75.0, "step": 0.5, "def": 62.0, "fmt": "%.1f%%"},
	{"section": "Power, Regen & Chassis", "key": "arb_front_balance_pct", "label": "ARB Front Bal", "min": 20.0, "max": 80.0, "step": 1.0, "def": 50.0, "fmt": "%.0f%%"},
	{"section": "Power, Regen & Chassis", "key": "throttle_exp", "label": "Throttle Exp", "min": 0.50, "max": 2.50, "step": 0.05, "def": 1.00, "fmt": "%.2f"},
]

var _veh_opt_tv: OptionButton = null
var _kin_opt_tv: OptionButton = null
var _chk_xray_susp_tv: CheckButton = null
var _event_setup_opt: OptionButton = null
var _btn_apply_event_preset: Button = null
var _event_setup_lbl: Label = null
var _map_opt: OptionButton = null
var _sliders: Dictionary = {}      # key -> HSlider
var _val_labels: Dictionary = {}   # key -> Label
var _defaults: Dictionary = {}     # key -> float
var _current: Dictionary = {}      # key -> float
var _syncing: bool = false

# Undo / Redo
var _undo_stack: Array[Dictionary] = []
var _redo_stack: Array[Dictionary] = []
var _btn_undo: Button = null
var _btn_redo: Button = null

# Setup A / B comparator
var _setup_a: Dictionary = {}
var _setup_b: Dictionary = {}
var _active_slot: String = "A"
var _btn_toggle_ab: Button = null

# Presets & Export
var _diag_lbl: Label = null
var _kin_lbl: Label = null
var _friction_ymd_widget: TvFrictionYmdWidgetScript = null
var _preset_name_edit: LineEdit = null
var _preset_opt: OptionButton = null
var _diff_lbl: Label = null
var _status_lbl: Label = null
var _active_preset_name: String = "default"

# Last Launch stats
var _launch_lbl: Label = null
var _launch_armed: bool = false
var _launch_active: bool = false
var _launch_t: float = 0.0
var _launch_dist_m: float = 0.0
var _t_0_25: float = -1.0
var _t_0_50: float = -1.0
var _t_0_100: float = -1.0
var _t_75m: float = -1.0
var _peak_ax_g: float = 0.0
var _last_launch_summary: String = "Last Launch: -- (stop & launch to record)"

# Sensors & CAN Latency controls
var _sensor_mode_opt: OptionButton = null
var _sld_can_latency_ms: HSlider = null
var _lbl_can_latency_ms: Label = null
var _sld_imu_noise_scale: HSlider = null
var _lbl_imu_noise_scale: Label = null
var _sld_sensor_seed: HSlider = null
var _lbl_sensor_seed: Label = null
var _btn_reset_ekf: Button = null

# CasADi Reference & 3D Racing Line controls
var _casadi_ref_opt: OptionButton = null
var _line_3d_mode_opt: OptionButton = null
var _sld_brake_3d_height: HSlider = null
var _lbl_brake_3d_height: Label = null
var _sld_speed_alert_gain: HSlider = null
var _lbl_speed_alert_gain: Label = null
var _btn_export_dil_session: Button = null

# Live DiL & Diagnostic Readout Panel
var _dil_diag_lbl: Label = null
var _dil_sector_lbl: Label = null
var _dil_hint_lbl: Label = null


func _ready() -> void:
	for p in PARAM_DEFS:
		var k: String = p["key"]
		var d: float = float(p["def"])
		_defaults[k] = d
		_current[k] = d
	_build_ui()
	_refresh_preset_list()


func _make_mono_font() -> SystemFont:
	var f := SystemFont.new()
	f.font_names = PackedStringArray(["DejaVu Sans Mono", "Liberation Mono", "Monospace"])
	return f


func _build_ui() -> void:
	add_theme_constant_override("separation", 5)
	var mono := _make_mono_font()

	# Clean Vehicle, Suspension Model & 3D X-Ray Selector Bar
	var setup_panel := PanelContainer.new()
	var setup_sb := StyleBoxFlat.new()
	setup_sb.bg_color = Color(0.10, 0.12, 0.16, 0.92)
	setup_sb.set_corner_radius_all(4)
	setup_sb.set_content_margin_all(6)
	setup_panel.add_theme_stylebox_override("panel", setup_sb)
	add_child(setup_panel)

	var setup_vb := VBoxContainer.new()
	setup_vb.add_theme_constant_override("separation", 4)
	setup_panel.add_child(setup_vb)

	var setup_hdr := Label.new()
	setup_hdr.text = "VEHICLE & SUSPENSION SETUP"
	setup_hdr.add_theme_font_size_override("font_size", 11)
	setup_hdr.add_theme_color_override("font_color", Color(0.74, 0.90, 1.0, 1.0))
	setup_vb.add_child(setup_hdr)

	var veh_row := HBoxContainer.new()
	veh_row.add_theme_constant_override("separation", 8)
	setup_vb.add_child(veh_row)
	var veh_lbl := Label.new()
	veh_lbl.text = "Vehicle"
	veh_lbl.custom_minimum_size = Vector2(96, 0)
	veh_lbl.add_theme_font_size_override("font_size", 11)
	veh_row.add_child(veh_lbl)
	_veh_opt_tv = OptionButton.new()
	_veh_opt_tv.name = "TvVehicleOpt"
	_veh_opt_tv.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_veh_opt_tv.focus_mode = Control.FOCUS_NONE
	_veh_opt_tv.add_item("TeR27 4WD", 0)
	_veh_opt_tv.set_item_metadata(0, "ter27_4wd")
	_veh_opt_tv.add_item("TeR26 Validado (RWD)", 1)
	_veh_opt_tv.set_item_metadata(1, "ter26_rwd")
	_veh_opt_tv.item_selected.connect(_on_tv_vehicle_selected)
	veh_row.add_child(_veh_opt_tv)

	var kin_row := HBoxContainer.new()
	kin_row.add_theme_constant_override("separation", 8)
	setup_vb.add_child(kin_row)
	var kin_lbl := Label.new()
	kin_lbl.text = "Susp Model"
	kin_lbl.custom_minimum_size = Vector2(96, 0)
	kin_lbl.add_theme_font_size_override("font_size", 11)
	kin_row.add_child(kin_lbl)
	_kin_opt_tv = OptionButton.new()
	_kin_opt_tv.name = "TvSuspModelOpt"
	_kin_opt_tv.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_kin_opt_tv.focus_mode = Control.FOCUS_NONE
	_kin_opt_tv.add_item("Doble Horquilla 3D", 0)
	_kin_opt_tv.add_item("Parametrico", 1)
	_kin_opt_tv.item_selected.connect(_on_tv_kinematics_selected)
	kin_row.add_child(_kin_opt_tv)

	_chk_xray_susp_tv = CheckButton.new()
	_chk_xray_susp_tv.name = "TvSuspXrayToggle"
	_chk_xray_susp_tv.text = "3D Suspension / X-Ray"
	_chk_xray_susp_tv.focus_mode = Control.FOCUS_NONE
	_chk_xray_susp_tv.toggled.connect(_on_tv_xray_susp_toggled)
	setup_vb.add_child(_chk_xray_susp_tv)

	var ev_row := HBoxContainer.new()
	ev_row.add_theme_constant_override("separation", 6)
	setup_vb.add_child(ev_row)
	var ev_lbl := Label.new()
	ev_lbl.text = "Event Setup Preset"
	ev_lbl.custom_minimum_size = Vector2(116, 0)
	ev_lbl.add_theme_font_size_override("font_size", 11)
	ev_row.add_child(ev_lbl)
	_event_setup_opt = OptionButton.new()
	_event_setup_opt.name = "EventSetupPresetOpt"
	_event_setup_opt.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_event_setup_opt.focus_mode = Control.FOCUS_NONE
	_event_setup_opt.clip_text = true
	for i in range(EVENT_SETUP_PRESET_LABELS.size()):
		_event_setup_opt.add_item(EVENT_SETUP_PRESET_LABELS[i], i)
	_event_setup_opt.select(0)
	_event_setup_opt.item_selected.connect(func(_idx: int): _event_setup_opt.release_focus())
	ev_row.add_child(_event_setup_opt)
	_btn_apply_event_preset = Button.new()
	_btn_apply_event_preset.name = "ApplyEventSetupPresetBtn"
	_btn_apply_event_preset.text = "Apply Preset"
	_btn_apply_event_preset.focus_mode = Control.FOCUS_NONE
	_btn_apply_event_preset.pressed.connect(_on_apply_event_preset_pressed)
	ev_row.add_child(_btn_apply_event_preset)

	_event_setup_lbl = Label.new()
	_event_setup_lbl.name = "EventSetupSummaryLabel"
	_event_setup_lbl.add_theme_font_override("font", mono)
	_event_setup_lbl.add_theme_font_size_override("font_size", 10)
	_event_setup_lbl.add_theme_color_override("font_color", Color(0.76, 0.92, 0.80, 1.0))
	_event_setup_lbl.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_event_setup_lbl.text = "Active Event Setup: Base (default)"
	setup_vb.add_child(_event_setup_lbl)

	# Live TV & Suspension Kinematics Diagnostics Row
	var diag_panel := PanelContainer.new()
	var diag_sb := StyleBoxFlat.new()
	diag_sb.bg_color = Color(0.10, 0.12, 0.16, 0.92)
	diag_sb.set_corner_radius_all(4)
	diag_sb.set_content_margin_all(6)
	diag_panel.add_theme_stylebox_override("panel", diag_sb)
	add_child(diag_panel)

	var diag_vb := VBoxContainer.new()
	diag_vb.add_theme_constant_override("separation", 3)
	diag_panel.add_child(diag_vb)

	_diag_lbl = Label.new()
	_diag_lbl.add_theme_font_override("font", mono)
	_diag_lbl.add_theme_font_size_override("font_size", 11)
	_diag_lbl.text = "Regime: BALANCED      | Launch State: IDLE      | Mz_TV:   +0.0 Nm\nOut/In:  1.00 | Pivot:   0% | Beta: +0.00 / +0.0 dps\nBalance: NEUTRAL    | P_dc: +0.0 kW"
	diag_vb.add_child(_diag_lbl)

	_kin_lbl = Label.new()
	_kin_lbl.name = "KinematicsDiagLabel"
	_kin_lbl.add_theme_font_override("font", mono)
	_kin_lbl.add_theme_font_size_override("font_size", 11)
	_kin_lbl.add_theme_color_override("font_color", Color(0.76, 0.90, 0.98, 1.0))
	_kin_lbl.text = "Susp: 3D-DW | Cam FL/FR: +0.00/+0.00 | Toe: +0.00/+0.00\nAck:  +0.0% | MR F/R: 0.885/0.885    | LLTD: 50.0%"
	diag_vb.add_child(_kin_lbl)

	# Live Friction Circles + Torque Split + Instant YMD Preview
	_friction_ymd_widget = TvFrictionYmdWidgetScript.new()
	_friction_ymd_widget.name = "TvFrictionYmdWidget"
	_friction_ymd_widget.focus_mode = Control.FOCUS_NONE
	add_child(_friction_ymd_widget)
	_friction_ymd_widget.update_ymd_preview(_current, _defaults)

	# ------------------------------------------------------------------
	# SENSORS & CAN LATENCY PANEL
	# ------------------------------------------------------------------
	var sens_panel := PanelContainer.new()
	var sens_sb := StyleBoxFlat.new()
	sens_sb.bg_color = Color(0.10, 0.12, 0.16, 0.92)
	sens_sb.set_corner_radius_all(4)
	sens_sb.set_content_margin_all(6)
	sens_panel.add_theme_stylebox_override("panel", sens_sb)
	add_child(sens_panel)

	var sens_vb := VBoxContainer.new()
	sens_vb.add_theme_constant_override("separation", 4)
	sens_panel.add_child(sens_vb)

	var sens_hdr := Label.new()
	sens_hdr.text = "SENSORS & CAN LATENCY"
	sens_hdr.add_theme_font_size_override("font_size", 11)
	sens_hdr.add_theme_color_override("font_color", Color(0.74, 0.90, 1.0, 1.0))
	sens_vb.add_child(sens_hdr)

	var sens_mode_row := HBoxContainer.new()
	sens_mode_row.add_theme_constant_override("separation", 8)
	sens_vb.add_child(sens_mode_row)
	var sens_mode_lbl := Label.new()
	sens_mode_lbl.text = "Sensor Profile"
	sens_mode_lbl.custom_minimum_size = Vector2(128, 0)
	sens_mode_lbl.add_theme_font_size_override("font_size", 11)
	sens_mode_row.add_child(sens_mode_lbl)
	_sensor_mode_opt = OptionButton.new()
	_sensor_mode_opt.name = "SensorRealismModeOpt"
	_sensor_mode_opt.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_sensor_mode_opt.focus_mode = Control.FOCUS_NONE
	_sensor_mode_opt.add_item("Ideal", 0)
	_sensor_mode_opt.add_item("Track Realism", 1)
	_sensor_mode_opt.item_selected.connect(_on_sensor_mode_selected)
	sens_mode_row.add_child(_sensor_mode_opt)

	var lat_row := HBoxContainer.new()
	lat_row.add_theme_constant_override("separation", 6)
	sens_vb.add_child(lat_row)
	var lat_lbl := Label.new()
	lat_lbl.text = "CAN Latency [ms]"
	lat_lbl.custom_minimum_size = Vector2(128, 0)
	lat_lbl.add_theme_font_size_override("font_size", 11)
	lat_row.add_child(lat_lbl)
	_sld_can_latency_ms = HSlider.new()
	_sld_can_latency_ms.name = "SensorCanLatencySlider"
	_sld_can_latency_ms.min_value = 0.0
	_sld_can_latency_ms.max_value = 200.0
	_sld_can_latency_ms.step = 5.0
	_sld_can_latency_ms.value = 125.0
	_sld_can_latency_ms.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_sld_can_latency_ms.focus_mode = Control.FOCUS_NONE
	_sld_can_latency_ms.value_changed.connect(_on_can_latency_changed)
	lat_row.add_child(_sld_can_latency_ms)
	_lbl_can_latency_ms = Label.new()
	_lbl_can_latency_ms.add_theme_font_override("font", mono)
	_lbl_can_latency_ms.add_theme_font_size_override("font_size", 11)
	_lbl_can_latency_ms.custom_minimum_size = Vector2(64, 0)
	_lbl_can_latency_ms.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	_lbl_can_latency_ms.text = "125 ms"
	lat_row.add_child(_lbl_can_latency_ms)

	var noise_row := HBoxContainer.new()
	noise_row.add_theme_constant_override("separation", 6)
	sens_vb.add_child(noise_row)
	var noise_lbl := Label.new()
	noise_lbl.text = "IMU Noise Scale"
	noise_lbl.custom_minimum_size = Vector2(128, 0)
	noise_lbl.add_theme_font_size_override("font_size", 11)
	noise_row.add_child(noise_lbl)
	_sld_imu_noise_scale = HSlider.new()
	_sld_imu_noise_scale.name = "SensorImuNoiseSlider"
	_sld_imu_noise_scale.min_value = 0.0
	_sld_imu_noise_scale.max_value = 2.0
	_sld_imu_noise_scale.step = 0.05
	_sld_imu_noise_scale.value = 1.0
	_sld_imu_noise_scale.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_sld_imu_noise_scale.focus_mode = Control.FOCUS_NONE
	_sld_imu_noise_scale.value_changed.connect(_on_imu_noise_scale_changed)
	noise_row.add_child(_sld_imu_noise_scale)
	_lbl_imu_noise_scale = Label.new()
	_lbl_imu_noise_scale.add_theme_font_override("font", mono)
	_lbl_imu_noise_scale.add_theme_font_size_override("font_size", 11)
	_lbl_imu_noise_scale.custom_minimum_size = Vector2(64, 0)
	_lbl_imu_noise_scale.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	_lbl_imu_noise_scale.text = "1.00x"
	noise_row.add_child(_lbl_imu_noise_scale)

	var seed_row := HBoxContainer.new()
	seed_row.add_theme_constant_override("separation", 6)
	sens_vb.add_child(seed_row)
	var seed_lbl := Label.new()
	seed_lbl.text = "Seed"
	seed_lbl.custom_minimum_size = Vector2(128, 0)
	seed_lbl.add_theme_font_size_override("font_size", 11)
	seed_row.add_child(seed_lbl)
	_sld_sensor_seed = HSlider.new()
	_sld_sensor_seed.name = "SensorSeedSlider"
	_sld_sensor_seed.min_value = 0.0
	_sld_sensor_seed.max_value = 999.0
	_sld_sensor_seed.step = 1.0
	_sld_sensor_seed.value = 42.0
	_sld_sensor_seed.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_sld_sensor_seed.focus_mode = Control.FOCUS_NONE
	_sld_sensor_seed.value_changed.connect(_on_sensor_seed_changed)
	seed_row.add_child(_sld_sensor_seed)
	_lbl_sensor_seed = Label.new()
	_lbl_sensor_seed.add_theme_font_override("font", mono)
	_lbl_sensor_seed.add_theme_font_size_override("font_size", 11)
	_lbl_sensor_seed.custom_minimum_size = Vector2(64, 0)
	_lbl_sensor_seed.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	_lbl_sensor_seed.text = "42"
	seed_row.add_child(_lbl_sensor_seed)

	_btn_reset_ekf = Button.new()
	_btn_reset_ekf.name = "ResetEkfFiltersBtn"
	_btn_reset_ekf.text = "Reset EKF / Filters"
	_btn_reset_ekf.focus_mode = Control.FOCUS_NONE
	_btn_reset_ekf.pressed.connect(_on_reset_ekf_pressed)
	sens_vb.add_child(_btn_reset_ekf)

	# ------------------------------------------------------------------
	# CASADI REFERENCE & 3D RACING LINE PANEL
	# ------------------------------------------------------------------
	var dil_panel := PanelContainer.new()
	var dil_sb := StyleBoxFlat.new()
	dil_sb.bg_color = Color(0.10, 0.12, 0.16, 0.92)
	dil_sb.set_corner_radius_all(4)
	dil_sb.set_content_margin_all(6)
	dil_panel.add_theme_stylebox_override("panel", dil_sb)
	add_child(dil_panel)

	var dil_vb := VBoxContainer.new()
	dil_vb.add_theme_constant_override("separation", 4)
	dil_panel.add_child(dil_vb)

	var dil_hdr := Label.new()
	dil_hdr.text = "CASADI REFERENCE & 3D RACING LINE"
	dil_hdr.add_theme_font_size_override("font_size", 11)
	dil_hdr.add_theme_color_override("font_color", Color(0.74, 0.90, 1.0, 1.0))
	dil_vb.add_child(dil_hdr)

	var ref_row := HBoxContainer.new()
	ref_row.add_theme_constant_override("separation", 8)
	dil_vb.add_child(ref_row)
	var ref_lbl := Label.new()
	ref_lbl.text = "CasADi Reference"
	ref_lbl.custom_minimum_size = Vector2(142, 0)
	ref_lbl.add_theme_font_size_override("font_size", 11)
	ref_row.add_child(ref_lbl)
	_casadi_ref_opt = OptionButton.new()
	_casadi_ref_opt.name = "CasadiReferenceOpt"
	_casadi_ref_opt.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_casadi_ref_opt.focus_mode = Control.FOCUS_NONE
	_casadi_ref_opt.add_item("Off", 0)
	_casadi_ref_opt.add_item("Live Delta", 1)
	_casadi_ref_opt.select(1)
	_casadi_ref_opt.item_selected.connect(_on_casadi_ref_selected)
	ref_row.add_child(_casadi_ref_opt)

	var line_row := HBoxContainer.new()
	line_row.add_theme_constant_override("separation", 8)
	dil_vb.add_child(line_row)
	var line_lbl := Label.new()
	line_lbl.text = "3D Line Mode"
	line_lbl.custom_minimum_size = Vector2(142, 0)
	line_lbl.add_theme_font_size_override("font_size", 11)
	line_row.add_child(line_lbl)
	_line_3d_mode_opt = OptionButton.new()
	_line_3d_mode_opt.name = "RacingLine3DModeOpt"
	_line_3d_mode_opt.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_line_3d_mode_opt.focus_mode = Control.FOCUS_NONE
	_line_3d_mode_opt.add_item("Off", 0)
	_line_3d_mode_opt.add_item("Standard", 1)
	_line_3d_mode_opt.add_item("Sim-Racing 3D", 2)
	_line_3d_mode_opt.select(1)
	_line_3d_mode_opt.item_selected.connect(_on_line_3d_mode_selected)
	line_row.add_child(_line_3d_mode_opt)

	var brk_h_row := HBoxContainer.new()
	brk_h_row.add_theme_constant_override("separation", 6)
	dil_vb.add_child(brk_h_row)
	var brk_h_lbl := Label.new()
	brk_h_lbl.text = "Red Brake 3D Height [m]"
	brk_h_lbl.custom_minimum_size = Vector2(142, 0)
	brk_h_lbl.add_theme_font_size_override("font_size", 11)
	brk_h_row.add_child(brk_h_lbl)
	_sld_brake_3d_height = HSlider.new()
	_sld_brake_3d_height.name = "Brake3DHeightSlider"
	_sld_brake_3d_height.min_value = 0.0
	_sld_brake_3d_height.max_value = 1.5
	_sld_brake_3d_height.step = 0.05
	_sld_brake_3d_height.value = 0.45
	_sld_brake_3d_height.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_sld_brake_3d_height.focus_mode = Control.FOCUS_NONE
	_sld_brake_3d_height.value_changed.connect(_on_brake_3d_height_changed)
	brk_h_row.add_child(_sld_brake_3d_height)
	_lbl_brake_3d_height = Label.new()
	_lbl_brake_3d_height.add_theme_font_override("font", mono)
	_lbl_brake_3d_height.add_theme_font_size_override("font_size", 11)
	_lbl_brake_3d_height.custom_minimum_size = Vector2(64, 0)
	_lbl_brake_3d_height.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	_lbl_brake_3d_height.text = "0.45 m"
	brk_h_row.add_child(_lbl_brake_3d_height)

	var spd_a_row := HBoxContainer.new()
	spd_a_row.add_theme_constant_override("separation", 6)
	dil_vb.add_child(spd_a_row)
	var spd_a_lbl := Label.new()
	spd_a_lbl.text = "Pre-Brake Alert Gain"
	spd_a_lbl.custom_minimum_size = Vector2(142, 0)
	spd_a_lbl.add_theme_font_size_override("font_size", 11)
	spd_a_row.add_child(spd_a_lbl)
	_sld_speed_alert_gain = HSlider.new()
	_sld_speed_alert_gain.name = "PreBrakeAlertGainSlider"
	_sld_speed_alert_gain.min_value = 0.0
	_sld_speed_alert_gain.max_value = 2.5
	_sld_speed_alert_gain.step = 0.05
	_sld_speed_alert_gain.value = 1.0
	_sld_speed_alert_gain.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_sld_speed_alert_gain.focus_mode = Control.FOCUS_NONE
	_sld_speed_alert_gain.value_changed.connect(_on_speed_alert_gain_changed)
	spd_a_row.add_child(_sld_speed_alert_gain)
	_lbl_speed_alert_gain = Label.new()
	_lbl_speed_alert_gain.add_theme_font_override("font", mono)
	_lbl_speed_alert_gain.add_theme_font_size_override("font_size", 11)
	_lbl_speed_alert_gain.custom_minimum_size = Vector2(64, 0)
	_lbl_speed_alert_gain.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	_lbl_speed_alert_gain.text = "1.00x"
	spd_a_row.add_child(_lbl_speed_alert_gain)

	_btn_export_dil_session = Button.new()
	_btn_export_dil_session.name = "ExportDilSessionBtn"
	_btn_export_dil_session.text = "Export DiL Session (JSON+CSV)"
	_btn_export_dil_session.focus_mode = Control.FOCUS_NONE
	_btn_export_dil_session.pressed.connect(_on_export_dil_session_pressed)
	dil_vb.add_child(_btn_export_dil_session)

	# ------------------------------------------------------------------
	# LIVE DIL & DIAGNOSTIC READOUT PANEL
	# ------------------------------------------------------------------
	var dil_readout_panel := PanelContainer.new()
	var dil_r_sb := StyleBoxFlat.new()
	dil_r_sb.bg_color = Color(0.08, 0.11, 0.15, 0.94)
	dil_r_sb.set_corner_radius_all(4)
	dil_r_sb.set_content_margin_all(6)
	dil_readout_panel.add_theme_stylebox_override("panel", dil_r_sb)
	add_child(dil_readout_panel)

	var dil_r_vb := VBoxContainer.new()
	dil_r_vb.add_theme_constant_override("separation", 3)
	dil_readout_panel.add_child(dil_r_vb)

	var dil_r_hdr := Label.new()
	dil_r_hdr.text = "LIVE DIL & DIAGNOSTIC READOUT"
	dil_r_hdr.add_theme_font_size_override("font_size", 11)
	dil_r_hdr.add_theme_color_override("font_color", Color(0.74, 0.90, 1.0, 1.0))
	dil_r_vb.add_child(dil_r_hdr)

	_dil_diag_lbl = Label.new()
	_dil_diag_lbl.name = "DilLiveDiagLabel"
	_dil_diag_lbl.add_theme_font_override("font", mono)
	_dil_diag_lbl.add_theme_font_size_override("font_size", 11)
	_dil_diag_lbl.text = "Delta t: +0.000 s | Delta vx: +0.0 km/h | e_lat: +0.00 m\nErr wz: +0.000 rad/s | Brake Prog: 0% (Mrg +0.0 m)\nMax Grip Util:  0.0% | Sensor Mode: IDEAL | Balance: NEUTRAL"
	dil_r_vb.add_child(_dil_diag_lbl)

	_dil_sector_lbl = Label.new()
	_dil_sector_lbl.name = "DilSectorDeltaLabel"
	_dil_sector_lbl.add_theme_font_override("font", mono)
	_dil_sector_lbl.add_theme_font_size_override("font_size", 10)
	_dil_sector_lbl.add_theme_color_override("font_color", Color(0.76, 0.90, 0.98, 1.0))
	_dil_sector_lbl.text = "Phase Deltas [s] | Brk: +0.000 | Ent: +0.000 | Apx: +0.000 | Ext: +0.000"
	dil_r_vb.add_child(_dil_sector_lbl)

	_dil_hint_lbl = Label.new()
	_dil_hint_lbl.name = "DilTimeGainHintLabel"
	_dil_hint_lbl.add_theme_font_override("font", mono)
	_dil_hint_lbl.add_theme_font_size_override("font_size", 10)
	_dil_hint_lbl.add_theme_color_override("font_color", Color(0.96, 0.84, 0.45, 1.0))
	_dil_hint_lbl.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_dil_hint_lbl.text = "Time-Gain Hint: On CasADi reference pace."
	dil_r_vb.add_child(_dil_hint_lbl)

	# TV Map Selector
	var map_row := HBoxContainer.new()
	map_row.add_theme_constant_override("separation", 8)
	add_child(map_row)
	var m_lbl := Label.new()
	m_lbl.text = "TV Map"
	m_lbl.custom_minimum_size = Vector2(96, 0)
	map_row.add_child(m_lbl)
	_map_opt = OptionButton.new()
	_map_opt.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_map_opt.focus_mode = Control.FOCUS_NONE
	for i in range(TV_MAP_NAMES.size()):
		_map_opt.add_item(TV_MAP_NAMES[i], i)
	_map_opt.item_selected.connect(_on_map_selected)
	map_row.add_child(_map_opt)

	# Undo / Redo & A/B Comparator row
	var cmp_row := HBoxContainer.new()
	cmp_row.add_theme_constant_override("separation", 4)
	add_child(cmp_row)

	_btn_undo = Button.new()
	_btn_undo.text = "Undo"
	_btn_undo.focus_mode = Control.FOCUS_NONE
	_btn_undo.disabled = true
	_btn_undo.pressed.connect(_on_undo_pressed)
	cmp_row.add_child(_btn_undo)

	_btn_redo = Button.new()
	_btn_redo.text = "Redo"
	_btn_redo.focus_mode = Control.FOCUS_NONE
	_btn_redo.disabled = true
	_btn_redo.pressed.connect(_on_redo_pressed)
	cmp_row.add_child(_btn_redo)

	var btn_a := Button.new()
	btn_a.text = "Store A"
	btn_a.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	btn_a.focus_mode = Control.FOCUS_NONE
	btn_a.pressed.connect(_on_store_a)
	cmp_row.add_child(btn_a)

	var btn_b := Button.new()
	btn_b.text = "Store B"
	btn_b.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	btn_b.focus_mode = Control.FOCUS_NONE
	btn_b.pressed.connect(_on_store_b)
	cmp_row.add_child(btn_b)

	_btn_toggle_ab = Button.new()
	_btn_toggle_ab.text = "A/B [A]"
	_btn_toggle_ab.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_btn_toggle_ab.focus_mode = Control.FOCUS_NONE
	_btn_toggle_ab.pressed.connect(_on_toggle_ab)
	cmp_row.add_child(_btn_toggle_ab)

	# Parameter sections
	var last_sec := ""
	for p in PARAM_DEFS:
		var sec: String = p["section"]
		if sec != last_sec:
			last_sec = sec
			var sep := HSeparator.new()
			add_child(sep)
			var sec_lbl := Label.new()
			sec_lbl.text = sec.to_upper()
			sec_lbl.add_theme_font_size_override("font_size", 11)
			sec_lbl.add_theme_color_override("font_color", Color(0.74, 0.90, 1.0, 1.0))
			add_child(sec_lbl)

		var k: String = p["key"]
		var row := HBoxContainer.new()
		row.add_theme_constant_override("separation", 6)
		add_child(row)

		var lbl := Label.new()
		lbl.text = String(p["label"])
		lbl.custom_minimum_size = Vector2(152, 0)
		lbl.add_theme_font_size_override("font_size", 11)
		row.add_child(lbl)

		var sld := HSlider.new()
		sld.min_value = float(p["min"])
		sld.max_value = float(p["max"])
		sld.step = float(p["step"])
		sld.value = float(p["def"])
		sld.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		sld.focus_mode = Control.FOCUS_NONE
		sld.value_changed.connect(func(v: float): _on_param_slider_changed(k, v))
		row.add_child(sld)
		_sliders[k] = sld

		var v_lbl := Label.new()
		v_lbl.add_theme_font_override("font", mono)
		v_lbl.add_theme_font_size_override("font_size", 11)
		v_lbl.custom_minimum_size = Vector2(72, 0)
		v_lbl.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
		row.add_child(v_lbl)
		_val_labels[k] = v_lbl
		_update_param_label(k)

	# Advanced Preset Manager, Diff Viewer & Export
	add_child(HSeparator.new())
	var pre_hdr := Label.new()
	pre_hdr.text = "PRESET MANAGER & DIFF VIEWER"
	pre_hdr.add_theme_font_size_override("font_size", 11)
	pre_hdr.add_theme_color_override("font_color", Color(0.74, 0.90, 1.0, 1.0))
	add_child(pre_hdr)

	var pre_row1 := HBoxContainer.new()
	pre_row1.add_theme_constant_override("separation", 4)
	add_child(pre_row1)

	_preset_name_edit = LineEdit.new()
	_preset_name_edit.placeholder_text = "Preset name..."
	_preset_name_edit.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_preset_name_edit.focus_mode = Control.FOCUS_CLICK
	_preset_name_edit.text_submitted.connect(func(_t: String): _preset_name_edit.release_focus())
	_preset_name_edit.gui_input.connect(_on_preset_name_gui_input)
	pre_row1.add_child(_preset_name_edit)

	var btn_save := Button.new()
	btn_save.text = "Save Named"
	btn_save.focus_mode = Control.FOCUS_NONE
	btn_save.pressed.connect(_on_save_preset)
	pre_row1.add_child(btn_save)

	var btn_def := Button.new()
	btn_def.text = "Reset Map"
	btn_def.focus_mode = Control.FOCUS_NONE
	btn_def.pressed.connect(_on_reset_default)
	pre_row1.add_child(btn_def)

	var pre_row2 := HBoxContainer.new()
	pre_row2.add_theme_constant_override("separation", 4)
	add_child(pre_row2)

	_preset_opt = OptionButton.new()
	_preset_opt.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_preset_opt.focus_mode = Control.FOCUS_NONE
	_preset_opt.clip_text = true
	pre_row2.add_child(_preset_opt)

	var btn_load := Button.new()
	btn_load.text = "Load"
	btn_load.focus_mode = Control.FOCUS_NONE
	btn_load.pressed.connect(_on_load_preset)
	pre_row2.add_child(btn_load)

	var btn_del := Button.new()
	btn_del.text = "Delete"
	btn_del.focus_mode = Control.FOCUS_NONE
	btn_del.pressed.connect(_on_delete_preset)
	pre_row2.add_child(btn_del)

	var pre_row3 := HBoxContainer.new()
	pre_row3.add_theme_constant_override("separation", 4)
	add_child(pre_row3)

	var btn_exp := Button.new()
	btn_exp.text = "Export STM32 Header"
	btn_exp.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	btn_exp.focus_mode = Control.FOCUS_NONE
	btn_exp.pressed.connect(_on_export_stm32)
	pre_row3.add_child(btn_exp)

	var btn_copy := Button.new()
	btn_copy.text = "Copy Setup Summary"
	btn_copy.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	btn_copy.focus_mode = Control.FOCUS_NONE
	btn_copy.pressed.connect(_on_copy_setup_summary)
	pre_row3.add_child(btn_copy)

	_diff_lbl = Label.new()
	_diff_lbl.add_theme_font_override("font", mono)
	_diff_lbl.add_theme_font_size_override("font_size", 10)
	_diff_lbl.add_theme_color_override("font_color", Color(0.80, 0.86, 0.92, 1.0))
	_diff_lbl.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_diff_lbl.text = "Diff vs Base Map: (none)"
	add_child(_diff_lbl)

	_status_lbl = Label.new()
	_status_lbl.add_theme_font_size_override("font_size", 11)
	_status_lbl.add_theme_color_override("font_color", Color(0.75, 0.82, 0.90, 1.0))
	_status_lbl.text = "Ready | 5-map live TV tuning"
	add_child(_status_lbl)

	# Last Launch stats block
	add_child(HSeparator.new())
	_launch_lbl = Label.new()
	_launch_lbl.add_theme_font_override("font", mono)
	_launch_lbl.add_theme_font_size_override("font_size", 11)
	_launch_lbl.text = _last_launch_summary
	add_child(_launch_lbl)


func _get_car() -> Node:
	var sim_node := get_node_or_null("/root/Sim")
	if sim_node != null and "car" in sim_node and sim_node.car != null:
		return sim_node.car
	var p := get_parent()
	while p != null:
		if p.has_node("Car"):
			return p.get_node("Car")
		p = p.get_parent()
	return null


func _get_track() -> Node:
	var sim_node := get_node_or_null("/root/Sim")
	if sim_node != null and "track" in sim_node and sim_node.track != null:
		return sim_node.track
	var p := get_parent()
	while p != null:
		if p.has_node("Track"):
			return p.get_node("Track")
		p = p.get_parent()
	return null


func _get_dynamics_model() -> Object:
	var car := _get_car()
	if car != null and "model" in car and car.model != null:
		return car.model
	return null


func _get_dil_session() -> Node:
	var trk := _get_track()
	if trk != null and "dil_session" in trk and trk.dil_session != null:
		return trk.dil_session
	return null


func _get_racing_line_vis() -> Node:
	var trk := _get_track()
	if trk != null and "racing_line_vis" in trk and trk.racing_line_vis != null:
		return trk.racing_line_vis
	return null


func _get_param_def(key: String) -> Dictionary:
	for p in PARAM_DEFS:
		if p["key"] == key:
			return p
	return {}


func _clamp_param(key: String, val: float) -> float:
	if key == "power_limit_kw":
		return clampf(val, 10.0, 80.0)
	elif key == "regen_limit_kw":
		return clampf(val, -25.0, 0.0)
	elif key == "friction_ellipse_margin":
		return clampf(val, 0.70, 1.05)
	elif key == "yaw_accel_ff_gain":
		return clampf(val, 0.30, 2.00)
	elif key == "centripetal_fx_gain":
		return clampf(val, 0.0, 50.0)
	elif key == "delta_alpha_eq_gain":
		return clampf(val, 0.0, 4000.0)
	return val


func _update_param_label(key: String) -> void:
	if not _val_labels.has(key):
		return
	var lbl: Label = _val_labels[key]
	var val: float = float(_current.get(key, 0.0))
	var def_val: float = float(_defaults.get(key, val))
	var p := _get_param_def(key)
	var fmt := String(p.get("fmt", "%.2f"))
	var scale := float(p.get("disp_scale", 1.0))
	var mod_mark := "*" if absf(val - def_val) > 1e-4 else ""
	lbl.text = (fmt % (val * scale)) + mod_mark


func _push_undo_snapshot() -> void:
	_undo_stack.append(_current.duplicate(true))
	if _undo_stack.size() > 30:
		_undo_stack.pop_front()
	_redo_stack.clear()
	_update_undo_buttons()


func _update_undo_buttons() -> void:
	if _btn_undo != null:
		_btn_undo.disabled = _undo_stack.is_empty()
	if _btn_redo != null:
		_btn_redo.disabled = _redo_stack.is_empty()


func _apply_params_dict(d: Dictionary, push_undo: bool = false) -> void:
	if push_undo:
		_push_undo_snapshot()
	_syncing = true
	for p in PARAM_DEFS:
		var k: String = p["key"]
		if d.has(k):
			var v: float = _clamp_param(k, float(d[k]))
			_current[k] = v
			if _sliders.has(k):
				(_sliders[k] as HSlider).value = v
			_update_param_label(k)
	_syncing = false
	_push_to_car()
	_update_diff_viewer()
	if _friction_ymd_widget != null:
		_friction_ymd_widget.update_ymd_preview(_current, _defaults)


func _push_to_car() -> void:
	var car := _get_car()
	if car == null:
		return
	var payload := _current.duplicate(true)
	payload["power_limit_kw"] = clampf(float(payload.get("power_limit_kw", 80.0)), 10.0, 80.0)
	payload["regen_limit_kw"] = clampf(float(payload.get("regen_limit_kw", -25.0)), -25.0, 0.0)
	payload["friction_ellipse_margin"] = clampf(float(payload.get("friction_ellipse_margin", 0.94)), 0.70, 1.05)
	payload["yaw_accel_ff_gain"] = clampf(float(payload.get("yaw_accel_ff_gain", 1.00)), 0.30, 2.00)
	payload["centripetal_fx_gain"] = clampf(float(payload.get("centripetal_fx_gain", 15.0)), 0.0, 50.0)
	payload["delta_alpha_eq_gain"] = clampf(float(payload.get("delta_alpha_eq_gain", 2620.0)), 0.0, 4000.0)
	if car.has_method("apply_live_tuning"):
		car.apply_live_tuning(payload)


func sync_from_car(car: Node, snap: Dictionary) -> void:
	if car == null:
		return
	var vid := String(snap.get("vehicle_id", "ter27_4wd"))
	var kin_sup := bool(snap.get("kinematics_3d_supported", false))
	var kin_en := bool(snap.get("kinematics_3d_enabled", false))
	var susp_sup := bool(snap.get("susp_supported", false))
	var susp_vis := bool(snap.get("susp_visible", false))

	_syncing = true
	if _veh_opt_tv != null:
		var v_idx := 1 if vid == "ter26_rwd" else 0
		if _veh_opt_tv.selected != v_idx:
			_veh_opt_tv.select(v_idx)
	if _kin_opt_tv != null:
		_kin_opt_tv.disabled = not kin_sup
		var k_idx := 0 if kin_en else 1
		if _kin_opt_tv.selected != k_idx:
			_kin_opt_tv.select(k_idx)
	if _chk_xray_susp_tv != null:
		_chk_xray_susp_tv.disabled = not susp_sup
		if _chk_xray_susp_tv.button_pressed != susp_vis:
			_chk_xray_susp_tv.button_pressed = susp_vis
	if _map_opt != null:
		_map_opt.disabled = (vid != "ter27_4wd")
	_syncing = false

	var d: Dictionary = {}
	if _diag_lbl != null and car.has_method("get_tv_live_diagnostics"):
		d = car.get_tv_live_diagnostics()
		if not d.is_empty():
			var reg := String(d.get("active_regime", "BALANCED"))
			var launch_st := String(d.get("launch_state", "IDLE"))
			var mz := float(d.get("mz_tv_nm", 0.0))
			var ratio := float(d.get("t_out_in_ratio", 1.0))
			var piv := float(d.get("pivot_blend", 0.0)) * 100.0
			var b_deg := float(d.get("beta_deg", 0.0))
			var b_dot := float(d.get("beta_dot_dps", 0.0))
			var bal := String(d.get("balance_state", "NEUTRAL"))
			var pdc := float(d.get("p_dc_kw", 0.0))
			_diag_lbl.text = "Regime: %-13s | Launch State: %-9s | Mz_TV: %+6.1f Nm\nOut/In: %5.2f | Pivot: %3.0f%% | Beta: %+5.2f / %+5.1f dps\nBalance: %-10s | P_dc: %+5.1f kW" % [
				reg, launch_st, mz, ratio, piv, b_deg, b_dot, bal, pdc
			]
	if _kin_lbl != null:
		var ks: Dictionary = snap.get("kinematics_summary", {}) if snap.get("kinematics_summary") is Dictionary else {}
		var susp_mode_str := "3D-DW" if kin_en else "PARAM"
		_kin_lbl.text = "Susp: %-5s | Cam FL/FR: %+5.2f/%+5.2f | Toe: %+5.2f/%+5.2f\nAck: %+5.1f%% | MR F/R: %.3f/%.3f    | LLTD: %4.1f%%" % [
			susp_mode_str,
			float(ks.get("camber_fl_deg", 0.0)),
			float(ks.get("camber_fr_deg", 0.0)),
			float(ks.get("toe_fl_deg", 0.0)),
			float(ks.get("toe_fr_deg", 0.0)),
			float(ks.get("ackermann_pct", 0.0)),
			float(ks.get("mr_front", 0.0)),
			float(ks.get("mr_rear", 0.0)),
			float(ks.get("lltd_pct", 50.0)),
		]
	if _friction_ymd_widget != null:
		_friction_ymd_widget.update_live_telemetry(car, snap, d)
	_sync_dil_and_sensor_readouts(car, snap, d)
	if _map_opt != null:
		var sim_mode: int = int(snap.get("mode_id", 1))
		var idx := TV_MAP_TO_SIM_MODE.find(sim_mode)
		if idx >= 0 and _map_opt.selected != idx:
			_syncing = true
			_map_opt.select(idx)
			_syncing = false
	if car.has_method("get_default_live_tuning"):
		var defs: Dictionary = car.get_default_live_tuning()
		if not defs.is_empty():
			for k in defs.keys():
				_defaults[k] = float(defs[k])
	if car.has_method("get_live_tuning") and _undo_stack.is_empty():
		var cur: Dictionary = car.get_live_tuning()
		if not cur.is_empty():
			_syncing = true
			for k in cur.keys():
				if _sliders.has(k):
					var v := float(cur[k])
					_current[k] = v
					(_sliders[k] as HSlider).value = v
					_update_param_label(k)
			_syncing = false
			_update_diff_viewer()


func step_launch_tracker(dt: float, snap: Dictionary, car: Node) -> void:
	if car == null or _launch_lbl == null:
		return
	var kmh := float(snap.get("kmh", 0.0))
	var ax_g := float(snap.get("ax_g", 0.0))
	var thr := float(car.gas) if "gas" in car else 0.0

	if not _launch_active:
		if kmh < 1.5:
			_launch_armed = true
		if _launch_armed and thr > 0.70 and kmh > 1.5:
			_launch_armed = false
			_launch_active = true
			_launch_t = 0.0
			_launch_dist_m = 0.0
			_t_0_25 = -1.0
			_t_0_50 = -1.0
			_t_0_100 = -1.0
			_t_75m = -1.0
			_peak_ax_g = maxf(0.0, ax_g)
	else:
		_launch_t += dt
		_launch_dist_m += (kmh / 3.6) * dt
		_peak_ax_g = maxf(_peak_ax_g, ax_g)
		if _t_0_25 < 0.0 and kmh >= 25.0:
			_t_0_25 = _launch_t
		if _t_0_50 < 0.0 and kmh >= 50.0:
			_t_0_50 = _launch_t
		if _t_0_100 < 0.0 and kmh >= 100.0:
			_t_0_100 = _launch_t
		if _t_75m < 0.0 and _launch_dist_m >= 75.0:
			_t_75m = _launch_t

		if _launch_t > 6.0 or (kmh < 2.0 and _launch_t > 0.5) or (_t_0_100 > 0.0 and _t_75m > 0.0):
			_launch_active = false

		_last_launch_summary = "Last Launch: 0-25:%s | 0-50:%s | 0-100:%s\n             75m:%s | Peak ax: %.2fg" % [
			("%.2fs" % _t_0_25) if _t_0_25 > 0.0 else "--",
			("%.2fs" % _t_0_50) if _t_0_50 > 0.0 else "--",
			("%.2fs" % _t_0_100) if _t_0_100 > 0.0 else "--",
			("%.2fs" % _t_75m) if _t_75m > 0.0 else "--",
			_peak_ax_g,
		]
		_launch_lbl.text = _last_launch_summary


func _on_tv_vehicle_selected(index: int) -> void:
	if _syncing or index < 0 or _veh_opt_tv == null:
		return
	var car := _get_car()
	var target_vid := String(_veh_opt_tv.get_item_metadata(index))
	if target_vid.is_empty():
		target_vid = "ter26_rwd" if index == 1 else "ter27_4wd"
	if car != null and car.has_method("switch_vehicle"):
		car.switch_vehicle(target_vid)
		_undo_stack.clear()
		_redo_stack.clear()
		_update_undo_buttons()
		_active_preset_name = "default"
		if car.has_method("get_hud_snapshot"):
			sync_from_car(car, car.get_hud_snapshot())
	if _status_lbl != null:
		_status_lbl.text = "Vehicle: %s" % _veh_opt_tv.get_item_text(index)


func _on_tv_kinematics_selected(index: int) -> void:
	if _syncing or index < 0:
		return
	var car := _get_car()
	var use_3d := (index == 0)
	if car != null and car.has_method("set_kinematics_3d_enabled"):
		car.set_kinematics_3d_enabled(use_3d)
		if car.has_method("get_hud_snapshot"):
			sync_from_car(car, car.get_hud_snapshot())
	if _status_lbl != null:
		_status_lbl.text = "Susp Model: %s" % ("Doble Horquilla 3D" if use_3d else "Parametrico")


func _on_tv_xray_susp_toggled(pressed: bool) -> void:
	if _syncing:
		return
	var car := _get_car()
	if car != null and car.has_method("set_suspension_visualizer_enabled"):
		var actual := bool(car.set_suspension_visualizer_enabled(pressed))
		if _chk_xray_susp_tv != null and _chk_xray_susp_tv.button_pressed != actual:
			_syncing = true
			_chk_xray_susp_tv.button_pressed = actual
			_syncing = false
		if car.has_method("get_hud_snapshot"):
			sync_from_car(car, car.get_hud_snapshot())
	if _status_lbl != null:
		_status_lbl.text = "3D Suspension / X-Ray: %s" % ("ON" if pressed else "OFF")


func _on_apply_event_preset_pressed() -> void:
	if _btn_apply_event_preset != null:
		_btn_apply_event_preset.release_focus()
	var idx := _event_setup_opt.selected if _event_setup_opt != null else 0
	apply_event_setup_preset(idx)


func apply_event_setup_preset(preset_idx: int, target_model: Object = null) -> bool:
	if preset_idx < 0 or preset_idx >= EVENT_SETUP_PRESET_STEMS.size():
		return false
	var stem := EVENT_SETUP_PRESET_STEMS[preset_idx]
	var label := EVENT_SETUP_PRESET_LABELS[preset_idx]
	var applied := false
	var mdl := target_model
	if mdl != null and mdl.has_method("apply_setup"):
		applied = bool(mdl.call("apply_setup", stem))
	else:
		var car := _get_car()
		if car != null and car.has_method("apply_vehicle_setup"):
			applied = bool(car.call("apply_vehicle_setup", stem))
			if applied:
				mdl = _get_dynamics_model()
				if car.has_method("get_hud_snapshot"):
					sync_from_car(car, car.get_hud_snapshot())
		else:
			mdl = _get_dynamics_model()
			if mdl != null and mdl.has_method("apply_setup"):
				applied = bool(mdl.call("apply_setup", stem))
	if applied:
		_active_preset_name = stem
		var summary := "Active Event Setup: %s (%s)" % [label, stem]
		if mdl != null and "config" in mdl and mdl.config != null:
			var cfg = mdl.config
			var rake_mm := (float(cfg.static_ride_height_rear) - float(cfg.static_ride_height_front)) * 1000.0
			summary = "Setup: %s | Cam F/R %+.2f/%+.2f | Toe %+.2f/%+.2f | BB %.1f%% | Rake %+.1f mm" % [
				label,
				float(cfg.static_camber_front_deg),
				float(cfg.static_camber_rear_deg),
				float(cfg.toe_front_deg),
				float(cfg.toe_rear_deg),
				float(cfg.brake_bias_front) * 100.0,
				rake_mm,
			]
		if _event_setup_lbl != null:
			_event_setup_lbl.text = summary
		if _status_lbl != null:
			_status_lbl.text = "Applied Event Setup: %s" % label
	else:
		if _status_lbl != null:
			_status_lbl.text = "Event Setup preset not found: %s.json" % stem
	return applied


func _on_map_selected(index: int) -> void:
	if _syncing or index < 0 or index >= TV_MAP_TO_SIM_MODE.size():
		return
	var car := _get_car()
	if car != null and car.has_method("set_sim_drive_mode"):
		car.set_sim_drive_mode(TV_MAP_TO_SIM_MODE[index])
	if car != null and car.has_method("get_default_live_tuning"):
		var defs: Dictionary = car.get_default_live_tuning()
		for k in defs.keys():
			_defaults[k] = float(defs[k])
	if car != null and car.has_method("get_live_tuning"):
		var cur: Dictionary = car.get_live_tuning()
		if not cur.is_empty():
			_apply_params_dict(cur, false)
	if _status_lbl != null:
		_status_lbl.text = "Map: %s" % TV_MAP_NAMES[index]


func _on_preset_name_gui_input(event: InputEvent) -> void:
	if event is InputEventKey:
		var ke := event as InputEventKey
		if ke.pressed and ke.keycode == KEY_ESCAPE and _preset_name_edit != null:
			_preset_name_edit.release_focus()
			_preset_name_edit.accept_event()


func get_active_preset_name() -> String:
	return _active_preset_name


func _update_diff_viewer() -> void:
	if _diff_lbl == null:
		return
	var use_ab := (not _setup_a.is_empty() and not _setup_b.is_empty())
	var diffs: Array[String] = []
	for p in PARAM_DEFS:
		var k: String = p["key"]
		var fmt: String = String(p["fmt"])
		var scale := float(p.get("disp_scale", 1.0))
		if use_ab:
			var va := float(_setup_a.get(k, p["def"]))
			var vb := float(_setup_b.get(k, p["def"]))
			if absf(va - vb) > 1e-4:
				diffs.append("%s: A=%s B=%s" % [String(p["label"]), fmt % (va * scale), fmt % (vb * scale)])
		else:
			var vc := float(_current.get(k, p["def"]))
			var vd := float(_defaults.get(k, p["def"]))
			if absf(vc - vd) > 1e-4:
				diffs.append("%s: %s (def %s)" % [String(p["label"]), fmt % (vc * scale), fmt % (vd * scale)])
	var hdr := "Diff A vs B:" if use_ab else "Diff vs Base Map:"
	if diffs.is_empty():
		_diff_lbl.text = "%s (none)" % hdr
	else:
		_diff_lbl.text = "%s\n  %s" % [hdr, "\n  ".join(diffs)]


func _on_param_slider_changed(key: String, val: float) -> void:
	if _syncing:
		return
	_push_undo_snapshot()
	val = _clamp_param(key, val)
	_current[key] = val
	_update_param_label(key)
	_push_to_car()
	_update_diff_viewer()
	if _friction_ymd_widget != null:
		_friction_ymd_widget.update_ymd_preview(_current, _defaults)


func _on_undo_pressed() -> void:
	if _undo_stack.is_empty():
		return
	_redo_stack.append(_current.duplicate(true))
	var prev: Dictionary = _undo_stack.pop_back()
	_apply_params_dict(prev, false)
	_update_undo_buttons()


func _on_redo_pressed() -> void:
	if _redo_stack.is_empty():
		return
	_undo_stack.append(_current.duplicate(true))
	var nxt: Dictionary = _redo_stack.pop_back()
	_apply_params_dict(nxt, false)
	_update_undo_buttons()


func _on_store_a() -> void:
	_setup_a = _current.duplicate(true)
	_active_slot = "A"
	if _btn_toggle_ab != null:
		_btn_toggle_ab.text = "A/B [A]"
	if _status_lbl != null:
		_status_lbl.text = "Stored Setup A"
	_update_diff_viewer()


func _on_store_b() -> void:
	_setup_b = _current.duplicate(true)
	_active_slot = "B"
	if _btn_toggle_ab != null:
		_btn_toggle_ab.text = "A/B [B]"
	if _status_lbl != null:
		_status_lbl.text = "Stored Setup B"
	_update_diff_viewer()


func _on_toggle_ab() -> void:
	if _setup_a.is_empty() and _setup_b.is_empty():
		if _status_lbl != null:
			_status_lbl.text = "Store A or B first"
		return
	if _active_slot == "A" and not _setup_b.is_empty():
		_active_slot = "B"
		_apply_params_dict(_setup_b, true)
	elif not _setup_a.is_empty():
		_active_slot = "A"
		_apply_params_dict(_setup_a, true)
	if _btn_toggle_ab != null:
		_btn_toggle_ab.text = "A/B [%s]" % _active_slot


func _on_reset_default() -> void:
	_active_preset_name = "default"
	var car := _get_car()
	if car != null and car.has_method("reset_map_live_tuning"):
		var defs: Dictionary = car.reset_map_live_tuning()
		if not defs.is_empty():
			_apply_params_dict(defs, true)
			if _status_lbl != null:
				_status_lbl.text = "Reset to map defaults"
			return
	_apply_params_dict(_defaults, true)
	if _status_lbl != null:
		_status_lbl.text = "Reset to defaults"


func _refresh_preset_list() -> void:
	if _preset_opt == null:
		return
	_preset_opt.clear()
	var abs_dir := ProjectSettings.globalize_path(PRESETS_DIR)
	DirAccess.make_dir_recursive_absolute(abs_dir)
	var da := DirAccess.open(PRESETS_DIR)
	if da == null:
		_preset_opt.add_item("(no presets)", 0)
		return
	var files: Array[String] = []
	da.list_dir_begin()
	var fn := da.get_next()
	while fn != "":
		if not da.current_is_dir() and fn.to_lower().ends_with(".json"):
			files.append(fn)
		fn = da.get_next()
	da.list_dir_end()
	files.sort()
	if files.is_empty():
		_preset_opt.add_item("(no presets)", 0)
	else:
		for i in range(files.size()):
			_preset_opt.add_item(files[i].get_basename(), i)
			_preset_opt.set_item_metadata(i, "%s/%s" % [PRESETS_DIR, files[i]])


func _sanitize_preset_name(raw: String) -> String:
	var s := raw.strip_edges().replace(" ", "_")
	var out := ""
	for i in range(s.length()):
		var ch := s.unicode_at(i)
		var ok := (ch >= 48 and ch <= 57) or (ch >= 65 and ch <= 90) or (ch >= 97 and ch <= 122) or ch == 95 or ch == 45
		if ok:
			out += char(ch)
	return out


func _on_save_preset() -> void:
	var abs_dir := ProjectSettings.globalize_path(PRESETS_DIR)
	DirAccess.make_dir_recursive_absolute(abs_dir)
	var map_lbl := TV_MAP_NAMES[_map_opt.selected] if (_map_opt != null and _map_opt.selected >= 0) else "custom"
	var custom_name := _sanitize_preset_name(_preset_name_edit.text) if _preset_name_edit != null else ""
	var raw_name := custom_name if not custom_name.is_empty() else ("%s_%d" % [map_lbl.to_lower().replace(" ", "_"), int(Time.get_unix_time_from_system())])
	var path := "%s/%s.json" % [PRESETS_DIR, raw_name]
	var payload := {
		"name": raw_name,
		"tv_map": TV_MAP_NAMES[_map_opt.selected] if (_map_opt != null and _map_opt.selected >= 0) else "Autocross",
		"params": _current,
	}
	var f := FileAccess.open(path, FileAccess.WRITE)
	if f != null:
		f.store_string(JSON.stringify(payload, "\t"))
		f.close()
		_active_preset_name = raw_name
		if _preset_name_edit != null:
			_preset_name_edit.release_focus()
		if _status_lbl != null:
			_status_lbl.text = "Saved: %s.json" % raw_name
		_refresh_preset_list()


func _on_load_preset() -> void:
	if _preset_opt == null or _preset_opt.selected < 0:
		return
	var meta: Variant = _preset_opt.get_item_metadata(_preset_opt.selected)
	var path := String(meta) if meta != null else ""
	if path.is_empty() or not FileAccess.file_exists(path):
		return
	var f := FileAccess.open(path, FileAccess.READ)
	if f == null:
		return
	var parsed: Variant = JSON.parse_string(f.get_as_text())
	f.close()
	if typeof(parsed) == TYPE_DICTIONARY and (parsed as Dictionary).has("params"):
		var p: Dictionary = (parsed as Dictionary)["params"]
		_apply_params_dict(p, true)
		_active_preset_name = path.get_file().get_basename()
		if _preset_name_edit != null:
			_preset_name_edit.text = _active_preset_name
		if _status_lbl != null:
			_status_lbl.text = "Loaded: %s" % path.get_file()


func _on_delete_preset() -> void:
	if _preset_opt == null or _preset_opt.selected < 0:
		return
	var meta: Variant = _preset_opt.get_item_metadata(_preset_opt.selected)
	var path := String(meta) if meta != null else ""
	if path.is_empty() or not FileAccess.file_exists(path):
		return
	var err := DirAccess.remove_absolute(ProjectSettings.globalize_path(path))
	if err == OK:
		if _status_lbl != null:
			_status_lbl.text = "Deleted: %s" % path.get_file()
		_refresh_preset_list()


func _on_export_stm32() -> void:
	var abs_dir := ProjectSettings.globalize_path(PRESETS_DIR)
	DirAccess.make_dir_recursive_absolute(abs_dir)
	var map_lbl := TV_MAP_NAMES[_map_opt.selected] if (_map_opt != null and _map_opt.selected >= 0) else "Autocross"
	var lines := PackedStringArray([
		"#ifndef TV_EXPORTED_PARAMS_H",
		"#define TV_EXPORTED_PARAMS_H",
		"",
		"/* Exported from MirenaSim Trackside Console | Active Map: %s | Preset: %s */" % [map_lbl, _active_preset_name],
	])
	for p in PARAM_DEFS:
		var k: String = p["key"]
		var v: float = _clamp_param(k, float(_current.get(k, p["def"])))
		lines.append("#define TV_PARAM_%s (%.4ff)" % [k.to_upper(), v])
	lines.append("")
	lines.append("#endif /* TV_EXPORTED_PARAMS_H */")

	var f := FileAccess.open(STM32_EXPORT_PATH, FileAccess.WRITE)
	if f != null:
		f.store_string("\n".join(lines) + "\n")
		f.close()
		if _status_lbl != null:
			_status_lbl.text = "Exported: %s" % STM32_EXPORT_PATH


func _on_copy_setup_summary() -> void:
	var map_lbl := TV_MAP_NAMES[_map_opt.selected] if (_map_opt != null and _map_opt.selected >= 0) else "Autocross"
	var lines := PackedStringArray([
		"MirenaSim Setup Summary | Map: %s | Preset: %s" % [map_lbl, _active_preset_name],
	])
	for p in PARAM_DEFS:
		var k: String = p["key"]
		var fmt: String = String(p["fmt"])
		var scale := float(p.get("disp_scale", 1.0))
		var v: float = float(_current.get(k, p["def"]))
		var d: float = float(_defaults.get(k, p["def"]))
		var mark := "*" if absf(v - d) > 1e-4 else ""
		lines.append("  %-27s: %s%s" % [String(p["label"]), fmt % (v * scale), mark])
	var text := "\n".join(lines)
	if DisplayServer.has_feature(DisplayServer.FEATURE_CLIPBOARD):
		DisplayServer.clipboard_set(text)
	if _status_lbl != null:
		_status_lbl.text = "Copied setup summary to clipboard (%d params)" % PARAM_DEFS.size()


# ==============================================================================
# SENSORS, CAN LATENCY, CASADI 3D LINE & LIVE DIL READOUT HANDLERS
# ==============================================================================
func _on_sensor_mode_selected(index: int) -> void:
	if _syncing or index < 0:
		return
	var mdl := _get_dynamics_model()
	if mdl != null and mdl.has_method("set_sensor_realism_mode"):
		mdl.call("set_sensor_realism_mode", index)
	if _status_lbl != null:
		_status_lbl.text = "Sensor Profile: %s" % ("Track Realism" if index == 1 else "Ideal")


func _on_can_latency_changed(val: float) -> void:
	if _lbl_can_latency_ms != null:
		_lbl_can_latency_ms.text = "%.0f ms" % val
	if _syncing:
		return
	var mdl := _get_dynamics_model()
	if mdl != null and mdl.has_method("set_sensor_can_latency_ms"):
		mdl.call("set_sensor_can_latency_ms", val)


func _on_imu_noise_scale_changed(val: float) -> void:
	if _lbl_imu_noise_scale != null:
		_lbl_imu_noise_scale.text = "%.2fx" % val
	if _syncing:
		return
	var mdl := _get_dynamics_model()
	if mdl != null and mdl.has_method("set_sensor_imu_noise_scale"):
		mdl.call("set_sensor_imu_noise_scale", val)


func _on_sensor_seed_changed(val: float) -> void:
	var s_int := int(round(val))
	if _lbl_sensor_seed != null:
		_lbl_sensor_seed.text = "%d" % s_int
	if _syncing:
		return
	var mdl := _get_dynamics_model()
	if mdl != null and mdl.has_method("reset_sensor_realism"):
		mdl.call("reset_sensor_realism", s_int)


func _on_reset_ekf_pressed() -> void:
	var s_int := int(round(_sld_sensor_seed.value)) if _sld_sensor_seed != null else 42
	var mdl := _get_dynamics_model()
	if mdl != null and mdl.has_method("reset_sensor_realism"):
		mdl.call("reset_sensor_realism", s_int)
	if _status_lbl != null:
		_status_lbl.text = "Reset EKF / Sensor filters (seed=%d)" % s_int


func _on_casadi_ref_selected(index: int) -> void:
	if _syncing or index < 0:
		return
	var enabled := (index == 1)
	var sess := _get_dil_session()
	if sess != null and sess.has_method("set_reference_enabled"):
		sess.call("set_reference_enabled", enabled)
	if _status_lbl != null:
		_status_lbl.text = "CasADi Reference: %s" % ("Live Delta" if enabled else "Off")


func _on_line_3d_mode_selected(index: int) -> void:
	if _syncing or index < 0:
		return
	var rlv := _get_racing_line_vis()
	if rlv != null and rlv.has_method("set_display_mode"):
		rlv.call("set_display_mode", index)
	elif rlv != null:
		rlv.visible = (index > 0)
	var mode_names := ["Off", "Standard", "Sim-Racing 3D"]
	if _status_lbl != null and index < mode_names.size():
		_status_lbl.text = "3D Line Mode: %s" % mode_names[index]


func _on_brake_3d_height_changed(val: float) -> void:
	if _lbl_brake_3d_height != null:
		_lbl_brake_3d_height.text = "%.2f m" % val
	if _syncing:
		return
	var rlv := _get_racing_line_vis()
	if rlv != null and rlv.has_method("set_brake_elevation_scale"):
		rlv.call("set_brake_elevation_scale", val)


func _on_speed_alert_gain_changed(val: float) -> void:
	if _lbl_speed_alert_gain != null:
		_lbl_speed_alert_gain.text = "%.2fx" % val
	if _syncing:
		return
	var rlv := _get_racing_line_vis()
	if rlv != null and rlv.has_method("set_speed_alert_gain"):
		rlv.call("set_speed_alert_gain", val)


func _on_export_dil_session_pressed() -> void:
	var sess := _get_dil_session()
	if sess != null and sess.has_method("export_dil_session"):
		var res: Variant = sess.call("export_dil_session", _active_preset_name)
		if res is Dictionary and bool((res as Dictionary).get("ok", false)):
			var jp := String((res as Dictionary).get("json_path", ""))
			if _status_lbl != null:
				_status_lbl.text = "Exported DiL session: %s" % jp.get_file()
			return
	if _status_lbl != null:
		_status_lbl.text = "Export DiL session: no active session"


func _compute_time_gain_hint(phase_deltas: Dictionary, delta_vx_kmh: float, e_lat_m: float, brake_margin_m: float, balance_state: String) -> String:
	var worst_phase := ""
	var worst_dt := 0.02
	for ph_key in ["braking", "entry", "apex", "exit"]:
		var dt_val := float(phase_deltas.get(ph_key, 0.0))
		if dt_val > worst_dt:
			worst_dt = dt_val
			worst_phase = ph_key
	if worst_phase == "braking":
		if brake_margin_m > 2.0:
			return "Time-Gain Hint: Losing +%.3fs in BRAKING (braking %.1fm early; delay brake point)." % [worst_dt, brake_margin_m]
		return "Time-Gain Hint: Losing +%.3fs in BRAKING (modulate trail-brake & front grip)." % worst_dt
	elif worst_phase == "entry":
		return "Time-Gain Hint: Losing +%.3fs in ENTRY (e_lat=%+.2fm, %s; sharpen turn-in)." % [worst_dt, e_lat_m, balance_state]
	elif worst_phase == "apex":
		return "Time-Gain Hint: Losing +%.3fs at APEX (delta_vx=%+.1f km/h; carry more apex speed)." % [worst_dt, delta_vx_kmh]
	elif worst_phase == "exit":
		return "Time-Gain Hint: Losing +%.3fs on EXIT (earlier throttle / optimize rear traction)." % worst_dt
	if absf(e_lat_m) > 0.65:
		return "Time-Gain Hint: Off optimal line (e_lat=%+.2fm); tighten line to CasADi trace." % e_lat_m
	if delta_vx_kmh < -4.0:
		return "Time-Gain Hint: %.1f km/h under CasADi ref speed; push earlier throttle." % absf(delta_vx_kmh)
	return "Time-Gain Hint: Tracking CasADi reference pace cleanly."


func _sync_dil_and_sensor_readouts(car: Node, _snap: Dictionary, tv_diag: Dictionary) -> void:
	var sensor_mode_str := "IDEAL"
	var mdl := _get_dynamics_model()
	if mdl == null and car != null and "model" in car:
		mdl = car.model
	if mdl != null and mdl.has_method("get_sensor_realism_diagnostics"):
		var sd: Variant = mdl.call("get_sensor_realism_diagnostics")
		if sd is Dictionary:
			var sdict := sd as Dictionary
			var mode_int := int(sdict.get("mode", 0))
			sensor_mode_str = "TRACK_REAL" if mode_int == 1 else "IDEAL"
			if _sensor_mode_opt != null and _sensor_mode_opt.selected != mode_int:
				_syncing = true
				_sensor_mode_opt.select(mode_int)
				_syncing = false

	var dil_t: Dictionary = {}
	var sess := _get_dil_session()
	if sess != null and sess.has_method("get_live_dil_telemetry"):
		var raw_t: Variant = sess.call("get_live_dil_telemetry")
		if raw_t is Dictionary:
			dil_t = raw_t as Dictionary

	var delta_t := float(dil_t.get("delta_t_s", 0.0))
	var delta_vx_kmh := float(dil_t.get("delta_vx_ms", 0.0)) * 3.6
	var e_lat := float(dil_t.get("e_lat_m", 0.0))
	var err_wz := float(dil_t.get("delta_wz_rads", 0.0))
	var brake_prog_pct := float(dil_t.get("brake_prog", 0.0)) * 100.0
	var brake_margin_m := float(dil_t.get("brake_margin_m", 0.0))
	var max_grip_pct := float(dil_t.get("max_grip_util", 0.0)) * 100.0
	var bal := String(tv_diag.get("balance_state", dil_t.get("balance_state", "NEUTRAL")))

	if _dil_diag_lbl != null:
		_dil_diag_lbl.text = "Delta t: %+6.3f s | Delta vx: %+5.1f km/h | e_lat: %+5.2f m\nErr wz: %+6.3f rad/s | Brake Prog: %3.0f%% (Mrg %+5.1f m)\nMax Grip Util: %5.1f%% | Sensor Mode: %-10s | Bal: %s" % [
			delta_t, delta_vx_kmh, e_lat, err_wz, brake_prog_pct, brake_margin_m, max_grip_pct, sensor_mode_str, bal
		]

	var phase_deltas: Dictionary = dil_t.get("phase_deltas_s", {}) if dil_t.get("phase_deltas_s") is Dictionary else {}
	var dt_brk := float(phase_deltas.get("braking", dil_t.get("delta_t_braking_s", 0.0)))
	var dt_ent := float(phase_deltas.get("entry", dil_t.get("delta_t_entry_s", 0.0)))
	var dt_apx := float(phase_deltas.get("apex", dil_t.get("delta_t_apex_s", 0.0)))
	var dt_ext := float(phase_deltas.get("exit", dil_t.get("delta_t_exit_s", 0.0)))
	if _dil_sector_lbl != null:
		_dil_sector_lbl.text = "Phase Deltas [s] | Brk: %+6.3f | Ent: %+6.3f | Apx: %+6.3f | Ext: %+6.3f" % [
			dt_brk, dt_ent, dt_apx, dt_ext
		]
	if _dil_hint_lbl != null:
		var pd_norm := {"braking": dt_brk, "entry": dt_ent, "apex": dt_apx, "exit": dt_ext}
		_dil_hint_lbl.text = _compute_time_gain_hint(pd_norm, delta_vx_kmh, e_lat, brake_margin_m, bal)
