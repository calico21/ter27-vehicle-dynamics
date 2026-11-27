extends PanelContainer

const MODE_LABELS: Array[String] = [
	"OPEN 4WD",
	"AUTOCROSS",
	"ENDURANCE",
	"SKIDPAD",
	"ACCELERATION",
	"TIRE WARMER",
]

const VEHICLE_IDS: Array[String] = ["ter27_4wd", "ter26_rwd"]
const VEHICLE_LABELS: Array[String] = ["TeR27 4WD", "TeR26 Validado (RWD)"]
const MODEL_IDS: Array[String] = ["full_14dof", "fast_3dof"]
const MODEL_LABELS: Array[String] = ["Full 14-DOF", "Fast 3-DOF"]
const INPUT_PROFILE_LABELS: Array[String] = ["Auto", "Gamepad", "Wheel (Direct)", "Keyboard"]
const GHOST_MODE_LABELS: Array[String] = ["Off", "Best Session Lap", "LTS Optimal"]
const GHOST_SOURCE_KEYS: Array[String] = ["OFF", "BEST_SESSION_LAP", "LTS_OPTIMAL"]
const CSV_RATE_LABELS: Array[String] = ["50 Hz", "100 Hz", "200 Hz"]
const CSV_RATE_VALUES: Array[int] = [50, 100, 200]
const PACE_RABBIT_LABELS: Array[String] = ["+0.0 s", "+0.5 s", "+1.0 s"]
const PACE_RABBIT_OFFSETS: Array[float] = [0.0, 0.5, 1.0]
const TIME_SCALE_LABELS: Array[String] = ["1.0x Normal", "0.5x Half", "0.25x Analysis"]
const TIME_SCALE_VALUES: Array[float] = [1.0, 0.5, 0.25]
const TIRE_STATE_LABELS: Array[String] = ["Cold 25C", "Warm 55C", "Optimal 70C"]
const TIRE_STATE_KEYS: Array[String] = ["COLD_25C", "WARM_55C", "OPT_70C"]
const AUDIO_PROFILE_LABELS: Array[String] = ["FS Electric", "F1 V10", "F1 V12"]
const ROAD_SURFACE_LABELS: Array[String] = [
	"Flat (Baseline)",
	"ISO 8608 Class A",
	"ISO 8608 Class B",
	"ISO 8608 Class C",
	"ISO B + Curbs",
]
const BUMP_MODE_LABELS: Array[String] = ["Off", "Speed Bumps", "Asym Potholes", "Apex Curbs"]
const LINE_COLOR_MODE_LABELS: Array[String] = [
	"Velocidad",
	"Fases de Conduccion",
	"Delta-v en tiempo real (Verde/Rojo)",
]

const LtsMenuControllerScript = preload("res://Scenes/Sim/UI/lts_menu_controller.gd")
const TracksideTuningControllerScript = preload("res://Scenes/Sim/UI/TracksideTuningController.gd")
const RewindBufferScript = preload("res://Scenes/Sim/UI/RewindBuffer.gd")
const TelemetryCsvLoggerScript = preload("res://Scenes/Sim/UI/TelemetryCsvLogger.gd")

@onready var _tabs: TabContainer = $MarginContainer/VBoxContainer/MenuTabs

# Tab 0: Vehicle
var _veh_opt: OptionButton = null
var _model_opt: OptionButton = null
var _kin_opt: OptionButton = null
var _setup_opt: OptionButton = null
var _mode_opt: OptionButton = null
var _last_setup_vehicle_id: String = ""

# Tab 1: Setup / TV
var _tv_tuning_ctrl: TracksideTuningControllerScript = null

# Tab 2: Track additions
var _time_scale_opt: OptionButton = null
var _pin_status_lbl: Label = null
var _skidpad_stats_lbl: Label = null
var _sld_mu_scale: HSlider = null
var _lbl_mu_scale: Label = null
var _tire_state_opt: OptionButton = null
var _sld_aero_scale: HSlider = null
var _lbl_aero_scale: Label = null
var _chk_split_mu: CheckButton = null
var _grip_profile_opt: OptionButton = null
var _road_surf_opt: OptionButton = null
var _sld_rough: HSlider = null
var _lbl_rough: Label = null
var _bump_mode_opt: OptionButton = null
var _sld_bump_h: HSlider = null
var _lbl_bump_h: Label = null
var _sld_bump_sp: HSlider = null
var _lbl_bump_sp: Label = null
var _session_lbl: Label = null
var _lap_history_lbl: Label = null
var _csv_rate_opt: OptionButton = null
var _btn_csv_rec: Button = null
var _chk_auto_csv: CheckButton = null
var _csv_status_lbl: Label = null

# Tab 3: View
var _cam_opt: OptionButton = null
var _sld_cam_fov: HSlider = null
var _lbl_cam_fov: Label = null
var _chk_susp_vis: CheckButton = null
var _chk_xray: CheckButton = null
var _chk_top: CheckButton = null
var _chk_arrows: CheckButton = null
var _chk_fz_arrows: CheckButton = null
var _chk_torque_bars: CheckButton = null
var _chk_ellipses: CheckButton = null
var _chk_skidpad_ring: CheckButton = null
var _chk_racing_line: CheckButton = null
var _line_color_opt: OptionButton = null
var _chk_line_markers: CheckButton = null
var _ghost_opt: OptionButton = null
var _chk_ghost_vis: CheckButton = null
var _sld_ghost_opacity: HSlider = null
var _lbl_ghost_opacity: Label = null
var _pace_rabbit_opt: OptionButton = null
var _susp_table: Label = null

# Tab 4: Audio
var _chk_dil_audio: CheckButton = null
var _audio_profile_opt: OptionButton = null
var _sld_master_db: HSlider = null
var _lbl_master_db: Label = null
var _sld_engine_db: HSlider = null
var _lbl_engine_db: Label = null
var _sld_tire_db: HSlider = null
var _lbl_tire_db: Label = null

# Tab 5: Controls
var _input_profile_opt: OptionButton = null
var _chk_haptics: CheckButton = null
var _ctrl_sliders: Dictionary = {}
var _ctrl_labels: Dictionary = {}
var _ctrl_steer_raw_bar: ColorRect = null
var _ctrl_steer_out_bar: ColorRect = null
var _ctrl_thr_raw_bar: ColorRect = null
var _ctrl_thr_out_bar: ColorRect = null
var _ctrl_brk_raw_bar: ColorRect = null
var _ctrl_brk_out_bar: ColorRect = null
var _ctrl_live_lbl: Label = null

# Tab 6: Telemetry
var _tel_kin_lbl: Label = null
var _tel_pwr_neg: ColorRect = null
var _tel_pwr_pos: ColorRect = null
var _tel_pwr_val: Label = null
var _tel_brk_reg: ColorRect = null
var _tel_brk_hyd: ColorRect = null
var _tel_brk_mark: ColorRect = null
var _tel_brk_title: Label = null
var _tel_brk_val: Label = null
var _tel_steer_lbl: Label = null
var _tel_tire_bars: Array[ColorRect] = []
var _tel_tire_lbls: Array[Label] = []
var _tel_ecu_lbl: Label = null
var _btn_rec: Button = null
var _btn_tune: Button = null
var _btn_scope: Button = null

# Top DiL timing strip
var _dil_timing_lbl: Label = null

# Tab 8: LTS controller
var _lts_ctrl: Node = null

# QoL runtime helpers
var _rewind_buf: RewindBufferScript = RewindBufferScript.new()
var _csv_logger: TelemetryCsvLoggerScript = TelemetryCsvLoggerScript.new()
var _has_practice_pin: bool = false
var _practice_pin_xf: Transform3D = Transform3D.IDENTITY
var _prev_lap_active: bool = false

var _syncing_ui: bool = false
var _last_cam_count: int = -1


func _ready() -> void:
	if _tabs == null:
		return

	var as_tab := _tabs.get_node_or_null("VEHICLE")
	var trk_tab := _tabs.get_node_or_null("TRACK")

	_build_vehicle_tab()
	_build_setup_tv_tab()
	_augment_track_tab(trk_tab)
	_build_view_tab()
	_build_audio_tab()
	_build_controls_tab()
	_build_telemetry_tab()
	_build_lts_tab()

	var veh_tab := _tabs.get_node_or_null("Vehicle")
	var setup_tab := _tabs.get_node_or_null("SetupTV")
	var view_tab := _tabs.get_node_or_null("View")
	var audio_tab := _tabs.get_node_or_null("Audio")
	var ctrl_tab := _tabs.get_node_or_null("Controls")
	var tel_tab := _tabs.get_node_or_null("Telemetry")
	var lts_tab := _tabs.get_node_or_null("LTS")

	var ordered: Array[Dictionary] = [
		{"node": veh_tab, "title": "Vehicle"},
		{"node": setup_tab, "title": "Setup / TV"},
		{"node": trk_tab, "title": "Track"},
		{"node": view_tab, "title": "View"},
		{"node": audio_tab, "title": "Audio"},
		{"node": ctrl_tab, "title": "Controls"},
		{"node": tel_tab, "title": "Telemetry"},
		{"node": as_tab, "title": "AS / Sys"},
		{"node": lts_tab, "title": "LTS"},
	]
	for idx in range(ordered.size()):
		var n: Node = ordered[idx]["node"]
		if n != null:
			_tabs.move_child(n, idx)
			_tabs.set_tab_title(idx, String(ordered[idx]["title"]))

	if get_parent() != null:
		_dil_timing_lbl = get_parent().get_node_or_null("DilTimingBar/DilTimingLabel") as Label

	_tabs.current_tab = 0
	if _tabs.has_method("get_tab_bar") and _tabs.get_tab_bar() != null:
		_tabs.get_tab_bar().focus_mode = Control.FOCUS_NONE
	_clear_focus_recursive(self)
	_tabs.tab_changed.connect(func(_idx: int): _release_any_focus())


func _clear_focus_recursive(node: Node) -> void:
	if node is Control:
		var ctrl := node as Control
		if ctrl is BaseButton or ctrl is Range or ctrl is TabContainer or ctrl is TabBar or ctrl is ScrollContainer:
			ctrl.focus_mode = Control.FOCUS_NONE
	for ch in node.get_children():
		_clear_focus_recursive(ch)


func _release_any_focus() -> void:
	var vp := get_viewport()
	if vp != null:
		var fo := vp.gui_get_focus_owner()
		if fo != null:
			fo.release_focus()


func _input(event: InputEvent) -> void:
	if not (event is InputEventKey):
		return
	var ke := event as InputEventKey
	if not ke.pressed or ke.echo:
		return
	var vp := get_viewport()
	if vp != null and vp.gui_get_focus_owner() is LineEdit:
		return
	var car := _get_car()
	if car != null and car.has_method("_is_overlay_typing") and car._is_overlay_typing():
		return
	match ke.keycode:
		KEY_R:
			_on_rewind_pressed()
		KEY_G:
			_on_drop_pin_pressed()
		KEY_B:
			_on_tp_pin_pressed()
		KEY_F5:
			_on_reset_lap_pressed()
		KEY_F6:
			_on_reset_cones_pressed()
		KEY_F7:
			_cycle_time_scale()


func _make_tab_margin(tab_name: String, scrollable: bool = false) -> VBoxContainer:
	var m := MarginContainer.new()
	m.name = tab_name
	m.add_theme_constant_override("margin_left", 10)
	m.add_theme_constant_override("margin_top", 8)
	m.add_theme_constant_override("margin_right", 10)
	m.add_theme_constant_override("margin_bottom", 8)
	_tabs.add_child(m)

	var vb := VBoxContainer.new()
	vb.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	vb.add_theme_constant_override("separation", 6)
	if scrollable:
		var sc := ScrollContainer.new()
		sc.focus_mode = Control.FOCUS_NONE
		sc.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
		sc.size_flags_vertical = Control.SIZE_EXPAND_FILL
		sc.add_child(vb)
		m.add_child(sc)
	else:
		m.add_child(vb)
	return vb


func _make_row_opt(parent: VBoxContainer, label_text: String, items: Array[String], cb: Callable) -> OptionButton:
	var row := HBoxContainer.new()
	row.add_theme_constant_override("separation", 8)
	parent.add_child(row)
	var lbl := Label.new()
	lbl.text = label_text
	lbl.custom_minimum_size = Vector2(104, 0)
	row.add_child(lbl)
	var opt := OptionButton.new()
	opt.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	opt.focus_mode = Control.FOCUS_NONE
	for i in range(items.size()):
		opt.add_item(items[i], i)
	opt.item_selected.connect(cb)
	row.add_child(opt)
	return opt


func _make_row_slider(parent: VBoxContainer, label_text: String, min_v: float, max_v: float, step_v: float, def_v: float, cb: Callable) -> Array:
	var row := HBoxContainer.new()
	row.add_theme_constant_override("separation", 6)
	parent.add_child(row)
	var lbl := Label.new()
	lbl.text = label_text
	lbl.custom_minimum_size = Vector2(118, 0)
	lbl.add_theme_font_size_override("font_size", 11)
	row.add_child(lbl)
	var sld := HSlider.new()
	sld.min_value = min_v
	sld.max_value = max_v
	sld.step = step_v
	sld.value = def_v
	sld.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	sld.focus_mode = Control.FOCUS_NONE
	sld.value_changed.connect(cb)
	row.add_child(sld)
	var v_lbl := Label.new()
	v_lbl.add_theme_font_override("font", _make_mono_font())
	v_lbl.add_theme_font_size_override("font_size", 11)
	v_lbl.custom_minimum_size = Vector2(56, 0)
	v_lbl.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	row.add_child(v_lbl)
	return [sld, v_lbl]


func _make_mono_font() -> SystemFont:
	var f := SystemFont.new()
	f.font_names = PackedStringArray(["DejaVu Sans Mono", "Liberation Mono", "Monospace"])
	return f


func _build_vehicle_tab() -> void:
	var vb := _make_tab_margin("Vehicle")
	_veh_opt = _make_row_opt(vb, "Vehicle", VEHICLE_LABELS, _on_vehicle_selected)
	_model_opt = _make_row_opt(vb, "Model", MODEL_LABELS, _on_model_selected)
	_kin_opt = _make_row_opt(vb, "Kinematics", ["Doble Horquilla 3D", "Parametrico"], _on_kin_selected)
	_setup_opt = _make_row_opt(vb, "Setup Profile", ["default"], _on_setup_selected)
	_mode_opt = _make_row_opt(vb, "Drive Mode [M]", MODE_LABELS, _on_mode_selected)


func _build_setup_tv_tab() -> void:
	var vb := _make_tab_margin("SetupTV", true)
	_tv_tuning_ctrl = TracksideTuningControllerScript.new()
	_tv_tuning_ctrl.name = "TracksideTuningController"
	_tv_tuning_ctrl.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	vb.add_child(_tv_tuning_ctrl)


func _build_view_tab() -> void:
	var vb := _make_tab_margin("View", true)
	var mono := _make_mono_font()

	var cam_row := HBoxContainer.new()
	cam_row.add_theme_constant_override("separation", 8)
	vb.add_child(cam_row)
	var cam_lbl := Label.new()
	cam_lbl.text = "Camera [C/F3]"
	cam_lbl.custom_minimum_size = Vector2(104, 0)
	cam_row.add_child(cam_lbl)
	_cam_opt = OptionButton.new()
	_cam_opt.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_cam_opt.focus_mode = Control.FOCUS_NONE
	_cam_opt.item_selected.connect(_on_cam_selected)
	cam_row.add_child(_cam_opt)

	var fov_pair := _make_row_slider(vb, "Camera FOV", 55.0, 95.0, 1.0, 75.0, _on_cam_fov_changed)
	_sld_cam_fov = fov_pair[0] as HSlider
	_lbl_cam_fov = fov_pair[1] as Label
	_lbl_cam_fov.text = "75 deg"

	var btn_focus := Button.new()
	btn_focus.text = "Focus FR Wheel"
	btn_focus.focus_mode = Control.FOCUS_NONE
	btn_focus.pressed.connect(_on_focus_susp_cam)
	vb.add_child(btn_focus)

	_chk_susp_vis = CheckButton.new()
	_chk_susp_vis.text = "Show 3D Suspension [V]"
	_chk_susp_vis.focus_mode = Control.FOCUS_NONE
	_chk_susp_vis.toggled.connect(_on_susp_toggled)
	vb.add_child(_chk_susp_vis)

	_chk_xray = CheckButton.new()
	_chk_xray.text = "X-Ray Body"
	_chk_xray.focus_mode = Control.FOCUS_NONE
	_chk_xray.toggled.connect(_on_xray_only_toggled)
	vb.add_child(_chk_xray)

	_chk_top = CheckButton.new()
	_chk_top.text = "Always On Top"
	_chk_top.focus_mode = Control.FOCUS_NONE
	_chk_top.toggled.connect(_on_always_on_top_toggled)
	vb.add_child(_chk_top)

	_chk_arrows = CheckButton.new()
	_chk_arrows.text = "Tire Force Arrows [F8]"
	_chk_arrows.focus_mode = Control.FOCUS_NONE
	_chk_arrows.toggled.connect(_on_arrows_toggled)
	vb.add_child(_chk_arrows)

	_chk_fz_arrows = CheckButton.new()
	_chk_fz_arrows.text = "Show Fz Arrows"
	_chk_fz_arrows.focus_mode = Control.FOCUS_NONE
	_chk_fz_arrows.toggled.connect(_on_fz_arrows_toggled)
	vb.add_child(_chk_fz_arrows)

	_chk_torque_bars = CheckButton.new()
	_chk_torque_bars.text = "Show Torque/Brake Bars"
	_chk_torque_bars.focus_mode = Control.FOCUS_NONE
	_chk_torque_bars.toggled.connect(_on_torque_bars_toggled)
	vb.add_child(_chk_torque_bars)

	_chk_ellipses = CheckButton.new()
	_chk_ellipses.text = "Friction Ellipse Ground Rings"
	_chk_ellipses.focus_mode = Control.FOCUS_NONE
	_chk_ellipses.toggled.connect(_on_ellipses_toggled)
	vb.add_child(_chk_ellipses)

	_chk_skidpad_ring = CheckButton.new()
	_chk_skidpad_ring.text = "Skidpad 9.125m Ring"
	_chk_skidpad_ring.focus_mode = Control.FOCUS_NONE
	_chk_skidpad_ring.toggled.connect(_on_skidpad_ring_toggled)
	vb.add_child(_chk_skidpad_ring)

	_chk_racing_line = CheckButton.new()
	_chk_racing_line.text = "Racing Line 3D"
	_chk_racing_line.focus_mode = Control.FOCUS_NONE
	_chk_racing_line.toggled.connect(_on_racing_line_toggled)
	vb.add_child(_chk_racing_line)

	_line_color_opt = _make_row_opt(vb, "Line Color Mode", LINE_COLOR_MODE_LABELS, _on_line_color_mode_selected)
	_line_color_opt.select(1) # Default: Fases de Conduccion

	_chk_line_markers = CheckButton.new()
	_chk_line_markers.text = "Track Braking/Apex Markers"
	_chk_line_markers.focus_mode = Control.FOCUS_NONE
	_chk_line_markers.button_pressed = true
	_chk_line_markers.toggled.connect(_on_line_markers_toggled)
	vb.add_child(_chk_line_markers)

	vb.add_child(HSeparator.new())
	_ghost_opt = _make_row_opt(vb, "Ghost Source", GHOST_MODE_LABELS, _on_ghost_mode_selected)

	_chk_ghost_vis = CheckButton.new()
	_chk_ghost_vis.text = "Show Ghost Car"
	_chk_ghost_vis.button_pressed = true
	_chk_ghost_vis.focus_mode = Control.FOCUS_NONE
	_chk_ghost_vis.toggled.connect(_on_ghost_vis_toggled)
	vb.add_child(_chk_ghost_vis)

	var gop_pair := _make_row_slider(vb, "Ghost Opacity", 0.05, 1.00, 0.05, 0.34, _on_ghost_opacity_changed)
	_sld_ghost_opacity = gop_pair[0] as HSlider
	_lbl_ghost_opacity = gop_pair[1] as Label
	_lbl_ghost_opacity.text = "34%"

	_pace_rabbit_opt = _make_row_opt(vb, "Pace Rabbit", PACE_RABBIT_LABELS, _on_pace_rabbit_selected)

	vb.add_child(HSeparator.new())
	_susp_table = Label.new()
	_susp_table.add_theme_font_override("font", mono)
	_susp_table.add_theme_font_size_override("font_size", 11)
	_susp_table.text = "Corner | dz[mm] | Cam[deg] | Fz[N] | Farb[N]\nFL     |   +0.0 |   +0.00  |     0 |    +0.0"
	vb.add_child(_susp_table)


func _build_audio_tab() -> void:
	var vb := _make_tab_margin("Audio", true)

	_chk_dil_audio = CheckButton.new()
	_chk_dil_audio.text = "DiL Procedural Audio"
	_chk_dil_audio.focus_mode = Control.FOCUS_NONE
	_chk_dil_audio.toggled.connect(_on_dil_audio_toggled)
	vb.add_child(_chk_dil_audio)

	_audio_profile_opt = _make_row_opt(vb, "Audio Profile", AUDIO_PROFILE_LABELS, _on_audio_profile_selected)

	var m_pair := _make_row_slider(vb, "Master Gain", -30.0, 6.0, 0.5, 0.0, _on_audio_gains_changed)
	_sld_master_db = m_pair[0] as HSlider
	_lbl_master_db = m_pair[1] as Label
	_lbl_master_db.text = "+0.0 dB"

	var e_pair := _make_row_slider(vb, "Engine / Motor", -30.0, 6.0, 0.5, 0.0, _on_audio_gains_changed)
	_sld_engine_db = e_pair[0] as HSlider
	_lbl_engine_db = e_pair[1] as Label
	_lbl_engine_db.text = "+0.0 dB"

	var t_pair := _make_row_slider(vb, "Tire Scrub", -30.0, 6.0, 0.5, 0.0, _on_audio_gains_changed)
	_sld_tire_db = t_pair[0] as HSlider
	_lbl_tire_db = t_pair[1] as Label
	_lbl_tire_db.text = "+0.0 dB"

	var btn_rev := Button.new()
	btn_rev.text = "Test Rev (1.2s Sweep)"
	btn_rev.focus_mode = Control.FOCUS_NONE
	btn_rev.pressed.connect(_on_test_rev_pressed)
	vb.add_child(btn_rev)


func _make_dual_bar_row(parent: VBoxContainer, label_text: String, bar_color_raw: Color, bar_color_out: Color) -> Array:
	var row := HBoxContainer.new()
	row.add_theme_constant_override("separation", 6)
	parent.add_child(row)
	var lbl := Label.new()
	lbl.text = label_text
	lbl.custom_minimum_size = Vector2(72, 0)
	lbl.add_theme_font_size_override("font_size", 11)
	row.add_child(lbl)

	var track := Control.new()
	track.custom_minimum_size = Vector2(260, 14)
	track.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	var bg := ColorRect.new()
	bg.color = Color(0.12, 0.14, 0.18, 1.0)
	bg.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	track.add_child(bg)

	var raw_bar := ColorRect.new()
	raw_bar.color = bar_color_raw
	raw_bar.position = Vector2(0, 1)
	raw_bar.size = Vector2(0, 5)
	track.add_child(raw_bar)

	var out_bar := ColorRect.new()
	out_bar.color = bar_color_out
	out_bar.position = Vector2(0, 8)
	out_bar.size = Vector2(0, 5)
	track.add_child(out_bar)

	row.add_child(track)
	return [raw_bar, out_bar]


func _build_controls_tab() -> void:
	var vb := _make_tab_margin("Controls", true)
	var mono := _make_mono_font()

	_input_profile_opt = _make_row_opt(vb, "Input Profile", INPUT_PROFILE_LABELS, _on_input_profile_selected)

	var pre_row := HBoxContainer.new()
	pre_row.add_theme_constant_override("separation", 4)
	vb.add_child(pre_row)
	for pair in [["Keyboard Sharp", "KEYBOARD_SHARP"], ["Gamepad Precision", "GAMEPAD_PRECISION"], ["Wheel Direct 1:1", "WHEEL_DIRECT"]]:
		var btn := Button.new()
		btn.text = pair[0]
		btn.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		btn.focus_mode = Control.FOCUS_NONE
		var pid: String = pair[1]
		btn.pressed.connect(func(): _on_input_preset_pressed(pid))
		pre_row.add_child(btn)

	_chk_haptics = CheckButton.new()
	_chk_haptics.name = "HapticsFFB"
	_chk_haptics.text = "Haptics / FFB Rumble"
	_chk_haptics.button_pressed = true
	_chk_haptics.focus_mode = Control.FOCUS_NONE
	_chk_haptics.toggled.connect(_on_haptics_toggled)
	vb.add_child(_chk_haptics)

	vb.add_child(HSeparator.new())
	var mon_hdr := Label.new()
	mon_hdr.text = "LIVE INPUT MONITOR (TOP: RAW | BOTTOM: CONDITIONED)"
	mon_hdr.add_theme_font_size_override("font_size", 11)
	mon_hdr.add_theme_color_override("font_color", Color(0.74, 0.90, 1.0, 1.0))
	vb.add_child(mon_hdr)

	var st_bars := _make_dual_bar_row(vb, "Steer", Color(0.45, 0.65, 0.90, 0.85), Color(0.20, 0.85, 0.95, 1.0))
	_ctrl_steer_raw_bar = st_bars[0] as ColorRect
	_ctrl_steer_out_bar = st_bars[1] as ColorRect

	var th_bars := _make_dual_bar_row(vb, "Throttle", Color(0.45, 0.78, 0.45, 0.85), Color(0.13, 0.85, 0.37, 1.0))
	_ctrl_thr_raw_bar = th_bars[0] as ColorRect
	_ctrl_thr_out_bar = th_bars[1] as ColorRect

	var br_bars := _make_dual_bar_row(vb, "Brake", Color(0.85, 0.45, 0.35, 0.85), Color(0.95, 0.25, 0.22, 1.0))
	_ctrl_brk_raw_bar = br_bars[0] as ColorRect
	_ctrl_brk_out_bar = br_bars[1] as ColorRect

	_ctrl_live_lbl = Label.new()
	_ctrl_live_lbl.add_theme_font_override("font", mono)
	_ctrl_live_lbl.add_theme_font_size_override("font_size", 11)
	_ctrl_live_lbl.text = "Str: +0.00->+0.00 | Thr: 0.00->0.00 | Brk: 0.00->0.00"
	vb.add_child(_ctrl_live_lbl)

	vb.add_child(HSeparator.new())
	var hdr := Label.new()
	hdr.text = "INPUT CONDITIONER TUNING"
	hdr.add_theme_font_size_override("font_size", 11)
	hdr.add_theme_color_override("font_color", Color(0.74, 0.90, 1.0, 1.0))
	vb.add_child(hdr)

	var defs := [
		{"key": "steer_deadzone", "label": "Steer Deadzone", "min": 0.00, "max": 0.25, "step": 0.01, "def": 0.06, "fmt": "%.2f"},
		{"key": "steer_gamma", "label": "Steer Gamma", "min": 0.50, "max": 3.00, "step": 0.05, "def": 1.60, "fmt": "%.2f"},
		{"key": "steer_speed_sensitivity", "label": "Steer Speed Sens", "min": 0.00, "max": 2.00, "step": 0.05, "def": 1.00, "fmt": "%.2f"},
		{"key": "throttle_gamma", "label": "Throttle Gamma", "min": 0.50, "max": 2.50, "step": 0.05, "def": 1.00, "fmt": "%.2f"},
		{"key": "brake_gamma", "label": "Brake Gamma", "min": 0.50, "max": 2.50, "step": 0.05, "def": 1.00, "fmt": "%.2f"},
		{"key": "rumble_intensity", "label": "Rumble Intensity", "min": 0.00, "max": 1.50, "step": 0.05, "def": 1.00, "fmt": "%.2f"},
	]
	for d in defs:
		var k: String = d["key"]
		var fmt: String = d["fmt"]
		var pair := _make_row_slider(vb, String(d["label"]), float(d["min"]), float(d["max"]), float(d["step"]), float(d["def"]), func(v: float): _on_ctrl_param_changed(k, v, fmt))
		_ctrl_sliders[k] = pair[0]
		_ctrl_labels[k] = pair[1]
		(pair[1] as Label).text = fmt % float(d["def"])

	vb.add_child(HSeparator.new())
	var hk_hdr := Label.new()
	hk_hdr.text = "ACTIVE HOTKEYS REFERENCE"
	hk_hdr.add_theme_font_size_override("font_size", 11)
	hk_hdr.add_theme_color_override("font_color", Color(0.74, 0.90, 1.0, 1.0))
	vb.add_child(hk_hdr)

	var hk_lbl := Label.new()
	hk_lbl.add_theme_font_override("font", mono)
	hk_lbl.add_theme_font_size_override("font_size", 11)
	hk_lbl.text = "\n".join([
		"W/S/A/D   : Throttle / Brake / Steer  | Space : EBS",
		"Esc       : Toggle Menu               | C/F3  : Cycle Camera",
		"M         : Cycle Drive Mode          | V     : 3D Suspension",
		"L         : Arm Launch Control        | R     : Rewind 3.0s",
		"G / B     : Set / Go Practice Pin 1   | T     : Track Menu",
		"F5 / F6   : Reset Lap / Reset Cones   | F7    : Cycle Slow-Mo",
		"F8        : 3D Wheel Forces (untouched)",
		"F9 / F10  : Record / TV Tune          | F11/12: Scope / Freeze",
	])
	vb.add_child(hk_lbl)


func _build_telemetry_tab() -> void:
	var vb := _make_tab_margin("Telemetry", true)
	var mono := _make_mono_font()

	_tel_kin_lbl = Label.new()
	_tel_kin_lbl.add_theme_font_override("font", mono)
	_tel_kin_lbl.add_theme_font_size_override("font_size", 12)
	_tel_kin_lbl.text = "0.0 km/h | beta +0.00 deg | mu_min 1.50\nax +0.00g | ay +0.00g"
	vb.add_child(_tel_kin_lbl)

	var pwr_title := Label.new()
	pwr_title.text = "HV DC (-25 .. +80 kW)"
	vb.add_child(pwr_title)

	var pwr_track := Control.new()
	pwr_track.custom_minimum_size = Vector2(350, 14)
	var pwr_bg := ColorRect.new()
	pwr_bg.color = Color(0.12, 0.14, 0.18, 1.0)
	pwr_bg.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	pwr_track.add_child(pwr_bg)
	_tel_pwr_neg = ColorRect.new()
	_tel_pwr_neg.color = Color(0.18, 0.78, 0.92, 1.0)
	_tel_pwr_neg.position = Vector2(83, 2)
	_tel_pwr_neg.size = Vector2(0, 10)
	pwr_track.add_child(_tel_pwr_neg)
	_tel_pwr_pos = ColorRect.new()
	_tel_pwr_pos.color = Color(0.95, 0.62, 0.15, 1.0)
	_tel_pwr_pos.position = Vector2(83, 2)
	_tel_pwr_pos.size = Vector2(0, 10)
	pwr_track.add_child(_tel_pwr_pos)
	var pwr_zero := ColorRect.new()
	pwr_zero.color = Color(0.90, 0.92, 0.96, 0.9)
	pwr_zero.position = Vector2(82, 0)
	pwr_zero.size = Vector2(2, 14)
	pwr_track.add_child(pwr_zero)
	vb.add_child(pwr_track)

	_tel_pwr_val = Label.new()
	_tel_pwr_val.add_theme_font_override("font", mono)
	_tel_pwr_val.add_theme_font_size_override("font_size", 12)
	_tel_pwr_val.text = "+0.0 kW | 540 V | SoC 95.0%"
	vb.add_child(_tel_pwr_val)

	_tel_brk_title = Label.new()
	_tel_brk_title.text = "Brake 35/65 (REGEN 0-35% / HYD 35-100%)"
	vb.add_child(_tel_brk_title)

	var brk_track := Control.new()
	brk_track.custom_minimum_size = Vector2(350, 12)
	var brk_bg := ColorRect.new()
	brk_bg.color = Color(0.12, 0.14, 0.18, 1.0)
	brk_bg.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	brk_track.add_child(brk_bg)
	_tel_brk_reg = ColorRect.new()
	_tel_brk_reg.color = Color(0.20, 0.85, 0.55, 1.0)
	_tel_brk_reg.position = Vector2(0, 2)
	_tel_brk_reg.size = Vector2(0, 8)
	brk_track.add_child(_tel_brk_reg)
	_tel_brk_hyd = ColorRect.new()
	_tel_brk_hyd.color = Color(0.95, 0.25, 0.22, 1.0)
	_tel_brk_hyd.position = Vector2(122.5, 2)
	_tel_brk_hyd.size = Vector2(0, 8)
	brk_track.add_child(_tel_brk_hyd)
	_tel_brk_mark = ColorRect.new()
	_tel_brk_mark.color = Color(0.95, 0.95, 0.95, 0.9)
	_tel_brk_mark.position = Vector2(122.5, 0)
	_tel_brk_mark.size = Vector2(2, 12)
	brk_track.add_child(_tel_brk_mark)
	vb.add_child(brk_track)

	_tel_brk_val = Label.new()
	_tel_brk_val.add_theme_font_override("font", mono)
	_tel_brk_val.add_theme_font_size_override("font_size", 12)
	_tel_brk_val.text = "BRK 0% | T_disc 25C | T_surf 25C"
	vb.add_child(_tel_brk_val)

	_tel_ecu_lbl = Label.new()
	_tel_ecu_lbl.add_theme_font_override("font", mono)
	_tel_ecu_lbl.add_theme_font_size_override("font_size", 12)
	_tel_ecu_lbl.text = "TC:-- ABS:-- CURB:-- LIMP:-- SAT Mz:0%"
	vb.add_child(_tel_ecu_lbl)

	_tel_steer_lbl = Label.new()
	_tel_steer_lbl.add_theme_font_override("font", mono)
	_tel_steer_lbl.add_theme_font_size_override("font_size", 12)
	_tel_steer_lbl.text = "STEER TORQUE: +0.0 Nm"
	vb.add_child(_tel_steer_lbl)

	var tg_title := Label.new()
	tg_title.text = "TIRE GRIP & TEMP (FL / FR / RL / RR)"
	tg_title.add_theme_font_size_override("font_size", 12)
	vb.add_child(tg_title)

	var grid := GridContainer.new()
	grid.columns = 2
	grid.add_theme_constant_override("h_separation", 8)
	grid.add_theme_constant_override("v_separation", 4)
	vb.add_child(grid)
	var c_names := ["FL", "FR", "RL", "RR"]
	_tel_tire_bars.clear()
	_tel_tire_lbls.clear()
	for ci in range(4):
		var cell := VBoxContainer.new()
		cell.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		cell.add_theme_constant_override("separation", 2)
		grid.add_child(cell)
		var lbl_c := Label.new()
		lbl_c.add_theme_font_override("font", mono)
		lbl_c.add_theme_font_size_override("font_size", 11)
		lbl_c.text = "%s:  0%% | 40.0 C" % c_names[ci]
		cell.add_child(lbl_c)
		_tel_tire_lbls.append(lbl_c)
		var track_c := Control.new()
		track_c.custom_minimum_size = Vector2(162, 8)
		var bg_c := ColorRect.new()
		bg_c.color = Color(0.12, 0.14, 0.18, 1.0)
		bg_c.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
		track_c.add_child(bg_c)
		var fill_c := ColorRect.new()
		fill_c.color = Color(0.13, 0.77, 0.37, 1.0)
		fill_c.position = Vector2(0, 1)
		fill_c.size = Vector2(0, 6)
		track_c.add_child(fill_c)
		_tel_tire_bars.append(fill_c)
		cell.add_child(track_c)

	var leg_lbl := Label.new()
	leg_lbl.add_theme_font_size_override("font_size", 11)
	leg_lbl.add_theme_color_override("font_color", Color(0.75, 0.82, 0.90, 1.0))
	leg_lbl.text = "DRV: Green | REGEN: Orange-Red | MECH: Deep Red"
	vb.add_child(leg_lbl)

	var tools_row := HBoxContainer.new()
	tools_row.add_theme_constant_override("separation", 6)
	vb.add_child(tools_row)

	_btn_rec = Button.new()
	_btn_rec.text = "Record [F9]"
	_btn_rec.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_btn_rec.focus_mode = Control.FOCUS_NONE
	_btn_rec.pressed.connect(_on_rec_pressed)
	tools_row.add_child(_btn_rec)

	_btn_tune = Button.new()
	_btn_tune.text = "TV Tune [F10]"
	_btn_tune.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_btn_tune.focus_mode = Control.FOCUS_NONE
	_btn_tune.pressed.connect(_on_tune_pressed)
	tools_row.add_child(_btn_tune)

	_btn_scope = Button.new()
	_btn_scope.text = "Scope [F11]"
	_btn_scope.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_btn_scope.focus_mode = Control.FOCUS_NONE
	_btn_scope.pressed.connect(_on_scope_pressed)
	tools_row.add_child(_btn_scope)


func _augment_track_tab(trk_tab: Node) -> void:
	if trk_tab == null:
		return
	var hb := trk_tab.get_node_or_null("HBoxContainer") as HBoxContainer
	var vb := trk_tab.get_node_or_null("HBoxContainer/VBoxContainer") as VBoxContainer
	if hb == null or vb == null:
		return

	# Wrap HBoxContainer inside a ScrollContainer (safe after TRACK._ready() @onready resolved)
	var sc := ScrollContainer.new()
	sc.focus_mode = Control.FOCUS_NONE
	sc.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	sc.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	sc.size_flags_vertical = Control.SIZE_EXPAND_FILL
	trk_tab.remove_child(hb)
	trk_tab.add_child(sc)
	sc.add_child(hb)
	hb.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	vb.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	vb.add_theme_constant_override("separation", 5)

	# Quick Test Zones & Multi-Pin
	vb.add_child(HSeparator.new())
	var sp_hdr := Label.new()
	sp_hdr.text = "TEST ZONES & MULTI-PIN"
	sp_hdr.add_theme_font_size_override("font_size", 11)
	sp_hdr.add_theme_color_override("font_color", Color(0.74, 0.90, 1.0, 1.0))
	vb.add_child(sp_hdr)

	var sp_row1 := HBoxContainer.new()
	sp_row1.add_theme_constant_override("separation", 4)
	vb.add_child(sp_row1)
	for pair in [["Launch Pad", "launch"], ["Hairpin Test", "hairpin"], ["Slalom Test", "slalom"]]:
		var btn := Button.new()
		btn.text = pair[0]
		btn.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		btn.focus_mode = Control.FOCUS_NONE
		var key: String = pair[1]
		btn.pressed.connect(func(): _on_quick_spawn_pressed(key))
		sp_row1.add_child(btn)

	var sp_row1b := HBoxContainer.new()
	sp_row1b.add_theme_constant_override("separation", 4)
	vb.add_child(sp_row1b)
	for pair in [["Trail-Brake Entry", "trail_brake"], ["Skidpad 9.125m Test", "skidpad"]]:
		var btn := Button.new()
		btn.text = pair[0]
		btn.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		btn.focus_mode = Control.FOCUS_NONE
		var key: String = pair[1]
		btn.pressed.connect(func(): _on_quick_spawn_pressed(key))
		sp_row1b.add_child(btn)

	var sp_row2 := HBoxContainer.new()
	sp_row2.add_theme_constant_override("separation", 4)
	vb.add_child(sp_row2)
	var btn_cones := Button.new()
	btn_cones.text = "Reset Cones Only [F6]"
	btn_cones.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	btn_cones.focus_mode = Control.FOCUS_NONE
	btn_cones.pressed.connect(_on_reset_cones_pressed)
	sp_row2.add_child(btn_cones)

	var btn_lap := Button.new()
	btn_lap.text = "Reset Lap [F5]"
	btn_lap.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	btn_lap.focus_mode = Control.FOCUS_NONE
	btn_lap.pressed.connect(_on_reset_lap_pressed)
	sp_row2.add_child(btn_lap)

	var pin_row := HBoxContainer.new()
	pin_row.add_theme_constant_override("separation", 4)
	vb.add_child(pin_row)
	for slot in range(3):
		var slot_box := HBoxContainer.new()
		slot_box.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		slot_box.add_theme_constant_override("separation", 2)
		pin_row.add_child(slot_box)

		var btn_set := Button.new()
		btn_set.text = "Pin %d Set" % (slot + 1)
		btn_set.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		btn_set.focus_mode = Control.FOCUS_NONE
		var s_idx := slot
		btn_set.pressed.connect(func(): _on_pin_set_pressed(s_idx))
		slot_box.add_child(btn_set)

		var btn_go := Button.new()
		btn_go.text = "Go"
		btn_go.focus_mode = Control.FOCUS_NONE
		btn_go.pressed.connect(func(): _on_pin_go_pressed(s_idx))
		slot_box.add_child(btn_go)

	_pin_status_lbl = Label.new()
	_pin_status_lbl.add_theme_font_size_override("font_size", 11)
	_pin_status_lbl.text = "Pins [G/B=Pin1]: P1:--  P2:--  P3:--"
	vb.add_child(_pin_status_lbl)

	# Skidpad 9.125m Stats block
	_skidpad_stats_lbl = Label.new()
	_skidpad_stats_lbl.add_theme_font_override("font", _make_mono_font())
	_skidpad_stats_lbl.add_theme_font_size_override("font_size", 11)
	_skidpad_stats_lbl.text = "Skidpad 9.125m | Last: -- | Best: -- | Avg ay: -- g | Avg beta: -- deg"
	vb.add_child(_skidpad_stats_lbl)

	# Rewind 3.0s & Slow-Mo
	var rw_row := HBoxContainer.new()
	rw_row.add_theme_constant_override("separation", 6)
	vb.add_child(rw_row)
	var btn_rew := Button.new()
	btn_rew.text = "Rewind 3.0s [R]"
	btn_rew.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	btn_rew.focus_mode = Control.FOCUS_NONE
	btn_rew.pressed.connect(_on_rewind_pressed)
	rw_row.add_child(btn_rew)

	_time_scale_opt = OptionButton.new()
	_time_scale_opt.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_time_scale_opt.focus_mode = Control.FOCUS_NONE
	for i in range(TIME_SCALE_LABELS.size()):
		_time_scale_opt.add_item(TIME_SCALE_LABELS[i], i)
	_time_scale_opt.item_selected.connect(_on_time_scale_selected)
	rw_row.add_child(_time_scale_opt)

	# Live Track Conditions
	vb.add_child(HSeparator.new())
	var tc_hdr := Label.new()
	tc_hdr.text = "LIVE TRACK CONDITIONS"
	tc_hdr.add_theme_font_size_override("font_size", 11)
	tc_hdr.add_theme_color_override("font_color", Color(0.74, 0.90, 1.0, 1.0))
	vb.add_child(tc_hdr)

	_grip_profile_opt = _make_row_opt(
		vb,
		"Grip / Aero Profile",
		["TTC Lab Peak (mu 1.00, ClA 6.0)", "FSG Real Track (mu 0.86, ClA 3.8)"],
		_on_grip_profile_selected
	)
	var mu_pair := _make_row_slider(vb, "Surface Grip (mu)", 0.60, 1.15, 0.01, 1.00, func(_v: float): _on_track_conditions_changed())
	_sld_mu_scale = mu_pair[0] as HSlider
	_lbl_mu_scale = mu_pair[1] as Label
	_lbl_mu_scale.text = "1.00"

	_tire_state_opt = _make_row_opt(vb, "Tire State", TIRE_STATE_LABELS, func(_idx: int): _on_track_conditions_changed())

	var aero_pair := _make_row_slider(vb, "Aero Scale", 0.50, 1.15, 0.01, 1.00, func(_v: float): _on_track_conditions_changed())
	_sld_aero_scale = aero_pair[0] as HSlider
	_lbl_aero_scale = aero_pair[1] as Label
	_lbl_aero_scale.text = "1.00x"

	_chk_split_mu = CheckButton.new()
	_chk_split_mu.text = "Split-Mu Test (L 0.65 / R 1.00)"
	_chk_split_mu.focus_mode = Control.FOCUS_NONE
	_chk_split_mu.toggled.connect(func(_on: bool): _on_track_conditions_changed())
	vb.add_child(_chk_split_mu)

	_road_surf_opt = _make_row_opt(vb, "Road Surface", ROAD_SURFACE_LABELS, _on_road_surface_changed)

	var r_pair := _make_row_slider(vb, "Roughness", 0.0, 3.0, 0.1, 1.0, func(_v: float): _on_road_surface_changed(0))
	_sld_rough = r_pair[0] as HSlider
	_lbl_rough = r_pair[1] as Label
	_lbl_rough.text = "1.0x"

	_bump_mode_opt = _make_row_opt(vb, "Pronounced Bumps", BUMP_MODE_LABELS, _on_road_surface_changed)

	var bh_pair := _make_row_slider(vb, "Bump Height", 0.0, 60.0, 1.0, 25.0, func(_v: float): _on_road_surface_changed(0))
	_sld_bump_h = bh_pair[0] as HSlider
	_lbl_bump_h = bh_pair[1] as Label
	_lbl_bump_h.text = "25mm"

	var bs_pair := _make_row_slider(vb, "Bump Spacing", 15.0, 80.0, 5.0, 35.0, func(_v: float): _on_road_surface_changed(0))
	_sld_bump_sp = bs_pair[0] as HSlider
	_lbl_bump_sp = bs_pair[1] as Label
	_lbl_bump_sp.text = "35m"

	# Session Lap History & Telemetry CSV Logger
	vb.add_child(HSeparator.new())
	_session_lbl = Label.new()
	_session_lbl.add_theme_font_override("font", _make_mono_font())
	_session_lbl.add_theme_font_size_override("font_size", 11)
	_session_lbl.text = "Session: 00:00.0 | E_net: +0.0 kJ"
	vb.add_child(_session_lbl)

	_csv_rate_opt = _make_row_opt(vb, "CSV Rate", CSV_RATE_LABELS, _on_csv_rate_selected)
	_csv_rate_opt.select(1) # Default 100 Hz

	var csv_row := HBoxContainer.new()
	csv_row.add_theme_constant_override("separation", 6)
	vb.add_child(csv_row)
	_btn_csv_rec = Button.new()
	_btn_csv_rec.text = "Record CSV"
	_btn_csv_rec.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_btn_csv_rec.focus_mode = Control.FOCUS_NONE
	_btn_csv_rec.pressed.connect(_on_csv_rec_pressed)
	csv_row.add_child(_btn_csv_rec)

	_chk_auto_csv = CheckButton.new()
	_chk_auto_csv.text = "Auto-Record Laps"
	_chk_auto_csv.focus_mode = Control.FOCUS_NONE
	_chk_auto_csv.toggled.connect(_on_auto_csv_toggled)
	csv_row.add_child(_chk_auto_csv)

	var csv_row2 := HBoxContainer.new()
	csv_row2.add_theme_constant_override("separation", 6)
	vb.add_child(csv_row2)
	var btn_exp_sess := Button.new()
	btn_exp_sess.text = "Export Session"
	btn_exp_sess.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	btn_exp_sess.focus_mode = Control.FOCUS_NONE
	btn_exp_sess.pressed.connect(_on_export_session_pressed)
	csv_row2.add_child(btn_exp_sess)

	var btn_open_logs := Button.new()
	btn_open_logs.text = "Open Logs Folder"
	btn_open_logs.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	btn_open_logs.focus_mode = Control.FOCUS_NONE
	btn_open_logs.pressed.connect(_on_open_logs_folder_pressed)
	csv_row2.add_child(btn_open_logs)

	_csv_status_lbl = Label.new()
	_csv_status_lbl.add_theme_font_size_override("font_size", 11)
	_csv_status_lbl.text = "CSV: Idle"
	vb.add_child(_csv_status_lbl)

	_lap_history_lbl = Label.new()
	_lap_history_lbl.add_theme_font_override("font", _make_mono_font())
	_lap_history_lbl.add_theme_font_size_override("font_size", 11)
	_lap_history_lbl.text = "Lap | S1/S2/S3          | Raw    | Cones | Official | Map/Preset\n(no completed laps)"
	vb.add_child(_lap_history_lbl)


func _build_lts_tab() -> void:
	var vb := _make_tab_margin("LTS", true)
	_lts_ctrl = LtsMenuControllerScript.new()
	_lts_ctrl.name = "LtsMenuController"
	_lts_ctrl.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	vb.add_child(_lts_ctrl)


func _get_sim() -> Node:
	return get_node_or_null("/root/Sim")


func _get_car() -> Node:
	var sim_node := _get_sim()
	if sim_node != null and "car" in sim_node and sim_node.car != null:
		return sim_node.car
	var p := get_parent()
	while p != null:
		if p.has_node("Car"):
			return p.get_node("Car")
		p = p.get_parent()
	return null


func _get_track() -> Node:
	var sim_node := _get_sim()
	if sim_node != null and "track" in sim_node and sim_node.track != null:
		return sim_node.track
	var p := get_parent()
	while p != null:
		if p.has_node("Track"):
			return p.get_node("Track")
		p = p.get_parent()
	return null


func _refresh_setup_profiles(car: Node, vehicle_id: String, active_setup: String) -> void:
	if _setup_opt == null or car == null or not car.has_method("list_setup_profiles"):
		return
	_setup_opt.clear()
	var profiles: Array = car.list_setup_profiles()
	if profiles.is_empty():
		_setup_opt.add_item("default", 0)
		_setup_opt.set_item_metadata(0, "default")
		_setup_opt.select(0)
	else:
		var sel_idx := 0
		for i in range(profiles.size()):
			var entry: Dictionary = profiles[i]
			var nm := String(entry.get("name", entry.get("stem", "default")))
			var stem := String(entry.get("stem", "default"))
			_setup_opt.add_item(nm, i)
			_setup_opt.set_item_metadata(i, stem)
			if stem == active_setup or nm == active_setup:
				sel_idx = i
		_setup_opt.select(sel_idx)
	_last_setup_vehicle_id = vehicle_id


func _physics_process(delta: float) -> void:
	var car := _get_car()
	if car == null or not car.has_method("get_hud_snapshot"):
		return
	var snap: Dictionary = car.get_hud_snapshot()
	_rewind_buf.record_step(delta, car)
	if _tv_tuning_ctrl != null:
		_tv_tuning_ctrl.step_launch_tracker(delta, snap, car)

	var ds: Dictionary = snap.get("dil_summary", {}) if snap.get("dil_summary") is Dictionary else {}
	var lap_active := bool(ds.get("lap_active", false))
	var trk := _get_track()
	var dil: Variant = trk.get("dil_session") if trk != null else null
	var map_name := String(snap.get("tv_map_name", MODE_LABELS[clampi(int(snap.get("mode_id", 1)), 0, MODE_LABELS.size() - 1)]))
	if dil != null and "current_tv_map_name" in dil:
		dil.current_tv_map_name = map_name

	var t_name: String = String(trk.get("last_loaded_track_path")).get_file().get_basename() if (trk != null and "last_loaded_track_path" in trk) else "track"
	var lap_num: int = (dil.lap_history.size() + 1) if (dil != null and "lap_history" in dil) else 1
	if dil != null and dil.has_method("set_active_context"):
		var preset_nm := _tv_tuning_ctrl.get_active_preset_name() if (_tv_tuning_ctrl != null and _tv_tuning_ctrl.has_method("get_active_preset_name")) else "default"
		dil.set_active_context(map_name, preset_nm)
	if _csv_logger.has_method("on_lap_update"):
		_csv_logger.on_lap_update(lap_active, lap_num, t_name, map_name)
	elif _csv_logger.auto_record_laps:
		if lap_active and not _prev_lap_active and not _csv_logger.is_recording:
			_csv_logger.start_recording(t_name, map_name)
		elif not lap_active and _prev_lap_active and _csv_logger.is_recording:
			_csv_logger.stop_recording()
	_prev_lap_active = lap_active

	if _csv_logger.is_recording:
		var s_cur: float = float(dil._s_cur_m) if (dil != null and "_s_cur_m" in dil) else 0.0
		_csv_logger.log_step(delta, snap, car, lap_num, s_cur, map_name)


func _process(delta: float) -> void:
	var car := _get_car()
	if car == null or not car.has_method("get_hud_snapshot"):
		return
	var snap: Dictionary = car.get_hud_snapshot()

	_syncing_ui = true
	var vid := String(snap.get("vehicle_id", "ter27_4wd"))
	var fid := String(snap.get("physics_fidelity", "full_14dof"))
	var is_ter27: bool = (vid == "ter27_4wd")
	var kin_sup: bool = bool(snap.get("kinematics_3d_supported", false))
	var kin_en: bool = bool(snap.get("kinematics_3d_enabled", false))

	if _veh_opt != null:
		var v_idx: int = VEHICLE_IDS.find(vid)
		if v_idx >= 0 and _veh_opt.selected != v_idx:
			_veh_opt.select(v_idx)

	if _model_opt != null:
		var m_idx: int = MODEL_IDS.find(fid)
		if m_idx >= 0 and _model_opt.selected != m_idx:
			_model_opt.select(m_idx)

	if _kin_opt != null:
		_kin_opt.disabled = not kin_sup
		var k_idx: int = 0 if kin_en else 1
		if _kin_opt.selected != k_idx:
			_kin_opt.select(k_idx)

	if vid != _last_setup_vehicle_id:
		_refresh_setup_profiles(car, vid, String(snap.get("active_setup_name", "default")))

	if _mode_opt != null:
		_mode_opt.disabled = not is_ter27
		var mid := int(snap.get("mode_id", 1))
		if not is_ter27:
			mid = 0
		if _mode_opt.selected != mid and mid >= 0 and mid < _mode_opt.item_count:
			_mode_opt.select(mid)

	if _tv_tuning_ctrl != null:
		_tv_tuning_ctrl.sync_from_car(car, snap)

	if _cam_opt != null and car.has_method("get_camera_names"):
		var cam_names: Array[String] = car.get_camera_names()
		if cam_names.size() != _last_cam_count:
			_cam_opt.clear()
			for i in range(cam_names.size()):
				_cam_opt.add_item(cam_names[i], i)
			_last_cam_count = cam_names.size()
		var active_cam: int = int(car.get_active_camera_index())
		if active_cam >= 0 and active_cam < _cam_opt.item_count and _cam_opt.selected != active_cam:
			_cam_opt.select(active_cam)

	var sim_node := _get_sim()
	if _sld_cam_fov != null and sim_node != null and sim_node.has_method("get_active_camera_fov"):
		var fov_v: float = float(sim_node.get_active_camera_fov())
		if absf(_sld_cam_fov.value - fov_v) > 0.5:
			_sld_cam_fov.value = fov_v
			if _lbl_cam_fov != null:
				_lbl_cam_fov.text = "%d deg" % int(roundf(fov_v))

	if _chk_susp_vis != null:
		_chk_susp_vis.disabled = not bool(snap.get("susp_supported", false))
		_chk_susp_vis.button_pressed = bool(snap.get("susp_visible", false))
	if _chk_xray != null:
		_chk_xray.button_pressed = bool(snap.get("xray_body_only", false))
	if _chk_top != null:
		_chk_top.button_pressed = bool(snap.get("always_on_top", false))
	if _chk_arrows != null:
		_chk_arrows.button_pressed = bool(snap.get("wheel_arrows", false))
	if _chk_fz_arrows != null:
		_chk_fz_arrows.button_pressed = bool(snap.get("fz_arrows", false))
	if _chk_torque_bars != null:
		_chk_torque_bars.button_pressed = bool(snap.get("torque_bars", false))
	if _chk_ellipses != null and "wheel_forces_vis" in car and car.wheel_forces_vis != null and "show_friction_ellipses" in car.wheel_forces_vis:
		_chk_ellipses.button_pressed = bool(car.wheel_forces_vis.show_friction_ellipses)
	var trk_sync := _get_track()
	if _chk_skidpad_ring != null and trk_sync != null and trk_sync.has_method("is_skidpad_ring_visible"):
		_chk_skidpad_ring.button_pressed = bool(trk_sync.is_skidpad_ring_visible())
	if _chk_racing_line != null:
		_chk_racing_line.button_pressed = bool(snap.get("racing_line", false))
	if _line_color_opt != null and trk_sync != null and "racing_line_vis" in trk_sync and trk_sync.racing_line_vis != null:
		var rlv_sync: Node = trk_sync.racing_line_vis
		if rlv_sync.has_method("get_color_mode"):
			var cm := int(rlv_sync.get_color_mode())
			if cm >= 0 and cm < _line_color_opt.item_count and _line_color_opt.selected != cm:
				_line_color_opt.select(cm)
		if rlv_sync.visible and rlv_sync.has_method("update_live_driver_speed"):
			var dil_v: Variant = trk_sync.get("dil_session")
			var s_now := float(dil_v._s_cur_m) if (dil_v != null and "_s_cur_m" in dil_v) else 0.0
			var v_mps := float(snap.get("kmh", 0.0)) / 3.6
			rlv_sync.update_live_driver_speed(s_now, v_mps, delta)
	if _chk_line_markers != null:
		if trk_sync != null and trk_sync.has_method("is_track_markers_visible"):
			_chk_line_markers.button_pressed = bool(trk_sync.is_track_markers_visible())
		else:
			_chk_line_markers.button_pressed = bool(snap.get("racing_line_markers", true))
	if _ghost_opt != null:
		var dil_sync: Variant = trk_sync.get("dil_session") if trk_sync != null else null
		if dil_sync != null and dil_sync.has_method("get_ghost_source"):
			var gs := String(dil_sync.get_ghost_source())
			var g_idx := GHOST_SOURCE_KEYS.find(gs)
			if g_idx >= 0 and _ghost_opt.selected != g_idx:
				_ghost_opt.select(g_idx)
		else:
			var gm := int(snap.get("ghost_mode", 0))
			if gm >= 0 and gm < _ghost_opt.item_count and _ghost_opt.selected != gm:
				_ghost_opt.select(gm)
	if _chk_split_mu != null and "split_mu_test_enabled" in car:
		_chk_split_mu.button_pressed = bool(car.split_mu_test_enabled)
	if _chk_dil_audio != null:
		_chk_dil_audio.button_pressed = bool(snap.get("dil_audio", false))
	if "dil_audio" in car and car.dil_audio != null:
		var da: Node = car.dil_audio
		if _audio_profile_opt != null and da.has_method("get_sound_profile"):
			var pid := int(da.get_sound_profile())
			if pid >= 0 and pid < _audio_profile_opt.item_count and _audio_profile_opt.selected != pid:
				_audio_profile_opt.select(pid)
	if _chk_haptics != null:
		_chk_haptics.button_pressed = bool(snap.get("haptics_enabled", true))
	var rs_cfg: Dictionary = snap.get("road_surface_cfg", {}) if snap.get("road_surface_cfg") is Dictionary else {}
	if not rs_cfg.is_empty():
		var sm := int(rs_cfg.get("surface_mode", 0))
		if _road_surf_opt != null and sm >= 0 and sm < _road_surf_opt.item_count and _road_surf_opt.selected != sm:
			_road_surf_opt.select(sm)
		var bm := int(rs_cfg.get("bump_mode", 0))
		if _bump_mode_opt != null and bm >= 0 and bm < _bump_mode_opt.item_count and _bump_mode_opt.selected != bm:
			_bump_mode_opt.select(bm)
	if _grip_profile_opt != null:
		var gp_idx := int(snap.get("track_grip_profile", 0))
		if gp_idx >= 0 and gp_idx < _grip_profile_opt.item_count and _grip_profile_opt.selected != gp_idx:
			_grip_profile_opt.select(gp_idx)
	if _input_profile_opt != null:
		var ip_idx := int(snap.get("input_profile", 0))
		if ip_idx >= 0 and ip_idx < _input_profile_opt.item_count and _input_profile_opt.selected != ip_idx:
			_input_profile_opt.select(ip_idx)
	_syncing_ui = false

	# Update View corner table
	if _susp_table != null:
		var corners: Array = snap.get("corners", [])
		if corners.is_empty():
			_susp_table.text = "No 3D double-wishbone telemetry."
		else:
			var lines := PackedStringArray(["Corner | dz[mm] | Cam[deg] | Fz[N] | Farb[N]"])
			for c in corners:
				lines.append("%-6s | %+6.1f |  %+6.2f  | %5.0f | %+7.1f" % [
					String(c.get("name", "?")),
					float(c.get("dz_mm", 0.0)),
					float(c.get("camber_deg", 0.0)),
					float(c.get("fz_n", 0.0)),
					float(c.get("f_arb_n", 0.0)),
				])
			_susp_table.text = "\n".join(lines)

	if _ctrl_live_lbl != null and "input_conditioner" in car and car.input_conditioner != null and car.input_conditioner.has_method("get_live_input_monitor"):
		var mon: Dictionary = car.input_conditioner.get_live_input_monitor()
		var sr := clampf(float(mon.get("steer_raw", 0.0)), -1.0, 1.0)
		var so := clampf(float(mon.get("steer_out", 0.0)), -1.0, 1.0)
		var tr := clampf(float(mon.get("thr_raw", 0.0)), 0.0, 1.0)
		var to := clampf(float(mon.get("thr_out", 0.0)), 0.0, 1.0)
		var br := clampf(float(mon.get("brk_raw", 0.0)), 0.0, 1.0)
		var bo := clampf(float(mon.get("brk_out", 0.0)), 0.0, 1.0)
		var max_w := 260.0
		if _ctrl_steer_raw_bar != null:
			var half := max_w * 0.5
			_ctrl_steer_raw_bar.position.x = half if sr >= 0.0 else (half + sr * half)
			_ctrl_steer_raw_bar.size.x = absf(sr) * half
		if _ctrl_steer_out_bar != null:
			var half := max_w * 0.5
			_ctrl_steer_out_bar.position.x = half if so >= 0.0 else (half + so * half)
			_ctrl_steer_out_bar.size.x = absf(so) * half
		if _ctrl_thr_raw_bar != null:
			_ctrl_thr_raw_bar.size.x = tr * max_w
		if _ctrl_thr_out_bar != null:
			_ctrl_thr_out_bar.size.x = to * max_w
		if _ctrl_brk_raw_bar != null:
			_ctrl_brk_raw_bar.size.x = br * max_w
		if _ctrl_brk_out_bar != null:
			_ctrl_brk_out_bar.size.x = bo * max_w
		_ctrl_live_lbl.text = "Str: %+.2f->%+.2f | Thr: %.2f->%.2f | Brk: %.2f->%.2f" % [sr, so, tr, to, br, bo]

	# Update Telemetry tab
	if _tel_kin_lbl != null:
		_tel_kin_lbl.text = "%5.1f km/h | beta %+5.2f deg | mu_min %.2f\nax %+5.2fg  | ay %+5.2fg" % [
			float(snap.get("kmh", 0.0)),
			float(snap.get("beta_deg", 0.0)),
			float(snap.get("mu_min", 1.50)),
			float(snap.get("ax_g", 0.0)),
			float(snap.get("ay_g", 0.0)),
		]

	if _tel_pwr_neg != null and _tel_pwr_pos != null and _tel_pwr_val != null:
		var raw_p: float = float(snap.get("p_dc_kw", 0.0))
		var p_kw: float = clampf(raw_p, -25.0, 80.0)
		if p_kw < 0.0:
			var w_neg: float = (-p_kw / 25.0) * 83.0
			_tel_pwr_neg.position.x = 83.0 - w_neg
			_tel_pwr_neg.size.x = w_neg
			_tel_pwr_pos.size.x = 0.0
		else:
			_tel_pwr_neg.size.x = 0.0
			_tel_pwr_pos.size.x = (p_kw / 80.0) * 267.0
			_tel_pwr_pos.color = Color(0.95, 0.25, 0.20, 1.0) if p_kw > 74.0 else Color(0.95, 0.62, 0.15, 1.0)
		_tel_pwr_val.text = "%+6.1f kW | %3.0f V | SoC %4.1f%%" % [
			raw_p, float(snap.get("v_dc_v", 540.0)), float(snap.get("soc_pct", 95.0))
		]

	if _tel_brk_reg != null and _tel_brk_hyd != null and _tel_brk_val != null:
		var b_u: float = float(snap.get("brake", 0.0))
		var mbs: float = float(snap.get("mech_brake_start", 0.35))
		var split_x: float = mbs * 350.0
		if _tel_brk_mark != null:
			_tel_brk_mark.position.x = split_x
		if _tel_brk_title != null:
			var pct_r := int(roundf(mbs * 100.0))
			_tel_brk_title.text = "Brake %d/%d (REGEN 0-%d%% / HYD %d-100%%)" % [pct_r, 100 - pct_r, pct_r, pct_r]
		_tel_brk_reg.size.x = minf(b_u, mbs) * 350.0
		if b_u > mbs:
			_tel_brk_hyd.position.x = split_x
			_tel_brk_hyd.size.x = (b_u - mbs) * 350.0
		else:
			_tel_brk_hyd.size.x = 0.0
		_tel_brk_val.text = "BRK %3.0f%% | T_disc %3.0fC | T_surf %3.0fC" % [
			b_u * 100.0, float(snap.get("max_disc_c", 25.0)), float(snap.get("max_tire_c", 25.0))
		]

	if _tel_ecu_lbl != null:
		_tel_ecu_lbl.text = "TC:%s ABS:%s CURB:%s LIMP:%s SAT Mz:%3.0f%%" % [
			"ON" if bool(snap.get("tc", false)) else "--",
			"ON" if bool(snap.get("abs", false)) else "--",
			"ON" if bool(snap.get("curb", false)) else "--",
			"ON" if bool(snap.get("limp", false)) else "--",
			float(snap.get("mz_sat_pct", 0.0)),
		]

	if _tel_steer_lbl != null:
		_tel_steer_lbl.text = "STEER TORQUE: %+5.1f Nm" % float(snap.get("steering_torque_nm", 0.0))

	if _tel_tire_bars.size() == 4 and _tel_tire_lbls.size() == 4:
		var sat_arr: Array = snap.get("tire_sat_pct", [0.0, 0.0, 0.0, 0.0])
		var tmp_arr: Array = snap.get("tire_temps_c", [40.0, 40.0, 40.0, 40.0])
		var cnms := ["FL", "FR", "RL", "RR"]
		for ci in range(4):
			var spct := clampf(float(sat_arr[ci]) if ci < sat_arr.size() else 0.0, 0.0, 120.0)
			var tc_v := float(tmp_arr[ci]) if ci < tmp_arr.size() else 40.0
			_tel_tire_lbls[ci].text = "%s:%3.0f%% | %4.1f C" % [cnms[ci], spct, tc_v]
			_tel_tire_bars[ci].size.x = clampf(spct / 100.0, 0.0, 1.0) * 162.0
			if spct < 70.0:
				_tel_tire_bars[ci].color = Color(0.13, 0.77, 0.37, 1.0)
			elif spct <= 90.0:
				_tel_tire_bars[ci].color = Color(0.96, 0.62, 0.04, 1.0)
			else:
				_tel_tire_bars[ci].color = Color(0.94, 0.25, 0.25, 1.0)

	if _dil_timing_lbl != null:
		var ds: Dictionary = snap.get("dil_summary", {}) if snap.get("dil_summary") is Dictionary else {}
		var cur_l := float(ds.get("current_lap_s", 0.0))
		var best_l := float(ds.get("best_lap_s", 0.0))
		var delta_l := float(ds.get("live_delta_s", 0.0))
		var cones_n := int(ds.get("cones_hit", 0))
		var pen_l := float(ds.get("penalty_s", 0.0))
		var cur_str := "%02d:%05.2f" % [int(cur_l / 60.0), fmod(cur_l, 60.0)]
		var best_str := ("%02d:%05.2f" % [int(best_l / 60.0), fmod(best_l, 60.0)]) if best_l > 0.0 else "--:--.--"
		_dil_timing_lbl.text = "LAP %s  |  BEST %s  |  DELTA %+.2fs  |  DOO: %d (+%.1fs)" % [
			cur_str, best_str, delta_l, cones_n, pen_l
		]

	if _btn_rec != null:
		_btn_rec.text = "REC ON [F9]" if bool(snap.get("rec_active", false)) else "Record [F9]"
	if _btn_tune != null:
		_btn_tune.text = "TV Tune* [F10]" if bool(snap.get("tv_tune_open", false)) else "TV Tune [F10]"
	if _btn_scope != null:
		_btn_scope.text = "Scope* [F11]" if bool(snap.get("scope_open", false)) else "Scope [F11]"

	if _session_lbl != null:
		var t_s: float = float(snap.get("session_time_s", 0.0))
		var mins: int = int(t_s / 60.0)
		var secs: float = t_s - float(mins) * 60.0
		_session_lbl.text = "Session: %02d:%04.1f | E_net: %+.1f kJ" % [
			mins, secs, float(snap.get("session_energy_kj", 0.0))
		]

	if _csv_status_lbl != null and _btn_csv_rec != null:
		if _csv_logger.is_recording:
			_btn_csv_rec.text = "Stop CSV (%d)" % _csv_logger.sample_count
			_csv_status_lbl.text = "CSV REC: %s" % _csv_logger.last_file_path.get_file()
		else:
			_btn_csv_rec.text = "Record Telemetry CSV"
			if not _csv_logger.last_file_path.is_empty():
				_csv_status_lbl.text = "CSV Saved: %s (%d)" % [_csv_logger.last_file_path.get_file(), _csv_logger.sample_count]

	if _skidpad_stats_lbl != null:
		var trk_sp := _get_track()
		if trk_sp != null and trk_sp.has_method("get_skidpad_stats"):
			var st: Dictionary = trk_sp.get_skidpad_stats()
			var last_s := float(st.get("last_skidpad_lap_s", 0.0))
			var best_s := float(st.get("best_skidpad_lap_s", 0.0))
			var avg_ay := float(st.get("avg_ay_g", 0.0))
			var avg_b := float(st.get("avg_beta_deg", 0.0))
			_skidpad_stats_lbl.text = "Skidpad 9.125m | Last: %s | Best: %s\nAvg ay: %.2fg | Avg beta: %+.2f deg" % [
				("%.2fs" % last_s) if last_s > 0.0 else "--",
				("%.2fs" % best_s) if best_s > 0.0 else "--",
				avg_ay,
				avg_b,
			]

	if _lap_history_lbl != null:
		var trk := _get_track()
		var dil: Variant = trk.get("dil_session") if trk != null else null
		if dil != null and dil.has_method("get_lap_history"):
			var hist: Array = dil.get_lap_history()
			if hist.is_empty():
				_lap_history_lbl.text = "Lap | S1/S2/S3          | Raw    | Cones | Official | Map/Preset\n(no completed laps)"
			else:
				var lines := PackedStringArray(["Lap | S1/S2/S3          | Raw    | Cones | Official | Map/Preset"])
				var start_i := maxi(0, hist.size() - 6)
				for i in range(start_i, hist.size()):
					var h: Dictionary = hist[i]
					var s1 := float(h.get("s1_s", 0.0))
					var s2 := float(h.get("s2_s", 0.0))
					var s3 := float(h.get("s3_s", 0.0))
					var s_str := "%4.1f/%4.1f/%4.1f" % [s1, s2, s3]
					var map_p := "%s/%s" % [String(h.get("tv_map", "")), String(h.get("preset_name", "def"))]
					lines.append("%3d | %-17s | %5.2fs | %2d+%1.0fs | %6.2fs  | %s" % [
						int(h.get("lap", i + 1)),
						s_str,
						float(h.get("raw_s", 0.0)),
						int(h.get("cones", 0)),
						float(h.get("penalty_s", 0.0)),
						float(h.get("official_s", 0.0)),
						map_p,
					])
				_lap_history_lbl.text = "\n".join(lines)


func _on_vehicle_selected(index: int) -> void:
	if _syncing_ui:
		return
	var car := _get_car()
	if car != null and index >= 0 and index < VEHICLE_IDS.size() and car.has_method("switch_vehicle"):
		car.switch_vehicle(VEHICLE_IDS[index])
		_last_setup_vehicle_id = ""
		if _tv_tuning_ctrl != null and car.has_method("get_hud_snapshot"):
			_tv_tuning_ctrl.sync_from_car(car, car.get_hud_snapshot())
	_release_any_focus()


func _on_model_selected(index: int) -> void:
	if _syncing_ui:
		return
	var car := _get_car()
	if car != null and index >= 0 and index < MODEL_IDS.size() and car.has_method("set_physics_fidelity"):
		car.set_physics_fidelity(MODEL_IDS[index])
		if _tv_tuning_ctrl != null and car.has_method("get_hud_snapshot"):
			_tv_tuning_ctrl.sync_from_car(car, car.get_hud_snapshot())
	_release_any_focus()


func _on_kin_selected(index: int) -> void:
	if _syncing_ui:
		return
	var car := _get_car()
	if car != null and car.has_method("set_kinematics_3d_enabled"):
		car.set_kinematics_3d_enabled(index == 0)
		if _tv_tuning_ctrl != null and car.has_method("get_hud_snapshot"):
			_tv_tuning_ctrl.sync_from_car(car, car.get_hud_snapshot())
	_release_any_focus()


func _on_setup_selected(index: int) -> void:
	if _syncing_ui or _setup_opt == null:
		return
	var car := _get_car()
	var stem := String(_setup_opt.get_item_metadata(index))
	if car != null and stem != "" and car.has_method("apply_vehicle_setup"):
		car.apply_vehicle_setup(stem)
	_release_any_focus()


func _on_mode_selected(index: int) -> void:
	if _syncing_ui:
		return
	var car := _get_car()
	if car != null and car.has_method("set_sim_drive_mode"):
		car.set_sim_drive_mode(index)
	_release_any_focus()


func _on_cam_selected(index: int) -> void:
	if _syncing_ui:
		return
	var car := _get_car()
	if car != null and car.has_method("select_camera_index"):
		car.select_camera_index(index)
	_release_any_focus()


func _on_cam_fov_changed(val: float) -> void:
	if _lbl_cam_fov != null:
		_lbl_cam_fov.text = "%d deg" % int(roundf(val))
	if _syncing_ui:
		return
	var sim_node := _get_sim()
	if sim_node != null and sim_node.has_method("set_active_camera_fov"):
		sim_node.set_active_camera_fov(val)
	_release_any_focus()


func _on_susp_toggled(pressed: bool) -> void:
	if _syncing_ui:
		return
	var car := _get_car()
	if car != null and car.has_method("set_suspension_visualizer_enabled"):
		car.set_suspension_visualizer_enabled(pressed)
		if _tv_tuning_ctrl != null and car.has_method("get_hud_snapshot"):
			_tv_tuning_ctrl.sync_from_car(car, car.get_hud_snapshot())
	_release_any_focus()


func _on_xray_only_toggled(pressed: bool) -> void:
	if _syncing_ui:
		return
	var car := _get_car()
	if car != null and car.has_method("set_suspension_xray_only"):
		car.set_suspension_xray_only(pressed)
	_release_any_focus()


func _on_always_on_top_toggled(pressed: bool) -> void:
	if _syncing_ui:
		return
	var car := _get_car()
	if car != null and car.has_method("set_suspension_always_on_top"):
		car.set_suspension_always_on_top(pressed)
	_release_any_focus()


func _on_arrows_toggled(pressed: bool) -> void:
	if _syncing_ui:
		return
	var car := _get_car()
	if car != null and car.has_method("set_wheel_arrows_enabled"):
		car.set_wheel_arrows_enabled(pressed)
	_release_any_focus()


func _on_focus_susp_cam() -> void:
	var car := _get_car()
	if car != null and car.has_method("focus_suspension_cam"):
		car.focus_suspension_cam()
	_release_any_focus()


func _on_rec_pressed() -> void:
	var car := _get_car()
	if car != null and car.has_method("toggle_recording_shortcut"):
		car.toggle_recording_shortcut()
	_release_any_focus()


func _on_tune_pressed() -> void:
	var car := _get_car()
	if car != null and car.has_method("toggle_tv_tune_overlay"):
		car.toggle_tv_tune_overlay()
	_release_any_focus()


func _on_scope_pressed() -> void:
	var car := _get_car()
	if car != null and car.has_method("toggle_telemetry_scope"):
		car.toggle_telemetry_scope()
	_release_any_focus()


func _on_fz_arrows_toggled(pressed: bool) -> void:
	if _syncing_ui:
		return
	var car := _get_car()
	if car != null and car.has_method("set_fz_arrows_enabled"):
		car.set_fz_arrows_enabled(pressed)
	_release_any_focus()


func _on_torque_bars_toggled(pressed: bool) -> void:
	if _syncing_ui:
		return
	var car := _get_car()
	if car != null and car.has_method("set_torque_bars_enabled"):
		car.set_torque_bars_enabled(pressed)
	_release_any_focus()


func _on_ellipses_toggled(pressed: bool) -> void:
	if _syncing_ui:
		return
	var car := _get_car()
	if car != null and "wheel_forces_vis" in car and car.wheel_forces_vis != null and car.wheel_forces_vis.has_method("set_friction_ellipses_enabled"):
		car.wheel_forces_vis.set_friction_ellipses_enabled(pressed)
	_release_any_focus()


func _on_skidpad_ring_toggled(pressed: bool) -> void:
	if _syncing_ui:
		return
	var trk := _get_track()
	if trk != null and trk.has_method("set_skidpad_ring_visible"):
		trk.set_skidpad_ring_visible(pressed)
	_release_any_focus()


func _on_racing_line_toggled(pressed: bool) -> void:
	if _syncing_ui:
		return
	var car := _get_car()
	if car != null and car.has_method("set_racing_line_enabled"):
		car.set_racing_line_enabled(pressed)
	_release_any_focus()


func _on_line_color_mode_selected(index: int) -> void:
	if _syncing_ui:
		return
	var trk := _get_track()
	if trk != null and "racing_line_vis" in trk and trk.racing_line_vis != null and trk.racing_line_vis.has_method("set_color_mode"):
		trk.racing_line_vis.set_color_mode(index)
	_release_any_focus()


func _on_line_markers_toggled(pressed: bool) -> void:
	if _syncing_ui:
		return
	var car := _get_car()
	if car != null and car.has_method("set_racing_line_markers_enabled"):
		car.set_racing_line_markers_enabled(pressed)
	var trk := _get_track()
	if trk != null and trk.has_method("set_track_markers_visible"):
		trk.set_track_markers_visible(pressed)
	_release_any_focus()


func _on_ghost_mode_selected(index: int) -> void:
	if _syncing_ui:
		return
	var car := _get_car()
	if car != null and car.has_method("set_ghost_mode"):
		car.set_ghost_mode(index)
	var trk := _get_track()
	if trk != null and "dil_session" in trk and trk.dil_session != null and trk.dil_session.has_method("set_ghost_source"):
		if index >= 0 and index < GHOST_SOURCE_KEYS.size():
			trk.dil_session.set_ghost_source(GHOST_SOURCE_KEYS[index])
	_release_any_focus()


func _on_ghost_vis_toggled(pressed: bool) -> void:
	if _syncing_ui:
		return
	var trk := _get_track()
	if trk != null and "dil_session" in trk and trk.dil_session != null and trk.dil_session.has_method("set_ghost_visible"):
		trk.dil_session.set_ghost_visible(pressed)
	_release_any_focus()


func _on_ghost_opacity_changed(val: float) -> void:
	if _lbl_ghost_opacity != null:
		_lbl_ghost_opacity.text = "%d%%" % int(roundf(val * 100.0))
	if _syncing_ui:
		return
	var trk := _get_track()
	if trk != null and "dil_session" in trk and trk.dil_session != null and trk.dil_session.has_method("set_ghost_opacity"):
		trk.dil_session.set_ghost_opacity(val)
	_release_any_focus()


func _on_pace_rabbit_selected(index: int) -> void:
	if _syncing_ui or index < 0 or index >= PACE_RABBIT_OFFSETS.size():
		return
	var trk := _get_track()
	if trk != null and "dil_session" in trk and trk.dil_session != null and trk.dil_session.has_method("set_pace_rabbit_offset"):
		trk.dil_session.set_pace_rabbit_offset(PACE_RABBIT_OFFSETS[index])
	_release_any_focus()


func _on_dil_audio_toggled(pressed: bool) -> void:
	if _syncing_ui:
		return
	var car := _get_car()
	if is_instance_valid(car) and car.has_method("set_dil_audio_enabled"):
		car.set_dil_audio_enabled(pressed)
	_release_any_focus()


func _on_audio_profile_selected(index: int) -> void:
	if _syncing_ui:
		return
	var car := _get_car()
	if is_instance_valid(car) and "dil_audio" in car and car.dil_audio != null and car.dil_audio.has_method("set_sound_profile"):
		car.dil_audio.set_sound_profile(index)
	_release_any_focus()


func _on_audio_gains_changed(_val: float) -> void:
	var m_db := _sld_master_db.value if _sld_master_db != null else 0.0
	var e_db := _sld_engine_db.value if _sld_engine_db != null else 0.0
	var t_db := _sld_tire_db.value if _sld_tire_db != null else 0.0
	if _lbl_master_db != null:
		_lbl_master_db.text = "%+.1f dB" % m_db
	if _lbl_engine_db != null:
		_lbl_engine_db.text = "%+.1f dB" % e_db
	if _lbl_tire_db != null:
		_lbl_tire_db.text = "%+.1f dB" % t_db
	if _syncing_ui:
		return
	var car := _get_car()
	if is_instance_valid(car) and "dil_audio" in car and car.dil_audio != null and car.dil_audio.has_method("set_volumes"):
		car.dil_audio.set_volumes(m_db, e_db, t_db)
	_release_any_focus()


func _on_test_rev_pressed() -> void:
	var car := _get_car()
	if is_instance_valid(car) and "dil_audio" in car and car.dil_audio != null and car.dil_audio.has_method("trigger_test_rev"):
		car.dil_audio.trigger_test_rev()
	_release_any_focus()


func _on_ctrl_param_changed(key: String, val: float, fmt: String) -> void:
	if _ctrl_labels.has(key):
		(_ctrl_labels[key] as Label).text = fmt % val
	if _syncing_ui:
		return
	var car := _get_car()
	if car == null:
		return
	if "input_conditioner" in car and car.input_conditioner != null:
		if key in car.input_conditioner:
			car.input_conditioner.set(key, val)
	if car.has_method("set_live_gamepad_param"):
		var gp_key := key
		if key == "steer_deadzone":
			gp_key = "steering_deadzone"
		elif key == "steer_gamma":
			gp_key = "steering_gamma"
		elif key == "steer_speed_sensitivity":
			gp_key = "speed_sensitivity"
		elif key == "rumble_intensity":
			gp_key = "rumble_effects"
		car.set_live_gamepad_param(gp_key, val)
	_release_any_focus()


func _on_quick_spawn_pressed(preset_name: String) -> void:
	var car := _get_car()
	var sim_node := _get_sim()
	if sim_node != null and sim_node.has_method("spawn_test_zone"):
		sim_node.spawn_test_zone(preset_name, car)
		_release_any_focus()
		return
	var trk := _get_track()
	if car == null or trk == null or not trk.has_method("get_quick_spawn_transform"):
		return
	var xf: Transform3D = trk.get_quick_spawn_transform(preset_name)
	if car.has_method("set_origin"):
		car.set_origin(xf, true)
	_release_any_focus()


func _on_reset_cones_pressed() -> void:
	var trk := _get_track()
	if trk != null and trk.has_method("reset_cones_only"):
		trk.reset_cones_only()
	_release_any_focus()


func _on_reset_lap_pressed() -> void:
	var trk := _get_track()
	if trk != null:
		if "dil_session" in trk and trk.dil_session != null:
			trk.dil_session.reset_session(true)
		if trk.has_method("reset_skidpad_stats"):
			trk.reset_skidpad_stats()
	_release_any_focus()


func _update_pin_status_label() -> void:
	if _pin_status_lbl == null:
		return
	var sim_node := _get_sim()
	if sim_node != null and sim_node.has_method("get_practice_pin_info"):
		var parts: Array[String] = []
		for i in range(3):
			var info: Dictionary = sim_node.get_practice_pin_info(i)
			if bool(info.get("set", false)):
				var p: Vector3 = info.get("position", Vector3.ZERO)
				parts.append("P%d:(%.0f,%.0f)" % [i + 1, p.x, p.z])
			else:
				parts.append("P%d:--" % (i + 1))
		_pin_status_lbl.text = "Pins [G/B=Pin1]: " + "  ".join(parts)
	elif _has_practice_pin:
		var p := _practice_pin_xf.origin
		_pin_status_lbl.text = "Practice Pin: (%.1f, %.1f, %.1f)" % [p.x, p.y, p.z]


func _on_pin_set_pressed(slot: int) -> void:
	var car := _get_car()
	var sim_node := _get_sim()
	if sim_node != null and sim_node.has_method("set_practice_pin"):
		sim_node.set_practice_pin(slot, car)
	elif slot == 0 and car != null and car is Node3D:
		_practice_pin_xf = (car as Node3D).global_transform
		_has_practice_pin = true
	_update_pin_status_label()
	_release_any_focus()


func _on_pin_go_pressed(slot: int) -> void:
	var car := _get_car()
	var sim_node := _get_sim()
	if sim_node != null and sim_node.has_method("teleport_to_pin"):
		sim_node.teleport_to_pin(slot, car)
	elif slot == 0 and _has_practice_pin and car != null and car.has_method("set_origin"):
		car.set_origin(_practice_pin_xf, true)
	_release_any_focus()


func _on_drop_pin_pressed() -> void:
	_on_pin_set_pressed(0)


func _on_tp_pin_pressed() -> void:
	_on_pin_go_pressed(0)


func _on_rewind_pressed() -> void:
	var car := _get_car()
	if car != null:
		_rewind_buf.rewind_seconds(3.0, car)
	_release_any_focus()


func _on_time_scale_selected(index: int) -> void:
	if index < 0 or index >= TIME_SCALE_VALUES.size():
		return
	var sim_node := _get_sim()
	if sim_node != null and sim_node.has_method("set_time_scale"):
		sim_node.set_time_scale(TIME_SCALE_VALUES[index])
	else:
		Engine.time_scale = TIME_SCALE_VALUES[index]
	_release_any_focus()


func _cycle_time_scale() -> void:
	if _time_scale_opt == null:
		return
	var nxt := (_time_scale_opt.selected + 1) % TIME_SCALE_VALUES.size()
	_time_scale_opt.select(nxt)
	_on_time_scale_selected(nxt)


func _on_track_conditions_changed() -> void:
	var mu_val := _sld_mu_scale.value if _sld_mu_scale != null else 1.0
	var aero_val := _sld_aero_scale.value if _sld_aero_scale != null else 1.0
	var ts_idx := _tire_state_opt.selected if (_tire_state_opt != null and _tire_state_opt.selected >= 0) else 1
	var ts_key := TIRE_STATE_KEYS[clampi(ts_idx, 0, TIRE_STATE_KEYS.size() - 1)]
	var split_mu := _chk_split_mu.button_pressed if _chk_split_mu != null else false
	if _lbl_mu_scale != null:
		_lbl_mu_scale.text = "%.2f" % mu_val
	if _lbl_aero_scale != null:
		_lbl_aero_scale.text = "%.2fx" % aero_val
	if _syncing_ui:
		return
	var car := _get_car()
	if car != null:
		if car.has_method("apply_track_conditions"):
			car.apply_track_conditions({
				"mu_scale": mu_val,
				"tire_temp_preset": ts_key,
				"aero_scale": aero_val,
				"split_mu_test_enabled": split_mu,
			})
		elif car.has_method("set_track_surface_mu"):
			car.set_track_surface_mu(mu_val)
	_release_any_focus()


func _on_csv_rate_selected(index: int) -> void:
	if index >= 0 and index < CSV_RATE_VALUES.size() and _csv_logger.has_method("set_sample_rate_hz"):
		var hz: int = int(_csv_logger.set_sample_rate_hz(CSV_RATE_VALUES[index]))
		if _csv_status_lbl != null:
			_csv_status_lbl.text = "CSV Rate: %d Hz" % hz
	_release_any_focus()


func _on_auto_csv_toggled(on: bool) -> void:
	if _csv_logger.has_method("set_auto_record_laps"):
		_csv_logger.set_auto_record_laps(on)
	else:
		_csv_logger.auto_record_laps = on
	_release_any_focus()


func _on_export_session_pressed() -> void:
	var trk := _get_track()
	var dil: Variant = trk.get("dil_session") if trk != null else null
	if dil != null and dil.has_method("export_session_summary"):
		var out_path := String(dil.export_session_summary())
		if _csv_status_lbl != null:
			_csv_status_lbl.text = "Session Exported: %s" % (out_path.get_file() if not out_path.is_empty() else "failed")
	_release_any_focus()


func _on_open_logs_folder_pressed() -> void:
	if _csv_logger.has_method("open_logs_folder"):
		var folder := String(_csv_logger.open_logs_folder())
		if _csv_status_lbl != null:
			_csv_status_lbl.text = "Logs: %s" % folder
	_release_any_focus()


func _on_csv_rec_pressed() -> void:
	var trk := _get_track()
	var car := _get_car()
	var t_name: String = String(trk.get("last_loaded_track_path")).get_file().get_basename() if (trk != null and "last_loaded_track_path" in trk) else "track"
	var m_name := "AUTOCROSS"
	if car != null and car.has_method("get_sim_drive_mode_name"):
		m_name = String(car.get_sim_drive_mode_name())
	_csv_logger.toggle_recording(t_name, m_name)
	_release_any_focus()


func _on_grip_profile_selected(index: int) -> void:
	if _syncing_ui:
		return
	var car := _get_car()
	if is_instance_valid(car) and car.has_method("set_track_grip_profile"):
		car.set_track_grip_profile(index)
	_release_any_focus()


func _on_road_surface_changed(_idx: int = 0) -> void:
	if _syncing_ui:
		return
	var sm := _road_surf_opt.selected if _road_surf_opt != null else 0
	var rg := _sld_rough.value if _sld_rough != null else 1.0
	var bm := _bump_mode_opt.selected if _bump_mode_opt != null else 0
	var bh := _sld_bump_h.value if _sld_bump_h != null else 25.0
	var bs := _sld_bump_sp.value if _sld_bump_sp != null else 35.0
	if _lbl_rough != null:
		_lbl_rough.text = "%.1fx" % rg
	if _lbl_bump_h != null:
		_lbl_bump_h.text = "%dmm" % int(roundf(bh))
	if _lbl_bump_sp != null:
		_lbl_bump_sp.text = "%dm" % int(roundf(bs))
	var car := _get_car()
	if car != null and car.has_method("set_road_surface_config"):
		car.set_road_surface_config(sm, rg, bm, bh, bs)
	_release_any_focus()


func _on_haptics_toggled(pressed: bool) -> void:
	if _syncing_ui:
		return
	var car := _get_car()
	if car != null and car.has_method("set_haptics_enabled"):
		car.set_haptics_enabled(pressed)
	_release_any_focus()


func _on_input_profile_selected(index: int) -> void:
	if _syncing_ui:
		return
	var car := _get_car()
	if car != null and car.has_method("set_input_profile"):
		car.set_input_profile(index)
	_release_any_focus()


func _on_input_preset_pressed(preset_id: String) -> void:
	var car := _get_car()
	if car == null or not ("input_conditioner" in car) or car.input_conditioner == null:
		return
	if car.input_conditioner.has_method("apply_input_preset"):
		var res: Dictionary = car.input_conditioner.apply_input_preset(preset_id)
		_syncing_ui = true
		if _input_profile_opt != null and res.has("profile"):
			var p_idx := int(res["profile"])
			if p_idx >= 0 and p_idx < _input_profile_opt.item_count:
				_input_profile_opt.select(p_idx)
		var fmt_map := {
			"steer_deadzone": "%.2f",
			"steer_gamma": "%.2f",
			"steer_speed_sensitivity": "%.2f",
			"throttle_gamma": "%.2f",
			"brake_gamma": "%.2f",
			"rumble_intensity": "%.2f",
		}
		for k in fmt_map.keys():
			if res.has(k) and _ctrl_sliders.has(k):
				var v := float(res[k])
				(_ctrl_sliders[k] as HSlider).value = v
				if _ctrl_labels.has(k):
					(_ctrl_labels[k] as Label).text = fmt_map[k] % v
		_syncing_ui = false
	_release_any_focus()
