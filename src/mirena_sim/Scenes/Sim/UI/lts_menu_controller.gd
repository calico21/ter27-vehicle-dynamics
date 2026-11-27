class_name LtsMenuController
extends VBoxContainer

## Controller for the 9th Options Menu tab ("LTS"):
##   - Solver selector: QSS + 14-DOF (Rapido) vs CasADi Driver Optimal (Realista) vs CasADi Theoretical Oracle
##   - Max Attack / Endurance Budget toggle, 22 km energy budget slider (2.8 - 6.0 kWh), cone margin slider (0.15 - 0.35 m)
##   - Non-blocking async subprocess runner (/home/alexr/project_gp_env/bin/python3 -u) with duplicate-run guard & OS.kill cancel
##   - Integrated LtsLivePlot (Convergence/Pareto/Benchmark + Distance-domain Delta-t(s), v(s), Throttle/Brake with clickable corners)
##   - Corner Loop ("Practicar esta curva") flying teleport 40 m before s_brake at optimal entry speed + Auto-Repeat toggle
##   - Compact Lap Library table with 1-click Ghost / Pace Rabbit (+0.0 / +0.5 / +1.0 s) loading.

const LtsLivePlotScript = preload("res://Scenes/Sim/UI/LtsLivePlot.gd")
const RacingLineVisScript = preload("res://Scenes/Track/RacingLineVisualizer.gd")

const CASADI_PYTHON_BIN := "/home/alexr/project_gp_env/bin/python3"
const LAP_LIBRARY_PATH := "user://lap_library.json"

const SOLVER_LABELS: Array[String] = [
	"QSS + 14-DOF (Rapido)",
	"CasADi Driver Optimal (Realista)",
	"CasADi Theoretical Oracle",
]

const TASK_LABELS: Array[String] = [
	"FSG 2023 Benchmark (Open vs Auto vs Eff)",
	"Optimize Skidpad Map",
	"Optimize FSG 2023 Autocross",
	"Optimize FSG 2023 Efficiency (Pareto)",
]

const TRIAL_OPTIONS: Array[int] = [8, 16, 24]
const PACE_RABBIT_LABELS: Array[String] = ["+0.0 s", "+0.5 s", "+1.0 s"]
const PACE_RABBIT_OFFSETS: Array[float] = [0.0, 0.5, 1.0]

const GRIP_PROFILE_LABELS: Array[String] = [
	"FSG_RUBBERED",
	"FSG_REAL_TRACK",
	"TTC_LAB_PEAK",
	"CUSTOM_CONDITIONS",
]
const GRIP_PROFILE_CLI_KEYS: Array[String] = [
	"fsg_rubbered",
	"fsg_real",
	"ttc_lab",
	"custom",
]
const GRIP_PROFILE_MU_DEFAULTS: Array[float] = [0.88, 0.80, 1.00, 0.88]
const GRIP_PROFILE_TEMP_DEFAULTS: Array[float] = [45.0, 40.0, 70.0, 45.0]
const TRACKSIDE_SETUP_LABELS: Array[String] = [
	"Fixed Base",
	"Co-Optimize in CasADi",
]
const WSL_PATH_FLAGS: Array[String] = [
	"--track-file",
	"--output",
	"--progress-file",
	"--save-14dof-csv",
]

# Solver & QSS controls
var _solver_opt: OptionButton = null
var _qss_box: VBoxContainer = null
var _casadi_box: VBoxContainer = null
var _task_opt: OptionButton = null
var _trials_opt: OptionButton = null

# CasADi controls
var _grip_profile_opt: OptionButton = null
var _sld_mu_scale: HSlider = null
var _lbl_mu_scale: Label = null
var _sld_temp_c: HSlider = null
var _lbl_temp_c: Label = null
var _syncing_conditions: bool = false
var _chk_endurance: CheckButton = null
var _sld_budget: HSlider = null
var _lbl_budget: Label = null
var _sld_cone_margin: HSlider = null
var _lbl_cone_margin: Label = null
var _trackside_setup_opt: OptionButton = null

# Action & status controls
var _btn_run: Button = null
var _btn_stop: Button = null
var _btn_load: Button = null
var _status_lbl: Label = null
var _best_lbl: Label = null
var _setup_rec_lbl: Label = null
var _plot: LtsLivePlotScript = null
var _poll_timer: Timer = null

# Corner Loop controls
var _corner_opt: OptionButton = null
var _btn_practice_corner: Button = null
var _chk_corner_loop: CheckButton = null
var _corner_info_lbl: Label = null
var _corners: Array = []
var _active_corner_idx: int = -1
var _corner_loop_cooldown: float = 0.0

# Lap Library controls
var _lib_table_lbl: Label = null
var _lib_opt: OptionButton = null
var _lib_pace_opt: OptionButton = null
var _btn_lib_ghost: Button = null
var _btn_lib_cmp: Button = null
var _btn_lib_save_driver: Button = null
var _lap_library: Array = []

# Async process state
var _pid: int = -1
var _run_is_casadi: bool = false
var _last_run_is_wsl: bool = false
var _progress_path: String = ""
var _casadi_log_path: String = ""
var _casadi_out_json: String = ""
var _read_byte_offset: int = 0
var _partial_buf: String = ""
var _progress_byte_offset: int = 0
var _progress_partial_buf: String = ""
var _casadi_stage_text: String = "Idle"
var _casadi_iter_count: int = 0
var _casadi_last_error: String = ""

# Loaded telemetry reference
var _ref_data: Dictionary = {}
var _ref_json_path: String = ""


func _ready() -> void:
	_build_ui()
	_load_lap_library()
	_discover_builtin_laps()
	_refresh_lap_library_ui()
	_auto_load_default_telemetry()


func _exit_tree() -> void:
	stop_process()


func _make_mono_font() -> SystemFont:
	var f := SystemFont.new()
	f.font_names = PackedStringArray(["DejaVu Sans Mono", "Liberation Mono", "Monospace"])
	return f


func _build_ui() -> void:
	add_theme_constant_override("separation", 5)
	var mono := _make_mono_font()

	# 1. Solver Selector
	var sol_row := HBoxContainer.new()
	sol_row.add_theme_constant_override("separation", 6)
	add_child(sol_row)
	var s_lbl := Label.new()
	s_lbl.text = "Solver"
	s_lbl.custom_minimum_size = Vector2(54, 0)
	sol_row.add_child(s_lbl)
	_solver_opt = OptionButton.new()
	_solver_opt.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_solver_opt.focus_mode = Control.FOCUS_NONE
	_solver_opt.clip_text = true
	for i in range(SOLVER_LABELS.size()):
		_solver_opt.add_item(SOLVER_LABELS[i], i)
	_solver_opt.select(1) # Default: CasADi Driver Optimal (Realista)
	_solver_opt.item_selected.connect(_on_solver_selected)
	sol_row.add_child(_solver_opt)

	# 2a. QSS Legacy Task Controls
	_qss_box = VBoxContainer.new()
	_qss_box.add_theme_constant_override("separation", 4)
	add_child(_qss_box)

	var task_row := HBoxContainer.new()
	task_row.add_theme_constant_override("separation", 6)
	_qss_box.add_child(task_row)
	var t_lbl := Label.new()
	t_lbl.text = "Task"
	t_lbl.custom_minimum_size = Vector2(54, 0)
	task_row.add_child(t_lbl)
	_task_opt = OptionButton.new()
	_task_opt.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_task_opt.focus_mode = Control.FOCUS_NONE
	_task_opt.clip_text = true
	for i in range(TASK_LABELS.size()):
		_task_opt.add_item(TASK_LABELS[i], i)
	_task_opt.item_selected.connect(func(_idx: int): _task_opt.release_focus())
	task_row.add_child(_task_opt)

	var tr_row := HBoxContainer.new()
	tr_row.add_theme_constant_override("separation", 6)
	_qss_box.add_child(tr_row)
	var tr_lbl := Label.new()
	tr_lbl.text = "Trials"
	tr_lbl.custom_minimum_size = Vector2(54, 0)
	tr_row.add_child(tr_lbl)
	_trials_opt = OptionButton.new()
	_trials_opt.focus_mode = Control.FOCUS_NONE
	for i in range(TRIAL_OPTIONS.size()):
		_trials_opt.add_item(str(TRIAL_OPTIONS[i]), TRIAL_OPTIONS[i])
	_trials_opt.select(1)
	_trials_opt.item_selected.connect(func(_idx: int): _trials_opt.release_focus())
	tr_row.add_child(_trials_opt)

	# 2b. CasADi OCP Controls
	_casadi_box = VBoxContainer.new()
	_casadi_box.add_theme_constant_override("separation", 4)
	add_child(_casadi_box)

	var gp_row := HBoxContainer.new()
	gp_row.add_theme_constant_override("separation", 6)
	_casadi_box.add_child(gp_row)
	var gp_lbl := Label.new()
	gp_lbl.text = "Grip Profile"
	gp_lbl.custom_minimum_size = Vector2(96, 0)
	gp_lbl.add_theme_font_size_override("font_size", 11)
	gp_row.add_child(gp_lbl)
	_grip_profile_opt = OptionButton.new()
	_grip_profile_opt.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_grip_profile_opt.focus_mode = Control.FOCUS_NONE
	_grip_profile_opt.clip_text = true
	for i in range(GRIP_PROFILE_LABELS.size()):
		_grip_profile_opt.add_item(GRIP_PROFILE_LABELS[i], i)
	_grip_profile_opt.select(0) # Default: FSG_RUBBERED
	_grip_profile_opt.item_selected.connect(_on_grip_profile_selected)
	gp_row.add_child(_grip_profile_opt)

	var mu_row := HBoxContainer.new()
	mu_row.add_theme_constant_override("separation", 6)
	_casadi_box.add_child(mu_row)
	var mu_lbl := Label.new()
	mu_lbl.text = "Grip Scale (mu)"
	mu_lbl.custom_minimum_size = Vector2(96, 0)
	mu_lbl.add_theme_font_size_override("font_size", 11)
	mu_row.add_child(mu_lbl)
	_sld_mu_scale = HSlider.new()
	_sld_mu_scale.min_value = 0.60
	_sld_mu_scale.max_value = 1.10
	_sld_mu_scale.step = 0.01
	_sld_mu_scale.value = 0.88
	_sld_mu_scale.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_sld_mu_scale.focus_mode = Control.FOCUS_NONE
	_sld_mu_scale.value_changed.connect(_on_mu_scale_changed)
	mu_row.add_child(_sld_mu_scale)
	_lbl_mu_scale = Label.new()
	_lbl_mu_scale.add_theme_font_override("font", mono)
	_lbl_mu_scale.add_theme_font_size_override("font_size", 11)
	_lbl_mu_scale.custom_minimum_size = Vector2(62, 0)
	_lbl_mu_scale.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	_lbl_mu_scale.text = "0.88"
	mu_row.add_child(_lbl_mu_scale)

	var tmp_row := HBoxContainer.new()
	tmp_row.add_theme_constant_override("separation", 6)
	_casadi_box.add_child(tmp_row)
	var tmp_lbl := Label.new()
	tmp_lbl.text = "Track Temp"
	tmp_lbl.custom_minimum_size = Vector2(96, 0)
	tmp_lbl.add_theme_font_size_override("font_size", 11)
	tmp_row.add_child(tmp_lbl)
	_sld_temp_c = HSlider.new()
	_sld_temp_c.min_value = 10.0
	_sld_temp_c.max_value = 95.0
	_sld_temp_c.step = 1.0
	_sld_temp_c.value = 45.0
	_sld_temp_c.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_sld_temp_c.focus_mode = Control.FOCUS_NONE
	_sld_temp_c.value_changed.connect(_on_temp_c_changed)
	tmp_row.add_child(_sld_temp_c)
	_lbl_temp_c = Label.new()
	_lbl_temp_c.add_theme_font_override("font", mono)
	_lbl_temp_c.add_theme_font_size_override("font_size", 11)
	_lbl_temp_c.custom_minimum_size = Vector2(62, 0)
	_lbl_temp_c.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	_lbl_temp_c.text = "45 C"
	tmp_row.add_child(_lbl_temp_c)

	_chk_endurance = CheckButton.new()
	_chk_endurance.text = "Endurance Budget (OFF = Max Attack)"
	_chk_endurance.focus_mode = Control.FOCUS_NONE
	_chk_endurance.button_pressed = false
	_chk_endurance.toggled.connect(_on_endurance_toggled)
	_casadi_box.add_child(_chk_endurance)

	var bud_row := HBoxContainer.new()
	bud_row.add_theme_constant_override("separation", 6)
	_casadi_box.add_child(bud_row)
	var b_lbl := Label.new()
	b_lbl.text = "22 km Budget"
	b_lbl.custom_minimum_size = Vector2(96, 0)
	b_lbl.add_theme_font_size_override("font_size", 11)
	bud_row.add_child(b_lbl)
	_sld_budget = HSlider.new()
	_sld_budget.min_value = 2.8
	_sld_budget.max_value = 6.0
	_sld_budget.step = 0.1
	_sld_budget.value = 3.5
	_sld_budget.editable = false
	_sld_budget.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_sld_budget.focus_mode = Control.FOCUS_NONE
	_sld_budget.value_changed.connect(func(v: float): _lbl_budget.text = "%.1f kWh" % v)
	bud_row.add_child(_sld_budget)
	_lbl_budget = Label.new()
	_lbl_budget.add_theme_font_override("font", mono)
	_lbl_budget.add_theme_font_size_override("font_size", 11)
	_lbl_budget.custom_minimum_size = Vector2(62, 0)
	_lbl_budget.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	_lbl_budget.text = "3.5 kWh"
	bud_row.add_child(_lbl_budget)

	var cm_row := HBoxContainer.new()
	cm_row.add_theme_constant_override("separation", 6)
	_casadi_box.add_child(cm_row)
	var cm_lbl := Label.new()
	cm_lbl.text = "Cone Margin"
	cm_lbl.custom_minimum_size = Vector2(96, 0)
	cm_lbl.add_theme_font_size_override("font_size", 11)
	cm_row.add_child(cm_lbl)
	_sld_cone_margin = HSlider.new()
	_sld_cone_margin.min_value = 0.15
	_sld_cone_margin.max_value = 0.35
	_sld_cone_margin.step = 0.01
	_sld_cone_margin.value = 0.25
	_sld_cone_margin.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_sld_cone_margin.focus_mode = Control.FOCUS_NONE
	_sld_cone_margin.value_changed.connect(func(v: float): _lbl_cone_margin.text = "%.2f m" % v)
	cm_row.add_child(_sld_cone_margin)
	_lbl_cone_margin = Label.new()
	_lbl_cone_margin.add_theme_font_override("font", mono)
	_lbl_cone_margin.add_theme_font_size_override("font_size", 11)
	_lbl_cone_margin.custom_minimum_size = Vector2(62, 0)
	_lbl_cone_margin.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	_lbl_cone_margin.text = "0.25 m"
	cm_row.add_child(_lbl_cone_margin)

	var ts_row := HBoxContainer.new()
	ts_row.add_theme_constant_override("separation", 6)
	_casadi_box.add_child(ts_row)
	var ts_lbl := Label.new()
	ts_lbl.text = "Trackside Setup"
	ts_lbl.custom_minimum_size = Vector2(96, 0)
	ts_lbl.add_theme_font_size_override("font_size", 11)
	ts_row.add_child(ts_lbl)
	_trackside_setup_opt = OptionButton.new()
	_trackside_setup_opt.name = "TracksideSetupOpt"
	_trackside_setup_opt.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_trackside_setup_opt.focus_mode = Control.FOCUS_NONE
	_trackside_setup_opt.clip_text = true
	for i in range(TRACKSIDE_SETUP_LABELS.size()):
		_trackside_setup_opt.add_item(TRACKSIDE_SETUP_LABELS[i], i)
	_trackside_setup_opt.select(0) # Default: Fixed Base
	_trackside_setup_opt.item_selected.connect(func(_idx: int): _trackside_setup_opt.release_focus())
	ts_row.add_child(_trackside_setup_opt)

	# 3. Run / Cancel / Load Row
	var run_row := HBoxContainer.new()
	run_row.add_theme_constant_override("separation", 6)
	add_child(run_row)
	_btn_run = Button.new()
	_btn_run.text = "Run Solver"
	_btn_run.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_btn_run.focus_mode = Control.FOCUS_NONE
	_btn_run.pressed.connect(_on_run_pressed)
	run_row.add_child(_btn_run)

	_btn_stop = Button.new()
	_btn_stop.text = "Cancelar"
	_btn_stop.disabled = true
	_btn_stop.focus_mode = Control.FOCUS_NONE
	_btn_stop.pressed.connect(_on_stop_pressed)
	run_row.add_child(_btn_stop)

	_btn_load = Button.new()
	_btn_load.text = "Load to Track"
	_btn_load.focus_mode = Control.FOCUS_NONE
	_btn_load.pressed.connect(_on_load_result_pressed)
	run_row.add_child(_btn_load)

	# 4. Integrated Telemetry Plot
	_plot = LtsLivePlotScript.new()
	_plot.focus_mode = Control.FOCUS_NONE
	_plot.corner_clicked.connect(_on_plot_corner_clicked)
	add_child(_plot)

	_status_lbl = Label.new()
	_status_lbl.add_theme_font_size_override("font_size", 11)
	_status_lbl.text = "Idle | Ready"
	add_child(_status_lbl)

	_best_lbl = Label.new()
	_best_lbl.add_theme_font_override("font", mono)
	_best_lbl.add_theme_font_size_override("font_size", 10)
	_best_lbl.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_best_lbl.text = "Ref: --"
	add_child(_best_lbl)

	_setup_rec_lbl = Label.new()
	_setup_rec_lbl.name = "RecommendedSetupLabel"
	_setup_rec_lbl.add_theme_font_override("font", mono)
	_setup_rec_lbl.add_theme_font_size_override("font_size", 10)
	_setup_rec_lbl.add_theme_color_override("font_color", Color(0.76, 0.92, 0.80, 1.0))
	_setup_rec_lbl.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_setup_rec_lbl.text = "Setup: Fixed Base (select Co-Optimize in CasADi for recommendations)"
	add_child(_setup_rec_lbl)

	# 5. Corner Loop ("Practicar esta curva")
	add_child(HSeparator.new())
	var cl_hdr := Label.new()
	cl_hdr.text = "CORNER LOOP (PRACTICAR ESTA CURVA)"
	cl_hdr.add_theme_font_size_override("font_size", 11)
	cl_hdr.add_theme_color_override("font_color", Color(0.74, 0.90, 1.0, 1.0))
	add_child(cl_hdr)

	var cl_row := HBoxContainer.new()
	cl_row.add_theme_constant_override("separation", 6)
	add_child(cl_row)
	_corner_opt = OptionButton.new()
	_corner_opt.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_corner_opt.focus_mode = Control.FOCUS_NONE
	_corner_opt.clip_text = true
	_corner_opt.item_selected.connect(_on_corner_selected)
	cl_row.add_child(_corner_opt)

	_btn_practice_corner = Button.new()
	_btn_practice_corner.text = "Practicar esta curva"
	_btn_practice_corner.focus_mode = Control.FOCUS_NONE
	_btn_practice_corner.pressed.connect(_on_practice_corner_pressed)
	cl_row.add_child(_btn_practice_corner)

	_chk_corner_loop = CheckButton.new()
	_chk_corner_loop.text = "Auto-Repeat Corner Loop (40m before s_brake)"
	_chk_corner_loop.focus_mode = Control.FOCUS_NONE
	add_child(_chk_corner_loop)

	_corner_info_lbl = Label.new()
	_corner_info_lbl.add_theme_font_override("font", mono)
	_corner_info_lbl.add_theme_font_size_override("font_size", 10)
	_corner_info_lbl.text = "Corner: -- (click plot or select above)"
	add_child(_corner_info_lbl)

	# 6. Compact Lap Library
	add_child(HSeparator.new())
	var lib_hdr := Label.new()
	lib_hdr.text = "LAP LIBRARY & GHOST / PACE RABBIT"
	lib_hdr.add_theme_font_size_override("font_size", 11)
	lib_hdr.add_theme_color_override("font_color", Color(0.74, 0.90, 1.0, 1.0))
	add_child(lib_hdr)

	_lib_table_lbl = Label.new()
	_lib_table_lbl.add_theme_font_override("font", mono)
	_lib_table_lbl.add_theme_font_size_override("font_size", 10)
	_lib_table_lbl.text = "Label              | Tiempo | v_max | ay_max | E_22km | Driv | Conos\n(no saved laps)"
	add_child(_lib_table_lbl)

	var lib_row1 := HBoxContainer.new()
	lib_row1.add_theme_constant_override("separation", 4)
	add_child(lib_row1)
	_lib_opt = OptionButton.new()
	_lib_opt.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_lib_opt.focus_mode = Control.FOCUS_NONE
	_lib_opt.clip_text = true
	lib_row1.add_child(_lib_opt)

	_lib_pace_opt = OptionButton.new()
	_lib_pace_opt.focus_mode = Control.FOCUS_NONE
	for i in range(PACE_RABBIT_LABELS.size()):
		_lib_pace_opt.add_item(PACE_RABBIT_LABELS[i], i)
	_lib_pace_opt.select(0)
	lib_row1.add_child(_lib_pace_opt)

	var lib_row2 := HBoxContainer.new()
	lib_row2.add_theme_constant_override("separation", 4)
	add_child(lib_row2)

	_btn_lib_ghost = Button.new()
	_btn_lib_ghost.text = "Load Ghost"
	_btn_lib_ghost.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_btn_lib_ghost.focus_mode = Control.FOCUS_NONE
	_btn_lib_ghost.pressed.connect(_on_lib_load_ghost_pressed)
	lib_row2.add_child(_btn_lib_ghost)

	_btn_lib_cmp = Button.new()
	_btn_lib_cmp.text = "Compare Plot"
	_btn_lib_cmp.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_btn_lib_cmp.focus_mode = Control.FOCUS_NONE
	_btn_lib_cmp.pressed.connect(_on_lib_compare_pressed)
	lib_row2.add_child(_btn_lib_cmp)

	_btn_lib_save_driver = Button.new()
	_btn_lib_save_driver.text = "Save Driver Best"
	_btn_lib_save_driver.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_btn_lib_save_driver.focus_mode = Control.FOCUS_NONE
	_btn_lib_save_driver.pressed.connect(_on_save_driver_best_pressed)
	lib_row2.add_child(_btn_lib_save_driver)

	# Poll timer (non-blocking 5 Hz tail)
	_poll_timer = Timer.new()
	_poll_timer.wait_time = 0.2
	_poll_timer.one_shot = false
	_poll_timer.timeout.connect(_on_poll_timeout)
	add_child(_poll_timer)

	_on_solver_selected(_solver_opt.selected)


func _physics_process(delta: float) -> void:
	if _chk_corner_loop == null or not _chk_corner_loop.button_pressed:
		return
	if _active_corner_idx < 0 or _active_corner_idx >= _corners.size():
		return
	if _corner_loop_cooldown > 0.0:
		_corner_loop_cooldown -= delta
		return
	var sim_node := get_node_or_null("/root/Sim")
	if sim_node == null or not ("track" in sim_node) or sim_node.track == null:
		return
	var dil: Variant = sim_node.track.get("dil_session")
	if dil == null or not ("_s_cur_m" in dil):
		return
	var c: Dictionary = _corners[_active_corner_idx]
	var s_exit := float(c.get("s_exit", float(c.get("s_apex", 0.0)) + 25.0))
	var s_cur := float(dil._s_cur_m)
	if s_cur >= s_exit + 15.0 and s_cur <= s_exit + 80.0:
		_corner_loop_cooldown = 1.2
		if sim_node.has_method("teleport_to_corner"):
			sim_node.teleport_to_corner(c)


func _get_sim_safe() -> Node:
	return get_node_or_null("/root/Sim") if is_inside_tree() else null


func _on_solver_selected(idx: int) -> void:
	if _solver_opt != null and _solver_opt.is_inside_tree():
		_solver_opt.release_focus()
	var is_qss := (idx == 0)
	if _qss_box != null:
		_qss_box.visible = is_qss
	if _casadi_box != null:
		_casadi_box.visible = not is_qss


func _on_endurance_toggled(on: bool) -> void:
	if _sld_budget != null:
		_sld_budget.editable = on


func _on_grip_profile_selected(idx: int) -> void:
	if _grip_profile_opt != null and _grip_profile_opt.is_inside_tree():
		_grip_profile_opt.release_focus()
	if idx < 0 or idx >= GRIP_PROFILE_LABELS.size():
		return
	if idx == 3: # CUSTOM_CONDITIONS keeps current slider values
		return
	_syncing_conditions = true
	if _sld_mu_scale != null:
		_sld_mu_scale.value = GRIP_PROFILE_MU_DEFAULTS[idx]
	if _lbl_mu_scale != null:
		_lbl_mu_scale.text = "%.2f" % GRIP_PROFILE_MU_DEFAULTS[idx]
	if _sld_temp_c != null:
		_sld_temp_c.value = GRIP_PROFILE_TEMP_DEFAULTS[idx]
	if _lbl_temp_c != null:
		_lbl_temp_c.text = "%d C" % int(roundf(GRIP_PROFILE_TEMP_DEFAULTS[idx]))
	_syncing_conditions = false


func _on_mu_scale_changed(v: float) -> void:
	if _lbl_mu_scale != null:
		_lbl_mu_scale.text = "%.2f" % v
	if not _syncing_conditions and _grip_profile_opt != null and _grip_profile_opt.selected != 3:
		_grip_profile_opt.select(3) # CUSTOM_CONDITIONS


func _on_temp_c_changed(v: float) -> void:
	if _lbl_temp_c != null:
		_lbl_temp_c.text = "%d C" % int(roundf(v))
	if not _syncing_conditions and _grip_profile_opt != null and _grip_profile_opt.selected != 3:
		_grip_profile_opt.select(3) # CUSTOM_CONDITIONS


static func _to_wsl_path(win_path: String) -> String:
	if win_path.is_empty():
		return ""
	var norm := win_path.replace("\\", "/")
	var lower := norm.to_lower()
	# UNC WSL paths: //wsl.localhost/<distro>/home/... or //wsl$/<distro>/home/...
	var unc_prefix := ""
	if lower.begins_with("//wsl.localhost/"):
		unc_prefix = "//wsl.localhost/"
	elif lower.begins_with("//wsl$/"):
		unc_prefix = "//wsl$/"
	if not unc_prefix.is_empty():
		var after_prefix := norm.substr(unc_prefix.length())
		var slash_idx := after_prefix.find("/")
		if slash_idx == -1:
			return "/"
		return after_prefix.substr(slash_idx)
	# Already a POSIX path
	if norm.begins_with("/"):
		return norm
	# Windows drive letter path: C:/Users/... or C:\Users\...
	if norm.length() >= 2 and norm[1] == ":":
		var drv := norm[0].to_lower()
		var code := drv.unicode_at(0)
		if code >= 97 and code <= 122:
			var rest := norm.substr(2)
			if rest.is_empty():
				return "/mnt/" + drv
			if not rest.begins_with("/"):
				rest = "/" + rest
			return "/mnt/" + drv + rest
	return norm


func _resolve_python_command(script_path: String, args: Array, platform_override: String = "") -> Dictionary:
	var plat := platform_override if not platform_override.is_empty() else OS.get_name()
	if plat == "Windows":
		var repo_root := resolve_repo_root()
		var venv_py := repo_root.path_join(".venv/Scripts/python.exe")
		if FileAccess.file_exists(venv_py):
			var probe_out := []
			var rc := OS.execute(venv_py, PackedStringArray(["-c", "import casadi"]), probe_out, true)
			if rc == 0:
				var win_args := PackedStringArray(["-u", script_path])
				for a in args:
					win_args.append(str(a))
				return {"exe": venv_py, "args": win_args, "is_wsl": false}
		# Fallback to WSL python3 with converted POSIX paths
		var wsl_args := PackedStringArray(["-e", CASADI_PYTHON_BIN, "-u", _to_wsl_path(script_path)])
		var next_is_path := false
		for a in args:
			var a_str := str(a)
			if next_is_path:
				wsl_args.append(_to_wsl_path(a_str))
				next_is_path = false
			elif WSL_PATH_FLAGS.has(a_str):
				wsl_args.append(a_str)
				next_is_path = true
			else:
				var handled_eq := false
				for pf in WSL_PATH_FLAGS:
					var pref := pf + "="
					if a_str.begins_with(pref):
						wsl_args.append(pref + _to_wsl_path(a_str.substr(pref.length())))
						handled_eq = true
						break
				if not handled_eq:
					wsl_args.append(a_str)
		return {"exe": "wsl.exe", "args": wsl_args, "is_wsl": true}

	# Linux / POSIX
	var py_bin := CASADI_PYTHON_BIN
	if platform_override.is_empty() and not FileAccess.file_exists(py_bin):
		var fb := resolve_python_executable(resolve_repo_root())
		if not fb.is_empty():
			py_bin = fb
	var lin_args := PackedStringArray(["-u", script_path])
	for a in args:
		lin_args.append(str(a))
	return {"exe": py_bin, "args": lin_args, "is_wsl": false}


static func resolve_repo_root() -> String:
	var res_dir := ProjectSettings.globalize_path("res://").rstrip("/")
	var candidate := res_dir.path_join("../..").simplify_path()
	if DirAccess.dir_exists_absolute(candidate.path_join("vd_correlation_lab/optimization/lap_time_sim")):
		return candidate
	var fallback := res_dir.get_base_dir().get_base_dir()
	return fallback


static func resolve_python_executable(repo_root: String) -> String:
	var py_candidates: Array[String] = []
	var env_py := OS.get_environment("PYTHON").strip_edges()
	if not env_py.is_empty():
		py_candidates.append(env_py)
	py_candidates.append(CASADI_PYTHON_BIN)
	if not repo_root.is_empty():
		py_candidates.append(repo_root.path_join(".venv/bin/python"))
		py_candidates.append(repo_root.path_join("venv/bin/python"))
		py_candidates.append(repo_root.path_join(".venv/bin/python3"))
		py_candidates.append(repo_root.path_join("venv/bin/python3"))
	py_candidates.append_array(["/usr/bin/python3", "/usr/local/bin/python3", "/bin/python3", "python3"])
	for py_bin in py_candidates:
		if py_bin.begins_with("/") and not FileAccess.file_exists(py_bin):
			continue
		var out := []
		var rc := OS.execute(py_bin, ["-c", "import sys"], out, true)
		if rc == 0:
			return py_bin
	return ""


func _resolve_active_track_file(repo_root: String) -> String:
	var sim_node := get_node_or_null("/root/Sim")
	if sim_node != null and "track" in sim_node and sim_node.track != null:
		var lp := String(sim_node.track.get("last_loaded_track_path"))
		if not lp.is_empty():
			var gp := ProjectSettings.globalize_path(lp) if lp.begins_with("res://") or lp.begins_with("user://") else lp
			if FileAccess.file_exists(gp):
				return gp
	var def_fsg := repo_root.path_join("src/mirena_sim/TrackFiles/fsg_2023.json")
	if FileAccess.file_exists(def_fsg):
		return def_fsg
	return repo_root.path_join("data/tracks/fsg_2023.json")


func _resolve_active_grip_profile() -> String:
	if _grip_profile_opt != null and _grip_profile_opt.selected >= 0 and _grip_profile_opt.selected < GRIP_PROFILE_CLI_KEYS.size():
		return GRIP_PROFILE_CLI_KEYS[_grip_profile_opt.selected]
	return "fsg_rubbered"


func is_trackside_setup_coopt_enabled() -> bool:
	return _trackside_setup_opt != null and _trackside_setup_opt.selected == 1


static func format_recommended_setup(setup_d: Dictionary, delta_time_s: float = 0.0) -> String:
	if setup_d.is_empty():
		return "Setup: Fixed Base"
	var lltd_raw := float(setup_d.get("lltd", setup_d.get("lltd_f", setup_d.get("lltd_front", 0.455))))
	var lltd_pct := lltd_raw * 100.0 if lltd_raw <= 1.0 else lltd_raw
	var cam_f := float(setup_d.get("camber_f_deg", setup_d.get("camber_front_deg", -2.0)))
	var cam_r := float(setup_d.get("camber_r_deg", setup_d.get("camber_rear_deg", -1.5)))
	var toe_f := float(setup_d.get("toe_f_deg", setup_d.get("toe_front_deg", 0.0)))
	var toe_r := float(setup_d.get("toe_r_deg", setup_d.get("toe_rear_deg", 0.0)))
	var bb_raw := float(setup_d.get("brake_bias", setup_d.get("brake_bias_front", 0.58)))
	var bb_pct := bb_raw * 100.0 if bb_raw <= 1.0 else bb_raw
	var rake_mm := float(setup_d.get("rake_mm", 0.0))
	var dt_val := float(setup_d.get("delta_time_s", setup_d.get("delta_lap_time_s", delta_time_s)))
	return "Recommended Setup: LLTD %.1f%% | Cam F/R %+.2f/%+.2f deg | Toe F/R %+.2f/%+.2f deg | BB %.1f%% | Rake %+.1f mm | Delta %+.3fs" % [
		lltd_pct, cam_f, cam_r, toe_f, toe_r, bb_pct, rake_mm, dt_val
	]


func _update_recommended_setup_display(data: Dictionary) -> void:
	if _setup_rec_lbl == null:
		return
	var meta: Dictionary = data.get("meta", {}) if data.get("meta") is Dictionary else {}
	var setup_d: Dictionary = {}
	for key in ["recommended_setup", "optimal_setup", "trackside_setup"]:
		if data.get(key) is Dictionary and not (data[key] as Dictionary).is_empty():
			setup_d = data[key]
			break
		if meta.get(key) is Dictionary and not (meta[key] as Dictionary).is_empty():
			setup_d = meta[key]
			break
	if setup_d.is_empty():
		_setup_rec_lbl.text = "Setup: Fixed Base"
	else:
		var dt_s := float(meta.get("setup_delta_time_s", meta.get("delta_time_s", 0.0)))
		_setup_rec_lbl.text = format_recommended_setup(setup_d, dt_s)


func _on_run_pressed() -> void:
	_btn_run.release_focus()
	# Duplicate-run guard
	if _pid > 0 and OS.is_process_running(_pid):
		_status_lbl.text = "Solver already running (PID %d). Click Cancelar first." % _pid
		return

	var solver_idx := _solver_opt.selected if _solver_opt != null else 0
	var repo_root := resolve_repo_root()
	_read_byte_offset = 0
	_partial_buf = ""
	_progress_byte_offset = 0
	_progress_partial_buf = ""
	_casadi_last_error = ""

	if solver_idx == 0:
		_launch_qss_solver(repo_root)
	else:
		_launch_casadi_solver(repo_root, solver_idx)


func _launch_qss_solver(repo_root: String) -> void:
	_run_is_casadi = false
	var task_idx := _task_opt.selected if _task_opt != null else 0
	var n_trials := TRIAL_OPTIONS[_trials_opt.selected] if (_trials_opt != null and _trials_opt.selected >= 0) else 16
	_progress_path = OS.get_user_data_dir().path_join("lts_live_progress.jsonl")
	var pf := FileAccess.open(_progress_path, FileAccess.WRITE)
	if pf != null:
		pf.store_string("")
		pf.close()

	var script_path := ""
	var script_args: Array = []
	if task_idx == 0:
		_plot.reset_plot(LtsLivePlotScript.PlotMode.BENCHMARK)
		script_path = repo_root.path_join("vd_correlation_lab/optimization/lap_time_sim/run_fsg_2023_benchmark.py")
		script_args = ["--progress-file", _progress_path]
	else:
		var t_names := ["all", "skidpad", "autocross", "efficiency"]
		var mode := LtsLivePlotScript.PlotMode.PARETO if task_idx == 3 else LtsLivePlotScript.PlotMode.CONVERGENCE
		_plot.reset_plot(mode)
		script_path = repo_root.path_join("vd_correlation_lab/control/torque_vectoring/optimize_tv_maps.py")
		script_args = ["--task", t_names[task_idx], "--trials", str(n_trials), "--progress-file", _progress_path]

	if not FileAccess.file_exists(script_path):
		_status_lbl.text = "LTS script not found: %s" % script_path
		return

	var cmd := _resolve_python_command(script_path, script_args)
	var py_exe := String(cmd.get("exe", ""))
	var py_args: PackedStringArray = cmd.get("args", PackedStringArray())
	_last_run_is_wsl = bool(cmd.get("is_wsl", false))
	if py_exe.is_empty():
		_status_lbl.text = "Error: Python 3 interpreter not found"
		return

	_pid = OS.create_process(py_exe, py_args, false)
	if _pid <= 0:
		_status_lbl.text = "Error launching %s" % script_path.get_file()
		return
	_btn_run.disabled = true
	_btn_stop.disabled = false
	_status_lbl.text = "Running QSS %s (PID %d)..." % [script_path.get_file(), _pid]
	_poll_timer.start()


func _launch_casadi_solver(repo_root: String, solver_idx: int) -> void:
	_run_is_casadi = true
	var script_path := repo_root.path_join("vd_correlation_lab/optimization/lap_time_sim/solve_full_lap_ocp.py")
	if not FileAccess.file_exists(script_path):
		_status_lbl.text = "CasADi script not found: %s" % script_path.get_file()
		return

	var track_file := _resolve_active_track_file(repo_root)
	var track_stem := track_file.get_file().get_basename().to_lower()
	var mode_str := "human_achievable" if solver_idx == 1 else "theoretical_oracle"
	var grip_prof := _resolve_active_grip_profile()
	var mu_scale := _sld_mu_scale.value if _sld_mu_scale != null else 0.88
	var temp_c := _sld_temp_c.value if _sld_temp_c != null else 45.0
	var cone_m := _sld_cone_margin.value if _sld_cone_margin != null else 0.25
	var use_budget := _chk_endurance != null and _chk_endurance.button_pressed
	var budget_kwh := _sld_budget.value if _sld_budget != null else 3.5
	var optimize_setup := is_trackside_setup_coopt_enabled()

	var profile_tag := "%s_%s%s" % [
		grip_prof,
		"endur" if use_budget else ("real" if solver_idx == 1 else "oracle"),
		"_optsetup" if optimize_setup else "",
	]
	var out_dir := repo_root.path_join("src/mirena_sim/Assets/RacingLines")
	DirAccess.make_dir_recursive_absolute(out_dir)
	_casadi_out_json = out_dir.path_join("%s_racing_line_casadi_%s.json" % [track_stem, profile_tag])
	_progress_path = OS.get_user_data_dir().path_join("casadi_live_progress.jsonl")
	var pf := FileAccess.open(_progress_path, FileAccess.WRITE)
	if pf != null:
		pf.store_string("")
		pf.close()
	_casadi_log_path = OS.get_user_data_dir().path_join("casadi_live_stdout.log")
	var lf := FileAccess.open(_casadi_log_path, FileAccess.WRITE)
	if lf != null:
		lf.store_string("")
		lf.close()

	_casadi_stage_text = "Homotopia: Iniciando..."
	_casadi_iter_count = 0
	_casadi_last_error = ""

	var script_args: Array = [
		"--track-file", track_file,
		"--mode", mode_str,
		"--grip-profile", grip_prof,
		"--mu-scale", "%.2f" % mu_scale,
		"--temp-c", "%.1f" % temp_c,
		"--cone-margin", "%.2f" % cone_m,
		"--progress-file", _progress_path,
	]
	if use_budget:
		script_args.append("--energy-budget-kwh")
		script_args.append("%.2f" % budget_kwh)
	if optimize_setup:
		script_args.append("--optimize-setup")
	script_args.append("--validate-14dof")
	script_args.append("--output")
	script_args.append(_casadi_out_json)

	var cmd := _resolve_python_command(script_path, script_args)
	var py_exe := String(cmd.get("exe", ""))
	var base_args: PackedStringArray = cmd.get("args", PackedStringArray())
	_last_run_is_wsl = bool(cmd.get("is_wsl", false))
	if py_exe.is_empty():
		_status_lbl.text = "Error: %s not found" % CASADI_PYTHON_BIN
		return

	# Wrap with stderr/stdout redirection inside Python so _pid is the direct process (clean OS.kill)
	var py_redir := "import sys, runpy; lf=open(sys.argv[1],'w',buffering=1); sys.stdout=lf; sys.stderr=lf; sys.argv=sys.argv[2:]; runpy.run_path(sys.argv[0], run_name='__main__')"
	var log_path_arg := _to_wsl_path(_casadi_log_path) if _last_run_is_wsl else _casadi_log_path
	var launch_args := PackedStringArray()
	var u_idx := base_args.find("-u")
	if u_idx >= 0:
		for i in range(u_idx + 1):
			launch_args.append(base_args[i])
		launch_args.append("-c")
		launch_args.append(py_redir)
		launch_args.append(log_path_arg)
		for i in range(u_idx + 1, base_args.size()):
			launch_args.append(base_args[i])
	else:
		launch_args = base_args

	_pid = OS.create_process(py_exe, launch_args, false)
	if _pid <= 0:
		_status_lbl.text = "Error launching CasADi OCP"
		return
	_btn_run.disabled = true
	_btn_stop.disabled = false
	_status_lbl.text = "CasADi PID %d | Etapa: %s" % [_pid, _casadi_stage_text]
	_poll_timer.start()


func _on_stop_pressed() -> void:
	_btn_stop.release_focus()
	stop_process()
	_status_lbl.text = "Cancelado por usuario"


func stop_process() -> void:
	if _poll_timer != null:
		_poll_timer.stop()
	if _pid > 0:
		if OS.is_process_running(_pid):
			OS.kill(_pid)
		_pid = -1
	if _last_run_is_wsl or OS.get_name() == "Windows":
		OS.execute("wsl.exe", PackedStringArray(["-e", "pkill", "-f", "solve_full_lap_ocp.py"]))
		_last_run_is_wsl = false
	if _btn_run != null:
		_btn_run.disabled = false
	if _btn_stop != null:
		_btn_stop.disabled = true


func _on_poll_timeout() -> void:
	if _run_is_casadi:
		_poll_casadi_log()
		if _pid > 0 and not OS.is_process_running(_pid):
			_poll_casadi_log()
			stop_process()
			if FileAccess.file_exists(_casadi_out_json):
				_on_casadi_finished(_casadi_out_json)
			else:
				var err_suffix := (" | %s" % _casadi_last_error) if not _casadi_last_error.is_empty() else ""
				_status_lbl.text = "CasADi exited without output JSON (%s)%s" % [_casadi_stage_text, err_suffix]
	else:
		poll_progress_file()
		if _pid > 0 and not OS.is_process_running(_pid):
			poll_progress_file()
			stop_process()
			if _plot.is_done:
				_status_lbl.text = "Done | Feasible: %d | Rejected: %d" % [_plot.feasible_count, _plot.rejected_count]
			elif _plot.events.is_empty():
				_status_lbl.text = "Process exited early without progress events"
			else:
				_status_lbl.text = "Finished | Feasible: %d | Rejected: %d" % [_plot.feasible_count, _plot.rejected_count]


func _parse_casadi_progress_line(ln: String) -> bool:
	var s := ln.strip_edges()
	if s.is_empty():
		return false
	if s.begins_with("{") and s.ends_with("}"):
		var parsed: Variant = JSON.parse_string(s)
		if parsed is Dictionary:
			var d: Dictionary = parsed
			if d.has("stage") and not String(d["stage"]).is_empty():
				_casadi_stage_text = String(d["stage"])
			elif d.has("message") and not String(d["message"]).is_empty():
				_casadi_stage_text = String(d["message"])
			if d.has("iter"):
				_casadi_iter_count = int(d["iter"])
			elif d.has("iteration"):
				_casadi_iter_count = int(d["iteration"])
			if d.has("error") and not String(d["error"]).is_empty():
				_casadi_last_error = String(d["error"]).substr(0, 80)
			return true
	if s.contains("[STAGE 1/3]") or s.to_lower().contains("homotopy stage a"):
		_casadi_stage_text = "Homotopia 1/3 (Warm-start)"
		return true
	elif s.contains("[STAGE 2/3]") or s.to_lower().contains("homotopy stage b"):
		_casadi_stage_text = "Homotopia 2/3 (IPOPT Full OCP)"
		return true
	elif s.contains("[STAGE 3/3]") or s.to_lower().contains("validating 14-dof"):
		_casadi_stage_text = "Validando 14-DOF..."
		return true
	elif s.contains("[DONE]"):
		_casadi_stage_text = "Completado"
		return true
	elif s.begins_with("iter") or (s.length() > 8 and s[0].is_valid_int() and s.contains("e")):
		_casadi_iter_count += 1
		return true
	elif s.contains("Traceback") or s.contains("Error:") or s.contains("Exception") or s.contains("Infeasible") or s.contains("Restoration_Failed"):
		_casadi_last_error = s.substr(0, 80)
		return true
	return false


func _poll_casadi_log() -> void:
	var updated := false
	# 1. Poll structured --progress-file (supports both JSONL append and JSON snapshot overwrite)
	if not _progress_path.is_empty() and FileAccess.file_exists(_progress_path):
		var pf := FileAccess.open(_progress_path, FileAccess.READ)
		if pf != null:
			var plen := pf.get_length()
			if plen < _progress_byte_offset:
				_progress_byte_offset = 0
				_progress_partial_buf = ""
			if plen > _progress_byte_offset:
				pf.seek(_progress_byte_offset)
				var pchunk := pf.get_buffer(plen - _progress_byte_offset).get_string_from_utf8()
				_progress_byte_offset = plen
				var ptrimmed := pchunk.strip_edges()
				if ptrimmed.begins_with("{") and ptrimmed.ends_with("}") and not ptrimmed.contains("\n"):
					if _parse_casadi_progress_line(ptrimmed):
						updated = true
				else:
					var pfull := _progress_partial_buf + pchunk
					var plines := pfull.split("\n", true)
					_progress_partial_buf = plines[plines.size() - 1]
					for i in range(plines.size() - 1):
						if _parse_casadi_progress_line(plines[i]):
							updated = true
			pf.close()

	# 2. Poll stdout/stderr log file for fallback stage/iter and error diagnostics
	if not _casadi_log_path.is_empty() and FileAccess.file_exists(_casadi_log_path):
		var f := FileAccess.open(_casadi_log_path, FileAccess.READ)
		if f != null:
			var flen := f.get_length()
			if flen > _read_byte_offset:
				f.seek(_read_byte_offset)
				var chunk := f.get_buffer(flen - _read_byte_offset).get_string_from_utf8()
				_read_byte_offset = flen
				var full_txt := _partial_buf + chunk
				var lines := full_txt.split("\n", true)
				_partial_buf = lines[lines.size() - 1]
				for i in range(lines.size() - 1):
					if _parse_casadi_progress_line(lines[i]):
						updated = true
			f.close()

	if updated and _status_lbl != null:
		var err_part := (" | Err: %s" % _casadi_last_error) if not _casadi_last_error.is_empty() else ""
		_status_lbl.text = "Etapa: %s | Iteracion IPOPT: %d%s" % [_casadi_stage_text, _casadi_iter_count, err_part]


func _on_casadi_finished(json_path: String) -> void:
	var loaded := _load_telemetry_reference_json(json_path)
	if not loaded:
		_status_lbl.text = "CasADi finished, failed to parse JSON"
		return
	var meta: Dictionary = _ref_data.get("meta", {})
	var label := String(meta.get("label", json_path.get_file().get_basename()))
	var driv := float(meta.get("drivability_score", -1.0))
	_upsert_lap_library_entry({
		"lap_id": String(meta.get("lap_id", json_path.get_file().get_basename())),
		"label": label,
		"path": json_path,
		"time_s": float(meta.get("sim14dof_lap_time_s", meta.get("ocp_lap_time_s", 0.0))),
		"v_max_kmh": float(meta.get("v_max_kmh", 0.0)),
		"ay_max_g": float(meta.get("ay_max_g", 0.0)),
		"e_22km_kwh": float(meta.get("e_22km_kwh", 0.0)),
		"drivability_score": driv,
		"cones_hit": int(meta.get("cones_hit", 0)),
	})
	var driv_str := (" | Driv %.0f/100" % driv) if driv >= 0.0 else ""
	_status_lbl.text = "CasADi Done | %s (%.2fs%s)" % [label, float(meta.get("sim14dof_lap_time_s", meta.get("ocp_lap_time_s", 0.0))), driv_str]
	_apply_json_to_track_and_ghost(json_path, 0.0)


func poll_progress_file() -> void:
	if _progress_path.is_empty() or not FileAccess.file_exists(_progress_path):
		return
	var f := FileAccess.open(_progress_path, FileAccess.READ)
	if f == null:
		return
	var flen := f.get_length()
	if flen <= _read_byte_offset:
		return
	f.seek(_read_byte_offset)
	var chunk := f.get_buffer(flen - _read_byte_offset).get_string_from_utf8()
	_read_byte_offset = flen
	var full_txt := _partial_buf + chunk
	var lines := full_txt.split("\n", true)
	_partial_buf = lines[lines.size() - 1]
	for i in range(lines.size() - 1):
		var ev := _plot.ingest_jsonl_line(lines[i])
		if not ev.is_empty():
			_refresh_labels()


func _refresh_labels() -> void:
	_status_lbl.text = "Trial %d / %d | Feasible: %d | Rejected: %d" % [
		_plot.events.size(),
		maxi(_plot.total_trials, _plot.events.size()),
		_plot.feasible_count,
		_plot.rejected_count,
	]
	if _plot.best_lap_s > 0.0:
		var extra := (" | E_22km %.2f kWh" % _plot.best_e_22km) if _plot.best_e_22km > 0.0 else ""
		_best_lbl.text = "Best: %.3f s%s | Params: %s" % [
			_plot.best_lap_s, extra, JSON.stringify(_plot.best_params)
		]


func _auto_load_default_telemetry() -> void:
	var repo_root := resolve_repo_root()
	var rl_dir := repo_root.path_join("src/mirena_sim/Assets/RacingLines")
	var candidates := [
		rl_dir.path_join("fsg_2023_racing_line_casadi_fsg_rubbered.json"),
		rl_dir.path_join("fsg_2023_racing_line_casadi_fsg_real.json"),
		rl_dir.path_join("fsg_2023_racing_line_casadi_fsg_real_real.json"),
		"res://Assets/RacingLines/fsg_2023_racing_line_real.json",
		"res://Assets/RacingLines/fsg_2023_racing_line.json",
	]
	for c in candidates:
		if FileAccess.file_exists(c):
			if _load_telemetry_reference_json(c):
				return


func _load_telemetry_reference_json(json_path: String) -> bool:
	var rlv := RacingLineVisScript.new()
	var data := rlv.load_casadi_line_json(json_path)
	rlv.free()
	if data.is_empty():
		return false
	_ref_data = data
	_ref_json_path = json_path
	var ref_pts: Array = data.get("waypoints", [])
	_corners = data.get("corners", [])
	if _corners.is_empty() and ref_pts.size() >= 10:
		_corners = _extract_fallback_corners(ref_pts)
	_populate_corner_dropdown()

	var cmp_pts := _get_comparison_waypoints(json_path)
	var meta: Dictionary = data.get("meta", {})
	var ref_lbl := String(meta.get("label", json_path.get_file().get_basename()))
	var cmp_lbl := "Driver Best" if _has_driver_best_trace() else "QSS Base"
	if _plot != null:
		_plot.set_telemetry_comparison(ref_pts, cmp_pts, _corners, ref_lbl, cmp_lbl)

	var t_14 := float(meta.get("sim14dof_lap_time_s", meta.get("ocp_lap_time_s", 0.0)))
	var vmax := float(meta.get("v_max_kmh", 0.0))
	var e22 := float(meta.get("e_22km_kwh", 0.0))
	var driv_str := (" | Driv %.0f/100" % float(meta["drivability_score"])) if meta.has("drivability_score") else ""
	if t_14 > 0.0:
		_best_lbl.text = "Ref: %s | %.2fs | vmax %.1f km/h | E_22km %.2f kWh%s" % [ref_lbl, t_14, vmax, e22, driv_str]
	else:
		_best_lbl.text = "Ref: %s (%d pts, %d corners)%s" % [ref_lbl, ref_pts.size(), _corners.size(), driv_str]
	_update_recommended_setup_display(data)
	return true


func _has_driver_best_trace() -> bool:
	var sim_node := _get_sim_safe()
	if sim_node != null and "track" in sim_node and sim_node.track != null and "dil_session" in sim_node.track and sim_node.track.dil_session != null:
		var bt: Array = sim_node.track.dil_session._best_trace
		return bt.size() >= 10
	return false


func _get_comparison_waypoints(ref_path: String) -> Array:
	var sim_node := _get_sim_safe()
	if sim_node != null and "track" in sim_node and sim_node.track != null and "dil_session" in sim_node.track and sim_node.track.dil_session != null:
		var bt: Array = sim_node.track.dil_session._best_trace
		if bt.size() >= 10:
			return bt
	# Fallback to QSS base racing line if different from ref
	var qss_path := "res://Assets/RacingLines/fsg_2023_racing_line.json"
	if ref_path != qss_path and FileAccess.file_exists(qss_path):
		var rlv := RacingLineVisScript.new()
		var q_data := rlv.load_casadi_line_json(qss_path)
		rlv.free()
		return q_data.get("waypoints", [])
	return []


static func _extract_fallback_corners(waypoints: Array) -> Array:
	var out: Array = []
	var n := waypoints.size()
	if n < 12:
		return out
	var cid := 1
	var last_s := -50.0
	for i in range(2, n - 2):
		var p: Dictionary = waypoints[i]
		var s_i := float(p.get("s", p.get("s_m", float(i) * 2.5)))
		if s_i - last_s < 35.0:
			continue
		var v_i := float(p.get("v_ref", p.get("v_mps", 15.0)))
		var v_p := float((waypoints[i - 1] as Dictionary).get("v_ref", (waypoints[i - 1] as Dictionary).get("v_mps", 15.0)))
		var v_n := float((waypoints[i + 1] as Dictionary).get("v_ref", (waypoints[i + 1] as Dictionary).get("v_mps", 15.0)))
		var is_apx := bool(p.get("is_apex", false)) or (v_i < 16.0 and v_i <= v_p and v_i <= v_n)
		if is_apx:
			last_s = s_i
			var sp_idx := maxi(0, i - 14)
			var sp: Dictionary = waypoints[sp_idx]
			var sp_nxt: Dictionary = waypoints[mini(n - 1, sp_idx + 1)]
			var dx := float(sp_nxt.get("x", 0.0)) - float(sp.get("x", 0.0))
			var dy := float(sp_nxt.get("y", 0.0)) - float(sp.get("y", 0.0))
			var sp_yaw := float(sp.get("yaw", sp.get("yaw_rad", atan2(dy, dx))))
			var sp_vx := maxf(float(sp.get("v_ref", sp.get("v_mps", 14.0))), 6.0)
			out.append({
				"corner_id": cid,
				"name": "C%d (s=%.0fm)" % [cid, s_i],
				"s_start": maxf(0.0, s_i - 40.0),
				"s_brake": maxf(0.0, s_i - 20.0),
				"s_turn_in": maxf(0.0, s_i - 10.0),
				"s_apex": s_i,
				"s_exit": s_i + 25.0,
				"v_entry_kmh": sp_vx * 3.6,
				"v_apex_kmh": v_i * 3.6,
				"v_exit_kmh": float((waypoints[mini(n - 1, i + 6)] as Dictionary).get("v_ref", v_i)) * 3.6,
				"spawn_pos": [float(sp.get("x", 0.0)), float(sp.get("y", 0.0)), 0.0],
				"spawn_yaw": sp_yaw,
				"spawn_vx": sp_vx,
			})
			cid += 1
			if cid > 16:
				break
	return out


func _populate_corner_dropdown() -> void:
	if _corner_opt == null:
		return
	_corner_opt.clear()
	if _corners.is_empty():
		_corner_opt.add_item("(no corners)", 0)
		_active_corner_idx = -1
		return
	for i in range(_corners.size()):
		var c: Dictionary = _corners[i]
		var nm := String(c.get("name", "Corner %d" % (i + 1)))
		var v_apx := float(c.get("v_apex_kmh", 0.0))
		_corner_opt.add_item("%s | apex %.0f km/h" % [nm, v_apx], i)
	_active_corner_idx = 0
	_corner_opt.select(0)
	_update_corner_info_label()


func _on_corner_selected(idx: int) -> void:
	if _corner_opt != null:
		_corner_opt.release_focus()
	if idx >= 0 and idx < _corners.size():
		_active_corner_idx = idx
		if _plot != null:
			_plot.set_selected_corner(idx)
		_update_corner_info_label()


func _on_plot_corner_clicked(corner_idx: int, _corner_data: Dictionary) -> void:
	if corner_idx >= 0 and corner_idx < _corners.size():
		_active_corner_idx = corner_idx
		if _corner_opt != null and corner_idx < _corner_opt.item_count:
			_corner_opt.select(corner_idx)
		_update_corner_info_label()


func _update_corner_info_label() -> void:
	if _corner_info_lbl == null:
		return
	if _active_corner_idx < 0 or _active_corner_idx >= _corners.size():
		_corner_info_lbl.text = "Corner: --"
		return
	var c: Dictionary = _corners[_active_corner_idx]
	_corner_info_lbl.text = "%s | Brake s=%.0fm | Entry %.0f -> Apex %.0f -> Exit %.0f km/h" % [
		String(c.get("name", "C%d" % (_active_corner_idx + 1))),
		float(c.get("s_brake", 0.0)),
		float(c.get("v_entry_kmh", float(c.get("spawn_vx", 14.0)) * 3.6)),
		float(c.get("v_apex_kmh", 0.0)),
		float(c.get("v_exit_kmh", 0.0)),
	]


func _on_practice_corner_pressed() -> void:
	if _btn_practice_corner != null:
		_btn_practice_corner.release_focus()
	if _active_corner_idx < 0 or _active_corner_idx >= _corners.size():
		_status_lbl.text = "Select a corner first"
		return
	var c: Dictionary = _corners[_active_corner_idx]
	var sim_node := get_node_or_null("/root/Sim")
	if sim_node != null and sim_node.has_method("teleport_to_corner"):
		var ok: bool = bool(sim_node.teleport_to_corner(c))
		if ok:
			_corner_loop_cooldown = 0.8
			_status_lbl.text = "Corner Loop: %s @ %.0f km/h" % [
				String(c.get("name", "C%d" % (_active_corner_idx + 1))),
				float(c.get("spawn_vx", 14.0)) * 3.6,
			]


# ------------------------------------------------------------------
# Compact Lap Library
# ------------------------------------------------------------------

func _load_lap_library() -> void:
	_lap_library.clear()
	if not FileAccess.file_exists(LAP_LIBRARY_PATH):
		return
	var f := FileAccess.open(LAP_LIBRARY_PATH, FileAccess.READ)
	if f == null:
		return
	var parsed: Variant = JSON.parse_string(f.get_as_text())
	f.close()
	if parsed is Array:
		_lap_library = parsed


func _save_lap_library() -> void:
	var f := FileAccess.open(LAP_LIBRARY_PATH, FileAccess.WRITE)
	if f == null:
		return
	f.store_string(JSON.stringify(_lap_library, "  "))
	f.close()


func _discover_builtin_laps() -> void:
	var repo_root := resolve_repo_root()
	var rl_dir := repo_root.path_join("src/mirena_sim/Assets/RacingLines")
	var da := DirAccess.open(rl_dir)
	if da == null:
		return
	da.list_dir_begin()
	var fn := da.get_next()
	while fn != "":
		if not da.current_is_dir() and fn.to_lower().ends_with(".json"):
			var full_p := rl_dir.path_join(fn)
			var lid := fn.get_basename()
			var exists := false
			for e_v in _lap_library:
				if String((e_v as Dictionary).get("lap_id", "")) == lid:
					exists = true
					break
			if not exists:
				var f := FileAccess.open(full_p, FileAccess.READ)
				if f != null:
					var parsed: Variant = JSON.parse_string(f.get_as_text())
					f.close()
					if parsed is Dictionary:
						var d: Dictionary = parsed
						var meta: Dictionary = d.get("meta", {}) if d.get("meta") is Dictionary else {}
						var lbl := String(meta.get("label", lid.replace("fsg_2023_racing_line", "FSG").strip_edges()))
						if lbl.is_empty() or lbl == "FSG":
							lbl = "QSS Base"
						_lap_library.append({
							"lap_id": lid,
							"label": lbl.substr(0, 18),
							"path": full_p,
							"time_s": float(meta.get("sim14dof_lap_time_s", meta.get("ocp_lap_time_s", float(d.get("lap_time_s", 0.0))))),
							"v_max_kmh": float(meta.get("v_max_kmh", 0.0)),
							"ay_max_g": float(meta.get("ay_max_g", 0.0)),
							"e_22km_kwh": float(meta.get("e_22km_kwh", 0.0)),
							"drivability_score": float(meta.get("drivability_score", -1.0)),
							"cones_hit": int(meta.get("cones_hit", 0)),
						})
		fn = da.get_next()
	da.list_dir_end()


func _upsert_lap_library_entry(entry: Dictionary) -> void:
	var lid := String(entry.get("lap_id", ""))
	var replaced := false
	for i in range(_lap_library.size()):
		if String((_lap_library[i] as Dictionary).get("lap_id", "")) == lid:
			_lap_library[i] = entry
			replaced = true
			break
	if not replaced:
		_lap_library.append(entry)
	_save_lap_library()
	_refresh_lap_library_ui()


func _refresh_lap_library_ui() -> void:
	if _lib_opt == null or _lib_table_lbl == null:
		return
	_lib_opt.clear()
	var lines := PackedStringArray(["Label              | Tiempo | v_max | ay_max | E_22km | Driv | Conos"])
	if _lap_library.is_empty():
		_lib_opt.add_item("(no saved laps)", 0)
		lines.append("(no saved laps)")
	else:
		for i in range(_lap_library.size()):
			var e: Dictionary = _lap_library[i]
			var lbl := String(e.get("label", "Lap")).substr(0, 18)
			var t_s := float(e.get("time_s", 0.0))
			var vm := float(e.get("v_max_kmh", 0.0))
			var ay := float(e.get("ay_max_g", 0.0))
			var e22 := float(e.get("e_22km_kwh", 0.0))
			var driv := float(e.get("drivability_score", -1.0))
			var cones := int(e.get("cones_hit", 0))
			_lib_opt.add_item("%s (%.2fs)" % [lbl, t_s] if t_s > 0.0 else lbl, i)
			lines.append("%-18s | %6s | %5s | %6s | %6s | %4s | %5d" % [
				lbl,
				("%.2fs" % t_s) if t_s > 0.0 else "  --  ",
				("%.0f" % vm) if vm > 0.0 else " --  ",
				("%.2fg" % ay) if ay > 0.0 else "  --  ",
				("%.2f" % e22) if e22 > 0.0 else "  --  ",
				("%4.0f" % driv) if driv >= 0.0 else " -- ",
				cones,
			])
	_lib_table_lbl.text = "\n".join(lines)


func _on_lib_load_ghost_pressed() -> void:
	if _btn_lib_ghost != null:
		_btn_lib_ghost.release_focus()
	if _lib_opt == null or _lib_opt.selected < 0 or _lib_opt.selected >= _lap_library.size():
		return
	var entry: Dictionary = _lap_library[_lib_opt.selected]
	var pace_idx := _lib_pace_opt.selected if (_lib_pace_opt != null and _lib_pace_opt.selected >= 0) else 0
	var pace_off := PACE_RABBIT_OFFSETS[clampi(pace_idx, 0, PACE_RABBIT_OFFSETS.size() - 1)]
	var path := String(entry.get("path", ""))
	if path == "DRIVER_BEST":
		var sim_node := get_node_or_null("/root/Sim")
		if sim_node != null and "track" in sim_node and sim_node.track != null and "dil_session" in sim_node.track and sim_node.track.dil_session != null:
			sim_node.track.dil_session.set_pace_rabbit_offset(pace_off)
			sim_node.track.dil_session.set_ghost_visible(true)
			sim_node.track.dil_session.set_ghost_mode(1)
			_status_lbl.text = "Ghost loaded: Mejor Vuelta Piloto (%+.1fs)" % pace_off
		return
	if _apply_json_to_track_and_ghost(path, pace_off):
		_status_lbl.text = "Ghost loaded: %s (%+.1fs)" % [String(entry.get("label", "")), pace_off]


func _on_lib_compare_pressed() -> void:
	if _btn_lib_cmp != null:
		_btn_lib_cmp.release_focus()
	if _lib_opt == null or _lib_opt.selected < 0 or _lib_opt.selected >= _lap_library.size():
		return
	var entry: Dictionary = _lap_library[_lib_opt.selected]
	var path := String(entry.get("path", ""))
	if not path.is_empty() and FileAccess.file_exists(path):
		_load_telemetry_reference_json(path)


func _on_save_driver_best_pressed() -> void:
	if _btn_lib_save_driver != null:
		_btn_lib_save_driver.release_focus()
	var sim_node := get_node_or_null("/root/Sim")
	if sim_node == null or not ("track" in sim_node) or sim_node.track == null or not ("dil_session" in sim_node.track) or sim_node.track.dil_session == null:
		_status_lbl.text = "No active DiL session"
		return
	var dil: Variant = sim_node.track.dil_session
	if float(dil.best_lap_s) <= 0.0:
		_status_lbl.text = "Complete a lap first to save Mejor Vuelta Piloto"
		return
	_upsert_lap_library_entry({
		"lap_id": "driver_best",
		"label": "Mejor Vuelta Piloto",
		"path": "DRIVER_BEST",
		"time_s": float(dil.best_lap_s),
		"v_max_kmh": 0.0,
		"ay_max_g": 0.0,
		"e_22km_kwh": 0.0,
		"cones_hit": int(dil.cones_hit),
	})
	_status_lbl.text = "Saved Mejor Vuelta Piloto (%.2fs)" % float(dil.best_lap_s)


func _apply_json_to_track_and_ghost(json_path: String, pace_off: float) -> bool:
	if json_path.is_empty() or not FileAccess.file_exists(json_path):
		return false
	var sim_node := get_node_or_null("/root/Sim")
	if sim_node != null and sim_node.has_method("load_ghost_from_casadi_json"):
		return bool(sim_node.load_ghost_from_casadi_json(json_path, pace_off))
	return false


func _on_load_result_pressed() -> void:
	_btn_load.release_focus()
	if not _ref_json_path.is_empty() and FileAccess.file_exists(_ref_json_path):
		if _apply_json_to_track_and_ghost(_ref_json_path, 0.0):
			_best_lbl.text = "Loaded %s into Ghost & Racing Line 3D" % _ref_json_path.get_file()
			return
	var csv_p := _plot.best_trace_csv
	if csv_p.is_empty():
		var repo_root := resolve_repo_root()
		csv_p = repo_root.path_join("vd_correlation_lab/results/lts/trace_fsg2023_tv4_auto.csv")
	if not FileAccess.file_exists(csv_p):
		_best_lbl.text = "Trace CSV not found: %s" % csv_p.get_file()
		return
	var sim_node := get_node_or_null("/root/Sim")
	if sim_node != null and "track" in sim_node and sim_node.get("track") != null:
		var trk: Node = sim_node.get("track")
		if "dil_session" in trk and trk.dil_session != null:
			trk.dil_session.load_lts_trace_from_csv(csv_p)
			trk.dil_session.set_ghost_mode(2)
		if "racing_line_vis" in trk and trk.racing_line_vis != null and trk.has_method("load_track"):
			var samples := _load_samples_from_csv(csv_p)
			if not samples.is_empty():
				trk.racing_line_vis.update_from_samples(samples, true)
				trk.racing_line_vis.visible = true
	if sim_node != null and "car" in sim_node and sim_node.get("car") != null:
		var car_n: Node = sim_node.get("car")
		if car_n.has_method("set_racing_line_enabled"):
			car_n.set_racing_line_enabled(true)
		if car_n.model != null and car_n.model.ecu_tv_4wd != null:
			car_n.model.ecu_tv_4wd.configure_from_vehicle_config(car_n.config)
	_best_lbl.text = "Loaded %s into Ghost & Racing Line 3D" % csv_p.get_file()


static func _load_samples_from_csv(csv_path: String) -> Array:
	var out: Array = []
	var f := FileAccess.open(csv_path, FileAccess.READ)
	if f == null:
		return out
	var header := f.get_csv_line()
	var col: Dictionary = {}
	for i in range(header.size()):
		col[header[i].strip_edges()] = i
	if not (col.has("x") and col.has("y")):
		return out
	var stride := 0
	while not f.eof_reached():
		var row := f.get_csv_line()
		if row.size() < header.size():
			continue
		if col.has("lap") and int(row[col["lap"]]) != 1:
			continue
		stride += 1
		if (stride % 5) != 1:
			continue
		out.append({
			"x_godot": -float(row[col["y"]]),
			"z_godot": -float(row[col["x"]]),
			"throttle": float(row[col["throttle"]]) if col.has("throttle") else 0.5,
			"brake": float(row[col["brake"]]) if col.has("brake") else 0.0,
			"ax": float(row[col["ax"]]) if col.has("ax") else 0.0,
		})
	return out
