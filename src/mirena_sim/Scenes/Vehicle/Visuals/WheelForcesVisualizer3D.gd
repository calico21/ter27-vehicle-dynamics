class_name WheelForcesVisualizer3D
extends Node3D

## Per-wheel 3D DiL visualizer anchored to FL / FR / RL / RR wheel nodes:
##   1. Vertical Fz arrow (height & color proportional to vertical wheel load)
##   2. Stacked vertical torque/brake bar outboard (+/-0.26 m) of each wheel:
##      - Upwards (+): Drive torque in Green (#22C55E)
##      - Downwards (-): Stacked Regenerative Brake in Orange-Red (#FF6B35)
##                       followed by Mechanical Hydraulic Brake in Deep Red (#DC2626)
## Preallocates all meshes and materials at setup for zero per-frame heap allocations.

const FZ_MAX_REF_N: float = 2200.0
const FZ_MAX_HEIGHT_M: float = 1.10
const FZ_DEADZONE_N: float = 1.0

const BAR_OFFSET_LAT_M: float = 0.26
const BAR_HALF_HEIGHT_M: float = 0.42
const BAR_WIDTH_M: float = 0.048
const TORQUE_REF_MAX_NM: float = 312.5 # 25 Nm * 12.5 gear ratio

const COL_DRIVE_GREEN: Color = Color(0.133, 0.773, 0.369, 0.92)     # #22C55E
const COL_REGEN_ORANGE: Color = Color(1.000, 0.420, 0.208, 0.94)    # #FF6B35
const COL_MECH_RED: Color = Color(0.863, 0.149, 0.149, 0.96)        # #DC2626
const COL_BG_DARK: Color = Color(0.08, 0.10, 0.14, 0.55)
const COL_ZERO_TICK: Color = Color(0.92, 0.94, 0.98, 0.88)

const COL_ELLIPSE_GREEN: Color = Color(0.133, 0.773, 0.369, 0.92)   # < 75%
const COL_ELLIPSE_AMBER: Color = Color(0.961, 0.620, 0.043, 0.94)   # 75-92%
const COL_ELLIPSE_RED: Color = Color(0.937, 0.267, 0.267, 0.96)     # > 92%
const ELLIPSE_INNER_R_M: float = 0.28
const ELLIPSE_OUTER_R_M: float = 0.38
const ELLIPSE_SEGMENTS: int = 32
const ELLIPSE_GROUND_Y_M: float = 0.035

var show_fz_arrows: bool = false
var show_torque_bars: bool = false
var show_friction_ellipses: bool = false

var _car: Node = null
var _built: bool = false

# Per-wheel Fz arrow nodes [4]
var _fz_roots: Array[Node3D] = []
var _fz_shafts: Array[MeshInstance3D] = []
var _fz_tips: Array[MeshInstance3D] = []
var _fz_mats: Array[StandardMaterial3D] = []

# Per-wheel Torque/Brake bar nodes [4]
var _bar_roots: Array[Node3D] = []
var _drv_fills: Array[MeshInstance3D] = []
var _reg_fills: Array[MeshInstance3D] = []
var _mech_fills: Array[MeshInstance3D] = []

# Per-wheel Friction Ellipse ground ring nodes [4]
var _ellipse_rings: Array[MeshInstance3D] = []
var _ellipse_mats: Array[StandardMaterial3D] = []
var _wheel_anchors: Array[Node3D] = []

var _wheel_node_names: Array[String] = ["FL_WHEEL", "FR_WHEEL", "RL_WHEEL", "RR_WHEEL"]


func bind_car(car: Node) -> void:
	_car = car


func _ensure_built() -> void:
	if _built or _car == null:
		return
	_built = true

	var shaft_mesh := CylinderMesh.new()
	shaft_mesh.top_radius = 0.016
	shaft_mesh.bottom_radius = 0.016
	shaft_mesh.height = 1.0
	shaft_mesh.radial_segments = 8

	var tip_mesh := CylinderMesh.new()
	tip_mesh.top_radius = 0.0
	tip_mesh.bottom_radius = 0.044
	tip_mesh.height = 0.14
	tip_mesh.radial_segments = 10

	var unit_box := BoxMesh.new()
	unit_box.size = Vector3(1.0, 1.0, 1.0)

	var mat_bg := _make_unshaded_mat(COL_BG_DARK, true)
	var mat_zero := _make_unshaded_mat(COL_ZERO_TICK, true)
	var mat_drv := _make_unshaded_mat(COL_DRIVE_GREEN, true)
	var mat_reg := _make_unshaded_mat(COL_REGEN_ORANGE, true)
	var mat_mech := _make_unshaded_mat(COL_MECH_RED, true)

	var ring_mesh := _build_annulus_mesh(ELLIPSE_INNER_R_M, ELLIPSE_OUTER_R_M, ELLIPSE_SEGMENTS)

	for i in range(4):
		var anchor := _car.get_node_or_null(_wheel_node_names[i]) as Node3D
		_wheel_anchors.append(anchor)
		var parent_node: Node = anchor if anchor != null else self

		# 1. Fz Arrow centered above wheel
		var fz_root := Node3D.new()
		fz_root.name = "FzArrow_%d" % i
		fz_root.position = Vector3(0.0, 0.24, 0.0)
		fz_root.visible = false
		parent_node.add_child(fz_root)

		var fz_mat := _make_unshaded_mat(Color(0.35, 0.82, 0.98, 0.90), true)

		var s_mi := MeshInstance3D.new()
		s_mi.mesh = shaft_mesh
		s_mi.material_override = fz_mat
		s_mi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
		fz_root.add_child(s_mi)

		var t_mi := MeshInstance3D.new()
		t_mi.mesh = tip_mesh
		t_mi.material_override = fz_mat
		t_mi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
		fz_root.add_child(t_mi)

		_fz_roots.append(fz_root)
		_fz_shafts.append(s_mi)
		_fz_tips.append(t_mi)
		_fz_mats.append(fz_mat)

		# 2. Torque / Brake Bar outboard of wheel (+/-0.26 m)
		var side_sign := -1.0 if (i % 2 == 0) else 1.0 # Godot -X left, +X right
		var b_root := Node3D.new()
		b_root.name = "TorqueBar_%d" % i
		b_root.position = Vector3(side_sign * BAR_OFFSET_LAT_M, 0.24, 0.0)
		b_root.visible = false
		parent_node.add_child(b_root)

		var bg_mi := MeshInstance3D.new()
		bg_mi.mesh = unit_box
		bg_mi.material_override = mat_bg
		bg_mi.scale = Vector3(BAR_WIDTH_M * 1.15, BAR_HALF_HEIGHT_M * 2.0, BAR_WIDTH_M * 0.45)
		bg_mi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
		b_root.add_child(bg_mi)

		var z_mi := MeshInstance3D.new()
		z_mi.mesh = unit_box
		z_mi.material_override = mat_zero
		z_mi.scale = Vector3(BAR_WIDTH_M * 1.5, 0.012, BAR_WIDTH_M * 0.70)
		z_mi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
		b_root.add_child(z_mi)

		var drv_mi := MeshInstance3D.new()
		drv_mi.mesh = unit_box
		drv_mi.material_override = mat_drv
		drv_mi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
		drv_mi.visible = false
		b_root.add_child(drv_mi)

		var reg_mi := MeshInstance3D.new()
		reg_mi.mesh = unit_box
		reg_mi.material_override = mat_reg
		reg_mi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
		reg_mi.visible = false
		b_root.add_child(reg_mi)

		var mech_mi := MeshInstance3D.new()
		mech_mi.mesh = unit_box
		mech_mi.material_override = mat_mech
		mech_mi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
		mech_mi.visible = false
		b_root.add_child(mech_mi)

		_bar_roots.append(b_root)
		_drv_fills.append(drv_mi)
		_reg_fills.append(reg_mi)
		_mech_fills.append(mech_mi)

		# 3. Ground Friction-Ellipse Utilization Ring under each wheel
		var e_mat := _make_unshaded_mat(COL_ELLIPSE_GREEN, true)
		e_mat.cull_mode = BaseMaterial3D.CULL_DISABLED
		e_mat.render_priority = 12
		var e_mi := MeshInstance3D.new()
		e_mi.name = "FrictionEllipseRing_%d" % i
		e_mi.mesh = ring_mesh
		e_mi.material_override = e_mat
		e_mi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
		e_mi.ignore_occlusion_culling = true
		e_mi.visible = false
		add_child(e_mi)
		_ellipse_rings.append(e_mi)
		_ellipse_mats.append(e_mat)


static func _build_annulus_mesh(inner_r: float, outer_r: float, segments: int) -> ArrayMesh:
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	for s in range(segments):
		var a0 := TAU * float(s) / float(segments)
		var a1 := TAU * float(s + 1) / float(segments)
		var c0 := cos(a0)
		var s0 := sin(a0)
		var c1 := cos(a1)
		var s1 := sin(a1)
		var p_in0 := Vector3(inner_r * c0, 0.0, inner_r * s0)
		var p_out0 := Vector3(outer_r * c0, 0.0, outer_r * s0)
		var p_in1 := Vector3(inner_r * c1, 0.0, inner_r * s1)
		var p_out1 := Vector3(outer_r * c1, 0.0, outer_r * s1)
		st.add_vertex(p_in0)
		st.add_vertex(p_out0)
		st.add_vertex(p_out1)
		st.add_vertex(p_in0)
		st.add_vertex(p_out1)
		st.add_vertex(p_in1)
	return st.commit()


static func _make_unshaded_mat(col: Color, alpha: bool = true) -> StandardMaterial3D:
	var mat := StandardMaterial3D.new()
	mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	mat.albedo_color = col
	if alpha:
		mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	mat.emission_enabled = true
	mat.emission = Color(col.r, col.g, col.b)
	mat.emission_energy_multiplier = 1.15
	return mat


static func compute_fz_arrow_height(fz_n: float) -> float:
	if fz_n <= FZ_DEADZONE_N or not is_finite(fz_n):
		return 0.0
	return clampf(fz_n / FZ_MAX_REF_N, 0.0, 1.35) * FZ_MAX_HEIGHT_M


static func fz_to_color(fz_n: float) -> Color:
	var c_low := Color(0.28, 0.72, 1.00, 0.90)
	var c_mid := Color(0.86, 0.96, 1.00, 0.92)
	var c_hi := Color(1.00, 0.56, 0.12, 0.95)
	if fz_n < 400.0:
		return c_low
	if fz_n < 1100.0:
		return c_low.lerp(c_mid, (fz_n - 400.0) / 700.0)
	return c_mid.lerp(c_hi, clampf((fz_n - 1100.0) / 800.0, 0.0, 1.0))


## Pure layout calculator for stacked 3D torque/brake bars (zero heap allocations on caller if reused).
static func compute_corner_bar_layout(
	t_drive_nm: float,
	t_regen_nm: float,
	t_mech_nm: float,
	t_max_nm: float = TORQUE_REF_MAX_NM,
	max_half_h: float = BAR_HALF_HEIGHT_M
) -> Dictionary:
	var ref := maxf(t_max_nm, 50.0)
	var h_drv := clampf(maxf(t_drive_nm, 0.0) / ref, 0.0, 1.0) * max_half_h
	var f_reg := clampf(maxf(t_regen_nm, 0.0) / ref, 0.0, 1.0)
	var f_mech := clampf(maxf(t_mech_nm, 0.0) / (ref * 1.6), 0.0, 1.0)
	var h_reg := f_reg * max_half_h
	var h_mech := f_mech * max_half_h
	var tot_down := h_reg + h_mech
	if tot_down > max_half_h and tot_down > 1e-6:
		var s := max_half_h / tot_down
		h_reg *= s
		h_mech *= s
	return {
		"drive_h": h_drv,
		"drive_cy": 0.5 * h_drv,
		"regen_h": h_reg,
		"regen_cy": -0.5 * h_reg,
		"mech_h": h_mech,
		"mech_cy": -(h_reg + 0.5 * h_mech),
	}


func set_fz_arrows_enabled(on: bool) -> void:
	show_fz_arrows = on
	if on:
		_ensure_built()
	for i in range(_fz_roots.size()):
		if is_instance_valid(_fz_roots[i]):
			_fz_roots[i].visible = on


func set_torque_bars_enabled(on: bool) -> void:
	show_torque_bars = on
	if on:
		_ensure_built()
	for i in range(_bar_roots.size()):
		if is_instance_valid(_bar_roots[i]):
			_bar_roots[i].visible = on


func set_friction_ellipses_enabled(on: bool) -> void:
	show_friction_ellipses = on
	if on:
		_ensure_built()
	for i in range(_ellipse_rings.size()):
		if is_instance_valid(_ellipse_rings[i]):
			_ellipse_rings[i].visible = on


func set_force_arrows_enabled(on: bool) -> void:
	set_fz_arrows_enabled(on)
	set_torque_bars_enabled(on)


static func compute_friction_utilization(fx: float, fy: float, fz: float, mu_eff: float) -> float:
	if fz <= FZ_DEADZONE_N or mu_eff <= 1e-4 or not is_finite(fx) or not is_finite(fy) or not is_finite(fz):
		return 0.0
	var f_lim := maxf(mu_eff * fz, 1e-3)
	return clampf(sqrt(fx * fx + fy * fy) / f_lim, 0.0, 1.5)


static func utilization_to_color(eta: float) -> Color:
	if eta < 0.75:
		return COL_ELLIPSE_GREEN
	if eta <= 0.92:
		return COL_ELLIPSE_AMBER
	return COL_ELLIPSE_RED


func _process(_delta: float) -> void:
	if show_friction_ellipses:
		_update_friction_ellipses()


func _update_friction_ellipses() -> void:
	if not show_friction_ellipses or _car == null or not is_instance_valid(_car):
		return
	_ensure_built()
	if not ("model" in _car) or _car.model == null or not ("wheels" in _car.model):
		return
	var wheels: Array = _car.model.wheels
	var mu_ref: float = 1.5
	if "config" in _car and _car.config != null and "tyre_mu_y" in _car.config:
		mu_ref = maxf(float(_car.config.tyre_mu_y), 0.2)

	for i in range(mini(4, _ellipse_rings.size())):
		var ring := _ellipse_rings[i]
		if not is_instance_valid(ring):
			continue
		if i >= wheels.size() or wheels[i] == null:
			ring.visible = false
			continue
		var w: Variant = wheels[i]
		var fz := float(w.F_z) if "F_z" in w else 0.0
		if fz <= FZ_DEADZONE_N:
			ring.visible = false
			continue
		ring.visible = true
		var fx := float(w.F_x) if "F_x" in w else 0.0
		var fy := float(w.F_y) if "F_y" in w else 0.0
		var surf_mu := float(w.surface_mu_scale) if "surface_mu_scale" in w else 1.0
		var eta := compute_friction_utilization(fx, fy, fz, maxf(mu_ref * surf_mu, 0.2))
		var col := utilization_to_color(eta)
		_ellipse_mats[i].albedo_color = col
		_ellipse_mats[i].emission = Color(col.r, col.g, col.b)

		var anchor := _wheel_anchors[i] if i < _wheel_anchors.size() else null
		if anchor != null and is_instance_valid(anchor) and is_inside_tree():
			var gp := anchor.global_position
			ring.global_position = Vector3(gp.x, ELLIPSE_GROUND_Y_M, gp.z)
			ring.global_rotation = Vector3(0.0, _car.global_rotation.y if _car is Node3D else 0.0, 0.0)
		var load_scale := clampf(sqrt(fz / 700.0), 0.55, 1.35)
		ring.scale = Vector3(load_scale, 1.0, load_scale)


func update_visuals(
	normal_forces: Array,
	t_drive_arr: Array,
	t_regen_arr: Array,
	t_mech_arr: Array,
	t_wheel_max_nm: float = TORQUE_REF_MAX_NM
) -> void:
	if not show_fz_arrows and not show_torque_bars:
		return
	_ensure_built()

	for i in range(4):
		if show_fz_arrows and i < _fz_roots.size():
			var fz := float(normal_forces[i]) if i < normal_forces.size() else 0.0
			var h := compute_fz_arrow_height(fz)
			if h <= 0.005:
				_fz_roots[i].visible = false
			else:
				_fz_roots[i].visible = true
				var shaft_h := maxf(h - 0.14, 0.02)
				_fz_shafts[i].scale = Vector3(1.0, shaft_h, 1.0)
				_fz_shafts[i].position = Vector3(0.0, 0.5 * shaft_h, 0.0)
				_fz_tips[i].position = Vector3(0.0, shaft_h + 0.07, 0.0)
				var col := fz_to_color(fz)
				_fz_mats[i].albedo_color = col
				_fz_mats[i].emission = Color(col.r, col.g, col.b)

		if show_torque_bars and i < _bar_roots.size():
			_bar_roots[i].visible = true
			var td := float(t_drive_arr[i]) if i < t_drive_arr.size() else 0.0
			var tr := float(t_regen_arr[i]) if i < t_regen_arr.size() else 0.0
			var tm := float(t_mech_arr[i]) if i < t_mech_arr.size() else 0.0
			var lay := compute_corner_bar_layout(td, tr, tm, t_wheel_max_nm, BAR_HALF_HEIGHT_M)

			var dh := float(lay["drive_h"])
			_drv_fills[i].visible = dh > 0.004
			if _drv_fills[i].visible:
				_drv_fills[i].scale = Vector3(BAR_WIDTH_M, dh, BAR_WIDTH_M * 0.6)
				_drv_fills[i].position = Vector3(0.0, float(lay["drive_cy"]), 0.0)

			var rh := float(lay["regen_h"])
			_reg_fills[i].visible = rh > 0.004
			if _reg_fills[i].visible:
				_reg_fills[i].scale = Vector3(BAR_WIDTH_M, rh, BAR_WIDTH_M * 0.6)
				_reg_fills[i].position = Vector3(0.0, float(lay["regen_cy"]), 0.0)

			var mh := float(lay["mech_h"])
			_mech_fills[i].visible = mh > 0.004
			if _mech_fills[i].visible:
				_mech_fills[i].scale = Vector3(BAR_WIDTH_M, mh, BAR_WIDTH_M * 0.6)
				_mech_fills[i].position = Vector3(0.0, float(lay["mech_cy"]), 0.0)
