class_name SuspensionVisualizer3D
extends Node3D

## Real-time 3D double-wishbone + pushrod + rocker + ARB visualizer.
## Runs purely in _process() when visible == true (zero overhead at 400 Hz physics).

const VehicleConfigScript = preload("../Config/VehicleConfig.gd")
const DoubleWishboneScript = preload("../Dynamics/Suspension/DoubleWishboneKinematics.gd")
const SuspensionSystemScript = preload("../Dynamics/Suspension/SuspensionSystem.gd")

const XRAY_ALPHA: float = 0.16

var _car: Node3D = null
var _built: bool = false
var _xray_active: bool = false
var _xray_body_only: bool = false
var _always_on_top: bool = false
var _saved_materials: Dictionary = {}  # MeshInstance3D -> Array of original override materials (null if none)
var _vis_kinematics: DoubleWishboneScript = null
var _vis_kin_key: String = ""
var _last_poses_cg_ros: Array[Dictionary] = []

# Pre-allocated shared geometries & materials (created once in _ready)
var _rod_mesh: CylinderMesh = null
var _thin_rod_mesh: CylinderMesh = null
var _damper_mesh: CylinderMesh = null
var _joint_mesh: SphereMesh = null

var _mat_wishbone: StandardMaterial3D = null
var _mat_tie: StandardMaterial3D = null
var _mat_pushrod: StandardMaterial3D = null
var _mat_rocker: StandardMaterial3D = null
var _mat_joint: StandardMaterial3D = null
var _damper_mats: Array[StandardMaterial3D] = []
var _arb_mats: Array[StandardMaterial3D] = []  # [0] front, [1] rear
var _all_vis_mats: Array[StandardMaterial3D] = []

# Per-corner node dictionaries (4 corners: FL=0, FR=1, RL=2, RR=3)
var _corner_nodes: Array[Dictionary] = []
var _arb_cross_bars: Array[MeshInstance3D] = []  # [0] front, [1] rear


func bind_car(car_node: Node3D) -> void:
	if _car != car_node and _xray_active:
		set_xray_enabled(false)
	_car = car_node
	_vis_kin_key = ""
	_last_poses_cg_ros.clear()
	if _car != null and (visible or _xray_body_only):
		set_xray_enabled(true)


func _ready() -> void:
	visible = false
	_build_geometries()


func _build_geometries() -> void:
	if _built:
		return
	_built = true

	_rod_mesh = CylinderMesh.new()
	_rod_mesh.top_radius = 0.011
	_rod_mesh.bottom_radius = 0.011
	_rod_mesh.height = 1.0
	_rod_mesh.radial_segments = 10

	_thin_rod_mesh = CylinderMesh.new()
	_thin_rod_mesh.top_radius = 0.009
	_thin_rod_mesh.bottom_radius = 0.009
	_thin_rod_mesh.height = 1.0
	_thin_rod_mesh.radial_segments = 10

	_damper_mesh = CylinderMesh.new()
	_damper_mesh.top_radius = 0.018
	_damper_mesh.bottom_radius = 0.018
	_damper_mesh.height = 1.0
	_damper_mesh.radial_segments = 12

	_joint_mesh = SphereMesh.new()
	_joint_mesh.radius = 0.016
	_joint_mesh.height = 0.032
	_joint_mesh.radial_segments = 10
	_joint_mesh.rings = 6

	_all_vis_mats.clear()
	_mat_wishbone = _make_mat(Color(0.84, 0.88, 0.94))
	_mat_tie = _make_mat(Color(0.98, 0.82, 0.20))
	_mat_pushrod = _make_mat(Color(0.98, 0.48, 0.14))
	_mat_rocker = _make_mat(Color(0.90, 0.94, 1.00), true)
	_mat_joint = _make_mat(Color(0.96, 0.98, 1.00))

	_damper_mats.clear()
	for _i in range(4):
		_damper_mats.append(_make_mat(Color(0.20, 0.88, 0.38)))

	_arb_mats.clear()
	for _a in range(2):
		_arb_mats.append(_make_mat(Color(0.32, 0.70, 0.98)))

	_corner_nodes.clear()
	for i in range(4):
		var axle_idx: int = 0 if i < 2 else 1
		var d: Dictionary = {
			"uca_f": _make_rod(_rod_mesh, _mat_wishbone),
			"uca_r": _make_rod(_rod_mesh, _mat_wishbone),
			"lca_f": _make_rod(_rod_mesh, _mat_wishbone),
			"lca_r": _make_rod(_rod_mesh, _mat_wishbone),
			"tie": _make_rod(_thin_rod_mesh, _mat_tie),
			"pushrod": _make_rod(_thin_rod_mesh, _mat_pushrod),
			"damper": _make_rod(_damper_mesh, _damper_mats[i]),
			"arb_link": _make_rod(_thin_rod_mesh, _arb_mats[axle_idx]),
			"arb_arm": _make_rod(_thin_rod_mesh, _arb_mats[axle_idx]),
			"rocker_mesh": ImmediateMesh.new(),
			"j_ubj": _make_joint(),
			"j_lbj": _make_joint(),
			"j_tie": _make_joint(),
			"j_pr_wb": _make_joint(),
			"j_pr_rk": _make_joint(),
			"j_rk_piv": _make_joint(),
		}
		var r_inst := MeshInstance3D.new()
		r_inst.mesh = d["rocker_mesh"]
		r_inst.material_override = _mat_rocker
		r_inst.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
		add_child(r_inst)
		d["rocker_node"] = r_inst
		_corner_nodes.append(d)

	_arb_cross_bars.clear()
	for a in range(2):
		_arb_cross_bars.append(_make_rod(_thin_rod_mesh, _arb_mats[a]))

	set_always_on_top(_always_on_top)


func _make_mat(col: Color, double_sided: bool = false) -> StandardMaterial3D:
	var mat := StandardMaterial3D.new()
	mat.albedo_color = col
	mat.roughness = 0.30
	mat.metallic = 0.20
	mat.emission_enabled = true
	mat.emission = col
	mat.emission_energy_multiplier = 0.35
	mat.no_depth_test = _always_on_top
	if double_sided:
		mat.cull_mode = BaseMaterial3D.CULL_DISABLED
	_all_vis_mats.append(mat)
	return mat


func _make_rod(mesh_res: Mesh, mat: Material) -> MeshInstance3D:
	var mi := MeshInstance3D.new()
	mi.mesh = mesh_res
	mi.material_override = mat
	mi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	add_child(mi)
	return mi


func _make_joint() -> MeshInstance3D:
	var mi := MeshInstance3D.new()
	mi.mesh = _joint_mesh
	mi.material_override = _mat_joint
	mi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	add_child(mi)
	return mi


static func _has_valid_kinematics_dict(kin_dict: Dictionary) -> bool:
	if kin_dict.is_empty():
		return false
	var fl: Variant = kin_dict.get("front_left")
	var rl: Variant = kin_dict.get("rear_left")
	if not (fl is Dictionary and rl is Dictionary):
		return false
	for k in DoubleWishboneScript.LINK_NAMES:
		if not (fl as Dictionary).has(k) or not (rl as Dictionary).has(k):
			return false
	return true


func is_supported() -> bool:
	if _car == null or not ("model" in _car) or _car.model == null:
		return false
	if not ("suspension" in _car.model) or _car.model.suspension == null:
		return false
	var susp = _car.model.suspension
	if ("use_3d_kinematics" in susp) and bool(susp.use_3d_kinematics):
		if ("kinematics_3d" in susp) and susp.kinematics_3d != null and bool(susp.kinematics_3d.is_ready()):
			return true
	if ("config" in _car) and _car.config != null and ("double_wishbone_kinematics" in _car.config):
		if _car.config.double_wishbone_kinematics is Dictionary:
			return _has_valid_kinematics_dict(_car.config.double_wishbone_kinematics)
	return false


func _resolve_visual_kinematics() -> DoubleWishboneScript:
	if _car == null or not ("model" in _car) or _car.model == null:
		return null
	var susp = _car.model.suspension
	if susp != null and ("use_3d_kinematics" in susp) and bool(susp.use_3d_kinematics):
		if ("kinematics_3d" in susp) and susp.kinematics_3d != null and bool(susp.kinematics_3d.is_ready()):
			return susp.kinematics_3d
	if not ("config" in _car) or _car.config == null:
		return null
	var cfg = _car.config
	var kin_dict: Dictionary = cfg.double_wishbone_kinematics
	if not _has_valid_kinematics_dict(kin_dict):
		return null
	var key: String = DoubleWishboneScript._build_cache_key(
		kin_dict["front_left"],
		kin_dict["rear_left"],
		cfg.wheel_radius,
		cfg.wheelbase,
		cfg.h_cog
	)
	if _vis_kinematics == null or not _vis_kinematics.is_ready() or _vis_kin_key != key:
		if _vis_kinematics == null:
			_vis_kinematics = DoubleWishboneScript.new()
		if not _vis_kinematics.configure(kin_dict, cfg.wheel_radius, cfg.wheelbase, cfg.h_cog):
			return null
		if not _vis_kinematics.build_luts():
			return null
		_vis_kin_key = key
	return _vis_kinematics


func set_visualizer_active(active: bool) -> bool:
	if active and not is_supported():
		visible = false
		_last_poses_cg_ros.clear()
		if not _xray_body_only:
			set_xray_enabled(false)
		return false
	visible = active
	set_xray_enabled(active or _xray_body_only)
	if active:
		update_from_car()
	else:
		_last_poses_cg_ros.clear()
	return visible


func toggle_visualizer() -> bool:
	return set_visualizer_active(not visible)


func is_xray_enabled() -> bool:
	return _xray_active


func is_xray_body_only() -> bool:
	return _xray_body_only


func set_xray_body_only(enabled: bool) -> void:
	_xray_body_only = enabled
	set_xray_enabled(visible or _xray_body_only)


func is_always_on_top() -> bool:
	return _always_on_top


func set_always_on_top(enabled: bool) -> void:
	_always_on_top = enabled
	for m in _all_vis_mats:
		if m != null:
			m.no_depth_test = enabled
			m.render_priority = 10 if enabled else 0


func set_xray_enabled(enabled: bool) -> void:
	if _car == null or _xray_active == enabled:
		return
	if enabled:
		_saved_materials.clear()
		for mi in _collect_body_meshes():
			_apply_xray_to_mesh(mi)
		_xray_active = true
	else:
		for mi in _saved_materials.keys():
			if is_instance_valid(mi):
				var arr: Array = _saved_materials[mi]
				for s in range(arr.size()):
					(mi as MeshInstance3D).set_surface_override_material(s, arr[s])
		_saved_materials.clear()
		_xray_active = false


func _is_excluded_mesh(mi: MeshInstance3D) -> bool:
	var n: Node = mi
	while n != null and n != _car:
		if n == self:
			return true
		var nm := String(n.name)
		if (
			nm.ends_with("_WHEEL")
			or nm.begins_with("WheelForceArrow")
			or nm == "WheelForcesVisualizer3D"
			or nm.begins_with("FrictionEllipseRing")
			or nm.begins_with("FzArrow_")
			or nm.begins_with("TorqueBar_")
		):
			return true
		n = n.get_parent()
	return false


func _collect_body_meshes() -> Array[MeshInstance3D]:
	var out: Array[MeshInstance3D] = []
	if _car == null:
		return out
	for node in _car.find_children("*", "MeshInstance3D", true, false):
		if node is MeshInstance3D:
			var mi := node as MeshInstance3D
			if not _is_excluded_mesh(mi):
				out.append(mi)
	return out


func _apply_xray_to_mesh(mi: MeshInstance3D) -> void:
	if mi.mesh == null:
		return
	var scount: int = mi.mesh.get_surface_count()
	var orig_arr: Array = []
	for s in range(scount):
		var ov: Material = mi.get_surface_override_material(s)
		orig_arr.append(ov)
		var base_mat: Material = ov if ov != null else mi.mesh.surface_get_material(s)
		var ghost := StandardMaterial3D.new()
		if base_mat is StandardMaterial3D:
			var sm := base_mat as StandardMaterial3D
			ghost.albedo_color = Color(sm.albedo_color.r, sm.albedo_color.g, sm.albedo_color.b, XRAY_ALPHA)
			ghost.roughness = sm.roughness
		else:
			ghost.albedo_color = Color(0.55, 0.60, 0.68, XRAY_ALPHA)
		ghost.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
		ghost.depth_draw_mode = BaseMaterial3D.DEPTH_DRAW_DISABLED
		ghost.cull_mode = BaseMaterial3D.CULL_DISABLED
		mi.set_surface_override_material(s, ghost)
	_saved_materials[mi] = orig_arr


func _process(_delta: float) -> void:
	if not visible:
		return
	update_from_car()


const CORNER_NAMES: Array[String] = ["FL", "FR", "RL", "RR"]


func update_from_car() -> void:
	if not is_supported():
		set_visualizer_active(false)
		return
	_build_geometries()
	var susp = _car.model.suspension
	var kin: DoubleWishboneScript = _resolve_visual_kinematics()
	if kin == null or not kin.is_ready():
		set_visualizer_active(false)
		return
	var is_3d_physics: bool = (
		("use_3d_kinematics" in susp)
		and bool(susp.use_3d_kinematics)
		and ("kinematics_3d" in susp)
		and susp.kinematics_3d != null
		and bool(susp.kinematics_3d.is_ready())
	)
	var a_cog: float = _car.config.a_cog
	var com: Vector3 = _car.center_of_mass if ("center_of_mass" in _car) else Vector3(0.0, _car.config.h_cog, -0.117)
	var steer_ang: float = _car.inputs.steer_angle if ("inputs" in _car and _car.inputs != null) else 0.0

	var arb_pivs: Array[Vector3] = [Vector3.ZERO, Vector3.ZERO, Vector3.ZERO, Vector3.ZERO]
	var arb_forces: Array[float] = [0.0, 0.0, 0.0, 0.0]
	_last_poses_cg_ros.clear()

	for i in range(4):
		var c = susp.corners[i]
		var is_front: bool = (i < 2)
		var is_left: bool = (i % 2 == 0)
		var pose: Dictionary = kin.compute_visual_pose_3d(is_front, is_left, c.delta_z, steer_ang, a_cog)
		if pose.is_empty():
			continue

		var lca_f := _to_godot(pose["lca_front"], com)
		var lca_r := _to_godot(pose["lca_rear"], com)
		var lbj := _to_godot(pose["lbj"], com)
		var uca_f := _to_godot(pose["uca_front"], com)
		var uca_r := _to_godot(pose["uca_rear"], com)
		var ubj := _to_godot(pose["ubj"], com)
		var tie_i := _to_godot(pose["tie_inner"], com)
		var tie_o := _to_godot(pose["tie_outer"], com)
		var pr_wb := _to_godot(pose["pushrod_wishbone"], com)
		var pr_rk := _to_godot(pose["pushrod_rocker"], com)
		var rk_piv := _to_godot(pose["rocker_pivot"], com)
		var dm_rk := _to_godot(pose["damper_rocker"], com)
		var dm_ch := _to_godot(pose["damper_chassis"], com)
		var arb_rk := _to_godot(pose["arb_link_rocker"], com)
		var arb_tip := _to_godot(pose["arb_arm_tip"], com)
		var arb_piv := _to_godot(pose["arb_chassis_piv"], com)
		arb_pivs[i] = arb_piv
		arb_forces[i] = c.arb_force

		var nd: Dictionary = _corner_nodes[i]
		_place_rod(nd["uca_f"], uca_f, ubj)
		_place_rod(nd["uca_r"], uca_r, ubj)
		_place_rod(nd["lca_f"], lca_f, lbj)
		_place_rod(nd["lca_r"], lca_r, lbj)
		_place_rod(nd["tie"], tie_i, tie_o)
		_place_rod(nd["pushrod"], pr_wb, pr_rk)
		_place_rod(nd["damper"], dm_ch, dm_rk)
		_place_rod(nd["arb_link"], arb_rk, arb_tip)
		_place_rod(nd["arb_arm"], arb_piv, arb_tip)

		(nd["j_ubj"] as Node3D).position = ubj
		(nd["j_lbj"] as Node3D).position = lbj
		(nd["j_tie"] as Node3D).position = tie_o
		(nd["j_pr_wb"] as Node3D).position = pr_wb
		(nd["j_pr_rk"] as Node3D).position = pr_rk
		(nd["j_rk_piv"] as Node3D).position = rk_piv

		var rm := nd["rocker_mesh"] as ImmediateMesh
		rm.clear_surfaces()
		rm.surface_begin(Mesh.PRIMITIVE_TRIANGLES)
		rm.surface_add_vertex(rk_piv); rm.surface_add_vertex(pr_rk); rm.surface_add_vertex(dm_rk)
		rm.surface_add_vertex(rk_piv); rm.surface_add_vertex(dm_rk); rm.surface_add_vertex(arb_rk)
		rm.surface_end()

		var d_col: Color = _spring_color(c.delta_z, c.spring_force, c.static_preload)
		_damper_mats[i].albedo_color = d_col
		_damper_mats[i].emission = d_col

		var entry := pose.duplicate(false)
		entry["name"] = CORNER_NAMES[i]
		entry["is_front"] = is_front
		entry["is_left"] = is_left
		entry["is_3d_kinematics"] = is_3d_physics
		entry["delta_z_m"] = float(c.delta_z)
		entry["steer_cmd_rad"] = steer_ang
		entry["physics_camber_rad"] = float(c.camber_rad)
		entry["physics_wheel_steer_rad"] = float(c.wheel_steer_rad) if is_3d_physics else (
			(SuspensionSystemScript.compute_ackermann_steer(steer_ang, is_left, _car.config) + float(c.toe_rad))
			if is_front else float(c.toe_rad)
		)
		entry["wheel_center_godot"] = _to_godot(pose["wheel_center"], com)
		entry["link_lengths"] = _pose_link_lengths(pose)
		entry["rendered_rod_lengths"] = {
			"uca_f": (nd["uca_f"] as MeshInstance3D).basis.y.length(),
			"uca_r": (nd["uca_r"] as MeshInstance3D).basis.y.length(),
			"lca_f": (nd["lca_f"] as MeshInstance3D).basis.y.length(),
			"lca_r": (nd["lca_r"] as MeshInstance3D).basis.y.length(),
			"tie": (nd["tie"] as MeshInstance3D).basis.y.length(),
			"pushrod": (nd["pushrod"] as MeshInstance3D).basis.y.length(),
			"damper": (nd["damper"] as MeshInstance3D).basis.y.length(),
			"arb_link": (nd["arb_link"] as MeshInstance3D).basis.y.length(),
			"arb_arm": (nd["arb_arm"] as MeshInstance3D).basis.y.length(),
		}
		_last_poses_cg_ros.append(entry)

	_place_rod(_arb_cross_bars[0], arb_pivs[0], arb_pivs[1])
	_place_rod(_arb_cross_bars[1], arb_pivs[2], arb_pivs[3])
	for a in range(2):
		var f_arb: float = absf(arb_forces[a * 2])
		var u_arb: float = clampf(f_arb / 800.0, 0.0, 1.0)
		var a_col := Color(0.28, 0.65, 0.92).lerp(Color(0.98, 0.28, 0.72), u_arb)
		_arb_mats[a].albedo_color = a_col
		_arb_mats[a].emission = a_col


func get_rendered_poses_cg_ros() -> Array[Dictionary]:
	if not visible or not is_supported():
		return []
	update_from_car()
	return _last_poses_cg_ros.duplicate(false)


func get_corner_metrics() -> Array[Dictionary]:
	var out: Array[Dictionary] = []
	if not is_supported():
		return out
	var susp = _car.model.suspension
	var is_3d_physics: bool = ("use_3d_kinematics" in susp) and bool(susp.use_3d_kinematics)
	var steer_ang: float = _car.inputs.steer_angle if ("inputs" in _car and _car.inputs != null) else 0.0
	for i in range(4):
		var c = susp.corners[i]
		var is_front: bool = (i < 2)
		var is_left: bool = (i % 2 == 0)
		var w_steer: float = float(c.wheel_steer_rad) if is_3d_physics else (
			(SuspensionSystemScript.compute_ackermann_steer(steer_ang, is_left, _car.config) + float(c.toe_rad))
			if is_front else float(c.toe_rad)
		)
		out.append({
			"name": CORNER_NAMES[i],
			"dz_mm": float(c.delta_z) * 1000.0,
			"camber_deg": rad_to_deg(float(c.camber_rad)),
			"toe_deg": rad_to_deg(float(c.toe_rad)),
			"wheel_steer_deg": rad_to_deg(w_steer),
			"fz_n": float(c.fz),
			"spring_force_n": float(c.spring_force),
			"f_arb_n": float(c.arb_force),
			"is_3d_kinematics": is_3d_physics,
		})
	return out


static func _pose_link_lengths(pose: Dictionary) -> Dictionary:
	return {
		"lca_f": (pose["lbj"] as Vector3).distance_to(pose["lca_front"] as Vector3),
		"lca_r": (pose["lbj"] as Vector3).distance_to(pose["lca_rear"] as Vector3),
		"uca_f": (pose["ubj"] as Vector3).distance_to(pose["uca_front"] as Vector3),
		"uca_r": (pose["ubj"] as Vector3).distance_to(pose["uca_rear"] as Vector3),
		"tie": (pose["tie_outer"] as Vector3).distance_to(pose["tie_inner"] as Vector3),
		"pushrod": (pose["pushrod_rocker"] as Vector3).distance_to(pose["pushrod_wishbone"] as Vector3),
		"arb_link": (pose["arb_arm_tip"] as Vector3).distance_to(pose["arb_link_rocker"] as Vector3),
		"arb_arm": (pose["arb_arm_tip"] as Vector3).distance_to(pose["arb_chassis_piv"] as Vector3),
		"kingpin": (pose["ubj"] as Vector3).distance_to(pose["lbj"] as Vector3),
	}


static func _to_godot(p_cg_ros: Vector3, com: Vector3) -> Vector3:
	return com + VehicleConfigScript.ros_to_godot_vec(p_cg_ros)


static func _place_rod(node: MeshInstance3D, p_a: Vector3, p_b: Vector3) -> void:
	var diff: Vector3 = p_b - p_a
	var len: float = diff.length()
	if len < 1e-5:
		node.visible = false
		return
	node.visible = true
	var y_ax: Vector3 = diff / len
	var ref: Vector3 = Vector3.FORWARD if absf(y_ax.z) < 0.9 else Vector3.RIGHT
	var x_ax: Vector3 = ref.cross(y_ax).normalized()
	var z_ax: Vector3 = x_ax.cross(y_ax).normalized()
	node.basis = Basis(x_ax, y_ax * len, z_ax)
	node.position = 0.5 * (p_a + p_b)


static func _spring_color(dz: float, spring_f: float, preload_f: float) -> Color:
	var norm: float = clampf((spring_f - preload_f) / maxf(preload_f * 1.2, 600.0), -1.0, 1.0)
	if absf(norm) < 0.02:
		norm = clampf(dz / 0.025, -1.0, 1.0)
	if norm < 0.0:
		return Color(0.20, 0.88, 0.38).lerp(Color(0.22, 0.55, 0.98), -norm)
	return Color(0.20, 0.88, 0.38).lerp(Color(0.95, 0.22, 0.20), norm)
