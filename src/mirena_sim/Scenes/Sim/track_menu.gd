extends Control

@onready var track_stats: Label = $HBoxContainer/VBoxContainer/TrackStats
@onready var track_preset: OptionButton = $HBoxContainer/VBoxContainer/TrackPreset


func _get_sim() -> Node:
	return get_node_or_null("/root/Sim")


func _ready() -> void:
	_populate_track_presets()


func _populate_track_presets() -> void:
	if track_preset == null:
		track_preset = get_node_or_null("HBoxContainer/VBoxContainer/TrackPreset") as OptionButton
	if track_stats == null:
		track_stats = get_node_or_null("HBoxContainer/VBoxContainer/TrackStats") as Label
	if track_preset == null:
		return
	track_preset.focus_mode = Control.FOCUS_NONE
	track_preset.clear()
	track_preset.add_item("Select Track Preset...")
	track_preset.set_item_metadata(0, "")

	var seen_files: Dictionary = {}
	var featured := [
		["FSG 2023 (Hockenheim)", "res://TrackFiles/fsg_2023.json"],
		["Skidpad (Official)", "res://TrackFiles/skidpad.json"],
		["Acceleration (Official)", "res://TrackFiles/acceleration.json"],
		["Track 10 (Mirena)", ProjectSettings.globalize_path("res://../mirena_planning/test/tracks/track10.json")],
	]
	for entry in featured:
		var label: String = entry[0]
		var path: String = entry[1]
		if _is_valid_track_json(path):
			_add_preset_item(label, path)
			seen_files[path.get_file().to_lower()] = true

	var scan_dirs := [
		"res://TrackFiles",
		ProjectSettings.globalize_path("res://../mirena_planning/test/tracks"),
	]
	for dir_path in scan_dirs:
		var da := DirAccess.open(dir_path)
		if da == null:
			continue
		var files: Array[String] = []
		da.list_dir_begin()
		var fname := da.get_next()
		while fname != "":
			if not da.current_is_dir() and fname.to_lower().ends_with(".json"):
				files.append(fname)
			fname = da.get_next()
		da.list_dir_end()
		files.sort_custom(func(a: String, b: String) -> bool: return a.naturalnocasecmp_to(b) < 0)
		for f in files:
			var key := f.to_lower()
			if seen_files.has(key):
				continue
			var full_path: String = dir_path.path_join(f)
			if _is_valid_track_json(full_path):
				_add_preset_item(_format_track_label(f), full_path)
				seen_files[key] = true


func _add_preset_item(label: String, path: String) -> void:
	var idx := track_preset.item_count
	track_preset.add_item(label)
	track_preset.set_item_metadata(idx, path)


func _format_track_label(filename: String) -> String:
	var stem := filename.get_basename()
	if stem.begins_with("track") and stem.substr(5).is_valid_int():
		return "Track %s" % stem.substr(5)
	return stem.replace("_", " ").capitalize()


func _is_valid_track_json(path: String) -> bool:
	if path.is_empty() or not FileAccess.file_exists(path):
		return false
	var f := FileAccess.open(path, FileAccess.READ)
	if f == null:
		return false
	var parsed: Variant = JSON.parse_string(f.get_as_text())
	if typeof(parsed) != TYPE_DICTIONARY:
		return false
	var d: Dictionary = parsed
	return d.has("cones") or d.has("path")


func _on_track_preset_selected(index: int) -> void:
	if track_preset != null:
		track_preset.release_focus()
		var meta: Variant = track_preset.get_item_metadata(index)
		var path := str(meta) if meta != null else ""
		if not path.is_empty():
			_on_track_file_selected(path)


func _on_generate_track():
	var sim := _get_sim()
	if sim == null or sim.get("track") == null:
		return
	sim.track.create_track()
	track_stats.text = _describe(sim.track.track_gen.last_stats)


## What the generator produced, against the rules it was held to.
##
## Not a pass or a fail: the generator never returns a layout that breaks a rule,
## so this is how much room the track left. A lap sitting on 200 m with the
## tightest corner near the limit is a hard track. The fallback line means the
## noise gave up and the circle was used instead.
func _describe(stats: Dictionary) -> String:
	if stats.is_empty():
		return "No track generated"
	if stats.get("fallback", false):
		return "Fallback circle: the noise produced nothing legal"

	return "Lap %.0f m (rules %.0f-%.0f)\nTightest turn %.1f m radius (min %.1f)\nLongest straight %.0f m (max %.0f)\n%d attempt(s)" % [
		stats["lap_length"], TrackGenerator.MIN_LAP_LENGTH, TrackGenerator.MAX_LAP_LENGTH,
		stats["min_turn_radius"], TrackGenerator.MIN_TURN_RADIUS,
		stats["longest_straight"], TrackGenerator.MAX_STRAIGHT_LENGTH,
		stats.get("attempts", 0)]


func _on_open_track_pressed():
	$FileDialog.access = FileDialog.ACCESS_FILESYSTEM
	$FileDialog.filters = PackedStringArray(["*.json ; Track files"])
	$FileDialog.current_dir = ProjectSettings.globalize_path("res://TrackFiles")
	$FileDialog.visible = true


func _on_export_track_pressed():
	$ExportDialog.visible = true


func _on_export_file_selected(path: String):
	var sim := _get_sim()
	if sim == null or sim.get("track") == null:
		return
	if sim.track.export_track(path):
		track_stats.text = "Exported to %s" % path.get_file()
	else:
		track_stats.text = "Could not write %s" % path.get_file()


func _on_track_file_selected(path: String):
	var sim := _get_sim()
	if sim == null or sim.get("track") == null:
		return
	sim.track.load_track(path)
	track_stats.text = "Loaded %s" % path.get_file()


func _on_enable_cone_collision_toggled(toggled_on: bool) -> void:
	var sim := _get_sim()
	if sim == null or sim.get("car") == null:
		return
	sim.car.cone_collision_set(toggled_on)


func _on_clear_track_pressed() -> void:
	var sim := _get_sim()
	if sim == null or sim.get("track") == null:
		return
	sim.track.clear_track()
	track_stats.text = "No track generated"
