extends Node

const _STUBS_PATH := "res://addons/rclgd/ros_stubs.gd"

static func _sync_ros_stubs() -> void:
	if ClassDB.class_exists("RosMsg"):
		if FileAccess.file_exists(_STUBS_PATH):
			DirAccess.remove_absolute(ProjectSettings.globalize_path(_STUBS_PATH))
	else:
		if not FileAccess.file_exists(_STUBS_PATH):
			var f := FileAccess.open(_STUBS_PATH, FileAccess.WRITE)
			if f:
				f.store_string("# Auto-generated fallback stub when native rclgd GDExtension is absent.\nclass_name RosMsg\nextends RefCounted\n")
				f.close()

static func _static_init() -> void:
	_sync_ros_stubs()
	if Engine.is_editor_hint(): return
	if Engine.has_singleton("rclgd"):
		var ros: Object = Engine.get_singleton("rclgd")
		var args := OS.get_cmdline_args()
		# Read ROS parameters from project.godot [ros] section and pass as ROS2 args.
		# Command-line args (from launch files) take precedence over project settings.
		var params := {
			"publish_sim_time": ProjectSettings.get_setting("ros/parameters/publish_sim_time", false),
			"use_sim_time": ProjectSettings.get_setting("ros/parameters/use_sim_time", false),
			"use_separate_thread": ProjectSettings.get_setting("ros/parameters/use_separate_thread", true),
		}
		for key in params:
			var already_set := false
			for i in range(args.size()):
				if args[i].begins_with("%s:=" % key):
					already_set = true
					break
			if not already_set:
				args.append("--ros-args")
				args.append("-p")
				args.append("%s:=%s" % [key, str(params[key]).to_lower()])
		ros.call("init", args)
		print("[ROS Init Args]: ", args)

func _ready() -> void:
	if Engine.is_editor_hint(): return
	if not Engine.has_singleton("rclgd") and not ClassDB.class_exists("rclgd"):
		push_warning("rclgd singleton/extension not loaded (e.g., Windows standalone mode). ROS 2 features disabled.")

func _exit_tree() -> void:
	if Engine.has_singleton("rclgd"):
		var ros: Object = Engine.get_singleton("rclgd")
		if ros and ros.call("ok"):
			ros.call("shutdown")
			print("[ROS] Shutdown called from exit_tree.")
