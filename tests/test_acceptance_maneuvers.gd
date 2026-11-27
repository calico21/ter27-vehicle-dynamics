extends SceneTree

const AcceptanceManeuversSuite = preload("acceptance_maneuvers_suite.gd")

func _init() -> void:
	var res: Dictionary = AcceptanceManeuversSuite.run(true)
	if int(res.get("failed", 1)) == 0:
		print(">> ACCEPTANCE MANEUVERS VERIFIED SUCCESSFULLY! <<\n")
		quit(0)
	else:
		for err in res.get("errors", []):
			print("  -> ERROR: %s" % err)
		print(">> ACCEPTANCE MANEUVERS FAILED! <<\n")
		quit(1)
