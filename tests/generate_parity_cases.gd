extends SceneTree

const TireModelScript = preload("res://Scenes/Vehicle/Dynamics/Tires/TireModel.gd")

func _init() -> void:
	var json_path = "res://../../tests/parity_test_cases.json"
	var file = FileAccess.open(json_path, FileAccess.READ)
	if not file:
		print("Failed to open " + json_path)
		quit(1)
		return
	var text = file.get_as_text()
	file.close()
	
	var data = JSON.parse_string(text)
	if not (data is Dictionary) or not data.has("cases"):
		print("Invalid JSON")
		quit(1)
		return
		
	var cases: Array = data["cases"]
	print("Updating %d cases with calibrated MF 6.2 model..." % cases.size())
	
	for idx in range(cases.size()):
		var c: Dictionary = cases[idx]
		var inp: Dictionary = c["inputs"]
		var k = float(inp["kappa"])
		var a = float(inp["alpha"])
		var fz = float(inp["f_z"])
		var g = float(inp["gamma"])
		var p = float(inp["p_kpa"])
		var vx = float(inp["v_x"])
		
		var res = TireModelScript.compute_forces(k, a, fz, null, g, p, vx, true)
		c["expected"] = {
			"Fx_mf": res["Fx_mf"],
			"Fy_mf": res["Fy_mf"],
			"Mz_mf": res["Mz_mf"],
			"Mx": res["Mx"],
			"My": res["My"],
			"delta_Fy": res["delta_Fy"],
			"delta_Mz": res["delta_Mz"],
			"Fx": res["Fx"],
			"Fy": res["Fy"],
			"Mz": res["Mz"]
		}
	
	var out_file = FileAccess.open(json_path, FileAccess.WRITE)
	if not out_file:
		print("Failed to write to " + json_path)
		quit(1)
		return
	out_file.store_string(JSON.stringify(data, "  "))
	out_file.close()
	print("Successfully updated parity_test_cases.json with %d cases." % cases.size())
	quit(0)
