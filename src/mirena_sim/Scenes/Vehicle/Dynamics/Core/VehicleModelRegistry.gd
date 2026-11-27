class_name VehicleModelRegistry
extends RefCounted

## registro central de solvers vehiculares disponibles en mirena dynamics
## permite instanciar y escalar nuevos modelos ("full_14dof", "fast_3dof", etc.)
## manteniendo una interfaz común hacia mirenacardynamics y vd_lab_replay

const Full14DofModelScript = preload("../Models/full_14dof/Full14DofModel.gd")
const Fast3DofModelScript = preload("../Models/fast_3dof/Fast3DofModel.gd")

const MODEL_FULL_14DOF: String = "full_14dof"
const MODEL_FAST_3DOF: String = "fast_3dof"


static func list_models() -> Array[Dictionary]:
	return [
		{
			"id": MODEL_FULL_14DOF,
			"name": "Full 14-DOF",
			"dof": 14,
			"recommended_hz": 1000,
			"description": "Modelo completo 6-DOF chasis + 4 suspensiones + 4 ruedas con transferencia dinámica de carga.",
		},
		{
			"id": MODEL_FAST_3DOF,
			"name": "Fast 3-DOF",
			"dof": 7,
			"recommended_hz": 200,
			"description": "Modelo plano 3-DOF chasis + 4 ruedas con LLTD cuasi-estático (35 ms) optimizado para Driverless.",
		},
	]


static func normalize_id(model_id: String) -> String:
	var clean := model_id.strip_edges().to_lower()
	if clean == MODEL_FAST_3DOF:
		return MODEL_FAST_3DOF
	return MODEL_FULL_14DOF


static func create_model(model_id: String, cfg = null):
	if normalize_id(model_id) == MODEL_FAST_3DOF:
		return Fast3DofModelScript.new(cfg)
	return Full14DofModelScript.new(cfg)
