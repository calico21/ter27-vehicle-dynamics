class_name TvSilBridge
extends RefCounted

## C/C++ Software-In-the-Loop (SIL) Bridge for ECU Torque Vectoring v3.
##
## Inspects whether the optional native GDExtension class `GpTorqueVectoringSIL`
## (backed by `addons/rclgd/bin/libter_tv_sil.so` compiled from TeR_ECU's
## `TRQ_VECTORING` C library) is registered in `ClassDB`. When absent, safely
## falls back to the 1:1 GDScript port in `EcuTorqueVectoringV3`.
##
## Canonical C ABI reference (`TeR_ECU/Libraries/TRQ_VECTORING/inc/gp_torque_vectoring.h`):
##   typedef struct {
##       uint8_t enable;
##       float   max_total_trq;
##       float   max_charge_power_w;
##   } gp_regen_limits_t;
##
##   void gp_tv_init(tv_state_t* state);
##
##   void gp_tv_step(
##       float fx_driver, float delta, float vx, float vy, float wz,
##       float ay, float ax, const float omega[4], float brake_norm,
##       float temp_inv_rl, float temp_inv_rr, float vy_gps, uint8_t gps_valid,
##       const gp_regen_limits_t* regen,
##       float dt, tv_state_t* state, float t_cmd_out[4]
##   );

enum ImplementationMode {
	GDSCRIPT = 0,
	C_SIL = 1
}

const IMPLEMENTATION_GDSCRIPT: int = ImplementationMode.GDSCRIPT
const IMPLEMENTATION_C_SIL: int = ImplementationMode.C_SIL

const NATIVE_SIL_CLASS_NAME: StringName = &"GpTorqueVectoringSIL"
const NATIVE_SIL_SHARED_LIB_PATH: String = "res://addons/rclgd/bin/libter_tv_sil.so"

const CAN_ID_TV_DYNAMICS: int = 0x250
const CAN_ID_TC_ESTIMATOR: int = 0x251
const CAN_ID_TV_ACTUATORS: int = 0x252
const CAN_ID_TV_DIAGNOSTICS: int = 0x253

var active_implementation: int = IMPLEMENTATION_GDSCRIPT
var _native_sil: Object = null
var _last_status_message: String = "GDScript implementation initialized (default)."
var _native_initialized: bool = false

func _init(prefer_native: bool = false) -> void:
	set_implementation(IMPLEMENTATION_C_SIL if prefer_native else IMPLEMENTATION_GDSCRIPT)

## Returns true only when the native GDExtension class `GpTorqueVectoringSIL` is registered in ClassDB.
static func is_native_sil_available() -> bool:
	return ClassDB.class_exists(NATIVE_SIL_CLASS_NAME) and ClassDB.can_instantiate(NATIVE_SIL_CLASS_NAME)

## Returns true if the optional compiled shared library exists on disk (even if not loaded as GDExtension).
static func has_native_shared_library() -> bool:
	return FileAccess.file_exists(NATIVE_SIL_SHARED_LIB_PATH)

## Switches active backend between IMPLEMENTATION_GDSCRIPT and IMPLEMENTATION_C_SIL.
## Automatically falls back to IMPLEMENTATION_GDSCRIPT when native GDExtension is unavailable.
func set_implementation(mode: int) -> int:
	if mode == IMPLEMENTATION_C_SIL:
		if is_native_sil_available():
			if _native_sil == null:
				_native_sil = ClassDB.instantiate(NATIVE_SIL_CLASS_NAME)
			if _native_sil != null:
				active_implementation = IMPLEMENTATION_C_SIL
				_native_initialized = false
				gp_tv_init(null)
				_last_status_message = "Native C SIL active via GDExtension '%s'." % String(NATIVE_SIL_CLASS_NAME)
				return active_implementation
		active_implementation = IMPLEMENTATION_GDSCRIPT
		_native_sil = null
		_native_initialized = false
		_last_status_message = (
			"Fallback to GDScript: GDExtension class '%s' not registered in ClassDB (lib_present=%s)."
			% [String(NATIVE_SIL_CLASS_NAME), str(has_native_shared_library())]
		)
		return active_implementation

	active_implementation = IMPLEMENTATION_GDSCRIPT
	_last_status_message = "GDScript implementation active."
	return active_implementation

func is_using_native_sil() -> bool:
	return active_implementation == IMPLEMENTATION_C_SIL and _native_sil != null

## Informative backend status dictionary for diagnostics, UI telemetry, and automated tests.
func sil_backend_status() -> Dictionary:
	return {
		"active_implementation": active_implementation,
		"backend_name": "C_SIL" if is_using_native_sil() else "GDSCRIPT",
		"native_class_name": String(NATIVE_SIL_CLASS_NAME),
		"native_class_exists": ClassDB.class_exists(NATIVE_SIL_CLASS_NAME),
		"native_lib_path": NATIVE_SIL_SHARED_LIB_PATH,
		"native_lib_present": has_native_shared_library(),
		"native_available": is_native_sil_available(),
		"fallback_active": not is_using_native_sil(),
		"native_initialized": _native_initialized,
		"message": _last_status_message,
		"c_api_symbols": ["gp_tv_init", "gp_tv_step", "gp_tv_state_sizeof"],
		"c_source_ref": "TeR_ECU/Libraries/TRQ_VECTORING/src/gp_torque_vectoring.c",
		"can_ids": {
			"dynamics": CAN_ID_TV_DYNAMICS,
			"tc_estimator": CAN_ID_TC_ESTIMATOR,
			"actuators": CAN_ID_TV_ACTUATORS,
			"diagnostics": CAN_ID_TV_DIAGNOSTICS
		}
	}

## Wrapper corresponding to C `void gp_tv_init(tv_state_t* state)`.
func gp_tv_init(gd_ecu: Object = null) -> void:
	if is_using_native_sil() and _native_sil.has_method("gp_tv_init"):
		_native_sil.call("gp_tv_init")
		_native_initialized = true
	if gd_ecu != null and gd_ecu.has_method("reset"):
		gd_ecu.call("reset")

## Wrapper corresponding to C `void gp_tv_step(...)` with 4WD MirenaSim extensions.
## Returns 4-element wheel torque command array [FL, FR, RL, RR] in Nm.
func gp_tv_step(
	fx_driver: float,
	delta: float,
	vx: float,
	vy: float,
	wz: float,
	ay: float,
	ax: float,
	omega: Array[float],
	brake_norm: float,
	dt: float,
	wheel_steers: Array[float] = [0.0, 0.0, 0.0, 0.0],
	vy_gps: float = 0.0,
	gps_valid: bool = false,
	temp_inv_rl: float = 40.0,
	temp_inv_rr: float = 40.0,
	regen_limits: Dictionary = {"enable": false, "max_total_trq": 0.0, "max_charge_power_w": 0.0},
	gd_ecu: Object = null
) -> Array[float]:
	if is_using_native_sil() and _native_sil.has_method("gp_tv_step"):
		if not _native_initialized:
			gp_tv_init(null)
		var res = _native_sil.call(
			"gp_tv_step",
			fx_driver,
			delta,
			vx,
			vy,
			wz,
			ay,
			ax,
			omega,
			brake_norm,
			temp_inv_rl,
			temp_inv_rr,
			vy_gps,
			gps_valid,
			regen_limits,
			dt,
			wheel_steers
		)
		if res is Array and res.size() >= 4:
			return [float(res[0]), float(res[1]), float(res[2]), float(res[3])]

	if gd_ecu != null and gd_ecu.has_method("_gp_tv_step"):
		gd_ecu.call(
			"_gp_tv_step",
			fx_driver,
			delta,
			vx,
			vy,
			wz,
			ay,
			ax,
			omega,
			brake_norm,
			dt,
			wheel_steers,
			vy_gps,
			gps_valid
		)
		var cmd: Array = gd_ecu.get("last_t_cmd")
		if cmd.size() >= 4:
			return [float(cmd[0]), float(cmd[1]), float(cmd[2]), float(cmd[3])]

	return [0.0, 0.0, 0.0, 0.0]

## Documents exact C interface contract for external GDExtension / C-SIL compilation.
static func get_c_interface_spec() -> Dictionary:
	return {
		"header": "TeR_ECU/Libraries/TRQ_VECTORING/inc/gp_torque_vectoring.h",
		"source": "TeR_ECU/Libraries/TRQ_VECTORING/src/gp_torque_vectoring.c",
		"gdextension_class": String(NATIVE_SIL_CLASS_NAME),
		"shared_library": NATIVE_SIL_SHARED_LIB_PATH,
		"functions": {
			"gp_tv_init": {
				"c_signature": "void gp_tv_init(tv_state_t* state)",
				"description": "Initializes EKF, RLS TC, QP warmstart, and LPF states."
			},
			"gp_tv_step": {
				"c_signature": "void gp_tv_step(float fx_driver, float delta, float vx, float vy, float wz, float ay, float ax, const float omega[4], float brake_norm, float temp_inv_rl, float temp_inv_rr, float vy_gps, uint8_t gps_valid, const gp_regen_limits_t* regen, float dt, tv_state_t* state, float t_cmd_out[4])",
				"description": "200 Hz (5 ms) 8-stage Torque Vectoring + EKF + KKT QP + RLS TC control step."
			}
		},
		"structs": {
			"gp_regen_limits_t": ["enable (uint8_t)", "max_total_trq (float)", "max_charge_power_w (float)"],
			"tv_state_t": [
				"wz_int", "delta_prev", "t_qp_prev[4]", "t_out_prev[4]",
				"tc (tc_state_t)", "ekf (gp_ekf_t)", "vy_est", "alpha_qp",
				"lam_prev", "mz_sat_ratio", "vy_gps_last", "vy_gps_age_ms",
				"ax_filt", "ay_filt", "t_ub_rl_filt", "t_ub_rr_filt",
				"t_lb_rl_filt", "t_lb_rr_filt", "qp_residual"
			]
		}
	}
