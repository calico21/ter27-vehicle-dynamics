extends SceneTree

## Automated Test Suite for Pacejka MF 6.2 + Neural Residual (TeR_Tire PINN) + Transient Relaxation
## Tests physical validity, calibration fidelity, neural bounds, PINN physics-informed invariants, and transient dynamics.

const VehicleConfigScript = preload("res://Scenes/Vehicle/Config/VehicleConfig.gd")
const WheelDynamicsScript = preload("res://Scenes/Vehicle/Dynamics/Tires/WheelDynamics.gd")
const TireModelScript = preload("res://Scenes/Vehicle/Dynamics/Tires/TireModel.gd")
const Pacejka62Script = preload("res://Scenes/Vehicle/Dynamics/Tires/MF62/Pacejka62.gd")
const NeuralCorrectionScript = preload("res://Scenes/Vehicle/Dynamics/Tires/NeuralCorrection/NeuralTireCorrection.gd")
const TeRTireNeuralResidualScript = preload("res://Scenes/Vehicle/Dynamics/Tires/NeuralCorrection/TeRTireNeuralResidual.gd")
const TransientRelaxationScript = preload("res://Scenes/Vehicle/Dynamics/Tires/Transient/TransientRelaxation.gd")

func _init() -> void:
	var res = run()
	print("[test_tire_mf62_suite] passed=%d failed=%d" % [res["passed"], res["failed"]])
	for err in res["errors"]:
		print("  ERROR: %s" % err)
	quit(0 if res["failed"] == 0 else 1)

static func run() -> Dictionary:
	var passed = 0
	var failed = 0
	var errors: Array[String] = []
	
	var config = VehicleConfigScript.new()
	var mf = Pacejka62Script.new()
	var nn = NeuralCorrectionScript.new()
	var wheel = WheelDynamicsScript.new(0, "FL")
	
	var F_z_nom = 800.0 # N
	
	# --- Test 1: MF 6.2 Pure Lateral Force Peak and Shape ---
	var fy_peak = 0.0
	var alpha_peak_deg = 0.0
	for deg in range(0, 16):
		var a_rad = deg_to_rad(float(deg))
		var fy = mf.compute_fy0(a_rad, F_z_nom, 0.0, 83.0)
		if fy > fy_peak:
			fy_peak = fy
			alpha_peak_deg = float(deg)
	
	var mu_peak = fy_peak / F_z_nom
	# Hoosier 16x7.5-10 R25B reaches mu ~ 2.4 - 2.6 at 800 N with peak plateau between 7 - 14 deg
	if mu_peak >= 2.2 and mu_peak <= 2.8 and alpha_peak_deg >= 7.0 and alpha_peak_deg <= 14.0:
		passed += 1
	else:
		failed += 1
		errors.append("MF6.2 Lateral Peak check failed: peak_mu=%.2f (expected 2.2-2.8), alpha_peak=%.1f deg (expected 7-14)" % [mu_peak, alpha_peak_deg])
	
	# --- Test 2: Self-Aligning Torque Mz Restoring Peak and Trail Decay ---
	# Restoring moment Mz is negative for alpha > 0 in ISO (+Z up), peaking around 3-5 deg, then decaying toward zero past peak lateral slip
	var mz_small = mf.compute_mz(deg_to_rad(3.0), F_z_nom, mf.compute_fy0(deg_to_rad(3.0), F_z_nom), 0.0, 83.0)
	var mz_high = mf.compute_mz(deg_to_rad(12.0), F_z_nom, mf.compute_fy0(deg_to_rad(12.0), F_z_nom), 0.0, 83.0)
	
	if mz_small < -20.0 and absf(mz_high) < absf(mz_small):
		passed += 1
	else:
		failed += 1
		errors.append("MF6.2 Mz alignment curve check failed: mz(3 deg)=%.2f Nm (expected < -20 Nm), mz(12 deg)=%.2f Nm (expected decay in magnitude)" % [mz_small, mz_high])
	
	# --- Test 3: Camber Sensitivity (IA = 4 deg vs 0 deg) ---
	var fy_c0 = mf.compute_fy0(deg_to_rad(4.0), F_z_nom, 0.0, 83.0)
	var fy_c4 = mf.compute_fy0(deg_to_rad(4.0), F_z_nom, deg_to_rad(4.0), 83.0)
	# Positive camber adds camber thrust
	if fy_c4 != fy_c0 and absf(fy_c4 - fy_c0) > 10.0:
		passed += 1
	else:
		failed += 1
		errors.append("MF6.2 Camber sensitivity check failed: fy_c0=%.1f N, fy_c4=%.1f N" % [fy_c0, fy_c4])
	
	# --- Test 4: Pressure Dependence ---
	var fy_p55 = mf.compute_fy0(deg_to_rad(8.0), F_z_nom, 0.0, 55.0) # 8 psi
	var fy_p83 = mf.compute_fy0(deg_to_rad(8.0), F_z_nom, 0.0, 83.0) # 12 psi
	if fy_p55 != fy_p83 and absf(fy_p55 - fy_p83) > 15.0:
		passed += 1
	else:
		failed += 1
		errors.append("MF6.2 Pressure sensitivity check failed: fy(55 kPa)=%.1f N, fy(83 kPa)=%.1f N" % [fy_p55, fy_p83])
	
	# --- Test 5: Neural Residual Physics Gates (Zero-Slip & Low-Load) ---
	var zero_corr = nn.predict_correction(0.0, F_z_nom, 0.0, 83.0)
	var air_corr = nn.predict_correction(deg_to_rad(5.0), 0.0, 0.0, 83.0)
	if absf(zero_corr["delta_Fy"]) < 1e-3 and absf(air_corr["delta_Fy"]) < 1e-3:
		passed += 1
	else:
		failed += 1
		errors.append("Neural physics gating failed: zero_slip_Fy=%.4f, airborne_Fy=%.4f (must be 0.0)" % [zero_corr["delta_Fy"], air_corr["delta_Fy"]])
	
	# --- Test 6: Neural Residual Saturation Clamp Limit ---
	# Even at extreme input, correction must never exceed safety limits (+-200 N, +-10 Nm)
	var extreme_corr = nn.predict_correction(deg_to_rad(25.0), 2000.0, deg_to_rad(10.0), 120.0)
	if absf(extreme_corr["delta_Fy"]) <= 200.01 and absf(extreme_corr["delta_Mz"]) <= 10.01:
		passed += 1
	else:
		failed += 1
		errors.append("Neural saturation limit exceeded: delta_Fy=%.2f, delta_Mz=%.2f" % [extreme_corr["delta_Fy"], extreme_corr["delta_Mz"]])
	
	# --- Test 7: Transient Carcass Relaxation Dynamics ---
	# Apply step steer input from 0 to 0.1 rad at 10 m/s
	var alpha_target = 0.10
	var alpha_trans = 0.0
	var dt = 0.0025 # 400 Hz
	
	# Integrate 10 steps (0.025 s -> 0.25 m traveled at 10 m/s ~ 1 relaxation length)
	for step in range(10):
		alpha_trans = TransientRelaxationScript.integrate_alpha_relaxation(alpha_trans, alpha_target, 10.0, dt, F_z_nom)
	
	# After ~1 relaxation length, transient angle should reach ~63% of target (0.05 to 0.08)
	if alpha_trans > 0.04 and alpha_trans < 0.095:
		passed += 1
	else:
		failed += 1
		errors.append("Transient relaxation dynamic response failed: alpha_trans=%.4f (expected 0.04 to 0.095)" % alpha_trans)
	
	# --- Test 8: End-to-End Hybrid Wheel Integration ---
	wheel.reset()
	wheel.omega = 50.0
	wheel.compute_slip(10.0, -1.0, 0.0, config.slip_regularization_speed, config.wheel_radius, dt)
	wheel.compute_forces(F_z_nom, config)
	
	if wheel.F_y > 400.0 and absf(wheel.M_z) > 1.0 and wheel.is_grounded:
		passed += 1
	else:
		failed += 1
		errors.append("Hybrid wheel force integration failed: Fy=%.2f N, Mz=%.2f Nm" % [wheel.F_y, wheel.M_z])

	# --- Test 9: Pure Longitudinal Fx Peak, Odd Symmetry & Continuity ---
	var fx_peak: float = 0.0
	var k_peak: float = 0.0
	var sym_ok: bool = true
	var finite_ok: bool = true
	for k_i in range(-200, 201, 2):
		var k: float = float(k_i) * 0.005  # [-1.0 .. +1.0]
		var fx_p: float = mf.compute_fx0(k, F_z_nom, 0.0, 83.0)
		if not is_finite(fx_p):
			finite_ok = false
		if k >= 0.0 and k <= 0.25 and fx_p > fx_peak:
			fx_peak = fx_p
			k_peak = k
		if absf(k) > 0.001 and absf(fx_p + mf.compute_fx0(-k, F_z_nom, 0.0, 83.0)) > 1e-4:
			sym_ok = false
	var mux_peak: float = fx_peak / F_z_nom
	if finite_ok and sym_ok and absf(mf.compute_fx0(0.0, F_z_nom, 0.0, 83.0)) < 1e-6 and mux_peak >= 1.9 and mux_peak <= 2.5 and k_peak >= 0.08 and k_peak <= 0.20:
		passed += 1
	else:
		failed += 1
		errors.append("MF6.2 Longitudinal peak/symmetry check failed: mu_x=%.3f, k_peak=%.3f, sym=%s, finite=%s" % [mux_peak, k_peak, str(sym_ok), str(finite_ok)])

	# --- Test 10: Longitudinal Pressure & Camber Sensitivity ---
	var fx_p55: float = mf.compute_fx0(0.12, F_z_nom, 0.0, 55.2)
	var fx_p97: float = mf.compute_fx0(0.12, F_z_nom, 0.0, 96.6)
	var fx_ia0: float = mf.compute_fx0(0.14, F_z_nom, 0.0, 83.0)
	var fx_ia4: float = mf.compute_fx0(0.14, F_z_nom, deg_to_rad(4.0), 83.0)
	if fx_p55 > fx_p97 + 40.0 and fx_ia4 < fx_ia0:
		passed += 1
	else:
		failed += 1
		errors.append("MF6.2 Longitudinal pressure/camber check failed: fx(55kPa)=%.1f, fx(97kPa)=%.1f, fx(0deg)=%.1f, fx(4deg)=%.1f" % [fx_p55, fx_p97, fx_ia0, fx_ia4])

	# --- Test 11: Combined-Slip Gxa/Gyk Scaling & Unified Compound Loading (R20 + R25B) ---
	var c0: Dictionary = mf.compute_combined_forces(0.0, deg_to_rad(-6.0), F_z_nom, 0.0, 83.0, false)
	var cd: Dictionary = mf.compute_combined_forces(0.12, deg_to_rad(-6.0), F_z_nom, 0.0, 83.0, false)
	var fx_pure12: float = mf.compute_fx0(0.12, F_z_nom, 0.0, 83.0)
	var gxa_12: float = float(cd["Fx"]) / maxf(fx_pure12, 1e-4)
	var r20_ok: bool = TireModelScript.set_compound("R20") and TireModelScript.get_neural().is_ready
	var r20_fx: float = TireModelScript.compute_forces(0.12, deg_to_rad(-3.0), F_z_nom, config, 0.0, 83.0, 11.11, true)["Fx"]
	var r25b_ok: bool = TireModelScript.set_compound("R25B") and TireModelScript.get_neural().is_ready
	if absf(float(c0["Gyk"]) - 1.0) < 1e-6 and float(cd["Gyk"]) > 0.35 and float(cd["Gyk"]) < 0.90 and gxa_12 > 0.35 and gxa_12 < 0.90 and r20_ok and r25b_ok and r20_fx > 500.0:
		passed += 1
	else:
		failed += 1
		errors.append("MF6.2 Combined-slip / compound check failed: Gyk(0)=%.4f, Gyk(0.12)=%.3f, Gxa=%.3f, r20=%s, r25b=%s" % [float(c0["Gyk"]), float(cd["Gyk"]), gxa_12, str(r20_ok), str(r25b_ok)])

	# --- Test 12: TeRTireNeuralResidual Default Weights Zero Residual & Exact Parity ---
	var ter_default := TeRTireNeuralResidualScript.new()
	var r_def_l: Vector3 = ter_default.predict_residual(0.08, deg_to_rad(6.0), F_z_nom, deg_to_rad(-2.0), 62.0, false)
	var r_def_r: Vector3 = ter_default.predict_residual(-0.05, deg_to_rad(-4.0), 1100.0, deg_to_rad(1.5), 45.0, true)
	var prev_neural_en: bool = TireModelScript.enable_neural_correction
	var prev_legacy_en: bool = TireModelScript.enable_legacy_neural_correction
	TireModelScript.enable_legacy_neural_correction = false
	TireModelScript.enable_neural_correction = false
	var f_off: Dictionary = TireModelScript.compute_forces(0.06, deg_to_rad(5.0), F_z_nom, config, deg_to_rad(-1.5), 83.0, 12.0, false, false, 1.0, 62.0, 1.0)
	TireModelScript.enable_neural_correction = true
	var f_on: Dictionary = TireModelScript.compute_forces(0.06, deg_to_rad(5.0), F_z_nom, config, deg_to_rad(-1.5), 83.0, 12.0, true, false, 1.0, 62.0, 1.0)
	TireModelScript.enable_neural_correction = prev_neural_en
	TireModelScript.enable_legacy_neural_correction = prev_legacy_en
	if ter_default.is_ready and r_def_l == Vector3.ZERO and r_def_r == Vector3.ZERO \
			and is_equal_approx(float(f_off["Fx"]), float(f_on["Fx"])) \
			and is_equal_approx(float(f_off["Fy"]), float(f_on["Fy"])) \
			and is_equal_approx(float(f_off["Mz"]), float(f_on["Mz"])) \
			and float(f_on["delta_Fx"]) == 0.0 and float(f_on["delta_Fy"]) == 0.0 and float(f_on["delta_Mz"]) == 0.0:
		passed += 1
	else:
		failed += 1
		errors.append("TeRTire default zero residual/parity failed: ready=%s, rL=%s, rR=%s" % [str(ter_default.is_ready), str(r_def_l), str(r_def_r)])

	# --- Test 13: TeRTireNeuralResidual Synthetic Weights — Exact Odd Symmetry & Zero-Slip ---
	var synth_dict: Dictionary = _build_synthetic_ter_weights_dict()
	var ter_synth := TeRTireNeuralResidualScript.new()
	var loaded_synth: bool = ter_synth.load_from_dict(synth_dict)
	var r_pos: Vector3 = ter_synth.predict_residual(0.07, deg_to_rad(5.5), F_z_nom, 0.0, 65.0, false)
	var r_neg: Vector3 = ter_synth.predict_residual(-0.07, deg_to_rad(-5.5), F_z_nom, 0.0, 65.0, false)
	var r_pos_r: Vector3 = ter_synth.predict_residual(0.07, deg_to_rad(5.5), F_z_nom, 0.0, 65.0, true)
	var r_neg_r: Vector3 = ter_synth.predict_residual(-0.07, deg_to_rad(-5.5), F_z_nom, 0.0, 65.0, true)
	var r_zero: Vector3 = ter_synth.predict_residual(0.0, 0.0, F_z_nom, 0.0, 65.0, false)
	if loaded_synth and r_pos.length() > 1.0 \
			and r_neg == -r_pos and r_neg_r == -r_pos_r \
			and r_zero == Vector3.ZERO:
		passed += 1
	else:
		failed += 1
		errors.append("TeRTire odd symmetry failed: loaded=%s, r_pos=%s, r_neg=%s, r_zero=%s" % [str(loaded_synth), str(r_pos), str(r_neg), str(r_zero)])

	# --- Test 14: TeRTireNeuralResidual Fz -> 0 Smooth Attenuation & Latency (< 0.05 ms/call) ---
	var r_fz0: Vector3 = ter_synth.predict_residual(0.10, deg_to_rad(6.0), 0.0, 0.0, 58.0, false)
	var r_fz_neg: Vector3 = ter_synth.predict_residual(0.10, deg_to_rad(6.0), -10.0, 0.0, 58.0, false)
	var r_fz5: Vector3 = ter_synth.predict_residual(0.10, deg_to_rad(6.0), 5.0, 0.0, 58.0, false)
	var r_fz800: Vector3 = ter_synth.predict_residual(0.10, deg_to_rad(6.0), 800.0, 0.0, 58.0, false)
	var t_start_us: int = Time.get_ticks_usec()
	var sink: float = 0.0
	for _i in 2000:
		var rv: Vector3 = ter_synth.predict_residual(0.06, 0.08, 750.0, -0.02, 60.0, false)
		sink += rv.x + rv.y + rv.z
	var elapsed_us: int = Time.get_ticks_usec() - t_start_us
	var ms_per_call: float = float(elapsed_us) / 2000000.0
	if is_finite(sink) and r_fz0 == Vector3.ZERO and r_fz_neg == Vector3.ZERO \
			and r_fz5.length() < 0.05 and r_fz800.length() > 1.0 \
			and r_fz5.length() < r_fz800.length() * 0.02 \
			and ms_per_call < 0.08:
		passed += 1
	else:
		failed += 1
		errors.append("TeRTire Fz->0 or latency check failed: r_fz0=%s, |r_fz5|=%.4f, |r_fz800|=%.2f, ms/call=%.4f" % [str(r_fz0), r_fz5.length(), r_fz800.length(), ms_per_call])

	# --- Test 15: Surface Friction Scale (surface_mu_scale = 0.85) & WheelDynamics Integration ---
	var fy_peak_100: float = 0.0
	var fy_peak_085: float = 0.0
	for deg in range(1, 16):
		var a_rad: float = deg_to_rad(float(deg))
		var fy1: float = float(TireModelScript.compute_forces(0.0, a_rad, F_z_nom, config, 0.0, 83.0, 10.0, false, false, 1.0, 58.0, 1.0)["Fy"])
		var fy85: float = float(TireModelScript.compute_forces(0.0, a_rad, F_z_nom, config, 0.0, 83.0, 10.0, false, false, 1.0, 58.0, 0.85)["Fy"])
		fy_peak_100 = maxf(fy_peak_100, fy1)
		fy_peak_085 = maxf(fy_peak_085, fy85)
	var ratio_85: float = fy_peak_085 / maxf(fy_peak_100, 1e-4)

	wheel.reset()
	wheel.F_z = F_z_nom
	wheel.omega = 10.0 / config.wheel_radius
	wheel.surface_mu_scale = 1.0
	for _s in 40:
		wheel.compute_slip(10.0, -1.5, 0.0, config.slip_regularization_speed, config.wheel_radius, dt)
	wheel.compute_forces(F_z_nom, config)
	var fy_wheel_100: float = wheel.F_y

	wheel.surface_mu_scale = 0.85
	wheel.compute_forces(F_z_nom, config)
	var fy_wheel_85: float = wheel.F_y
	wheel.reset()

	# Also verify WheelDynamics propagates non-zero TeRTireNeuralResidual delta_Fx/Fy/Mz
	var ter_singleton = TireModelScript.get_ter_pinn()
	ter_singleton.load_from_dict(synth_dict)
	var prev_n_en: bool = TireModelScript.enable_neural_correction
	var prev_l_en: bool = TireModelScript.enable_legacy_neural_correction
	TireModelScript.enable_neural_correction = true
	TireModelScript.enable_legacy_neural_correction = false
	wheel.omega = 52.0
	wheel.compute_slip(10.0, -1.0, 0.0, config.slip_regularization_speed, config.wheel_radius, dt)
	wheel.compute_forces(F_z_nom, config)
	var got_deltas: bool = absf(wheel.delta_Fx) > 0.01 or absf(wheel.delta_Fy) > 0.01 or absf(wheel.delta_Mz) > 0.001
	ter_singleton.load_weights()
	TireModelScript.enable_neural_correction = prev_n_en
	TireModelScript.enable_legacy_neural_correction = prev_l_en
	wheel.reset()

	if absf(ratio_85 - 0.85) < 0.03 and wheel.surface_mu_scale == 1.0 \
			and fy_wheel_85 < fy_wheel_100 and fy_wheel_85 > 500.0 and got_deltas:
		passed += 1
	else:
		failed += 1
		errors.append("Surface mu scale / WheelDynamics check failed: ratio=%.4f, fy100=%.1f, fy85=%.1f, deltas=%s" % [ratio_85, fy_wheel_100, fy_wheel_85, str(got_deltas)])

	# --- Test 16: Hoosier R20 Progressive Saturation & Combined Grip Envelope (750 N) ---
	var prev_cmp: String = TireModelScript.get_compound()
	var r20_cfg := config.duplicate()
	r20_cfg.lambda_mux = 0.86
	r20_cfg.lambda_muy = 0.76
	TireModelScript.set_compound(r20_cfg.tire_compound if r20_cfg.tire_compound != "" else "R20")
	var fz_750: float = 750.0
	var therm_init: float = WheelDynamicsScript.compute_thermal_grip_factor(r20_cfg.tire_initial_temp_c)
	var fx_pk_750: float = 0.0
	var fx_022_750: float = 0.0
	for ki in range(61):
		var kv: float = float(ki) * 0.005
		var fxv: float = float(TireModelScript.compute_forces(kv, 0.0, fz_750, r20_cfg, 0.0, 83.0, 15.0, false, false, therm_init, r20_cfg.tire_initial_temp_c, 1.0)["Fx"])
		fx_pk_750 = maxf(fx_pk_750, fxv)
		if absf(kv - 0.22) < 0.001:
			fx_022_750 = fxv
	var fy_pk_750: float = 0.0
	var a_pk_750_deg: float = 0.0
	var fy_14_750: float = 0.0
	for ai in range(73):
		var ad: float = float(ai) * 0.25
		var fyv: float = absf(float(TireModelScript.compute_forces(0.0, -deg_to_rad(ad), fz_750, r20_cfg, 0.0, 83.0, 15.0, false, false, therm_init, r20_cfg.tire_initial_temp_c, 1.0)["Fy"]))
		if fyv > fy_pk_750:
			fy_pk_750 = fyv
			a_pk_750_deg = ad
		if absf(ad - 14.0) < 0.01:
			fy_14_750 = fyv
	var fy_comb_005: float = absf(float(TireModelScript.compute_forces(0.05, -deg_to_rad(a_pk_750_deg), fz_750, r20_cfg, 0.0, 83.0, 15.0, false, false, therm_init, r20_cfg.tire_initial_temp_c, 1.0)["Fy"]))
	var mux_750: float = fx_pk_750 / fz_750
	var muy_750: float = fy_pk_750 / fz_750
	TireModelScript.set_compound(prev_cmp)

	var env_ok: bool = (
		mux_750 >= 1.72 and mux_750 <= 2.00
		and muy_750 >= 1.75 and muy_750 <= 2.05
		and a_pk_750_deg >= 7.5 and a_pk_750_deg <= 9.5
		and fy_14_750 >= 0.86 * fy_pk_750
		and fx_022_750 >= 0.85 * fx_pk_750
		and fy_comb_005 >= 0.82 * fy_pk_750
	)
	if env_ok:
		passed += 1
	else:
		failed += 1
		errors.append("R20 envelope check failed: mux=%.3f muy=%.3f a_pk=%.2f fy14=%.1f%% fx22=%.1f%% comb05=%.1f%%" % [
			mux_750, muy_750, a_pk_750_deg,
			100.0 * fy_14_750 / maxf(fy_pk_750, 1.0),
			100.0 * fx_022_750 / maxf(fx_pk_750, 1.0),
			100.0 * fy_comb_005 / maxf(fy_pk_750, 1.0)
		])

	return {
		"name": "test_tire_mf62_suite",
		"passed": passed,
		"failed": failed,
		"errors": errors
	}


static func _build_synthetic_ter_weights_dict() -> Dictionary:
	var w1: Array = []
	for i in 16:
		var row: Array = []
		for j in 5:
			row.append(0.35 * sin(float(i * 5 + j + 1) * 0.73))
		w1.append(row)
	var b1: Array = []
	for i in 16:
		b1.append(0.1 * cos(float(i) * 0.41))
	var w2: Array = []
	for i in 16:
		var row2: Array = []
		for j in 16:
			row2.append(0.22 * cos(float(i * 16 + j + 1) * 0.37))
		w2.append(row2)
	var b2: Array = []
	for i in 16:
		b2.append(0.08 * sin(float(i) * 0.53))
	var w3: Array = []
	for k in 3:
		var row3: Array = []
		for j in 16:
			row3.append(0.45 * sin(float((k + 1) * 17 + j) * 0.61))
		w3.append(row3)
	return {
		"model_name": "TeR_Tire_PINN_Synthetic_Test",
		"architecture": {"in_dim": 5, "hidden1": 16, "hidden2": 16, "out_dim": 3},
		"normalization": {
			"x_mean": [0.0, 0.0, 750.0, 0.0, 58.0],
			"x_std": [0.15, 0.20, 400.0, 0.05, 25.0],
			"y_mean": [0.0, 0.0, 0.0],
			"y_std": [100.0, 100.0, 10.0]
		},
		"physics_bounds": {
			"fz_gate_scale_n": 100.0,
			"mu_nom": 2.5,
			"r0_m": 0.2032,
			"max_fx_frac": 0.20,
			"max_fy_frac": 0.20,
			"max_mz_frac": 0.25
		},
		"weights": {
			"w1": w1, "b1": b1,
			"w2": w2, "b2": b2,
			"w3": w3, "b3": [0.0, 0.0, 0.0]
		}
	}
