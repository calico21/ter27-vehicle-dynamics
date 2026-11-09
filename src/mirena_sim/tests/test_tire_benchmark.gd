extends SceneTree

## Performance Benchmark for Mirena Dynamics Hybrid Tire Model
## Evaluates computational throughput and real-time latency at 200 Hz for:
## 1. Pure Pacejka MF 6.2
## 2. Hybrid (MF 6.2 + PINN Residual Correction)
## 3. Full Model (MF 6.2 + PINN Residual + Transient Relaxation)

const TireModelScript = preload("res://Scenes/Vehicle/Dynamics/Tires/TireModel.gd")
const WheelDynamicsScript = preload("res://Scenes/Vehicle/Dynamics/Tires/WheelDynamics.gd")
const VehicleConfigScript = preload("res://Scenes/Vehicle/Dynamics/Core/VehicleConfig.gd")

func _init() -> void:
	print("\n========================================================")
	print("    MIRENA DYNAMICS — TIRE MODEL PERFORMANCE BENCHMARK   ")
	print("========================================================\n")
	
	var config = VehicleConfigScript.new()
	var wheel = WheelDynamicsScript.new(0, "FL")
	
	var iterations = 50000
	var fz = 800.0
	var alpha = 0.08 # ~4.5 deg
	var kappa = 0.05
	var dt = 0.005 # 200 Hz physics step
	
	# Warmup JIT / cache
	for i in range(1000):
		wheel.compute_slip(10.0, -1.0, 0.0, 0.5, 0.20, dt)
		wheel.compute_forces(fz, config)
	
	# --- Benchmark 1: Pure MF 6.2 ---
	TireModelScript.enable_neural_correction = false
	TireModelScript.enable_transient_relaxation = false
	
	var t0 = Time.get_ticks_usec()
	for i in range(iterations):
		wheel.compute_slip(10.0, -1.0, 0.0, 0.5, 0.20, dt)
		wheel.compute_forces(fz, config)
	var t1 = Time.get_ticks_usec()
	var us_per_call_mf = float(t1 - t0) / float(iterations)
	var evals_sec_mf = 1000000.0 / us_per_call_mf
	
	# --- Benchmark 2: Hybrid (MF 6.2 + PINN Residual) ---
	TireModelScript.enable_neural_correction = true
	TireModelScript.enable_transient_relaxation = false
	
	t0 = Time.get_ticks_usec()
	for i in range(iterations):
		wheel.compute_slip(10.0, -1.0, 0.0, 0.5, 0.20, dt)
		wheel.compute_forces(fz, config)
	t1 = Time.get_ticks_usec()
	var us_per_call_hyb = float(t1 - t0) / float(iterations)
	var evals_sec_hyb = 1000000.0 / us_per_call_hyb
	
	# --- Benchmark 3: Full (MF 6.2 + PINN Residual + Transient Relaxation) ---
	TireModelScript.enable_neural_correction = true
	TireModelScript.enable_transient_relaxation = true
	
	t0 = Time.get_ticks_usec()
	for i in range(iterations):
		wheel.compute_slip(10.0, -1.0, 0.0, 0.5, 0.20, dt)
		wheel.compute_forces(fz, config)
	t1 = Time.get_ticks_usec()
	var us_per_call_full = float(t1 - t0) / float(iterations)
	var evals_sec_full = 1000000.0 / us_per_call_full
	
	# 200 Hz budget: 5000 us total physics tick. Target: 4 tires < 50 us (< 1% of budget)
	var tick_budget_200hz = 5000.0 # us
	
	print("Iteration Count: %d per mode (Headless CPU)" % iterations)
	print("--------------------------------------------------------")
	print("1. Pure Pacejka MF 6.2:")
	print("   Latency per tire call:  %.3f us (microseconds)" % us_per_call_mf)
	print("   Throughput:             %.0f evaluations/sec" % evals_sec_mf)
	print("   4-Tire Step at 200 Hz:  %.3f us (%.2f%% of 5000 us budget)" % [us_per_call_mf * 4.0, (us_per_call_mf * 4.0 / tick_budget_200hz) * 100.0])
	print("")
	print("2. Hybrid (MF 6.2 + PINN Residual):")
	print("   Latency per tire call:  %.3f us" % us_per_call_hyb)
	print("   Throughput:             %.0f evaluations/sec" % evals_sec_hyb)
	print("   4-Tire Step at 200 Hz:  %.3f us (%.2f%% of 5000 us budget)" % [us_per_call_hyb * 4.0, (us_per_call_hyb * 4.0 / tick_budget_200hz) * 100.0])
	print("")
	print("3. Full Model (MF 6.2 + PINN Residual + Transient Relaxation):")
	print("   Latency per tire call:  %.3f us" % us_per_call_full)
	print("   Throughput:             %.0f evaluations/sec" % evals_sec_full)
	print("   4-Tire Step at 200 Hz:  %.3f us (%.2f%% of 5000 us budget)" % [us_per_call_full * 4.0, (us_per_call_full * 4.0 / tick_budget_200hz) * 100.0])
	print("--------------------------------------------------------")
	
	# Requirement: Target is 800 evaluations/s.
	if evals_sec_full >= 800.0:
		print(">> REAL-TIME 200 Hz BENCHMARK: PASSED (Exceeds 800 evals/s by %.1fx!) <<" % [evals_sec_full / 800.0])
	else:
		print(">> REAL-TIME 200 Hz BENCHMARK: FAILED (Below 800 evals/s)")
	print("========================================================\n")
	
	# Save benchmark results to json
	var bench_data = {
		"iterations": iterations,
		"target_evals_per_sec": 800,
		"pure_mf62": {
			"us_per_call": us_per_call_mf,
			"evals_per_sec": evals_sec_mf,
			"four_tires_pct_200hz_tick": (us_per_call_mf * 4.0 / tick_budget_200hz) * 100.0
		},
		"hybrid_pinn": {
			"us_per_call": us_per_call_hyb,
			"evals_per_sec": evals_sec_hyb,
			"four_tires_pct_200hz_tick": (us_per_call_hyb * 4.0 / tick_budget_200hz) * 100.0
		},
		"full_model": {
			"us_per_call": us_per_call_full,
			"evals_per_sec": evals_sec_full,
			"four_tires_pct_200hz_tick": (us_per_call_full * 4.0 / tick_budget_200hz) * 100.0
		}
	}
	var f = FileAccess.open("res://Scenes/Vehicle/Dynamics/Tires/Validation/inference_benchmark.json", FileAccess.WRITE)
	if f:
		f.store_string(JSON.stringify(bench_data, "  "))
	
	quit(0)
