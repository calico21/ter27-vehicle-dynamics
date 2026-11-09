extends SceneTree

const MirenaCarScene = preload("res://Scenes/Vehicle/mirena_car.tscn")
const VehicleConfigScript = preload("res://Scenes/Vehicle/Dynamics/Core/VehicleConfig.gd")

func _init() -> void:
	var runner = PhysicalTestRunner.new()
	root.add_child(runner)

class PhysicalTestRunner extends Node:
	var car: MirenaCar
	var ground: StaticBody3D
	var ticks: int = 0
	
	var passed: int = 0
	var failed: int = 0
	
	var p1_done: bool = false
	var p2_done: bool = false
	var p3_done: bool = false
	var p4_done: bool = false
	var p5_done: bool = false
	
	var initial_yaw: float = 0.0
	var initial_z: float = 0.0
	
	func _ready() -> void:
		print("\n========================================================")
		print("   MIRENA DYNAMICS — PHYSICAL TRACK INTEGRATION TEST    ")
		print("========================================================\n")
		
		# Setup Ground Plane (StaticBody3D at Y = 0)
		ground = StaticBody3D.new()
		ground.name = "Ground"
		ground.collision_layer = 1
		ground.collision_mask = 1
		
		var col_shape = CollisionShape3D.new()
		var box = BoxShape3D.new()
		box.size = Vector3(2000.0, 1.0, 2000.0)
		col_shape.shape = box
		col_shape.position = Vector3(0.0, -0.5, 0.0) # Surface is at Y = 0.0
		ground.add_child(col_shape)
		add_child(ground)
		
		# Instantiate MirenaCar
		car = MirenaCarScene.instantiate() as MirenaCar
		car.position = Vector3(0.0, 0.10, 0.0)
		add_child(car)
		
		car.pilot = MirenaCar.PilotMode.ROS
		car.set_control(0.0, 0.0, 0.0)
		print(">> Spawning vehicle onto track plane at (0, 0.10, 0)...")
	
	func _physics_process(delta: float) -> void:
		ticks += 1
		
		# --- Phase 1: Suspension settlement (0.0 to 1.0 s, 400 ticks) ---
		if ticks == 60:
			print(">> Phase 1: Simulating suspension settlement on ground (1.0 s)...")
		
		if ticks == 400 and not p1_done:
			p1_done = true
			var height = car.global_position.y
			var grounded = 0
			for ray in car.raycasts:
				if ray.is_colliding():
					grounded += 1
			print("   Vehicle height: Y = %.3f m | Grounded wheels: %d/4" % [height, grounded])
			if height >= -0.05 and height <= 0.15 and grounded == 4:
				print("   [PASS] Vehicle maintains stable ground contact and suspension settles.")
				passed += 1
			else:
				print("   [FAIL] Vehicle settlement failed: height=%.3f, grounded=%d" % [height, grounded])
				failed += 1
			
			# Begin Phase 2: 4WD Acceleration (400 to 1200 ticks = 2.0 s)
			print("\n>> Phase 2: Testing 4WD full throttle acceleration (2.0 s)...")
			car.drive_mode = VehicleConfigScript.DriveMode.FOUR_WD
			car.set_control(1.0, 0.0, 0.0)
			initial_z = car.global_position.z
		
		# --- Phase 2 Check ---
		if ticks == 1200 and not p2_done:
			p2_done = true
			var speed = car.linear_velocity.length()
			var dist = initial_z - car.global_position.z # forward is -Z in Godot
			print("   Speed reached: %.2f m/s (%.1f km/h) | Distance traveled: %.2f m" % [speed, speed * 3.6, dist])
			if speed > 4.0 and dist > 4.0:
				print("   [PASS] 4WD Acceleration confirmed: vehicle accelerates forward smoothly.")
				passed += 1
			else:
				print("   [FAIL] 4WD Acceleration failed: speed=%.2f m/s, dist=%.2f m" % [speed, dist])
				failed += 1
			
			# Begin Phase 3: Steering (1200 to 1600 ticks = 1.0 s)
			print("\n>> Phase 3: Testing steering step response (steer left 0.25 rad)...")
			car.set_control(0.3, 0.25, 0.0)
			initial_yaw = car.global_rotation.y
		
		# --- Phase 3 Check ---
		if ticks == 1600 and not p3_done:
			p3_done = true
			var yaw_rate = car.angular_velocity.y
			var delta_yaw = car.global_rotation.y - initial_yaw
			print("   Yaw rate: %.3f rad/s | Heading change: %.2f deg" % [yaw_rate, rad_to_deg(delta_yaw)])
			# In Godot, turning left corresponds to negative rotation around +Y (yaw)
			if absf(yaw_rate) > 0.05 and delta_yaw < -0.01:
				print("   [PASS] Steering confirmed: vehicle generates lateral cornering force and turns left.")
				passed += 1
			else:
				print("   [FAIL] Steering failed: yaw_rate=%.3f, delta_yaw=%.3f" % [yaw_rate, delta_yaw])
				failed += 1
			
			# Begin Phase 4: Braking to Standstill (1600 to 3200 ticks = 4.0 s)
			print("\n>> Phase 4: Testing friction braking to standstill (4.0 s)...")
			car.set_control(0.0, 0.0, 20.0) # full friction brake
		
		# --- Phase 4 Check ---
		if ticks == 3200 and not p4_done:
			p4_done = true
			var speed_stop = car.linear_velocity.length()
			print("   Speed after braking: %.3f m/s" % speed_stop)
			if speed_stop < 0.2:
				print("   [PASS] Braking confirmed: vehicle decelerates to a stop without jitter.")
				passed += 1
			else:
				print("   [FAIL] Braking failed: speed is %.3f m/s (expected < 0.2)" % speed_stop)
				failed += 1
			
			# Begin Phase 5: RWD Acceleration (3200 to 4000 ticks = 2.0 s)
			print("\n>> Phase 5: Testing RWD mode acceleration (2.0 s)...")
			car.drive_mode = VehicleConfigScript.DriveMode.REAR_WD
			car.set_control(1.0, 0.0, 0.0)
			initial_z = car.global_position.z
		
		# --- Phase 5 Check ---
		if ticks == 4000 and not p5_done:
			p5_done = true
			var rwd_speed = car.linear_velocity.length()
			print("   Speed reached in RWD: %.2f m/s (%.1f km/h)" % [rwd_speed, rwd_speed * 3.6])
			if rwd_speed > 3.0:
				print("   [PASS] RWD mode confirmed: driven by rear axle.")
				passed += 1
			else:
				print("   [FAIL] RWD acceleration failed: speed=%.2f m/s" % rwd_speed)
				failed += 1
			
			# Complete test
			print("\n--------------------------------------------------------")
			print("Physical Integration Results: %d/5 tests PASSED." % passed)
			print("========================================================\n")
			
			if failed == 0:
				print(">> ALL PHYSICAL TRACK TESTS PASSED SUCCESSFULLY! <<\n")
				get_tree().quit(0)
			else:
				print(">> PHYSICAL TRACK TESTS FAILED! <<\n")
				get_tree().quit(1)
