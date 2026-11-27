class_name TransientRelaxation
extends RefCounted

## transient tire relaxation dynamics for mirena dynamics
## implements first-order contact patch relaxation dynamics (pacejka ch. 7)
## for longitudinal (sigma_kappa) and lateral (sigma_alpha) contact patch deflection
## includes robust standstill stabilization to eliminate chatter and spurious forces at rest

## nominal relaxation lengths for hoosier 16.0x7.5-10 racing slick (m)
const SIGMA_ALPHA_NOMINAL: float = 0.180 # lateral relaxation length at nominal load (m)
const SIGMA_KAPPA_NOMINAL: float = 0.100 # longitudinal relaxation length at nominal load (m)
const V_LOW_SPEED_REG: float = 0.50     # speed below which kinematic slip is attenuated (m/s)
const V_MIN_DAMPING: float = 1.00       # baseline damping speed at standstill (m/s)

## calculates load-dependent lateral relaxation length sigma_alpha (m)
static func get_lateral_relaxation_length(f_z: float, f_z0: float = 800.0) -> float:
	var dfz = (f_z - f_z0) / maxf(f_z0, 1.0)
	var sigma = SIGMA_ALPHA_NOMINAL * (1.0 + 0.25 * dfz)
	return clampf(sigma, 0.06, 0.40)

## calculates load-dependent longitudinal relaxation length sigma_kappa (m)
static func get_longitudinal_relaxation_length(f_z: float, f_z0: float = 800.0) -> float:
	var dfz = (f_z - f_z0) / maxf(f_z0, 1.0)
	var sigma = SIGMA_KAPPA_NOMINAL * (1.0 + 0.20 * dfz)
	return clampf(sigma, 0.04, 0.25)

## integrates transient slip angle alpha_t using exact exponential discretization:
## d(alpha_t)/dt = (|vx| / sigma_alpha) * (alpha_target - alpha_t)
## at near-zero speed, target slip is smoothly attenuated to prevent standstill drift
static func integrate_alpha_relaxation(
	alpha_current: float,
	alpha_target: float,
	v_x: float,
	dt: float,
	f_z: float,
	f_z0: float = 800.0
) -> float:
	if dt <= 0.0:
		return alpha_current
	
	var abs_vx = absf(v_x)
	var sigma_alpha = get_lateral_relaxation_length(f_z, f_z0)
	
	# standstill attenuation: as speed approaches zero, target slip angle smoothly vanishes
	var speed_factor = clampf(abs_vx / V_LOW_SPEED_REG, 0.0, 1.0)
	var target_conditioned = alpha_target * speed_factor
	
	# effective velocity ensures gentle dissipation of residual deflection at standstill
	var v_eff = maxf(abs_vx, V_MIN_DAMPING)
	var rate = (v_eff / sigma_alpha) * dt
	var decay = exp(-rate)
	
	return target_conditioned + (alpha_current - target_conditioned) * decay

## integrates transient longitudinal slip kappa_t using exact exponential discretization:
## d(kappa_t)/dt = (|vx| / sigma_kappa) * (kappa_target - kappa_t)
static func integrate_kappa_relaxation(
	kappa_current: float,
	kappa_target: float,
	v_x: float,
	dt: float,
	f_z: float,
	f_z0: float = 800.0
) -> float:
	if dt <= 0.0:
		return kappa_current
	
	var abs_vx = absf(v_x)
	var sigma_kappa = get_longitudinal_relaxation_length(f_z, f_z0)
	
	var speed_factor = clampf(abs_vx / V_LOW_SPEED_REG, 0.0, 1.0)
	var target_conditioned = kappa_target * speed_factor
	
	var v_eff = maxf(abs_vx, V_MIN_DAMPING)
	var rate = (v_eff / sigma_kappa) * dt
	var decay = exp(-rate)
	
	return target_conditioned + (kappa_current - target_conditioned) * decay
