#ifndef TER27_STOCHASTIC_TUBE_H
#define TER27_STOCHASTIC_TUBE_H

/*
 * TeR27 4WD Stochastic Tube-MPC (Chance-Constrained Control)
 * Deterministic float32, 0 B heap allocation, 200 Hz STM32F405RGT6 (168 MHz) C99 module.
 * 1-to-1 parity with Python EmbeddedDiagonalTubePropagator and Godot EcuTorqueVectoring4Wd.gd.
 */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float z_quantile;            /* Phi^{-1}(1 - eps), default 3.7190165f for eps=1e-4 */
    float min_cone_clearance_m;  /* 0.150 m strict FSAE clearance floor */
    float track_half_width_m;    /* 1.500 m lane half-width */
    float vehicle_half_width_m;  /* 0.720 m TeR27 half-width */
    float friction_limit_nom;    /* 0.995 low-uncertainty physical limit utilization */
    float friction_limit_min;    /* 0.680 floor under severe grip/sensor uncertainty */
    float friction_backoff_gain; /* 0.580 normalized std -> ellipse backoff gain */
    float p_dc_max_kw;           /* 80.0 kW strict FSAE HV discharge cap */
    float p_dc_regen_min_kw;     /* -25.0 kW strict HV regen charge cap */
    float p_dc_hard_margin_kw;   /* 0.05 kW hard safety margin */
    float p_dc_backoff_max_kw;   /* 6.0 kW max stochastic HV backoff */
    float cone_backoff_max_m;    /* 0.550 m max lateral tube radius */
    float speed_backoff_gain;    /* 0.450 speed-reference tightening gain */
    float k_ey_fb;               /* 4.2 1/s closed-loop lateral error contraction */
    float k_epsi_fb;             /* 6.5 1/s closed-loop heading error contraction */
    float k_vy_fb;               /* 8.0 1/s closed-loop lateral velocity contraction */
    float k_wz_fb;               /* 9.5 1/s closed-loop yaw rate contraction */
    float k_innov_var;           /* 0.0028 innovation-triggered grip variance inflation */
    float can_accel_latency_s;   /* 0.125 s TeR26/27 CAN accelerometer latency */
    float can_steer_latency_s;   /* 0.020 s CAN steering/gyro latency */
    float can_jitter_s;          /* 0.015 s CAN transport jitter */
    float imu_noise_scale;       /* 1.0 nominal SENSOR_TRACK_REALISM scale */
    float ay_noise_std_ms2;      /* 0.18 m/s^2 accelerometer noise std */
    float wz_noise_std_rads;     /* 0.012 rad/s gyro noise std */
    float omega_noise_std_rads;  /* 0.08 rad/s wheel speed encoder noise std */
    float accel_quant_ms2;       /* 0.01 m/s^2 ADC quantization step */
} Ter27StochasticTubeParams;

typedef struct {
    float p_yy; /* var(e_y) [m^2] */
    float p_yp; /* cov(e_y, e_psi) [m*rad] */
    float p_pp; /* var(e_psi) [rad^2] */
    float p_vy; /* var(v_y) [(m/s)^2] */
    float p_wz; /* var(omega_z) [(rad/s)^2] */
} Ter27StochasticTubeState;

typedef struct {
    float b_cone_m;
    float b_friction;
    float b_pdc_kw;
    float friction_ellipse_scale;
    float p_dc_max_eff_kw;
    float p_dc_regen_eff_kw;
    float v_ref_scale;
    float sigma_ey_m;
    float sigma_fric_norm;
    float sigma_pdc_kw;
} Ter27StochasticTubeOutput;

void ter27_stube_default_params(Ter27StochasticTubeParams *p);
void ter27_stube_reset(Ter27StochasticTubeState *st);
void ter27_stube_step(
    const Ter27StochasticTubeParams *p,
    Ter27StochasticTubeState *st,
    float vx_ms,
    float ay_ms2,
    float lambda_mu_hat,
    float p_cov_mu,
    float innov_ms2,
    float p_dc_nom_kw,
    float dt_s,
    Ter27StochasticTubeOutput *out
);

#ifdef __cplusplus
}
#endif

#endif /* TER27_STOCHASTIC_TUBE_H */
