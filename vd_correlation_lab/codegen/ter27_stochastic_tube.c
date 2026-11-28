#include "ter27_stochastic_tube.h"
#include <math.h>

static float stube_clampf(float x, float lo, float hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

static float stube_maxf(float a, float b) {
    return (a > b) ? a : b;
}

static float stube_minf(float a, float b) {
    return (a < b) ? a : b;
}

void ter27_stube_default_params(Ter27StochasticTubeParams *p) {
    p->z_quantile = 3.7190165f;
    p->min_cone_clearance_m = 0.150f;
    p->track_half_width_m = 1.500f;
    p->vehicle_half_width_m = 0.720f;
    p->friction_limit_nom = 1.000f;
    p->friction_limit_min = 0.680f;
    p->friction_backoff_gain = 0.580f;
    p->p_dc_max_kw = 80.0f;
    p->p_dc_regen_min_kw = -25.0f;
    p->p_dc_hard_margin_kw = 0.05f;
    p->p_dc_backoff_max_kw = 6.0f;
    p->cone_backoff_max_m = 0.550f;
    p->speed_backoff_gain = 0.450f;
    p->k_ey_fb = 4.2f;
    p->k_epsi_fb = 6.5f;
    p->k_vy_fb = 8.0f;
    p->k_wz_fb = 9.5f;
    p->k_innov_var = 0.0028f;
    p->can_accel_latency_s = 0.125f;
    p->can_steer_latency_s = 0.020f;
    p->can_jitter_s = 0.015f;
    p->imu_noise_scale = 1.0f;
    p->ay_noise_std_ms2 = 0.18f;
    p->wz_noise_std_rads = 0.012f;
    p->omega_noise_std_rads = 0.08f;
    p->accel_quant_ms2 = 0.01f;
}

void ter27_stube_reset(Ter27StochasticTubeState *st) {
    st->p_yy = 0.0004f;
    st->p_yp = 0.0f;
    st->p_pp = 0.0001f;
    st->p_vy = 0.0025f;
    st->p_wz = 0.0004f;
}

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
) {
    float dt_safe = stube_clampf(dt_s, 0.0005f, 0.025f);
    float vx_eff = stube_maxf(fabsf(vx_ms), 1.5f);
    float lam_safe = stube_maxf(lambda_mu_hat, 0.25f);
    float ay_abs = fabsf(ay_ms2);

    /* Closed-loop stabilizing contraction factors (A_k - B_k K_k), preserving dMz/dbeta < 0 */
    float rho_y = stube_clampf(1.0f - p->k_ey_fb * dt_safe, 0.70f, 0.995f);
    float rho_p = stube_clampf(1.0f - p->k_epsi_fb * dt_safe, 0.65f, 0.995f);
    float rho_vy = stube_clampf(1.0f - p->k_vy_fb * dt_safe, 0.60f, 0.990f);
    float rho_wz = stube_clampf(1.0f - p->k_wz_fb * dt_safe, 0.60f, 0.990f);

    /* Disturbance covariance Q_w(sigma_lambda_mu^2, sigma_IMU^2, sigma_CAN^2) */
    float innov_norm = innov_ms2 / stube_maxf(ay_abs, 4.0f);
    float sigma_mu_var = stube_maxf(p_cov_mu, 1.0e-6f) + p->k_innov_var * innov_norm * innov_norm;
    float imu_ay_std = p->imu_noise_scale * p->ay_noise_std_ms2;
    float imu_wz_std = p->imu_noise_scale * p->wz_noise_std_rads;
    float sigma_imu_ay_var = imu_ay_std * imu_ay_std;
    float ay_dot_rms = stube_maxf(ay_abs * 1.8f, 3.0f);
    float can_ay_std = (p->can_accel_latency_s + p->can_jitter_s) * ay_dot_rms;
    float sigma_can_ay_var = can_ay_std * can_ay_std + (p->accel_quant_ms2 * p->accel_quant_ms2) / 12.0f;
    float g_mu_ay = stube_maxf(ay_abs, 2.0f) / lam_safe;
    float q_ay = g_mu_ay * g_mu_ay * sigma_mu_var + sigma_imu_ay_var + sigma_can_ay_var;

    float g_mu_wz = (0.45f * g_mu_ay) / stube_maxf(vx_eff, 3.0f);
    float wz_dot_rms = stube_maxf(ay_abs * 0.25f, 0.4f);
    float can_wz_std = (p->can_steer_latency_s + 0.5f * p->can_jitter_s) * wz_dot_rms;
    float q_wz = g_mu_wz * g_mu_wz * sigma_mu_var + imu_wz_std * imu_wz_std + can_wz_std * can_wz_std;

    /* Propagate diagonal velocity variances + exact 2x2 (e_y, e_psi) tracking covariance block */
    float dt2 = dt_safe * dt_safe;
    st->p_vy = stube_clampf(rho_vy * rho_vy * st->p_vy + q_ay * dt2, 1.0e-6f, 4.0f);
    st->p_wz = stube_clampf(rho_wz * rho_wz * st->p_wz + q_wz * dt2, 1.0e-7f, 2.0f);

    float a01 = vx_eff * dt_safe;
    float p_yy_n = rho_y * rho_y * st->p_yy + 2.0f * rho_y * a01 * st->p_yp
        + a01 * a01 * st->p_pp + st->p_vy * dt2 * 6.0f;
    float p_yp_n = rho_y * rho_p * st->p_yp + a01 * rho_p * st->p_pp;
    float p_pp_n = rho_p * rho_p * st->p_pp + st->p_wz * dt2 * 6.0f;
    st->p_yy = stube_clampf(p_yy_n, 1.0e-7f, 0.50f);
    st->p_yp = stube_clampf(p_yp_n, -0.25f, 0.25f);
    st->p_pp = stube_clampf(p_pp_n, 1.0e-7f, 0.25f);

    /* Projected chance-constraint variances c^T * Sigma_k * c */
    float var_ey = st->p_yy;
    float var_fric = sigma_mu_var / (lam_safe * lam_safe)
        + 0.015f * st->p_vy
        + 0.040f * st->p_wz
        + 0.0080f * (sigma_imu_ay_var + sigma_can_ay_var) / (9.81f * 9.81f);
    float om_std = p->imu_noise_scale * p->omega_noise_std_rads;
    float pdc_rel = p_dc_nom_kw * 0.004f;
    float pdc_can = p->can_accel_latency_s * 2.5f;
    float var_pdc = pdc_rel * pdc_rel + 0.35f * om_std * om_std + 0.60f * pdc_can * pdc_can;

    float sig_ey = sqrtf(stube_maxf(var_ey, 0.0f));
    float sig_fr = sqrtf(stube_maxf(var_fric, 0.0f));
    float sig_p = sqrtf(stube_maxf(var_pdc, 0.0f));

    /* Gaussian quantile backoffs b_k = Phi^{-1}(1 - eps) * sqrt(c^T Sigma_k c) */
    float b_cone = stube_clampf(p->z_quantile * sig_ey, 0.0f, p->cone_backoff_max_m);
    float max_f_drop = stube_maxf(p->friction_limit_nom - p->friction_limit_min, 0.01f);
    float b_fric = stube_clampf(p->z_quantile * p->friction_backoff_gain * sig_fr, 0.0f, max_f_drop);
    float fric_scale = stube_clampf(
        p->friction_limit_nom - b_fric,
        p->friction_limit_min,
        p->friction_limit_nom
    );
    float b_pdc = stube_clampf(p->z_quantile * sig_p, 0.0f, p->p_dc_backoff_max_kw);

    float p_max_hard = p->p_dc_max_kw - p->p_dc_hard_margin_kw;
    float p_min_hard = p->p_dc_regen_min_kw + p->p_dc_hard_margin_kw;
    float p_max_eff = stube_clampf(p_max_hard - b_pdc, 20.0f, p_max_hard);
    float p_reg_eff = stube_clampf(p_min_hard + 0.5f * b_pdc, p_min_hard, -5.0f);

    float max_nom_ey = stube_maxf(
        p->track_half_width_m - p->vehicle_half_width_m - p->min_cone_clearance_m,
        0.05f
    );
    float tube_rat = b_cone / max_nom_ey;
    float v_from_f = sqrtf(stube_maxf(fric_scale, 0.25f));
    float v_from_c = stube_clampf(
        1.0f - p->speed_backoff_gain * stube_maxf(tube_rat - 0.12f, 0.0f),
        0.70f,
        1.0f
    );
    float v_scale = stube_clampf(stube_minf(v_from_f, v_from_c), 0.70f, 1.0f);

    out->b_cone_m = b_cone;
    out->b_friction = b_fric;
    out->b_pdc_kw = b_pdc;
    out->friction_ellipse_scale = fric_scale;
    out->p_dc_max_eff_kw = p_max_eff;
    out->p_dc_regen_eff_kw = p_reg_eff;
    out->v_ref_scale = v_scale;
    out->sigma_ey_m = sig_ey;
    out->sigma_fric_norm = sig_fr;
    out->sigma_pdc_kw = sig_p;
}
