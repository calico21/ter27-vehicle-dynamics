#ifndef MIRENA_STATE_EKF_HPP
#define MIRENA_STATE_EKF_HPP

#include <Eigen/Dense>

namespace mirena
{

/**
 * Extended Kalman Filter over the planar car state.
 *
 * State vector (same order as mirena_common/Car covariance, 6x6 row-major):
 *   x[0] = x      position in odom frame [m]
 *   x[1] = y      position in odom frame [m]
 *   x[2] = psi    yaw in odom frame [rad]
 *   x[3] = u      body-frame longitudinal velocity (REP-103 x, forward) [m/s]
 *   x[4] = v      body-frame lateral velocity (REP-103 y, left) [m/s]
 *   x[5] = omega  yaw rate [rad/s]
 *
 * Pure Eigen, no ROS types: unit-testable in isolation.
 */
class Ekf
{
public:
    using Vec6 = Eigen::Matrix<double, 6, 1>;
    using Mat6 = Eigen::Matrix<double, 6, 6>;

    struct Config
    {
        Vec6 process_noise_diag;    // continuous-time process noise q (scaled by dt each predict)
        Vec6 odom_meas_floor_diag;  // minimum measurement variance per odom-update axis
        double gyro_meas_var;       // variance of the gyro-as-measurement omega update
        double max_dt;              // predict steps longer than this are clamped [s]
    };

    explicit Ekf(const Config &config) : cfg_(config) { reset(); }

    void reset();
    bool is_initialized() const { return initialized_; }

    /// Seed state + covariance from a full measurement (first INS odometry fix).
    void init(const Vec6 &z, const Vec6 &r_diag, double stamp);

    /**
     * Propagate the state to `stamp` using body-frame accelerations (gravity-free,
     * REP-103: ax forward, ay left). Kinematic bicycle-free mechanization:
     *   x'   = u cos(psi) - v sin(psi)     u' = ax + v*omega
     *   y'   = u sin(psi) + v cos(psi)     v' = ay - u*omega
     *   psi' = omega                       omega' = 0
     */
    void predict(double stamp, double ax, double ay);

    /// Fuse a gyro yaw-rate reading as a scalar measurement of omega.
    void update_gyro(double omega_meas);

    /**
     * Fuse a full INS odometry measurement z = [x, y, psi, u, v, omega].
     * `r_diag` is the measurement variance per axis (floored by config).
     * The psi innovation is wrapped to [-pi, pi].
     */
    void update_odom(const Vec6 &z, const Vec6 &r_diag);

    const Vec6 &state() const { return x_; }
    const Mat6 &covariance() const { return P_; }
    double stamp() const { return stamp_; }

    static double wrap_angle(double a);

private:
    void joseph_symmetrize();

    Config cfg_;
    Vec6 x_;
    Mat6 P_;
    double stamp_ = 0.0;
    bool initialized_ = false;
};

} // namespace mirena

#endif // MIRENA_STATE_EKF_HPP
