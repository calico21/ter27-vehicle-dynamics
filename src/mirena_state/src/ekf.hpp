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

    /// Outcome of a filter step, so the caller can log the two failure kinds apart.
    enum class Step
    {
        Ok,         ///< applied
        Ignored,    ///< input was unusable (non-finite, or out-of-order in time); state untouched
        Diverged,   ///< state or covariance went non-finite; the filter reset itself
    };

    explicit Ekf(const Config &config) : cfg_(config) { reset(); }

    void reset();
    bool is_initialized() const { return initialized_; }

    /// Times the filter has reset itself after divergence, since construction.
    unsigned divergences() const { return divergences_; }

    /// Seed state + covariance from a full measurement (first INS odometry fix).
    Step init(const Vec6 &z, const Vec6 &r_diag, double stamp);

    /**
     * Propagate the state to `stamp` using body-frame accelerations (gravity-free,
     * REP-103: ax forward, ay left). Kinematic bicycle-free mechanization:
     *   x'   = u cos(psi) - v sin(psi)     u' = ax + v*omega
     *   y'   = u sin(psi) + v cos(psi)     v' = ay - u*omega
     *   psi' = omega                       omega' = 0
     *
     * Samples that are out of order or duplicate in time are Ignored, and
     * crucially leave the filter clock where it was: adopting their stamp would
     * let one bogus future timestamp disable prediction for the rest of the run.
     */
    Step predict(double stamp, double ax, double ay);

    /// Fuse a gyro yaw-rate reading as a scalar measurement of omega.
    Step update_gyro(double omega_meas);

    /**
     * Fuse a full INS odometry measurement z = [x, y, psi, u, v, omega].
     * `r_diag` is the measurement variance per axis (floored by config).
     * The psi innovation is wrapped to [-pi, pi].
     */
    Step update_odom(const Vec6 &z, const Vec6 &r_diag);

    const Vec6 &state() const { return x_; }
    const Mat6 &covariance() const { return P_; }
    double stamp() const { return stamp_; }

    static double wrap_angle(double a);

private:
    void symmetrize();

    /// Post-step sanity check. A single NaN anywhere in x_ or P_ is permanent --
    /// it spreads through every subsequent product -- so the only recovery is to
    /// drop the filter and let the next INS fix re-seed it.
    Step commit();

    Config cfg_;
    Vec6 x_;
    Mat6 P_;
    double stamp_ = 0.0;
    bool initialized_ = false;
    unsigned divergences_ = 0;
};

} // namespace mirena

#endif // MIRENA_STATE_EKF_HPP
