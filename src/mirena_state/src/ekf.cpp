#include "ekf.hpp"

#include <cmath>

namespace mirena
{

double Ekf::wrap_angle(double a)
{
    while (a > M_PI) a -= 2.0 * M_PI;
    while (a < -M_PI) a += 2.0 * M_PI;
    return a;
}

void Ekf::reset()
{
    x_.setZero();
    P_.setIdentity();
    stamp_ = 0.0;
    initialized_ = false;
}

void Ekf::init(const Vec6 &z, const Vec6 &r_diag, double stamp)
{
    x_ = z;
    x_(2) = wrap_angle(x_(2));
    P_ = r_diag.cwiseMax(cfg_.odom_meas_floor_diag).asDiagonal();
    stamp_ = stamp;
    initialized_ = true;
}

void Ekf::predict(double stamp, double ax, double ay)
{
    if (!initialized_) return;

    double dt = stamp - stamp_;
    stamp_ = stamp;
    if (dt <= 0.0) return; // out-of-order or duplicate sample
    if (dt > cfg_.max_dt) dt = cfg_.max_dt;

    const double psi = x_(2), u = x_(3), v = x_(4), omega = x_(5);
    const double c = std::cos(psi), s = std::sin(psi);

    // Nonlinear state propagation (forward Euler)
    x_(0) += (u * c - v * s) * dt;
    x_(1) += (u * s + v * c) * dt;
    x_(2) = wrap_angle(psi + omega * dt);
    x_(3) += (ax + v * omega) * dt;
    x_(4) += (ay - u * omega) * dt;
    // omega: random walk (fused separately via update_gyro)

    // Jacobian F = I + A*dt
    Mat6 F = Mat6::Identity();
    F(0, 2) = (-u * s - v * c) * dt;
    F(0, 3) = c * dt;
    F(0, 4) = -s * dt;
    F(1, 2) = (u * c - v * s) * dt;
    F(1, 3) = s * dt;
    F(1, 4) = c * dt;
    F(2, 5) = dt;
    F(3, 4) = omega * dt;
    F(3, 5) = v * dt;
    F(4, 3) = -omega * dt;
    F(4, 5) = -u * dt;

    P_ = F * P_ * F.transpose();
    P_ += (cfg_.process_noise_diag * dt).asDiagonal().toDenseMatrix();
    joseph_symmetrize();
}

void Ekf::update_gyro(double omega_meas)
{
    if (!initialized_) return;

    // Scalar update on omega: H = [0 0 0 0 0 1]
    const double innovation = omega_meas - x_(5);
    const double s_var = P_(5, 5) + cfg_.gyro_meas_var;
    const Vec6 K = P_.col(5) / s_var;

    x_ += K * innovation;
    x_(2) = wrap_angle(x_(2));
    P_ -= K * P_.row(5);
    joseph_symmetrize();
}

void Ekf::update_odom(const Vec6 &z, const Vec6 &r_diag)
{
    if (!initialized_) return;

    // Full-state measurement: H = I6
    Vec6 innovation = z - x_;
    innovation(2) = wrap_angle(innovation(2));

    const Mat6 R = r_diag.cwiseMax(cfg_.odom_meas_floor_diag).asDiagonal();
    const Mat6 S = P_ + R;
    const Mat6 K = P_ * S.inverse();

    x_ += K * innovation;
    x_(2) = wrap_angle(x_(2));
    P_ = (Mat6::Identity() - K) * P_;
    joseph_symmetrize();
}

void Ekf::joseph_symmetrize()
{
    P_ = (0.5 * (P_ + P_.transpose())).eval();
}

} // namespace mirena
