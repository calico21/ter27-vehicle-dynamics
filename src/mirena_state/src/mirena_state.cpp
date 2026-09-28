#include <rclcpp/rclcpp.hpp>
#include <chrono>
#include <cmath>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

// Transforms
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2/LinearMath/Quaternion.h>

// JSON library for string packaging
#include <nlohmann/json.hpp>

// Interfaces
#include <sensor_msgs/msg/imu.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include "mirena_common/msg/car.hpp"
#include "mirena_common/msg/heartbeat.hpp"
#include "mirena_common/msg/wheel_speeds.hpp"

#include "ekf.hpp"

// Topics and Frames
#define IMU_TOPIC "/sensors/imu"
#define INS_ODOM_TOPIC "/sensors/imu/odom"
#define WHEEL_SPEEDS_TOPIC "/sensors/wheel_speeds"
#define CAR_STATE_TOPIC "/state/car"

using namespace std::chrono_literals;

namespace
{
    /// Frames and rate. Constants rather than parameters: no launch file in this
    /// workspace has ever set them, and the nodes downstream hard-code the same
    /// strings (mirena_perception's CAR_FRAME, mirena_slam's frames).
    constexpr const char *kOdomFrame = "odom";
    constexpr const char *kBaseFrame = "car/cog";
    constexpr auto kReportPeriod = 100ms;

    /// The odom frame is *defined* by the pose the filter starts at, so in odom
    /// that pose has no error by construction. Ekf::init floors this by
    /// init_cov_floor_diag regardless, which is what keeps the position gain
    /// off zero so the INS can still pull once it anchors.
    constexpr double kInitPoseVar = 0.0;

    /// Zero-velocity update. The car stands still in staging while the INS
    /// converges, so "stopped" is a measurement rather than an assumption --
    /// as long as the IMU agrees it is stopped.
    constexpr double kZuptVelVar = 0.0025;   // (0.05 m/s)^2
    constexpr double kUnknownVelVar = 25.0;  // (5 m/s)^2, if it is already rolling
    constexpr double kStationaryAccel = 0.5; // m/s^2, gravity already removed
    constexpr double kStationaryRate = 0.05; // rad/s
} // namespace

/**
 * MirenaState — sensor fusion / state estimation node.
 *
 * Owns /state/car and the odom->base TF: predicts from /sensors/imu (body accel
 * + gyro) and corrects with the INS solution on /sensors/imu/odom
 * (nav_msgs/Odometry) and with the rear wheel speeds on /sensors/wheel_speeds.
 *
 * New sensors (GSS, GPS fix, ...) should be added here as additional EKF
 * measurement updates — never as direct /state/car publishers.
 *
 * Sign conventions: state and Car fields follow REP-103 body axes (x forward,
 * y left, z up): u = forward velocity, v = LEFT-positive lateral velocity,
 * omega = CCW-positive yaw rate. This matches the sim's established Car
 * convention that SLAM/MPC were developed against.
 */
class MirenaState : public rclcpp::Node
{
    // Callback Groups for thread safety
    rclcpp::CallbackGroup::SharedPtr _filter_cb_group;
    rclcpp::CallbackGroup::SharedPtr _report_cb_group;

    // Subscribers
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr _imu_sub;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr _ins_odom_sub;
    rclcpp::Subscription<mirena_common::msg::WheelSpeeds>::SharedPtr _encoder_sub;

    // Publishers
    rclcpp::Publisher<mirena_common::msg::Car>::SharedPtr _car_pub;
    rclcpp::Publisher<mirena_common::msg::Heartbeat>::SharedPtr _report_pub;

    // Transform broadcaster (odom -> base)
    std::unique_ptr<tf2_ros::TransformBroadcaster> _tf_broadcaster;

    // Timers
    rclcpp::TimerBase::SharedPtr _report_timer;

    // Filter state & mutex (shared between filter and report threads)
    std::mutex _filter_mutex;
    std::unique_ptr<mirena::Ekf> _ekf;
    rclcpp::Time _last_imu_stamp{0, 0, RCL_ROS_TIME};
    rclcpp::Time _last_odom_stamp{0, 0, RCL_ROS_TIME};
    rclcpp::Time _last_encoder_stamp{0, 0, RCL_ROS_TIME};

    /// Stamp of the last state actually published. The state is published
    /// synchronously on every IMU prediction/update step.
    double _last_published_stamp = 0.0;

    /// T_odom <- ins, latched once the INS first speaks.
    bool _ins_anchored = false;
    double _anchor_tx = 0.0, _anchor_ty = 0.0, _anchor_dpsi = 0.0;

    // Parameters (resolved in the constructor)
    double _odom_timeout_sec = 0.5;
    double _imu_timeout_sec = 0.2;
    double _encoder_timeout_sec = 0.2;
    double _gyro_meas_var = 0.0001;
    bool _use_encoders = true;

    // Wheel encoders (rear wheels)
    double _wheel_radius = 0.20;       // m
    double _track_width = 1.2;        // m
    double _encoder_base_var = 0.01;  // (m/s)^2

public:
    MirenaState() : rclcpp::Node("mirena_state")
    {
        this->declare_parameter<double>("odom_timeout_sec", 0.5);
        this->declare_parameter<double>("imu_timeout_sec", 0.2);
        this->declare_parameter<double>("encoder_timeout_sec", 0.2);
        this->declare_parameter<double>("max_predict_dt_sec", 0.1);
        this->declare_parameter<std::vector<double>>(
            "process_noise_diag", {0.01, 0.01, 0.005, 0.5, 0.5, 0.5});
        this->declare_parameter<std::vector<double>>(
            "odom_meas_floor_diag", {1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2});
        this->declare_parameter<std::vector<double>>(
            "init_cov_floor_diag", {0.01, 0.01, 0.001, 0.01, 0.01, 0.001});
        this->declare_parameter<double>("gyro_meas_var", 1.0e-5);
        this->declare_parameter<double>("wheel_radius", 0.2);
        this->declare_parameter<double>("track_width", 1.2);
        this->declare_parameter<double>("encoder_var", 0.01);
        this->declare_parameter<bool>("use_encoders", true);

        RCLCPP_INFO(this->get_logger(), "Starting Mirena State");

        _odom_timeout_sec = this->get_parameter("odom_timeout_sec").as_double();
        _imu_timeout_sec = this->get_parameter("imu_timeout_sec").as_double();
        _encoder_timeout_sec = this->get_parameter("encoder_timeout_sec").as_double();
        _wheel_radius = this->get_parameter("wheel_radius").as_double();
        _track_width = this->get_parameter("track_width").as_double();
        _encoder_base_var = this->get_parameter("encoder_var").as_double();
        _use_encoders = this->get_parameter("use_encoders").as_bool();

        _filter_cb_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        _report_cb_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

        auto filter_sub_options = rclcpp::SubscriptionOptions();
        filter_sub_options.callback_group = _filter_cb_group;

        // Publishers & TF
        _car_pub = this->create_publisher<mirena_common::msg::Car>(CAR_STATE_TOPIC, 10);
        _report_pub = this->create_publisher<mirena_common::msg::Heartbeat>("health", 10);
        _tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        mirena::Ekf::Config cfg;
        cfg.process_noise_diag = param_vec6("process_noise_diag");
        cfg.odom_meas_floor_diag = param_vec6("odom_meas_floor_diag");
        cfg.init_cov_floor_diag = param_vec6("init_cov_floor_diag");
        _gyro_meas_var = this->get_parameter("gyro_meas_var").as_double();
        cfg.gyro_meas_var = _gyro_meas_var;
        cfg.max_dt = this->get_parameter("max_predict_dt_sec").as_double();
        _ekf = std::make_unique<mirena::Ekf>(cfg);

        // Sensor subscriptions (SensorDataQoS)
        _imu_sub = this->create_subscription<sensor_msgs::msg::Imu>(
            IMU_TOPIC, rclcpp::SensorDataQoS(),
            std::bind(&MirenaState::imu_callback, this, std::placeholders::_1), filter_sub_options);
        _ins_odom_sub = this->create_subscription<nav_msgs::msg::Odometry>(
            INS_ODOM_TOPIC, rclcpp::SensorDataQoS(),
            std::bind(&MirenaState::ins_odom_callback, this, std::placeholders::_1), filter_sub_options);
        if (_use_encoders)
        {
            _encoder_sub = this->create_subscription<mirena_common::msg::WheelSpeeds>(
                WHEEL_SPEEDS_TOPIC, rclcpp::SensorDataQoS(),
                std::bind(&MirenaState::encoder_callback, this, std::placeholders::_1), filter_sub_options);
            RCLCPP_INFO(this->get_logger(), "Subscribed to %s for wheel encoder speeds (radius=%.3f m, var=%.4f)",
                        WHEEL_SPEEDS_TOPIC, _wheel_radius, _encoder_base_var);
        }
        else
        {
            RCLCPP_WARN(this->get_logger(),
                        "Wheel encoder fusion DISABLED (use_encoders=false). Not subscribing to %s.",
                        WHEEL_SPEEDS_TOPIC);
        }

        // Health/status report timer
        _report_timer = rclcpp::create_timer(
            this, this->get_clock(), kReportPeriod,
            std::bind(&MirenaState::report_callback, this), _report_cb_group);
    }

private:

    mirena::Ekf::Vec6 param_vec6(const std::string &name)
    {
        auto v = this->get_parameter(name).as_double_array();
        if (v.size() != 6)
        {
            RCLCPP_FATAL(this->get_logger(), "Parameter %s must have 6 elements, got %zu.",
                         name.c_str(), v.size());
            throw std::runtime_error("invalid " + name);
        }
        return mirena::Ekf::Vec6(v.data());
    }

    static double yaw_from_quat(const geometry_msgs::msg::Quaternion &q)
    {
        return std::atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));
    }

    void log_step(mirena::Ekf::Step step, const char *what)
    {
        using Step = mirena::Ekf::Step;
        if (step == Step::Diverged)
        {
          RCLCPP_ERROR(this->get_logger(),
             "EKF diverged on %s (non-finite state or covariance) — filter reset, "
             "re-initializing from the next INS fix. Reset #%u.",
             what, static_cast<unsigned int>(_ekf->divergences()));
        }
        else if (step == Step::Ignored)
        {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                 "EKF ignored %s: non-finite values or a stamp that did not "
                                 "advance. Check the sensor layer.", what);
        }
    }

    void init_at_origin(const sensor_msgs::msg::Imu &msg)
    {
        const double accel = std::hypot(msg.linear_acceleration.x, msg.linear_acceleration.y);
        const double rate = std::abs(msg.angular_velocity.z);
        const bool stationary = accel < kStationaryAccel && rate < kStationaryRate;
        const double vel_var = stationary ? kZuptVelVar : kUnknownVelVar;

        mirena::Ekf::Vec6 r;
        r << kInitPoseVar, kInitPoseVar, kInitPoseVar, vel_var, vel_var, _gyro_meas_var;

        const double t = rclcpp::Time(msg.header.stamp).seconds();
        if (_ekf->init(mirena::Ekf::Vec6::Zero(), r, t) != mirena::Ekf::Step::Ok)
        {
            log_step(mirena::Ekf::Step::Ignored, "origin initialization");
            return;
        }
        RCLCPP_INFO(this->get_logger(),
                    "odom origin defined at the current pose; dead reckoning until the INS "
                    "converges (%s).",
                    stationary ? "stationary, zero-velocity update applied"
                               : "MOVING at startup — velocity seeded as unknown");
    }

    void imu_callback(const sensor_msgs::msg::Imu::ConstSharedPtr &msg)
    {
        std::lock_guard<std::mutex> lock(_filter_mutex);
        _last_imu_stamp = msg->header.stamp;
        if (!_ekf->is_initialized())
        {
            if (!_ins_anchored) init_at_origin(*msg);
            return;
        }

        const double t = rclcpp::Time(msg->header.stamp).seconds();
        log_step(_ekf->predict(t, msg->linear_acceleration.x, msg->linear_acceleration.y), "IMU predict");
        if (!_ekf->is_initialized()) return;
        log_step(_ekf->update_gyro(msg->angular_velocity.z), "gyro update");

        // Synchronous publish directly following the IMU step
        publish_state();
    }
    
    
    void encoder_callback(const mirena_common::msg::WheelSpeeds::ConstSharedPtr &msg)
    {
        if (!_use_encoders) return;

        // Validación de marca de tiempo nula: sin stamp no hay forma de
        // sincronizar con el EKF. (builtin_interfaces::msg::Time no tiene
        // nanoseconds(); se revisan los campos sec y nanosec.)
        if (msg->header.stamp.sec == 0 && msg->header.stamp.nanosec == 0)
        {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                 "[Encoder] Ignoring message with null timestamp on %s.", WHEEL_SPEEDS_TOPIC);
            return;
        }

        std::lock_guard<std::mutex> lock(_filter_mutex);
        _last_encoder_stamp = msg->header.stamp;

        RCLCPP_INFO_ONCE(this->get_logger(),
                         "[Encoder] Callback activated: received first message on %s (rl=%.2f RPM, rr=%.2f RPM)",
                         WHEEL_SPEEDS_TOPIC, msg->rl, msg->rr);

        if (!_ekf->is_initialized())
        {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                 "[Encoder] Callback active, but EKF is not initialized yet (waiting for initial IMU/INS fix).");
            return;
        }

        // Validación de datos de entrada: descartar NaN/INF antes de tocar el EKF.
        if (!std::isfinite(msg->rl) || !std::isfinite(msg->rr))
        {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                 "[Encoder] Ignoring message with non-finite wheel speeds (rl=%.2f, rr=%.2f).",
                                 msg->rl, msg->rr);
            return;
        }

        const double v_left = (msg->rl * 2.0 * M_PI / 60.0) * _wheel_radius;
        const double v_right = (msg->rr * 2.0 * M_PI / 60.0) * _wheel_radius;

        const double u = 0.5 * (v_left + v_right);

        // Filtro de outliers: rechazar velocidades irrealistas para el vehículo.
        if (std::abs(u) > 15.0)
        {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                 "[Encoder] Rejecting outlier velocity u=%.2f m/s (|u| > 15.0 m/s).", u);
            return;
        }

        // Varianza dinámica: durante giros cerrados o deslizamientos las ruedas
        // giran a velocidades muy distintas y la medición de u pierde
        // confianza. Se escala cuadráticamente con la diferencia entre ruedas
        // y se limita para no anular la corrección por completo.
        const double v_diff = std::abs(v_left - v_right);
        const double dynamic_var = std::min(_encoder_base_var + 0.10 * (v_diff * v_diff), 1.0);

        // Sincronización temporal con el EKF: nunca aplicar una medición
        // en el pasado; si el EKF está atrasado, avanzar el tiempo primero.
        const double t = rclcpp::Time(msg->header.stamp).seconds();
        if (t < _ekf->stamp())
        {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                 "[Encoder] Discarding out-of-order message: stamp %.6f s < EKF stamp %.6f s.",
                                 t, _ekf->stamp());
            return;
        }
        if (t > _ekf->stamp())
        {
            log_step(_ekf->predict(t, 0.0, 0.0), "encoder time sync predict");
        }

        RCLCPP_INFO_ONCE(this->get_logger(),
                         "[Encoder] EKF is initialized; started fusing wheel encoder velocity updates.");

        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                             "[Encoder] Fusing: rl=%.1f RPM, rr=%.1f RPM -> u_meas=%.2f m/s (EKF u=%.2f m/s, var=%.4f)",
                             msg->rl, msg->rr, u, _ekf->state()(3), dynamic_var);

        log_step(_ekf->update_velocity_u(u, dynamic_var), "encoder speed update");
    }
    

    void latch_anchor(double ins_x, double ins_y, double ins_psi)
    {
        const auto &x = _ekf->state();
        _anchor_dpsi = mirena::Ekf::wrap_angle(x(2) - ins_psi);
        const double c = std::cos(_anchor_dpsi), s = std::sin(_anchor_dpsi);
        _anchor_tx = x(0) - (c * ins_x - s * ins_y);
        _anchor_ty = x(1) - (s * ins_x + c * ins_y);
        _ins_anchored = true;

        RCLCPP_INFO(this->get_logger(),
                    "INS anchored into odom: offset (%.2f, %.2f) m, %.2f deg. Dead reckoning had "
                    "carried the pose to (%.2f, %.2f); absolute corrections start now.",
                    _anchor_tx, _anchor_ty, _anchor_dpsi * 180.0 / M_PI, x(0), x(1));
    }

    void ins_to_odom(double ix, double iy, double ipsi,
                     double &ox, double &oy, double &opsi) const
    {
        const double c = std::cos(_anchor_dpsi), s = std::sin(_anchor_dpsi);
        ox = c * ix - s * iy + _anchor_tx;
        oy = s * ix + c * iy + _anchor_ty;
        opsi = mirena::Ekf::wrap_angle(ipsi + _anchor_dpsi);
    }

    void ins_odom_callback(const nav_msgs::msg::Odometry::ConstSharedPtr &msg)
    {
        std::lock_guard<std::mutex> lock(_filter_mutex);
        _last_odom_stamp = msg->header.stamp;
        const double t = rclcpp::Time(msg->header.stamp).seconds();

        const double ins_x = msg->pose.pose.position.x;
        const double ins_y = msg->pose.pose.position.y;
        const double ins_psi = yaw_from_quat(msg->pose.pose.orientation);

        if (!_ins_anchored)
        {
            if (_ekf->is_initialized())
            {
                latch_anchor(ins_x, ins_y, ins_psi);
            }
            else
            {
                _anchor_tx = _anchor_ty = _anchor_dpsi = 0.0;
                _ins_anchored = true;
                RCLCPP_INFO(this->get_logger(),
                            "INS arrived before any IMU; odom origin adopted from the INS.");
            }
        }

        mirena::Ekf::Vec6 z;
        double zx, zy, zpsi;
        ins_to_odom(ins_x, ins_y, ins_psi, zx, zy, zpsi);
        z << zx, zy, zpsi,
            msg->twist.twist.linear.x,
            msg->twist.twist.linear.y,
            msg->twist.twist.angular.z;

        mirena::Ekf::Vec6 r;
        r << msg->pose.covariance[0],   // x
            msg->pose.covariance[7],    // y
            msg->pose.covariance[35],   // yaw
            msg->twist.covariance[0],   // u
            msg->twist.covariance[7],   // v
            msg->twist.covariance[35];  // omega

        if (!_ekf->is_initialized())
        {
            if (_ekf->init(z, r, t) == mirena::Ekf::Step::Ok)
            {
                RCLCPP_INFO(this->get_logger(), "EKF re-initialized from INS at (%.2f, %.2f)", z(0), z(1));
                publish_state();
            }
            else
            {
                log_step(mirena::Ekf::Step::Ignored, "INS initialization");
            }
            return;
        }

        log_step(_ekf->update_odom(z, r), "INS odometry update");
    }

    void publish_state()
    {
        if (!_ekf->is_initialized()) return;

        if (_ekf->stamp() <= _last_published_stamp) return;
        _last_published_stamp = _ekf->stamp();

        const auto &x = _ekf->state();
        const auto &P = _ekf->covariance();
        const rclcpp::Time stamp(static_cast<int64_t>(_ekf->stamp() * 1e9), RCL_ROS_TIME);

        mirena_common::msg::Car car;
        car.header.stamp = stamp;
        car.header.frame_id = kOdomFrame;
        car.child_frame_id = kBaseFrame;
        car.x = x(0);
        car.y = x(1);
        car.psi = x(2);
        car.u = x(3);
        car.v = x(4);
        car.omega = x(5);
        for (int i = 0; i < 6; i++)
            for (int j = 0; j < 6; j++)
                car.covariance[i * 6 + j] = P(i, j);
        _car_pub->publish(car);

        send_tf(x(0), x(1), x(2), stamp);
    }

    void send_tf(double x, double y, double psi, const rclcpp::Time &stamp)
    {
        geometry_msgs::msg::TransformStamped t;
        t.header.stamp = stamp;
        t.header.frame_id = kOdomFrame;
        t.child_frame_id = kBaseFrame;
        t.transform.translation.x = x;
        t.transform.translation.y = y;
        t.transform.translation.z = 0.0;

        tf2::Quaternion q;
        q.setRPY(0, 0, psi);
        t.transform.rotation.x = q.x();
        t.transform.rotation.y = q.y();
        t.transform.rotation.z = q.z();
        t.transform.rotation.w = q.w();

        _tf_broadcaster->sendTransform(t);
    }

    void report_callback()
    {
        mirena_common::msg::Heartbeat msg;
        msg.timestamp = this->now();
        msg.node_name = this->get_name();

        nlohmann::json report_json;
        {
            std::lock_guard<std::mutex> lock(_filter_mutex);
            const double now_s = this->now().seconds();

            const bool have_odom = _last_odom_stamp.nanoseconds() > 0;
            const bool have_imu = _last_imu_stamp.nanoseconds() > 0;
            const bool have_encoder = _last_encoder_stamp.nanoseconds() > 0;
            const double odom_age = now_s - _last_odom_stamp.seconds();
            const double imu_age = now_s - _last_imu_stamp.seconds();
            const double encoder_age = now_s - _last_encoder_stamp.seconds();

            const bool odom_stale = !have_odom || odom_age > _odom_timeout_sec;
            const bool imu_stale = !have_imu || imu_age > _imu_timeout_sec;
            // A disabled encoder is intentionally absent, not stale.
            const bool encoder_stale = _use_encoders &&
                                        (!have_encoder || encoder_age > _encoder_timeout_sec);

            std::string mode;
            if (!_ekf->is_initialized())
                mode = "INIT";
            else if (!_ins_anchored)
                mode = "NO_INS_YET";
            else if (odom_stale && imu_stale)
                mode = "STALE";
            else if (odom_stale)
                mode = "DEAD_RECKONING";
            else if (imu_stale)
                mode = "NO_IMU";
            else
                mode = "FUSED";

            report_json["mode"] = mode;
            report_json["ins_anchored"] = _ins_anchored;
            report_json["encoders_enabled"] = _use_encoders;
            report_json["odom_stale"] = odom_stale;
            report_json["imu_stale"] = imu_stale;
            report_json["encoder_stale"] = encoder_stale;
            if (have_odom) report_json["odom_age_s"] = odom_age;
            if (have_imu) report_json["imu_age_s"] = imu_age;
            if (have_encoder) report_json["encoder_age_s"] = encoder_age;
            report_json["ekf_resets"] = _ekf->divergences();
            if (_ekf->is_initialized())
            {
                const auto &x = _ekf->state();
                report_json["car_pos_x"] = x(0);
                report_json["car_pos_y"] = x(1);
                report_json["car_pos_yaw"] = x(2);
                report_json["car_vel_u"] = x(3);
            }
        }

        msg.report = report_json.dump();
        _report_pub->publish(msg);
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MirenaState>();

    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}