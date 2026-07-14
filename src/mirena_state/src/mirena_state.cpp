#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <rclcpp_lifecycle/lifecycle_publisher.hpp>
#include <lifecycle_msgs/msg/state.hpp>
#include <chrono>
#include <mutex>

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

#include "ekf.hpp"

// Topics and Frames
#define IMU_TOPIC "/sensors/imu"
#define INS_ODOM_TOPIC "/sensors/imu/odom"
#define GROUND_TRUTH_TOPIC "/debug/state/car"
#define CAR_STATE_TOPIC "/state/car"

#define REPORT_PERIOD 100ms

using namespace std::chrono_literals;
using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

/**
 * MirenaState — sensor fusion / state estimation node.
 *
 * Owns /state/car and the odom->base TF in every mode:
 *  - EKF mode (default): predicts from /sensors/imu (body accel + gyro) and
 *    corrects with the INS solution on /sensors/imu/odom (nav_msgs/Odometry).
 *  - Ground-truth passthrough (`ground_truth_passthrough:=true`): republishes
 *    the simulator's ground truth from /debug/state/car. Same output contract,
 *    so the rest of the pipeline never knows the difference.
 *
 * New sensors (GSS, wheel speeds, GPS fix, ...) should be added here as
 * additional EKF measurement updates — never as direct /state/car publishers.
 *
 * Sign conventions: state and Car fields follow REP-103 body axes (x forward,
 * y left, z up): u = forward velocity, v = LEFT-positive lateral velocity,
 * omega = CCW-positive yaw rate. This matches the sim's established Car
 * convention that SLAM/MPC were developed against.
 */
class MirenaState : public rclcpp_lifecycle::LifecycleNode
{
    // Callback Groups for thread safety
    rclcpp::CallbackGroup::SharedPtr _filter_cb_group;
    rclcpp::CallbackGroup::SharedPtr _report_cb_group;

    // Subscribers
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr _imu_sub;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr _ins_odom_sub;
    rclcpp::Subscription<mirena_common::msg::Car>::SharedPtr _gt_sub;

    // Lifecycle Publishers
    rclcpp_lifecycle::LifecyclePublisher<mirena_common::msg::Car>::SharedPtr _car_pub;
    rclcpp_lifecycle::LifecyclePublisher<mirena_common::msg::Heartbeat>::SharedPtr _report_pub;

    // Transform broadcaster (odom -> base)
    std::unique_ptr<tf2_ros::TransformBroadcaster> _tf_broadcaster;

    // Timers
    rclcpp::TimerBase::SharedPtr _publish_timer;
    rclcpp::TimerBase::SharedPtr _report_timer;

    // Filter state & mutex (shared between filter and report/publish threads)
    std::mutex _filter_mutex;
    std::unique_ptr<mirena::Ekf> _ekf;
    rclcpp::Time _last_imu_stamp{0, 0, RCL_ROS_TIME};
    rclcpp::Time _last_odom_stamp{0, 0, RCL_ROS_TIME};

    // Parameters (resolved in on_configure)
    std::string _odom_frame, _base_frame;
    bool _passthrough = false;
    bool _broadcast_tf = true;
    double _odom_timeout_sec = 0.5;
    double _imu_timeout_sec = 0.2;

public:
    MirenaState() : rclcpp_lifecycle::LifecycleNode("mirena_state")
    {
        this->declare_parameter<bool>("ground_truth_passthrough", false);
        this->declare_parameter<bool>("broadcast_tf", true);
        this->declare_parameter<double>("publish_rate_hz", 100.0);
        this->declare_parameter<std::string>("odom_frame", "odom");
        this->declare_parameter<std::string>("base_frame", "car/cog");
        this->declare_parameter<double>("odom_timeout_sec", 0.5);
        this->declare_parameter<double>("imu_timeout_sec", 0.2);
        this->declare_parameter<double>("max_predict_dt_sec", 0.1);
        // Continuous-time process noise diagonal over [x, y, psi, u, v, omega]
        this->declare_parameter<std::vector<double>>(
            "process_noise_diag", {0.01, 0.01, 0.005, 0.5, 0.5, 0.5});
        // Floor on INS odometry measurement variances (guards zero covariances)
        this->declare_parameter<std::vector<double>>(
            "odom_meas_floor_diag", {0.01, 0.01, 0.001, 0.01, 0.01, 0.001});
        this->declare_parameter<double>("gyro_meas_var", 0.0001);
    }

    CallbackReturn on_configure(const rclcpp_lifecycle::State &) override
    {
        RCLCPP_INFO(this->get_logger(), "Configuring Mirena State");

        _passthrough = this->get_parameter("ground_truth_passthrough").as_bool();
        _broadcast_tf = this->get_parameter("broadcast_tf").as_bool();
        _odom_frame = this->get_parameter("odom_frame").as_string();
        _base_frame = this->get_parameter("base_frame").as_string();
        _odom_timeout_sec = this->get_parameter("odom_timeout_sec").as_double();
        _imu_timeout_sec = this->get_parameter("imu_timeout_sec").as_double();

        _filter_cb_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        _report_cb_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

        auto filter_sub_options = rclcpp::SubscriptionOptions();
        filter_sub_options.callback_group = _filter_cb_group;

        // Publishers & TF
        _car_pub = this->create_publisher<mirena_common::msg::Car>(CAR_STATE_TOPIC, 10);
        _report_pub = this->create_publisher<mirena_common::msg::Heartbeat>("health", 10);
        _tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        if (_passthrough)
        {
            RCLCPP_WARN(this->get_logger(), "Ground-truth passthrough mode: republishing %s", GROUND_TRUTH_TOPIC);
            _gt_sub = this->create_subscription<mirena_common::msg::Car>(
                GROUND_TRUTH_TOPIC, 10,
                std::bind(&MirenaState::gt_callback, this, std::placeholders::_1), filter_sub_options);
        }
        else
        {
            mirena::Ekf::Config cfg;
            cfg.process_noise_diag = param_vec6("process_noise_diag");
            cfg.odom_meas_floor_diag = param_vec6("odom_meas_floor_diag");
            cfg.gyro_meas_var = this->get_parameter("gyro_meas_var").as_double();
            cfg.max_dt = this->get_parameter("max_predict_dt_sec").as_double();
            _ekf = std::make_unique<mirena::Ekf>(cfg);

            // Sensor subscriptions: SensorDataQoS (best-effort) — the hardware
            // drivers publish best-effort; a reliable subscriber would starve.
            _imu_sub = this->create_subscription<sensor_msgs::msg::Imu>(
                IMU_TOPIC, rclcpp::SensorDataQoS(),
                std::bind(&MirenaState::imu_callback, this, std::placeholders::_1), filter_sub_options);
            _ins_odom_sub = this->create_subscription<nav_msgs::msg::Odometry>(
                INS_ODOM_TOPIC, rclcpp::SensorDataQoS(),
                std::bind(&MirenaState::ins_odom_callback, this, std::placeholders::_1), filter_sub_options);

            const double rate = this->get_parameter("publish_rate_hz").as_double();
            _publish_timer = this->create_wall_timer(
                std::chrono::duration<double>(1.0 / rate),
                std::bind(&MirenaState::publish_timer_callback, this), _filter_cb_group);
        }

        _report_timer = this->create_wall_timer(
            REPORT_PERIOD, std::bind(&MirenaState::report_callback, this), _report_cb_group);

        return CallbackReturn::SUCCESS;
    }

    CallbackReturn on_activate(const rclcpp_lifecycle::State &) override
    {
        RCLCPP_INFO(this->get_logger(), "State Node Active");
        _car_pub->on_activate();
        _report_pub->on_activate();
        return CallbackReturn::SUCCESS;
    }

    CallbackReturn on_deactivate(const rclcpp_lifecycle::State &) override
    {
        _car_pub->on_deactivate();
        _report_pub->on_deactivate();
        return CallbackReturn::SUCCESS;
    }

    CallbackReturn on_cleanup(const rclcpp_lifecycle::State &) override
    {
        _publish_timer.reset();
        _report_timer.reset();
        _imu_sub.reset();
        _ins_odom_sub.reset();
        _gt_sub.reset();
        _car_pub.reset();
        _report_pub.reset();
        _ekf.reset();
        return CallbackReturn::SUCCESS;
    }

private:
    mirena::Ekf::Vec6 param_vec6(const std::string &name)
    {
        auto v = this->get_parameter(name).as_double_array();
        if (v.size() != 6)
        {
            RCLCPP_ERROR(this->get_logger(), "Parameter %s must have 6 elements, got %zu — using defaults", name.c_str(), v.size());
            return mirena::Ekf::Vec6::Constant(0.1);
        }
        return mirena::Ekf::Vec6(v.data());
    }

    static double yaw_from_quat(const geometry_msgs::msg::Quaternion &q)
    {
        return std::atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));
    }

    bool is_active()
    {
        return this->get_current_state().id() == lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE;
    }

    // --- EKF mode -----------------------------------------------------------

    void imu_callback(const sensor_msgs::msg::Imu::ConstSharedPtr &msg)
    {
        if (!is_active()) return;
        std::lock_guard<std::mutex> lock(_filter_mutex);
        _last_imu_stamp = msg->header.stamp;
        if (!_ekf->is_initialized()) return; // wait for the first INS fix

        const double t = rclcpp::Time(msg->header.stamp).seconds();
        // Body accel is gravity-free forward/left (REP-103) after the driver's
        // gravity compensation and frame alignment to the base frame.
        _ekf->predict(t, msg->linear_acceleration.x, msg->linear_acceleration.y);
        _ekf->update_gyro(msg->angular_velocity.z);
    }

    void ins_odom_callback(const nav_msgs::msg::Odometry::ConstSharedPtr &msg)
    {
        if (!is_active()) return;
        std::lock_guard<std::mutex> lock(_filter_mutex);
        _last_odom_stamp = msg->header.stamp;

        const double t = rclcpp::Time(msg->header.stamp).seconds();

        mirena::Ekf::Vec6 z;
        z << msg->pose.pose.position.x,
            msg->pose.pose.position.y,
            yaw_from_quat(msg->pose.pose.orientation),
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
            _ekf->init(z, r, t);
            RCLCPP_INFO(this->get_logger(), "EKF initialized from INS at (%.2f, %.2f)", z(0), z(1));
            return;
        }
        _ekf->update_odom(z, r);
    }

    void publish_timer_callback()
    {
        if (!is_active()) return;
        std::lock_guard<std::mutex> lock(_filter_mutex);
        if (!_ekf->is_initialized()) return; // no state before the first INS fix

        const auto &x = _ekf->state();
        const auto &P = _ekf->covariance();
        // Stamp with the filter's own time (derived from message stamps) so the
        // output is consistent under use_sim_time.
        const rclcpp::Time stamp(static_cast<int64_t>(_ekf->stamp() * 1e9), RCL_ROS_TIME);

        mirena_common::msg::Car car;
        car.header.stamp = stamp;
        car.header.frame_id = _odom_frame;
        car.child_frame_id = _base_frame;
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

        if (_broadcast_tf) send_tf(x(0), x(1), x(2), stamp);
    }

    // --- Ground-truth passthrough mode ---------------------------------------

    void gt_callback(const mirena_common::msg::Car::ConstSharedPtr &msg)
    {
        if (!is_active()) return;
        std::lock_guard<std::mutex> lock(_filter_mutex);
        _last_odom_stamp = msg->header.stamp;

        mirena_common::msg::Car car = *msg;
        car.header.frame_id = _odom_frame;
        car.child_frame_id = _base_frame;
        _car_pub->publish(car);

        if (_broadcast_tf) send_tf(msg->x, msg->y, msg->psi, msg->header.stamp);
    }

    // --- Shared helpers -------------------------------------------------------

    void send_tf(double x, double y, double psi, const rclcpp::Time &stamp)
    {
        geometry_msgs::msg::TransformStamped t;
        t.header.stamp = stamp;
        t.header.frame_id = _odom_frame;
        t.child_frame_id = _base_frame;
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
        if (!_report_pub->is_activated()) return;

        mirena_common::msg::Heartbeat msg;
        msg.timestamp = this->now();
        msg.node_name = this->get_name();

        nlohmann::json report_json;
        {
            std::lock_guard<std::mutex> lock(_filter_mutex);
            const double now_s = this->now().seconds();
            const double odom_age = now_s - rclcpp::Time(_last_odom_stamp).seconds();
            const double imu_age = now_s - rclcpp::Time(_last_imu_stamp).seconds();

            std::string mode;
            if (_passthrough)
                mode = "GT_PASSTHROUGH";
            else if (!_ekf || !_ekf->is_initialized())
                mode = "INIT";
            else if (odom_age > _odom_timeout_sec)
                mode = "STALE";
            else
                mode = "FUSED";

            report_json["mode"] = mode;
            report_json["odom_age_s"] = odom_age;
            report_json["imu_age_s"] = imu_age;
            if (_ekf && _ekf->is_initialized())
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

    bool autostart_flag = false;
    for (int i = 1; i < argc; i++)
    {
        if (std::string(argv[i]) == "-a")
        {
            autostart_flag = true;
            break;
        }
    }

    if (autostart_flag)
    {
        RCLCPP_INFO(node->get_logger(), "Autoactivating");
        node->configure();
        node->activate();
    }
    else
    {
        RCLCPP_INFO(node->get_logger(), "Awaiting activation. Use argument '-a' when running for autoactivation");
    }

    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node->get_node_base_interface());
    executor.spin();
    rclcpp::shutdown();
    return 0;
}
