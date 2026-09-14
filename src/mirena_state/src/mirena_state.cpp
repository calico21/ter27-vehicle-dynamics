#include <rclcpp/rclcpp.hpp>
#include <chrono>
#include <mutex>
#include <stdexcept>

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
#define CAR_STATE_TOPIC "/state/car"

using namespace std::chrono_literals;

namespace
{
    /// Frames and rate. Constants rather than parameters: no launch file in this
    /// workspace has ever set them, and the nodes downstream hard-code the same
    /// strings (mirena_perception's CAR_FRAME, mirena_slam's frames).
    constexpr const char *kOdomFrame = "odom";
    constexpr const char *kBaseFrame = "car/cog";
    constexpr auto kPublishPeriod = 10ms; // 100 Hz, matching the IMU stream
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
 * (nav_msgs/Odometry).
 *
 * New sensors (GSS, wheel speeds, GPS fix, ...) should be added here as
 * additional EKF measurement updates — never as direct /state/car publishers.
 *
 * Sign conventions: state and Car fields follow REP-103 body axes (x forward,
 * y left, z up): u = forward velocity, v = LEFT-positive lateral velocity,
 * omega = CCW-positive yaw rate. This matches the sim's established Car
 * convention that SLAM/MPC were developed against.
 *
 * WHAT THE PUBLISHED COVARIANCE MEANS -- read before consuming it.
 *
 * It is the filter's uncertainty *about the odom frame*, and it is a relative
 * quantity, not an absolute one. The odom origin was pinned wherever the INS
 * first converged, which on a status-0 fix is itself some metres from the true
 * position; that offset is common to every pose, unobservable, and deliberately
 * not represented here, because it cancels for everything that works in odom.
 *
 * It is also not a bound on absolute GNSS accuracy. INS position error is a
 * slowly varying bias rather than white noise, and this filter fuses 100
 * position measurements a second as though they were independent -- so P
 * settles near 0.03 m^2 (sigma 0.17 m) while the INS reports sigma 2.8 m. That
 * is the right number for "how well do I know where I am relative to where I
 * started", which is what odom means, and the wrong number for "how well do I
 * know where I am on Earth". Measured against revisited poses the relative
 * figure is optimistic by roughly 2x, not by the 16x the absolute comparison
 * suggests.
 *
 * Consumers wanting the uncertainty of a *relative* motion (SLAM's odometry
 * factors, say) must not difference this: a converged filter's absolute
 * covariance is flat, so the difference is ~0 and says nothing. Integrate the
 * velocity block (indices 21, 28, 35) over the interval instead -- that grows
 * honestly when the INS drops out, which is exactly when a relative constraint
 * should loosen.
 */
class MirenaState : public rclcpp::Node
{
    // Callback Groups for thread safety
    rclcpp::CallbackGroup::SharedPtr _filter_cb_group;
    rclcpp::CallbackGroup::SharedPtr _report_cb_group;

    // Subscribers
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr _imu_sub;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr _ins_odom_sub;

    // Publishers
    rclcpp::Publisher<mirena_common::msg::Car>::SharedPtr _car_pub;
    rclcpp::Publisher<mirena_common::msg::Heartbeat>::SharedPtr _report_pub;

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

    /// Stamp of the last state actually published. The filter clock only moves
    /// when an IMU sample arrives, so a timer tick that lands between two
    /// samples has nothing new to say -- and republishing anyway hands tf2 a
    /// duplicate stamp, which it answers with TF_REPEATED_DATA.
    double _last_published_stamp = 0.0;

    /// T_odom <- ins, latched once the INS first speaks. Until then the filter
    /// dead reckons and there is no absolute reference at all.
    bool _ins_anchored = false;
    double _anchor_tx = 0.0, _anchor_ty = 0.0, _anchor_dpsi = 0.0;

    // Parameters (resolved in the constructor)
    double _odom_timeout_sec = 0.5;
    double _imu_timeout_sec = 0.2;
    double _gyro_meas_var = 0.0001;
    //(encoder) double _encoder_timeout_sec = 0.2; 

    //Wheel (encoder) (Falta rellenar con los datos)
/*

    this->declare_parameter<double>("wheel_radius", 0.202);
    this->declare_parameter<double>("encoder_var", 0.01);
    _wheel_radius = this->get_parameter("wheel_radius").as_double();
    _encoder_base_var = this->get_parameter("encoder_var").as_double();

    // Crear la suscripción para que empiece a recibir datos de la CAN
    _encoder_sub = this->create_subscription<mirena_common::msg::WheelSpeeds>(
        "/sensors/wheel_speeds", rclcpp::SensorDataQoS(),
        std::bind(&MirenaState::encoder_callback, this, std::placeholders::_1), filter_sub_options);
*/
public:
    MirenaState() : rclcpp::Node("mirena_state")
    {
        this->declare_parameter<double>("odom_timeout_sec", 0.5);
        this->declare_parameter<double>("imu_timeout_sec", 0.2);
        //(encoder) this->declare_parameter<double>("encoder_timeout_sec", 0.2);
        this->declare_parameter<double>("max_predict_dt_sec", 0.1);
        // Continuous-time process noise diagonal over [x, y, psi, u, v, omega]
        this->declare_parameter<std::vector<double>>(
            "process_noise_diag", {0.01, 0.01, 0.005, 0.5, 0.5, 0.5});
        // Floor on INS odometry measurement variances. Purely a guard against a
        // zero or negative variance reaching the gain.
        //
        // It used to be {0.01, 0.01, 0.001, 0.01, 0.01, 0.001}, which was not a
        // guard but a silent override: measured against the 2026-08-08 log, the
        // yaw and angular-rate entries replaced the sensor's own uncertainty on
        // 100% of 27690 messages and the velocity entries on 23%. The VN-300
        // reported 0.478 deg of yaw uncertainty; the filter was told 1.81 deg.
        this->declare_parameter<std::vector<double>>(
            "odom_meas_floor_diag", {1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9});
        // Floor on the covariance the filter initialises P with. These are the
        // values the floor above used to carry, kept here because init() is the
        // one place they were doing real work: the odom frame is defined by the
        // initial pose, so its reported variance is zero and P needs seeding
        // from something.
        this->declare_parameter<std::vector<double>>(
            "init_cov_floor_diag", {0.01, 0.01, 0.001, 0.01, 0.01, 0.001});
        // Variance of the gyro-as-measurement omega update, rad^2/s^2.
        //
        // Measured, not assumed: stationary whitened yaw-rate noise on the
        // 2026-08-08 log was 9.7e-7, and this sits an order of magnitude above
        // it to cover vibration under load. The previous 1e-4 had no derivation
        // and was in any case clamped to 1e-3 by the old floor. Keep it in step
        // with mirena_imu's gyro_variance -- they describe the same gyro.
        this->declare_parameter<double>("gyro_meas_var", 1.0e-5);


        RCLCPP_INFO(this->get_logger(), "Starting Mirena State");

        _odom_timeout_sec = this->get_parameter("odom_timeout_sec").as_double();
        _imu_timeout_sec = this->get_parameter("imu_timeout_sec").as_double();
        // (encoder) _encoder_timeout_sec = this->get_parameter("encoder_timeout_sec").as_double();
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

        // Sensor subscriptions: SensorDataQoS (best-effort) — the hardware
        // drivers publish best-effort; a reliable subscriber would starve.
        _imu_sub = this->create_subscription<sensor_msgs::msg::Imu>(
            IMU_TOPIC, rclcpp::SensorDataQoS(),
            std::bind(&MirenaState::imu_callback, this, std::placeholders::_1), filter_sub_options);
        _ins_odom_sub = this->create_subscription<nav_msgs::msg::Odometry>(
            INS_ODOM_TOPIC, rclcpp::SensorDataQoS(),
            std::bind(&MirenaState::ins_odom_callback, this, std::placeholders::_1), filter_sub_options);

        // Node clock, not wall clock. Under use_sim_time these two are different
        // clocks running at different rates, and a wall timer would publish at
        // 100 Hz of real time while the stamps it carries advance at whatever
        // rate the simulator manages.
        _publish_timer = rclcpp::create_timer(
            this, this->get_clock(), kPublishPeriod,
            std::bind(&MirenaState::publish_timer_callback, this), _filter_cb_group);
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
            // Fatal rather than substituted. This is filter tuning: carrying on
            // with a silent stand-in means the car drives on a filter nobody
            // configured, which is worse than not starting.
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

    /// Log a filter step that did not apply. Divergence is an error (the filter
    /// dropped itself and now waits for a new INS fix); Ignored is a producer
    /// problem, throttled because a broken producer breaks every sample.
    void log_step(mirena::Ekf::Step step, const char *what)
    {
        using Step = mirena::Ekf::Step;
        if (step == Step::Diverged)
        {
            RCLCPP_ERROR(this->get_logger(),
                         "EKF diverged on %s (non-finite state or covariance) — filter reset, "
                         "re-initializing from the next INS fix. Reset #%u.",
                         what, _ekf->divergences());
        }
        else if (step == Step::Ignored)
        {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                 "EKF ignored %s: non-finite values or a stamp that did not "
                                 "advance. Check the sensor layer.", what);
        }
    }

    /// Define the odom frame here and start dead reckoning.
    ///
    /// REP-105 makes the odom origin arbitrary -- conventionally wherever the
    /// robot booted -- so there is nothing to wait for. The filter used to sit
    /// idle until the INS converged, which on rosbag2_2026_08_05-20_26_23 was
    /// 62 s of a 271 s run: 23% of the session with cones being detected and
    /// discarded because no pose existed to place them against.
    ///
    /// Velocity is seeded at zero. That is a real measurement, not a guess, for
    /// a car sitting in staging -- but only if the IMU agrees it is sitting, so
    /// a car that is already rolling gets a variance wide enough to say the
    /// filter does not know.
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
            // Once the INS has anchored, recovery from a divergence goes back
            // through the INS instead (ins_odom_callback). Re-defining the
            // origin mid-run would put a step in the one frame whose entire
            // contract is that it never steps.
            if (!_ins_anchored) init_at_origin(*msg);
            return;
        }

        const double t = rclcpp::Time(msg->header.stamp).seconds();
        // Body accel is gravity-free forward/left (REP-103) after the driver's
        // gravity compensation and frame alignment to the base frame.
        log_step(_ekf->predict(t, msg->linear_acceleration.x, msg->linear_acceleration.y), "IMU predict");
        if (!_ekf->is_initialized()) return; // predict diverged and reset the filter
        log_step(_ekf->update_gyro(msg->angular_velocity.z), "gyro update");
    }

    /// Latch the rigid transform that carries INS poses into odom.
    ///
    /// The INS reports in a frame pinned wherever its own filter converged,
    /// which is not where this node defined odom. Rather than adopt the INS
    /// frame -- which is what waiting for it used to amount to -- the offset
    /// between the two is measured once and every later INS pose is carried
    /// through it. Because that changes what the *measurement* means and not
    /// the *state*, odom does not move when the INS arrives: no step, which is
    /// odom's whole contract.
    ///
    /// Only position and yaw need it. The twist is body-frame (child_frame_id
    /// is the base frame), and body axes are the same physical axes whichever
    /// world frame references them, so u, v and omega carry over untouched.
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
                // The INS beat the IMU to this node, so there is no dead-reckoned
                // pose to anchor against. Adopt the INS frame as odom -- identity
                // anchor -- which is what this node did unconditionally before.
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

        // The position sigma the driver reports is isotropic, so rotating the
        // measurement by the anchor leaves its covariance unchanged; a yaw
        // offset does not touch the yaw variance either.
        mirena::Ekf::Vec6 r;
        r << msg->pose.covariance[0],   // x
            msg->pose.covariance[7],    // y
            msg->pose.covariance[35],   // yaw
            msg->twist.covariance[0],   // u
            msg->twist.covariance[7],   // v
            msg->twist.covariance[35];  // omega

        if (!_ekf->is_initialized())
        {
            // Divergence recovery. The measurement is already in odom, so
            // re-seeding from it puts the filter back without moving the frame.
            if (_ekf->init(z, r, t) == mirena::Ekf::Step::Ok)
                RCLCPP_INFO(this->get_logger(), "EKF re-initialized from INS at (%.2f, %.2f)", z(0), z(1));
            else
                log_step(mirena::Ekf::Step::Ignored, "INS initialization");
            return;
        }
        // The measurement is fused at the filter's current time rather than
        // back-propagated to its own stamp. Both streams come from the same
        // driver at the same rate, so the mismatch is bounded by one sample
        // (~10 ms); anything that changes that has to revisit this.
        log_step(_ekf->update_odom(z, r), "INS odometry update");
    }

    void publish_timer_callback()
    {
        std::lock_guard<std::mutex> lock(_filter_mutex);
        // No state before the filter is initialised -- which the first IMU
        // message does via init_at_origin, not the first INS fix. The stack
        // runs without GNSS; it just runs dead-reckoning (mode NO_INS_YET).
        if (!_ekf->is_initialized()) return;

        // Nothing new since the last tick: the filter clock only moves on an IMU
        // sample, and republishing the same instant is a duplicate, not an update.
        if (_ekf->stamp() <= _last_published_stamp) return;
        _last_published_stamp = _ekf->stamp();

        const auto &x = _ekf->state();
        const auto &P = _ekf->covariance();
        // Stamp with the filter's own time (derived from message stamps) so the
        // output is consistent under use_sim_time.
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

            // Before the first message of a stream there is no age to report --
            // subtracting an epoch-zero stamp yields the seconds since 1970,
            // which reads as a number rather than as "nothing has arrived".
            const bool have_odom = _last_odom_stamp.nanoseconds() > 0;
            const bool have_imu = _last_imu_stamp.nanoseconds() > 0;
            // (encoder) const bool have_encoder = _last_encoder_stamp.nanoseconds() > 0; 
            const double odom_age = now_s - _last_odom_stamp.seconds();
            const double imu_age = now_s - _last_imu_stamp.seconds();
            // (encoder) const double encoder_age = now_s - _last_encoder_stamp.seconds(); // NUEVO


            // Both inputs are checked: losing the INS leaves the filter dead
            // reckoning on the accelerometer, losing the IMU leaves it with no
            // propagation between fixes. They fail differently, so they report
            // differently rather than collapsing into one "STALE".
            const bool odom_stale = !have_odom || odom_age > _odom_timeout_sec;
            const bool imu_stale = !have_imu || imu_age > _imu_timeout_sec;
        //(encoder)    const bool encoder_stale = !have_encoder || encoder_age > _encoder_timeout_sec; 
            std::string mode;
            if (!_ekf->is_initialized())
                mode = "INIT";
            else if (!_ins_anchored)
                // Running, but on the IMU alone and with no absolute reference
                // yet. Distinct from DEAD_RECKONING, which means the INS was
                // there and went away.
                mode = "NO_INS_YET";
            else if (odom_stale && imu_stale)
                mode = "STALE";           // nothing is arriving; the pose is frozen
            else if (odom_stale)
                mode = "DEAD_RECKONING";  // IMU only; drifting, trust it briefly
            else if (imu_stale)
                mode = "NO_IMU";          // INS fixes only; no propagation between them
            else
                mode = "FUSED";

            report_json["mode"] = mode;
            report_json["ins_anchored"] = _ins_anchored;
            report_json["odom_stale"] = odom_stale;
            report_json["imu_stale"] = imu_stale;
            if (have_odom) report_json["odom_age_s"] = odom_age;
            if (have_imu) report_json["imu_age_s"] = imu_age;
            // (encoder) if (have_encoder) report_json["encoder_age_s"] = encoder_age; 
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
