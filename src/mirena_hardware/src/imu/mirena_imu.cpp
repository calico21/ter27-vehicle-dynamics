#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <exception>
#include <string>
#include <thread>
#include <vector>

#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Transform.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <rclcpp/rclcpp.hpp>

#include "imu_wrapper.hpp"
#include "imu_pipeline.hpp"

#include <vectornav/Interface/Sensor.hpp>
#include <vectornav/Interface/Registers.hpp>

#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <nav_msgs/msg/odometry.hpp>

using namespace std::chrono_literals;

/// VectorNav INS mode, as reported in insStatus.
enum class InsMode
{
    NotTracking, ///< no solution
    Aligning,    ///< filter running but not yet converged with GNSS
    Tracking,    ///< converged
    GpsLoss      ///< was tracking, now dead reckoning
};

/// Where a measurement's timestamp came from. Worth naming because a silent
/// change of source is a silent jump in the clock of a 100 Hz sensor.
enum class StampSource
{
    None,
    GpsHardware,
    SdkReception,
    PublishTime
};

class ImuNode : public rclcpp::Node
{
public:
    ImuNode()
        : Node("mirena_imu"),
          imu_(this->get_logger()),
          tf_buffer_(this->get_clock()),
          tf_listener_(tf_buffer_)
    {
        // ---- Parameters (everything tunable is exposed here) ----
        imu_dev_path_ = this->declare_parameter<std::string>("imu_device", "/dev/ter_IMU");

        // Antenna geometry (metres). Defaults match the legacy hard-coded values.
        antenna_a_offset_ = this->declare_parameter<std::vector<double>>("antenna_a_offset", {-0.11, 0.0, 0.52});
        gnss_baseline_ = this->declare_parameter<std::vector<double>>("gnss_compass_baseline", {1.50, 0.0, -0.36});
        gnss_baseline_unc_ = this->declare_parameter<std::vector<double>>("gnss_compass_uncertainty", {0.04, 0.04, 0.04});

        // GPS->UNIX leap-second offset (18 since Dec 2016). Exposed so it can be
        // bumped without a rebuild if a future leap second is announced.
        gps_leap_seconds_ns_ =
            static_cast<int64_t>(this->declare_parameter<int>("gps_leap_seconds", 18)) * 1'000'000'000LL;

        // Output rate: the binary message is streamed at base_rate / divisor.
        // VN-300 base IMU rate is 400 Hz; VN-200/100 is 800 Hz.
        imu_base_rate_hz_ = this->declare_parameter<int>("imu_base_rate_hz", 400);
        imu_output_rate_hz_ = this->declare_parameter<int>("imu_output_rate_hz", 100);

        reconnect_delay_s_ = this->declare_parameter<double>("reconnect_delay_sec", 1.0);

        // QoS: IMU is high-rate sensor data, best-effort by default.
        use_sensor_qos_ = this->declare_parameter<bool>("use_sensor_data_qos", true);
        qos_depth_ = this->declare_parameter<int>("qos_depth", 10);

        frames_.lidar = this->declare_parameter<std::string>("lidar_frame", "car/lidar");
        frames_.odom = this->declare_parameter<std::string>("odom_frame", "odom");
        frames_.cog = this->declare_parameter<std::string>("base_frame", "car/cog");
        frames_.gnss = this->declare_parameter<std::string>("gnss_frame", "car/gnss");
        imu_frame_ = this->declare_parameter<std::string>("imu_frame", "car/imu");

        if (antenna_a_offset_.size() != 3 || gnss_baseline_.size() != 3 || gnss_baseline_unc_.size() != 3)
        {
            RCLCPP_FATAL(get_logger(), "antenna_a_offset / gnss_compass_baseline / gnss_compass_uncertainty must each have 3 elements.");
            throw std::runtime_error("invalid antenna geometry parameters");
        }

        rclcpp::QoS qos = use_sensor_qos_
                              ? rclcpp::QoS(rclcpp::SensorDataQoS())
                              : rclcpp::QoS(rclcpp::KeepLast(static_cast<size_t>(qos_depth_)));

        RCLCPP_INFO(get_logger(), "Initializing IMU bridge on %s (output %d Hz)", imu_dev_path_.c_str(), imu_output_rate_hz_);

        // Sensor-layer outputs only: raw IMU streams, the onboard INS solution
        // as odometry, and the GNSS fix. State fusion (/state/car) and health
        // reporting are the pipeline's job (mirena_state), not the driver's.
        imu_in_lidar_pub_ = this->create_publisher<sensor_msgs::msg::Imu>("/sensors/lidar/imu", qos);
        imu_in_cog_pub_ = this->create_publisher<sensor_msgs::msg::Imu>("/sensors/imu", qos);
        odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/sensors/imu/odom", qos);
        fix_pub_ = this->create_publisher<sensor_msgs::msg::NavSatFix>("/sensors/gps/fix", qos);

        keep_running_ = true;
        publish_thread_ = std::thread(&ImuNode::publish_loop, this);

        RCLCPP_INFO(get_logger(), "VectorNav IMU streaming layer online.");
    }

    ~ImuNode()
    {
        keep_running_ = false;
        if (publish_thread_.joinable())
        {
            publish_thread_.join();
        }

        if (imu_.is_connected())
        {
            imu_.disconnect();
        }
        RCLCPP_INFO(get_logger(), "IMU bridge node destroyed cleanly.");
    }

private:
    // ---- Parameters ----
    std::string imu_dev_path_;
    std::vector<double> antenna_a_offset_;
    std::vector<double> gnss_baseline_;
    std::vector<double> gnss_baseline_unc_;
    int64_t gps_leap_seconds_ns_;
    int imu_base_rate_hz_;
    int imu_output_rate_hz_;
    double reconnect_delay_s_;
    bool use_sensor_qos_;
    int qos_depth_;
    mirena::IMU::Frames frames_;
    std::string imu_frame_;

    // ---- State ----
    StampSource stamp_source_ = StampSource::None;

    mirena::IMU::Wrapper imu_;
    mirena::IMU::Pipeline pipeline_;

    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;

    // The mounting geometry, looked up once.
    //
    // These are bolts, not motion: car/imu, car/lidar and car/cog do not move
    // relative to each other, so re-querying them for every one of a hundred
    // measurements a second bought nothing and cost a failure mode -- a lookup
    // that missed dropped the whole measurement, silently taking the INS
    // odometry that mirena_state runs on down with it.
    bool have_mounting_ = false;
    tf2::Transform cog_from_imu_;
    tf2::Transform lidar_from_imu_;

    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_in_lidar_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_in_cog_pub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    rclcpp::Publisher<sensor_msgs::msg::NavSatFix>::SharedPtr fix_pub_;

    std::thread publish_thread_;
    std::atomic<bool> keep_running_{false};

    // ==========================================
    // Hardware Timestamp (camera-style approach)
    // ==========================================
    rclcpp::Time get_measurement_stamp(const VN::Sensor::CompositeDataQueueReturn &msg)
    {
        // Priority 1: GPS hardware timestamp — absolute wall-clock (sub-µs accuracy)
        if (msg->time.timeStatus.has_value() && msg->time.timeGps.has_value() &&
            msg->time.timeStatus.value().towValid)
        {
            // GPS epoch = Jan 6 1980 = 315,964,800 s after UNIX epoch.
            // GPS time does not count leap seconds; subtract the configured offset.
            constexpr int64_t GPS_TO_UNIX_NS = 315'964'800LL * 1'000'000'000LL;
            const int64_t unix_ns = static_cast<int64_t>(msg->time.timeGps.value().nanoseconds()) +
                                    GPS_TO_UNIX_NS - gps_leap_seconds_ns_;
            note_stamp_source(StampSource::GpsHardware);
            return rclcpp::Time(unix_ns, RCL_SYSTEM_TIME);
        }

        // Priority 2: SDK packet reception time — steady_clock → CLOCK_REALTIME offset
        using tp = std::chrono::steady_clock::time_point;
        if (msg->timestamp != tp{})
        {
            struct timespec ts_mono, ts_real;
            clock_gettime(CLOCK_MONOTONIC, &ts_mono);
            clock_gettime(CLOCK_REALTIME, &ts_real);
            const int64_t mono_ns = static_cast<int64_t>(ts_mono.tv_sec) * 1'000'000'000LL + ts_mono.tv_nsec;
            const int64_t real_ns = static_cast<int64_t>(ts_real.tv_sec) * 1'000'000'000LL + ts_real.tv_nsec;
            const int64_t sdk_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                       msg->timestamp.time_since_epoch())
                                       .count();
            note_stamp_source(StampSource::SdkReception);
            return rclcpp::Time(sdk_ns + (real_ns - mono_ns), RCL_SYSTEM_TIME);
        }

        note_stamp_source(StampSource::PublishTime);
        return this->get_clock()->now();
    }

    /// Log every *change* of timestamp source, not just the first one.
    ///
    /// This used to latch after the first measurement, so losing towValid
    /// mid-session swapped GPS time for the host clock in silence. Those two
    /// agree only if the host is disciplined to GPS; if it is not, every
    /// stamp downstream steps by the difference with nothing in the log to
    /// say why.
    void note_stamp_source(StampSource source)
    {
        if (source == stamp_source_)
        {
            return;
        }
        const char *name = source == StampSource::GpsHardware  ? "GPS hardware time (sub-us)"
                           : source == StampSource::SdkReception ? "SDK reception time (no GPS fix)"
                                                                 : "publish time (no SDK timestamp)";
        if (stamp_source_ == StampSource::None)
        {
            RCLCPP_INFO(get_logger(), "IMU timestamp source: %s", name);
        }
        else
        {
            RCLCPP_WARN(get_logger(), "IMU timestamp source CHANGED to %s — expect a step in the stamps.", name);
        }
        stamp_source_ = source;
    }

    /// Look the mounting geometry up once, at whatever time TF has it for.
    ///
    /// Deliberately TimePointZero rather than the measurement stamp: these are
    /// static transforms, and asking for them at a GPS-derived wall-clock time
    /// only invites a clock-domain mismatch with the buffer for no benefit.
    bool ensure_mounting()
    {
        if (have_mounting_)
        {
            return true;
        }
        try
        {
            tf2::fromMsg(tf_buffer_.lookupTransform(frames_.cog, imu_frame_, tf2::TimePointZero).transform,
                         cog_from_imu_);
            tf2::fromMsg(tf_buffer_.lookupTransform(frames_.lidar, imu_frame_, tf2::TimePointZero).transform,
                         lidar_from_imu_);
        }
        catch (const tf2::TransformException &e)
        {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                                 "Waiting for mounting transforms: %s", e.what());
            return false;
        }

        const tf2::Vector3 &arm = cog_from_imu_.getOrigin();
        RCLCPP_INFO(get_logger(), "Mounting geometry latched: IMU sits at (%.3f, %.3f, %.3f) in %s.",
                    arm.x(), arm.y(), arm.z(), frames_.cog.c_str());
        have_mounting_ = true;
        return true;
    }

    void publish_loop()
    {
        while (keep_running_ && rclcpp::ok())
        {
            if (!imu_.is_connected())
            {
                imu_.connect_and_config(imu_dev_path_.c_str(), antenna_a_offset_, gnss_baseline_,
                                        gnss_baseline_unc_, compute_rate_divisor());
                if (!imu_.is_connected())
                {
                    std::this_thread::sleep_for(std::chrono::duration<double>(reconnect_delay_s_));
                    continue;
                }
            }

            VN::Sensor::CompositeDataQueueReturn msg = imu_.get_next_measurement();
            if (!msg)
            {
                RCLCPP_ERROR(get_logger(), "IMU read timeout — assuming dead, disconnecting.");
                imu_.disconnect();
                stamp_source_ = StampSource::None; // re-log the source on the next session
                continue;
            }

            if (!ensure_mounting())
            {
                continue;
            }

            try
            {
                const rclcpp::Time stamp = get_measurement_stamp(msg);

                // Only a *converged* INS gets to speak. Aligning means the
                // onboard filter has not agreed with GNSS yet, and mirena_state
                // seeds its EKF from the first odometry it sees -- so accepting
                // one of those pins the whole run's origin on a position the
                // sensor itself does not stand behind.
                const bool ins_active = read_mode(msg) == InsMode::Tracking;
                if (!ins_active)
                {
                    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 5000,
                                         "Waiting for the INS to converge (no odometry until it does)...");
                }

                mirena::IMU::Output out = pipeline_.process(
                    frames_, stamp, msg, cog_from_imu_, lidar_from_imu_, ins_active);

                imu_in_lidar_pub_->publish(out.lidar_imu);
                imu_in_cog_pub_->publish(out.cog_imu);
                if (out.odom)
                {
                    odom_pub_->publish(*out.odom);
                }
                if (out.fix)
                {
                    fix_pub_->publish(*out.fix);
                }
            }
            catch (const std::exception &e)
            {
                RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                                     "Skipping measurement, processing error: %s", e.what());
            }
        }
    }

    uint16_t compute_rate_divisor()
    {
        const int rate = std::max(1, imu_output_rate_hz_);
        const int divisor = std::max(1, imu_base_rate_hz_ / rate);
        const int actual = imu_base_rate_hz_ / divisor;
        if (actual != rate)
        {
            RCLCPP_WARN(get_logger(),
                        "imu_output_rate_hz=%d does not divide imu_base_rate_hz=%d; streaming at %d Hz instead.",
                        rate, imu_base_rate_hz_, actual);
        }
        return static_cast<uint16_t>(divisor);
    }

    static InsMode read_mode(const VN::Sensor::CompositeDataQueueReturn &msg)
    {
        if (!msg->ins.insStatus.has_value())
        {
            return InsMode::NotTracking;
        }
        // VectorNav INS mode field: 0=NotTracking, 1=Aligning, 2=Tracking, 3=GpsLoss
        switch (msg->ins.insStatus.value().mode)
        {
        case 1:
            return InsMode::Aligning;
        case 2:
            return InsMode::Tracking;
        case 3:
            return InsMode::GpsLoss;
        default:
            return InsMode::NotTracking;
        }
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ImuNode>());
    rclcpp::shutdown();
    return 0;
}
