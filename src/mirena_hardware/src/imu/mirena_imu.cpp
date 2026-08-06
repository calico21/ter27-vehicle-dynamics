#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <exception>
#include <stdexcept>
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
#include "stamp_estimator.hpp"

#include <vectornav/Interface/Sensor.hpp>

#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <nav_msgs/msg/odometry.hpp>

using namespace std::chrono_literals;

namespace
{
    /// Frame names. Constants rather than parameters because nothing in this
    /// workspace has ever overridden them and the nodes on the other end of
    /// these topics hard-code the same strings (mirena_lidar's "car/lidar",
    /// mirena_perception's CAR_FRAME). Parameters that only ever take their
    /// default are configuration in appearance only.
    const mirena::IMU::Frames kFrames{"odom", "car/cog", "car/lidar", "car/gnss"};
    constexpr const char *kImuFrame = "car/imu";

    /// VN-300 base IMU rate. Fixed by the hardware, not a choice.
    constexpr int kBaseRateHz = 400;

    constexpr auto kReconnectDelay = 1s;

    /// GPS epoch (Jan 6 1980) as UNIX nanoseconds, and the leap seconds GPS
    /// time does not count. Only used to report how far the host clock has
    /// drifted; a stale leap count would move that report by a second and
    /// change no stamp.
    constexpr int64_t kGpsToUnixNs = 315'964'800LL * 1'000'000'000LL;
    constexpr int64_t kLeapSecondsNs = 18LL * 1'000'000'000LL;
} // namespace

class ImuNode : public rclcpp::Node
{
public:
    ImuNode()
        : Node("mirena_imu"),
          imu_(this->get_logger()),
          tf_buffer_(this->get_clock()),
          tf_listener_(tf_buffer_)
    {
        imu_dev_path_ = this->declare_parameter<std::string>("imu_device", "/dev/ter_IMU");
        imu_output_rate_hz_ = this->declare_parameter<int>("imu_output_rate_hz", 100);

        // Antenna geometry (metres). The only genuinely per-car settings here:
        // they describe where the antennas are bolted, and the onboard filter
        // is wrong without them.
        antenna_a_offset_ = this->declare_parameter<std::vector<double>>("antenna_a_offset", {-0.11, 0.0, 0.52});
        gnss_baseline_ = this->declare_parameter<std::vector<double>>("gnss_compass_baseline", {1.50, 0.0, -0.36});
        gnss_baseline_unc_ = this->declare_parameter<std::vector<double>>("gnss_compass_uncertainty", {0.04, 0.04, 0.04});
        if (antenna_a_offset_.size() != 3 || gnss_baseline_.size() != 3 || gnss_baseline_unc_.size() != 3)
        {
            RCLCPP_FATAL(get_logger(), "antenna_a_offset / gnss_compass_baseline / gnss_compass_uncertainty must each have 3 elements.");
            throw std::runtime_error("invalid antenna geometry parameters");
        }

        RCLCPP_INFO(get_logger(), "Initializing IMU bridge on %s (output %d Hz)",
                    imu_dev_path_.c_str(), imu_output_rate_hz_);

        // Sensor-layer outputs only: raw IMU streams, the onboard INS solution
        // as odometry, and the GNSS fix. State fusion (/state/car) and health
        // reporting are the pipeline's job (mirena_state), not the driver's.
        const rclcpp::QoS qos = rclcpp::SensorDataQoS();
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
    std::string imu_dev_path_;
    int imu_output_rate_hz_;
    std::vector<double> antenna_a_offset_;
    std::vector<double> gnss_baseline_;
    std::vector<double> gnss_baseline_unc_;

    mirena::IMU::Wrapper imu_;
    mirena::IMU::Pipeline pipeline_;
    mirena::IMU::StampEstimator stamp_estimator_;

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

    /// The measurement instant, in the clock domain the rest of the stack uses.
    ///
    /// The sensor knows *when* far better than this process does: timeStartup
    /// is stamped at the sample, inside the VN-300, before the serial link and
    /// the SDK's reader thread add their 10 ms of jitter. What the sensor does
    /// not know is ROS time. StampEstimator holds the two together.
    rclcpp::Time get_measurement_stamp(const VN::Sensor::CompositeDataQueueReturn &msg)
    {
        // Read first: this bounds when the sample was taken, and every
        // microsecond spent before reading it is bias handed to the estimator.
        const rclcpp::Time arrival = this->now();

        if (!msg->time.timeStartup.has_value())
        {
            // Requested explicitly in BinaryOutput1, so its absence means the
            // sensor is not streaming what it was configured to. Better to say
            // so and drop the measurement than to quietly stamp it with a
            // worse clock for the rest of the session.
            throw std::runtime_error("measurement has no timeStartup");
        }

        return rclcpp::Time(
            stamp_estimator_.stamp(static_cast<int64_t>(msg->time.timeStartup.value().nanoseconds()),
                                   arrival.nanoseconds()),
            arrival.get_clock_type());
    }

    /// Report how far the host clock has drifted from GPS UTC.
    ///
    /// This is the number that used to leak into the stamps. It does not stop
    /// existing just because they no longer carry it: it is still the offset
    /// every bag has to be retimed by (tools/retime_bag.py) and the reason
    /// absolute times in the logs cannot be trusted. Measured against
    /// CLOCK_REALTIME rather than now() so it means the same thing under
    /// use_sim_time, and reported rather than corrected -- the fix is gpsd +
    /// chrony on the Jetson, not arithmetic in a driver.
    void report_clock_skew(const VN::Sensor::CompositeDataQueueReturn &msg)
    {
        if (!msg->time.timeGps.has_value() || !msg->time.timeStatus.has_value() ||
            !msg->time.timeStatus.value().towValid)
        {
            return;
        }

        struct timespec ts_real;
        clock_gettime(CLOCK_REALTIME, &ts_real);
        const int64_t host_ns = static_cast<int64_t>(ts_real.tv_sec) * 1'000'000'000LL + ts_real.tv_nsec;
        const int64_t gps_ns = static_cast<int64_t>(msg->time.timeGps.value().nanoseconds()) +
                               kGpsToUnixNs - kLeapSecondsNs;
        const double skew_s = static_cast<double>(host_ns - gps_ns) * 1e-9;

        // 0.1 s is already an order of magnitude past what any time alignment
        // in this stack tolerates.
        if (std::abs(skew_s) > 0.1)
        {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 30000,
                                 "Host clock is %.3f s %s GPS UTC. Stamps are in ROS time so the stack "
                                 "stays consistent, but absolute times and bag arrival times carry this "
                                 "error (discipline the clock with gpsd + chrony; retime existing bags).",
                                 std::abs(skew_s), skew_s > 0 ? "ahead of" : "behind");
        }
    }

    /// Look the mounting geometry up once, at whatever time TF has it for.
    ///
    /// Deliberately TimePointZero rather than the measurement stamp: these are
    /// static transforms, and asking for them at a specific time only invites a
    /// mismatch with the buffer for no benefit.
    bool ensure_mounting()
    {
        if (have_mounting_)
        {
            return true;
        }
        try
        {
            tf2::fromMsg(tf_buffer_.lookupTransform(kFrames.cog, kImuFrame, tf2::TimePointZero).transform,
                         cog_from_imu_);
            tf2::fromMsg(tf_buffer_.lookupTransform(kFrames.lidar, kImuFrame, tf2::TimePointZero).transform,
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
                    arm.x(), arm.y(), arm.z(), kFrames.cog.c_str());
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
                                        gnss_baseline_unc_, rate_divisor());
                if (!imu_.is_connected())
                {
                    std::this_thread::sleep_for(kReconnectDelay);
                    continue;
                }
            }

            VN::Sensor::CompositeDataQueueReturn msg = imu_.get_next_measurement();
            if (!msg)
            {
                RCLCPP_ERROR(get_logger(), "IMU read timeout — assuming dead, disconnecting.");
                imu_.disconnect();
                // It may come back power-cycled, with timeStartup restarted at
                // zero. The estimator detects that itself, but there is nothing
                // worth keeping across a reconnect either way.
                stamp_estimator_.reset();
                continue;
            }

            if (!ensure_mounting())
            {
                continue;
            }

            try
            {
                const rclcpp::Time stamp = get_measurement_stamp(msg);
                report_clock_skew(msg);

                // Only a *converged* INS gets to speak. Aligning means the
                // onboard filter has not agreed with GNSS yet, and mirena_state
                // seeds its EKF from the first odometry it sees -- so accepting
                // one of those pins the whole run's origin on a position the
                // sensor itself does not stand behind.
                const bool ins_active = ins_tracking(msg);
                if (!ins_active)
                {
                    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 5000,
                                         "Waiting for the INS to converge (no odometry until it does)...");
                }

                mirena::IMU::Output out = pipeline_.process(
                    kFrames, stamp, msg, cog_from_imu_, lidar_from_imu_, ins_active);

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

    /// The binary message streams at kBaseRateHz / divisor.
    uint16_t rate_divisor()
    {
        const int rate = std::max(1, imu_output_rate_hz_);
        const int divisor = std::max(1, kBaseRateHz / rate);
        const int actual = kBaseRateHz / divisor;
        if (actual != rate)
        {
            RCLCPP_WARN(get_logger(),
                        "imu_output_rate_hz=%d does not divide the %d Hz base rate; streaming at %d Hz instead.",
                        rate, kBaseRateHz, actual);
        }
        return static_cast<uint16_t>(divisor);
    }

    /// VectorNav INS mode 2 is "tracking"; 0, 1 and 3 are no solution,
    /// aligning and dead reckoning after GPS loss. Nothing here needs to tell
    /// those three apart -- none of them may publish odometry.
    static bool ins_tracking(const VN::Sensor::CompositeDataQueueReturn &msg)
    {
        return msg->ins.insStatus.has_value() && msg->ins.insStatus.value().mode == 2;
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ImuNode>());
    rclcpp::shutdown();
    return 0;
}
