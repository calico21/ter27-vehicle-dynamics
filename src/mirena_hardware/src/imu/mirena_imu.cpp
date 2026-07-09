#include <chrono>
#include <string>
#include <thread>
#include <atomic>
#include <ctime>
#include <vector>
#include <algorithm>
#include <exception>

#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <rclcpp/rclcpp.hpp>

#include "imu_wrapper.hpp"
#include "ahrs_pipeline.hpp"
#include "ins_pipeline.hpp"

#include <vectornav/Interface/Sensor.hpp>
#include <vectornav/Interface/Registers.hpp>

#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <mirena_common/msg/car.hpp>
#include <mirena_common/msg/heartbeat.hpp>

#include <limits>
#include <cstdint>

using namespace std::chrono_literals;

enum InsMode
{
    NotTracking, // Dead
    Aligning,    // INS Kalman filter working but diverging from GNSS
    Tracking,    // INS Kalman filter and converged with GNSS
    GpsLoss      // Tracking lost
};

struct InsStatus
{
    InsMode mode;
    bool GpsFix;
    bool GpsCompassFix;
};

class ImuNode : public rclcpp::Node
{
public:
    ImuNode()
        : Node("mirena_imu"),
          imu(this->get_logger()),
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
        gps_leap_seconds_ = this->declare_parameter<int>("gps_leap_seconds", 18);

        // Output rate: the binary message is streamed at base_rate / divisor.
        // VN-300 base IMU rate is 400 Hz; VN-200/100 is 800 Hz.
        imu_base_rate_hz_ = this->declare_parameter<int>("imu_base_rate_hz", 400);
        imu_output_rate_hz_ = this->declare_parameter<int>("imu_output_rate_hz", 100);

        // Persist the sensor config to non-volatile flash. Off by default: the
        // driver reconfigures the sensor on every connect, so flash writes (which
        // wear the chip) are unnecessary.
        persist_config_ = this->declare_parameter<bool>("persist_config", false);

        reconnect_delay_s_ = this->declare_parameter<double>("reconnect_delay_sec", 1.0);

        // QoS: IMU is high-rate sensor data, best-effort by default.
        use_sensor_qos_ = this->declare_parameter<bool>("use_sensor_data_qos", true);
        qos_depth_ = this->declare_parameter<int>("qos_depth", 10);

        // Frames
        lidar_frame_ = this->declare_parameter<std::string>("lidar_frame", "car/lidar");
        odom_frame_ = this->declare_parameter<std::string>("odom_frame", "car/odom");
        base_frame_ = this->declare_parameter<std::string>("base_frame", "car/cog");
        gnss_frame_ = this->declare_parameter<std::string>("gnss_frame", "car/gnss");
        imu_frame_ = this->declare_parameter<std::string>("imu_frame", "car/imu");

        gps_leap_seconds_ns_ = static_cast<int64_t>(gps_leap_seconds_) * 1'000'000'000LL;

        if (antenna_a_offset_.size() != 3 || gnss_baseline_.size() != 3 || gnss_baseline_unc_.size() != 3)
        {
            RCLCPP_FATAL(get_logger(), "antenna_a_offset / gnss_compass_baseline / gnss_compass_uncertainty must each have 3 elements.");
            throw std::runtime_error("invalid antenna geometry parameters");
        }

        rclcpp::QoS qos = use_sensor_qos_
                              ? rclcpp::QoS(rclcpp::SensorDataQoS())
                              : rclcpp::QoS(rclcpp::KeepLast(static_cast<size_t>(qos_depth_)));

        RCLCPP_INFO(get_logger(), "Initializing IMU bridge on %s (output %d Hz)", imu_dev_path_.c_str(), imu_output_rate_hz_);

        imu_in_lidar_pub_ = this->create_publisher<sensor_msgs::msg::Imu>("/sensors/lidar/imu", qos);
        imu_in_cog_pub_ = this->create_publisher<sensor_msgs::msg::Imu>("/sensors/cog/imu", qos);
        heartbeat_pub_ = this->create_publisher<mirena_common::msg::Heartbeat>("/health", 10);
        car_pub_ = this->create_publisher<mirena_common::msg::Car>("/sensors/car", qos);
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

        if (imu.is_connected())
        {
            imu.disconnect();
        }
        RCLCPP_INFO(get_logger(), "IMU bridge node destroyed cleanly.");
    }

private:
    // ---- Parameters ----
    std::string imu_dev_path_;
    std::vector<double> antenna_a_offset_;
    std::vector<double> gnss_baseline_;
    std::vector<double> gnss_baseline_unc_;
    int gps_leap_seconds_;
    int64_t gps_leap_seconds_ns_;
    int imu_base_rate_hz_;
    int imu_output_rate_hz_;
    bool persist_config_;
    double reconnect_delay_s_;
    bool use_sensor_qos_;
    int qos_depth_;
    std::string odom_frame_, base_frame_, gnss_frame_, lidar_frame_, imu_frame_;

    // ---- State ----
    bool logged_stamp_source_ = false;
    InsStatus last_status_;

    mirena::IMU::Wrapper imu;
    mirena::AHRS::Pipeline ahrs_pipeline;
    mirena::INS::Pipeline ins_pipeline;

    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;

    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_in_lidar_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_in_cog_pub_;
    rclcpp::Publisher<mirena_common::msg::Car>::SharedPtr car_pub_;
    rclcpp::Publisher<sensor_msgs::msg::NavSatFix>::SharedPtr fix_pub_;
    rclcpp::Publisher<mirena_common::msg::Heartbeat>::SharedPtr heartbeat_pub_;

    std::thread publish_thread_;
    std::atomic<bool> keep_running_{false};

    // voy a suicidarme OZUBA DEJA EL PUTO CLAUDE XFI?????
    // ==========================================
    // Hardware Timestamp (camera-style approach)
    // ==========================================
    rclcpp::Time get_measurement_stamp(const VN::Sensor::CompositeDataQueueReturn &msg)
    {
        // Priority 1: GPS hardware timestamp — absolute wall-clock (sub-µs accuracy)
        if (msg->time.timeStatus.has_value() && msg->time.timeGps.has_value())
        {
            if (msg->time.timeStatus.value().towValid)
            {
                // GPS epoch = Jan 6 1980 = 315,964,800 s after UNIX epoch.
                // GPS time does not count leap seconds; subtract the configured offset.
                constexpr int64_t GPS_TO_UNIX_NS = 315'964'800LL * 1'000'000'000LL;
                int64_t unix_ns = static_cast<int64_t>(msg->time.timeGps.value().nanoseconds()) + GPS_TO_UNIX_NS - gps_leap_seconds_ns_;
                if (!logged_stamp_source_)
                {
                    RCLCPP_INFO(get_logger(), "IMU Timestamp: GPS hardware time active (sub-µs accuracy).");
                    logged_stamp_source_ = true;
                }
                return rclcpp::Time(unix_ns, RCL_SYSTEM_TIME);
            }
        }

        // Priority 2: SDK packet reception time — steady_clock → CLOCK_REALTIME offset (camera-style)
        using tp = std::chrono::steady_clock::time_point;
        if (msg->timestamp != tp{})
        {
            struct timespec ts_mono, ts_real;
            clock_gettime(CLOCK_MONOTONIC, &ts_mono);
            clock_gettime(CLOCK_REALTIME, &ts_real);
            int64_t mono_ns = static_cast<int64_t>(ts_mono.tv_sec) * 1'000'000'000LL + ts_mono.tv_nsec;
            int64_t real_ns = static_cast<int64_t>(ts_real.tv_sec) * 1'000'000'000LL + ts_real.tv_nsec;
            int64_t sdk_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                 msg->timestamp.time_since_epoch())
                                 .count();
            if (!logged_stamp_source_)
            {
                RCLCPP_WARN(get_logger(), "IMU Timestamp: No GPS fix — SDK reception time (steady_clock → epoch).");
                logged_stamp_source_ = true;
            }
            return rclcpp::Time(sdk_ns + (real_ns - mono_ns), RCL_SYSTEM_TIME);
        }

        // Priority 3: fallback to publish time
        if (!logged_stamp_source_)
        {
            RCLCPP_WARN(get_logger(), "IMU Timestamp: SDK timestamp unavailable — falling back to publish time.");
            logged_stamp_source_ = true;
        }
        return this->get_clock()->now();
    }

    bool get_transform(const std::string &target_frame, const std::string &source_frame, const rclcpp::Time &stamp, tf2::Transform &transform)
    {
        try
        {
            geometry_msgs::msg::TransformStamped tf_msg = tf_buffer_.lookupTransform(target_frame, source_frame, stamp);
            tf2::fromMsg(tf_msg.transform, transform);
            return true;
        }
        catch (const tf2::TransformException &e)
        {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "TF lookup failed: %s", e.what());
            return false;
        }
    }

    void publish_loop()
    {
        while (keep_running_ && rclcpp::ok())
        {
            if (!imu.is_connected())
            {
                imu.connect_and_config(imu_dev_path_.c_str(), antenna_a_offset_, gnss_baseline_, gnss_baseline_unc_, compute_rate_divisor());
                if (!imu.is_connected())
                {
                    std::this_thread::sleep_for(std::chrono::duration<double>(reconnect_delay_s_));
                    continue;
                }
            }

            VN::Sensor::CompositeDataQueueReturn msg = imu.get_next_measurement();
            if (!msg)
            {
                RCLCPP_ERROR(get_logger(), "IMU read timeout — assuming dead, disconnecting.");
                imu.disconnect();
                logged_stamp_source_ = false; // re-log the stamp source on the next session
                continue;
            }

            try
            {
                rclcpp::Time stamp = get_measurement_stamp(msg);
                last_status_ = extract_imu_status(msg);
                bool ins_active = last_status_.mode == InsMode::Aligning || last_status_.mode == InsMode::Tracking;

                tf2::Transform t_imu_to_cog, t_imu_to_lidar;
                if (!get_transform(base_frame_, imu_frame_, stamp, t_imu_to_cog) ||
                    !get_transform(lidar_frame_, imu_frame_, stamp, t_imu_to_lidar))
                {
                    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "Missing transforms, skipping measurement");
                    continue;
                }

                tf2::Vector3 t_imu_to_cog_vec = t_imu_to_cog.getOrigin();
                tf2::Vector3 t_imu_to_lidar_vec = t_imu_to_lidar.getOrigin();
                tf2::Quaternion r_body_to_ned = t_imu_to_cog.getRotation();
                

                if (!ins_active)
                {
                    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 5000, "Waiting for GPS Fix...");
                    auto ret = ahrs_pipeline.process_imu(
                        base_frame_,
                        lidar_frame_,
                        stamp,
                        msg,
                        t_imu_to_cog_vec,
                        t_imu_to_lidar_vec,
                        r_body_to_ned);

                    imu_in_lidar_pub_->publish(ret->lidar_imu);
                    imu_in_cog_pub_->publish(ret->cog_imu);
                    heartbeat_pub_->publish(ret->report);
                }
                else
                {
                    auto ret = ins_pipeline.process_imu(
                        odom_frame_,
                        base_frame_,
                        lidar_frame_,
                        gnss_frame_,
                        stamp,
                        msg,
                        t_imu_to_cog_vec,
                        t_imu_to_lidar_vec,
                        r_body_to_ned);

                    imu_in_lidar_pub_->publish(ret->lidar_imu);
                    imu_in_cog_pub_->publish(ret->cog_imu);
                    car_pub_->publish(ret->car);
                    fix_pub_->publish(ret->fix_msg);
                    heartbeat_pub_->publish(ret->report);
                }
            }
            catch (const std::exception &e)
            {
                RCLCPP_WARN(get_logger(), "Skipping measurement, processing error: %s", e.what());
            }
        }
    }

    uint16_t compute_rate_divisor() const
    {
        int rate = std::max(1, imu_output_rate_hz_);
        int divisor = std::max(1, imu_base_rate_hz_ / rate);
        return static_cast<uint16_t>(divisor);
    }

    void print_status_info(VN::Sensor::CompositeDataQueueReturn &msg)
    {
        if (!msg->ins.insStatus.has_value())
        {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "InsStatus not being published.");
        }
        auto ins_status = msg->ins.insStatus.value();

        auto highestCno = std::numeric_limits<uint8_t>::min();

        for (int i = 0; i < msg->gnss.gnss1SatInfo.value().numSats; i++)
        {
            auto currentCno = msg->gnss.gnss1SatInfo.value().cno[i];
            if (currentCno > highestCno)
            {
                highestCno = currentCno;
            }
        }
        auto highestCno2 = std::numeric_limits<uint8_t>::min();

        for (int i = 0; i < msg->gnss2.gnss2SatInfo.value().numSats; i++)
        {
            auto currentCno2 = msg->gnss2.gnss2SatInfo.value().cno[i];
            if (currentCno2 > highestCno2)
            {
                highestCno2 = currentCno2;
            }
        }

        RCLCPP_DEBUG(get_logger(), "SatInfo1=%d SatInfo2=%d",
                     highestCno, highestCno2);

        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 5000, "GpsFix=%s GpsCompass=%s GpsErr=%i mode=%i",
                             ins_status.gnssFix ? "yes" : "no", ins_status.gnssCompassFix ? "yes" : "no",
                             ins_status.gnssErr, ins_status.mode);
    }

    InsStatus extract_imu_status(const VN::Sensor::CompositeDataQueueReturn &msg)
    {
        InsStatus ret;
        if (!msg->ins.insStatus.has_value())
        {
            ret.mode = InsMode::NotTracking;
            ret.GpsFix = false;
            ret.GpsCompassFix = false;
            return ret;
        }

        auto ins_status = msg->ins.insStatus.value();
        ret.GpsFix = ins_status.gnssFix == 1;
        ret.GpsCompassFix = ins_status.gnssCompassFix == 1;
        // VectorNav INS mode field: 0=NotTracking, 1=Aligning, 2=Tracking, 3=GpsLoss
        switch (ins_status.mode)
        {
        case 1:
            ret.mode = InsMode::Aligning;
            break;
        case 2:
            ret.mode = InsMode::Tracking;
            break;
        case 3:
            ret.mode = InsMode::GpsLoss;
            break;
        default:
            ret.mode = InsMode::NotTracking;
            break;
        }
        return ret;
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ImuNode>());
    rclcpp::shutdown();
    return 0;
}
