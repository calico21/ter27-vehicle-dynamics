#include "lidar_packet.hpp"
#include "pointcloud_builder.hpp"
#include "common/udp_listener.hpp"
#include "lidar_socket_pair.hpp"

#include <iostream>
#include <chrono>
#include <thread>
#include <poll.h>
#include <mutex>
#include <deque>
#include <atomic>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>

#define POINTCLOUD_PUB_TOPIC "/sensors/lidar/points"

using namespace std::chrono_literals;
using namespace mirena;

class MirenaLidar : public rclcpp::Node
{
    // Member Variables
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_pub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub;
    rclcpp::TimerBase::SharedPtr process_timer_;
    mirena::PointcloudBuilder pc_builder_;

    in_addr_t _self_ip;
    in_addr_t _lidar_ip;
    uint16_t _msop_port;
    uint16_t _difop_port;
    mirena::LidarSocketPair sockets;

    std::thread capture_thread_;
    std::atomic<bool> keep_running_{false};
    std::mutex pc_builder_configure_mtx_;
    std::mutex msop_mtx_;
    std::deque<StampedMSOPPacket> msop_queue_;

    // Added Parameter Variables & Callbacks
    std::atomic<bool> use_imu_{true};
    std::atomic<double> forced_packet_offset_{0.0};
    std::atomic<double> last_imu_stamp_{0.0};
    rclcpp::Node::OnSetParametersCallbackHandle::SharedPtr param_callback_handle_;

public:
    MirenaLidar() : rclcpp::Node("mirena_lidar"),
                    pc_builder_("car/lidar")
    {
        // 1. Declare Parameters with default fallback choices
        this->declare_parameter("self_ip", "192.168.1.102");
        this->declare_parameter("lidar_ip", "192.168.1.200");
        this->declare_parameter("msop_port", 6699);
        this->declare_parameter("difop_port", 7788);
        this->declare_parameter("use_imu", true);
        this->declare_parameter("forced_packet_offset", 0.00);

        // 2. Fetch the parameter values (automatically respects any YAML config overrides)
        auto self_ip_str = this->get_parameter("self_ip").as_string();
        auto lidar_ip_str = this->get_parameter("lidar_ip").as_string();
        auto msop_port_val = this->get_parameter("msop_port").as_int();
        auto difop_port_val = this->get_parameter("difop_port").as_int();
        this->_self_ip = inet_addr(self_ip_str.c_str());
        this->_lidar_ip = inet_addr(lidar_ip_str.c_str());
        this->_msop_port = htons(static_cast<uint16_t>(msop_port_val));
        this->_difop_port = htons(static_cast<uint16_t>(difop_port_val));

        use_imu_ = this->get_parameter("use_imu").as_bool();
        forced_packet_offset_ = this->get_parameter("forced_packet_offset").as_double();

        // Register parameter modification callback
        param_callback_handle_ = this->add_on_set_parameters_callback(
            std::bind(&MirenaLidar::on_parameter_change, this, std::placeholders::_1));

        // RCLCPP_INFO(get_logger(), "Initializing Sockets on IP %s (Lidar Target: %s)...",
        //             self_ip_str.c_str(), lidar_ip_str.c_str());

        // Initialize Standard Publisher
        pointcloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            POINTCLOUD_PUB_TOPIC, rclcpp::QoS(rclcpp::SensorDataQoS()));

        // Motion de-skew consumes the lidar-frame, lever-arm-compensated IMU
        // stream (not the cog stream on /sensors/imu)
        imu_sub = this->create_subscription<sensor_msgs::msg::Imu>("/sensors/lidar/imu", rclcpp::QoS(rclcpp::SensorDataQoS()), std::bind(&MirenaLidar::imu_callback, this, std::placeholders::_1));

        // Start capture thread execution instantly
        keep_running_ = true;
        capture_thread_ = std::thread(&MirenaLidar::capture_loop, this);

        // Setup the localized parsing timer loop
        process_timer_ = this->create_wall_timer(
            1ms, std::bind(&MirenaLidar::process_packets, this));

        RCLCPP_INFO(get_logger(), "Mirena Lidar Bridge Core Initialized and Streaming.");
    }

    ~MirenaLidar()
    {
        // Safely pull thread down cleanly on node shutdown to prevent memory crashes
        keep_running_ = false;
        if (capture_thread_.joinable())
        {
            capture_thread_.join();
        }

        if (process_timer_)
        {
            process_timer_->cancel();
        }

        sockets.invalidate();
        RCLCPP_INFO(get_logger(), "Lidar Thread Joined and Sockets Closed cleanly.");
    }

private:
    rcl_interfaces::msg::SetParametersResult on_parameter_change(const std::vector<rclcpp::Parameter> &parameters)
    {
        rcl_interfaces::msg::SetParametersResult result;
        result.successful = true;
        result.reason = "success";

        for (const auto &param : parameters)
        {
            if (param.get_name() == "use_imu")
            {
                use_imu_ = param.as_bool();
                RCLCPP_INFO(get_logger(), "Parameter 'use_imu' updated to: %s", use_imu_ ? "true" : "false");
                if (!use_imu_)
                {
                    pc_builder_.set_angular_velocity_z(0);
                }
            }
            else if (param.get_name() == "forced_packet_offset")
            {
                forced_packet_offset_ = param.as_double();
                RCLCPP_INFO(get_logger(), "Parameter 'forced_packet_offset' updated to: %f", forced_packet_offset_.load());
            }
        }
        return result;
    }

    void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg)
    {
        if (!use_imu_)
        {
            return;
        }
        last_imu_stamp_.store(rclcpp::Time(msg->header.stamp).seconds());
        std::lock_guard<std::mutex> lock(pc_builder_configure_mtx_);
        this->pc_builder_.set_angular_velocity_z(msg->angular_velocity.z);
    }

    void capture_loop()
    {
        while (keep_running_ && rclcpp::ok())
        {
            if (!sockets.is_valid())
            {
                RCLCPP_INFO(this->get_logger(), "Trying to Adquire sockets...");
                reset_capture();
                std::this_thread::sleep_for(1s);
                sockets = mirena::LidarSocketPair(this->_self_ip, this->_lidar_ip, this->_msop_port, this->_difop_port);
                if(sockets.is_valid()){
                    RCLCPP_INFO(this->get_logger(), "Adquired sockets successfully; Starting capture");
                }
                continue;
            } else {
                //RCLCPP_INFO(this->get_logger(), "Valid Sockets");
            }
            capture();
        }
    }

    void capture()
    {
        struct pollfd pfd;
        pfd.fd = sockets.get_msop_fd();
        pfd.events = POLLIN;

        int ret = poll(&pfd, 1, 10);
        if (ret > 0 && (pfd.revents & POLLIN))
        {
            mirena::lidar::MSOPPacket packet;
            ssize_t size = sockets.get_msop_socket()->_recvfrom(&packet, sizeof(packet), 0);
            if (size == sizeof(mirena::lidar::MSOPPacket))
            {
                std::lock_guard<std::mutex> lock(msop_mtx_);
                msop_queue_.push_back({packet, get_clock()->now()});
            }
        }
    }

    void reset_capture()
    {
        sockets.invalidate();
        {
            std::lock_guard<std::mutex> lock(msop_mtx_);
            msop_queue_.clear();
        }
        {
            std::lock_guard<std::mutex> lock(pc_builder_configure_mtx_);
            pc_builder_.flush();
        }
        return;
    }

    void process_packets()
    {
        sensor_msgs::msg::PointCloud2::SharedPtr msg = nullptr;
        StampedMSOPPacket packet;
        while (true)
        {
            {
                std::lock_guard<std::mutex> lock(msop_mtx_);
                if (msop_queue_.empty() || (use_imu_.load() && (last_imu_stamp_.load() - msop_queue_.at(0).timestamp.seconds()) < forced_packet_offset_))
                {
                    return;
                }
                packet = msop_queue_.at(0);
                msop_queue_.pop_front();
            }
            {
                std::lock_guard<std::mutex> lock(pc_builder_configure_mtx_);
                msg = pc_builder_.process_packet(packet);
            }
            if (msg != nullptr)
            {
                pointcloud_pub_->publish(*msg);
                msg = nullptr;
            }
        }
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MirenaLidar>();

    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();

    rclcpp::shutdown();
    return 0;
}
