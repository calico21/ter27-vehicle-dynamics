#pragma once

#include "sensor_msgs/msg/point_cloud2.hpp"
#include "lidar_packet.hpp"
#include "rclcpp/time.hpp"
#include <string>
#include <cstdint>
#include <iostream>

namespace mirena {

struct Timestamp {
    int64_t seconds; // Changed to int64_t because 6 bytes (48 bits) will overflow a standard 32-bit int
    int32_t microseconds; // 4 bytes fits perfectly inside a standard 32-bit int

    double diff_seconds(const Timestamp& other) const {
        if(this->seconds < other.seconds || ((this->seconds == other.seconds) && (this->microseconds < other.microseconds))){
            std::cout << "obsceno" << std::endl;
            exit(39);
        }
        double diff_seconds = (double)(this->seconds - other.seconds);
        return diff_seconds + ((double) this->microseconds - (double)other.microseconds) * 1e-6;
    }
};

struct StampedMSOPPacket {
    mirena::lidar::MSOPPacket packet;
    rclcpp::Time timestamp;
};

    class PointcloudBuilder {
    public:
        /**
         * @brief Constructor
         * @param frame_id_ The TF frame name for the pointcloud header
         * @param max_seq_num_ Estimated packets per frame (default 630 for RS-M1)
         */
        PointcloudBuilder(const char* frame_id_, uint16_t max_seq_num_ = 630) 
            : frame_id(frame_id_), max_seq_num(max_seq_num_) {}

        /**
         * @brief Process an incoming MSOP packet and append to the current frame.
         * @return SharedPtr to completed PointCloud2 if a frame boundary is reached, else nullptr.
         */
        std::unique_ptr<sensor_msgs::msg::PointCloud2> process_packet(const mirena::StampedMSOPPacket &stamped_packet);
        void set_angular_velocity_z(double value) {angular_vel_z = value;};
        void flush(){
          last_sequence_number = 0;
          total_integrated_time = 0;
          frame_start_time = Timestamp();
          accumulated_error_azimuth = 0;
          product = nullptr;
        };
    private:
        
        // Configuration
        const std::string frame_id;
        const uint16_t max_seq_num;
        double angular_vel_z = 0;
        
        // State tracking
        uint16_t last_sequence_number = 0;
        Timestamp frame_start_time;
        double total_integrated_time = 0;
        double accumulated_error_azimuth = 0;

        // The pointcloud message currently being "filled"
        std::unique_ptr<sensor_msgs::msg::PointCloud2> product;
        /**
         * @brief Creates and pre-allocates a new PointCloud2 message.
         */
        std::unique_ptr<sensor_msgs::msg::PointCloud2> make_new_pointcloud(const builtin_interfaces::msg::Time & stamp);
    };

} // namespace mirena
