#include "pointcloud_builder.hpp"
#include "rclcpp/rclcpp.hpp"
#include <cstring>
#include <cmath>
#include <arpa/inet.h>

// Global optimization for the whole file
#pragma GCC optimize("unroll-loops")
#pragma GCC optimize("-ffast-math")

#define NUM_CHANNELS 5
#define NUM_DATA_BLOCKS 25
#define RADIAL_RES 0.005f
#define ANG_RES (0.01f * M_PI / 180.0f)

using namespace mirena;

struct alignas(4) LidarPoint
{
    float x, y, z, intensity;
};

Timestamp parse_timestamp(const uint8_t timestamp[10])
{
    Timestamp ts;

    ts.seconds = ((int64_t)timestamp[0] << 40) |
                 ((int64_t)timestamp[1] << 32) |
                 ((int64_t)timestamp[2] << 24) |
                 ((int64_t)timestamp[3] << 16) |
                 ((int64_t)timestamp[4] << 8) |
                 ((int64_t)timestamp[5]);

    ts.microseconds = ((int64_t)timestamp[6] << 24) |
                      ((int64_t)timestamp[7] << 16) |
                      ((int64_t)timestamp[8] << 8) |
                      ((int64_t)timestamp[9]);

    return ts;
}
// Switching return type to unique_ptr for Zero-Copy
std::unique_ptr<sensor_msgs::msg::PointCloud2> PointcloudBuilder::process_packet(const StampedMSOPPacket &stamped_packet)
{
    // Header check
    const lidar::MSOPPacket *packet = &stamped_packet.packet;
    uint8_t expected_pkt_header[4] = {0x55, 0xaa, 0x5a, 0xa5};
    if (std::memcmp(packet->header.pkt_header, expected_pkt_header, 4) != 0)
    {
        return nullptr;
    }

    Timestamp packet_timestamp = parse_timestamp(packet->header.timestamp);

    if (product == nullptr)
    {
        product = this->make_new_pointcloud(stamped_packet.timestamp);
        frame_start_time = packet_timestamp;
        total_integrated_time = 0;
        accumulated_error_azimuth = 0;
    }

    uint16_t seq_number = ntohs(packet->header.pkt_psn);
    std::unique_ptr<sensor_msgs::msg::PointCloud2> ret = nullptr;

    // Frame boundary
    if (seq_number < last_sequence_number && (last_sequence_number - seq_number) > (max_seq_num / 10))
    {
        if (!product->data.empty())
        {
            product->width = product->data.size() / product->point_step;
            product->row_step = product->data.size();
            ret = std::move(product); // Transfer ownership
        }
        product = this->make_new_pointcloud(stamped_packet.timestamp);
        frame_start_time = packet_timestamp;
        total_integrated_time = 0;
        accumulated_error_azimuth = 0;
    }
    last_sequence_number = seq_number;

    size_t current_offset = product->data.size();
    product->data.resize(current_offset + (NUM_DATA_BLOCKS * NUM_CHANNELS * 16));
    uint8_t *write_ptr = &product->data[current_offset];
    size_t bytes_written = 0;

    double base_block_frame_offset = packet_timestamp.diff_seconds(frame_start_time);
    for (int block_n = 0; block_n < NUM_DATA_BLOCKS; block_n++)
    {
        const lidar::MSOPDataBlock *block = &packet->blocks[block_n];
        double block_frame_offset = base_block_frame_offset + ((double)(block->time_offset & 0x00111111)) * 1e-6;
        accumulated_error_azimuth += angular_vel_z * (block_frame_offset - total_integrated_time);
        total_integrated_time = block_frame_offset;

// Explicitly helping the compiler vectorize this loop
#pragma GCC ivdep
        for (int channel_n = 0; channel_n < NUM_CHANNELS; channel_n++)
        {
            const lidar::ChannelBlock *src = &block->channels[channel_n];

            float radius = (float)((src->radius[0] << 8) | src->radius[1]) * RADIAL_RES;
            if (radius < 0.4f)
                continue;

            float elevation = (float)((src->elevation[0] << 8 | src->elevation[1]) - 32768) * ANG_RES;
            float azimuth = (float)((src->azimuth[0] << 8 | src->azimuth[1]) - 32768) * ANG_RES;
            azimuth += accumulated_error_azimuth;

            // Using sincosf for Jetson hardware acceleration
            float s_el, c_el, s_az, c_az;
            sincosf(elevation, &s_el, &c_el);
            sincosf(azimuth, &s_az, &c_az);

            float r_c_el = radius * c_el;

            LidarPoint p;
            p.x = r_c_el * c_az;
            p.y = r_c_el * s_az;
            p.z = radius * s_el;
            p.intensity = (float)src->intensity / 255.0f;

            std::memcpy(write_ptr, &p, 16);
            write_ptr += 16;
            bytes_written += 16;
        }
    }

    product->data.resize(current_offset + bytes_written);
    return ret;
}

std::unique_ptr<sensor_msgs::msg::PointCloud2> PointcloudBuilder::make_new_pointcloud(const builtin_interfaces::msg::Time &stamp)
{
    auto msg = std::make_unique<sensor_msgs::msg::PointCloud2>();
    msg->header.frame_id = frame_id;
    msg->header.stamp = stamp;
    msg->height = 1;
    msg->point_step = 16;
    msg->fields.resize(4);
    const char *names[] = {"x", "y", "z", "intensity"};
    for (int i = 0; i < 4; i++)
    {
        msg->fields[i].name = names[i];
        msg->fields[i].offset = i * 4;
        msg->fields[i].datatype = sensor_msgs::msg::PointField::FLOAT32;
        msg->fields[i].count = 1;
    }

    msg->data.reserve(max_seq_num * NUM_DATA_BLOCKS * NUM_CHANNELS * 16);
    return msg;
}