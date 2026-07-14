#pragma once

#include <vectornav/Interface/Sensor.hpp>
#include <vectornav/Interface/Registers.hpp>

#include <rclcpp/time.hpp>

#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include "ang_accel_estimator.hpp"

#include <tf2/LinearMath/Quaternion.h>

namespace mirena::AHRS
{
    struct PipelineOutput
    {
        sensor_msgs::msg::Imu cog_imu;
        sensor_msgs::msg::Imu lidar_imu;
    };

    class Pipeline
    {
        mirena::AngAccelEstimator::AngAccelEstimator estimator = mirena::AngAccelEstimator::AngAccelEstimator(20);

    public:
        std::unique_ptr<PipelineOutput> process_imu(
            const std::string &cog_frame_id,
            const std::string &lidar_frame_id,
            const rclcpp::Time &stamp,
            const VN::Sensor::CompositeDataQueueReturn &msg,
            const tf2::Vector3 t_imu_to_cog,    // Translation from the IMU cog to the car cog
            const tf2::Vector3 t_imu_to_lidar,  // Translation from the IMU cog to the LiDAR cog
            const tf2::Quaternion r_body_to_ned // Rotation from the IMU body frame to the NED frame given in the IMU body frame
        )
        {
            const double GRAVITY_CONSTANT = 9.80665;

            if (!msg->attitude.quaternion.has_value() ||
                !msg->imu.accel.has_value() ||
                !msg->imu.angularRate.has_value())
            {
                return nullptr;
            }

            auto imu_quat = msg->attitude.quaternion.value();
            auto imu_lin_accel = msg->imu.accel.value();
            auto imu_ang_vel = msg->imu.angularRate.value();

            std::unique_ptr<PipelineOutput> ret = std::make_unique<PipelineOutput>();

            const tf2::Quaternion ned_to_nwu = tf2::Quaternion(1, 0, 0, 0);
            const tf2::Quaternion nwu_to_ned = tf2::Quaternion(-1, 0, 0, 0);
            tf2::Quaternion ned_to_body = r_body_to_ned.inverse();
            tf2::Quaternion body_to_nwu = ned_to_nwu * r_body_to_ned;

            tf2::Quaternion body_orientation = tf2::Quaternion(imu_quat.vector[0], imu_quat.vector[1], imu_quat.vector[2], imu_quat.scalar);
            tf2::Vector3 body_ang_vel = tf2::Vector3(imu_ang_vel[0], imu_ang_vel[1], imu_ang_vel[2]);
            tf2::Vector3 body_lin_accel = tf2::Vector3(imu_lin_accel[0], imu_lin_accel[1], imu_lin_accel[2]);
            // tf2::Vector3 body_ang_accel = estimator.estimate_ang_accel(body_ang_vel, stamp.seconds());
            tf2::Vector3 body_ang_accel = tf2::Vector3(0, 0, 0);

            tf2::Vector3 gravity_compensation = tf2::Vector3(0, 0, GRAVITY_CONSTANT);
            body_lin_accel += tf2::quatRotate(body_orientation.inverse(), gravity_compensation);

            // ----------------------- ------------------------------
            // ----------------------- COG IMU ------------------------------
            // ----------------------- ------------------------------
            {
                auto cog_imu_msg = sensor_msgs::msg::Imu();
                cog_imu_msg.header.frame_id = cog_frame_id;
                cog_imu_msg.header.stamp = stamp;

                // ----------------------- RESPECTIVE FRAMES TO IMU-CENTERED NWU ------------------------------
                auto nwu_orientation = ned_to_nwu * body_orientation * ned_to_body * nwu_to_ned;
                auto nwu_ang_vel = tf2::quatRotate(body_to_nwu, body_ang_vel);
                auto nwu_lin_accel = tf2::quatRotate(body_to_nwu, body_lin_accel);
                auto nwu_ang_accel = tf2::quatRotate(body_to_nwu, body_ang_accel);

                // ----------------------- NWU TO COG (lever arm lmao) ------------------------------
                auto cog_lin_accel = nwu_lin_accel - nwu_ang_accel.cross(t_imu_to_cog) - nwu_ang_vel.cross(nwu_ang_vel.cross(t_imu_to_cog));

                cog_imu_msg.orientation.x = nwu_orientation.x();
                cog_imu_msg.orientation.y = nwu_orientation.y();
                cog_imu_msg.orientation.z = nwu_orientation.z();
                cog_imu_msg.orientation.w = nwu_orientation.w();

                cog_imu_msg.angular_velocity.x = nwu_ang_vel.x();
                cog_imu_msg.angular_velocity.y = nwu_ang_vel.y();
                cog_imu_msg.angular_velocity.z = nwu_ang_vel.z();

                cog_imu_msg.linear_acceleration.x = cog_lin_accel.x();
                cog_imu_msg.linear_acceleration.y = cog_lin_accel.y();
                cog_imu_msg.linear_acceleration.z = cog_lin_accel.z();

                ret->cog_imu = cog_imu_msg;
            }

            // ----------------------- ------------------------------
            // ----------------------- LIDAR IMU ------------------------------
            // ----------------------- ------------------------------
            {
                auto lidar_imu_msg = sensor_msgs::msg::Imu();
                lidar_imu_msg.header.frame_id = lidar_frame_id;
                lidar_imu_msg.header.stamp = stamp;

                // ----------------------- RESPECTIVE FRAMES TO IMU-CENTERED NWU ------------------------------
                auto nwu_orientation = ned_to_nwu * body_orientation * nwu_to_ned;
                auto nwu_ang_vel = tf2::quatRotate(ned_to_nwu, body_ang_vel);
                auto nwu_lin_accel = tf2::quatRotate(ned_to_nwu, body_lin_accel);
                auto nwu_ang_accel = tf2::quatRotate(ned_to_nwu, body_ang_accel);

                // -----------------------  TO COG (lever arm lmao) ------------------------------
                auto cog_lin_accel = nwu_lin_accel - nwu_ang_accel.cross(t_imu_to_lidar) - nwu_ang_vel.cross(nwu_ang_vel.cross(t_imu_to_lidar));

                lidar_imu_msg.orientation.x = nwu_orientation.x();
                lidar_imu_msg.orientation.y = nwu_orientation.y();
                lidar_imu_msg.orientation.z = nwu_orientation.z();
                lidar_imu_msg.orientation.w = nwu_orientation.w();

                lidar_imu_msg.angular_velocity.x = nwu_ang_vel.x();
                lidar_imu_msg.angular_velocity.y = nwu_ang_vel.y();
                lidar_imu_msg.angular_velocity.z = nwu_ang_vel.z();

                lidar_imu_msg.linear_acceleration.x = cog_lin_accel.x();
                lidar_imu_msg.linear_acceleration.y = cog_lin_accel.y();
                lidar_imu_msg.linear_acceleration.z = cog_lin_accel.z();
                ret->lidar_imu = lidar_imu_msg;
            }

            return ret;
        }
    };
}