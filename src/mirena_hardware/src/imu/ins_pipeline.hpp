#pragma once

#include <vectornav/Interface/Sensor.hpp>
#include <vectornav/Interface/Registers.hpp>

#include <rclcpp/time.hpp>

#include "gnss_conversions.hpp"

#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>

namespace mirena::INS
{
    struct PipelineOutput
    {
        sensor_msgs::msg::NavSatFix fix_msg;
        sensor_msgs::msg::Imu cog_imu;
        sensor_msgs::msg::Imu lidar_imu;
        // Onboard INS solution (REP-105): pose in the odom frame, twist in the
        // body frame. Consumed by mirena_state for sensor fusion.
        nav_msgs::msg::Odometry odom;
    };

    struct GeodesicRef
    {
        double latitude;
        double longitude;
        double altitude;
    };

    class Pipeline
    {
        std::optional<GeodesicRef> geodesic_ref;

    public:
        std::unique_ptr<PipelineOutput> process_imu(
            const std::string &odom_frame_id,
            const std::string &cog_frame_id,
            const std::string &lidar_frame_id,
            const std::string &gnss_frame_id,
            const rclcpp::Time &stamp,
            const VN::Sensor::CompositeDataQueueReturn &msg,
            const tf2::Vector3 t_imu_to_cog,    // Translation from the IMU cog to the car cog
            const tf2::Vector3 t_imu_to_lidar,  // Translation from the IMU cog to the LiDAR cog
            const tf2::Quaternion r_body_to_ned // Rotation from the IMU body frame to the NED frame given in the IMU body frame
        )
        {
            if (!msg->attitude.quaternion.has_value() ||
                !msg->imu.angularRate.has_value() ||
                !msg->ins.posLla.has_value() ||
                !msg->ins.velBody.has_value() ||
                !msg->attitude.linBodyAcc.has_value())
            {
                throw std::runtime_error("Missing parameters // mode: " + std::to_string(msg->ins.insStatus.value().mode) + " gps_fix " + std::to_string(msg->ins.insStatus.value().gnssFix) + " gps_compass " +  std::to_string(msg->ins.insStatus.value().gnssCompassFix) + " error?" + std::to_string(msg->ins.insStatus.value().imuErr || msg->ins.insStatus.value().gnssErr || msg->ins.insStatus.value().magPresErr));
            }

            auto imu_quat = msg->attitude.quaternion.value();
            auto imu_ang_vel = msg->imu.angularRate.value();
            auto imu_gnss_pos = msg->ins.posLla.value();
            auto imu_lin_vel = msg->ins.velBody.value();
            auto imu_lin_accel = msg->attitude.linBodyAcc.value();

            std::unique_ptr<PipelineOutput> ret = std::make_unique<PipelineOutput>();
            const tf2::Quaternion ned_to_nwu = tf2::Quaternion(1, 0, 0, 0);
            const tf2::Quaternion nwu_to_ned = tf2::Quaternion(-1, 0, 0, 0);
            tf2::Quaternion ned_to_body = r_body_to_ned.inverse();
            tf2::Quaternion body_to_nwu = ned_to_nwu * r_body_to_ned;

            if (!geodesic_ref.has_value())
            {
                geodesic_ref = {imu_gnss_pos.lat, imu_gnss_pos.lon, imu_gnss_pos.alt};
            }

            double N, E, D;
            mirena::GNSSConversion::latLonAltToNED(imu_gnss_pos.lat, imu_gnss_pos.lon, imu_gnss_pos.alt,
                                                   geodesic_ref->latitude, geodesic_ref->longitude, geodesic_ref->altitude,
                                                   N, E, D);

            tf2::Quaternion body_orientation = tf2::Quaternion(imu_quat.vector[0], imu_quat.vector[1], imu_quat.vector[2], imu_quat.scalar);
            tf2::Vector3 body_ang_vel = tf2::Vector3(imu_ang_vel[0], imu_ang_vel[1], imu_ang_vel[2]);
            tf2::Vector3 ned_imu_pos = tf2::Vector3(N, E, D);
            tf2::Vector3 body_lin_vel = tf2::Vector3(imu_lin_vel[0], imu_lin_vel[1], imu_lin_vel[2]);
            tf2::Vector3 body_lin_accel = tf2::Vector3(imu_lin_accel[0], imu_lin_accel[1], imu_lin_accel[2]);
            // tf2::Vector3 body_ang_accel = estimator.estimate_ang_accel(body_ang_vel, stamp.seconds());
            tf2::Vector3 body_ang_accel = tf2::Vector3(0, 0, 0);

            // ----------------------- ------------------------------
            // ----------------------- COG IMU ------------------------------
            // ----------------------- ------------------------------
            {
                auto cog_imu_msg = sensor_msgs::msg::Imu();
                cog_imu_msg.header.frame_id = cog_frame_id;
                cog_imu_msg.header.stamp = stamp;

                auto odom_msg = nav_msgs::msg::Odometry();
                odom_msg.header.frame_id = odom_frame_id;
                odom_msg.header.stamp = stamp;
                odom_msg.child_frame_id = cog_frame_id;

                // ----------------------- RESPECTIVE FRAMES TO IMU-CENTERED NWU ------------------------------
                auto nwu_orientation = ned_to_nwu * body_orientation * ned_to_body * nwu_to_ned;
                auto nwu_ang_vel = tf2::quatRotate(body_to_nwu, body_ang_vel);
                auto nwu_imu_pos = tf2::quatRotate(ned_to_nwu, ned_imu_pos);
                auto nwu_lin_vel = tf2::quatRotate(body_to_nwu, body_lin_vel);
                auto nwu_lin_accel = tf2::quatRotate(body_to_nwu, body_lin_accel);
                auto nwu_ang_accel = tf2::quatRotate(body_to_nwu, body_ang_accel);

                // ----------------------- NWU TO COG (lever arm lmao) ------------------------------
                auto nwu_cog_pos = nwu_imu_pos + t_imu_to_cog;
                auto cog_lin_accel = nwu_lin_accel - nwu_ang_accel.cross(t_imu_to_cog) - nwu_ang_vel.cross(nwu_ang_vel.cross(t_imu_to_cog));
                auto cog_lin_vel = nwu_lin_vel - nwu_ang_vel.cross(t_imu_to_cog);

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

                // ----------------------- INS ODOMETRY (REP-105) ------------------------------
                // Pose in the odom frame
                odom_msg.pose.pose.position.x = nwu_cog_pos[0];
                odom_msg.pose.pose.position.y = nwu_cog_pos[1];
                odom_msg.pose.pose.position.z = nwu_cog_pos[2];
                odom_msg.pose.pose.orientation.x = nwu_orientation.x();
                odom_msg.pose.pose.orientation.y = nwu_orientation.y();
                odom_msg.pose.pose.orientation.z = nwu_orientation.z();
                odom_msg.pose.pose.orientation.w = nwu_orientation.w();

                // Twist in the body (cog) frame, REP-103: x forward, y left.
                // NOTE: cog_lin_vel inherits the historical NWU/cog lever-arm
                // frame mixing above (t_imu_to_cog is a cog-frame offset applied
                // to NWU vectors); TODO express the lever arm in NWU first.
                auto body_cog_lin_vel = tf2::quatRotate(nwu_orientation.inverse(), cog_lin_vel);
                auto body_ang_vel_cog = tf2::quatRotate(nwu_orientation.inverse(), nwu_ang_vel);
                odom_msg.twist.twist.linear.x = body_cog_lin_vel[0];
                odom_msg.twist.twist.linear.y = body_cog_lin_vel[1];
                odom_msg.twist.twist.linear.z = body_cog_lin_vel[2];
                odom_msg.twist.twist.angular.x = body_ang_vel_cog[0];
                odom_msg.twist.twist.angular.y = body_ang_vel_cog[1];
                odom_msg.twist.twist.angular.z = body_ang_vel_cog[2];

                // Covariance diagonals from the onboard filter's 1-sigma
                // uncertainties, with conservative fallbacks when not streamed.
                const double pos_var = msg->ins.posU.has_value()
                                           ? static_cast<double>(msg->ins.posU.value()) * msg->ins.posU.value()
                                           : 1.0;
                const double vel_var = msg->ins.velU.has_value()
                                           ? static_cast<double>(msg->ins.velU.value()) * msg->ins.velU.value()
                                           : 0.5;
                double yaw_var = 0.05;
                if (msg->attitude.yprU.has_value())
                {
                    const double yaw_unc_rad = msg->attitude.yprU.value()[0] * M_PI / 180.0;
                    yaw_var = yaw_unc_rad * yaw_unc_rad;
                }
                constexpr double GYRO_VAR = 1e-4; // not streamed by the filter

                for (int i = 0; i < 3; i++)
                {
                    odom_msg.pose.covariance[i * 7] = pos_var;
                    odom_msg.pose.covariance[(i + 3) * 7] = yaw_var;
                    odom_msg.twist.covariance[i * 7] = vel_var;
                    odom_msg.twist.covariance[(i + 3) * 7] = GYRO_VAR;
                }

                ret->cog_imu = cog_imu_msg;
                ret->odom = odom_msg;
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

            // ----------------------- ------------------------------
            // ----------------------- FIX MSG ------------------------------
            // ----------------------- ------------------------------
            {
                auto gnss_pos = msg->ins.posLla.value();

                auto fix_msg = sensor_msgs::msg::NavSatFix();
                fix_msg.header.frame_id = gnss_frame_id;
                fix_msg.header.stamp = stamp;

                // publish_fix is only called when we have a fix.
                fix_msg.status.status = sensor_msgs::msg::NavSatStatus::STATUS_FIX;
                fix_msg.status.service = sensor_msgs::msg::NavSatStatus::SERVICE_GPS;

                fix_msg.latitude = gnss_pos.lat;
                fix_msg.longitude = gnss_pos.lon;
                fix_msg.altitude = gnss_pos.alt;

                fix_msg.position_covariance_type = sensor_msgs::msg::NavSatFix::COVARIANCE_TYPE_UNKNOWN;

                ret->fix_msg = fix_msg;
            }

            return ret;
        }

        void set_geodesic_ref(const GeodesicRef &ref)
        {
            geodesic_ref = ref;
        }
    };
}