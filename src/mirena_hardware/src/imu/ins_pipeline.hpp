#pragma once

#include <vectornav/Interface/Sensor.hpp>
#include <vectornav/Interface/Registers.hpp>

#include <rclcpp/time.hpp>

#include "gnss_conversions.hpp"

#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <mirena_common/msg/car.hpp>
#include <mirena_common/msg/heartbeat.hpp>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>

namespace mirena::INS
{
    struct PipelineOutput
    {
        sensor_msgs::msg::NavSatFix fix_msg;
        sensor_msgs::msg::Imu cog_imu;
        sensor_msgs::msg::Imu lidar_imu;
        mirena_common::msg::Car car;
        mirena_common::msg::Heartbeat report;
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

                auto car_msg = mirena_common::msg::Car();
                car_msg.header.frame_id = odom_frame_id;
                car_msg.header.stamp = stamp;
                car_msg.child_frame_id = cog_frame_id;

                // ----------------------- RESPECTIVE FRAMES TO IMU-CENTERED NWU ------------------------------
                auto nwu_orientation = ned_to_nwu * body_orientation * ned_to_body * nwu_to_ned;
                auto nwu_ang_vel = tf2::quatRotate(body_to_nwu, body_ang_vel);
                auto nwu_imu_pos = tf2::quatRotate(ned_to_nwu, ned_imu_pos);
                auto nwu_lin_vel = tf2::quatRotate(body_to_nwu, body_lin_vel);
                auto nwu_lin_accel = tf2::quatRotate(body_to_nwu, body_lin_accel);
                auto nwu_ang_accel = tf2::quatRotate(body_to_nwu, body_ang_accel);

                double roll, pitch, yaw;
                tf2::Matrix3x3(nwu_orientation).getRPY(roll, pitch, yaw);

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

                car_msg.x = nwu_cog_pos[0];
                car_msg.y = nwu_cog_pos[1];
                car_msg.psi = yaw;
                car_msg.u = cog_lin_vel[0];
                car_msg.v = -cog_lin_vel[1];
                car_msg.omega = nwu_ang_vel[2];

                ret->cog_imu = cog_imu_msg;
                ret->car = car_msg;
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

            {
                auto msg = mirena_common::msg::Heartbeat();
                msg.node_name = "mirena_imu";
                msg.estado.status = mirena_common::msg::NodeStatus::WORKING;

                ret->report = msg;
            }

            return ret;
        }

        void set_geodesic_ref(const GeodesicRef &ref)
        {
            geodesic_ref = ref;
        }
    };
}