#pragma once

#include <vectornav/Interface/Sensor.hpp>
#include <vectornav/Interface/Registers.hpp>

#include <rclcpp/time.hpp>

#include "gnss_conversions.hpp"

#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>

#include <optional>
#include <stdexcept>
#include <string>

namespace mirena::IMU
{
    /// Where the odom frame's origin sits on the earth. Latched once, from the
    /// first INS solution the driver decides to trust.
    struct GeodesicRef
    {
        double latitude;
        double longitude;
        double altitude;
    };

    /// Frames the driver publishes into.
    struct Frames
    {
        std::string odom;
        std::string cog;
        std::string lidar;
        std::string gnss;
    };

    /// One measurement, converted.
    ///
    /// The IMU messages are always produced; the INS solution only once the
    /// onboard filter has one, which is why those two are optional. Consumers
    /// (mirena_state) must tolerate their absence -- there is no odometry before
    /// GPS lock, and saying nothing is better than saying zero.
    struct Output
    {
        sensor_msgs::msg::Imu cog_imu;
        sensor_msgs::msg::Imu lidar_imu;
        std::optional<nav_msgs::msg::Odometry> odom;
        std::optional<sensor_msgs::msg::NavSatFix> fix;
    };

    namespace detail
    {
        /// A 180-degree turn about X.
        ///
        /// It appears twice for two different reasons that happen to be the same
        /// rotation: NED -> NWU between world frames, and the sensor's own
        /// x-forward/y-right/z-down body axes -> the ROS x-forward/y-left/z-up
        /// convention every frame in this workspace uses.
        inline const tf2::Quaternion &flip_x()
        {
            static const tf2::Quaternion q(1, 0, 0, 0);
            return q;
        }

        /// A vector the sensor measured in its own body axes, expressed in the
        /// frame that `frame_from_imu` targets.
        ///
        /// The order matters and is the whole reason this is a named function.
        /// The axis flip has to come first, because the mounting rotation that
        /// TF hands back is defined between two ROS frames and expects a ROS
        /// vector. Applying the mounting rotation to raw sensor axes and
        /// flipping afterwards lands the mounting pitch on the wrong side --
        /// with the 10-degree mainhoop pitch in transforms.yaml that is a
        /// 20-degree error that mixes forward with vertical.
        inline tf2::Vector3 to_frame(const tf2::Transform &frame_from_imu,
                                     const tf2::Vector3 &sensor_vector)
        {
            return tf2::quatRotate(frame_from_imu.getRotation(),
                                   tf2::quatRotate(flip_x(), sensor_vector));
        }

        /// The sensor's NED attitude, re-expressed as the orientation of the
        /// target frame in the ROS world frame:
        ///
        ///   world <- frame = (NWU<-NED)(NED<-sensor)(sensor<-imu)(imu<-frame)
        inline tf2::Quaternion orientation_of(const tf2::Transform &frame_from_imu,
                                              const tf2::Quaternion &ned_attitude)
        {
            return flip_x() * ned_attitude * flip_x() *
                   frame_from_imu.getRotation().inverse();
        }

        /// Rigid-body transfer of a velocity from the IMU to the origin of the
        /// frame, both already in that frame's axes.
        ///
        /// `frame_from_imu.getOrigin()` is where the IMU sits in the target
        /// frame, so the arm pointing from the IMU to that origin is its
        /// negation -- which is where the minus signs come from. Angular
        /// acceleration is neglected: estimating it means differentiating the
        /// gyro, and the term it feeds is small next to the centripetal one.
        inline tf2::Vector3 transfer_velocity(const tf2::Transform &frame_from_imu,
                                              const tf2::Vector3 &velocity,
                                              const tf2::Vector3 &angular_rate)
        {
            return velocity - angular_rate.cross(frame_from_imu.getOrigin());
        }

        inline tf2::Vector3 transfer_acceleration(const tf2::Transform &frame_from_imu,
                                                  const tf2::Vector3 &acceleration,
                                                  const tf2::Vector3 &angular_rate)
        {
            const tf2::Vector3 arm = frame_from_imu.getOrigin();
            return acceleration - angular_rate.cross(angular_rate.cross(arm));
        }

        inline void set(geometry_msgs::msg::Vector3 &out, const tf2::Vector3 &v)
        {
            out.x = v.x();
            out.y = v.y();
            out.z = v.z();
        }

        inline void set(geometry_msgs::msg::Quaternion &out, const tf2::Quaternion &q)
        {
            out.x = q.x();
            out.y = q.y();
            out.z = q.z();
            out.w = q.w();
        }
    } // namespace detail

    /// Turns one VectorNav measurement into the messages the rest of the stack
    /// consumes.
    ///
    /// One class rather than the two it replaced. The AHRS and INS paths shared
    /// every line of frame handling and had drifted apart in three of them, each
    /// difference silently changing what a downstream node was told; merging
    /// them makes that class of bug impossible rather than merely absent.
    ///
    /// NOTE on `linear_acceleration`: the Imu messages carry gravity-free
    /// acceleration, which sensor_msgs/Imu does *not* specify -- that field is
    /// defined as the raw accelerometer reading. This is deliberate and load
    /// bearing: mirena_state's EKF mechanises from gravity-free body
    /// acceleration and would integrate a 9.81 m/s^2 bias if this were changed.
    /// Anything else consuming these topics has to know.
    class Pipeline
    {
        std::optional<GeodesicRef> geodesic_ref_;

    public:
        /// @param ins_active the onboard INS has a usable solution
        /// @param cog_from_imu   TF car/cog <- car/imu
        /// @param lidar_from_imu TF car/lidar <- car/imu
        Output process(const Frames &frames,
                       const rclcpp::Time &stamp,
                       const VN::Sensor::CompositeDataQueueReturn &msg,
                       const tf2::Transform &cog_from_imu,
                       const tf2::Transform &lidar_from_imu,
                       bool ins_active)
        {
            using namespace detail;

            if (!msg->attitude.quaternion.has_value() ||
                !msg->imu.angularRate.has_value())
            {
                throw std::runtime_error("measurement has no attitude or angular rate");
            }

            const auto q = msg->attitude.quaternion.value();
            const tf2::Quaternion ned_attitude(q.vector[0], q.vector[1], q.vector[2], q.scalar);
            const auto rate = msg->imu.angularRate.value();
            const tf2::Vector3 sensor_ang_vel(rate[0], rate[1], rate[2]);

            // Linear acceleration, gravity removed, still in sensor axes.
            //
            // With the INS running the onboard filter has already done it and
            // does it better. Without one, gravity is subtracted using the
            // attitude: the accelerometer measures specific force, so a level
            // stationary sensor reads -g and adding g back gives zero.
            tf2::Vector3 sensor_lin_accel;
            if (ins_active && msg->attitude.linBodyAcc.has_value())
            {
                const auto a = msg->attitude.linBodyAcc.value();
                sensor_lin_accel = tf2::Vector3(a[0], a[1], a[2]);
            }
            else if (msg->imu.accel.has_value())
            {
                constexpr double kGravity = 9.80665;
                const auto a = msg->imu.accel.value();
                sensor_lin_accel = tf2::Vector3(a[0], a[1], a[2]) +
                                   tf2::quatRotate(ned_attitude.inverse(),
                                                   tf2::Vector3(0, 0, kGravity));
            }
            else
            {
                throw std::runtime_error("measurement has no acceleration");
            }

            Output out;
            out.cog_imu = imu_message(frames.cog, stamp, cog_from_imu, ned_attitude,
                                      sensor_ang_vel, sensor_lin_accel);
            out.lidar_imu = imu_message(frames.lidar, stamp, lidar_from_imu, ned_attitude,
                                        sensor_ang_vel, sensor_lin_accel);

            if (!ins_active)
            {
                return out;
            }
            if (!msg->ins.posLla.has_value() || !msg->ins.velBody.has_value())
            {
                throw std::runtime_error("INS is active but reported no position or velocity");
            }

            out.odom = odom_message(frames, stamp, msg, cog_from_imu, ned_attitude,
                                    sensor_ang_vel);
            out.fix = fix_message(frames.gnss, stamp, msg);
            return out;
        }

    private:
        static sensor_msgs::msg::Imu imu_message(const std::string &frame_id,
                                                 const rclcpp::Time &stamp,
                                                 const tf2::Transform &frame_from_imu,
                                                 const tf2::Quaternion &ned_attitude,
                                                 const tf2::Vector3 &sensor_ang_vel,
                                                 const tf2::Vector3 &sensor_lin_accel)
        {
            using namespace detail;

            const tf2::Vector3 ang_vel = to_frame(frame_from_imu, sensor_ang_vel);
            const tf2::Vector3 lin_accel = transfer_acceleration(
                frame_from_imu, to_frame(frame_from_imu, sensor_lin_accel), ang_vel);

            sensor_msgs::msg::Imu imu;
            imu.header.frame_id = frame_id;
            imu.header.stamp = stamp;
            set(imu.orientation, orientation_of(frame_from_imu, ned_attitude));
            set(imu.angular_velocity, ang_vel);
            set(imu.linear_acceleration, lin_accel);
            return imu;
        }

        nav_msgs::msg::Odometry odom_message(const Frames &frames,
                                             const rclcpp::Time &stamp,
                                             const VN::Sensor::CompositeDataQueueReturn &msg,
                                             const tf2::Transform &cog_from_imu,
                                             const tf2::Quaternion &ned_attitude,
                                             const tf2::Vector3 &sensor_ang_vel)
        {
            using namespace detail;

            const auto lla = msg->ins.posLla.value();
            const auto vel = msg->ins.velBody.value();

            // The odom frame is pinned to wherever the car was when the INS
            // first had a solution worth trusting. Latched here rather than by
            // the caller so there is exactly one place it can happen.
            if (!geodesic_ref_.has_value())
            {
                geodesic_ref_ = GeodesicRef{lla.lat, lla.lon, lla.alt};
            }

            double north, east, down;
            GNSSConversion::latLonAltToNED(lla.lat, lla.lon, lla.alt,
                                           geodesic_ref_->latitude,
                                           geodesic_ref_->longitude,
                                           geodesic_ref_->altitude,
                                           north, east, down);

            const tf2::Quaternion cog_orientation = orientation_of(cog_from_imu, ned_attitude);
            const tf2::Vector3 imu_position = tf2::quatRotate(flip_x(), tf2::Vector3(north, east, down));

            // The IMU is not the car. Shift the reported position along the arm
            // from the IMU to the cog, rotated into the world by the car's own
            // attitude -- the arm is a body-frame quantity and adding it to a
            // world-frame position unrotated is a heading-dependent error the
            // size of the arm, which here is about 0.4 m horizontally.
            const tf2::Vector3 cog_position =
                imu_position - tf2::quatRotate(cog_orientation, cog_from_imu.getOrigin());

            // Velocity and rate stay in the body frame, which is what
            // nav_msgs/Odometry asks for in twist and what the EKF consumes as
            // u, v and omega.
            const tf2::Vector3 cog_ang_vel = to_frame(cog_from_imu, sensor_ang_vel);
            const tf2::Vector3 cog_lin_vel = transfer_velocity(
                cog_from_imu, to_frame(cog_from_imu, tf2::Vector3(vel[0], vel[1], vel[2])),
                cog_ang_vel);

            nav_msgs::msg::Odometry odom;
            odom.header.frame_id = frames.odom;
            odom.header.stamp = stamp;
            odom.child_frame_id = frames.cog;

            odom.pose.pose.position.x = cog_position.x();
            odom.pose.pose.position.y = cog_position.y();
            odom.pose.pose.position.z = cog_position.z();
            set(odom.pose.pose.orientation, cog_orientation);
            set(odom.twist.twist.linear, cog_lin_vel);
            set(odom.twist.twist.angular, cog_ang_vel);

            // Covariance diagonals from the onboard filter's 1-sigma
            // uncertainties, with conservative fallbacks when not streamed.
            // mirena_state reads [0], [7] and [35] of each and floors them, so
            // the ones it does not read still have to be honest rather than
            // zero -- a zero variance claims perfect knowledge.
            const double pos_var = variance_of(msg->ins.posU, 1.0);
            const double vel_var = variance_of(msg->ins.velU, 0.5);
            constexpr double kGyroVar = 1e-4; // not streamed by the filter

            double yaw_var = 0.05, pitch_var = 0.05, roll_var = 0.05;
            if (msg->attitude.yprU.has_value())
            {
                // Reported as (yaw, pitch, roll), in degrees.
                const auto ypr = msg->attitude.yprU.value();
                const double to_rad = M_PI / 180.0;
                yaw_var = (ypr[0] * to_rad) * (ypr[0] * to_rad);
                pitch_var = (ypr[1] * to_rad) * (ypr[1] * to_rad);
                roll_var = (ypr[2] * to_rad) * (ypr[2] * to_rad);
            }

            for (int i = 0; i < 3; i++)
            {
                odom.pose.covariance[i * 7] = pos_var;
                odom.twist.covariance[i * 7] = vel_var;
                odom.twist.covariance[(i + 3) * 7] = kGyroVar;
            }
            odom.pose.covariance[21] = roll_var;
            odom.pose.covariance[28] = pitch_var;
            odom.pose.covariance[35] = yaw_var;

            return odom;
        }

        static sensor_msgs::msg::NavSatFix fix_message(const std::string &frame_id,
                                                       const rclcpp::Time &stamp,
                                                       const VN::Sensor::CompositeDataQueueReturn &msg)
        {
            const auto lla = msg->ins.posLla.value();

            sensor_msgs::msg::NavSatFix fix;
            fix.header.frame_id = frame_id;
            fix.header.stamp = stamp;
            fix.status.status = sensor_msgs::msg::NavSatStatus::STATUS_FIX;
            fix.status.service = sensor_msgs::msg::NavSatStatus::SERVICE_GPS;
            fix.latitude = lla.lat;
            fix.longitude = lla.lon;
            fix.altitude = lla.alt;

            // The filter reports a single position sigma, so the best that can
            // honestly be said is an isotropic approximation -- but that is a
            // great deal more than the UNKNOWN this used to publish while the
            // number sat unused two functions away.
            if (msg->ins.posU.has_value())
            {
                const double var = variance_of(msg->ins.posU, 1.0);
                fix.position_covariance[0] = var;
                fix.position_covariance[4] = var;
                fix.position_covariance[8] = var;
                fix.position_covariance_type =
                    sensor_msgs::msg::NavSatFix::COVARIANCE_TYPE_APPROXIMATED;
            }
            return fix;
        }

        static double variance_of(const std::optional<float> &sigma, double fallback)
        {
            if (!sigma.has_value() || !(sigma.value() > 0.0f))
            {
                return fallback;
            }
            return static_cast<double>(sigma.value()) * static_cast<double>(sigma.value());
        }
    };
} // namespace mirena::IMU
