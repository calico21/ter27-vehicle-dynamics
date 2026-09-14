#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/filter.h>
#include <pcl_conversions/pcl_conversions.h>

#include <mutex>
#include <vector>
#include <cmath>
#include <memory>
#include <unordered_map>
#include <tuple>
#include <chrono>
#include <algorithm>

#include "mirena_common/msg/entity_list.hpp"

using namespace std::chrono_literals;

namespace fast_lio
{

inline Eigen::Matrix3d Exp(const Eigen::Vector3d &v)
{
    double norm = v.norm();
    if (norm < 1e-7) return Eigen::Matrix3d::Identity();
    return Eigen::AngleAxisd(norm, v / norm).toRotationMatrix();
}

inline Eigen::Vector3d Log(const Eigen::Matrix3d &R)
{
    Eigen::AngleAxisd aa(R);
    return aa.angle() * aa.axis();
}

inline Eigen::Matrix3d Skew(const Eigen::Vector3d &v)
{
    Eigen::Matrix3d m;
    m <<     0, -v.z(),  v.y(),
         v.z(),      0, -v.x(),
        -v.y(),  v.x(),      0;
    return m;
}

inline pcl::PointCloud<pcl::PointXYZ>::Ptr downsampleVoxel(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &input_cloud, float leaf_size)
{
    auto output_cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    if (!input_cloud || input_cloud->empty()) return output_cloud;

    struct VoxelHash {
        std::size_t operator()(const std::tuple<int, int, int> &k) const {
            auto [x, y, z] = k;
            return (std::hash<int>()(x) ^ (std::hash<int>()(y) << 1)) ^ (std::hash<int>()(z) << 2);
        }
    };

    std::unordered_map<std::tuple<int, int, int>, pcl::PointXYZ, VoxelHash> grid;
    float inv_leaf = 1.0f / leaf_size;

    for (const auto &pt : input_cloud->points) {
        if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) continue;
        
        auto key = std::make_tuple(
            static_cast<int>(std::floor(pt.x * inv_leaf)),
            static_cast<int>(std::floor(pt.y * inv_leaf)),
            static_cast<int>(std::floor(pt.z * inv_leaf))
        );
        grid.emplace(key, pt);
    }

    output_cloud->points.reserve(grid.size());
    for (const auto &[key, pt] : grid) {
        output_cloud->points.push_back(pt);
    }
    output_cloud->width = static_cast<uint32_t>(output_cloud->points.size());
    output_cloud->height = 1;
    output_cloud->is_dense = true;

    return output_cloud;
}

struct alignas(16) State
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Vector3d pos = Eigen::Vector3d::Zero();
    Eigen::Matrix3d rot = Eigen::Matrix3d::Identity();
    Eigen::Vector3d vel = Eigen::Vector3d::Zero();
    Eigen::Vector3d bg  = Eigen::Vector3d::Zero();
    Eigen::Vector3d ba  = Eigen::Vector3d::Zero();
    Eigen::Vector3d grav = Eigen::Vector3d(0.0, 0.0, -9.81);

    // Error-state manifold boxminus operator: self (-) prior
    Eigen::Matrix<double, 15, 1> boxMinus(const State &prior) const {
        Eigen::Matrix<double, 15, 1> dx;
        dx.segment<3>(0) = pos - prior.pos;
        dx.segment<3>(3) = Log(prior.rot.transpose() * rot);
        dx.segment<3>(6) = vel - prior.vel;
        dx.segment<3>(9) = bg - prior.bg;
        dx.segment<3>(12) = ba - prior.ba;
        return dx;
    }

    // Error-state manifold boxplus operator: self (+) dx
    void boxPlus(const Eigen::Matrix<double, 15, 1> &dx) {
        pos += dx.segment<3>(0);
        rot = rot * Exp(dx.segment<3>(3));
        vel += dx.segment<3>(6);
        bg  += dx.segment<3>(9);
        ba  += dx.segment<3>(12);
    }
};

class FastLioNode : public rclcpp::Node
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    FastLioNode() : Node("fast_lio_node")
    {
        auto sensor_qos = rclcpp::QoS(rclcpp::KeepLast(10)).best_effort();

        // Subscribing to raw IMU measurements (angular velocity & acceleration)
        imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
            "/sensors/imu/data", sensor_qos,
            std::bind(&FastLioNode::imuCallback, this, std::placeholders::_1));

        cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
            "/sensors/lidar/points", sensor_qos,
            std::bind(&FastLioNode::cloudCallback, this, std::placeholders::_1));

        perception_sub_ = create_subscription<mirena_common::msg::EntityList>(
            "/perception/cones", 10,
            std::bind(&FastLioNode::perceptionCallback, this, std::placeholders::_1));

        odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("/state/fast_lio/odometry", 10);
        
        auto map_qos = rclcpp::QoS(rclcpp::KeepLast(1)).transient_local().reliable();
        map_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("/state/fast_lio/global_map", map_qos);
        global_cones_pub_ = create_publisher<mirena_common::msg::EntityList>("/state/fast_lio/global_cones", map_qos);

        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        odom_timer_ = create_wall_timer(20ms, std::bind(&FastLioNode::timerCallback, this));

        // State covariance and process noise initialization
        P_ = Eigen::Matrix<double, 15, 15>::Identity() * 1e-2;
        Q_ = Eigen::Matrix<double, 15, 15>::Zero();
        Q_.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * 1e-4; // Position noise
        Q_.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * 1e-4; // Gyro noise
        Q_.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity() * 1e-3; // Accel noise
        Q_.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity() * 1e-6; // Gyro bias drift
        Q_.block<3, 3>(12, 12) = Eigen::Matrix3d::Identity() * 1e-6; // Accel bias drift

        global_map_ = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        kdtree_map_ = pcl::make_shared<pcl::KdTreeFLANN<pcl::PointXYZ>>();

        RCLCPP_INFO(get_logger(), "FAST-LIO Perception Node initialized.");
    }

    ~FastLioNode() override
    {
        tf_broadcaster_.reset();
    }

private:
    void timerCallback()
    {
        publishOdometry(this->now());
    }

    void imuCallback(const sensor_msgs::msg::Imu::ConstSharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        double stamp = rclcpp::Time(msg->header.stamp).seconds();

        if (stamp <= 0.0) return;

        if (last_imu_time_ < 0.0) {
            last_imu_time_ = stamp;
            return;
        }

        double dt = stamp - last_imu_time_;
        if (dt <= 0.0 || dt > 1.0) {
            last_imu_time_ = stamp;
            return;
        }

        Eigen::Vector3d gyro(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);
        Eigen::Vector3d accel(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);

        Eigen::Vector3d un_gyro = gyro - state_.bg;
        Eigen::Vector3d un_accel = accel - state_.ba;

        // Discrete IMU integration
        Eigen::Matrix3d R_prev = state_.rot;
        Eigen::Matrix3d R_curr = R_prev * Exp(un_gyro * dt);

        Eigen::Vector3d acc_w = R_prev * un_accel + state_.grav;
        state_.pos += state_.vel * dt + 0.5 * acc_w * dt * dt;
        state_.vel += acc_w * dt;
        state_.rot = R_curr;

        // Covariance Propagation F_x
        Eigen::Matrix<double, 15, 15> F_x = Eigen::Matrix<double, 15, 15>::Identity();
        F_x.block<3, 3>(0, 6) = Eigen::Matrix3d::Identity() * dt;
        F_x.block<3, 3>(3, 3) = Exp(-un_gyro * dt);
        F_x.block<3, 3>(3, 9) = -Eigen::Matrix3d::Identity() * dt;
        F_x.block<3, 3>(6, 3) = -R_prev * Skew(un_accel) * dt;
        F_x.block<3, 3>(6, 12) = -R_prev * dt;

        P_ = F_x * P_ * F_x.transpose() + Q_ * dt;

        last_imu_time_ = stamp;
    }

    void cloudCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
    {
        auto raw_cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        pcl::fromROSMsg(*msg, *raw_cloud);

        std::vector<int> dummy_indices;
        pcl::removeNaNFromPointCloud(*raw_cloud, *raw_cloud, dummy_indices);

        if (raw_cloud->empty()) return;

        auto filtered_cloud = downsampleVoxel(raw_cloud, 0.1f);
        if (filtered_cloud->empty()) return;

        State state_prior;
        Eigen::Matrix<double, 15, 15> P_prior;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            state_prior = state_;
            P_prior = P_;
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr active_map;
        pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr active_kdtree;
        {
            std::lock_guard<std::mutex> map_lock(map_mutex_);
            active_map = global_map_;
            active_kdtree = kdtree_map_;
        }

        if (active_map->points.size() < 10) {
            auto init_map = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
            for (const auto &pt : filtered_cloud->points) {
                Eigen::Vector3d p_w = state_prior.rot * Eigen::Vector3d(pt.x, pt.y, pt.z) + state_prior.pos;
                if (p_w.allFinite()) {
                    init_map->push_back(pcl::PointXYZ(static_cast<float>(p_w.x()), 
                                                       static_cast<float>(p_w.y()), 
                                                       static_cast<float>(p_w.z())));
                }
            }

            if (init_map->points.size() >= 10) {
                auto new_tree = pcl::make_shared<pcl::KdTreeFLANN<pcl::PointXYZ>>();
                new_tree->setInputCloud(init_map);

                std::lock_guard<std::mutex> map_lock(map_mutex_);
                global_map_ = init_map;
                kdtree_map_ = new_tree;
            }

            publishMap(msg->header.stamp);
            return;
        }

        State state_iter = state_prior;
        constexpr int kMaxIter = 4;
        constexpr double kMeasNoiseVar = 1e-2;
        constexpr double kInvNoise = 1.0 / kMeasNoiseVar;
        const Eigen::Matrix<double, 15, 15> P_prior_inv = P_prior.inverse();

        Eigen::Matrix<double, 15, 15> Ht_Rinv_H_final = Eigen::Matrix<double, 15, 15>::Zero();

        for (int iter = 0; iter < kMaxIter; ++iter) {
            Eigen::Matrix<double, 15, 15> Ht_Rinv_H = Eigen::Matrix<double, 15, 15>::Zero();
            Eigen::Matrix<double, 15, 1> Ht_Rinv_z = Eigen::Matrix<double, 15, 1>::Zero();
            int valid_matches = 0;

            for (const auto &pt : filtered_cloud->points) {
                Eigen::Vector3d p_l(pt.x, pt.y, pt.z);
                Eigen::Vector3d p_w = state_iter.rot * p_l + state_iter.pos;

                if (!p_w.allFinite()) continue;

                pcl::PointXYZ search_pt(static_cast<float>(p_w.x()), 
                                       static_cast<float>(p_w.y()), 
                                       static_cast<float>(p_w.z()));

                std::vector<int> k_indices(5);
                std::vector<float> k_sqr_distances(5);

                if (active_kdtree->nearestKSearch(search_pt, 5, k_indices, k_sqr_distances) == 5) {
                    if (k_sqr_distances[4] > 1.0f) continue;

                    Eigen::Matrix<double, 5, 3> A;
                    Eigen::Vector3d center = Eigen::Vector3d::Zero();

                    for (int i = 0; i < 5; ++i) {
                        const auto &mp = active_map->points[k_indices[i]];
                        Eigen::Vector3d v(mp.x, mp.y, mp.z);
                        A.row(i) = v;
                        center += v;
                    }
                    center /= 5.0;
                    A.rowwise() -= center.transpose();

                    Eigen::Matrix3d cov = A.transpose() * A;
                    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);

                    if (es.info() != Eigen::Success) continue;

                    Eigen::Vector3d norm_vec = es.eigenvectors().col(0);
                    if (!norm_vec.allFinite()) continue;

                    double res = norm_vec.dot(p_w - center);
                    if (std::abs(res) < 0.2) {
                        Eigen::Matrix<double, 1, 15> H_i = Eigen::Matrix<double, 1, 15>::Zero();
                        H_i.block<1, 3>(0, 0) = norm_vec.transpose();
                        H_i.block<1, 3>(0, 3) = -norm_vec.transpose() * state_iter.rot * Skew(p_l);

                        Ht_Rinv_H += kInvNoise * (H_i.transpose() * H_i);
                        Ht_Rinv_z += kInvNoise * (H_i.transpose() * (-res));
                        valid_matches++;
                    }
                }
            }

            if (valid_matches < 10) break;

            // Prior state offset check
            Eigen::Matrix<double, 15, 1> dx_prior = state_iter.boxMinus(state_prior);

            // Corrected IEKF System update matrix
            Eigen::Matrix<double, 15, 15> A_mat = P_prior_inv + Ht_Rinv_H;
            Eigen::Matrix<double, 15, 1> b_vec = Ht_Rinv_z - P_prior_inv * dx_prior;

            Eigen::Matrix<double, 15, 1> dx = A_mat.ldlt().solve(b_vec);

            if (!dx.allFinite() || dx.norm() > 2.0) break;

            state_iter.boxPlus(dx);
            Ht_Rinv_H_final = Ht_Rinv_H;

            if (dx.norm() < 1e-3) break;
        }

        // Post-measurement state and covariance updates
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (state_iter.pos.allFinite() && state_iter.rot.allFinite()) {
                state_ = state_iter;
                P_ = (P_prior_inv + Ht_Rinv_H_final).inverse();
            }
        }

        // Global map building
        auto combined_map = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        *combined_map = *active_map;

        for (const auto &pt : filtered_cloud->points) {
            Eigen::Vector3d p_w = state_iter.rot * Eigen::Vector3d(pt.x, pt.y, pt.z) + state_iter.pos;
            if (p_w.allFinite()) {
                combined_map->push_back(pcl::PointXYZ(static_cast<float>(p_w.x()), 
                                                       static_cast<float>(p_w.y()), 
                                                       static_cast<float>(p_w.z())));
            }
        }

        auto downsampled_map = downsampleVoxel(combined_map, 0.2f);
        pcl::removeNaNFromPointCloud(*downsampled_map, *downsampled_map, dummy_indices);

        if (!downsampled_map->empty() && downsampled_map->points.size() >= 10) {
            auto new_kdtree = pcl::make_shared<pcl::KdTreeFLANN<pcl::PointXYZ>>();
            new_kdtree->setInputCloud(downsampled_map);

            std::lock_guard<std::mutex> map_lock(map_mutex_);
            global_map_ = downsampled_map;
            kdtree_map_ = new_kdtree;
        }

        publishMap(msg->header.stamp);
    }

    void perceptionCallback(const mirena_common::msg::EntityList::ConstSharedPtr &msg)
    {
        Eigen::Matrix3d current_rot;
        Eigen::Vector3d current_pos;

        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            current_rot = state_.rot;
            current_pos = state_.pos;
        }

        constexpr double kMaxRadiusSq = 0.5 * 0.5; // 0.5m matching radius threshold
        std::lock_guard<std::mutex> lock(cones_mutex_);

        for (const auto &entity : msg->entities) {
            Eigen::Vector3d p_local(entity.position.x, entity.position.y, entity.position.z);
            Eigen::Vector3d p_world = current_rot * p_local + current_pos;

            if (!p_world.allFinite()) continue;

            int best_match_idx = -1;
            double min_dist_sq = kMaxRadiusSq;

            for (size_t i = 0; i < global_cones_map_.entities.size(); ++i) {
                auto &global_entity = global_cones_map_.entities[i];

                // Ensure same class type
                if (global_entity.type != entity.type) continue;

                // 2D Distance matching check
                double dx = global_entity.position.x - p_world.x();
                double dy = global_entity.position.y - p_world.y();
                double dist_sq = dx * dx + dy * dy;

                if (dist_sq < min_dist_sq) {
                    min_dist_sq = dist_sq;
                    best_match_idx = static_cast<int>(i);
                }
            }

            if (best_match_idx >= 0) {
                // Recursive weighted average to stabilize cone centers
                auto &matched = global_cones_map_.entities[best_match_idx];
                matched.confidence += 1.0;
                double n = std::min(static_cast<double>(matched.confidence), 20.0);
                
                matched.position.x = ((n - 1.0) * matched.position.x + p_world.x()) / n;
                matched.position.y = ((n - 1.0) * matched.position.y + p_world.y()) / n;
                matched.position.z = ((n - 1.0) * matched.position.z + p_world.z()) / n;
            } else {
                mirena_common::msg::Entity new_cone = entity;
                new_cone.position.x = p_world.x();
                new_cone.position.y = p_world.y();
                new_cone.position.z = p_world.z();
                new_cone.confidence = 1.0;
                global_cones_map_.entities.push_back(new_cone);
            }
        }

        global_cones_map_.header.stamp = msg->header.stamp;
        global_cones_map_.header.frame_id = "odom";
        global_cones_pub_->publish(global_cones_map_);
    }

    void publishMap(const rclcpp::Time &stamp)
    {
        sensor_msgs::msg::PointCloud2 map_msg;
        {
            std::lock_guard<std::mutex> map_lock(map_mutex_);
            if (global_map_->empty()) return;

            RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000, 
                "Publishing global map with %zu points.", global_map_->size());

            pcl::toROSMsg(*global_map_, map_msg);
        }

        map_msg.header.stamp = stamp;
        map_msg.header.frame_id = "odom";
        map_pub_->publish(map_msg);
    }

    void publishOdometry(const rclcpp::Time &stamp)
    {
        Eigen::Vector3d pos, vel;
        Eigen::Matrix3d rot;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            pos = state_.pos;
            vel = state_.vel;
            rot = state_.rot;
        }

        nav_msgs::msg::Odometry odom;
        odom.header.stamp = stamp;
        odom.header.frame_id = "odom";
        odom.child_frame_id = "car/cog";

        odom.pose.pose.position.x = pos.x();
        odom.pose.pose.position.y = pos.y();
        odom.pose.pose.position.z = pos.z();

        Eigen::Quaterniond q(rot);
        odom.pose.pose.orientation.x = q.x();
        odom.pose.pose.orientation.y = q.y();
        odom.pose.pose.orientation.z = q.z();
        odom.pose.pose.orientation.w = q.w();

        odom.twist.twist.linear.x = vel.x();
        odom.twist.twist.linear.y = vel.y();
        odom.twist.twist.linear.z = vel.z();

        odom_pub_->publish(odom);

        geometry_msgs::msg::TransformStamped tf;
        tf.header.stamp = stamp;
        tf.header.frame_id = "odom";
        tf.child_frame_id = "car/cog";
        tf.transform.translation.x = pos.x();
        tf.transform.translation.y = pos.y();
        tf.transform.translation.z = pos.z();
        tf.transform.rotation = odom.pose.pose.orientation;
        tf_broadcaster_->sendTransform(tf);
    }

    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
    rclcpp::Subscription<mirena_common::msg::EntityList>::SharedPtr perception_sub_;
    
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_pub_;
    rclcpp::Publisher<mirena_common::msg::EntityList>::SharedPtr global_cones_pub_;
    
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    rclcpp::TimerBase::SharedPtr odom_timer_;

    std::mutex state_mutex_;
    std::mutex map_mutex_;
    std::mutex cones_mutex_;
    State state_;
    Eigen::Matrix<double, 15, 15> P_;
    Eigen::Matrix<double, 15, 15> Q_;
    double last_imu_time_ = -1.0;

    pcl::PointCloud<pcl::PointXYZ>::Ptr global_map_;
    pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kdtree_map_;
    mirena_common::msg::EntityList global_cones_map_;
};

} // namespace fast_lio

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<fast_lio::FastLioNode>());
    rclcpp::shutdown();
    return 0;
}