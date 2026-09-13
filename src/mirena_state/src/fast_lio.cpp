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
};

class FastLioNode : public rclcpp::Node
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    FastLioNode() : Node("fast_lio_node")
    {
        auto sensor_qos = rclcpp::QoS(rclcpp::KeepLast(10)).best_effort();

        imu_sub_ = create_subscription<nav_msgs::msg::Odometry>(
            "/sensors/imu/odom", sensor_qos,
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

        // 50 Hz Timer for high-frequency odometry publishing
        odom_timer_ = create_wall_timer(20ms, std::bind(&FastLioNode::timerCallback, this));

        P_ = Eigen::Matrix<double, 15, 15>::Identity() * 1e-3;
        global_map_ = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        kdtree_map_ = pcl::make_shared<pcl::KdTreeFLANN<pcl::PointXYZ>>();

        RCLCPP_INFO(get_logger(), "FAST-LIO Perception-to-Map Node initialized.");
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

    void imuCallback(const nav_msgs::msg::Odometry::ConstSharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        double stamp = rclcpp::Time(msg->header.stamp).seconds();

        if (stamp <= 0.0) return;

        if (last_imu_time_ < 0.0) {
            last_imu_time_ = stamp;
            return;
        }

        double dt = stamp - last_imu_time_;
        if (dt <= 0.0 || dt > 2.0) {
            last_imu_time_ = stamp;
            return;
        }

        Eigen::Vector3d gyro(
            msg->twist.twist.angular.x,
            msg->twist.twist.angular.y,
            msg->twist.twist.angular.z
        );

        Eigen::Vector3d body_vel(
            msg->twist.twist.linear.x,
            msg->twist.twist.linear.y,
            msg->twist.twist.linear.z
        );

        Eigen::Vector3d un_gyro = gyro - state_.bg;
        state_.rot = state_.rot * Exp(un_gyro * dt);

        Eigen::Vector3d world_vel = state_.rot * body_vel;
        state_.pos += world_vel * dt;
        state_.vel = world_vel;

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

        State current_state;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            current_state = state_;
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr active_map;
        pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr active_kdtree;
        {
            std::lock_guard<std::mutex> map_lock(map_mutex_);
            active_map = global_map_;
            active_kdtree = kdtree_map_;
        }

        if (active_map->points.size() < 5) {
            auto init_map = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
            for (const auto &pt : filtered_cloud->points) {
                Eigen::Vector3d p_w = current_state.rot * Eigen::Vector3d(pt.x, pt.y, pt.z) + current_state.pos;
                if (p_w.allFinite()) {
                    init_map->push_back(pcl::PointXYZ(static_cast<float>(p_w.x()), 
                                                       static_cast<float>(p_w.y()), 
                                                       static_cast<float>(p_w.z())));
                }
            }

            if (init_map->points.size() >= 5) {
                auto new_tree = pcl::make_shared<pcl::KdTreeFLANN<pcl::PointXYZ>>();
                new_tree->setInputCloud(init_map);

                std::lock_guard<std::mutex> map_lock(map_mutex_);
                global_map_ = init_map;
                kdtree_map_ = new_tree;
            }

            publishMap(msg->header.stamp);
            return;
        }

        State state_iter = current_state;
        constexpr int kMaxIter = 3;
        constexpr double kMeasNoiseVar = 1e-3;
        constexpr double kInvNoise = 1.0 / kMeasNoiseVar;
        const Eigen::Matrix<double, 15, 15> P_inv = P_.inverse();

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
                        Ht_Rinv_z += kInvNoise * (H_i.transpose() * res);
                        valid_matches++;
                    }
                }
            }

            if (valid_matches < 10) break;

            Eigen::Matrix<double, 15, 15> A_mat = P_inv + Ht_Rinv_H;
            Eigen::Matrix<double, 15, 1> dx = A_mat.ldlt().solve(-Ht_Rinv_z);

            if (!dx.allFinite() || dx.norm() > 5.0) break;

            state_iter.pos += dx.segment<3>(0);
            state_iter.rot  = state_iter.rot * Exp(dx.segment<3>(3));
            state_iter.vel += dx.segment<3>(6);
            state_iter.bg  += dx.segment<3>(9);
            state_iter.ba  += dx.segment<3>(12);

            if (dx.norm() < 1e-3) break;
        }

        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (state_iter.pos.allFinite() && state_iter.rot.allFinite()) {
                state_ = state_iter;
            }
        }

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

        if (!downsampled_map->empty() && downsampled_map->points.size() >= 5) {
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

        constexpr double kDupRadiusSq = 0.3 * 0.3;
        std::lock_guard<std::mutex> lock(cones_mutex_);

        for (const auto &entity : msg->entities) {
            Eigen::Vector3d p_local(entity.position.x, entity.position.y, entity.position.z);
            Eigen::Vector3d p_world = current_rot * p_local + current_pos;

            if (!p_world.allFinite()) continue;

            bool is_duplicate = false;
            for (auto &global_entity : global_cones_map_.entities) {
                double dx = global_entity.position.x - p_world.x();
                double dy = global_entity.position.y - p_world.y();
                double dz = global_entity.position.z - p_world.z();

                if ((dx * dx + dy * dy + dz * dz) <= kDupRadiusSq) {
                    global_entity.position.x = 0.8 * global_entity.position.x + 0.2 * p_world.x();
                    global_entity.position.y = 0.8 * global_entity.position.y + 0.2 * p_world.y();
                    global_entity.position.z = 0.8 * global_entity.position.z + 0.2 * p_world.z();
                    is_duplicate = true;
                    break;
                }
            }

            if (!is_duplicate) {
                mirena_common::msg::Entity new_cone = entity;
                new_cone.position.x = p_world.x();
                new_cone.position.y = p_world.y();
                new_cone.position.z = p_world.z();
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

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr imu_sub_;
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