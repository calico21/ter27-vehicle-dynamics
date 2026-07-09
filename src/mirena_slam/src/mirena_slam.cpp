#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <rclcpp_lifecycle/lifecycle_publisher.hpp>
#include <lifecycle_msgs/msg/state.hpp>
#include <chrono>
#include <mutex> // NUEVO: Para seguridad en hilos (Thread-safety)

// Message filters for time-synchronization
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

// Transforms
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2/LinearMath/Quaternion.h>

// JSON library for string packaging
#include <nlohmann/json.hpp>

// Custom Interfaces
#include "mirena_common/msg/car.hpp"
#include "mirena_common/msg/entity_list.hpp"
#include "mirena_common/msg/heartbeat.hpp"
#include "slam.hpp"

// Topics and Frames
#define CAR_STATE_TOPIC "/state/car"
#define LANDMARK_TOPIC "/perception/cones"
#define MAP_TOPIC "slam/map"

#define FIXED_FRAME "map"
#define MOVING_FRAME "odom"

// Parameters
#define MAP_PUBLISH_PERIOD 100ms 
#define REPORT_PERIOD 100ms
#define DIST_THRESH 0.5    // Meters moved before creating a keyframe
#define ANGLE_THRESH 0.3   // Radians turned before creating a keyframe

using namespace std::chrono_literals;
using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

class MirenaMapping : public rclcpp_lifecycle::LifecycleNode
{
    // Synchronize the Car state with the detected Landmarks
    using SyncPolicy = message_filters::sync_policies::ApproximateTime<mirena_common::msg::Car, mirena_common::msg::EntityList>;

    // Callback Groups for thread safety
    rclcpp::CallbackGroup::SharedPtr _slam_cb_group;
    rclcpp::CallbackGroup::SharedPtr _report_cb_group;

    // Subscribers & Synchronizer
    std::shared_ptr<message_filters::Subscriber<mirena_common::msg::Car, rclcpp_lifecycle::LifecycleNode>> _sync_car_sub;
    std::shared_ptr<message_filters::Subscriber<mirena_common::msg::EntityList, rclcpp_lifecycle::LifecycleNode>> _sync_landmark_sub;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> _sync;

    // Lifecycle Publishers
    rclcpp_lifecycle::LifecyclePublisher<mirena_common::msg::EntityList>::SharedPtr _map_pub;
    rclcpp_lifecycle::LifecyclePublisher<mirena_common::msg::Heartbeat>::SharedPtr _report_pub;

    // Transform broadcaster (map -> odom)
    std::unique_ptr<tf2_ros::TransformBroadcaster> _tf_broadcaster;

    // Timers
    rclcpp::TimerBase::SharedPtr _map_publish_timer;
    rclcpp::TimerBase::SharedPtr _report_timer;

    // Report State & Mutex (NUEVO: Protege las variables que comparten ambos hilos)
    std::mutex _report_mutex;
    rclcpp::Time _last_update_timestamp;
    uint16_t _last_update_total_landmark_count;
    double _last_update_ms_elapsed;
    double _current_car_x = 0.0;
    double _current_car_y = 0.0;
    double _current_car_psi = 0.0;

    // SLAM Engine
    std::unique_ptr<mirena::Slam> _slam;
    gtsam::Pose2 _last_keyframe_raw_odom;
    bool _is_initialized = false;

public:
    MirenaMapping() : rclcpp_lifecycle::LifecycleNode("mirena_mapping") {
        this->_last_update_timestamp = this->now();
        this->_last_update_total_landmark_count = 0;
    }

    CallbackReturn on_configure(const rclcpp_lifecycle::State &) override
    {
        RCLCPP_INFO(this->get_logger(), "Configuring Mirena Mapping");

        _slam_cb_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        _report_cb_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

        auto slam_sub_options = rclcpp::SubscriptionOptions();
        slam_sub_options.callback_group = _slam_cb_group;

        // Publishers & TF
        _map_pub = this->create_publisher<mirena_common::msg::EntityList>(MAP_TOPIC, 10);
        _report_pub = this->create_publisher<mirena_common::msg::Heartbeat>("health", 10);
        _tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        // Filtered Subscribers
        _sync_car_sub = std::make_shared<message_filters::Subscriber<mirena_common::msg::Car, rclcpp_lifecycle::LifecycleNode>>(
            this, CAR_STATE_TOPIC, rmw_qos_profile_default, slam_sub_options);

        _sync_landmark_sub = std::make_shared<message_filters::Subscriber<mirena_common::msg::EntityList, rclcpp_lifecycle::LifecycleNode>>(
            this, LANDMARK_TOPIC, rmw_qos_profile_default, slam_sub_options);

        _sync = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
            SyncPolicy(100), *_sync_car_sub, *_sync_landmark_sub);
        _sync->registerCallback(std::bind(&MirenaMapping::synchronized_callback, this, std::placeholders::_1, std::placeholders::_2));

        // Low-frequency Map and TF update timer
        _map_publish_timer = this->create_wall_timer(
            MAP_PUBLISH_PERIOD, std::bind(&MirenaMapping::map_publish_timer_callback, this), _slam_cb_group);

        _report_timer = this->create_wall_timer(
            REPORT_PERIOD, std::bind(&MirenaMapping::report_callback, this), _report_cb_group);

        _is_initialized = false;
        return CallbackReturn::SUCCESS;
    }

    CallbackReturn on_activate(const rclcpp_lifecycle::State &) override {
        RCLCPP_INFO(this->get_logger(), "SLAM Node Active");
        _map_pub->on_activate();
        _report_pub->on_activate();
        return CallbackReturn::SUCCESS;
    }

    CallbackReturn on_deactivate(const rclcpp_lifecycle::State &) override {
        _map_pub->on_deactivate();
        _report_pub->on_deactivate();
        return CallbackReturn::SUCCESS;
    }

    CallbackReturn on_cleanup(const rclcpp_lifecycle::State &) override {
        _map_publish_timer.reset();
        _report_timer.reset();
        _sync.reset();
        _sync_car_sub.reset();
        _sync_landmark_sub.reset();
        _map_pub.reset();
        _report_pub.reset();
        _slam.reset();
        _is_initialized = false;
        return CallbackReturn::SUCCESS;
    }

private:
    void synchronized_callback(const mirena_common::msg::Car::ConstSharedPtr &car_msg,
                               const mirena_common::msg::EntityList::ConstSharedPtr &landmark_msg)
    {
        if (this->get_current_state().id() != lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE) return;

        // NUEVO: Guardamos la última posición recibida del coche de forma segura
        {
            std::lock_guard<std::mutex> lock(_report_mutex);
            _current_car_x = car_msg->x;
            _current_car_y = car_msg->y;
            _current_car_psi = car_msg->psi;
        }

        gtsam::Pose2 current_raw(car_msg->x, car_msg->y, car_msg->psi);

        if (!_is_initialized) {
            _slam = std::make_unique<mirena::Slam>(current_raw);
            _last_keyframe_raw_odom = current_raw;
            _is_initialized = true;
            RCLCPP_INFO(this->get_logger(), "SLAM Engine Initialized at (%.2f, %.2f)", car_msg->x, car_msg->y);
            return;
        }

        // 2. Check Displacement
        gtsam::Pose2 delta = _last_keyframe_raw_odom.between(current_raw);
        if (delta.translation().norm() >= DIST_THRESH || std::abs(delta.theta()) >= ANGLE_THRESH || _slam->get_map().empty())
        {
            gtsam::Matrix33 odom_cov = gtsam::Matrix33::Zero();
            odom_cov(0,0) = std::max(car_msg->covariance[0], 0.5);
            odom_cov(1,1) = std::max(car_msg->covariance[7], 0.5);
            odom_cov(2,2) = std::max(car_msg->covariance[14], 0.05);
            auto dynamic_noise = gtsam::noiseModel::Gaussian::Covariance(odom_cov);

            std::vector<mirena::Landmark> obs;
            for (const auto &e : landmark_msg->entities) {
                obs.push_back({e.position.x, e.position.y, e.type, e.confidence, {}});
            }

            // 5. Add to Graph
            _slam->add_keyframe(current_raw, obs, dynamic_noise);
            _last_keyframe_raw_odom = current_raw;
            
            {
                std::lock_guard<std::mutex> lock(_report_mutex);
                _last_update_timestamp = this->now();
                _last_update_total_landmark_count = this->_slam->get_map().size();
                _last_update_ms_elapsed = this->_slam->get_last_update_elapsed_ms();
            }
        }
    }

    void map_publish_timer_callback()
    {
        if (this->get_current_state().id() != lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE || !_is_initialized)
            return;

        auto now = this->now();

        // 1. Update and Broadcast map -> odom TF
        gtsam::Pose2 map_to_odom = _slam->get_correction();
        
        geometry_msgs::msg::TransformStamped t;
        t.header.stamp = now + MAP_PUBLISH_PERIOD; 
        t.header.frame_id = FIXED_FRAME;
        t.child_frame_id = MOVING_FRAME;
        
        t.transform.translation.x = map_to_odom.x();
        t.transform.translation.y = map_to_odom.y();
        t.transform.translation.z = 0.0;

        tf2::Quaternion q;
        q.setRPY(0, 0, map_to_odom.theta());
        t.transform.rotation.x = q.x();
        t.transform.rotation.y = q.y();
        t.transform.rotation.z = q.z();
        t.transform.rotation.w = q.w();

        _tf_broadcaster->sendTransform(t);

        auto map_msg = std::make_shared<mirena_common::msg::EntityList>();
        auto landmarks = _slam->get_map();
        for (const auto &l : landmarks) {
            mirena_common::msg::Entity e;
            e.position.x = l.x; 
            e.position.y = l.y; 
            e.type = l.color;
            map_msg->entities.push_back(e);
        }
        map_msg->header.stamp = now;
        map_msg->header.frame_id = FIXED_FRAME;
        _map_pub->publish(*map_msg);
    }

    void report_callback() {
        if (!_report_pub->is_activated()) { return; }
        
        mirena_common::msg::Heartbeat msg;
        msg.timestamp = this->now();
        msg.node_name = this->get_name();

        nlohmann::json report_json;
        report_json["status_state"] = "WORKING";

        {
            std::lock_guard<std::mutex> lock(_report_mutex);
            report_json["last_update_ms_elapsed"] = this->_last_update_ms_elapsed;
            report_json["last_update_total_landmark_count"] = this->_last_update_total_landmark_count;
            report_json["last_run_cones_seen"] = this->_last_update_total_landmark_count;
            
            // Añadimos la posición actual del coche al JSON
            report_json["car_pos_x"] = this->_current_car_x;
            report_json["car_pos_y"] = this->_current_car_y;
            report_json["car_pos_yaw"] = this->_current_car_psi;
        }

        msg.report = report_json.dump();
        _report_pub->publish(msg);
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MirenaMapping>();

    bool autostart_flag = false;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "-a") {
            autostart_flag = true;
            break;
        }
    }

    if(autostart_flag){
        RCLCPP_INFO(node->get_logger(), "Autoactivating");
        node->configure();
        node->activate();
    } else {
        RCLCPP_INFO(node->get_logger(), "Awaiting activation. Use argument '-a' when running for autoactivation");
    }

    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node->get_node_base_interface());
    executor.spin();
    rclcpp::shutdown();
    return 0;
}