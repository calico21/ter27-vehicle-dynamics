#include <rclcpp/rclcpp.hpp>
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
#define MAX_SYNC_DELTA_MS 10.0 // Pose/detection pairs further apart are dropped
#define REPORT_STALE_AFTER_S 1.0

using namespace std::chrono_literals;
class MirenaMapping : public rclcpp::Node
{
    // Synchronize the Car state with the detected Landmarks
    using SyncPolicy = message_filters::sync_policies::ApproximateTime<mirena_common::msg::Car, mirena_common::msg::EntityList>;

    // Callback Groups for thread safety
    rclcpp::CallbackGroup::SharedPtr _slam_cb_group;
    rclcpp::CallbackGroup::SharedPtr _report_cb_group;
    // The map->odom broadcast lives on its own group on purpose: an iSAM2
    // update takes hundreds of milliseconds once the graph grows, and sharing
    // a mutually exclusive group with it stalls the transform for that whole
    // time. Downstream that shows up as control failing its lookups with
    // "extrapolation into the future".
    rclcpp::CallbackGroup::SharedPtr _tf_cb_group;

    // Subscribers & Synchronizer
    std::shared_ptr<message_filters::Subscriber<mirena_common::msg::Car, rclcpp::Node>> _sync_car_sub;
    std::shared_ptr<message_filters::Subscriber<mirena_common::msg::EntityList, rclcpp::Node>> _sync_landmark_sub;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> _sync;

    // Publishers
    rclcpp::Publisher<mirena_common::msg::EntityList>::SharedPtr _map_pub;
    rclcpp::Publisher<mirena_common::msg::Heartbeat>::SharedPtr _report_pub;

    // Transform broadcaster (map -> odom)
    std::unique_ptr<tf2_ros::TransformBroadcaster> _tf_broadcaster;

    // Timers
    rclcpp::TimerBase::SharedPtr _map_publish_timer;
    rclcpp::TimerBase::SharedPtr _report_timer;

    // Report State & Mutex (NUEVO: Protege las variables que comparten ambos hilos)
    std::mutex _report_mutex;
    rclcpp::Time _last_update_timestamp;
    uint16_t _last_update_total_landmark_count = 0;
    double _last_update_ms_elapsed = 0.0; // read by the report timer before the first keyframe
    double _current_car_x = 0.0;
    double _current_car_y = 0.0;
    double _current_car_psi = 0.0;
    bool _has_keyframe = false;
    int _last_matched = 0;
    int _last_new = 0;
    uint64_t _dropped_sync_pairs = 0;

    // Result of the last optimisation, handed to the publisher thread.
    // Everything the timer needs is copied out here so it never touches _slam
    // while the optimiser is running -- note get_map() hands back a reference to
    // a vector that update_isam() clears and refills.
    std::mutex _snapshot_mutex;
    gtsam::Pose2 _map_to_odom_snapshot;
    std::shared_ptr<const mirena_common::msg::EntityList> _map_snapshot;
    bool _has_snapshot = false;

    // SLAM Engine.
    //
    // Guarded because the sync callback runs in _slam_cb_group under a
    // MultiThreadedExecutor while the map and report timers run in their own.
    std::mutex _slam_mutex;
    std::unique_ptr<mirena::Slam> _slam;
    gtsam::Pose2 _last_keyframe_raw_odom;
    bool _is_initialized = false;

    // Absolute pose covariance at the last keyframe, used to derive the
    // *relative* covariance of the next odometry delta.
    double _last_kf_cov[3] = {0.0, 0.0, 0.0};

public:
    MirenaMapping() : rclcpp::Node("mirena_mapping") {
        this->_last_update_timestamp = this->now();
        this->_last_update_total_landmark_count = 0;

        RCLCPP_INFO(this->get_logger(), "Starting Mirena Mapping");

        _slam_cb_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        _report_cb_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        _tf_cb_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

        auto slam_sub_options = rclcpp::SubscriptionOptions();
        slam_sub_options.callback_group = _slam_cb_group;

        // Publishers & TF
        _map_pub = this->create_publisher<mirena_common::msg::EntityList>(MAP_TOPIC, 10);
        _report_pub = this->create_publisher<mirena_common::msg::Heartbeat>("health", 10);
        _tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        // Filtered Subscribers
        _sync_car_sub = std::make_shared<message_filters::Subscriber<mirena_common::msg::Car, rclcpp::Node>>(
            this, CAR_STATE_TOPIC, rmw_qos_profile_default, slam_sub_options);

        _sync_landmark_sub = std::make_shared<message_filters::Subscriber<mirena_common::msg::EntityList, rclcpp::Node>>(
            this, LANDMARK_TOPIC, rmw_qos_profile_default, slam_sub_options);

        _sync = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
            SyncPolicy(100), *_sync_car_sub, *_sync_landmark_sub);
        // Refuse pairs the policy would otherwise match across an arbitrary gap.
        // At 15 m/s a 100 ms mismatch is 1.5 m of pose error baked into every
        // landmark of that keyframe.
        _sync->setMaxIntervalDuration(rclcpp::Duration::from_seconds(MAX_SYNC_DELTA_MS / 1000.0));
        _sync->registerCallback(std::bind(&MirenaMapping::synchronized_callback, this, std::placeholders::_1, std::placeholders::_2));

        // Low-frequency Map and TF update timer
        _map_publish_timer = this->create_wall_timer(
            MAP_PUBLISH_PERIOD, std::bind(&MirenaMapping::map_publish_timer_callback, this), _tf_cb_group);

        _report_timer = this->create_wall_timer(
            REPORT_PERIOD, std::bind(&MirenaMapping::report_callback, this), _report_cb_group);

        _is_initialized = false;
    }

private:
    void synchronized_callback(const mirena_common::msg::Car::ConstSharedPtr &car_msg,
                               const mirena_common::msg::EntityList::ConstSharedPtr &landmark_msg)
    {

        rclcpp::Time car_stamp(car_msg->header.stamp);
        rclcpp::Time landmark_stamp(landmark_msg->header.stamp);
        
        double delta_ms = std::abs((car_stamp - landmark_stamp).seconds() * 1000.0);

        // Belt and braces on top of setMaxIntervalDuration: a pose and a cone
        // frame this far apart describe different moments, and pretending
        // otherwise writes the difference straight into the map.
        if (delta_ms > MAX_SYNC_DELTA_MS)
        {
            {
                std::lock_guard<std::mutex> lock(_report_mutex);
                ++_dropped_sync_pairs;
            }
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                 "Dropping sync pair, delta %.1f ms exceeds %.1f ms",
                                 delta_ms, MAX_SYNC_DELTA_MS);
            return;
        }

        RCLCPP_DEBUG(this->get_logger(), "Sync delta: %.2f ms", delta_ms);

        {
            std::lock_guard<std::mutex> lock(_report_mutex);
            _current_car_x = car_msg->x;
            _current_car_y = car_msg->y;
            _current_car_psi = car_msg->psi;
        }

        gtsam::Pose2 current_raw(car_msg->x, car_msg->y, car_msg->psi);

        std::lock_guard<std::mutex> slam_lock(_slam_mutex);

        if (!_is_initialized) {
            _slam = std::make_unique<mirena::Slam>(current_raw);
            _last_keyframe_raw_odom = current_raw;
            _last_kf_cov[0] = car_msg->covariance[0];
            _last_kf_cov[1] = car_msg->covariance[7];
            _last_kf_cov[2] = car_msg->covariance[14];
            _is_initialized = true;
            RCLCPP_INFO(this->get_logger(), "SLAM Engine Initialized at (%.2f, %.2f)", car_msg->x, car_msg->y);
            publish_snapshot(car_msg->header.stamp);
            return;
        }

        // 2. Check Displacement. Keyframe on motion only -- the old "or the map
        //    is empty" clause meant that until the first cone landed, every
        //    synchronised pair became a keyframe: a pile of near-zero-delta poses
        //    right where the prior is stiffest.
        gtsam::Pose2 delta = _last_keyframe_raw_odom.between(current_raw);
        const bool first_keyframe = !_has_keyframe;
        if (first_keyframe || delta.translation().norm() >= DIST_THRESH ||
            std::abs(delta.theta()) >= ANGLE_THRESH)
        {
            auto dynamic_noise = make_odometry_noise(*car_msg, delta);

            std::vector<mirena::Landmark> obs;
            obs.reserve(landmark_msg->entities.size());
            for (const auto &e : landmark_msg->entities) {
                obs.push_back({e.position.x, e.position.y, e.type, e.confidence, {}});
            }

            // 5. Add to Graph
            _slam->add_keyframe(current_raw, obs, dynamic_noise);
            _last_keyframe_raw_odom = current_raw;

            publish_snapshot(car_msg->header.stamp);

            {
                std::lock_guard<std::mutex> lock(_report_mutex);
                _last_update_timestamp = this->now();
                _last_update_total_landmark_count = static_cast<uint16_t>(_slam->get_map().size());
                _last_update_ms_elapsed = _slam->get_last_update_elapsed_ms();
                _last_matched = _slam->get_last_matched_count();
                _last_new = _slam->get_last_new_count();
                _has_keyframe = true;
            }
        }
    }

    /// Noise for the odometry *delta*, not for the absolute pose.
    ///
    /// The message carries the covariance of the absolute pose in odom, which
    /// grows monotonically as the car drives; feeding it straight into a
    /// BetweenFactor made the odometry weaker and weaker for no physical reason,
    /// and the old 0.5 m^2 floor meant a 0.5 m keyframe step was constrained by
    /// a sigma of 0.71 m -- effectively no constraint at all. What a relative
    /// factor deserves is the growth since the last keyframe, floored by a motion
    /// model so a reset or an over-confident estimator cannot lock the graph.
    /// Caller must hold _slam_mutex.
    gtsam::SharedNoiseModel make_odometry_noise(const mirena_common::msg::Car &car,
                                                const gtsam::Pose2 &delta)
    {
        const double cov_now[3] = {car.covariance[0], car.covariance[7], car.covariance[14]};

        const double dist = delta.translation().norm();
        const double turn = std::abs(delta.theta());
        const double model_xy = 0.05 * dist + 0.10 * turn;
        const double model_th = 0.02 * dist + 0.05 * turn;

        auto sigma_from = [](double increment, double model, double floor) {
            const double from_msg = increment > 0.0 ? std::sqrt(increment) : 0.0;
            return std::max({from_msg, model, floor});
        };

        const double sx = sigma_from(cov_now[0] - _last_kf_cov[0], model_xy, 0.02);
        const double sy = sigma_from(cov_now[1] - _last_kf_cov[1], model_xy, 0.02);
        const double st = sigma_from(cov_now[2] - _last_kf_cov[2], model_th, 0.005);

        _last_kf_cov[0] = cov_now[0];
        _last_kf_cov[1] = cov_now[1];
        _last_kf_cov[2] = cov_now[2];

        return gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector3(sx, sy, st));
    }

    /// Copy the optimiser's result somewhere the publisher thread can read it
    /// without touching _slam. Building the message here rather than in the timer
    /// keeps the timer's critical section down to two pointer copies.
    /// Caller must hold _slam_mutex.
    void publish_snapshot(const builtin_interfaces::msg::Time &stamp)
    {
        auto map_msg = std::make_shared<mirena_common::msg::EntityList>();
        const auto &landmarks = _slam->get_map();
        map_msg->entities.reserve(landmarks.size());
        for (const auto &l : landmarks) {
            mirena_common::msg::Entity e;
            e.position.x = l.x;
            e.position.y = l.y;
            e.type = l.type;
            e.confidence = l.confidence;
            map_msg->entities.push_back(e);
        }
        map_msg->header.frame_id = FIXED_FRAME;
        map_msg->header.stamp = stamp;

        const gtsam::Pose2 correction = _slam->get_correction();

        std::lock_guard<std::mutex> lock(_snapshot_mutex);
        _map_to_odom_snapshot = correction;
        _map_snapshot = std::move(map_msg);
        _has_snapshot = true;
    }

    void map_publish_timer_callback()
    {
        // Read the last optimisation result and get straight out of the way.
        gtsam::Pose2 map_to_odom;
        std::shared_ptr<const mirena_common::msg::EntityList> map_msg;
        {
            std::lock_guard<std::mutex> lock(_snapshot_mutex);
            if (!_has_snapshot)
                return;
            map_to_odom = _map_to_odom_snapshot;
            map_msg = _map_snapshot;
        }

        // 1. Broadcast map -> odom.
        //
        // Stamped at broadcast time, NOT at the time of the data it was derived
        // from. map->odom is a piecewise-constant correction that only changes on
        // a keyframe, so re-publishing it under an old stamp adds no new sample to
        // the tf buffer -- the newest map->odom would then only ever advance as
        // fast as perception delivers a synchronised pair, and any consumer
        // looking up at a fresher time (control uses the latest /state/car stamp)
        // fails with "extrapolation into the future".
        geometry_msgs::msg::TransformStamped t;
        t.header.stamp = this->now();
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

        // 2. Republish the map as-is; it keeps the stamp of the update that
        // produced it, which is the time the landmarks were actually valid.
        if (map_msg)
            _map_pub->publish(*map_msg);
    }

    void report_callback() {
        if (!_report_pub) { return; }

        mirena_common::msg::Heartbeat msg;
        msg.timestamp = this->now();
        msg.node_name = this->get_name();

        nlohmann::json report_json;

        {
            std::lock_guard<std::mutex> lock(_report_mutex);

            // The supervisor could not previously tell healthy SLAM from SLAM
            // that had not produced a keyframe in five seconds: this field was
            // the constant "WORKING".
            const double since_update = (this->now() - _last_update_timestamp).seconds();
            const bool stale = _has_keyframe && since_update > REPORT_STALE_AFTER_S;
            report_json["status_state"] = !_has_keyframe ? "IDLE" : (stale ? "PARTIAL" : "WORKING");

            report_json["seconds_since_update"] = since_update;
            report_json["last_update_ms_elapsed"] = this->_last_update_ms_elapsed;
            report_json["last_update_total_landmark_count"] = this->_last_update_total_landmark_count;
            report_json["last_run_cones_seen"] = this->_last_update_total_landmark_count;

            // Once the track has been driven, new cones should stop appearing. If
            // they keep coming, association is failing to recognise cones already
            // mapped.
            report_json["matched_last_keyframe"] = _last_matched;
            report_json["new_cones_last_keyframe"] = _last_new;
            report_json["dropped_sync_pairs"] = _dropped_sync_pairs;

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

    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}