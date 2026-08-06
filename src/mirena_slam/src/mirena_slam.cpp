#include <rclcpp/rclcpp.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <mutex> // NUEVO: Para seguridad en hilos (Thread-safety)

// Lie-group interpolation, for easing the published correction
#include <gtsam/base/Lie.h>

// Message filters for time-synchronization
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

// Debug visualisation
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

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
#define DEBUG_TOPIC "slam/debug/associations"

#define FIXED_FRAME "map"
#define MOVING_FRAME "odom"

// Parameters
#define MAP_PUBLISH_PERIOD 100ms 
#define REPORT_PERIOD 100ms
#define DIST_THRESH 0.5    // Meters moved before creating a keyframe
#define ANGLE_THRESH 0.3   // Radians turned before creating a keyframe
#define MAX_SYNC_DELTA_MS 10.0 // Pose/detection pairs further apart are dropped
#define REPORT_STALE_AFTER_S 1.0
// Longest a keyframe may be deferred for want of motion. A car waiting on the
// line does not move, so the thresholds above never fire and the heartbeat used
// to go PARTIAL a second into every start -- healthy SLAM, reported as failing.
// Kept below REPORT_STALE_AFTER_S so standing still stays inside the window.
#define KEYFRAME_MAX_INTERVAL_S 0.5
// Time constant of the filter on the published map->odom, seconds.
//
// The correction absorbs accumulated odometry drift, so what it has to say
// changes over metres and laps. What it actually did on a measured run was
// travel 75 m of path to end up 3 cm from where it started, and swing 64 deg of
// heading to drift a third of one -- because it is re-derived from a fresh
// solve several times a second, and every solve moves the whole window a little.
// That is not error to be fixed in the optimiser, it is estimator noise on a
// quantity that is genuinely slowly varying, and the place to take it out is
// here. Against a signal drifting a third of a degree a lap, half a second of
// lag is nothing; against noise arriving at 3 Hz it is most of it.
#define CORRECTION_TIME_CONSTANT_S 0.5

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
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr _debug_pub;
    bool _publish_debug = true;

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
    int _last_removed = 0;
    int _window_size = 0;
    uint64_t _dropped_sync_pairs = 0;

    // Result of the last optimisation, handed to the publisher thread.
    // Everything the timer needs is copied out here so it never touches _slam
    // while the optimiser is running -- note get_map() hands back a reference to
    // a vector that update_isam() clears and refills.
    std::mutex _snapshot_mutex;
    gtsam::Pose2 _map_to_odom_snapshot;
    std::shared_ptr<const mirena_common::msg::EntityList> _map_snapshot;
    bool _has_snapshot = false;

    // What was last put on the wire, and when. Only the publisher touches these,
    // so they need no lock. The transform is filtered towards the snapshot
    // rather than snapped to it, which is why they have to persist between
    // ticks.
    gtsam::Pose2 _published_correction;
    rclcpp::Time _last_broadcast;
    bool _has_broadcast = false;

    // SLAM Engine.
    //
    // Guarded because the sync callback runs in _slam_cb_group under a
    // MultiThreadedExecutor while the map and report timers run in their own.
    std::mutex _slam_mutex;
    std::unique_ptr<mirena::Slam> _slam;
    gtsam::Pose2 _last_keyframe_raw_odom;
    /// Stamp of the last keyframe, from the /state/car header rather than the
    /// wall clock: make_odometry_noise integrates the estimator's velocity
    /// uncertainty over this interval.
    rclcpp::Time _last_keyframe_time;
    bool _is_initialized = false;

public:
    MirenaMapping() : rclcpp::Node("mirena_mapping") {
        this->_last_update_timestamp = this->now();
        this->_last_keyframe_time = this->now();
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

        _publish_debug = this->declare_parameter<bool>("publish_debug", true);
        if (_publish_debug)
            _debug_pub = this->create_publisher<visualization_msgs::msg::MarkerArray>(DEBUG_TOPIC, 1);
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
            _last_keyframe_time = car_msg->header.stamp;
            _is_initialized = true;
            RCLCPP_INFO(this->get_logger(), "SLAM Engine Initialized at (%.2f, %.2f)", car_msg->x, car_msg->y);
            publish_snapshot(car_msg->header.stamp);
            return;
        }

        // 2. Every frame, match the cones against the map. This is where cones
        //    are confirmed and where ones that were never there are dropped, and
        //    it runs at the rate perception delivers rather than the rate the car
        //    moves far enough to be worth a new pose -- which on a slow lap is
        //    ten times as much evidence.
        std::vector<mirena::Observation> obs;
        obs.reserve(landmark_msg->entities.size());
        for (const auto &e : landmark_msg->entities) {
            obs.push_back({e.position.x, e.position.y, e.type,
                           {e.covariance[0], e.covariance[1],
                            e.covariance[2], e.covariance[3]}});
        }

        mirena::SlamDebug debug;
        mirena::SlamDebug *sink = _publish_debug ? &debug : nullptr;
        _slam->observe(current_raw, obs, sink);

        // 3. Only commit a pose when the car has actually gone somewhere. The old
        //    "or the map is empty" clause meant that until the first cone landed,
        //    every synchronised pair became a keyframe: a pile of near-zero-delta
        //    poses right where the prior is stiffest.
        gtsam::Pose2 delta = _last_keyframe_raw_odom.between(current_raw);
        const bool first_keyframe = !_has_keyframe;
        // Standing still is a state to keep observing, not one to fall silent
        // in: the cones in front of the car are still being seen, and the graph
        // should keep taking them.
        // Measured against the message stamps, not now(): the interval is a
        // property of the data, and it is what the odometry noise below
        // integrates over. Mixing a wall reading into that would make the
        // factor weights depend on how busy this node happened to be.
        const double interval_s = (car_stamp - _last_keyframe_time).seconds();
        const bool overdue = interval_s >= KEYFRAME_MAX_INTERVAL_S;
        if (first_keyframe || overdue || delta.translation().norm() >= DIST_THRESH ||
            std::abs(delta.theta()) >= ANGLE_THRESH)
        {
            _slam->add_keyframe(make_odometry_noise(*car_msg, delta, interval_s), sink);
            _last_keyframe_raw_odom = current_raw;
            _last_keyframe_time = car_stamp;

            {
                std::lock_guard<std::mutex> lock(_report_mutex);
                _last_update_timestamp = this->now();
                _last_update_ms_elapsed = _slam->get_last_update_elapsed_ms();
                _window_size = _slam->get_window_size();
                _has_keyframe = true;
            }
        }

        // The map changes on every frame now, not only on keyframes, because a
        // cone can be born or dropped between them.
        publish_snapshot(car_msg->header.stamp);
        if (_publish_debug)
            publish_debug(debug, car_msg->header.stamp);

        {
            std::lock_guard<std::mutex> lock(_report_mutex);
            _last_update_total_landmark_count = static_cast<uint16_t>(_slam->get_map().size());
            _last_matched = _slam->get_last_matched_count();
            _last_new = _slam->get_last_new_count();
            _last_removed = _slam->get_last_removed_count();
        }
    }

    /// Noise for the odometry *delta*, not for the absolute pose.
    ///
    /// This used to difference the absolute position covariance across the
    /// keyframe interval. That cannot work: a converged filter's absolute
    /// covariance is flat -- measured at 0.173 m sigma, varying by under a
    /// millimetre between keyframes -- so the difference was always ~0 and the
    /// message covariance never once influenced a factor. Every odometry
    /// constraint in the graph came from the motion model below, silently.
    ///
    /// What a relative factor actually deserves is the uncertainty accumulated
    /// by integrating the estimator's *velocity* over the interval. Velocity
    /// error out of an INS is bias-like over these timescales, so it integrates
    /// as sigma_v * dt rather than sqrt(dt), and unlike the absolute position
    /// covariance it does grow when the estimator loses its corrections.
    ///
    /// The motion model stays as a floor and still governs normal driving: it
    /// covers scale and lever-arm error the filter does not model, and it stops
    /// an over-confident estimator from locking the graph. Replayed over
    /// rosbag2_2026_08_05-20_26_23 (2780 keyframes) the median is unchanged at
    /// 0.030 m and the velocity term overrides the model on 2.0% of keyframes --
    /// the INS dropouts, where sigma_u rises from 0.12 to 0.88 m/s and this
    /// factor loosens to 0.071 m, 2.4x the model. That is the whole intent: no
    /// change when the odometry is healthy, a real loosening when it is not.
    ///
    /// Note this moves the position term only. sigma_omega out of the filter is
    /// ~0.01 rad/s, so over a 0.07 s keyframe the heading drift term is 0.0007
    /// rad against the model's 0.012, and heading stays motion-model governed on
    /// every keyframe in that bag. If the heading weight ever needs to respond
    /// to filter confidence, gyro_meas_var in mirena_state is what sets it.
    ///
    /// Caller must hold _slam_mutex.
    gtsam::SharedNoiseModel make_odometry_noise(const mirena_common::msg::Car &car,
                                                const gtsam::Pose2 &delta,
                                                double interval_s)
    {
        // Car covariance is row-major 6x6 over [x, y, psi, u, v, omega], so the
        // velocity block sits on the diagonal at 21, 28 and 35.
        const double dt = std::clamp(interval_s, 0.0, KEYFRAME_MAX_INTERVAL_S);
        const double drift_x = std::sqrt(std::max(0.0, car.covariance[21])) * dt;
        const double drift_y = std::sqrt(std::max(0.0, car.covariance[28])) * dt;
        const double drift_th = std::sqrt(std::max(0.0, car.covariance[35])) * dt;

        const double dist = delta.translation().norm();
        const double turn = std::abs(delta.theta());
        const double model_xy = 0.05 * dist + 0.10 * turn;
        const double model_th = 0.02 * dist + 0.05 * turn;

        const double sx = std::max({drift_x, model_xy, 0.02});
        const double sy = std::max({drift_y, model_xy, 0.02});
        const double st = std::max({drift_th, model_th, 0.005});

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

    /// One marker per thing worth seeing, all under the map frame.
    ///
    /// Published straight from the keyframe rather than off the snapshot timer:
    /// this describes one optimisation step, and re-sending it at 10 Hz would
    /// make a single bad association look like a persistent one.
    void publish_debug(const mirena::SlamDebug &debug,
                       const builtin_interfaces::msg::Time &stamp)
    {
        using visualization_msgs::msg::Marker;
        visualization_msgs::msg::MarkerArray array;

        std_msgs::msg::Header header;
        header.frame_id = FIXED_FRAME;
        header.stamp = stamp;

        // Wipe first. Every keyframe redraws the lot, and without this the
        // labels of landmarks that dropped out of range stay on screen for ever.
        Marker clear;
        clear.header = header;
        clear.action = Marker::DELETEALL;
        array.markers.push_back(clear);

        const auto marker = [&](const char *ns, int32_t type, double scale,
                                float r, float g, float b, float a) {
            Marker m;
            m.header = header;
            m.ns = ns;
            m.id = 0;
            m.type = type;
            m.action = Marker::ADD;
            m.pose.orientation.w = 1.0;
            m.scale.x = m.scale.y = m.scale.z = scale;
            m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = a;
            return m;
        };
        const auto point = [](const gtsam::Point2 &p) {
            geometry_msgs::msg::Point out;
            out.x = p.x();
            out.y = p.y();
            return out;
        };

        // Each line runs from where the cone was seen to the landmark it was
        // matched to, so its length is the innovation the gate let through. Short
        // and tidy is healthy; lines growing in one direction is the pose
        // drifting away from the map.
        Marker links = marker("associations", Marker::LINE_LIST, 0.03, 0.2f, 0.9f, 0.3f, 0.9f);
        for (const auto &[observed, landmark] : debug.associations) {
            links.points.push_back(point(observed));
            links.points.push_back(point(landmark));
        }
        array.markers.push_back(links);

        // New landmarks. After the first lap these should stop appearing; if they
        // do not, association is failing to recognise cones already mapped.
        Marker created = marker("created", Marker::SPHERE_LIST, 0.45, 0.2f, 0.8f, 1.0f, 0.9f);
        for (const auto &p : debug.created)
            created.points.push_back(point(p));
        array.markers.push_back(created);

        // Detections dropped for landing on a cone already mapped. A few are
        // normal -- two detections of one cone in a frame, and the assignment can
        // only claim it once. A steady stream means the gate is too tight to
        // recognise its own landmarks.
        Marker suppressed = marker("suppressed", Marker::CUBE_LIST, 0.3, 1.0f, 0.3f, 0.2f, 0.9f);
        for (const auto &p : debug.suppressed)
            suppressed.points.push_back(point(p));
        array.markers.push_back(suppressed);

        // Cones deleted this keyframe: looked at repeatedly from close range and
        // never seen again, so they were never cones.
        Marker removed = marker("removed", Marker::CUBE_LIST, 0.4, 1.0f, 0.9f, 0.1f, 0.9f);
        for (const auto &p : debug.removed)
            removed.points.push_back(point(p));
        array.markers.push_back(removed);

        // How many times each nearby cone has been seen. A landmark stuck at one
        // while the car drives past it repeatedly is about to be removed.
        int id = 0;
        for (const auto &[position, sightings] : debug.sightings) {
            Marker label = marker("sightings", Marker::TEXT_VIEW_FACING, 0.3,
                                  0.9f, 0.9f, 0.9f, 0.8f);
            label.id = id++;
            label.pose.position = point(position);
            label.pose.position.z = 0.4;
            label.text = std::to_string(sightings);
            array.markers.push_back(label);
        }

        _debug_pub->publish(array);
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

        // 1. Ease the published correction towards the one SLAM last solved.
        //
        // A first-order filter, stepped with the time actually elapsed rather
        // than the nominal period, so a late tick catches up by the right
        // amount instead of falling behind. Interpolation is along the SE(2)
        // geodesic: doing it on x, y and yaw separately would drag the car
        // sideways through the turn between two corrections that differ in
        // heading.
        //
        // The first correction is taken as-is. Easing in from identity would
        // spend the first seconds of every run publishing a transform that is
        // knowingly wrong.
        const rclcpp::Time now = this->now();
        if (!_has_broadcast)
        {
            _published_correction = map_to_odom;
            _has_broadcast = true;
        }
        else
        {
            const double dt = std::clamp((now - _last_broadcast).seconds(), 0.0, 1.0);
            const double alpha = 1.0 - std::exp(-dt / CORRECTION_TIME_CONSTANT_S);
            _published_correction =
                gtsam::interpolate<gtsam::Pose2>(_published_correction, map_to_odom, alpha);
        }
        _last_broadcast = now;
        map_to_odom = _published_correction;

        // 2. Broadcast map -> odom.
        //
        // Stamped at broadcast time, NOT at the time of the data it was derived
        // from. map->odom is a piecewise-constant correction that only changes on
        // a keyframe, so re-publishing it under an old stamp adds no new sample to
        // the tf buffer -- the newest map->odom would then only ever advance as
        // fast as perception delivers a synchronised pair, and any consumer
        // looking up at a fresher time (control uses the latest /state/car stamp)
        // fails with "extrapolation into the future".
        geometry_msgs::msg::TransformStamped t;
        t.header.stamp = now;
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

        // 3. Republish the map as-is; it keeps the stamp of the update that
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
            report_json["removed_last_keyframe"] = _last_removed;
            // Constant once the window fills. If it is not, the optimiser is
            // being handed a bigger problem every keyframe and will not keep up.
            report_json["window_keyframes"] = _window_size;
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