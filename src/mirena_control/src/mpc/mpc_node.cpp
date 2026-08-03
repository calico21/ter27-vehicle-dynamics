// Model-predictive contouring controller for the Mirena driverless car.
//
// Three loops run concurrently on their own callback groups:
//   * 20 Hz  solver     -- projects the car onto the planned centreline, builds
//                          the local reference and runs the acados NLP;
//   * 100 Hz commander  -- replays the latest optimal input horizon at the
//                          wall-clock offset reached since the solve started;
//   * 10 Hz  heartbeat  -- reports liveness and progress to the supervisor.
//
// The commander is the loop with the hard deadline, so it never takes a lock
// that the solver can hold: the solved horizon reaches it through a seqlock.

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include <rclcpp/rclcpp.hpp>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include "mirena_common/msg/car.hpp"
#include "mirena_common/msg/car_control.hpp"
#include "mirena_common/msg/heartbeat.hpp"
#include "mirena_common/msg/track.hpp"

#include "mirena_control/cubic_spline.hpp"
#include "mirena_control/mpc_controller.hpp"
#include "mirena_control/seqlock.hpp"

namespace mirena
{

using Mpc = MpcController;

namespace
{

constexpr int kPathPoints = Mpc::kPathPoints;
constexpr int kHorizon = Mpc::kN;

double yawFromQuaternion(const geometry_msgs::msg::Quaternion & q)
{
  const double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
  const double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
  return std::atan2(siny_cosp, cosy_cosp);
}

/// Everything the solver needs from the state estimate.  Kept as a small POD so
/// the subscription callback can publish it without allocating.
struct CarState
{
  int64_t stamp_ns;
  double u;
  double v;
  double omega;
  bool valid;
};

/// The slice of the solution the 100 Hz commander replays.
struct CommandPlan
{
  int64_t solve_start_ns;
  double gas[kHorizon];
  double steer[kHorizon];
};

/// Immutable centreline snapshot, swapped in wholesale on each Track message.
struct TrackSnapshot
{
  CubicSpline2D spline;
  bool closed = false;
};

}  // namespace

class MpcNode : public rclcpp::Node
{
public:
  MpcNode()
  : rclcpp::Node("mirena_control")
  {
    // The mission's values arrive as launch arguments from the mission launch
    // file the supervisor started, so they are already in place by the time
    // they are read below.
    declare_parameter<int>("laps", 0);
    declare_parameter<double>("stop_distance", 20.0);
    declare_parameter<double>("min_path_length", 5.0);
    declare_parameter<double>("stop_speed_eps", 0.3);
    declare_parameter<double>("stop_hold_time", 0.5);
    declare_parameter<double>("lookahead_base", 15.0);
    // Raising lookahead_gain raises cruise speed (horizon equilibrium is
    // 2u ~= lookahead_base + lookahead_gain*u) but braking distance grows as u^2
    // while the truncation onset only grows as lookahead_gain*u. Past ~2.5 the
    // car can no longer stop inside the reference that is left to it.
    declare_parameter<double>("lookahead_gain", 1.2);
    declare_parameter<std::string>("robot_frame", "car/cog");
    declare_parameter<std::string>("work_frame", "odom");
    declare_parameter<std::string>("map_frame", "map");
    declare_parameter<double>("solver_period", 0.05);
    declare_parameter<double>("command_period", 0.01);
    declare_parameter<double>("heartbeat_period", 0.1);
    declare_parameter<double>("tf_timeout", 0.005);
    declare_parameter<double>("projection_window_back", 5.0);
    declare_parameter<double>("projection_window_forward", 20.0);
    declare_parameter<double>("projection_coarse_step", 0.25);

    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    // A dedicated listener thread keeps /tf ingestion off the solver's
    // callback group and is what makes a bounded canTransform() legal here.
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_, this, true);

    RCLCPP_INFO(get_logger(), "Starting local-frame control node...");

    laps_ = get_parameter("laps").as_int();
    stop_distance_ = get_parameter("stop_distance").as_double();
    min_path_length_ = std::max(get_parameter("min_path_length").as_double(), 0.1);
    stop_speed_eps_ = get_parameter("stop_speed_eps").as_double();
    stop_hold_time_ = get_parameter("stop_hold_time").as_double();
    lookahead_base_ = get_parameter("lookahead_base").as_double();
    lookahead_gain_ = get_parameter("lookahead_gain").as_double();
    robot_frame_ = get_parameter("robot_frame").as_string();
    work_frame_ = get_parameter("work_frame").as_string();
    map_frame_ = get_parameter("map_frame").as_string();
    solver_period_ = get_parameter("solver_period").as_double();
    command_period_ = get_parameter("command_period").as_double();
    heartbeat_period_ = get_parameter("heartbeat_period").as_double();
    tf_timeout_ = get_parameter("tf_timeout").as_double();
    projection_back_ = get_parameter("projection_window_back").as_double();
    projection_forward_ = get_parameter("projection_window_forward").as_double();
    projection_step_ = get_parameter("projection_coarse_step").as_double();

    if (laps_ > 0) {
      RCLCPP_INFO(
        get_logger(), "Lap mission: %d laps, stopping %.1f m past the final crossing.",
        laps_, stop_distance_);
    } else {
      RCLCPP_INFO(get_logger(), "Open-track mission: stopping at the last gate.");
    }

    mpc_ = std::make_unique<Mpc>();   // throws; main() reports and exits non-zero
    step_duration_ = mpc_->stepDuration();
    RCLCPP_INFO(
      get_logger(), "acados solver ready: %d states, %d inputs, %d steps of %.3f s",
      Mpc::kNx, Mpc::kNu, kHorizon, step_duration_);

    solver_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    command_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    state_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    track_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

    rclcpp::SubscriptionOptions state_opts;
    state_opts.callback_group = state_group_;
    rclcpp::SubscriptionOptions track_opts;
    track_opts.callback_group = track_group_;

    track_sub_ = create_subscription<mirena_common::msg::Track>(
      "planning/track", 10,
      std::bind(&MpcNode::trackCallback, this, std::placeholders::_1), track_opts);
    car_sub_ = create_subscription<mirena_common::msg::Car>(
      "state/car", 10,
      std::bind(&MpcNode::carCallback, this, std::placeholders::_1), state_opts);

    control_pub_ = create_publisher<mirena_common::msg::CarControl>("/control", 10);
    viz_pub_ = create_publisher<visualization_msgs::msg::Marker>("control/horizon_markers", 10);
    heartbeat_pub_ = create_publisher<mirena_common::msg::Heartbeat>("health", 10);

    initMarkers();

    CommandPlan plan{};
    plan.solve_start_ns = get_clock()->now().nanoseconds();
    plan_.store(plan);

    heartbeat_timer_ = rclcpp::create_timer(
      this, get_clock(), rclcpp::Duration::from_seconds(heartbeat_period_),
      std::bind(&MpcNode::publishHeartbeat, this), track_group_);
    solver_timer_ = rclcpp::create_timer(
      this, get_clock(), rclcpp::Duration::from_seconds(solver_period_),
      std::bind(&MpcNode::solverIteration, this), solver_group_);
    command_timer_ = rclcpp::create_timer(
      this, get_clock(), rclcpp::Duration::from_seconds(command_period_),
      std::bind(&MpcNode::commandIteration, this), command_group_);

    // Configuring starts a fresh run: drop any progress carried over from a
    // previous activation so the centreline is re-acquired from scratch.
    laps_done_.store(0, std::memory_order_relaxed);
    stop_since_ns_ = 0;
    global_s_.store(0.0, std::memory_order_relaxed);
    finish_armed_.store(false, std::memory_order_relaxed);
    finished_.store(false, std::memory_order_relaxed);

    // ...including what the heartbeat reports about it, so a re-configured node
    // does not advertise the previous run's track and solver health.
    have_track_.store(false, std::memory_order_relaxed);
    solver_status_.store(-1, std::memory_order_relaxed);
    last_solve_ok_ns_.store(0, std::memory_order_relaxed);
    car_.store(CarState{0, 0.0, 0.0, 0.0, false});
  }

private:
  // ---------------------------------------------------------------- callbacks

  void trackCallback(const mirena_common::msg::Track::SharedPtr msg)
  {
    if (msg->gates.size() < 2) {
      return;
    }

    // Closed tracks repeat the first gate so the periodic fit has a seam to
    // match; scratch is reused between messages to keep this allocation-free
    // once the track size settles.
    const std::size_t n = msg->gates.size() + (msg->is_closed ? 1 : 0);
    fit_scratch_.resize(2 * n);
    for (std::size_t i = 0; i < msg->gates.size(); ++i) {
      fit_scratch_[2 * i] = msg->gates[i].x;
      fit_scratch_[2 * i + 1] = msg->gates[i].y;
    }
    if (msg->is_closed) {
      fit_scratch_[2 * (n - 1)] = msg->gates.front().x;
      fit_scratch_[2 * (n - 1) + 1] = msg->gates.front().y;
    }

    auto snapshot = std::make_shared<TrackSnapshot>();
    snapshot->closed = msg->is_closed;
    const auto bc = msg->is_closed ?
      CubicSpline2D::Boundary::Periodic : CubicSpline2D::Boundary::NotAKnot;

    if (!snapshot->spline.build(fit_scratch_.data(), n, bc)) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Could not fit a spline through %zu gates; keeping the previous track.", n);
      return;
    }

    {
      std::lock_guard<std::mutex> lock(track_mutex_);
      track_ = std::move(snapshot);
    }
    have_track_.store(true, std::memory_order_relaxed);
  }

  void carCallback(const mirena_common::msg::Car::SharedPtr msg)
  {
    CarState s;
    s.stamp_ns = rclcpp::Time(msg->header.stamp).nanoseconds();
    s.u = msg->u;
    s.v = msg->v;
    s.omega = msg->omega;
    s.valid = true;
    car_.store(s);
  }

  // -------------------------------------------------------- reference tracking

  /// Local reference in the work frame, plus the pose the solver starts from.
  struct Reference
  {
    double points[2 * kPathPoints];
    double length;         // arc length the points span, the solver's path_length
    double offset;         // where the car sits inside that window, in metres
    double pose[3];        // x, y, yaw in the work frame
  };

  bool generateReference(const CarState & car, Reference & ref)
  {
    std::shared_ptr<const TrackSnapshot> track;
    {
      std::lock_guard<std::mutex> lock(track_mutex_);
      track = track_;
    }
    if (!track || !track->spline.valid()) {
      return false;
    }

    const rclcpp::Time target_time(car.stamp_ns, get_clock()->get_clock_type());
    const auto timeout = rclcpp::Duration::from_seconds(tf_timeout_);

    try {
      std::string tf_error;
      if (!tf_buffer_->canTransform(
          map_frame_, robot_frame_, target_time, timeout, &tf_error))
      {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "TF %s -> %s unavailable: %s", map_frame_.c_str(), robot_frame_.c_str(),
          tf_error.c_str());
        return false;
      }

      const auto t_map_car = tf_buffer_->lookupTransform(map_frame_, robot_frame_, target_time);
      const double car_map_x = t_map_car.transform.translation.x;
      const double car_map_y = t_map_car.transform.translation.y;

      const auto t_odom_car = tf_buffer_->lookupTransform(work_frame_, robot_frame_, target_time);
      ref.pose[0] = t_odom_car.transform.translation.x;
      ref.pose[1] = t_odom_car.transform.translation.y;
      ref.pose[2] = yawFromQuaternion(t_odom_car.transform.rotation);

      const CubicSpline2D & spline = track->spline;
      const double s_max = spline.length();
      if (s_max <= 0.0) {
        return false;
      }

      // Progress must stay unwrapped so it can pass laps * s_max; only the
      // sampling below folds back into the spline's domain.
      const bool wrap = track->closed;

      // Always seeded from the progress we already believe in, including the
      // very first call, where it is the 0 set at configure time -- the car
      // starts on the line.  A global nearest-point sweep must not be used as a
      // substitute: the unrolled skidpad path overlaps itself exactly, so the
      // nearest point is ambiguous between passes and can jump a whole circle.
      double progress = spline.project(
        car_map_x, car_map_y, global_s_.load(std::memory_order_relaxed), wrap,
        projection_back_, projection_forward_, projection_step_);
      global_s_.store(progress, std::memory_order_relaxed);

      // Where the run ends, in the same unwrapped arc length as progress.  A
      // lap mission needs the planner to have closed the loop before s_max
      // means "one lap"; until then nothing is truncated and the car keeps
      // driving.  An open path simply ends at its last gate.
      const bool finish_active = (laps_ == 0) || track->closed;
      double s_finish;
      double s_stop;
      if (laps_ > 0) {
        s_finish = static_cast<double>(laps_) * s_max;
        // Capped so a stop_distance >= one lap cannot fold the stop point onto
        // the following lap when the sampling below wraps.
        s_stop = s_finish + std::min(stop_distance_, 0.5 * s_max);
      } else {
        s_finish = s_max;
        s_stop = s_max;
      }
      s_stop = std::max(s_stop, min_path_length_);

      if (finish_active) {
        updateLapCount(progress, s_max, track->closed);
        if (progress >= s_finish) {
          finish_armed_.store(true, std::memory_order_relaxed);
        }
      }

      // Cut the reference at s_stop by sliding the whole window back, never by
      // clamping individual samples: the solver evaluates the reference at
      // p_norm = p / path_length over a uniform 20-point grid, so bunching the
      // tail would stop p meaning arc length exactly in the braking region, and
      // a fully collapsed window leaves the progress reward -0.08*vp with no
      // curvature to oppose it.  Sliding keeps the samples equispaced and
      // path_length honest, and simply stops the reference advancing -- which
      // is what makes the MPC decelerate of its own accord.
      const double lookahead = lookahead_base_ + car.u * lookahead_gain_;
      double ref_start = progress;
      double ref_len = lookahead;
      if (finish_active) {
        ref_start = std::max(std::min(progress, s_stop - min_path_length_), 0.0);
        ref_len = std::max(std::min(lookahead, s_stop - ref_start), min_path_length_);
      }
      ref.length = ref_len;
      ref.offset = std::min(std::max(progress - ref_start, 0.0), ref_len);

      // Sample the centreline over that window, in map coordinates.
      const double span = ref_len / static_cast<double>(kPathPoints - 1);
      for (int i = 0; i < kPathPoints; ++i) {
        double s = ref_start + span * static_cast<double>(i);
        if (track->closed) {
          s = std::fmod(s, s_max);
          if (s < 0.0) {
            s += s_max;
          }
        } else {
          s = std::min(std::max(s, 0.0), s_max);
        }
        spline.eval(s, ref.points[2 * i], ref.points[2 * i + 1]);
      }

      // ...then rotate them into the work frame in one pass.
      const auto t_odom_map = tf_buffer_->lookupTransform(work_frame_, map_frame_, target_time);
      const double yaw = yawFromQuaternion(t_odom_map.transform.rotation);
      const double c = std::cos(yaw);
      const double s = std::sin(yaw);
      const double tx = t_odom_map.transform.translation.x;
      const double ty = t_odom_map.transform.translation.y;
      for (int i = 0; i < kPathPoints; ++i) {
        const double px = ref.points[2 * i];
        const double py = ref.points[2 * i + 1];
        ref.points[2 * i] = c * px - s * py + tx;
        ref.points[2 * i + 1] = s * px + c * py + ty;
      }

      return true;

    } catch (const tf2::TransformException & e) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000, "TF lookup failed in generateReference: %s", e.what());
      return false;
    }
  }

  // ------------------------------------------------------------- solver loop

  void solverIteration()
  {
    if (!mpc_) {
      return;
    }

    const CarState car = car_.load();
    if (!car.valid) {
      return;
    }

    // Ahead of generateReference, so a TF dropout on the final metres cannot
    // stop the run from ever being declared finished.
    updateStopLatch(car);

    Reference ref;
    if (!generateReference(car, ref)) {
      return;
    }

    const auto & previous = mpc_->states();
    double x0[Mpc::kNx];
    x0[0] = ref.pose[0];
    x0[1] = ref.pose[1];
    x0[2] = ref.pose[2];
    x0[3] = car.u;
    x0[4] = car.v;
    x0[5] = car.omega;
    x0[6] = previous[1][6];      // carry the actuator state one step forward
    x0[7] = previous[1][7];
    x0[8] = ref.offset;          // where the car sits inside the reference window
    x0[9] = car.u;

    double params[Mpc::kNpGlobal];
    for (int i = 0; i < kPathPoints; ++i) {
      params[i] = ref.points[2 * i];
      params[kPathPoints + i] = ref.points[2 * i + 1];
    }
    params[2 * kPathPoints] = ref.length;

    const int64_t t_start = get_clock()->now().nanoseconds();
    const int status = mpc_->solve(x0, params);
    const int64_t t_end = get_clock()->now().nanoseconds();

    if (status == 0) {
      const auto & states = mpc_->states();
      CommandPlan plan;
      plan.solve_start_ns = t_start;
      for (int i = 0; i < kHorizon; ++i) {
        plan.gas[i] = states[i][6];
        plan.steer[i] = states[i][7];
      }
      plan_.store(plan);
    } else {
      mpc_->reset();
    }

    logDiagnostics(status, t_end, car.stamp_ns);
    broadcastDebug(ref);
  }

  void logDiagnostics(int status, int64_t end_ns, int64_t car_stamp_ns)
  {
    solver_status_.store(status, std::memory_order_relaxed);
    if (status != 0) {
      // Throttled: once the car is told to stop the reference stops tracking it
      // and every solve fails, which at 20 Hz would otherwise flood the log.
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 500, "MPC solve failed with status %d", status);
      return;
    }
    const Mpc::Stats s = mpc_->stats();
    const double latency_ms = static_cast<double>(end_ns - car_stamp_ns) * 1e-6;

    solve_time_ms_.store(s.solve_time_ms, std::memory_order_relaxed);
    solve_latency_ms_.store(latency_ms, std::memory_order_relaxed);
    solve_iterations_.store(s.sqp_iterations, std::memory_order_relaxed);
    solve_residual_.store(s.max_residual, std::memory_order_relaxed);
    last_solve_ok_ns_.store(end_ns, std::memory_order_relaxed);

    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 500,
      "LOCAL OK | Solve: %4.1fms | Lat: %4.1fms | Iters: %d | Res: %.1e",
      s.solve_time_ms, latency_ms, s.sqp_iterations, s.max_residual);
  }

  /// Laps come straight from how far the car has travelled along the
  /// centreline; the spline is anchored at the start gate, so s == 0 (mod
  /// s_max) is the finish line by construction.
  void updateLapCount(double progress, double s_max, bool closed)
  {
    if (laps_ <= 0 || !closed) {
      return;
    }
    const int done = static_cast<int>(std::floor(progress / s_max));
    if (done > laps_done_.load(std::memory_order_relaxed)) {
      laps_done_.store(done, std::memory_order_relaxed);
      RCLCPP_INFO(get_logger(), "Lap %d/%d completed.", done, laps_);
    }
  }

  /// The run is only reported finished once the car is actually stationary.
  /// The supervisor tears the stack down the moment it sees the flag, so
  /// latching any earlier would cut the controller off mid-braking.
  void updateStopLatch(const CarState & car)
  {
    if (!finish_armed_.load(std::memory_order_relaxed) ||
      finished_.load(std::memory_order_relaxed))
    {
      return;
    }
    const int64_t now_ns = get_clock()->now().nanoseconds();
    if (std::fabs(car.u) >= stop_speed_eps_) {
      stop_since_ns_ = 0;              // still rolling, re-arm
      return;
    }
    if (stop_since_ns_ == 0) {
      stop_since_ns_ = now_ns;
    } else if (static_cast<double>(now_ns - stop_since_ns_) * 1e-9 >= stop_hold_time_) {
      finished_.store(true, std::memory_order_relaxed);
      RCLCPP_INFO(
        get_logger(), "Stopped at s = %.1f m after %d lap(s). Mission complete.",
        global_s_.load(std::memory_order_relaxed), laps_done_.load(std::memory_order_relaxed));
    }
  }

  // ------------------------------------------------------------ command loop

  void commandIteration()
  {
    // No gate here: commands go out from the moment there is a solution. The
    // car is what decides whether to act on them -- it ignores this node until
    // its own state machine reaches DRIVING.
    const auto now = get_clock()->now();
    command_msg_.header.stamp = now;

    // No finish override lives here on purpose: bringing the car to a stop is
    // the job of the truncated reference, which the solver brakes against on
    // its own. Holding the car once stopped belongs to the braking system.
    const CommandPlan plan = plan_.load();
    const double elapsed = static_cast<double>(now.nanoseconds() - plan.solve_start_ns) * 1e-9;
    int index = static_cast<int>(elapsed / step_duration_);
    index = std::min(std::max(index, 0), kHorizon - 1);

    command_msg_.gas = plan.gas[index];
    command_msg_.steer_angle = plan.steer[index];
    control_pub_->publish(command_msg_);
  }

  // ----------------------------------------------------------- visualisation

  void initMarkers()
  {
    auto setup = [this](visualization_msgs::msg::Marker & m, const char * ns, int id,
        float r, float g, float b, std::size_t points) {
        m.header.frame_id = work_frame_;
        m.ns = ns;
        m.id = id;
        m.type = visualization_msgs::msg::Marker::LINE_STRIP;
        m.action = visualization_msgs::msg::Marker::ADD;
        m.scale.x = 0.1;
        m.color.r = r;
        m.color.g = g;
        m.color.b = b;
        m.color.a = 1.0;
        m.pose.orientation.w = 1.0;
        m.points.resize(points);
        for (auto & p : m.points) {
          p.z = 0.1;
        }
      };

    setup(ref_marker_, "local_ref", 0, 1.0f, 0.0f, 0.0f, kPathPoints);
    setup(pred_marker_, "local_pred", 1, 0.0f, 1.0f, 1.0f, kHorizon + 1);
  }

  void broadcastDebug(const Reference & ref)
  {
    if (!viz_pub_) {
      return;
    }
    const auto stamp = get_clock()->now();

    ref_marker_.header.stamp = stamp;
    for (int i = 0; i < kPathPoints; ++i) {
      ref_marker_.points[i].x = ref.points[2 * i];
      ref_marker_.points[i].y = ref.points[2 * i + 1];
    }
    viz_pub_->publish(ref_marker_);

    const auto & states = mpc_->states();
    pred_marker_.header.stamp = stamp;
    for (int i = 0; i <= kHorizon; ++i) {
      pred_marker_.points[i].x = states[i][0];
      pred_marker_.points[i].y = states[i][1];
    }
    viz_pub_->publish(pred_marker_);
  }

  void publishHeartbeat()
  {
    if (!heartbeat_pub_) {
      return;
    }
    mirena_common::msg::Heartbeat msg;
    msg.timestamp = get_clock()->now();
    msg.node_name = get_name();

    // Everything below is read from another thread's writes, so it comes from
    // the seqlock or from atomics; the heartbeat never blocks the solver.
    const CarState car = car_.load();
    const bool have_track = have_track_.load(std::memory_order_relaxed);
    const int solver_status = solver_status_.load(std::memory_order_relaxed);
    const int64_t last_ok_ns = last_solve_ok_ns_.load(std::memory_order_relaxed);
    const double since_solve = last_ok_ns == 0 ?
      -1.0 : static_cast<double>(get_clock()->now().nanoseconds() - last_ok_ns) * 1e-9;

    // Waiting on an input is a different failure from a solver that runs but
    // will not converge, and the supervisor can only act on the difference if
    // the report carries it.
    const char * status_state = "WORKING";
    if (!have_track || !car.valid) {
      status_state = "IDLE";
    } else if (solver_status != 0) {
      status_state = "PARTIAL";
    }

    nlohmann::json report;

    // "mission_finished" is the exact key the supervisor ends the mission off:
    // the lap goal is met and the car has come to a stop. Deciding that is this
    // node's job alone -- nothing else in the stack may declare a run over --
    // and the inspection node reports the same key when its sweep completes.
    report["mission_finished"] = finished_.load(std::memory_order_relaxed);
    report["mode"] = "mpc";
    report["status_state"] = status_state;
    report["progress"] = global_s_.load(std::memory_order_relaxed);
    report["laps"] = laps_done_.load(std::memory_order_relaxed);

    report["laps_target"] = laps_;
    report["finish_armed"] = finish_armed_.load(std::memory_order_relaxed);
    report["have_track"] = have_track;
    report["have_state"] = car.valid;
    report["speed_mps"] = car.u;

    report["solver_status"] = solver_status;
    report["solve_time_ms"] = solve_time_ms_.load(std::memory_order_relaxed);
    report["solve_latency_ms"] = solve_latency_ms_.load(std::memory_order_relaxed);
    report["sqp_iterations"] = solve_iterations_.load(std::memory_order_relaxed);
    report["max_residual"] = solve_residual_.load(std::memory_order_relaxed);
    report["seconds_since_solve"] = since_solve;

    msg.report = report.dump();
    heartbeat_pub_->publish(msg);
  }

  // -------------------------------------------------------------------- state

  // Parameters. Declared in the constructor but read in on_configure, because
  // the supervisor pushes the mission's values between the two.
  int laps_ = 0;
  double stop_distance_;
  double min_path_length_;
  double stop_speed_eps_;
  double stop_hold_time_;
  double lookahead_base_;
  double lookahead_gain_;
  std::string robot_frame_;
  std::string work_frame_;
  std::string map_frame_;
  double solver_period_;
  double command_period_;
  double heartbeat_period_;
  double tf_timeout_;
  double projection_back_;
  double projection_forward_;
  double projection_step_;

  std::unique_ptr<Mpc> mpc_;
  double step_duration_ = 0.1;

  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  rclcpp::CallbackGroup::SharedPtr solver_group_;
  rclcpp::CallbackGroup::SharedPtr command_group_;
  rclcpp::CallbackGroup::SharedPtr state_group_;
  rclcpp::CallbackGroup::SharedPtr track_group_;

  rclcpp::Subscription<mirena_common::msg::Track>::SharedPtr track_sub_;
  rclcpp::Subscription<mirena_common::msg::Car>::SharedPtr car_sub_;

  rclcpp::Publisher<mirena_common::msg::CarControl>::SharedPtr control_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr viz_pub_;
  rclcpp::Publisher<mirena_common::msg::Heartbeat>::SharedPtr heartbeat_pub_;

  rclcpp::TimerBase::SharedPtr solver_timer_;
  rclcpp::TimerBase::SharedPtr command_timer_;
  rclcpp::TimerBase::SharedPtr heartbeat_timer_;

  // Cross-thread hand-offs.
  Seqlock<CarState> car_{CarState{0, 0.0, 0.0, 0.0, false}};
  Seqlock<CommandPlan> plan_{CommandPlan{}};

  std::mutex track_mutex_;
  std::shared_ptr<const TrackSnapshot> track_;

  std::atomic<double> global_s_{0.0};
  std::atomic<bool> finish_armed_{false};   // the car has passed s_finish
  std::atomic<bool> finished_{false};       // ...and has since come to a stop
  std::atomic<bool> have_track_{false};

  // Written by the solver, read by the heartbeat: the last solve's outcome.
  std::atomic<int> laps_done_{0};
  std::atomic<int> solver_status_{-1};      // acados status, -1 until the first solve
  std::atomic<int> solve_iterations_{0};
  std::atomic<double> solve_time_ms_{0.0};
  std::atomic<double> solve_latency_ms_{0.0};
  std::atomic<double> solve_residual_{0.0};
  std::atomic<int64_t> last_solve_ok_ns_{0};

  // Track-thread only.
  std::vector<double> fit_scratch_;

  // Solver-thread only.
  int64_t stop_since_ns_ = 0;

  // Reused outgoing messages, so the hot loops do not allocate.
  mirena_common::msg::CarControl command_msg_;
  visualization_msgs::msg::Marker ref_marker_;
  visualization_msgs::msg::Marker pred_marker_;
};

}  // namespace mirena

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  std::shared_ptr<mirena::MpcNode> node;
  try {
    node = std::make_shared<mirena::MpcNode>();
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("mirena_control"), "Startup failed: %s", e.what());
    rclcpp::shutdown();
    return 1;
  }

  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 4);
  executor.add_node(node);

  try {
    executor.spin();
  } catch (const std::exception & e) {
    RCLCPP_ERROR(node->get_logger(), "Executor terminated: %s", e.what());
  }

  executor.remove_node(node);
  rclcpp::shutdown();
  return 0;
}
