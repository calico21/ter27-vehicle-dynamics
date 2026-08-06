// Inspection controller for the Mirena driverless car.
//
// The static inspection mission: the car is on stands, the steering is swept
// with a sine and the drivetrain is run through a short throttle profile, for a
// fixed window, to demonstrate that both actuators respond. Nothing is planned
// and nothing is solved, so this node replaces the MPC one for the length of
// that mission -- it is the alternative occupant of the same slot in the stack,
// which is why it carries the same default node name, subscribes to nothing
// else and publishes the same /control and health topics.
//
// The throttle profile is three parameterised stages, timed from the same clock
// as the sweep: launch_gas until launch_time to get the wheels turning against
// their inertia, then hold_gas until hold_time, then full braking for whatever
// is left of the sweep. Every edge is a parameter because the numbers are what
// the scrutineers ask to see changed, and a profile compiled into the binary is
// one nobody can change on the day.
//
// Two loops run on their own callback groups:
//   * 100 Hz sweep      -- publishes the open-loop steering and gas command;
//   * 10 Hz  heartbeat  -- reports liveness and progress to the supervisor.
//
// Like the MPC node, this one is the sole authority on when its run is over: it
// latches mission_finished in the heartbeat once the window closes, and the
// supervisor ends the mission off that flag.

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>

#include <nlohmann/json.hpp>

#include <rclcpp/rclcpp.hpp>

#include "mirena_common/msg/as_info.hpp"
#include "mirena_common/msg/car_control.hpp"
#include "mirena_common/msg/heartbeat.hpp"

namespace mirena
{

namespace
{

/// Sentinel for "the car has not started driving yet".  Zero will not do: under
/// sim time the clock legitimately starts at zero, so a zero sentinel would
/// make a sweep that began at t = 0 look like one that never began.
constexpr int64_t kUnsetTime = std::numeric_limits<int64_t>::min();

/// Held once the window closes, and whenever a stage boundary leaves nothing
/// else to command: wheels straight, full braking.
constexpr double kFailsafeGas = -1.0;
constexpr double kFailsafeSteer = 0.0;

/// Topic naming follows the stack's convention -- see the MPC node.
constexpr char kAsStatusTopic[] = "/as_status";
constexpr char kControlTopic[] = "/control";
constexpr char kHealthTopic[] = "health";

}  // namespace


class InspectionNode : public rclcpp::Node
{
public:
  InspectionNode()
  : rclcpp::Node("mirena_control")
  {
    // The mission's values arrive as launch arguments from the mission launch
    // file the supervisor started, so they are already in place by the time the
    // constructor reads them below.  Nothing re-reads them afterwards: a node's
    // lifetime is the mission's, so a parameter that changes mid-sweep has
    // nowhere sensible to take effect.
    declare_parameter<double>("inspection_frequency", 0.5);   // Hz
    declare_parameter<double>("inspection_amplitude", 0.3);   // rad, ~17 deg
    declare_parameter<double>("inspection_duration", 25.0);   // s
    // Throttle profile, in seconds from the same start as the sweep.
    declare_parameter<double>("inspection_launch_gas", 0.24);
    declare_parameter<double>("inspection_launch_time", 4.0);
    declare_parameter<double>("inspection_hold_gas", 0.18);
    declare_parameter<double>("inspection_hold_time", 20.0);
    declare_parameter<double>("inspection_brake_gas", -1.0);
    declare_parameter<double>("command_period", 0.01);
    declare_parameter<double>("heartbeat_period", 0.1);

    RCLCPP_INFO(get_logger(), "Starting inspection control node...");

    frequency_ = get_parameter("inspection_frequency").as_double();
    amplitude_ = get_parameter("inspection_amplitude").as_double();
    duration_ = get_parameter("inspection_duration").as_double();
    launch_gas_ = get_parameter("inspection_launch_gas").as_double();
    launch_time_ = get_parameter("inspection_launch_time").as_double();
    hold_gas_ = get_parameter("inspection_hold_gas").as_double();
    hold_time_ = get_parameter("inspection_hold_time").as_double();
    brake_gas_ = get_parameter("inspection_brake_gas").as_double();
    const double command_period = get_parameter("command_period").as_double();
    const double heartbeat_period = get_parameter("heartbeat_period").as_double();

    // The stages have to be ordered and have to fit inside the sweep, or the
    // profile silently means something other than it reads: a hold_time past
    // the end of the window is a brake stage that never runs, and a launch_time
    // past hold_time is a hold stage that never runs.  Shortening the sweep
    // must shorten the profile with it rather than quietly drop a stage.
    hold_time_ = std::min(std::max(hold_time_, 0.0), duration_);
    launch_time_ = std::min(std::max(launch_time_, 0.0), hold_time_);

    RCLCPP_INFO(
      get_logger(), "Inspection mission: %.1f s steering sweep at %.2f Hz, +-%.2f rad.",
      duration_, frequency_, amplitude_);
    RCLCPP_INFO(
      get_logger(),
      "Throttle profile: gas %.2f to %.1f s, %.2f to %.1f s, %.2f thereafter.",
      launch_gas_, launch_time_, hold_gas_, hold_time_, brake_gas_);

    // The car's own state machine is what starts the clock. The sweep must be
    // a window of the wheel actually moving, and the wheel does not move until
    // the car reaches DRIVING, so counting from anything earlier -- launch,
    // mission select -- would short the run by however long that took.
    as_sub_ = create_subscription<mirena_common::msg::AsInfo>(
      kAsStatusTopic, 10,
      std::bind(&InspectionNode::asCallback, this, std::placeholders::_1));

    control_pub_ = create_publisher<mirena_common::msg::CarControl>(kControlTopic, 10);
    heartbeat_pub_ = create_publisher<mirena_common::msg::Heartbeat>(kHealthTopic, 10);

    sweep_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

    heartbeat_timer_ = rclcpp::create_timer(
      this, get_clock(), rclcpp::Duration::from_seconds(heartbeat_period),
      std::bind(&InspectionNode::publishHeartbeat, this));
    sweep_timer_ = rclcpp::create_timer(
      this, get_clock(), rclcpp::Duration::from_seconds(command_period),
      std::bind(&InspectionNode::sweepIteration, this), sweep_group_);
  }

private:
  // ------------------------------------------------------------- sweep loop

  /// Latches the moment the car starts driving; the sweep is timed from there.
  void asCallback(const mirena_common::msg::AsInfo::SharedPtr msg)
  {
    if (msg->status != mirena_common::msg::AsInfo::AS_STATUS_DRIVING) {
      return;
    }
    int64_t unset = kUnsetTime;
    if (start_ns_.compare_exchange_strong(
        unset, get_clock()->now().nanoseconds(), std::memory_order_relaxed))
    {
      RCLCPP_INFO(get_logger(), "AS is DRIVING: starting the %.1f s sweep.", duration_);
    }
  }

  /// The commanded throttle at `elapsed` seconds into the sweep.  Boundaries
  /// belong to the stage that is ending, so no instant falls into two stages.
  double gasAt(double elapsed) const
  {
    if (elapsed <= launch_time_) {
      return launch_gas_;
    }
    if (elapsed <= hold_time_) {
      return hold_gas_;
    }
    return brake_gas_;
  }

  void sweepIteration()
  {
    const int64_t start_ns = start_ns_.load(std::memory_order_relaxed);
    if (start_ns == kUnsetTime) {
      return;   // not driving yet; the sweep has not begun
    }

    const auto stamp = get_clock()->now();
    const double elapsed = static_cast<double>(stamp.nanoseconds() - start_ns) * 1e-9;
    elapsed_.store(elapsed, std::memory_order_relaxed);

    command_msg_.header.stamp = stamp;

    if (elapsed <= duration_) {
      command_msg_.gas = gasAt(elapsed);
      command_msg_.steer_angle = amplitude_ * std::sin(2.0 * M_PI * frequency_ * elapsed);
      control_pub_->publish(command_msg_);
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 500, "Sweep active: %.2f rad, gas %.2f (%.1f/%.1f s)",
        command_msg_.steer_angle, command_msg_.gas, elapsed, duration_);
      return;
    }

    // Past the window: hold neutral and latch. Keep publishing afterwards --
    // the supervisor tears the stack down on the flag, but until it does the
    // vehicle interface must not be left without a fresh command.
    command_msg_.steer_angle = kFailsafeSteer;
    command_msg_.gas = kFailsafeGas;
    control_pub_->publish(command_msg_);

    if (!finished_.exchange(true, std::memory_order_relaxed)) {
      RCLCPP_INFO(get_logger(), "Sweep finished after %.1f s. Mission complete.", elapsed);
    }
  }

  // --------------------------------------------------------------- heartbeat

  void publishHeartbeat()
  {
    if (!heartbeat_pub_) {
      return;
    }
    mirena_common::msg::Heartbeat msg;
    msg.timestamp = get_clock()->now();
    msg.node_name = get_name();

    const bool finished = finished_.load(std::memory_order_relaxed);

    nlohmann::json report;

    // "mission_finished" is the exact key the supervisor ends the mission off,
    // spelled exactly as the MPC node spells it -- the supervisor watches one
    // control node and does not care which of the two it got.
    report["mission_finished"] = finished;
    report["mode"] = "inspection";
    report["status_state"] = finished ? "IDLE" : "WORKING";
    report["sweep_elapsed_s"] = elapsed_.load(std::memory_order_relaxed);
    report["sweep_duration_s"] = duration_;

    msg.report = report.dump();
    heartbeat_pub_->publish(msg);
  }

  // ------------------------------------------------------------------- state

  // Parameters, fixed at construction.
  double frequency_;
  double amplitude_;
  double duration_;
  double launch_gas_;
  double launch_time_;
  double hold_gas_;
  double hold_time_;
  double brake_gas_;

  rclcpp::CallbackGroup::SharedPtr sweep_group_;

  rclcpp::Subscription<mirena_common::msg::AsInfo>::SharedPtr as_sub_;

  rclcpp::Publisher<mirena_common::msg::CarControl>::SharedPtr control_pub_;
  rclcpp::Publisher<mirena_common::msg::Heartbeat>::SharedPtr heartbeat_pub_;

  rclcpp::TimerBase::SharedPtr sweep_timer_;
  rclcpp::TimerBase::SharedPtr heartbeat_timer_;

  // Written by the AS callback, read by the sweep loop and the heartbeat.
  std::atomic<int64_t> start_ns_{kUnsetTime};
  std::atomic<double> elapsed_{0.0};
  std::atomic<bool> finished_{false};

  // Reused outgoing message, so the sweep loop does not allocate.
  mirena_common::msg::CarControl command_msg_;
};

}  // namespace mirena

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  std::shared_ptr<mirena::InspectionNode> node;
  try {
    node = std::make_shared<mirena::InspectionNode>();
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("mirena_control"), "Startup failed: %s", e.what());
    rclcpp::shutdown();
    return 1;
  }

  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
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
