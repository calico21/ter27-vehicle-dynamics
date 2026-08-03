// Inspection controller for the Mirena driverless car.
//
// The static inspection mission: the car is on stands and the steering is swept
// with a sine for a fixed window to demonstrate the actuators. Nothing is
// planned and nothing is solved, so this node replaces the MPC one for the
// length of that mission -- it is the alternative occupant of the same slot in
// the stack, which is why it carries the same default node name, subscribes to
// nothing and publishes the same /control and health topics.
//
// Two loops run on their own callback groups:
//   * 100 Hz sweep      -- publishes the open-loop steering command;
//   * 10 Hz  heartbeat  -- reports liveness and progress to the supervisor.
//
// Like the MPC node, this one is the sole authority on when its run is over: it
// latches mission_finished in the heartbeat once the window closes, and the
// supervisor ends the mission off that flag.

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <memory>

#include <nlohmann/json.hpp>

#include <rclcpp/rclcpp.hpp>

#include "mirena_common/msg/as_info.hpp"
#include "mirena_common/msg/car_control.hpp"
#include "mirena_common/msg/heartbeat.hpp"

namespace mirena
{


class InspectionNode : public rclcpp::Node
{
public:
  InspectionNode()
  : rclcpp::Node("mirena_control")
  {
    // The mission's values arrive as launch arguments from the mission launch
    // file the supervisor started, so they are already in place by the time
    // they are read below.
    declare_parameter<double>("inspection_frequency", 0.5);   // Hz
    declare_parameter<double>("inspection_amplitude", 0.3);   // rad, ~17 deg
    declare_parameter<double>("inspection_duration", 25.0);   // s
    declare_parameter<double>("command_period", 0.01);
    declare_parameter<double>("heartbeat_period", 0.1);

    RCLCPP_INFO(get_logger(), "Starting inspection control node...");

    frequency_ = get_parameter("inspection_frequency").as_double();
    amplitude_ = get_parameter("inspection_amplitude").as_double();
    duration_ = get_parameter("inspection_duration").as_double();
    const double command_period = get_parameter("command_period").as_double();
    const double heartbeat_period = get_parameter("heartbeat_period").as_double();

    RCLCPP_INFO(
      get_logger(), "Inspection mission: %.1f s steering sweep at %.2f Hz, +-%.2f rad.",
      duration_, frequency_, amplitude_);

    // The car's own state machine is what starts the clock. The sweep must be
    // a window of the wheel actually moving, and the wheel does not move until
    // the car reaches DRIVING, so counting from anything earlier -- launch,
    // mission select -- would short the run by however long that took.
    as_sub_ = create_subscription<mirena_common::msg::AsInfo>(
      "/as_status", 10,
      std::bind(&InspectionNode::asCallback, this, std::placeholders::_1));

    control_pub_ = create_publisher<mirena_common::msg::CarControl>("/control", 10);
    heartbeat_pub_ = create_publisher<mirena_common::msg::Heartbeat>("health", 10);

    sweep_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

    heartbeat_timer_ = rclcpp::create_timer(
      this, get_clock(), rclcpp::Duration::from_seconds(heartbeat_period),
      std::bind(&InspectionNode::publishHeartbeat, this));
    sweep_timer_ = rclcpp::create_timer(
      this, get_clock(), rclcpp::Duration::from_seconds(command_period),
      std::bind(&InspectionNode::sweepIteration, this), sweep_group_);

    start_ns_.store(0, std::memory_order_relaxed);
    elapsed_.store(0.0, std::memory_order_relaxed);
    finished_.store(false, std::memory_order_relaxed);
  }

private:
  // ------------------------------------------------------------- sweep loop

  /// Latches the moment the car starts driving; the sweep is timed from there.
  void asCallback(const mirena_common::msg::AsInfo::SharedPtr msg)
  {
    if (msg->status != mirena_common::msg::AsInfo::AS_STATUS_DRIVING) {
      return;
    }
    int64_t unset = 0;
    if (start_ns_.compare_exchange_strong(
        unset, get_clock()->now().nanoseconds(), std::memory_order_relaxed))
    {
      RCLCPP_INFO(get_logger(), "AS is DRIVING: starting the %.1f s sweep.", duration_);
    }
  }

  void sweepIteration()
  {
    const int64_t start_ns = start_ns_.load(std::memory_order_relaxed);
    if (start_ns == 0) {
      return;   // not driving yet; the sweep has not begun
    }

    const auto stamp = get_clock()->now();
    const double elapsed = static_cast<double>(stamp.nanoseconds() - start_ns) * 1e-9;
    elapsed_.store(elapsed, std::memory_order_relaxed);

    command_msg_.header.stamp = stamp;
    if (elapsed <= 4.0){
      command_msg_.gas = 0.24;
    } else if ((elapsed >= 4) && (elapsed <= 20)){
      command_msg_.gas = 0.18;
    } else{
      command_msg_.gas = -1;
    }
    

    if (elapsed <= duration_) {
      command_msg_.steer_angle = amplitude_ * std::sin(2.0 * M_PI * frequency_ * elapsed);
      control_pub_->publish(command_msg_);
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 500, "Sweep active: %.2f rad (%.1f/%.1f s)",
        command_msg_.steer_angle, elapsed, duration_);
      return;
    }

    // Past the window: hold neutral and latch. Keep publishing afterwards --
    // the supervisor tears the stack down on the flag, but until it does the
    // vehicle interface must not be left without a fresh command.
    command_msg_.steer_angle = 0.0;
    command_msg_.gas = -1;
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

  rclcpp::CallbackGroup::SharedPtr sweep_group_;

  rclcpp::Subscription<mirena_common::msg::AsInfo>::SharedPtr as_sub_;

  rclcpp::Publisher<mirena_common::msg::CarControl>::SharedPtr control_pub_;
  rclcpp::Publisher<mirena_common::msg::Heartbeat>::SharedPtr heartbeat_pub_;

  rclcpp::TimerBase::SharedPtr sweep_timer_;
  rclcpp::TimerBase::SharedPtr heartbeat_timer_;

  // Written by the AS callback, read by the sweep loop and the heartbeat.
  std::atomic<int64_t> start_ns_{0};
  std::atomic<double> elapsed_{0.0};
  std::atomic<bool> finished_{false};

  // Reused outgoing message, so the sweep loop does not allocate.
  mirena_common::msg::CarControl command_msg_;
};

}  // namespace mirena

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<mirena::InspectionNode>();

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
