// Open-loop steering sweep used for the inspection mission: sends a sine on the
// steering wheel for a fixed window, then holds neutral and exits.

#include <chrono>
#include <cmath>
#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "mirena_common/msg/car_control.hpp"

namespace mirena
{

class SteeringSineNode : public rclcpp::Node
{
public:
  SteeringSineNode()
  : rclcpp::Node("steering_sine_node")
  {
    frequency_ = declare_parameter<double>("frequency", 0.5);      // Hz
    amplitude_ = declare_parameter<double>("amplitude", 0.3);      // rad, ~17 deg
    duration_ = declare_parameter<double>("duration", 25.0);       // s

    control_pub_ = create_publisher<mirena_common::msg::CarControl>("control", 10);
    start_time_ = now();

    timer_ = rclcpp::create_timer(
      this, get_clock(), rclcpp::Duration::from_seconds(0.01),
      std::bind(&SteeringSineNode::tick, this));

    RCLCPP_INFO(get_logger(), "Steering sine node started (%.1f s sweep).", duration_);
  }

  void publishStop()
  {
    mirena_common::msg::CarControl msg;
    msg.header.stamp = now();
    msg.gas = 0.0;
    msg.steer_angle = 0.0;
    control_pub_->publish(msg);
  }

private:
  void tick()
  {
    const auto stamp = now();
    const double elapsed = (stamp - start_time_).seconds();

    mirena_common::msg::CarControl msg;
    msg.header.stamp = stamp;
    msg.gas = 0.0;
    msg.steer_angle = 0.0;

    if (elapsed <= duration_) {
      msg.steer_angle = amplitude_ * std::sin(2.0 * M_PI * frequency_ * elapsed);
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 500, "Sine active: %.2f rad", msg.steer_angle);
    } else {
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 10000, "Sweep finished; holding neutral.");
      control_pub_->publish(msg);
      rclcpp::shutdown();
      return;
    }

    control_pub_->publish(msg);
  }

  rclcpp::Publisher<mirena_common::msg::CarControl>::SharedPtr control_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Time start_time_;
  double frequency_;
  double amplitude_;
  double duration_;
};

}  // namespace mirena

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<mirena::SteeringSineNode>();
  rclcpp::spin(node);
  node->publishStop();
  rclcpp::shutdown();
  return 0;
}
