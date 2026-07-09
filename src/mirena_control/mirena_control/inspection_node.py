import numpy as np
import rclpy
from rclpy.node import Node
from mirena_common.msg import CarControl

class SteeringSineNode(Node):
    def __init__(self):
        super().__init__('steering_sine_node')

    # Publisher for control commands
        self.control_pub = self.create_publisher(CarControl, 'control', 10)

        # Timing and state
        self.start_time = self.get_clock().now()
        self.frequency = 0.5  # Hz (one full oscillation every 2 seconds)
        self.amplitude = 0.3  # Radians (~17 degrees)
        
        # Timer to run at 100Hz (consistent with your original 0.01s timer)
        self.timer = self.create_timer(0.01, self.timer_callback)
        
        self.get_logger().info("Dummy Sine Node Started. Waiting for 20s mark...")

    def timer_callback(self):
        # Calculate elapsed time in seconds
        now = self.get_clock().now()
        elapsed = (now - self.start_time).nanoseconds / 1e9

        msg = CarControl()
        msg.header.stamp = now.to_msg()
        
        # Default state: Neutral
        msg.gas = 0.0
        msg.steer_angle = 0.0

        if elapsed <= 25.0:
            # Sine wave movement
            # steer = A * sin(2 * pi * f * t)
            # We subtract 20 from elapsed to start the sine wave at 0 amplitude
            msg.steer_angle = self.amplitude * np.sin(2 * np.pi * self.frequency * elapsed)

            self.get_logger().info(f"Sine Active: {msg.steer_angle:.2f} rad", throttle_duration_sec=0.5)

        elif elapsed > 25.0:
            # Final stop log
            self.get_logger().info("30s reached. Stopping movement.", throttle_duration_sec=10.0)
            rclpy.shutdown()
            # Leave gas and steer at 0.0

        self.control_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = SteeringSineNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Send one final zero-command for safety
        stop_msg = CarControl(gas=0.0, steer_angle=0.0)
        node.control_pub.publish(stop_msg)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()