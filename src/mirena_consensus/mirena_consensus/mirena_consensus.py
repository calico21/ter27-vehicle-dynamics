import rclpy
from rclpy.node import Node
from rclpy.time import Time
import numpy as np
if not hasattr(np, 'float'):
    np.float = float

from std_msgs.msg import Header
import sensor_msgs.msg as sensor_msgs
from geometry_msgs.msg import TransformStamped
from tf_transformations import euler_from_quaternion, quaternion_from_euler
from tf2_ros import TransformBroadcaster

# Import your custom messages (you'll need to create these Python versions)
from mirena_common.msg import CarControl, Car

# Placeholder imports for your custom modules
from .conversion import Conversion 
from .filters import X_SIZE, U_SIZE, Z_GPS_SIZE, Z_GSS_SIZE, Z_IMU_SIZE, Filters
from .gps_receiver import GpsReceiver, Point2D
from .state import State
from .estimator import Estimator, Measure

# Constants
GPS_SUB_TOPIC = "/car/gps/fix"
IMU_SUB_TOPIC = "/car/imu/data"
GSS_SUB_TOPIC = "/void"
CONTROL_SUB_TOPIC = "/car/inferred_control"
#CONTROL_SUB_TOPIC = "/control/car_control"
CAR_PUB_TOPIC = "/consensus/car_state"

FIXED_FRAME = "odom"
CHILD_FRAME = "car/cog"

NODE_PUBLISH_PERIOD = 0.1  # 100ms
MAX_H_NS_BETWEEN_UPDATES = 1e7  # 10ms
UPDATE_WINDOW_NS = 2e8  # 0.2s

class MirenaConsensusNode(Node):
    def __init__(self):
        super().__init__('mirena_consensus')
        
        # Subscribers
        self._gps_sub = self.create_subscription(
            sensor_msgs.NavSatFix, 
            GPS_SUB_TOPIC, 
            self.gps_callback, 
            10
        )
        self._imu_sub = self.create_subscription(
            sensor_msgs.Imu, 
            IMU_SUB_TOPIC, 
            self.imu_callback, 
            10
        )
        
        self._control_sub = self.create_subscription(
             CarControl, 
             CONTROL_SUB_TOPIC, 
             self.control_callback, 
             10
        )
        # Tranform broadcaster
        self._tf_broadcaster = TransformBroadcaster(self)

        # Publisher
        self._car_pub = self.create_publisher(Car, CAR_PUB_TOPIC, 10)
        
        # Timers
        self._node_publish_timer = self.create_timer(
            NODE_PUBLISH_PERIOD, 
            self.node_publish_callback
        )
        
        self._gps_receiver = GpsReceiver()
        self._filters = Filters()
        self._estimator = Estimator(self._filters, self.get_clock().now().nanoseconds, UPDATE_WINDOW_NS, MAX_H_NS_BETWEEN_UPDATES)

    def gps_callback(self, msg: sensor_msgs.NavSatFix):
        cartesian_point = self._gps_receiver.receive_gps(msg.latitude, msg.longitude)
        if cartesian_point is None:
            if not self._gps_receiver.is_initialized():
                curr_state = self._estimator.get_estimate(self.get_clock().now().nanoseconds)
                if curr_state is None:
                    return
                self._gps_receiver.initialize(
                    msg.latitude, msg.longitude, 
                    curr_state.x, curr_state.y
                )
            return
        _measure = np.array([cartesian_point[0], cartesian_point[1]], dtype=np.float64)
        _measure_time = Time.from_msg(msg.header.stamp).nanoseconds
        self._estimator.insert_measure(_measure_time, Measure(_measure, "gps"))

    def imu_callback(self, msg: sensor_msgs.Imu):
        quaternion = [
            msg.orientation.x,
            msg.orientation.y, 
            msg.orientation.z,
            msg.orientation.w
        ]
        _, _, psi = euler_from_quaternion(quaternion)
        omega = msg.angular_velocity.z
        _measure = np.array([psi, omega], dtype=np.float64)
        _measure_time = Time.from_msg(msg.header.stamp).nanoseconds
        self._estimator.insert_measure(_measure_time, Measure(_measure, "imu"))

    def control_callback(self, msg: CarControl):  # msg: CarControl
        #print("control_uwu")
        self._estimator.insert_control(Time.from_msg(msg.header.stamp).nanoseconds, np.array([msg.gas, msg.steer_angle], dtype=np.float64))


    def publish_transform(self, state):
        """Helper to broadcast the odom -> car/cog transform."""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = FIXED_FRAME
        t.child_frame_id = CHILD_FRAME

        # Translation from state (assuming state has x and y attributes)
        t.transform.translation.x = float(state.x)
        t.transform.translation.y = float(state.y)
        t.transform.translation.z = 0.0

        # Rotation from state yaw (assuming state has psi/yaw attribute)
        # Assuming state.psi is the heading in radians
        q = quaternion_from_euler(0, 0, state.psi)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        self._tf_broadcaster.sendTransform(t)


    def node_publish_callback(self):
            now = self.get_clock().now()
            state = self._estimator.get_estimate(now.nanoseconds)
            
            if state is None: 
                return

            # 1. Publish the custom Car state message
            msg = Conversion.state_to_msg(state)
            msg.child_frame_id = CHILD_FRAME
            msg.header.frame_id = FIXED_FRAME
            msg.header.stamp = now.to_msg()
            self._car_pub.publish(msg)

            # 2. Publish the TF transform
            self.publish_transform(state)

def main(args=None):
    rclpy.init(args=args)
    node = MirenaConsensusNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()