import rclpy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from rclpy.node import Node
from mirena_common.msg import EntityList, Track, Heartbeat
from ament_index_python.packages import get_package_share_directory
from mirena_planning.planners import UnknownPlanner, KnownPlanner, BasePlanner
import json
import os
import sys

SLAM_MAP_TOPIC = 'slam/map'
PLANNER_TRACK_TOPIC = 'planning/track'

class PlannerNode(LifecycleNode): 
    def __init__(self):
        super().__init__('mirena_planning')
        # Publishers and subscribers
        self.map_sub = None
        self.track_pub = None
        # Heartbeat
        self.report_pub = None
        self.report_timer = None
        # Planning Strategy
        self.planner: BasePlanner = None # Function that takes cones in and returns track
        self.declare_parameter('mission', 'trackdrive')
        self._last_update_time = self.get_clock().now()
        self._last_update_start_point = None
        self._last_update_end_point = None
        

    def on_configure(self, state):
        self.get_logger().info(f"Configuring Planner. Mission selected: {self.get_parameter('mission').get_parameter_value().string_value}")
        #Pub and sub
        self.map_sub = self.create_subscription(EntityList, SLAM_MAP_TOPIC, self.map_callback, 10)
        self.track_pub = self.create_lifecycle_publisher(Track, PLANNER_TRACK_TOPIC, 10)
        # Heartbeat timer and pub
        self.report_pub = self.create_lifecycle_publisher(Heartbeat, "health", 10)
        self.report_timer = self.create_timer(0.1, self.publish_heartbeat)
        # Configure planning strategy
        mission = self.get_parameter('mission').get_parameter_value().string_value

        if mission in ["skidpad", "acceleration"]:
            self.planner = KnownPlanner(track_path=os.path.join(get_package_share_directory('mirena_planning'), 'resource', f'{mission}.json'))
        elif mission in ["trackdrive", "autocross"]:
            self.planner = UnknownPlanner()
        else:
            self.get_logger().error(f"Mission '{mission}' not supported")
            return TransitionCallbackReturn.FAILURE
        
        
        return TransitionCallbackReturn.SUCCESS


    def on_activate(self, state):
        self.get_logger().info("Activating Planner.")
        return super().on_activate(state)

    def on_deactivate(self, state):
        self.get_logger().info("Deactivating Planner.")
        return super().on_deactivate(state)

    def on_cleanup(self, state):
        self.get_logger().info("Cleaning up Planner.")
        self.destroy_publisher(self.track_pub)
        self.destroy_timer(self.report_timer)
        self.destroy_publisher(self.report_pub)
        return TransitionCallbackReturn.SUCCESS
    

    def publish_heartbeat(self):
        if self.report_pub is not None and self.report_pub.is_activated:
            msg = Heartbeat()
            msg.node_name = self.get_name()
            msg.timestamp = self.get_clock().now().to_msg()

            # Serialize the start/finish gate as plain numbers so the supervisor can
            # parse it from JSON (a Vector3 message is not JSON-serializable). Null
            # until the planner has produced its first plan.
            start_point = self._last_update_start_point
            report_dict = {
                "start_point": (
                    {"x": start_point.x, "y": start_point.y, "z": start_point.z}
                    if start_point is not None else None
                ),
            }
            msg.report = json.dumps(report_dict)
            self.report_pub.publish(msg)


    def map_callback(self, msg):
        if self.planner is not None:
            plan = self.planner.plan(msg) 
            if len(msg.entities) > 2:# Pass entities to planner and return track
                if plan is not None:
                    track = plan.track
                    # Header of track
                    track.header.frame_id = "map"
                    track.header.stamp = self.get_clock().now().to_msg()
                    # Publish track
                    self.track_pub.publish(track)

                    self._last_update_time = self.get_clock().now()
                    self._last_update_start_point = plan.start_point
                    self._last_update_end_point = plan.end_point


def main(args=None):
    rclpy.init(args=args)
    node = PlannerNode()   
    
    autostart_flag = False
    for i in range(1, len(sys.argv)):
        if sys.argv[i] == "-a":
            autostart_flag = True
            break

    if autostart_flag:
        node.get_logger().info("Autoactivating")
        node.trigger_configure()
        node.trigger_activate()
    else:
        node.get_logger().info("Awaiting activation. Use argument '-a' when running for autoactivation")

    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
