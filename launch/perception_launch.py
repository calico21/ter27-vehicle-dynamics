import os
import sys
import lifecycle_msgs.msg
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node, LifecycleNode
from launch.actions import EmitEvent, RegisterEventHandler
from launch_ros.events.lifecycle import ChangeState
from launch_ros.event_handlers import OnStateTransition

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def generate_launch_description():
    print(os.getcwd())
    #1. Setup Paths
    hw_config = os.path.join(
        get_package_share_directory('mirena_hardware'),
        'config', 'camera.yaml'
    )

    # 2. Define the Perception Lifecycle Node
    # Explicitly using keyword arguments to satisfy the 'namespace' requirement
    perception_node = LifecycleNode(
        package='mirena_perception',
        executable='mirena_perception',
        name='mirena_perception',
        namespace='',  # <--- This must be a keyword argument
        output='screen',
        arguments = ['-a']
    )

    camera_node = LifecycleNode(
        package='mirena_hardware',
        executable='mirena_camera',
        name = 'mirena_camera',
        namespace='',
        parameters=[hw_config],
        output='screen',
        arguments = ['-a']
    )

    lidar_node = LifecycleNode(
        package='mirena_hardware',
        executable='mirena_lidar',
        name = 'mirena_lidar',
        namespace='',
        output='screen',
        arguments = ['-a']
    )
    transform_node = LifecycleNode(
        package='mirena_hardware',
        executable='mirena_transform',
        name = 'mirena_transform',
        namespace='',
        parameters=[hw_config],
        output='screen',
        arguments = ['-a']
    )



    return LaunchDescription([
        transform_node,
        camera_node,
        lidar_node,
        perception_node
       
    ])