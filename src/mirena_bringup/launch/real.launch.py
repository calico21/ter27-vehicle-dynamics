"""
real.launch.py — hardware drivers + autonomy stack on the TeR vehicle.

  ros2 launch mirena_bringup real.launch.py mission:=trackdrive

Override hardware params with a params file:
  ros2 launch mirena_bringup real.launch.py \
    mission:=trackdrive params:=/path/to/custom.yaml
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    hw_config = os.path.join(
        get_package_share_directory('mirena_hardware'),
        'config',
        'TeR.yaml'
    )

    bringup_share = get_package_share_directory('mirena_bringup')

    autonomy = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([FindPackageShare('mirena_bringup'), 'launch', 'autonomy.launch.py'])
        ]),
        launch_arguments={
            'use_sim_time': 'false',
        }.items(),
    )

    return LaunchDescription([

        # ── Lidar ───────────────────────────────────────────────────────────────
        Node(
            package='mirena_hardware',
            executable='mirena_lidar',
            name='mirena_lidar',
            parameters=[hw_config],
        ),

        # ── Camera ──────────────────────────────────────────────────────────────
        Node(
            package='mirena_hardware',
            executable='mirena_camera',
            name='mirena_camera',
            parameters=[hw_config],
        ),

        # ── IMU ─────────────────────────────────────────────────────────────────
        Node(
            package='mirena_hardware',
            executable='mirena_imu',
            name='mirena_imu',
            parameters=[hw_config],
        ),



        # ── CAN Bus ─────────────────────────────────────────────────────────────
        Node(
            package='mirena_hardware',
            executable='mirena_can',
            name='mirena_can',
            parameters=[hw_config],
        ),
        
        Node(
            package='mirena_hardware',
            executable='mirena_transform',
            name='mirena_transform',
            parameters=[hw_config],
        ),

        # ── Autonomy pipeline ───────────────────────────────────────────────────
        autonomy,
    ])
