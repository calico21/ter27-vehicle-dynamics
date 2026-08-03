"""
hardware.launch.py -- the whole system, on the TeR vehicle.

  ros2 launch mirena_supervisor hardware.launch.py

Starts the sensor and actuator drivers, and the supervisor. The autonomy
pipeline is deliberately *not* here. The supervisor brings it up when the AS
selects a mission -- launching launch/missions/<mission>.launch.py on AS READY
and holding the mission in CONFIGURING until every node in it is beating -- and
tears it down when the run ends. So the car can do inspection, then skidpad,
then trackdrive without anything being restarted between them.

The drivers' parameters live below rather than in a YAML file: this is one
vehicle's configuration, and a launch file that says both what it starts and how
is easier to follow than one that points somewhere else for half the answer. The
device paths are the udev symlinks from mirena_hardware/config/99-mirena.rules.

Its counterpart is sim.launch.py, which is the same file with the simulator
where the drivers are.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # Intrinsics from the last calibration, installed with mirena_hardware.
    camera_info_url = 'file://' + os.path.join(
        get_package_share_directory('mirena_hardware'), 'config', 'camera_info.yaml')

    return LaunchDescription([
        # -- Static transforms ------------------------------------------------
        # First: every driver and every mission resolves frames against car/cog,
        # and the tree only exists once this has published /tf_static. Its
        # transforms.yaml stays a file -- that is a geometry table rather than
        # settings, and the node finds it through the ament index on its own.
        Node(
            package='mirena_hardware',
            executable='mirena_transform',
            name='mirena_transform',
            parameters=[{'use_sim_time': False}],
        ),

        # -- Sensors ----------------------------------------------------------
        Node(
            package='mirena_hardware',
            executable='mirena_lidar',
            name='mirena_lidar',
            parameters=[{
                'use_sim_time': False,
                'self_ip': '192.168.1.102',
                'lidar_ip': '192.168.1.200',
                'msop_port': 6699,
                'difop_port': 7788,
            }],
        ),

        Node(
            package='mirena_hardware',
            executable='mirena_camera',
            name='mirena_camera',
            parameters=[{
                'use_sim_time': False,
                'video_device': '/dev/ter_camera_video',
                'image_width': 640,
                'image_height': 480,
                'fps': 90,
                'frame_id': 'camera_optical_frame',
                'camera_name': 'mirena_camera',
                'camera_info_url': camera_info_url,
            }],
        ),

        Node(
            package='mirena_hardware',
            executable='mirena_imu',
            name='mirena_imu',
            parameters=[{
                'use_sim_time': False,
                'imu_device': '/dev/ter_IMU',
            }],
        ),

        # -- Vehicle bus ------------------------------------------------------
        # Also the source of /as_status, which is what selects the mission and
        # what the car uses to decide when to act on our commands.
        Node(
            package='mirena_hardware',
            executable='mirena_can',
            name='mirena_can',
            parameters=[{
                'use_sim_time': False,
                'can_interface': 'can0',
                'can_bitrate': 500000,
                # The node runs `ip link set can0 up type can bitrate ...` itself.
                'auto_bringup': True,
            }],
        ),

        # -- Supervisor -------------------------------------------------------
        # Runs for the life of the session; the pipeline comes and goes beneath
        # it, one launch file per mission.
        Node(
            package='mirena_supervisor',
            executable='mirena_supervisor',
            name='mirena_supervisor',
            parameters=[{
                'use_sim_time': False,
                'mission_launch_package': 'mirena_supervisor',
                'mission_launch_directory': 'launch/missions',
                'mission_launch_arguments': ['use_sim_time:=false'],
            }],
        ),
    ])
