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
                # De-skew: the sweep takes 100 ms and the car covers metres in
                # it, so the points are motion-corrected against /sensors/lidar/imu
                # before they leave the driver. Off means perception sees a
                # smeared cloud at speed. Live-reconfigurable in the node.
                'use_imu': True,
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
                # The VN-300's own IMU runs at 400 Hz; this is what it is asked
                # to emit. 100 Hz is what mirena_state's predict step expects.
                'imu_output_rate_hz': 100,
                # Antenna geometry, in metres, car/cog-relative. These are the
                # only genuinely per-car numbers in this driver -- they say where
                # the antennas are bolted, and the VN-300's onboard filter is
                # wrong without them, so they belong here in the vehicle's own
                # launch file rather than buried as node defaults. Remeasure and
                # update if either antenna moves.
                'antenna_a_offset': [-0.11, 0.0, 0.52],
                'gnss_compass_baseline': [1.50, 0.0, -0.36],
                'gnss_compass_uncertainty': [0.04, 0.04, 0.04],
            }],
        ),

        # -- Vehicle bus ------------------------------------------------------
        # Also the source of /as_status, which is what selects the mission and
        # what the car uses to decide when to act on our commands, and of
        # /sensors/wheel_speeds off the rear WheelInfo frame.
        #
        # can0 must already be up. This node does NOT bring it up: it opens the
        # socket and nothing more, and if the link is down it logs one error and
        # runs inactive -- which looks like a healthy stack that never gets a
        # mission, because /as_status never arrives. Bringing the link up is
        # systemd-networkd's job (or `ip link set can0 up type can bitrate
        # 1000000`). This block used to pass can_bitrate and auto_bringup with a
        # comment claiming the node did it itself; neither parameter was ever
        # declared, so rclcpp discarded both silently and the bitrate here had no
        # effect on anything.
        Node(
            package='mirena_hardware',
            executable='mirena_can',
            name='mirena_can',
            parameters=[{
                'use_sim_time': False,
                'can_interface': 'can0',
                # Record the bus. A session is one file, opened when TER_STATUS
                # reaches PRECHARGED and closed when it returns to WAIT_SL, named
                # for the moment it opened. Format is candump -L, so canplayer
                # and cantools read it directly.
                #
                # This makes the socket run unfiltered (error frames included),
                # which is the point -- but it means the node reads every frame
                # on the bus rather than the two it decodes. Set log False to put
                # the narrow kernel filter back.
                'log': True,
                'log_path': '/home/mirena/logs',
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
                # How long the pipeline gets to come up before its silence is a
                # failed mission. Sized for the slowest node in any mission --
                # control building its acados solver.
                'startup_timeout_sec': 60.0,
                # The pipeline is discovered from whoever publishes /health, so
                # CONFIGURING ends when that set has held still this long.
                'discovery_settle_sec': 2.0,
                # Past CONFIGURING, a mission node quiet for this long is dead
                # and the mission fails. Nodes beat at 10 Hz, so this is 30
                # missed heartbeats.
                'heartbeat_timeout_sec': 3.0,
            }],
        ),
    ])
