"""
acceleration.launch.py -- the ACCELERATION mission, start to finish.

This file *is* the mission definition: which nodes run and with what
parameters. It replaces mission_config/ACCELERATION.yaml, whose contents are now
the parameters below. mirena_supervisor starts it through the launch API when
the AS selects this mission, and shuts it down when the run ends, so the whole
pipeline lives exactly as long as the mission does.

Nothing here is a lifecycle node: a node's lifetime is the mission's, so
"configured but inactive" has nowhere left to live. The supervisor starts this
file on AS READY and holds the mission in CONFIGURING until every node below is
publishing a heartbeat. Commands go out from the moment there is a solution --
the car is what decides whether to act on them, and it does not until its own
state machine reaches DRIVING.

use_sim_time is the only argument, forwarded by the supervisor from whichever
top-level launch file started it.

  ros2 launch mirena_supervisor acceleration.launch.py use_sim_time:=true
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),

        # -- State estimation ------------------------------------------------
        # Owns /state/car and the odom->car/cog TF.
        Node(
            package='mirena_state',
            executable='mirena_state',
            name='mirena_state',
            parameters=[{'use_sim_time': use_sim_time}],
        ),

        # -- Perception ------------------------------------------------------
        Node(
            package='mirena_perception',
            executable='mirena_perception',
            name='mirena_perception',
            parameters=[{'use_sim_time': use_sim_time}],
        ),

        # -- SLAM ------------------------------------------------------------
        Node(
            package='mirena_slam',
            executable='mirena_slam',
            name='mirena_slam',
            parameters=[{'use_sim_time': use_sim_time}],
        ),

        # -- Planning --------------------------------------------------------
        # Surveyed layout: the known planner registers resource/acceleration.json.
        Node(
            package='mirena_planning',
            executable='mirena_planning_known',
            name='mirena_planning',
            parameters=[{
                'use_sim_time': use_sim_time,
                'mission': 'acceleration',
            }],
        ),

        # -- Control ---------------------------------------------------------
        # laps is what mission_config carried: 0 selects open-track behaviour -- stop at the end of the straight.
        Node(
            package='mirena_control',
            executable='mirena_control',
            name='mirena_control',
            parameters=[{
                'use_sim_time': use_sim_time,
                'laps': 0,
            }],
        ),
    ])
