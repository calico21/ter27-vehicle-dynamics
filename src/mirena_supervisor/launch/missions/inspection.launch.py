"""
inspection.launch.py -- the INSPECTION mission, start to finish.

The car sits on stands and the steering is swept to demonstrate the actuators.
There is no track, nothing to perceive, nothing to map and nothing to plan, so
this mission launches exactly one node: the inspection controller. That is the
whole point of a launch file per mission -- an inspection run costs one process,
not a pipeline sitting idle behind it.

It replaces mission_config/INSPECTION.yaml, whose contents are the parameters
below. mirena_supervisor starts this through the launch API when the AS selects
INSPECTION and shuts it down when the sweep reports itself finished.

The sweep is timed from AS DRIVING, not from launch: the node subscribes to
/as_status and starts its clock there, because the seconds the rules ask for are
seconds of the wheel actually moving.

  ros2 launch mirena_supervisor inspection.launch.py use_sim_time:=true
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),

        # -- Control, and nothing else ---------------------------------------
        Node(
            package='mirena_control',
            executable='mirena_control_inspection',
            name='mirena_control',
            parameters=[{
                'use_sim_time': use_sim_time,
                'inspection_duration': 25.0,    # s of sweep before the run is done
                'inspection_frequency': 0.5,    # Hz
                'inspection_amplitude': 0.3,    # rad, ~17 deg at the wheel
            }],
        ),
    ])
