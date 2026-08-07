"""
inspection.launch.py -- the INSPECTION mission, start to finish.

The car sits on stands, the steering is swept and the wheels are spun to a
target speed, to demonstrate that both actuators respond. There is no track,
nothing to perceive, nothing to map and nothing to plan, so this mission
launches exactly one node: the inspection controller. That is the whole point of
a launch file per mission -- an inspection run costs one process, not a pipeline
sitting idle behind it.

The drivetrain runs closed-loop on the CAN bridge's wheel speeds, so the one
thing this mission does depend on is /sensors/wheel_speeds arriving. Without it
the controller brakes rather than spinning the wheels open-loop, reports
NOT_WORKING, and still completes its steering sweep and ends the mission.

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

                # The drivetrain half of the demonstration. The wheels are held
                # at this speed by a PID closing on the mean of the two rear
                # wheel speeds off the CAN bus, so this is a speed that is
                # reached rather than a throttle opening that is hoped for.
                # inspection_gas_max is the loop's authority, and is what limits
                # the reachable speed -- raise it if the target cannot be met.
                'inspection_target_rpm': 200.0,
                'inspection_gas_max': 0.5,
            }],
        ),
    ])
