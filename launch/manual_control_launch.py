from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='mirena_examples',
            executable='remo_con',
            name='remo_con'
        ),
        Node(
            package='mirena_sim',
            executable='MirenaSim',
            name='MirenaSim'
        ),
    ])