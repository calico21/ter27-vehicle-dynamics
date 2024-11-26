from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

child_launch_file = FindPackageShare('mirena_sim').find('mirena_sim') + '/launch/mirena_sim_launch.py'

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='mirena_examples',
            executable='remo_con',
            name='remo_con'
        ),
            IncludeLaunchDescription(
            PythonLaunchDescriptionSource(child_launch_file),
        ),
    ])