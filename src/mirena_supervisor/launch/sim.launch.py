import os
from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Ruta relativa al paquete instalado
    rviz_config_file = PathJoinSubstitution([
        FindPackageShare('mirena_supervisor'),
        'config',
        'sim.rviz'
    ])

    return LaunchDescription([
        # -- Data source ------------------------------------------------------
        Node(
            package='mirena_sim',
            executable='mirena_sim',
            parameters=[{'use_sim_time': True}],
        ),

        # -- Supervisor -------------------------------------------------------
        # Runs for the life of the session; the pipeline comes and goes beneath
        # it, one launch file per mission.
        Node(
            package='mirena_supervisor',
            executable='mirena_supervisor',
            name='mirena_supervisor',
            parameters=[{
                'use_sim_time': True,
                'mission_launch_package': 'mirena_supervisor',
                'mission_launch_directory': 'launch/missions',
                'mission_launch_arguments': ['use_sim_time:=true'],
            }],
        ),

        # -- State EKF --------------------------------------------------------
        Node(
            package='mirena_state',
            executable='mirena_state_ekf',
            parameters=[{'use_encoders': True}],
        ),

        # -- RViz2 ------------------------------------------------------------
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config_file],
            parameters=[{'use_sim_time': True}],
            output='screen',
        ),
    ])