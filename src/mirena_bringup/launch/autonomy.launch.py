from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import UnlessCondition # ── Added ──
from launch_ros.actions import LifecycleNode
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')

    # Capture the forwarded ground truth parameters
    gt_state   = LaunchConfiguration('ground_truth_state')
    gt_percep  = LaunchConfiguration('ground_truth_perception')
    gt_slam    = LaunchConfiguration('ground_truth_slam')

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('ground_truth_state', default_value='false'),
        DeclareLaunchArgument('ground_truth_perception', default_value='false'),
        DeclareLaunchArgument('ground_truth_slam', default_value='false'),

        # ── State Estimation ────────────────────────────────────────────────────
        # Always runs: owns /state/car and the odom->car/cog TF in every mode.
        # With ground_truth_state=true it republishes the sim's ground truth
        # (/debug/state/car) instead of running the EKF.
        LifecycleNode(
            package='mirena_state',
            executable='mirena_state',
            name='mirena_state',
            namespace='',
            parameters=[{
                'use_sim_time': use_sim_time,
                'ground_truth_passthrough': ParameterValue(gt_state, value_type=bool),
            }],
        ),

        # ── Perception ─────────────────────────────────────────────────────────
        # Only run if we are NOT using ground truth perception
        LifecycleNode(
            package='mirena_perception',
            executable='mirena_perception',
            name='mirena_perception',
            namespace='',
            condition=UnlessCondition(gt_percep),
            parameters=[{'use_sim_time': use_sim_time}],
            
        ),

        # ── SLAM ────────────────────────────────────────────────────────────────
        # Only run if we are NOT bypassing slam entirely
        LifecycleNode(
            package='mirena_slam',
            executable='mirena_slam',
            name='mirena_slam',
            namespace='',
            condition=UnlessCondition(gt_slam),
            parameters=[{'use_sim_time': use_sim_time}],
        ),

        # ── Planning ────────────────────────────────────────────────────────────
        LifecycleNode(
            package='mirena_planning',
            executable='mirena_planning',
            name='mirena_planning',
            namespace='',
            parameters=[{'use_sim_time': use_sim_time}],
        ),

        # ── Control ─────────────────────────────────────────────────────────────
        LifecycleNode(
            package='mirena_control',
            executable='mirena_control',
            name='mirena_control',
            namespace='',
            parameters=[{'use_sim_time': use_sim_time}],
        ),

        # ── Supervisor ──────────────────────────────────────────────────────────
        LifecycleNode(
            package='mirena_supervisor',
            executable='mirena_supervisor',
            name='mirena_supervisor',
            namespace='',
            parameters=[{
                'use_sim_time': use_sim_time,
                'nodes_to_monitor': ['mirena_state', 'mirena_slam', 'mirena_planning', 'mirena_control','mirena_perception'] 
            }],
        ),
    ])