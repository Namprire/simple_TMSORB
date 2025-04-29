from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='mono_slam',
            executable='slam_node',
            name='slam_node',
            output='screen',
        ),
        Node(
            package='mono_slam',
            executable='trajectory_logger_node',
            name='trajectory_logger',
            output='screen',
        ),
    ])
