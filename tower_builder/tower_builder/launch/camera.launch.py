"""
카메라 노드 Launch 파일
ros2 launch tower_builder camera.launch.py
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='tower_builder',
            executable='camera_node',
            name='camera_node',
            output='screen',
            emulate_tty=True,  # 터미널 입력 허용
        ),
    ])
