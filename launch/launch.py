import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    ld = LaunchDescription()
    config = os.path.join(
        get_package_share_directory('cfar_cpp'),
        'cfg',
        'config.yaml'
        )
        
    node=Node(
        package = 'cfar_cpp',
        name = 'cfar_node',
        executable = 'cfar_node',
        parameters = [config]
    )
    ld.add_action(node)
    return ld