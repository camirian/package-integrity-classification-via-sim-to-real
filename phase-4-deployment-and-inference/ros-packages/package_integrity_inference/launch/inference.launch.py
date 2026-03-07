import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    """Launch the YOLOv8 Inference Node and a V4L2 USB camera node."""

    # 1. Declare Launch Arguments
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='best.onnx',
        description='Absolute path to the exported YOLOv8 ONNX or TensorRT engine'
    )

    image_topic_arg = DeclareLaunchArgument(
        'image_topic',
        default_value='/image_raw',
        description='The ROS 2 image topic to subscribe to'
    )

    conf_thresh_arg = DeclareLaunchArgument(
        'conf_thresh',
        default_value='0.5',
        description='Confidence threshold for bounding box detection'
    )

    camera_id_arg = DeclareLaunchArgument(
        'camera_id',
        default_value='0',
        description='USB Camera ID (e.g. 0 for /dev/video0)'
    )

    # 2. Define Nodes

    # Standard USB Webcam driver node
    # Assumes `sudo apt install ros-humble-v4l2-camera`
    v4l2_camera_node = Node(
        package='v4l2_camera',
        executable='v4l2_camera_node',
        name='v4l2_camera',
        namespace='',
        parameters=[{
            'video_device': ['/dev/video', LaunchConfiguration('camera_id')],
            'image_size': [640, 480],
            'pixel_format': 'YUYV' # standard for most cheap webcams
        }],
        remappings=[
            ('/image_raw', LaunchConfiguration('image_topic'))
        ]
    )

    # Our custom YOLO inference node
    yolo_node = Node(
        package='package_integrity_inference',
        executable='yolo_inference_node',
        name='yolo_inference_node',
        output='screen',
        parameters=[{
            'model_path': LaunchConfiguration('model_path'),
            'image_topic': LaunchConfiguration('image_topic'),
            'confidence_threshold': LaunchConfiguration('conf_thresh')
        }]
    )

    # 3. Create Launch Description
    ld = LaunchDescription()
    
    ld.add_action(model_path_arg)
    ld.add_action(image_topic_arg)
    ld.add_action(conf_thresh_arg)
    ld.add_action(camera_id_arg)

    ld.add_action(v4l2_camera_node)
    ld.add_action(yolo_node)

    return ld
