#!/usr/bin/env python3
"""ROS 2 Edge Inference Node for YOLOv8 Package Integrity Classification.

This node subscribes to a standard `sensor_msgs/Image` topic, performs
object detection using an exported YOLOv8 TensorRT engine (`best.onnx`), 
and publishes both the raw detections (`vision_msgs/Detection2DArray`) 
and an annotated image block for easy visualization.
"""

import os
import cv2
import numpy as np
from typing import List

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from vision_msgs.msg import BoundingBox2D, Detection2D, Detection2DArray, ObjectHypothesisWithPose

# Dynamically loaded at runtime to avoid heavy imports on startup if failing
from ultralytics import YOLO


class YOLOInferenceNode(Node):
    """ROS 2 wrapper for Ultralytics YOLOv8 Edge Inference."""

    def __init__(self):
        super().__init__('yolo_inference_node')

        # Declare parameters
        self.declare_parameter('model_path', 'best.onnx')
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('confidence_threshold', 0.5)
        
        # Load parameters
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self._conf_thresh = self.get_parameter('confidence_threshold').get_parameter_value().double_value

        self.get_logger().info(f"Loading YOLO engine from: {model_path}")
        if not os.path.exists(model_path):
            self.get_logger().error(f"Model file not found at {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Initialize YOLO
        # Note: If passing an ONNX file on a Jetson, ultralytics automatically leverages TensorRT
        self._model = YOLO(model_path, task='detect')
        
        # We need cv_bridge to convert ROS Image -> OpenCV Mat
        self._bridge = CvBridge()

        # QoS Settings (Best Effort for high-fps camera streams)
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscribers
        self.get_logger().info(f"Subscribing to image topic: {image_topic}")
        self._sub_image = self.create_subscription(
            Image,
            image_topic,
            self._image_callback,
            qos_profile
        )

        # Publishers
        self._pub_detections = self.create_publisher(
            Detection2DArray,
            '~/detections',
            10
        )
        self._pub_annotated = self.create_publisher(
            Image,
            '~/annotated_image',
            10
        )

        self.get_logger().info("YOLO Inference Node initialized and ready.")

    def _image_callback(self, msg: Image) -> None:
        """Process incoming image, run inference, and publish results."""
        try:
            # Convert ROS Image to CV2 format (BGR8 is standard for OpenCV/YOLO)
            cv_image = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        # Run inference (imgsz=640 is standard from our Phase 3 export)
        # Using half=True expects the tensorrt engine to use FP16
        results = self._model(cv_image, conf=self._conf_thresh, imgsz=640, verbose=False)
        
        # We expect exactly 1 result since we passed 1 image
        if not results:
            return
            
        result = results[0]

        # 1. Publish raw vision_msgs
        detection_array_msg = self._build_detection_array(result, msg.header)
        self._pub_detections.publish(detection_array_msg)

        # 2. Publish annotated image for debugging/visualization
        if self._pub_annotated.get_subscription_count() > 0:
            annotated_frame = result.plot()
            try:
                annotated_msg = self._bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
                annotated_msg.header = msg.header
                self._pub_annotated.publish(annotated_msg)
            except Exception as e:
                self.get_logger().error(f"Failed to publish annotated image: {e}")

    def _build_detection_array(self, result, header) -> Detection2DArray:
        """Construct standard ROS 2 vision_msgs from YOLO results."""
        msg = Detection2DArray()
        msg.header = header

        if result.boxes is None or len(result.boxes) == 0:
            return msg

        # Parse boxes (format: xywh = x_center, y_center, width, height)
        boxes_xywh = result.boxes.xywh.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()
        class_names = result.names

        for box, cls, conf in zip(boxes_xywh, classes, confidences):
            detection = Detection2D()
            detection.header = header
            
            # Setup Bounding Box (vision_msgs standard uses center + size)
            x_c, y_c, w, h = box
            
            bbox = BoundingBox2D()
            # Pose2D is expected for the center
            bbox.center.position.x = float(x_c)
            bbox.center.position.y = float(y_c)
            bbox.center.theta = 0.0 # No rotation in standard YOLO output
            
            bbox.size_x = float(w)
            bbox.size_y = float(h)
            
            detection.bbox = bbox

            # Setup Hypothesis (Class + Confidence)
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = class_names[cls]
            hyp.hypothesis.score = float(conf)
            
            detection.results.append(hyp)
            msg.detections.append(detection)

        return msg


def main(args=None):
    rclpy.init(args=args)
    try:
        node = YOLOInferenceNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Node initialization failed: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
