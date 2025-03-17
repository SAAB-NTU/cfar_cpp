import rclpy
from rclpy.node import Node
import cv2
import os
import glob
import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import argparse


class ImagePublisher(Node):
    def __init__(self, image_folder, frequency):
        super().__init__('image_publisher')
        self.publisher = self.create_publisher(Image, '/oculus_sonar/ping_image', 10)
        self.bridge = CvBridge()
        self.image_paths = sorted(glob.glob(os.path.join(image_folder, "*.*")))  # Load images
        self.frequency = frequency
        self.timer = self.create_timer(1.0 / frequency, self.publish_image)
        self.index = 0
        self.get_logger().info(f"Loaded {len(self.image_paths)} images. Publishing at {frequency} Hz.")

    def publish_image(self):
        if not self.image_paths:
            self.get_logger().warn("No images found in the folder.")
            return
        
        img_path = self.image_paths[self.index]
        image = cv2.imread(img_path)
        
        if image is None:
            self.get_logger().warn(f"Failed to load image: {img_path}")
        else:
            msg = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
            self.publisher.publish(msg)
            self.get_logger().info(f"Published {img_path}")

        self.index = (self.index + 1) % len(self.image_paths)  # Loop back to first image


def main(args=None):
    parser = argparse.ArgumentParser(description="ROS2 Image Publisher")
    parser.add_argument("image_folder", type=str, help="Path to folder containing images")
    parser.add_argument("frequency", type=float, help="Publishing frequency (Hz)")
    parsed_args = parser.parse_args()

    rclpy.init(args=args)
    node = ImagePublisher(parsed_args.image_folder, parsed_args.frequency)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
