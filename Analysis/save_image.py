import rclpy
import rclpy.serialization
import rosbag2_py
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import os
import rosidl_runtime_py.utilities as msg_utils

def extract_images_from_bag(bag_path, image_topic, output_dir="extracted_images"):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Initialize ROS and CvBridge
    rclpy.init()
    reader = rosbag2_py.SequentialReader()
    
    # Open the ROS 2 bag file
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions()
    reader.open(storage_options, converter_options)

    # Get topic types
    topic_types = reader.get_all_topics_and_types()
    type_map = {topic.name: topic.type for topic in topic_types}

    bridge = CvBridge()
    counter = 0  # Image counter

    # Read messages
    while reader.has_next():
        (topic, data, timestamp) = reader.read_next()
        
        if topic == image_topic:
            try:
                image_msg = rclpy.serialization.deserialize_message(data, Image)
                cv_image = bridge.imgmsg_to_cv2(image_msg)
                filename = os.path.join(output_dir, f'image_{counter:04d}.png')
                cv2.imwrite(filename, cv_image)
                print(f"Saved {filename}")
                counter += 1
            except Exception as e:
                print(f"Failed to process image: {e}")

    print(f"Extracted {counter} images to {output_dir}")
    rclpy.shutdown()

if __name__ == "__main__":
    bag_path = "fls_bag/20241210110455/20241210110455.db3"  # Change this to your ROS bag file path
    image_topic = "/oculus_sonar/ping_image"
    output_dirs = "fls_data"      # Change this to your image topic

    extract_images_from_bag(bag_path, image_topic, output_dirs)