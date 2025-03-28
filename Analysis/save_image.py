import rclpy
import rclpy.serialization
import rosbag2_py
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import os
import rosidl_runtime_py.utilities as msg_utils
import matplotlib.pyplot as plt
import numpy as np

def extract_images_from_bag(bag_path, image_topic1, image_topic2, output_dir="extracted_images"):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Initialize ROS and CvBridge
    # rclpy.init()
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
        
        if topic == image_topic1:
            try:
                image_msg = rclpy.serialization.deserialize_message(data, Image)
                # print(type(image_msg.data))
                np_image = np.frombuffer(image_msg.data, dtype=np.uint8)
                np_image = np_image.reshape((image_msg.height, image_msg.width, -1))  # Reshape based on dimensions
                # cv_image = bridge.imgmsg_to_cv2(image_msg)
                # cv_image_flip =cv2.flip(cv_image, -1)
                
                filename = os.path.join(output_dir, f'image_{counter:04d}_1.png')
                cv2.imwrite(filename, np_image)
                print(f"Saved {filename}")
                # counter += 1
                
                # return cv_image
            except Exception as e:
                print(f"Failed to process image: {e}")
        elif topic == image_topic2:
            try:
                image_msg = rclpy.serialization.deserialize_message(data, Image)
                # print(type(image_msg.data))
                np_image = np.frombuffer(image_msg.data, dtype=np.uint8)
                np_image = np_image.reshape((image_msg.height, image_msg.width, -1))  # Reshape based on dimensions
                # cv_image = bridge.imgmsg_to_cv2(image_msg)
                # cv_image_flip =cv2.flip(cv_image, -1)
                
                filename = os.path.join(output_dir, f'image_{counter:04d}_0.png')
                cv2.imwrite(filename, np_image)
                print(f"Saved {filename}")
                # counter += 1
                
                # return cv_image
            except Exception as e:
                print(f"Failed to process image: {e}")
            counter += 1

    print(f"Extracted {counter} images to {output_dir}")
    # rclpy.shutdown()

if __name__ == "__main__":
    bag_path = "/media/saab/f7ee81f1-4052-4c44-b470-0a4a650ee479/cfar_cpp/Analysis/2d_localization_output/2d_localization_output_0.db3" 
    image_topic1 = "/cfar/image"
    image_topic2 = "/oculus_sonar/ping_image"
    output_dirs = "2d_localization_img"      # Change this to your image topic

    cv_image = extract_images_from_bag(bag_path, image_topic1, image_topic2, output_dirs)
    # cv_image_flip_0 = cv2.flip(cv_image,0)
    # cv_image_flip_1 = cv2.flip(cv_image,1)
    # cv_image_flip_2 = cv2.flip(cv_image,-1)

    # fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    # titles = ["Original", "Flipped Vertical", "Flipped Horizontal", "Flipped Both"]
    # images = [cv_image, cv_image_flip_0, cv_image_flip_1, cv_image_flip_2]
    # for ax, img, title in zip(axes, images, titles):
    #     ax.imshow(img)
    #     ax.set_title(title)
    #     ax.axis("off")  # Hide axes
    # plt.tight_layout()
    # plt.show()