#include <cfar.h>
#include <string>
#include <filesystem>

#include "opencv2/opencv.hpp"

int main()
{
    std::string path = "/home/trgknng/ros2_ws/src/pingviewer_ros/pingviewer_ros/data";
    std::string image_name = "image_010.png";
    
    std::filesystem::path image_path = std::filesystem::path(path) / image_name;

    // Read the image using OpenCV
    cv::Mat image = cv::imread(image_path.string());
    cv::imshow("original image", image);

    // Check if the image was loaded successfully
    if (image.empty()) {
        std::cerr << "Error importing image" << std::endl;
        return EXIT_FAILURE; // Exit with failure code
    }

    CFAR cfar(12,4,0.97);
    cv::Mat result = cfar.soca(image);
    cv::Mat final_result;

    // cv::normalize(result, result, 0, 255, cv::NORM_MINMAX, CV_8U);
    // std::cout << final_result << std::endl;
    // If you need integer values
    result.convertTo(final_result, CV_8U, 255.0);  // Convert to 8-bit unsigned integer

    cv::imshow("Image", final_result);
    cv::waitKey(0); // Wait for a key press

    return EXIT_SUCCESS; // Exit successfully
}