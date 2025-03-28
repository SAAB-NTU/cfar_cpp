#include <cfar.h>
#include <string>
#include <filesystem>

#include "opencv2/opencv.hpp"

int main()
{
    std::string path = "/media/saab/f7ee81f1-4052-4c44-b470-0a4a650ee479/cfar_cpp/Analysis/Oculus_reader/20250212_181120/polar/";
    std::string image_name = "1739355080.836.png";
    
    std::filesystem::path image_path = std::filesystem::path(path) / image_name;

    // Read the image using OpenCV
    cv::Mat image = cv::imread(image_path.string());

    // Check if the image was loaded successfully
    if (image.empty()) {
        std::cerr << "Error importing image" << std::endl;
        return EXIT_FAILURE; // Exit with failure code
    }

    CFAR cfar(40,10,0.160);
    cv::Mat result_1d = cv::Mat::zeros(image.size(), CV_32F);
    cv::Mat result_2d = cv::Mat::zeros(image.size(), CV_32F);
    cfar.soca_1d(image, result_1d);
    cfar.soca_2d_integral(image, result_2d);
    cv::normalize(result_1d, result_1d, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::normalize(result_2d, result_2d, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    cv::imwrite("original.png", image);
    cv::imwrite("cfar1d.png", result_1d);
    cv::imwrite("cfar2d.png", result_2d);

    return EXIT_SUCCESS; // Exit successfully
}