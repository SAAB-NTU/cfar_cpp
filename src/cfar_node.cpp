#include "cfar.h"

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cfar_cpp/msg/cfar_info.hpp"

#include <cv_bridge/cv_bridge.h>

#include<chrono>

class CfarNode: public rclcpp::Node
{
    public:
        CfarNode() : Node("cfar_node"), count_(0) {


            sonar_subscriber_ = this->create_subscription<sensor_msgs::msg::Image>("/oculus_sonar/ping_image", 10, std::bind(&CfarNode::topic_callback, this, std::placeholders::_1));
            cfar_publisher_ = this->create_publisher<sensor_msgs::msg::Image>("/cfar/image", 10);
            
            rclcpp::QoS qos_latch(1);
            qos_latch.durability(rclcpp::DurabilityPolicy::TransientLocal);
            cfar_info_publisher_ = this->create_publisher<cfar_cpp::msg::CfarInfo>("/cfar/info", qos_latch);

            // Read from config.yaml
            this->declare_parameter<std::string>("mode","soca");
            this->declare_parameter<int>("train_cells",16);
            this->declare_parameter<int>("guard_cells",2);
            this->declare_parameter<float>("false_alarm_rate",0.97);

            this->get_parameter("mode", this->mode);
            this->get_parameter("train_cells", this->train_cells);
            this->get_parameter("guard_cells", this->guard_cells);
            this->get_parameter("false_alarm_rate", this->false_alarm_rate);
            
            cfar_filter = CFAR(
                this->train_cells, 
                this->guard_cells, 
                this->false_alarm_rate
            );
            
            this->publish_cfar_info();

            elapse_total = 0;
            elapse_count = 0;
        }
    
    private:
        void publish_cfar_info() {
            auto info = cfar_cpp::msg::CfarInfo();
            info.mode = this->mode;
            info.train_cells = cfar_filter.get_train_cells();
            info.guard_cells = cfar_filter.get_guard_cells();
            // Round to 3 decimal places
            info.false_alarm_rate = round(cfar_filter.get_false_alarm_rate() * 1000.0) / 1000.0;
            info.threshold_factor = cfar_filter.get_threshold_factor_soca();

            cfar_info_publisher_->publish(info);
        }

        void topic_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
            try {
                cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::MONO8);
                if (cv_ptr->image.empty()) {
                    RCLCPP_ERROR(this->get_logger(), "Received empty image");
                    return;
                }
                if (result.empty()) { 
                    result = cv::Mat::zeros(cv_ptr->image.rows, cv_ptr->image.cols, CV_32F);
                }
                auto start = std::chrono::high_resolution_clock::now();
                // cfar_filter.soca(cv_ptr->image, result);
                // cfar_filter.soca_1d(cv_ptr->image, result);
                cfar_filter.soca_1d_integral(cv_ptr->image, result);
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> elapsed = end - start; // Time in milliseconds
                // RCLCPP_INFO(this->get_logger(), "Execution time: %.3f ms", elapsed.count());

                if (elapse_count < 100)
                {
                    elapse_total += elapsed.count();
                    elapse_count ++;
                }
                else if (elapse_count == 100)
                {
                    double average = elapse_total/elapse_count;
                    elapse_count = 0;
                    elapse_total = 0;
                    RCLCPP_INFO(this->get_logger(), "Average execution time: %.3f ms", average);
                }

        
                // TODO: move uint8 conversion to inside cfar_filter
                cv::Mat result_uint8;
                cv::normalize(result, result_uint8, 0, 255, cv::NORM_MINMAX, CV_8UC1);

                cv_bridge::CvImage cv_image(
                    msg->header,
                    sensor_msgs::image_encodings::MONO8,
                    result_uint8
                );
                auto img_msg = cv_image.toImageMsg();
                img_msg->step = result.step[0];
                cfar_publisher_->publish(*img_msg);
            } 
            catch (const cv_bridge::Exception& e) {
                RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            } 
            catch (const std::exception& e) {
                RCLCPP_ERROR(this->get_logger(), "Exception: %s", e.what());
            }
        }

    private:
        size_t count_;
        std::string mode;
        int train_cells;
        int guard_cells;
        float false_alarm_rate;

        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sonar_subscriber_;
        rclcpp::Publisher<cfar_cpp::msg::CfarInfo>::SharedPtr cfar_info_publisher_;
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr cfar_publisher_;

        CFAR cfar_filter;
        cv::Mat result;
        double elapse_total;
        int elapse_count;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CfarNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
