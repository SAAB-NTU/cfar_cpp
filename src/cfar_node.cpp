#include "cfar.h"

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cfar_cpp/msg/cfar_info.hpp"

#include <cv_bridge/cv_bridge.h>

class CfarNode: public rclcpp::Node
{
    public:
        CfarNode() : Node("cfar_node"), count_(0), cfar_filter(12, 4, 0.97) {
            sonar_subscriber_ = this->create_subscription<sensor_msgs::msg::Image>("/oculus_sonar/ping_image", 10, std::bind(&CfarNode::topic_callback, this, std::placeholders::_1));
            cfar_info_publisher_ = this->create_publisher<cfar_cpp::msg::CfarInfo>("/cfar/info", 10);
            cfar_publisher_ = this->create_publisher<sensor_msgs::msg::Image>("/cfar/image", 10);

            // Read from config.yaml
            this->declare_parameter<std::string>("mode","soca");
            this->declare_parameter<int>("train_cells",12);
            this->declare_parameter<int>("guard_cells",4);
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
            
            RCLCPP_INFO(this->get_logger(), "Constructor finished");

        }
    
    private:
        void publish_cfar_info() {
            auto info = cfar_cpp::msg::CfarInfo();
            info.mode = this->mode;
            info.train_cells = cfar_filter.get_train_cells();
            info.guard_cells = cfar_filter.get_guard_cells();
            info.false_alarm_rate = cfar_filter.get_false_alarm_rate();
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

                cv::Mat result = cfar_filter.soca(cv_ptr->image);
                result.convertTo(result, CV_8U, 255.0); 

                cv_bridge::CvImage cv_image;
                cv_image.image = result;
                cv_image.encoding = "mono8"; 
                cv_image.header = msg->header;
                auto msg = cv_image.toImageMsg();

                // Convert to sensor_msgs::Image message
                sensor_msgs::msg::Image img_msg;
                cv_image.toImageMsg(img_msg);
                msg->step = result.cols;
                cfar_publisher_->publish(*msg);
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
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CfarNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
