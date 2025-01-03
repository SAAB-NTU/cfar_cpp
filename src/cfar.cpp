#include <cfar.h>

CFAR::CFAR(int train_cells, int guard_cells, float false_alarm_rate)
{
    assert(train_cells%2 == 0);
    assert(guard_cells%2 == 0);
    this->train_cells = train_cells;
    this->guard_cells = guard_cells;
    this->false_alarm_rate = false_alarm_rate;

    this->train_hs = this->train_cells/2;
    this->guard_hs = this->guard_cells/2;
    
    this->threshold_factor_SOCA = CFAR::retrieve_params(
        this->train_cells, this->guard_cells, this->false_alarm_rate);

    this->rank = this->train_cells/2;
}

CFAR::~CFAR()
{
}

double CFAR::retrieve_params(int train_cells, int guard_cells, float false_alarm_rate)
{
    int train_num = (train_cells - 2)/2;
    int guard_num = (guard_cells - 2)/2;
    int false_rate_num = (int)((false_alarm_rate - 0.90)/0.005);
    int line = train_num*200 + guard_num*20 + false_rate_num + 1;

    // TODO: change path to dynamic
    std::fstream inputFile("/home/trgknng/ros2_ws/src/cfar_cpp/parameter_map.txt");

    if (!inputFile.is_open()) {
        std::cerr << "Error opening the file!" << std::endl;
        return -1; // Exit with error
    }
    int current_line = 0;
    std::string data;

    while (getline(inputFile, data)) {
        current_line++;
        if (current_line == line) {
            // Extracts the threshold data from data string
            std::stringstream ss(data);
            std::string value;

            // Read values until we reach the last two
            for (int i = 0; i < 5; ++i) {
                getline(ss, value, ',');
                if (i == 3) {
                    return stod(value);
                }
            }
            break;
        }
    }
    return -1;
}

cv::Mat CFAR::soca(cv::Mat& img)
{
    // auto start = std::chrono::high_resolution_clock::now();
    
    // There's a CUDA variation for this function
    cv::Mat img_gray;
    if (img.channels() == 3) {
        cv::cvtColor(img, img_gray,cv::COLOR_BGR2GRAY);
    } else {
        img_gray = img;
    }

    cv::Mat blurred_image;
    cv::GaussianBlur(img_gray, blurred_image, cv::Size(7,7), 0);

    cv::Mat denoised_image;
    // There's a CUDA variation for this function
    cv::fastNlMeansDenoising(blurred_image, denoised_image, 200);

    cv::Mat integral_image;
    cv::integral(denoised_image, integral_image);

    cv::Mat trimmed_image = integral_image(cv::Rect(1,1,integral_image.cols-1, integral_image.rows-1));
    trimmed_image.convertTo(trimmed_image, CV_32F);

    int rows = trimmed_image.rows;
    int cols = trimmed_image.cols;
    cv::Mat result = cv::Mat::zeros(rows, cols, CV_32F);

    int total_train_cells = this->train_hs * (2 * this->train_hs + 2 * this->guard_hs + 1);
    
    for (int col = (this->train_hs + this->guard_hs) + 1; col < (cols - this->train_hs - this->guard_hs); col++) {
        for (int row = (this->train_hs + this->guard_hs) + 1; row < (rows - this->train_hs - this->guard_hs); row++) {
            float leading_guard = calc_rect_sum(trimmed_image,
                                            row - this->guard_hs, col - this->guard_hs,
                                            this->guard_hs, 2*this->guard_hs+1);
            float leading_sum = calc_rect_sum(trimmed_image,
                                            row - this->guard_hs - this->train_hs, col - this->guard_hs - this->train_hs,
                                            this->guard_hs + this->train_hs,
                                            (2 * this->guard_hs + 2 * this->train_hs + 1));
            float leading_train = leading_sum - leading_guard;

            // Calculate lagging guard and lagging sum
            float lagging_guard = calc_rect_sum(trimmed_image,
                                            row - this->guard_hs, col + 1,
                                            this->guard_hs, 
                                            (2 * this->guard_hs + 1));
            float lagging_sum = calc_rect_sum(trimmed_image,
                                            row - this->guard_hs - this->train_hs, col + 1,
                                            this->guard_hs + this->train_hs,
                                            (2 * this->guard_hs + 2 * this->train_hs + 1));
            float lagging_train = lagging_sum - lagging_guard;

            // Calculate minimum of leading and lagging train sums
            float sum_train = std::min(leading_train, lagging_train);
            
            // Assuming ret is a cv::Mat to store results
            int num = (this->threshold_factor_SOCA * sum_train / total_train_cells);
            result.at<float>(row, col) = num;
        }
    }

    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> duration = end - start;
    // std::cout << "Execution time: " << duration.count() << " seconds." << std::endl;
    return result;
}

float CFAR::calc_rect_sum(cv::Mat& img, int x, int y, int w, int h) {
    // Check if the coordinates are within bounds
    if (x < 0 || y < 0 || x + h - 1 >= img.rows || y + w - 1 >= img.cols) {
        throw std::out_of_range("Coordinates are out of bounds");
    }

    // Calculate the sum of the rectangular block in the integral image
    float sum = img.at<float>(x + h - 1, y + w - 1)
              - (x > 0 ? img.at<float>(x - 1, y + w - 1) : 0)
              - (y > 0 ? img.at<float>(x + h - 1, y - 1) : 0)
              + (x > 0 && y > 0 ? img.at<float>(x - 1, y - 1) : 0);

    return sum;
}

int CFAR::get_train_cells() {
    return this->train_cells;
}

int CFAR::get_guard_cells() {
    return this->guard_cells;
}

float CFAR::get_false_alarm_rate() {
    return this->false_alarm_rate;
}

float CFAR::get_threshold_factor_soca() {
    return this->threshold_factor_SOCA;
}