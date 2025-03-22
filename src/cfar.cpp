#include <cfar.h>

CFAR::CFAR()
{
    
}

CFAR::CFAR(int train_cells, int guard_cells, float false_alarm_rate)
{
    assert(train_cells%2 == 0);
    assert(guard_cells%2 == 0);
    this->train_cells = train_cells;
    this->guard_cells = guard_cells;
    this->false_alarm_rate = false_alarm_rate;

    this->train_hs = this->train_cells/2;
    this->guard_hs = this->guard_cells/2;
    this->total_hs = this->train_hs + this->guard_hs;
    
    this->threshold_factor_SOCA = CFAR::retrieve_params(
        this->train_cells, this->guard_cells, this->false_alarm_rate);

    this->rank = this->train_cells/2;
    this->total_train_cells = this->train_hs * (2 * this->train_hs + 2 * this->guard_hs + 1);
}

CFAR::~CFAR()
{
}

double CFAR::retrieve_params(int train_cells, int guard_cells, float false_alarm_rate)
{
    
    int train_num = (train_cells - 2)/2;
    int guard_num = (guard_cells - 2)/2;
    int false_rate_num = (int)((false_alarm_rate - 0.005)/0.005);
    int line = train_num*1990 + guard_num*199 + false_rate_num + 1;

    // TODO: change path to dynamic
    std::fstream inputFile("parameter_map.txt");

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

void CFAR::soca_2d(cv::Mat& img, cv::Mat& des)
{
    cv::Mat img_gray;
    if (img.channels() == 3) {
        cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
    } else {
        img_gray = img.clone();
    }

    int rows = img_gray.rows;
    int cols = img_gray.cols;

    for (int row = (this->train_hs + this->guard_hs) + 1; row < (rows - this->train_hs - this->guard_hs); row++) {
        for (int col = (this->train_hs + this->guard_hs) + 1; col < (cols - this->train_hs - this->guard_hs); col++) {
            float leading_sum = 0.0, lagging_sum = 0.0;
            for (int i = row - this->train_hs - this->guard_hs; i <= row + this->train_hs + this->guard_hs; ++i) {
                for (int j = col - this->train_hs - this->guard_hs; j <= col + this->train_hs + this->guard_hs; ++j) {
                    if (std::abs(i - row) <= this->guard_hs && std::abs(j - col) <= this->guard_hs) {
                        continue;
                    }
                    if ((i < row && j < col) || (i < row && j == col) || (i == row && j < col)) {
                        leading_sum += img_gray.at<uchar>(i, j);
                    }
                    else {
                        lagging_sum += img_gray.at<uchar>(i, j);
                    }
                }
            }

            float sum_train = std::min(leading_sum, lagging_sum);
            float num = (this->threshold_factor_SOCA * sum_train / total_train_cells);

            des.at<float>(row, col) = (img_gray.at<uchar>(row, col) > num) ? img_gray.at<uchar>(row, col) : 0.0f;
        }
    }
}

void CFAR::soca_2d_integral(cv::Mat& img, cv::Mat& des)
{
    cv::Mat img_gray;
    if (img.channels() == 3) {
        cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
    } else {
        img_gray = img.clone();
    }

    cv::Mat integral_image;
    cv::integral(img_gray, integral_image, CV_32F);
    cv::Mat trimmed_image = integral_image(cv::Rect(1, 1, integral_image.cols - 1, integral_image.rows - 1));

    int rows = trimmed_image.rows;
    int cols = trimmed_image.cols;

    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            int available_rows = 2 * this->train_hs;
            int available_leading_cols = this->train_hs;
            int available_lagging_cols = this->train_hs;

            if (row - this->guard_hs < 0 || row + this->guard_hs >= rows) {
                available_rows = this->train_hs;
            } else if (row - this->total_hs < 0) {
                available_rows = 2 * this->train_hs + (row - this->total_hs);
            } else if (row + this->guard_hs + this->train_hs >= rows) {
                available_rows = 2 * this->train_hs - (row + this->total_hs - (rows - 1));
            }

            if (col - this->guard_hs < 0) {
                available_leading_cols = 0;
            } else if (col - this->total_hs < 0) {
                available_leading_cols = this->train_hs + (col - this->total_hs);
            }

            if (col + this->guard_hs >= cols) {
                available_lagging_cols = 0;
            } else if (col + this->total_hs >= cols) {
                available_lagging_cols = this->train_hs - (col + this->total_hs - (cols - 1));
            }

            int total_leading_cells = 2*available_rows*available_leading_cols;
            int total_lagging_cells = 2*available_rows*available_lagging_cols;

            float leading_guard = calc_rect_sum(trimmed_image, row - this->guard_hs, col - this->guard_hs, this->guard_hs, 2*this->guard_hs+1);
            float leading_sum = calc_rect_sum(trimmed_image, row - this->total_hs, col - this->total_hs, this->total_hs, (2*this->total_hs + 1));
            float leading_train = (total_leading_cells > 0)?(leading_sum - leading_guard)/total_leading_cells:0.0f;

            float lagging_guard = calc_rect_sum(trimmed_image, row - this->guard_hs, col + 1, this->guard_hs, (2 * this->guard_hs + 1));
            float lagging_sum = calc_rect_sum(trimmed_image, row - this->total_hs, col + 1, this->total_hs, (2 * this->total_hs + 1));
            float lagging_train = (total_lagging_cells > 0)?(lagging_sum - lagging_guard)/total_lagging_cells:0.0f;

            // float sum_train = std::min(leading_train, lagging_train);
            float num = (this->threshold_factor_SOCA * std::min(leading_train, lagging_train));

            des.at<float>(row, col) = (img_gray.at<uchar>(row, col) > num) ? img_gray.at<uchar>(row, col) : 0.0f;
        }
    }
}

void CFAR::soca_1d(cv::Mat& img, cv::Mat& des) {
    // Similar implementation to bruce-slam
    cv::Mat img_gray;
    if (img.channels() == 3) {
        cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
    } else {
        img_gray = img.clone();
    }

    int rows = img_gray.rows;
    int cols = img_gray.cols;

    for (int row = this->train_hs + this->guard_hs; row < rows - this->train_hs - this->guard_hs; ++row)
    {
        for (int col = 0; col < cols; ++col)
        {
            float leading_sum = 0.0, lagging_sum = 0.0;
            for (int i = row - this->train_hs - this->guard_hs; i < row + this->train_hs + this->guard_hs + 1; ++i)
            {
                if ((i - row) > this->guard_hs)
                lagging_sum += img_gray.at<uchar>(i, col);
                else if ((i - row) < -this->guard_hs)
                leading_sum += img_gray.at<uchar>(i, col);
            }
            float sum_train = std::min(leading_sum, lagging_sum);
            float num = (this->threshold_factor_SOCA * sum_train / total_train_cells);
            des.at<float>(row, col) = (img_gray.at<uchar>(row, col) > num) ? img_gray.at<uchar>(row, col) : 0.0f;
        }
    }
}

void CFAR::soca_1d_integral(cv::Mat& img, cv::Mat& des) {
    cv::Mat img_gray;
    if (img.channels() == 3) {
        cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
    } else {
        img_gray = img;
    }

    int rows = img_gray.rows;
    int cols = img_gray.cols;

    for (int row = this->train_hs + this->guard_hs; row < rows - this->train_hs - this->guard_hs; ++row)
    {
        for (int col = 0; col < cols; ++col)
        {
            float leading_sum = calc_rect_sum(img_gray, row - this->guard_hs - this->train_hs, col, 0, this->train_hs);
            float lagging_sum = calc_rect_sum(img_gray, row + this->guard_hs, col, 0, this->train_hs);
            float sum_train = std::min(leading_sum, lagging_sum);
            float num = (this->threshold_factor_SOCA * sum_train / total_train_cells);
            des.at<float>(row, col) = (img_gray.at<uchar>(row, col) > num) ? img_gray.at<uchar>(row, col) : 0.0f;
        }
    }
}

// float CFAR::calc_rect_sum(cv::Mat& img, int x, int y, int w, int h) {
//     // TODO: make integral image consider edge cases within the training cells
//     if (x < 0 || y < 0 || x + h - 1 >= img.rows || y + w - 1 >= img.cols) {
//         throw std::out_of_range("Coordinates are out of bounds");
//     }

//     // Calculate the sum of the rectangular block in the integral image
//     float sum = img.at<float>(x + h - 1, y + w - 1)
//               - (x > 0 ? img.at<float>(x - 1, y + w - 1) : 0)
//               - (y > 0 ? img.at<float>(x + h - 1, y - 1) : 0)
//               + (x > 0 && y > 0 ? img.at<float>(x - 1, y - 1) : 0);

//     return sum;
// }

float CFAR::calc_rect_sum(cv::Mat& img, int x, int y, int w, int h) {
    int x1 = std::max(0, x);
    int y1 = std::max(0, y);
    int x2 = std::min(img.rows - 1, x + h - 1);
    int y2 = std::min(img.cols - 1, y + w - 1);

    float sum = img.at<float>(x2, y2)
              - (x1 > 0 ? img.at<float>(x1 - 1, y2) : 0)
              - (y1 > 0 ? img.at<float>(x2, y1 - 1) : 0)
              + (x1 > 0 && y1 > 0 ? img.at<float>(x1 - 1, y1 - 1) : 0);

    return sum;
}

