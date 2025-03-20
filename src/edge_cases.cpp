int available_rows = 2 * this->train_hs;
int available_leading_cols = this->train_hs;
int available_lagging_cols = this->train_hs;  // Ensure it is always initialized

// Handling row edge cases
if (row - this->guard_hs < 0 || row + this->guard_hs >= rows) {
    available_rows = this->train_hs;
} else if (row - this->guard_hs - this->train_hs < 0) {
    available_rows = 2 * this->train_hs + (row - this->guard_hs - this->train_hs);
} else if (row + this->guard_hs + this->train_hs >= rows) {
    available_rows = 2 * this->train_hs - (row + this->guard_hs + this->train_hs - (rows - 1));
}

// Handling column edge cases
if (col - this->guard_hs < 0) {
    available_leading_cols = 0;
} else if (col - this->guard_hs - this->train_hs < 0) {
    available_leading_cols = this->train_hs + (col - this->guard_hs - this->train_hs);
}

if (col + this->guard_hs >= cols) {
    available_lagging_cols = 0;
} else if (col + this->guard_hs + this->train_hs >= cols) {
    available_lagging_cols = this->train_hs - (col + this->guard_hs + this->train_hs - (cols - 1));
}

float CFAR::calc_rect_sum(cv::Mat& img, int x, int y, int w, int h) {
    // Clamp coordinates to valid image bounds
    int x1 = std::max(0, x);                // Top-left x
    int y1 = std::max(0, y);                // Top-left y
    int x2 = std::min(img.rows - 1, x + h - 1); // Bottom-right x
    int y2 = std::min(img.cols - 1, y + w - 1); // Bottom-right y

    // Calculate the sum using the clamped coordinates in the integral image
    float sum = img.at<float>(x2, y2)
              - (x1 > 0 ? img.at<float>(x1 - 1, y2) : 0)
              - (y1 > 0 ? img.at<float>(x2, y1 - 1) : 0)
              + (x1 > 0 && y1 > 0 ? img.at<float>(x1 - 1, y1 - 1) : 0);

    return sum;
}
