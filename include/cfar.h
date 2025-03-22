#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/photo.hpp>

class CFAR
{
    private:
        int train_cells; // Number of training cells
        int guard_cells; // Number of guard cells
        int train_hs;
        int guard_hs;
        int total_hs;
        float false_alarm_rate; // False alarm rate
        int rank;

        float threshold_factor_SOCA;
        float threshold_factor_GOCA;

        int total_train_cells;
    public:
        CFAR();
        CFAR(int train_cells, int guard_cells, float false_alarm_rate);
        ~CFAR();

        double retrieve_params(int train_cells, int guard_cells, float false_alarm_rate);
        int get_train_cells();
        int get_guard_cells();
        float get_false_alarm_rate();
        float get_threshold_factor_soca();

        void soca_1d(cv::Mat& img, cv::Mat& des);
        void soca_2d(cv::Mat& img, cv::Mat& des);
        void soca_1d_integral(cv::Mat& img, cv::Mat& des);
        void soca_2d_integral(cv::Mat& img, cv::Mat& des);
        float calc_rect_sum(cv::Mat& img, int x, int y, int w, int h);
};