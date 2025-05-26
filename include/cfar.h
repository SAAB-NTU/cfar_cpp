#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>

#include <armadillo>
#include <boost/math/special_functions/binomial.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/photo.hpp>

class CFAR
{
    private:
        int train_cells; // Number of training cells
        int guard_cells; // Number of guard cells
        int total_train_cells;
        int train_hs;
        int guard_hs;
        int total_hs;
        float Pfa; // False alarm rate
        int rank;

        // float threshold_factor_SOCA;
        // float threshold_factor_GOCA;

        float threshold_mul;    // threshold multiplier
    public:
        CFAR();
        CFAR(int train_cells, int guard_cells, float Pfa);
        ~CFAR();

        double calcMultiplier();
        int getTrainCells();
        int getGuardCells();
        float getPfa();
        float getThresholdMultiplier();

        void soca_1d(cv::Mat& img, cv::Mat& des);
        void soca_2d(cv::Mat& img, cv::Mat& des);
        void soca_1d_integral(cv::Mat& img, cv::Mat& des);
        void soca_2d_integral(cv::Mat& img, cv::Mat& des);
        float calc_rect_sum(cv::Mat& img, int x, int y, int w, int h);
};