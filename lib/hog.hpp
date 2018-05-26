#ifndef HOG_HPP
#define HOG_HPP

#include <vector>
#include "opencv2/objdetect.hpp"

void computeHOG(const std::vector<cv::Mat> &imgs, std::vector<cv::Mat> &gradients);

#endif