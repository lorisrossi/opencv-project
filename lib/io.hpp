#ifndef IO_HPP
#define IO_HPP

#include <string>
#include "opencv2/opencv.hpp"

void loadDataset(const std::string &dirname, const std::string &label, std::vector<cv::Mat> &imgs, bool show_images = false);

#endif