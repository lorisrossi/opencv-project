#ifndef IO_HPP
#define IO_HPP

#include <string>
#include "opencv2/opencv.hpp"

void loadDataset(const std::string &dirname, std::vector<std::string> &labels,
                 std::vector<std::vector<std::vector<cv::Mat>>> &imgs,
                 bool show_images = false);

#endif