#ifndef HOG_HPP
#define HOG_HPP

#include <vector>
#include "opencv2/objdetect.hpp"

void computeHOG(std::vector<std::vector<std::vector<cv::Mat>>> &imgs,
  std::vector<std::vector<cv::Mat>> &gradients, std::vector<std::vector<int>> &labels, std::vector<cv::Mat> &dataForSvm);

#endif