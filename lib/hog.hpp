#ifndef HOG_HPP
#define HOG_HPP

#include <vector>
#include "opencv2/objdetect.hpp"

void computeHOG(std::vector<std::vector<cv::Mat>> &imgs,
  std::vector<cv::Mat> &gradients, std::vector<int> &labels, cv::Mat &dataForSvm);

#endif