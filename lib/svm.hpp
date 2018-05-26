#include <vector>
#include "opencv2/ml.hpp"

std::vector<float> get_svm_detector(const cv::Ptr<cv::ml::SVM>& svm);