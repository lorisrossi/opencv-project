#ifndef SVM_HPP
#define SVM_HPP

#include <vector>
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/ml.hpp"

void initSvm(cv::Ptr<cv::ml::SVM> &svm);

void crossValidation(cv::Ptr<cv::ml::SVM> &svm, std::vector<cv::Mat> data, std::vector<std::vector<int>> labels);

#endif