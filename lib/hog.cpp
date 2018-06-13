#include <iostream>
#include <vector>
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"

using namespace std;
using namespace cv;

void _computeHOG(const vector<vector<Mat>> &imgs, vector<Mat> &gradients) {
  HOGDescriptor hog;
  hog.winSize = imgs.at(0).at(0).size() / 8 * 8;
  vector<float> descriptors;
  Mat gray;

  for (size_t j = 0; j < imgs.size(); ++j) {
    for (size_t i = 0; i < imgs.at(j).size(); ++i) {
      if (imgs.at(j).at(i).rows >= hog.winSize.height &&
          imgs.at(j).at(i).cols >= hog.winSize.width) {
        Rect r = Rect((imgs.at(j).at(i).rows - hog.winSize.height) / 2,
                      (imgs.at(j).at(i).cols - hog.winSize.width) / 2,
                      hog.winSize.width, hog.winSize.height);

        cvtColor(imgs.at(j).at(i)(r), gray, COLOR_BGR2GRAY);
        hog.compute(gray, descriptors, Size(8, 8), Size(0, 0));
        gradients.push_back(Mat(descriptors).t());
      }
    }
  }
}

void computeHOG(vector<vector<vector<Mat>>> &imgs,
                vector<vector<Mat>> &gradients, vector<vector<int>> &labels,
                vector<Mat> &data_for_svm) {
  for (uint8_t k = 0; k < 3; ++k) {
    _computeHOG(imgs.at(k), gradients.at(k));
    for (size_t i = 0; i < imgs.at(0).size(); ++i) {
      labels.at(k).insert(labels.at(k).end(), imgs.at(k).at(i).size(), i + 1);
    }

    data_for_svm.at(k) =
        Mat(gradients.at(k).size(), gradients.at(k).at(0).cols, CV_32FC1);
    
    for (size_t i = 0; i < gradients.at(k).size(); ++i) {
      gradients.at(k).at(i).copyTo(data_for_svm.at(k).row(i));
    }
  }
}