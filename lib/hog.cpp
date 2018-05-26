#include <iostream>
#include <vector>
#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;
using namespace cv;

void _computeHOG(const vector<vector<Mat>> &imgs, vector<Mat> &gradients) {
  HOGDescriptor hog;
  hog.winSize = imgs.at(0).at(0).size() / 8 * 8;
  vector<float> descriptors;
  Mat gray;

  for (size_t j = 0; j < imgs.size(); ++j) {
    for (size_t i = 0; i < imgs.at(j).size(); ++i) {
      if (imgs.at(j).at(i).rows >= hog.winSize.height && imgs.at(j).at(i).cols >= hog.winSize.width) {
        Rect r = Rect((imgs.at(j).at(i).rows - hog.winSize.height) / 2,
                      (imgs.at(j).at(i).cols - hog.winSize.width) / 2,
                      hog.winSize.width,
                      hog.winSize.height);
        
        cvtColor(imgs.at(j).at(i)(r), gray, COLOR_BGR2GRAY);
        hog.compute(gray, descriptors, Size(8, 8), Size(0, 0));
        gradients.push_back(Mat(descriptors).t());
      }
    }
  }
}

void computeHOG(vector<vector<Mat>> &imgs, vector<Mat> &gradients, vector<int> &labels, Mat &data_for_svm) {
  _computeHOG(imgs, gradients);
  for (size_t i = 0; i < imgs.size(); ++i) {
    labels.insert(labels.end(), imgs.at(i).size(), i+1);
  }

  cout << gradients.size() << " gradients inserted\n";
  cout << labels.size() << " labels inserted\n";
  
  data_for_svm = Mat(gradients.size(), gradients[0].cols, CV_32FC1);
  for (size_t i = 0; i < gradients.size(); ++i) {
    gradients.at(i).copyTo(data_for_svm.row(i));
  }
  cout << data_for_svm.size() << " data size\n";
}