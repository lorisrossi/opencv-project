#include <vector>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/ml.hpp"

using namespace std;
using namespace cv;
using namespace cv::ml;

vector<float> get_svm_detector(const Ptr<SVM>& svm) {
  // get the support vectors. The method returns all the support vectors as a
  // floating-point matrix, where support vectors are stored as matrix rows.
  Mat sv = svm->getSupportVectors();
  const int sv_total = sv.rows;

  cout << "Getting Decision Functions..." << endl;
  Mat alpha, svidx;
  // get the decision function
  // If the problem solved is regression, 1-class or 2-class classification,
  // then there will be just one decision function and the index should always
  // be 0. Otherwise, in the case of N-class classification, there will be
  // N(N−1)/2 decision functions.
  double rho = svm->getDecisionFunction(0, alpha, svidx);

  // CV_Assert(alpha.total() == 1 && svidx.total() == 1 && sv_total == 1);
  // CV_Assert((alpha.type() == CV_64F && alpha.at<double>(0) == 1.) ||
  //           (alpha.type() == CV_32F && alpha.at<float>(0) == 1.f));
  // CV_Assert(sv.type() == CV_32F);

  vector<float> hog_detector(sv.cols + 1);
  memcpy(&hog_detector[0], sv.ptr(), sv.cols * sizeof(hog_detector[0]));
  hog_detector[sv.cols] = (float)-rho;
  return hog_detector;
}