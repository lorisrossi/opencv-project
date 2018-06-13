#include <iomanip>
#include <iostream>
#include "hog.hpp"
#include "io.hpp"
#include "opencv2/opencv.hpp"
#include "svm.hpp"

using namespace cv;
using namespace cv::ml;
using namespace std;

int main() {
  string dataset_path = "./dataset";
  vector<string> classes{"Hbv", "He", "IPCL", "Le"};
  vector<vector<vector<Mat>>> imgs(3, vector<vector<Mat>>(4, vector<Mat>()));

  cout << "Loading dataset..." << flush;

  loadDataset(dataset_path, classes, imgs);

  cout << " success!\n";

  /* HOG */

  vector<vector<Mat>> gradients(3, vector<Mat>());
  vector<vector<int>> labels(3, vector<int>());
  vector<Mat> data(3, Mat());

  computeHOG(imgs, gradients, labels, data);

  /* SVM */

  cout << "Init SVM\n";

  Ptr<SVM> svm;
  initSvm(svm);

  cout << "Starting Cross Validation\n";

  crossValidation(svm, data, labels);

  return 0;
}