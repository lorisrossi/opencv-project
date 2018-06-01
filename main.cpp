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

  // /* HOG */

  vector<vector<Mat>> gradients(3, vector<Mat>());
  vector<vector<int>> labels(3, vector<int>());
  vector<Mat> data(3, Mat());

  computeHOG(imgs, gradients, labels, data);

  // /* SVM */

  // cout << "Init SVM\n";

  // Ptr<SVM> svm;
  // initSvm(svm);

  // cout << "Training..." << flush;

  // svm->train(train_data, ROW_SAMPLE, train_labels);

  // cout << "  Testing...\n";

  // unsigned int matches = 0;
  // for (size_t i = 0; i < test_labels.size(); ++i) {
  //   if (svm->predict(test_data.row(i)) == test_labels[i]) ++matches;
  // }

  // cout << endl << matches << " matches out of " << test_labels.size() <<
  // endl; cout << "Accuracy " << fixed << setprecision(2) << (float(matches) /
  // test_labels.size() * 100) << '%' << endl;

  return 0;
}