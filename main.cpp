#include <iostream>
#include <iomanip>
#include "opencv2/opencv.hpp"
#include "hog.hpp"
#include "io.hpp"
#include "svm.hpp"

using namespace cv;
using namespace cv::ml;
using namespace std;

int main() {
  string dataset_path = "./dataset";
  vector<string> classes {"Hbv", "He", "IPCL", "Le"};
  vector<vector<Mat>> train_imgs(4, vector<Mat>()), test_imgs(4, vector<Mat>());

  cout << "Loading dataset..." << flush;

  loadDataset(dataset_path, classes, train_imgs, "train");
  loadDataset(dataset_path, classes, test_imgs, "test");

  cout << " success!\n";

  /* HOG */

  vector<Mat> train_gradients, test_gradients;
  vector<int> train_labels, test_labels;
  Mat train_data, test_data;

  computeHOG(train_imgs, train_gradients, train_labels, train_data);
  computeHOG(test_imgs, test_gradients, test_labels, test_data);

  /* SVM */

  cout << "Init SVM\n";

  Ptr<SVM> svm;
  initSvm(svm);

  cout << "Training..." << flush;
  
  svm->train(train_data, ROW_SAMPLE, train_labels);
  
  cout << "  Testing...\n";
  
  unsigned int matches = 0;
  for (size_t i = 0; i < test_labels.size(); ++i) {
    if (svm->predict(test_data.row(i)) == test_labels[i]) ++matches;
  }

  cout << endl << matches << " matches out of " << test_labels.size() << endl;
  cout << "Accuracy " << fixed << setprecision(2) << (float(matches) / test_labels.size() * 100) << '%' << endl;

  return 0;
}