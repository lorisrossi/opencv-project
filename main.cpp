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
  string svm_pretrained_model = "./svm_pretrained_model";
  vector<string> classes {"Hbv", "He", "IPCL", "Le"};
  vector<vector<Mat>> train_imgs(4, vector<Mat>());
  vector<vector<Mat>> test_imgs(4, vector<Mat>());

  cout << "Loading dataset...\n";

  loadDataset(dataset_path, classes, train_imgs, "train");
  loadDataset(dataset_path, classes, test_imgs, "test");

  cout << "Dataset loaded correctly\n";

  vector<Mat> train_gradients;
  vector<Mat> test_gradients;
  vector<int> train_labels;
  vector<int> test_labels;

  computeHOG(train_imgs, train_gradients);
  for (size_t i = 0; i < train_imgs.size(); ++i) {
    train_labels.insert(train_labels.end(), train_imgs[i].size(), i+1);
  }
  computeHOG(test_imgs, test_gradients);
  for (size_t i = 0; i < test_imgs.size(); ++i) {
    test_labels.insert(test_labels.end(), test_imgs[i].size(), i+1);
  }
  
  cout << train_gradients.size() << " train gradients inserted\n";
  cout << train_labels.size() << " train labels inserted\n";
  cout << test_gradients.size() << " test gradients inserted\n";
  cout << test_labels.size() << " test labels inserted\n";

  Mat trainingData = Mat(train_gradients.size(), train_gradients[0].cols, CV_32FC1);
  for (size_t i = 0; i < train_gradients.size(); ++i) {
    train_gradients.at(i).copyTo(trainingData.row(i));
  }
  cout << trainingData.size() << " training data size\n";

  Mat testingData = Mat(test_gradients.size(), test_gradients[0].cols, CV_32FC1);
  for (size_t i = 0; i < test_gradients.size(); ++i) {
    test_gradients.at(i).copyTo(testingData.row(i));
  }
  cout << testingData.size() << " testing data size\n";

  cout << "Creating SVM\n";

  Ptr<SVM> svm;
  initSvm(svm);

  cout << "Training...\n";
  
  svm->train(trainingData, ROW_SAMPLE, train_labels);
  
  cout << "Testing...\n";
  
  unsigned int matches = 0;
  for (size_t i = 0; i < test_labels.size(); ++i) {
    if (svm->predict(testingData.row(i)) == test_labels[i]) ++matches;
  }

  cout << endl << matches << " matches out of " << test_labels.size() << endl;
  cout << "Accuracy " << fixed << setprecision(2) << (float(matches) / test_labels.size() * 100) << '%' << endl;

  return 0;
}