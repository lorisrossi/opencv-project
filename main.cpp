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
  vector<Mat> hbv_imgs, he_imgs, ipcl_imgs, le_imgs;
  vector<Mat> test_hbv_imgs, test_he_imgs, test_ipcl_imgs, test_le_imgs;

  cout << "Loading dataset...\n";

  loadTrainingDataset(dataset_path, "Hbv", hbv_imgs);
  loadTrainingDataset(dataset_path, "He", he_imgs);
  loadTrainingDataset(dataset_path, "IPCL", ipcl_imgs);
  loadTrainingDataset(dataset_path, "Le", le_imgs);

  loadTestingDataset(dataset_path, "Hbv", test_hbv_imgs);
  loadTestingDataset(dataset_path, "He", test_he_imgs);
  loadTestingDataset(dataset_path, "IPCL", test_ipcl_imgs);
  loadTestingDataset(dataset_path, "Le", test_le_imgs);

  cout << "Dataset loaded correctly\n";

  vector<Mat> gradients;
  vector<int> labels;

  computeHOG(hbv_imgs, gradients);
  labels.insert(labels.end(), hbv_imgs.size(), 1);
  computeHOG(he_imgs, gradients);
  labels.insert(labels.end(), he_imgs.size(), 2);
  computeHOG(ipcl_imgs, gradients);
  labels.insert(labels.end(), ipcl_imgs.size(), 3);
  computeHOG(le_imgs, gradients);
  labels.insert(labels.end(), le_imgs.size(), 4);

  vector<Mat> test_gradients;
  vector<int> test_labels;

  computeHOG(test_hbv_imgs, test_gradients);
  test_labels.insert(test_labels.end(), test_hbv_imgs.size(), 1);
  computeHOG(test_he_imgs, test_gradients);
  test_labels.insert(test_labels.end(), test_he_imgs.size(), 2);
  computeHOG(test_ipcl_imgs, test_gradients);
  test_labels.insert(test_labels.end(), test_ipcl_imgs.size(), 3);
  computeHOG(test_le_imgs, test_gradients);
  test_labels.insert(test_labels.end(), test_le_imgs.size(), 4);

  // cout << gradients.size() << " gradients inserted\n";
  // cout << labels.size() << " labels inserted\n";

  Mat trainingData = Mat(gradients.size(), gradients[0].cols, CV_32FC1);
  for (size_t i = 0; i < gradients.size(); ++i) {
    gradients.at(i).copyTo(trainingData.row(i));
  }
  cout << trainingData.size() << " training data size\n";

  Mat testingData =
      Mat(test_gradients.size(), test_gradients[0].cols, CV_32FC1);
  for (size_t i = 0; i < test_gradients.size(); ++i) {
    test_gradients.at(i).copyTo(testingData.row(i));
  }
  cout << testingData.size() << " testing data size\n";

  cout << "Creating SVM\n";
  Ptr<SVM> svm;
  initSvm(svm);

  cout << "Training...\n";
  svm->train(trainingData, ROW_SAMPLE, labels);
  
  cout << "Testing...\n";
  unsigned int matches = 0;
  for (size_t i = 0; i < test_labels.size(); ++i) {
    if (svm->predict(testingData.row(i)) == test_labels[i]) ++matches;
  }

  cout << endl << matches << " matches out of " << test_labels.size() << endl;
  cout << "Accuracy " << fixed << setprecision(2) << (float(matches) / test_labels.size() * 100) << '%' << endl;

  return 0;
}