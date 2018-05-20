#include "opencv2/opencv.hpp"

#include "io.hpp"
#include "hog.hpp"

using namespace cv;
using namespace std;

int main() {
  string dataset_path = "./dataset";
  vector<Mat> hbv_imgs, he_imgs, ipcl_imgs, le_imgs;

  cout << "Loading dataset...\n";

  loadDataset(dataset_path, "Hbv", hbv_imgs);
  loadDataset(dataset_path, "He", he_imgs);
  loadDataset(dataset_path, "Ipcl", ipcl_imgs);
  loadDataset(dataset_path, "Le", le_imgs);

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

  cout << gradients.size() << " gradients inserted\n";
  cout << labels.size() << " labels inserted\n";

  Mat trainingData = Mat(gradients.size(), gradients[0].cols, CV_32FC1);
  for (size_t i = 0; i < gradients.size(); ++i) {
    gradients.at(i).copyTo(trainingData.row(i));
  }

  cout << trainingData.size() << " training data size\n";

  return 0;
}