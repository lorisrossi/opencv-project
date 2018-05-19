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

  computeHOG(hbv_imgs);

  return 0;
}