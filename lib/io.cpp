#include "io.hpp"

using namespace std;
using namespace cv;

void loadTrainingDataset(const string &dirname, const string &label,
                         vector<Mat> &imgs, bool show_images) {
  vector<cv::String> files;
  for (char k = '1'; k <= '2'; ++k) {
    try {
      glob(dirname + "/FOLD " + k + '/' + label, files);
    } catch (exception &e) {
      cout << "Error loading dataset/FOLD " << k << "\n\nExit\n";
      exit(1);
    }

    for (size_t i = 0; i < files.size(); ++i) {
      Mat img = imread(files[i]);
      if (img.empty()) {
        cout << files.at(i) << " is not valid." << endl;
        continue;
      }

      if (show_images) {
        imshow("image", img);
        waitKey(100);
      }

      imgs.push_back(img);
    }
  }
}

void loadTestingDataset(const string &dirname, const string &label,
                     vector<Mat> &imgs, bool show_images) {
  vector<cv::String> files;
  char k = '3';
  // for (char k = '1'; k <= '2'; ++k) {
  try {
    glob(dirname + "/FOLD " + k + '/' + label, files);
  } catch (exception &e) {
    cout << "Error loading dataset/FOLD " << k << "\n\nExit\n";
    exit(1);
  }

  for (size_t i = 0; i < files.size(); ++i) {
    Mat img = imread(files[i]);
    if (img.empty()) {
      cout << files.at(i) << " is not valid." << endl;
      continue;
    }

    if (show_images) {
      imshow("image", img);
      waitKey(100);
    }

    imgs.push_back(img);
  }
  // }
}
