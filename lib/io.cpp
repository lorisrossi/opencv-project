#include "io.hpp"

using namespace std;
using namespace cv;

void loadDataset(const string &dirname, vector<string> &labels,
                 vector<vector<Mat>> &imgs, const string &type,
                 bool show_images) {
  vector<cv::String> files;
  for (size_t j = 0; j < labels.size(); ++j) {
    for (char k = '1'; k <= '2'; ++k) {
      if(type == "test") k = '3';
      
      try {
        glob(dirname + "/FOLD " + k + '/' + labels.at(j), files);
      } catch (exception &e) {
        cout << "Error loading dataset/FOLD " << k << "\n\nExit\n";
        exit(1);
      }

      for (size_t i = 0; i < files.size(); ++i) {
        Mat img = imread(files.at(i));
        if (img.empty()) {
          cout << files.at(i) << " is not valid." << endl;
          continue;
        }

        if (show_images) {
          imshow("image", img);
          waitKey(100);
        }

        imgs.at(j).push_back(img);
      }
    }
  }
}