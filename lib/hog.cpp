#include <iostream>
#include <vector>
#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;
using namespace cv;

void computeHOG(const vector<Mat> &imgs, vector<Mat> &gradients) {
  HOGDescriptor hog;
  hog.winSize = imgs[0].size() / 8 * 8;
  vector<float> descriptors;
  Mat gray;

  for (size_t i = 0; i < imgs.size(); ++i) {
    if (imgs[i].rows >= hog.winSize.height && imgs[i].cols >= hog.winSize.width) {
      Rect r = Rect((imgs[i].rows - hog.winSize.height) / 2,
                    (imgs[i].cols - hog.winSize.width) / 2,
                    hog.winSize.width,
                    hog.winSize.height);
      
      cvtColor(imgs[i](r), gray, COLOR_BGR2GRAY);
      hog.compute(gray, descriptors, Size(8, 8), Size(0, 0));
      gradients.push_back(Mat(descriptors).t());
    }
  }
}