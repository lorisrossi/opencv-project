#include <iostream>
#include <vector>
#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;
using namespace cv;

void computeHOG(const vector<Mat> &imgs) {
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

      if ((i+1) % 10 == 0) {
        cout << i + 1 << " image has " << descriptors.size() << " descriptors\n";
      }

    }
  }
}