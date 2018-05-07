#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;

int main() {
    Mat box = Mat::zeros(220, 450, CV_8UC3);
    namedWindow("Hello!");
    putText(box, "Hello world!!", Point(50, 50),
            FONT_HERSHEY_COMPLEX, 1, Scalar(255, 255, 255));
    imshow("Hello!", box);

    // select the window and press a key
    waitKey(0);

    return 0;
}