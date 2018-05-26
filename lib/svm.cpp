#include "svm.hpp"

#ifndef CV_TERMCRIT_ITER
#define CV_TERMCRIT_ITER 1
#endif

#ifndef CV_TERMCRIT_EPS
#define CV_TERMCRIT_EPS 2
#endif

using namespace cv;
using namespace cv::ml;

void initSvm(Ptr<SVM> &svm) {
  svm = SVM::create();
  svm->setCoef0(10.0);
  svm->setDegree(6);
  svm->setTermCriteria(
      TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, 1e-3));
  svm->setGamma(3);
  svm->setKernel(SVM::POLY);
  svm->setNu(0.5);
  svm->setP(0.1);
  svm->setC(0.01);
  svm->setType(SVM::NU_SVC);
}