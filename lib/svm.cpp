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
  svm->setCoef0(0.0);
  svm->setDegree(4);
  svm->setTermCriteria(
      TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, 1e-3));
  svm->setGamma(3);
  // svm->setKernel(SVM::LINEAR);
  svm->setKernel(SVM::POLY);
  svm->setNu(0.5);
  svm->setP(0.1);  // for EPSILON_SVR, epsilon in loss function?
  svm->setC(0.01); // From paper, soft classifier
  // svm->setType(SVM::EPS_SVR);  // C_SVC; // EPSILON_SVR; // may be also
  // NU_SVR;
  svm->setType(SVM::C_SVC);
}