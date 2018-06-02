#include "svm.hpp"
#include <iomanip>
#include <vector>
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"

#ifndef CV_TERMCRIT_ITER
#define CV_TERMCRIT_ITER 1
#endif

#ifndef CV_TERMCRIT_EPS
#define CV_TERMCRIT_EPS 2
#endif

using namespace std;
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

void crossValidation(cv::Ptr<cv::ml::SVM> &svm, vector<Mat> data,
                     vector<vector<int>> labels) {
  unsigned int final_accuracy = 0;

  for (uint8_t k = 0; k < 3; ++k) {
    Mat train_data, test_data;
    vector<int> train_labels, test_labels;
    cout << "Epoch " << k + 1 << endl;

    train_data.push_back(data.at(k));
    train_data.push_back(data.at((k + 1) % 3));
    train_labels.insert(train_labels.end(), labels.at(k).begin(),
                        labels.at(k).end());
    train_labels.insert(train_labels.end(), labels.at((k + 1) % 3).begin(),
                        labels.at((k + 1) % 3).end());
    test_data.push_back(data.at((k + 2) % 3));
    test_labels.insert(test_labels.end(), labels.at((k + 2) % 3).begin(),
                       labels.at((k + 2) % 3).end());

    // cout << "train_data size " << train_data.size() << endl;
    // cout << "train_labels size " << train_labels.size() << endl;
    // cout << "test_data size " << test_data.size() << endl;
    // cout << "test_labels size " << test_labels.size() << endl;

    cout << "Training..." << flush;

    svm->train(train_data, ROW_SAMPLE, train_labels);

    cout << "  Testing...\n";

    unsigned int matches = 0;
    for (size_t i = 0; i < test_labels.size(); ++i) {
      if (svm->predict(test_data.row(i)) == test_labels[i]) ++matches;
    }

    cout << endl << matches << " matches out of " << test_labels.size() << endl;
    cout << "Accuracy " << fixed << setprecision(2)
         << (float(matches) / test_labels.size() * 100) << '%' << endl;
    final_accuracy = final_accuracy + float(matches) / test_labels.size() * 100;
  }

  cout << "Total Accuracy " << float(final_accuracy) / 3 << '%' << endl;
}
