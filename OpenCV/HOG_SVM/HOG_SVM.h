#pragma once
#include <cv.h>
#include <opencv2/objdetect/objdetect.hpp>
#include <ml.h>
#include <cxcore.h>
#include <cvaux.h>
#include "highgui.h"


using namespace cv ;

class HOG_SVM //: 	public HOGDescriptor
{
public:
	HOGDescriptor hog;
	CvSVM svm;

void HOG_SVM::detect(const Mat& img, vector<Point>& hits, vector<double>& weights, double hitThreshold=0, 
						Size winStride=Size(), Size padding=Size(), const vector<Point>& locations = vector<Point>()) const ;

void HOG_SVM::detect(const Mat& img, vector<Point>& hits, double hitThreshold=0, 
						Size winStride=Size(), Size padding=Size(), const vector<Point>& locations = vector<Point>()) const ;

# if 1
void HOG_SVM::detectMultiScale(
    const Mat& img, vector<Rect>& foundLocations, vector<double>& foundWeights,
    double hitThreshold, Size winStride, Size padding,
    double scale0, double finalThreshold, bool useMeanshiftGrouping) const  ;

void HOG_SVM::detectMultiScale(const Mat& img, vector<Rect>& foundLocations, 
                                     double hitThreshold, Size winStride, Size padding,
                                     double scale0, double finalThreshold, bool useMeanshiftGrouping) const  ;

#endif

	HOG_SVM(void);
	~HOG_SVM(void);
};
