#pragma once
#ifndef FACEFILTER_HPP
#define FACEFILTER_HPP
#include <iostream>
#include <string>
#include <vector>
#include "FaceDetect.hpp"
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

class FaceFilter
{
private:
	double MAX_FACE_SIZE = 200;
	double MIN_FACE_SIZE = 20;
	string facefile;
	vector<Rect> faces;
	Mat Mask;

	Mat ResizedMask;
	Mat TempSourceFrame;
	Mat TransparentWhite;
	Mat TransparentedMask;
	Mat MaskAlpha;

public:

	string getFaceFile() { return facefile; }
	void setMask(string f) { facefile = f; }
	void ReadMaskFile();
	FaceFilter(const string facefile);
	cv::Mat LocateTheMask(Mat frame);
	~FaceFilter();
	cv::Mat CreateMask(Mat& source_frame, Point face_centre, Size face_size);
};
#endif
