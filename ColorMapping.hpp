#pragma once

#define _CRT_SECURE_NO_WARNINGS
#ifndef COLORMAPPING_HPP
#define COLORMAPPING_HPP

#include <iostream>

#include <string>
#include <vector>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>

using namespace std;
using namespace cv;

enum ListColor
{
	BONE = 1,
	COOL = 2,
	HOT = 3,
	HSV = 4,
	OCEAN = 6,
	SUMMER = 10,
	WINTER = 11
};

class ColorMapping
{
public: 
	ColorMapping(ListColor cmap);
	void ConvertColor(Mat inputframe,Mat outputFrame);
	void setColorTone(ListColor color);

private:
	int colorframe;
	Mat inputFrame;
	Mat OutputFrmae;
	ListColor colorlist;
};


#endif