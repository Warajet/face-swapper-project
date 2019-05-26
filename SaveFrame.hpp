#pragma once

#ifndef SAVEFRAME_HPP
#define SAVEFRAME_HPP

#include <iostream>

#include <sstream>
#include <fstream>
#include <vector>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

class SaveImage
{

public:
	SaveImage(Mat& frame,string imagename);

	SaveImage(string Path,string imagename, Mat& frame);

	void setFolderPath(string Path);

	void printFilename();

	string getFolderPath();

	Mat getFrame();

	vector<int> getImageQuality();

	void setImageName(string imagename) { filename = imagename; }

	void SaveImagetoFile(Mat Frame);

	string intToString(int number);

	void setSaveFile();

private:
	int imagecount;
	int filenum;
	vector<int> imagequality;
	Mat Frame;
	string folderpath;
	string filename;
};
#endif