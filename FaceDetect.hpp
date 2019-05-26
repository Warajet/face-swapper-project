#pragma once
#define _CRT_SECURE_NO_WARNINGS
#ifndef FACEDETECT_HPP
#define FACEDETECT_HPP
#include "FaceFilter.hpp"

#include <opencv2/core/core.hpp>
#include <vector>
#include <string>
#include <memory>

namespace cv
{
	class VideoCapture;
	class CascadeClassifier;
}

class FaceDetector
{
public:
	
	//Initializes detector with cascade file, initializes camera with camera index and sets number of faces to track
	
	FaceDetector(const std::string cascadeFilePath, const int cameraIndex, size_t numFaces);
	~FaceDetector();

	//Return next frame and operate face detection
	void operator >> (cv::Mat &frame);

	// Return all face detected in the form of vector

	std:: vector<cv::Rect> faces(); //
	static cv::Rect doubleRectSize(const cv::Rect &rect, const cv::Size &frameSize);

	//Face Detection and Tracking
	void detect();
	void track();

	//Program's instruction
	virtual void instruction();

private:

	//VideoCapture for camera 
	std::unique_ptr<cv::VideoCapture> CamFrame;

	//Create Cascade Classifier used for facial detection
	std::unique_ptr<cv::CascadeClassifier> Face_Cascade;

	//Adjust the frame to smaller size in order to speed up the swap operation
	cv::Mat AdjustedFrame;

	//Fixed Width of the downscaled frame
	const int FixedWidth = 256;

	//Vector used to cover the face in a camera frame
	std::vector<cv::Rect> FaceRects;

	//Vector of the face vector used for tracking a face in the camera frame (Note : 1 vector per 1 detected face)
	std::vector<cv::Rect> TempFacesRect;  //Stored a deteced face as the temp face rect

	//Used to determin if the pixels are in region of interest
	std::vector<bool>                       INRoi;

	//Store the face as face template
	std::vector<cv::Mat>                    FaceTemp;

	//Rectangular area covering facial region of interest
	std::vector<cv::Rect>                   FaceRegionOfInterest;

	//Matching the output of face detected
	cv::Mat                                 MatchingResult;

	//Adjust framesize
	cv::Size                                AdjustedSize;

	//Original Frame Size
	cv::Size                                OriginalFrameSize;
	cv::Point2f                             RatioForAdjustment;
	
	//Used to determine the status of face detection
	bool                                    Istracking = false;

	//Number of face being found in a frame
	unsigned int FaceFound = 0; 

	//Control the time for face detection
	//Time being measured in millisecond Thus, long long must be used

	std::vector<long long>					BeginAddface;
	std::vector<long long>                  StopAddFace;
	const double                            MaxTimeOnFrame = 2.0;

};

#endif