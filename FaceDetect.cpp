#include "FaceDetect.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>

FaceDetector::FaceDetector(const std::string cascadeFilePath, const int cameraIndex, size_t numFaces)
{
	try {
		CamFrame = std::make_unique<cv::VideoCapture>(cameraIndex);
	}

	catch (exception &e)
	{
		if (CamFrame->isOpened() == false)
		{
			std::cerr << "Failed opening camera" << std::endl;
			cout << "Error: " << e.what();
			exit(-1);
		}
	}

	try
	{

		Face_Cascade = std::make_unique<cv::CascadeClassifier>(cascadeFilePath);
	}

	catch (exception &e) {
		if (Face_Cascade->empty())
		{
			std::cerr << "Error loading cascade file " << cascadeFilePath << endl <<
				"Make sure the file exists" << endl;
			cout << "Error: " << e.what() << endl;
			system("pause");
			//exit(-1);
		}
	}

	OriginalFrameSize.width = (int)CamFrame->get(cv::CAP_PROP_FRAME_WIDTH);
	OriginalFrameSize.height = (int)CamFrame->get(cv::CAP_PROP_FRAME_HEIGHT);


	AdjustedSize.width = FixedWidth;
	AdjustedSize.height = (AdjustedSize.width * OriginalFrameSize.height) / OriginalFrameSize.width;

	RatioForAdjustment.x = (double)OriginalFrameSize.width / AdjustedSize.width;
	RatioForAdjustment.y = (double)OriginalFrameSize.height / AdjustedSize.height;

	FaceFound = numFaces;
}

FaceDetector::~FaceDetector()
{
}

void FaceDetector:: instruction()
{
	cout << "////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////" << endl;
	cout << "////////////////////                   WELCOME TO PROJECT SNOOP PHOTO                            ///////////////////////" << endl;
	cout << "////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////" << endl << endl <<endl;
	cout << "\t\t This Project consist of 3 main features : " << endl << "\t\t 1 ) Face Swapper(Press A)" << endl << "\t\t 2 ) Mask (Press B)" << endl << "\t\t 3 ) Change Color (Press C)" << endl;
	cout << endl << endl << endl << "\t\t There are some option to play with!!" << endl;
	cout << endl << endl << "\t\t [[[[[[[[[   Mask Option   ]]]]]]]]]]" << endl << endl << "\t\t Press 1 : Toxic Mask" << "\t\t Press 2 : Wooden Mask" << endl << endl;
	cout << endl << endl << "\t\t              Window Color Option              " << endl << endl << "\t\t Press q  : Bone " << endl << "\t\t Press w  : Winter" << endl;
	cout << "\t\t Press e  : Ocean" << endl << "\t\t Press r  : Summer" << endl << "\t\t Press t  : COOL" << endl << "\t\t Press y  : HSV " << endl << "\t\t Press u  : Hot" << endl;
	cout << endl;
	cout << "///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////";
}
void FaceDetector::operator >> (cv::Mat &frame)
{
	if (CamFrame->isOpened() == false)
	{
		frame.release();
		return;
	}
	*CamFrame >> frame;

	cv::resize(frame, AdjustedFrame, AdjustedSize);

	if (!Istracking) // Search for faces on whole frame until 2 faces are found
	{
		detect();
		return;
	}
	else // if (m_tracking)
	{
		track();
	}
}

std::vector<cv::Rect> FaceDetector::faces()
{
	vector<cv::Rect> faces;

	faces.clear();
	for (const Rect& face : FaceRects)
	{
		int AdjustX = face.x * RatioForAdjustment.x;
		int AdjustY = face.y * RatioForAdjustment.y;
		int AdjustW = face.width * RatioForAdjustment.x;
		int AdjustH = face.height * RatioForAdjustment.y;

		faces.push_back(cv::Rect(AdjustX, AdjustY, AdjustW, AdjustH));
	}
	return faces;
}

void FaceDetector :: detect()
{
	//Min face size is 1/5 of screen height
	int getMinFaceWidth = AdjustedFrame.rows / 5;
	int getMinFaceHeight = AdjustedFrame.rows / 5;
	//Max face size must be 2/3 of screen height
	int getMaxFaceWidth = AdjustedFrame.rows * 2 / 3;
	int getMaxFaceHeight = AdjustedFrame.rows * 2 / 3;

	Face_Cascade ->detectMultiScale(AdjustedFrame, FaceRects, 1.1, 3, 0,
		cv::Size(getMinFaceWidth, getMinFaceHeight), cv::Size(getMaxFaceWidth, getMaxFaceHeight));

	if (FaceRects.size() < FaceFound)
	{
		return;
	}
	else if (FaceRects.size() >= FaceFound)
	{
		FaceRects.resize(FaceFound);
	}

	// Get face templates
	FaceTemp.clear();
	for (cv::Rect face : FaceRects)
	{
		face.width /= 2;
		face.height /= 2;
		face.x += face.width / 2;
		face.y += face.height / 2;

		//Duplicate the downscaledframe and put to face_template
		FaceTemp.push_back(AdjustedFrame(face).clone());
	}

	// Get face ROIs
	FaceRegionOfInterest.clear();
	for (const cv::Rect& face : FaceRects)
	{
		FaceRegionOfInterest.push_back(doubleRectSize(face, AdjustedSize));
	}

	// Initialize template matching timers
	INRoi.clear();
	BeginAddface.clear();
	StopAddFace.clear();

	INRoi.resize(FaceRects.size(), false);
	BeginAddface.resize(FaceRects.size());
	StopAddFace.resize(FaceRects.size());

	// Turn on tracking
	Istracking = true;
}

void FaceDetector :: track()
{
	for (int i = 0; i < FaceRegionOfInterest.size(); i++)
	{
		const cv::Rect &roi = FaceRegionOfInterest[i]; // roi

		//Min FaceWidthROI 4/10 of detected face & Min FaceHeightROI 4/10 of the detected face
		int getMinROIW = roi.width * 4 / 10;
		int getMinROIH = roi.height * 4 / 10;
		int getMaxROIW = roi.width * 6 / 10;
		int getMaxROIH = roi.width * 6 / 10;
										 // Detect faces sized +/-20% off biggest face in previous search
		const cv::Mat &faceRoi = AdjustedFrame(roi);
		Face_Cascade ->detectMultiScale(faceRoi, TempFacesRect, 1.1, 3, 0,
			cv::Size(getMinROIW,getMinROIH), cv::Size(getMaxROIW, getMaxROIH));

		if (TempFacesRect.empty()) //In case that no face can be detected
		{
			if (BeginAddface[i] == 0) // Start the time for adding face
			{
				BeginAddface[i] = cv::getCPUTickCount();  // Use as timer
			}

			if (FaceTemp[i].cols <= 1 || FaceTemp[i].rows <= 1)
			{
				FaceRects.clear();
				Istracking = false;
				return;
			}

			// Template matching

			cv::matchTemplate(faceRoi, FaceTemp[i], MatchingResult, CV_TM_SQDIFF_NORMED);
			cv::normalize(MatchingResult, MatchingResult, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
			double min, max;
			cv::Point minLoc, maxLoc;
			cv::minMaxLoc(MatchingResult, &min, &max, &minLoc, &maxLoc);

			// Add roi offset to face position
			FaceRects[i].x = minLoc.x + roi.x - FaceTemp[i].cols / 2;
			FaceRects[i].y = minLoc.y + roi.y - FaceTemp[i].rows / 2;
			FaceRects[i].width = FaceTemp[i].cols * 2;
			FaceRects[i].height = FaceTemp[i].rows * 2;

			StopAddFace[i] = cv::getCPUTickCount();

			double duration = (double)(StopAddFace[i] - BeginAddface[i]) / cv::getTickFrequency();
			if (duration > MaxTimeOnFrame)
			{
				FaceRects.clear();
				Istracking = false;
				return; // Stop tracking faces
			}
		}
		else
		{
			INRoi[i] = false;

			BeginAddface[i] = 0;
			StopAddFace[i] = 0;

			FaceRects[i] = TempFacesRect[0];

			FaceRects[i].x += roi.x;
			FaceRects[i].y += roi.y;
		}
	}

	for (int i = 0; i < FaceRects.size(); i++)
	{
		for (int j = i + 1; j < FaceRects.size(); j++)
		{
			if ((FaceRects[i] & FaceRects[j]).area() > 0)
			{
				FaceRects.clear();
				Istracking = false;
				return;
			}
		}
	}
}

cv::Rect FaceDetector :: doubleRectSize(const cv::Rect &inputRect, const cv::Size &frameSize)
{
	cv::Rect outputRect;
	// Double rect size
	outputRect.width = inputRect.width * 2;
	outputRect.height = inputRect.height * 2;

	// Center rect around original center
	outputRect.x = inputRect.x - inputRect.width / 2;
	outputRect.y = inputRect.y - inputRect.height / 2;

	// Handle edge cases
	if (outputRect.x < 0) {
		outputRect.width += outputRect.x;
		outputRect.x = 0;
	}

	if (outputRect.y < 0) {
		outputRect.height += outputRect.y;
		outputRect.y = 0;
	}

	if (outputRect.x + outputRect.width > frameSize.width) {
		outputRect.width = frameSize.width - outputRect.x;
	}
	if (outputRect.y + outputRect.height > frameSize.height) {
		outputRect.height = frameSize.height - outputRect.y;
	}

	return outputRect;
}
