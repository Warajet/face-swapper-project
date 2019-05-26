#include "FaceFilter.hpp"

FaceFilter:: FaceFilter(const string mask)
{
	facefile = "C:\\Users\\TCCOM\\Desktop\\YEAR1 TERM2\\C++ Slide\\OPENCVTESTER\\OPENCVTESTER\\" + mask;
}

FaceFilter::~FaceFilter()
{
}


void FaceFilter::ReadMaskFile()
{
	Mask = imread(facefile.c_str());
}

cv::Mat FaceFilter:: LocateTheMask(Mat frame)
{
	CascadeClassifier face_cascade("C:\\Users\\TCCOM\\Documents\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt.xml");
	face_cascade.detectMultiScale(frame, faces, 1.2, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(MIN_FACE_SIZE, MIN_FACE_SIZE), Size(MAX_FACE_SIZE, MAX_FACE_SIZE));
	
		// Draw circles on the detected faces
	for (int i = 0; i < faces.size(); i++)
	{
		MIN_FACE_SIZE = faces[i].width*0.7;
		MAX_FACE_SIZE = faces[i].width*1.5;
		Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);
		frame = CreateMask(frame, center, Size(faces[i].width, faces[i].height));
	}
	return frame;
}

cv::Mat FaceFilter::CreateMask(Mat& source_frame, Point centre, Size face_size)
{
	ReadMaskFile();

	cv::resize(Mask, ResizedMask, face_size);

	// ROI selection
	Rect RegionOfInterest(centre.x - face_size.width / 2, centre.y - face_size.width / 2, face_size.width, face_size.width);

	source_frame(RegionOfInterest).copyTo(TempSourceFrame);

	//Make white region transparent

	cv::cvtColor(ResizedMask, TransparentWhite, CV_BGR2GRAY);
	cv::threshold(TransparentWhite, TransparentWhite, 230, 255, CV_THRESH_BINARY_INV);

	vector<Mat> maskChannels(3), result_mask(3);

	//Separate the resized mask from mask channel
	cv::split(ResizedMask, maskChannels);

	cv::bitwise_and(maskChannels[0], TransparentWhite, result_mask[0]);  // Get transparent mask
	cv::bitwise_and(maskChannels[1], TransparentWhite, result_mask[1]);
	cv::bitwise_and(maskChannels[2], TransparentWhite, result_mask[2]);
	cv::merge(result_mask, TransparentedMask);

	TransparentWhite = 255 - TransparentWhite; // White color - mask itself

	vector<Mat> srcChannels(3);

	cv::split(TempSourceFrame, srcChannels);

	cv::bitwise_and(srcChannels[0], TransparentWhite, result_mask[0]);  //Get black mask region
	cv::bitwise_and(srcChannels[1], TransparentWhite, result_mask[1]);
	cv::bitwise_and(srcChannels[2], TransparentWhite, result_mask[2]);
	cv::merge(result_mask, MaskAlpha);

	cv::addWeighted(TransparentedMask, 1, MaskAlpha, 1, 0, MaskAlpha);  //(Source, alpha , maskalpha(src 2), beta, gamma, outputmask)

	MaskAlpha.copyTo(source_frame(RegionOfInterest));

	return source_frame;
}