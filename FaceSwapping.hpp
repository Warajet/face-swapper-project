#pragma once
#define _CRT_SECURE_NO_WARNINGS
#ifndef FACESWAPPING_HPP
#define FACESWAPPING_HPP
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <iostream>

#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>


class FaceSwapper
{
public:
	// Initialize face swapped with landmarks
	FaceSwapper(const std::string landmarkFile);
	~FaceSwapper();

	//Swaps faces in rects on frame
	void swapFaces(cv::Mat &frame, cv::Rect &face1, cv::Rect &face2);

	// Returns minimal Mat containing both faces
	cv::Mat getMinFrame(const cv::Mat &frame, cv::Rect &rect_face1, cv::Rect &rect_face2);

	// Get useful face position
	void getFacePosition(const cv::Mat &frame);

	// Reshape the matrix of pixels in an image on particular points on each face
	void getTransformationMatrices();

	// Create the mask obtained in frame along with face position
	void getMasks();

	// Create the reshaped mask
	void getReshapedMasks();

	// Return the mask that has been adjusted and reshaped (Note : Not bigger than original face)
	cv::Mat getAdjusted_ReshapedMasks();

	// Extract the face from getmask
	void extractFaces();

	// Create the reshaped face obtained from function extractfaces
	cv::Mat getReshapedFaces();

	// Match the tone of each inviduals' faces
	void colorCorrectFaces();

	// Blur the edge of the mask being transfered on individuals faces
	void featherMask(cv::Mat &refined_masks);

	// Paste each face on the original frame
	void pasteFacesOnFrame();

	// Calculates source image histogram and changes target_image to match source hist
	//Adjust the color to have similar level to the face of a person that has been swapped
	void CalculateHistogram(const cv::Mat source_image, cv::Mat target_image, cv::Mat mask);

private:
	
	cv::Rect face1, face2;  //
	cv::Rect ReshapedFace1; //
	cv::Rect ReshapedFace2; //

	dlib::shape_predictor face_model;  //
	dlib::full_object_detection ShapeDetection[2];  //
	dlib::rectangle dlib_rects[2];     //
	dlib::cv_image<dlib::bgr_pixel> dlib_frame;    //
	cv::Point2f affine_transform_keypoints_Face1[3], affine_transform_keypoints_Face2[3];  //

	cv::Mat Alpha_Reshaped_face1_to2, Alpha_Reshaped_face2_to1;  //Alpha and reshaped face 1 -> 2 and face 2 -> 1
	cv::Mat warpped_Face1, warpped_Face2;  //

	cv::Point2i Points_on_Face1[9], Points_on_Face2[9];  //
	cv::Mat Add_Face1_to_Face2, Add_Face2_to_Face1;  //
	cv::Mat Face1_Mask, Face2_Mask;  //
	cv::Mat ReshapedMask_face1, ReshapedMask_face2; //Using image warping
	cv::Mat refined_masks;         //Alpha_reshaped that has been copied an put to refined_mask container
	cv::Mat RealFace1, RealFace2;  //
	cv::Mat AllReshaped_face;//

	cv::Mat AdjustedFrame;  //being adjust to smaller size to allow all methods operate faster

	cv::Size FrameSize;   //Frame Size

	cv::Size feather_amount;

	uint8_t LookUPTable[3][256];  //Lookup table
	int source_hist_int[3][256];
	int target_hist_int[3][256];
	float source_histogram[3][256];
	float target_histogram[3][256];
};
#endif