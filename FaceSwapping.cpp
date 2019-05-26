#include "FaceSwapping.hpp"

#include <iostream>
using namespace std;

FaceSwapper::FaceSwapper(const std::string landmarks_path)
{
	try
	{
		dlib::deserialize(landmarks_path) >> face_model;
	}
	catch (std::exception& e)
	{
		std::cerr << "Error loading landmarks from " << landmarks_path << endl;
		exit(-1);
	}
}


FaceSwapper::~FaceSwapper()
{
}

void FaceSwapper::swapFaces(cv::Mat &frame, cv::Rect &Face1, cv::Rect &Face2)
{
	AdjustedFrame = getMinFrame(frame, Face1, Face2);

	FrameSize = cv::Size(AdjustedFrame.cols, AdjustedFrame.rows);

	getFacePosition(AdjustedFrame);

	getTransformationMatrices();

	Face1_Mask.create(FrameSize, CV_8UC1);
	Face2_Mask.create(FrameSize, CV_8UC1);
	getMasks();

	getReshapedMasks();

	refined_masks = getAdjusted_ReshapedMasks();

	extractFaces();

	AllReshaped_face = getReshapedFaces();

	colorCorrectFaces();

	cv::Mat refined_mask_ann = refined_masks(ReshapedFace1);
	cv::Mat refined_mask_bob = refined_masks(ReshapedFace2);
	featherMask(refined_mask_ann);
	featherMask(refined_mask_bob);

	pasteFacesOnFrame();
}

cv::Mat FaceSwapper::getMinFrame(const cv::Mat &frame, cv::Rect &Face1, cv::Rect &Face2)
{
	cv::Rect bounding_rect = Face1 | Face2;

	bounding_rect -= cv::Point(50, 50);
	bounding_rect += cv::Size(100, 100);

	bounding_rect &= cv::Rect(0, 0, frame.cols, frame.rows);

	this->face1 = Face1 - bounding_rect.tl();
	this->face2 = Face2 - bounding_rect.tl();

	ReshapedFace1 = ((this->face1 - cv::Point(Face1.width / 4, Face1.height / 4)) + cv::Size(Face1.width / 2, Face1.height / 2)) & cv::Rect(0, 0, bounding_rect.width, bounding_rect.height);
	ReshapedFace2 = ((this->face2 - cv::Point(Face2.width / 4, Face2.height / 4)) + cv::Size(Face2.width / 2, Face2.height / 2)) & cv::Rect(0, 0, bounding_rect.width, bounding_rect.height);

	return frame(bounding_rect);
}

void FaceSwapper::getFacePosition(const cv::Mat &frame)
{
	using namespace dlib;

	dlib_rects[0] = rectangle(face1.x, face1.y, face1.x + face1.width, face2.y + face2.height);
	dlib_rects[1] = rectangle(face2.x, face2.y, face2.x + face2.width, face2.y + face2.height);

	dlib_frame = frame;

	ShapeDetection[0] = face_model(dlib_frame, dlib_rects[0]);
	ShapeDetection[1] = face_model(dlib_frame, dlib_rects[1]);

	auto getPoint = [&](int ShapeDetected, int PointOnFace) -> const cv::Point2i
	{
		const auto &p = ShapeDetection[ShapeDetected].part(PointOnFace);

	return cv::Point2i(p.x(), p.y());
	};

	Points_on_Face1[0] = getPoint(0, 0);
	Points_on_Face1[1] = getPoint(0, 3);
	Points_on_Face1[2] = getPoint(0, 5);
	Points_on_Face1[3] = getPoint(0, 8);
	Points_on_Face1[4] = getPoint(0, 11);
	Points_on_Face1[5] = getPoint(0, 13);
	Points_on_Face1[6] = getPoint(0, 16);

	cv::Point2i nose_length = getPoint(0, 27) - getPoint(0, 30);
	Points_on_Face1[7] = getPoint(0, 26) + nose_length;
	Points_on_Face1[8] = getPoint(0, 17) + nose_length;


	Points_on_Face2[0] = getPoint(1, 0);
	Points_on_Face2[1] = getPoint(1, 3);
	Points_on_Face2[2] = getPoint(1, 5);
	Points_on_Face2[3] = getPoint(1, 8);
	Points_on_Face2[4] = getPoint(1, 11);
	Points_on_Face2[5] = getPoint(1, 13);
	Points_on_Face2[6] = getPoint(1, 16);

	nose_length = getPoint(1, 27) - getPoint(1, 30);
	Points_on_Face2[7] = getPoint(1, 26) + nose_length;
	Points_on_Face2[8] = getPoint(1, 17) + nose_length;

	affine_transform_keypoints_Face1[0] = Points_on_Face1[3];
	affine_transform_keypoints_Face1[1] = getPoint(0, 36);
	affine_transform_keypoints_Face1[2] = getPoint(0, 45);

	affine_transform_keypoints_Face2[0] = Points_on_Face2[3];
	affine_transform_keypoints_Face2[1] = getPoint(1, 36);
	affine_transform_keypoints_Face2[2] = getPoint(1, 45);

	feather_amount.width = feather_amount.height = (int)cv::norm(Points_on_Face1[0] - Points_on_Face1[6]) / 8;
}

void FaceSwapper::getTransformationMatrices()
{
	Add_Face1_to_Face2 = cv::getAffineTransform(affine_transform_keypoints_Face1, affine_transform_keypoints_Face2);
	cv::invertAffineTransform(Add_Face1_to_Face2, Add_Face2_to_Face1);
}

void FaceSwapper::getMasks()
{
	Face1_Mask.setTo(cv::Scalar::all(0));
	Face2_Mask.setTo(cv::Scalar::all(0));

	cv::fillConvexPoly(Face1_Mask, Points_on_Face1, 9, cv::Scalar(255));
	cv::fillConvexPoly(Face2_Mask, Points_on_Face2, 9, cv::Scalar(255));
}

void FaceSwapper::getReshapedMasks()
{
	cv::warpAffine(Face1_Mask, ReshapedMask_face1, Add_Face1_to_Face2, FrameSize, cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));
	cv::warpAffine(Face2_Mask, ReshapedMask_face2, Add_Face2_to_Face1, FrameSize, cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));
}

cv::Mat FaceSwapper::getAdjusted_ReshapedMasks()
{
	//Obtain the black region due to both face mask and reshapedmask
	cv::bitwise_and(Face1_Mask, ReshapedMask_face2, Alpha_Reshaped_face1_to2);
	cv::bitwise_and(Face2_Mask, ReshapedMask_face1, Alpha_Reshaped_face2_to1);

	cv::Mat refined_masks(FrameSize, CV_8UC1, cv::Scalar(0));
	Alpha_Reshaped_face1_to2.copyTo(refined_masks, Alpha_Reshaped_face1_to2);
	Alpha_Reshaped_face2_to1.copyTo(refined_masks, Alpha_Reshaped_face2_to1);

	return refined_masks;
}

void FaceSwapper::extractFaces()
{
	AdjustedFrame.copyTo(RealFace1, Face1_Mask);
	AdjustedFrame.copyTo(RealFace2, Face2_Mask);
}

cv::Mat FaceSwapper::getReshapedFaces()
{
	cv::Mat warpped_faces(FrameSize, CV_8UC3, cv::Scalar::all(0));

	cv::warpAffine(RealFace1, warpped_Face1, Add_Face1_to_Face2, FrameSize, cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
	cv::warpAffine(RealFace2, warpped_Face2, Add_Face2_to_Face1, FrameSize, cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

	warpped_Face1.copyTo(warpped_faces, ReshapedMask_face1);
	warpped_Face2.copyTo(warpped_faces, ReshapedMask_face2);

	return warpped_faces;
}

void FaceSwapper::colorCorrectFaces()
{
	CalculateHistogram(AdjustedFrame(ReshapedFace1), AllReshaped_face(ReshapedFace1), ReshapedMask_face2(ReshapedFace1));
	CalculateHistogram(AdjustedFrame(ReshapedFace2), AllReshaped_face(ReshapedFace2), ReshapedMask_face1(ReshapedFace2));
}

void FaceSwapper::featherMask(cv::Mat &refined_masks)
{
	cv::erode(refined_masks, refined_masks, getStructuringElement(cv::MORPH_RECT, feather_amount), cv::Point(-1, -1), 1, cv::BORDER_CONSTANT, cv::Scalar(0));

	cv::blur(refined_masks, refined_masks, feather_amount, cv::Point(-1, -1), cv::BORDER_CONSTANT);
}

inline void FaceSwapper::pasteFacesOnFrame()
{
	for (size_t i = 0; i < AdjustedFrame.rows; i++)
	{
		auto frame_pixel = AdjustedFrame.row(i).data;
		auto faces_pixel = AllReshaped_face.row(i).data;
		auto masks_pixel = refined_masks.row(i).data;

		for (size_t j = 0; j < AdjustedFrame.cols; j++)
		{
			if (*masks_pixel != 0)
			{
				*frame_pixel = ((255 - *masks_pixel) * (*frame_pixel) + (*masks_pixel) * (*faces_pixel)) / 256 ;
				*(frame_pixel + 1) = ((255 - *(masks_pixel + 1)) * (*(frame_pixel + 1)) + (*(masks_pixel + 1)) * (*(faces_pixel + 1))) / 256;
				*(frame_pixel + 2) = ((255 - *(masks_pixel + 2)) * (*(frame_pixel + 2)) + (*(masks_pixel + 2)) * (*(faces_pixel + 2))) / 256;
			}

			frame_pixel += 3;
			faces_pixel += 3;
			masks_pixel++;
		}
	}
}

void FaceSwapper::CalculateHistogram(const cv::Mat source_image, cv::Mat target_image, cv::Mat mask)
{

	std::memset(source_hist_int, 0, sizeof(int) * 3 * 256);
	std::memset(target_hist_int, 0, sizeof(int) * 3 * 256);

	for (size_t i = 0; i < mask.rows; i++)
	{
		auto current_mask_pixel = mask.row(i).data;
		auto current_source_pixel = source_image.row(i).data;
		auto current_target_pixel = target_image.row(i).data;

		for (size_t j = 0; j < mask.cols; j++)
		{
			if (*current_mask_pixel != 0) {
				source_hist_int[0][*current_source_pixel]++;
				source_hist_int[1][*(current_source_pixel + 1)]++;
				source_hist_int[2][*(current_source_pixel + 2)]++;

				target_hist_int[0][*current_target_pixel]++;
				target_hist_int[1][*(current_target_pixel + 1)]++;
				target_hist_int[2][*(current_target_pixel + 2)]++;
			}

			// Advance to next pixel
			current_source_pixel += 3;
			current_target_pixel += 3;
			current_mask_pixel++;
		}
	}

	// Calc CDF
	for (size_t i = 1; i < 256; i++)
	{
		source_hist_int[0][i] += source_hist_int[0][i - 1];
		source_hist_int[1][i] += source_hist_int[1][i - 1];
		source_hist_int[2][i] += source_hist_int[2][i - 1];

		target_hist_int[0][i] += target_hist_int[0][i - 1];
		target_hist_int[1][i] += target_hist_int[1][i - 1];
		target_hist_int[2][i] += target_hist_int[2][i - 1];
	}

	// Normalize CDF
	for (size_t i = 0; i < 256; i++)
	{
		source_histogram[0][i] = (source_hist_int[0][255] ? (float)source_hist_int[0][i] / source_hist_int[0][255] : 0);
		source_histogram[1][i] = (source_hist_int[1][255] ? (float)source_hist_int[1][i] / source_hist_int[1][255] : 0);
		source_histogram[2][i] = (source_hist_int[2][255] ? (float)source_hist_int[2][i] / source_hist_int[2][255] : 0);

		target_histogram[0][i] = (target_hist_int[0][255] ? (float)target_hist_int[0][i] / target_hist_int[0][255] : 0);
		target_histogram[1][i] = (target_hist_int[1][255] ? (float)target_hist_int[1][i] / target_hist_int[1][255] : 0);
		target_histogram[2][i] = (target_hist_int[2][255] ? (float)target_hist_int[2][i] / target_hist_int[2][255] : 0);
	}

	// Create lookup table

	auto binary_search = [&](const float needle, const float haystack[]) -> uint8_t
	{
		uint8_t l = 0, r = 255, m;
		while (l < r)
		{
			m = (l + r) / 2;
			if (needle > haystack[m])
				l = m + 1;
			else
				r = m - 1;
		}
		// TODO check closest value
		return m;
	};

	for (size_t i = 0; i < 256; i++)
	{
		LookUPTable[0][i] = binary_search(target_histogram[0][i], source_histogram[0]);
		LookUPTable[1][i] = binary_search(target_histogram[1][i], source_histogram[1]);
		LookUPTable[2][i] = binary_search(target_histogram[2][i], source_histogram[2]);
	}

	// repaint pixels
	for (size_t i = 0; i < mask.rows; i++)
	{
		auto current_mask_pixel = mask.row(i).data;
		auto current_target_pixel = target_image.row(i).data;
		for (size_t j = 0; j < mask.cols; j++)
		{
			if (*current_mask_pixel != 0)
			{
				*current_target_pixel = LookUPTable[0][*current_target_pixel];
				*(current_target_pixel + 1) = LookUPTable[1][*(current_target_pixel + 1)];
				*(current_target_pixel + 2) = LookUPTable[2][*(current_target_pixel + 2)];
			}

			// Advance to next pixel
			current_target_pixel += 3;
			current_mask_pixel++;
		}
	}
}
