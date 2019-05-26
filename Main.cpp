#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <algorithm>
#include <sstream>
#include <fstream>
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
#include <opencv2/video.hpp>

//Programmer's created header files
#include "FaceSwapping.hpp"
#include "FaceDetect.hpp"
#include "FaceFilter.hpp"
#include "SaveFrame.hpp"
#include "ColorMapping.hpp"

using namespace std;
using namespace cv;


enum Mode { normal , swap , mask , save, colormap };

int main()
{

	const unsigned int faceinframe = 2;
	
	string text = "Enter ESC to exit";
	FaceDetector facedetect("C:\\Users\\TCCOM\\Documents\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt.xml", 0, faceinframe);
	FaceFilter filter("Mask1.jpg");
	FaceSwapper face_swapper("C:\\Users\\TCCOM\\Documents\\dlib-19.4\\shape_predictor_68_face_landmarks.dat");
	Mode mode = normal;

	ColorMapping colormap(BONE);

	facedetect.instruction();
	try {
		while (true)
		{
			Mat ColorMapFrame;
			Mat Frame;
			facedetect >> Frame;
			SaveImage save(Frame, "Swapper.png");

			putText(Frame, text, cvPoint(20, 20), FONT_HERSHEY_TRIPLEX, 0.8, cvScalar(0, 255, 0), 1, CV_AA);

			vector<Rect> cv_faces = facedetect.faces();

			switch (mode)
			{
			case normal:
				break;

			case Mode::swap:

				if (cv_faces.size() == faceinframe)
					face_swapper.swapFaces(Frame, cv_faces[0], cv_faces[1]);
				break;

			case Mode::mask:
				for (int i = 0; i < cv_faces.size(); i++)
				{
					Point center(cv_faces[i].x + cv_faces[i].width*0.5, cv_faces[i].y + cv_faces[i].height*0.5);
					filter.CreateMask(Frame, center, Size(cv_faces[i].width, cv_faces[i].height));
				}
				break;

			case Mode::colormap:
				colormap.ConvertColor(Frame, Frame);
				break;
			}

			imshow("Face Swap", Frame);

			switch (cv::waitKey(1))
			{
				//Mode !
			case 'a':
				mode = Mode::swap;
				break;

			case 'b':
				mode = Mode::mask;
				break;
			case 'c':
				mode = Mode::colormap;
				break;

				//Action !
			case '1':
				filter.setMask("Mask1.jpg");
				break;
			case '2':
				filter.setMask("Mask2.jpg");
				break;

			case 'q':
				colormap.setColorTone(BONE);
				break;
			case 'w':
				colormap.setColorTone(WINTER);
				break;
			case 'e':
				colormap.setColorTone(OCEAN);
				break;
			case 'r':
				colormap.setColorTone(SUMMER);
				break;
			case 't':
				colormap.setColorTone(COOL);
				break;
			case 'y':
				colormap.setColorTone(HSV);
				break;
			case 'u':
				colormap.setColorTone(HOT);
				break;

			case 's':
				save.setSaveFile();
				save.SaveImagetoFile(Frame);
				break;

			case 27:
				return 0;
				break;

			default:
				break;
			}
		}
	}
	catch (exception& e)
	{
		cout << e.what() << endl;
	}
	
	
}