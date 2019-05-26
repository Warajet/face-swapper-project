#include "SaveFrame.hpp";
#include <fstream>

SaveImage::SaveImage(Mat& frame,string imagename)
{
	ifstream in_file;
	in_file.open("C:\\Users\\TCCOM\\Desktop\\YEAR1 TERM2\\C++ Slide\\OPENCVTESTER\\OPENCVTESTER\\ProjectPic.txt");
	if (!in_file.is_open())
	{
		throw exception();
	}
	in_file >> imagecount;
	in_file.close();
	folderpath = "C:\\Users\\TCCOM\\Desktop\\YEAR1 TERM2\\C++ Slide\\OPENCVTESTER\\OPENCVTESTER\\SavedImage\\" + imagename;
	Frame = frame;
	imagequality.push_back(CV_IMWRITE_PNG_COMPRESSION);
	imagequality.push_back(98);

}

SaveImage::SaveImage(string path,string imagename, Mat& frame)
{
	folderpath = path + imagename;
	Frame = frame;
	imagequality.push_back(CV_IMWRITE_PNG_COMPRESSION);
	imagequality.push_back(98);
}


string SaveImage::getFolderPath()
{
	return folderpath;
}

vector<int> SaveImage:: getImageQuality() 
{
	return imagequality;
}

Mat SaveImage :: getFrame()
{
	return Frame;
}

string SaveImage :: intToString(int number) 
{

	std::stringstream ss;
	ss << number;
	return ss.str();
}

void SaveImage::setSaveFile()
{
	string outnum = intToString(imagecount);
	string filenum = outnum.c_str();
	folderpath = "C:\\Users\\TCCOM\\Desktop\\YEAR1 TERM2\\C++ Slide\\OPENCVTESTER\\OPENCVTESTER\\SavedImage\\Swapper" + filenum + ".png";
	imagecount++;
	ofstream out_file;
	out_file.open("C:\\Users\\TCCOM\\Desktop\\YEAR1 TERM2\\C++ Slide\\OPENCVTESTER\\OPENCVTESTER\\ProjectPic.txt");
	out_file << imagecount;
	out_file.close();
}

void SaveImage :: SaveImagetoFile(Mat frame)
{
	try {
		bool SaveToFile = imwrite(folderpath, frame, imagequality);
	}
	catch (exception& e)
	{
		std::cerr << "Fail to Save Image because " << e.what();
	}

}

void SaveImage::setFolderPath(string Path)
{
	folderpath = Path;
}

void SaveImage::printFilename()
{
	cout << folderpath;
}