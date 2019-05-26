#include "ColorMapping.hpp"

ColorMapping::ColorMapping(ListColor cmap)
{
	colorlist = cmap;
}

void ColorMapping :: setColorTone(ListColor color)
{
	switch (color)
	{
	case BONE:
		colorframe = 1;
		break;

	case WINTER:
		colorframe = 2;
		break;

	case OCEAN:
		colorframe = 3;
		break;

	case SUMMER:
		colorframe = 4;
		break;

	case COOL:
		colorframe = 6;
		break;

	case HSV :
		colorframe = 10;
		break;

	case HOT :
		colorframe = 11;
		break;
	}
}

void ColorMapping::ConvertColor(Mat inf, Mat outF)
{
	try {
		inputFrame = inf;
		applyColorMap(inputFrame, outF, colorframe);
	}

	catch (runtime_error &error)
	{
		std::cerr << "Error : " << error.what();
	}

	catch (out_of_range &another_error)
	{
		std::cerr << "Out Of Range Error : " << another_error.what();
	}
}