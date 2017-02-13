//============================================================================
// Name        : test_opencv.cpp
// Author      : tony
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================
#include "main.hpp"


using namespace std;
using namespace cv;

int kalmen_camshift(VideoCapture capture, Mat background = Mat());


int main()
{
	VideoCapture cap(0);
	kalmen_camshift(cap);
};



