#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <algorithm>

using namespace cv;
using namespace std;

void Get_Remap_Parameters(Mat& map11, Mat& map12, Mat& map21, Mat& map22, Rect& rectified_roi);

