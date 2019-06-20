#include "GetRemapParam.h"

using namespace cv;
using namespace std;

void Get_Remap_Parameters(Mat& map11, Mat& map12, Mat& map21, Mat& map22, Rect& rectified_roi) {

	Size img_size = {1280,720};

	string param_path = "/home/nvidia/wali_ws/src/stereo/param/";
    cv::FileStorage fs;
    fs.open(param_path + "param1022_2_1280_30.yml", FileStorage::READ);

    bool flag = fs.isOpened();
    if(!fs.isOpened()){
        cout<<"failed to open file test.yml "<<endl<<endl;
        return;
    }

	Mat M1, D1, M2, D2, R, T;
	fs["M1"] >> M1;
	fs["D1"] >> D1;
	fs["M2"] >> M2;
	fs["D2"] >> D2;
	fs["R"]  >> R;
	fs["T"]  >> T;

	Mat R1, P1, R2, P2;
	Rect roi1, roi2;
	Mat Q;
	stereoRectify(M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, -1, img_size, &roi1, &roi2);
	
	//���Remap�Ĳ���
	initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
	initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);

	Point2i pa(std::max(roi1.tl().x, roi2.tl().x), std::max(roi1.tl().y, roi2.tl().y));
	Point2i pb(std::min(roi1.br().x, roi2.br().x), std::min(roi1.br().y, roi2.br().y));
	Rect rectified_roi_temp(pa, pb);

	rectified_roi = rectified_roi_temp;

	M1.release();
	D1.release();
	M2.release();
	D2.release();
	R.release();
	T.release();
	R1.release();
	P1.release();
	R2.release();
	P2.release();
	Q.release();

	return;
}
