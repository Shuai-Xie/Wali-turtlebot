//
// Created by itx on 19-1-8.
//
#pragma once
#ifndef CAMERA_LIBELAS_GLOBELVARIABLE_H
#define CAMERA_LIBELAS_GLOBELVARIABLE_H

#endif //CAMERA_LIBELAS_GLOBELVARIABLE_H

#include "../../../../../../../usr/include/c++/5/iostream"
#include "../../../../../../../usr/include/stdio.h"
#include "../../../../../../../usr/include/c++/5/string"
#include "../../../../../../../usr/include/stdlib.h"
#include "../../../../../../../opt/ros/kinetic/include/opencv-3.3.1-dev/opencv2/opencv.hpp"

using namespace std;
using namespace cv;

#define showsleep 0

//typedef struct picMessage
//{
//    Mat img[2];
//    bool left_has;
//    bool right_has;
//};

typedef struct picMessage
{
    Mat img[2];
    bool has;
};

typedef struct  transMessage
{
    Mat leftpic;
    Mat depthpic;
    int index;
};
