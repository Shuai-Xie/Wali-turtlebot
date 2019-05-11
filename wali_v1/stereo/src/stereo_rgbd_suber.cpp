#include <ros/ros.h>
#include <std_msgs/Header.h>
#include <sensor_msgs/Image.h>

#include <stereo/RGBD_Image.h> // self-defined rgbd
#include <cv_bridge/cv_bridge.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

cv_bridge::CvImagePtr cv_ptr_rgb, cv_ptr_depth;
cv::Mat rgb, depth;

void chatterCallback(const stereo::RGBD_Image::ConstPtr &msg) {
    ROS_INFO("get RGBD %d", msg->header.seq);
    cv_ptr_rgb = cv_bridge::toCvCopy(msg->rgb, "bgr8");
    cv_ptr_depth = cv_bridge::toCvCopy(msg->depth, "mono16"); // 16bits depth
    rgb = cv_ptr_rgb->image;
    depth = cv_ptr_depth->image;
    cv::imshow("rgb", rgb);

    // cvt 1C depth to 3C for imshow
    cv::Mat depth_show(depth.rows, depth.cols, CV_8UC3);
    for (int h = 0; h < depth.rows; ++h) {
        for (int w = 0; w < depth.cols; ++w) { // ushort for 16 bit
            uchar val = (uchar) (depth.at<ushort>(h, w) / 256); // 16->8
            depth_show.at<cv::Vec3b>(h, w) = {val, val, val};
        }
    }
    cv::imshow("depth", depth_show); // default 3 channels
    cv::waitKey(30);
}


int main(int argc, char **argv) {
    ros::init(argc, argv, "stereo_rgbd_suber");
    ros::NodeHandle n;
    ros::Subscriber sub = n.subscribe("/stereo/rgbd/image", 1000, chatterCallback);
    ros::spin();
    return 0;
}
