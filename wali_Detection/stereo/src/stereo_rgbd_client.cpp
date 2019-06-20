#include <ros/ros.h> // make this cpp as ros node
#include <std_msgs/Header.h>
#include <sensor_msgs/Image.h>

//#include <stereo/RGBD_Image.h> // topic
#include <kinect/RGBD_Image.h> // service
#include <cv_bridge/cv_bridge.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

cv::Mat Depth_16U_2_8UC1(cv::Mat depth) {
    cv::Mat depth_show(depth.rows, depth.cols, CV_8UC1);
    for (int h = 0; h < depth.rows; ++h) {
        for (int w = 0; w < depth.cols; ++w) { // ushort for 16 bit
            depth_show.at<uchar>(h, w) = (uchar) (depth.at<ushort>(h, w) / 256); // 16->8
        }
    }
    return depth_show;
}

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "stereo_rgbd_client"); // make this cpp as a node, then no multi_main error!
    ros::NodeHandle n;

    ros::ServiceClient client = n.serviceClient<kinect::RGBD_Image>("get_stereo_rgbd_image");
    kinect::RGBD_Image srv;
    srv.request.start = static_cast<unsigned char>(true);

    cv_bridge::CvImagePtr cv_ptr_rgb, cv_ptr_depth;
    cv::Mat rgb, depth, seg;

    int seq = 0;
    while (ros::ok()) {
        if (client.call(srv)) {
            seq = srv.response.rgb.header.seq;
            std::cout << seq << std::endl;

            if (seq > 0) {
                ROS_INFO("get RGBG %d", seq);
                // 1.get RGBD img
                cv_ptr_rgb = cv_bridge::toCvCopy(srv.response.rgb); // cvt ros::sensor_msgs/Image to cv::Mat
                cv_ptr_depth = cv_bridge::toCvCopy(srv.response.depth);
                rgb = cv_ptr_rgb->image;
                depth = cv_ptr_depth->image;

                // 2.show results
                cv::imshow("stereo_rgb", rgb);

                depth = Depth_16U_2_8UC1(depth);
                double minv = 0.0, maxv = 0.0;
                double *minp = &minv;
                double *maxp = &maxv;
                cv::Mat DispRgb;
                cv::minMaxIdx(depth, minp, maxp);
                cv::applyColorMap((depth / maxv * 255), DispRgb, cv::COLORMAP_JET);
                cv::imshow("stereo_depth", DispRgb);

                if (cv::waitKey(10) == 'q') {
                    break;
                }
            } else {
                ROS_INFO("waiting rgbd server...");
            }
        } else {
            ROS_ERROR("Failed to call service get_rgbd_image");
            return 1;
        }
    }
}