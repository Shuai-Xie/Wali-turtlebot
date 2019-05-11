#include <ros/ros.h>
#include <std_msgs/Header.h>
#include <sensor_msgs/Image.h>

#include <stereo/RGBD_Image.h> // self-defined rgbd
#include <cv_bridge/cv_bridge.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


int main(int argc, char **argv) {
    ros::init(argc, argv, "stereo_rgbd_puber");
    ros::NodeHandle n;
    ros::Publisher chatter_pub = n.advertise<stereo::RGBD_Image>("/stereo/rgbd/image", 1000); // Note: change msg type!
    ros::Rate loop_rate(10); // 10fps

    stereo::RGBD_Image rgbd_msg;

    // msg source
    cv::Mat rgb = cv::imread("/home/nvidia/wali_ws/src/stereo/img/rgb.jpg", cv::IMREAD_COLOR);
    cv::Mat depth = cv::imread("/home/nvidia/wali_ws/src/stereo/img/depth.png", cv::IMREAD_ANYDEPTH);

    // header
    std_msgs::Header header; // not necessary
    header.frame_id = "/stereo/rgbd/image"; // detailed header info
    header.stamp = ros::Time::now(); // time stamp

    unsigned int count = 0;
    while (ros::ok()) {
        header.seq = count;
        rgbd_msg.header = header;
        cv_bridge::CvImage(std_msgs::Header(), "bgr8", rgb).toImageMsg(rgbd_msg.rgb);
        cv_bridge::CvImage(std_msgs::Header(), "mono16", depth).toImageMsg(rgbd_msg.depth);
        // though default sensor_msgs/Image use uint8[] data, we use "mono16" can get 16bit data
        ROS_INFO("send RGBD %d", count);

        chatter_pub.publish(rgbd_msg);

        ros::spinOnce();
        loop_rate.sleep();
        ++count;
    }

    return 0;
}
