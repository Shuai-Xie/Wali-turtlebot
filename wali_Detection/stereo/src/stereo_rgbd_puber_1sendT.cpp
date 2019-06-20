#include <thread>
#include <chrono>
#include <unistd.h>
#include <sys/time.h>
#include "../include/stereo/elas.h"
#include "../include/stereo/image.h"
#include "../include/stereo/GetRemapParam.h"
#include "../include/stereo/GlobelVariable.h"

#include <ros/ros.h> // make this cpp as ros node
#include <std_msgs/Header.h>
#include <sensor_msgs/Image.h>

#include <stereo/RGBD_Image.h> // self-defined rgbd
#include <cv_bridge/cv_bridge.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <csignal>


#define NUM_THREADS 4
#define num_counts  100

using namespace std;
using namespace cv;
using namespace std::chrono; //test time


static volatile int keepRunning = 1;

void sig_handler(int sig) {
    if (sig == SIGINT) {
        keepRunning = 0;
    }
}

//Get rectified parameters
Mat map11, map12, map21, map22;
Rect rectified_roi;
string img_head_a = "";
string img_head_b = "";
string file_type = ".jpg";
int32_t count_num = 0;
int32_t count_num_all = 0;
string rectified_path_left = "../img/img_rgb_left/";
string rectified_path_right = "../img/img_rgb_right/";
string disparity_path = "../img/img_dis/";
string disp_rgb_path = "../img/img_rgb/";

picMessage get_pic[NUM_THREADS];
static pthread_mutex_t picMutex[NUM_THREADS];

pthread_mutex_t countlock = PTHREAD_MUTEX_INITIALIZER;
high_resolution_clock::time_point t1, t2;


transMessage trans_pic[1];
static pthread_mutex_t trans_picMutex[1];


// compute disparities of pgm image input pair file_1, file_2
void process(const Mat &img_1, const Mat &img_2, Mat &Disp, const int num) {

    Mat d1(img_1.size(), CV_32F);
    Mat d2(img_2.size(), CV_32F);
    int32_t dims[] = {img_1.cols, img_1.rows, img_1.cols};

    Elas::parameters param;
    param.postprocess_only_left = true;
    Elas elas(param);
    elas.process(img_1.data, img_2.data, (float *) d1.data, (float *) d2.data, dims);

    d1.convertTo(d1, CV_16U);
    d1 = 951.009746 * 301.268297 / d1;
    Disp = d1;

//    Mat d1_temp;
//    d1.convertTo(d1_temp, CV_16U);
//
//    ushort *tem_16 = d1.ptr<ushort>(100);
//    cout << "16U:" << (int) tem_16[100] << endl;
//
//    Mat d1_temp_disp;
//    d1_temp_disp = 951.009746 * 301.268297 / d1_temp;
//
//    //ushort *temp_16 = d1_temp_disp.ptr<ushort>(100);
//    //cout << "16U:" << (int) temp_16[100] << endl;
//
//    Disp = d1_temp_disp;

    d1.release();
    d2.release();
}

void getImgFromMes(Mat &img_1, Mat &img_2, const int count) {
    int cur_get_point = count % NUM_THREADS;
    while (1) {
        if (get_pic[cur_get_point].has) {
            if (pthread_mutex_lock(&(picMutex[cur_get_point])) != 0) {
                fprintf(stdout, "lock error!\n");
            }

            img_1 = get_pic[cur_get_point].img[0].clone();
            img_2 = get_pic[cur_get_point].img[1].clone();
            get_pic[cur_get_point].has = false;
#ifdef logMess
            cout<<"get      img"<<count<<endl;
#endif

            pthread_mutex_unlock(&(picMutex[cur_get_point]));//����
            break;
        } else {
            if (showsleep) {
                cout << "sleep for not get" << count << endl;
            }
            usleep(200000);
        }
    }
}

void *thread_pic(void *thread_mess) {
    Mat Disp;
    Mat img_1, img_2;
    Mat img_1Gray, img_2Gray, DispRgb;
    char strnum[10];

    int transindex = 0;

    while (ros::ok) {
        pthread_mutex_lock(&countlock);
        const int32_t num = count_num;
        transindex = count_num_all;
        count_num++;
        count_num_all++;
        pthread_mutex_unlock(&countlock);

        getImgFromMes(img_1, img_2, num);
        sprintf(strnum, "%04d", num);

//        string path_left = "img/ori_left/";
//        string path_right = "img/ori_right/";
//        imwrite(path_left + img_head_a + strnum + file_type, img_1);
//        imwrite(path_right + img_head_b + strnum + file_type, img_2);

        if (count_num >= num_counts) {
            count_num = 0;
            t2 = high_resolution_clock::now();
            duration<double, std::ratio<1, 1>> duration_s(t2 - t1);

            printf("The time : %.4f\n", (double) duration_s.count());
            printf("The num of img per second: %.4f\n", num_counts / (double) duration_s.count());
            printf("The time of per 100 img: %.4f\n", (double) duration_s.count() / num_counts * 100);

            t1 = high_resolution_clock::now();
        }

        //Recified
        remap(img_1, img_1, map11, map12, INTER_LINEAR);
        remap(img_2, img_2, map21, map22, INTER_LINEAR);

        img_1 = Mat(img_1, rectified_roi);
        img_2 = Mat(img_2, rectified_roi);

        // imwrite("/home/nvidia/wali_ws/src/stereo/img/ori.png", img_1);

        // 1224x575 -> 240x320 0.2615, 0.418
        // 1224x575 -> 1022x480  0.8347
        // 1224x575 -> 511x240
        double r = 0.4173;
        resize(img_1, img_1, Size(), r, r, INTER_CUBIC); // 511x240
        resize(img_2, img_2, Size(), r, r, INTER_CUBIC);

        img_1 = img_1(Rect(55, 0, 400, 240)); // 400x240
        img_2 = img_2(Rect(55, 0, 400, 240));

        cvtColor(img_1, img_1Gray, CV_BGR2GRAY);
        cvtColor(img_2, img_2Gray, CV_BGR2GRAY);

        process(img_1Gray, img_2Gray, Disp, num);

        // cut middle 191, 640x480
        Disp = Disp(Rect(40, 0, 320, 240)); // 320x240
        img_1 = img_1(Rect(40, 0, 320, 240));

        // imwrite("/home/nvidia/wali_ws/src/stereo/img/ori_s.png", img_1);
//		cv::imshow("thread_pic", img_1);
//		if (cv::waitKey(100) == 'q') {
//			break;
//		}


        //put pic message into trans_pic
        if (transindex > trans_pic[0].index) {
            if (pthread_mutex_lock(&(trans_picMutex[0])) != 0) {
                fprintf(stdout, "lock error!\n");
            }
            printf("put                                    :%d\n",transindex);
            trans_pic[0].leftpic = img_1.clone();
            trans_pic[0].depthpic = Disp.clone();
            trans_pic[0].index = transindex;
            pthread_mutex_unlock(&(trans_picMutex[0]));
        } else {
            continue;
        }
    }
    pthread_exit(NULL);
}


void *thread_getPic(void *thread_mess) {

    string gst_left = "rtspsrc location=rtsp://admin:vipa404404@192.168.1.66:554 latency=0 ! rtph264depay ! h264parse ! omxh264dec ! videoconvert ! appsink";
    string gst_right = "rtspsrc location=rtsp://admin:vipa404404@192.168.1.67:554 latency=0 ! rtph264depay ! h264parse ! omxh264dec ! videoconvert ! appsink";

//    string gst_left = "rtsp://admin:vipa404404@192.168.1.66:554";
//    string gst_right = "rtsp://admin:vipa404404@192.168.1.67:554";

    cv::VideoCapture leftcapture(gst_left);
    if (!leftcapture.isOpened())
    {
        cout << "Failed to open camera." << endl;
    }
    cv::VideoCapture rightcapture(gst_right);
    if (!rightcapture.isOpened())
    {
        cout << "Failed to open camera." << endl;
    }

    int picIndexAll = 0;
    int picIndex = 0;
    cv::Mat leftframe, rightframe;
    bool readerror = false;
    while (ros::ok) {
        readerror = false;
        if (!leftcapture.read(leftframe)) {
            readerror = true;
            printf("error in read left camera\n");
        }
        if (!rightcapture.read(rightframe)) {
            readerror = true;
            printf("error in read right camera\n");
        }
        if (readerror) {
            continue;
        }

//        cv::imshow("leftimg",leftframe);
//        cv::waitKey(1);

        picIndex = picIndexAll % NUM_THREADS;

        if (get_pic[picIndex].has) {
#ifdef logMess
            printf("waiting for load img\n");
#endif
            usleep(20000);
            continue;
        }

        if (pthread_mutex_lock(&(picMutex[picIndex])) != 0) {
            fprintf(stdout, "lock error!\n");
        }

        get_pic[picIndex].img[0] = leftframe.clone();
        get_pic[picIndex].img[1] = rightframe.clone();
        get_pic[picIndex].has = true;
        pthread_mutex_unlock(&(picMutex[picIndex]));

#ifdef logMess
		cout << "put img" << picIndexAll << endl;
#endif
        picIndexAll++;
        usleep(20000);
    }
}

// global var
ros::Publisher chatter_pub;
stereo::RGBD_Image rgbd_msg;
std_msgs::Header header;

void *thread_transPic(void *thread_mess) {
    int transindex = 0;
    Mat leftpic, depthpic;

    signal(SIGINT, sig_handler);

    header.frame_id = "/stereo/rgbd/image";
//    cv::namedWindow("thread_transPic");

    while (keepRunning) {
        if (trans_pic[0].index < 0) {
            //printf("none pic, waiting for...\n");
            usleep(100000);
            continue;
        }

        if (transindex == trans_pic[0].index) {
            usleep(100000);
            continue;
        }
        if (pthread_mutex_lock(&(trans_picMutex[0])) != 0) {
            fprintf(stdout, "lock error!\n");
        }
        leftpic = trans_pic[0].leftpic;
        depthpic = trans_pic[0].depthpic;
        transindex = trans_pic[0].index;
        pthread_mutex_unlock(&(trans_picMutex[0]));

        cv::imshow("thread_transPic_left", leftpic);
        //cv::imshow("thread_transPic_depth", depthpic);
        if (cv::waitKey(1) == 'q') {
            ROS_INFO("close ros!");
            ros::shutdown();
        }

        header.stamp = ros::Time::now();
        header.seq = (unsigned int) transindex;
        rgbd_msg.header = header;

        cv_bridge::CvImage(std_msgs::Header(), "bgr8", leftpic).toImageMsg(rgbd_msg.rgb);
        cv_bridge::CvImage(std_msgs::Header(), "mono16", depthpic).toImageMsg(rgbd_msg.depth);

        ROS_INFO("send RGBD %d", transindex);

        chatter_pub.publish(rgbd_msg);
    }
}


int main(int argc, char *argv[]) {
    cout << "OpenCV version : " << CV_VERSION << endl;

    ros::init(argc, argv, "stereo_rgbd_puber"); // make this cpp as a node, then no multi_main error!
    ros::NodeHandle n;
    chatter_pub = n.advertise<stereo::RGBD_Image>("/stereo/rgbd/image", 10000); // Note: change msg type!
    Get_Remap_Parameters(map11, map12, map21, map22, rectified_roi);

    for (int i = 0; i < NUM_THREADS; i++) {
        get_pic[i].has = false;
    }

    trans_pic[0].index = -1;

    // Multi-thread compute disparity
    pthread_t tids[NUM_THREADS + 2];
    int32_t thread_mess[NUM_THREADS + 2];
    int ret;

    t1 = high_resolution_clock::now();

    for (int i = 0; i < NUM_THREADS + 2; ++i) {
        thread_mess[i] = i;
        if (i == 0) {
            ret = pthread_create(&tids[i], NULL, thread_getPic, (void *) &(thread_mess[i]));
            sleep(1);
        } else if (i == NUM_THREADS + 1) {
            ret = pthread_create(&tids[i], NULL, thread_transPic, (void *) &(thread_mess[i]));
        } else {
            ret = pthread_create(&tids[i], NULL, thread_pic, (void *) &(thread_mess[i]));
        }

        if (ret != 0) {
            cout << "pthread_create error:error_code=" << ret << endl;
        }
        usleep(500000);
    }
    pthread_join(tids[NUM_THREADS + 1], NULL);

//    for (int i = 1; i < NUM_THREADS + 1; ++i) {
//        pthread_join(tids[i], NULL);
//    }

    pthread_mutex_destroy(&countlock);
//    t2 = high_resolution_clock::now();
//    duration<double, std::ratio<1, 1>> duration_s(t2 - t1);
//    printf("threads num : %d\n", NUM_THREADS);
//    printf("predict img num : %d\n", count_num_all);
//    printf("The time : %.4f\n", (double) duration_s.count());
//    printf("The num of img per second: %.4f\n", count_num_all / (double) duration_s.count());
//    printf("The time of per 100 img: %.4f\n", (double) duration_s.count() / count_num_all * 100);

    return 0;
}