#include <iostream>
#include <fstream>
#include <assert.h>
#include <time.h> // cal program time

#include <ros/ros.h>
#include <stereo/RGBD_Image.h>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include "common.h"
#include "NvOnnxConfig.h"
#include "NvOnnxParser.h"

using namespace nvinfer1;
using namespace nvonnxparser;

static Logger gLogger;

// attrs of model
static const int scale1[] = {240, 320};
static const int scale2[] = {360, 480};
static const int scale3[] = {480, 640};

static const string onnx_name = "bise_240.onnx";

static const int INPUT_H = scale1[0];
static const int INPUT_W = scale1[1];
static const int INPUT_C = 4;
static const int INPUT_SIZE = INPUT_C * INPUT_H * INPUT_W;
static const int INPUT_STEP = INPUT_H * INPUT_W;

static const int OUTPUT_H = INPUT_H;
static const int OUTPUT_W = INPUT_W;
static const int OUTPUT_C = 37; // classes
static const int OUTPUT_SIZE = OUTPUT_C * OUTPUT_H * OUTPUT_W;
static const int OUTPUT_STEP = OUTPUT_H * OUTPUT_W; // for float argmax

// float array is too big, define in main will cause segmentation fault!
// need define global and use new to allocate heap_memory
float *input = new float[INPUT_SIZE];
float *output = new float[OUTPUT_SIZE];

static uchar label_colors[OUTPUT_C][3] = {
        {148, 65,  137}, // wall
        {255, 116, 69},  // floor
        {86,  156, 137}, // cabinet
        {202, 179, 158}, // bed
        {155, 99,  235}, // chair
        {161, 107, 108}, // sofa
        {133, 160, 103}, // table
        {76,  152, 126}, // door
        {84,  62,  35},  // window
        {44,  80,  130}, // bookshelf
        {31,  184, 157}, // picture
        {101, 144, 77},  // counter
        {23,  197, 62},  // blinds
        {141, 168, 145}, // desk
        {142, 151, 136}, // shelves
        {115, 201, 77},  // curtain
        {100, 216, 255}, // dresser
        {57,  156, 36},  // pillow
        {88,  108, 129}, // mirror
        {105, 129, 112}, // floor_mat
        {42,  137, 126}, // clothes
        {155, 108, 249}, // ceiling
        {166, 148, 143}, // books
        {81,  91,  87},  // fridge
        {100, 124, 51},  // tv
        {73,  131, 121}, // paper
        {157, 210, 220}, // towel
        {134, 181, 60},  // shower_curtain
        {221, 223, 147}, // box
        {123, 108, 131}, // whiteboard
        {161, 66,  179}, // person
        {163, 221, 160}, // night_stand
        {31,  146, 98},  // toilet
        {99,  121, 30},  // sink
        {49,  89,  240}, // lamp
        {116, 108, 9},   // bathtub
        {161, 176, 169}, // bag
};

const std::vector<std::string> directories{ // model path
        "/home/nvidia/wali_ws/src/model/ckpts/",
};

void onnxToTRTModel(const std::string &modelFile,      // name of the onnx model
                    unsigned int maxBatchSize,         // batch size - NB must be at least as large as the batch we want to run with
                    IHostMemory *&trtModelStream)      // output buffer for the TensorRT model
{
    // create the builder
    IBuilder *builder = createInferBuilder(gLogger);

    // create a 16-bit model if it's natively supported
    bool useFp16 = builder->platformHasFastFp16();
    useFp16 = false; // test FP32
    DataType modelDataType = useFp16 ? DataType::kHALF
                                     : DataType::kFLOAT;

    nvonnxparser::IOnnxConfig *config = nvonnxparser::createONNXConfig();
    config->setModelFileName(locateFile(modelFile, directories).c_str()); // str -> const* char
    nvonnxparser::IONNXParser *parser = nvonnxparser::createONNXParser(*config);

    // parse model file, use fp16 datatype!!
    if (!parser->parse(locateFile(modelFile, directories).c_str(), modelDataType)) {
        string msg("failed to parse onnx file");
        gLogger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
        exit(EXIT_FAILURE);
    } else {
        if (useFp16) {
            cout << "  ->parse onnx file using FP16 done!" << endl;
        } else {
            cout << "  ->parse onnx file using FP32 done!" << endl;
        }
    }

    if (!parser->convertToTRTNetwork()) {
        string msg("ERROR, failed to convert onnx network into TRT network");
        gLogger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
        exit(EXIT_FAILURE);
    } else {
        cout << "  ->convert to TRT done!" << endl;
    }

    nvinfer1::INetworkDefinition *network = parser->getTRTNetwork();

    // build engine
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(1 << 20);

    // set up the network for paired-fp16 format if available
    if (useFp16)
        builder->setFp16Mode(true);

    ICudaEngine *engine = builder->buildCudaEngine(*network);
    assert(engine);
    cout << "  ->build engine finished!" << endl;

    // we don't need the network any more, and we can destroy the parser
    network->destroy();
    parser->destroy();

    // serialize the engine, then close everything down
    trtModelStream = engine->serialize(); // store in IHostMemory *&trtModelStream
    engine->destroy();
    builder->destroy();
}


void doInference(IExecutionContext &context, // context
                 float *input,
                 float *output, // input output pointer address
                 int batchSize) // batchSize=1
{
    const ICudaEngine &engine = context.getEngine();

    // input and output buffer pointers that we pass to the engine IEngine::getNbBindings(),
    assert(engine.getNbBindings() == 2); // assert one input and one output
    void *buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // note that indices are guaranteed to be less than IEngine::getNbBindings() = 2
    int inputIndex, outputIndex;
    for (int b = 0; b < engine.getNbBindings(); ++b) { // 0,1
        if (engine.bindingIsInput(b)) {
            inputIndex = b;
        } else
            outputIndex = b;
    }

    // create GPU buffers and a stream
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * INPUT_SIZE * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * INPUT_SIZE * sizeof(float),
                          cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float),
                          cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // release the stream and the buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

// normalize rgb
static const float rgb_norm_means[] = {0.406, 0.456, 0.485}; // RGB
static const float rgb_norm_stds[] = {0.225, 0.224, 0.229};
// normalize depth
//static const float depth_norm_mean = 19050.0;
//static const float depth_norm_std = 9650.0;

static const float depth_norm_mean = 2280.0;
static const float depth_norm_std = 1200.0;

// max val index of an array
template<class ForwardIterator>
inline int argmax(ForwardIterator first, ForwardIterator last) {
    return static_cast<int>(std::distance(first, std::max_element(first, last)));
}

// preprocess src img, pass to float array
void preprocess(cv::Mat rgb, cv::Mat depth, float *data) {
    // resize
    cv::resize(rgb, rgb, cv::Size(INPUT_W, INPUT_H), cv::INTER_LINEAR);
    cv::resize(depth, depth, cv::Size(INPUT_W, INPUT_H), cv::INTER_NEAREST);

    // cvt to float
    rgb.convertTo(rgb, CV_32FC3);
    depth.convertTo(depth, CV_32F);
    rgb = rgb / 255.0; // scale to [0,1]
    depth = (depth - depth_norm_mean) / depth_norm_std; // norm depth

    // split rgb and norm each channel
    vector<cv::Mat> channels;
    cv::split(rgb, channels); // B,G,R order
    for (int i = 0; i < 3; ++i) {
        cv::Mat channel = (channels[2 - i] - rgb_norm_means[i]) / rgb_norm_stds[i]; // R begin
        float *c_data = (float *) channel.data;
        std::copy(c_data, c_data + INPUT_STEP, data + INPUT_STEP * i); // channel data
    }
    float *D_data = (float *) depth.data;
    std::copy(D_data, D_data + INPUT_STEP, data + INPUT_STEP * 3);
}

float prob[OUTPUT_C];


// use argmax to get class_vals matrix from output float array
cv::Mat getClassVals(const float *data) {
    cv::Mat class_val = cv::Mat::zeros(OUTPUT_H, OUTPUT_W, CV_32S); // use int as color_index of mapVals2RGB
    int index = 0;
    for (int h = 0; h < OUTPUT_H; ++h) {
        for (int w = 0; w < OUTPUT_W; ++w) {
            for (int c = 0; c < OUTPUT_C; ++c) { // 37
                prob[c] = data[index + c * OUTPUT_STEP]; // 不是连续的，不能 copy
            }
            class_val.at<int>(h, w) = argmax(prob, prob + OUTPUT_C);
            index++; // finish this point, goto next
        }
    }
    return class_val;
}

// map class vals to RGB, and scale to input size
cv::Mat mapVals2RGB(cv::Mat class_val) {
    cv::Mat color = cv::Mat::zeros(OUTPUT_H, OUTPUT_W, CV_8UC3);
    uchar *a;
    for (int h = 0; h < OUTPUT_H; ++h) {
        for (int w = 0; w < OUTPUT_W; ++w) {
            a = label_colors[class_val.at<int>(h, w)]; // class_val need be CV_32S
            color.at<cv::Vec3b>(h, w) = {a[2], a[1], a[0]};
        }
    }
    // 240,320 -> 480,640
    cv::resize(color, color, cv::Size(), 2, 2, cv::INTER_NEAREST);
    return color;
}

// merge getClassVal & mapVals2RGB
cv::Mat getColorSeg(const float *data) {
    cv::Mat color = cv::Mat::zeros(OUTPUT_H, OUTPUT_W, CV_8UC3);
    int index = 0;
    uchar *a;
    for (int h = 0; h < OUTPUT_H; ++h) {
        for (int w = 0; w < OUTPUT_W; ++w) {
            for (int c = 0; c < OUTPUT_C; ++c) { // 37
                prob[c] = data[index + c * OUTPUT_STEP]; // 不是连续的，不能 copy
            }
            a = label_colors[argmax(prob, prob + OUTPUT_C)]; // class_val map color
            color.at<cv::Vec3b>(h, w) = {a[2], a[1], a[0]};
            index++; // finish this point, goto next
        }
    }
    // 240,320 -> 480,640
    // cv::resize(color, color, cv::Size(), 2, 2, cv::INTER_NEAREST);
    return color;
}

cv::Mat Depth_16U_2_8UC3(cv::Mat depth) {
    cv::Mat depth_show(depth.rows, depth.cols, CV_8UC3);
    for (int h = 0; h < depth.rows; ++h) {
        for (int w = 0; w < depth.cols; ++w) { // ushort for 16 bit
            uchar val = (uchar) (depth.at<ushort>(h, w) / 256); // 16->8
            depth_show.at<cv::Vec3b>(h, w) = {val, val, val};
        }
    }
    return depth_show;
}

cv::Mat Depth_16U_2_8UC1(cv::Mat depth) {
    cv::Mat depth_show(depth.rows, depth.cols, CV_8UC1);
    for (int h = 0; h < depth.rows; ++h) {
        for (int w = 0; w < depth.cols; ++w) { // ushort for 16 bit
            depth_show.at<uchar>(h, w) = (uchar) (depth.at<ushort>(h, w) / 256); // 16->8
        }
    }
    return depth_show;
}


cv_bridge::CvImagePtr cv_ptr_rgb, cv_ptr_depth;
cv::Mat rgb, depth, seg;
IExecutionContext *context;


void chatterCallback(const stereo::RGBD_Image::ConstPtr &msg) {
    ROS_INFO("get RGBD %d", msg->header.seq);
    // std_msg -> mat
    cv_ptr_rgb = cv_bridge::toCvCopy(msg->rgb, "bgr8");
    cv_ptr_depth = cv_bridge::toCvCopy(msg->depth, "mono16"); // 16bits depth
    rgb = cv_ptr_rgb->image;
    depth = cv_ptr_depth->image;

    // preprocess rgbd, infer, color_seg
    preprocess(rgb, depth, input);
    doInference(*context, input, output, 1); // batch=1
    seg = getColorSeg(output);

    // show results
    cv::imshow("stereo_rgb", rgb);
    depth = Depth_16U_2_8UC1(depth);

    double minv = 0.0, maxv = 0.0;
    double *minp = &minv;
    double *maxp = &maxv;
    cv::Mat DispRgb;
    cv::minMaxIdx(depth, minp, maxp);
    cv::applyColorMap((depth / maxv * 255), DispRgb, cv::COLORMAP_JET);

    cv::imshow("stereo_depth", DispRgb);
    cv::imshow("stereo_seg", seg);

    if (cv::waitKey(1) == 'q') {
        ROS_INFO("close ros!");
        ros::shutdown();
    }
}


int main(int argc, char *argv[]) {
    clock_t t1, t2;
    // 1.create a TensorRT model from the onnx model and serialize it to a stream {trtModelStream}
    t1 = clock();
    std::cout << "loading model..." << std::endl;
    IHostMemory *trtModelStream{nullptr};
    onnxToTRTModel(onnx_name, 1, trtModelStream);
    assert(trtModelStream != nullptr);
    std::cout << "done!" << std::endl;
    t2 = clock();
    std::cout << "model load time: " << (double) (t2 - t1) / CLOCKS_PER_SEC << "s" << std::endl;

    // 2.deserialize the engine {trtModelStream} and get engine_execution_context
    IRuntime *runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine *engine = runtime->deserializeCudaEngine(trtModelStream->data(), trtModelStream->size(), nullptr);
    assert(engine != nullptr);
    trtModelStream->destroy();
    context = engine->createExecutionContext();
    assert(context != nullptr);

    // 3.init ros node
    ros::init(argc, argv, "stereo_bisenet");
    ros::NodeHandle n;
    ros::Subscriber sub = n.subscribe("/stereo/rgbd/image", 10000, chatterCallback);
    ros::spin();

    // 4.destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
    return 0;
}
