#include <iostream>
#include <fstream>
#include <assert.h>

#include <ros/ros.h>
#include <kinect/RGBD_Image.h>
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

static Logger gLogger;

// attrs of model
static const int INPUT_H = 256;
static const int INPUT_W = 320;
static const int INPUT_C = 3;
static const int INPUT_SIZE = INPUT_C * INPUT_H * INPUT_W;

static const int OUTPUT_H = 32;
static const int OUTPUT_W = 40;
static const int OUTPUT_C = 14; // classes
static const int OUTPUT_SIZE = OUTPUT_C * OUTPUT_H * OUTPUT_W;
static const int OUTPUT_STEP = OUTPUT_H * OUTPUT_W; // for float argmax

const std::vector<std::string> directories{ // model path
        "/home/nvidia/wali_ws/src/model/ckpts/",
};

void onnxToTRTModel(const std::string &modelFile,      // name of the onnx model
                    unsigned int maxBatchSize,         // batch size - NB must be at least as large as the batch we want to run with
                    IHostMemory *&trtModelStream)      // output buffer for the TensorRT model
{
    // create the builder
    IBuilder *builder = createInferBuilder(gLogger);

    nvonnxparser::IOnnxConfig *config = nvonnxparser::createONNXConfig();
    config->setModelFileName(locateFile(modelFile, directories).c_str()); // str -> const* char

    nvonnxparser::IONNXParser *parser = nvonnxparser::createONNXParser(*config);

    if (!parser->parse(locateFile(modelFile, directories).c_str(), DataType::kFLOAT)) {
        string msg("failed to parse onnx file");
        gLogger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
        exit(EXIT_FAILURE);
    } else {
        cout << "  ->parse onnx file done!" << endl;
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
    builder->setMaxWorkspaceSize(1 << 30); // todo 1GB

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
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * INPUT_SIZE * sizeof(float))); // todo inputsize!!
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * INPUT_SIZE * sizeof(float), // todo
                          cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), // todo
                          cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // release the stream and the buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

// normal img means, stds
static const float norm_means[] = {0.406, 0.456, 0.485}; // BGR
static const float norm_stds[] = {0.225, 0.224, 0.229}; // BGR

// max val index of an array
template<class ForwardIterator>
inline int argmax(ForwardIterator first, ForwardIterator last) {
    return static_cast<int>(std::distance(first, std::max_element(first, last)));
}

// preprocess src img, pass to float array
void preprocess(cv::Mat src, float *data) {
    // 1.resize, cvt RGB
    cv::resize(src, src, cv::Size(INPUT_W, INPUT_H), cv::INTER_LINEAR); // equals BILINEAR
    // cv::cvtColor(src, src, cv::COLOR_BGR2RGB);

    // 2.uchar->CV_32F, scale to [0,1]
    src.convertTo(src, CV_32FC3);
    src = src / 255.0; // src-1 equals PIL.Image

    // 3.split R,G,B and normalize each channel using norm_means,norm_stds
    vector<cv::Mat> channels;
    cv::split(src, channels);
    // normalize, same to torchvision.transforms.Normalize
    for (int i = 0; i < 3; ++i) {
        channels[i] = (channels[i] - norm_means[i]) / norm_stds[i];
    }

    // 4.pass to data, ravel()
    int index = 0;
    for (int c = 2; c >= 0; --c) { // BGR -> RGB
        for (int h = 0; h < INPUT_H; ++h) {
            for (int w = 0; w < INPUT_W; ++w) {
                data[index] = channels[c].at<float>(h, w);
                index++;
            }
        }
    }
}


float prob[OUTPUT_C];

// use argmax to get class_vals matrix from output float array
cv::Mat getClassVals(const float *data) {
    cv::Mat out = cv::Mat::zeros(OUTPUT_H, OUTPUT_W, CV_32S);
    int index = 0;
    for (int h = 0; h < OUTPUT_H; ++h) {
        for (int w = 0; w < OUTPUT_W; ++w) {
            for (int c = 0; c < OUTPUT_C; ++c) { // 14
                prob[c] = data[index + c * OUTPUT_STEP];
            }
            out.at<int>(h, w) = argmax(prob, prob + OUTPUT_C);
            index++; // finish this point, goto next
        }
    }
    return out;
}

void printClassVals(cv::Mat val_mat) {
    for (int h = 0; h < OUTPUT_H; ++h) {
        for (int w = 0; w < OUTPUT_W; ++w) {
            cout << val_mat.at<int>(h, w);
        }
        cout << endl;
    }
}

static int label_colors[OUTPUT_C][3] = {
        {148, 65,  137}, // wall
        {255, 116, 69},  // floor
        {86,  156, 137}, // cabinet
        {49,  89,  240}, // chair
        {161, 107, 108}, // sofa
        {133, 160, 103}, // table
        {76,  152, 126}, // door
        {84,  62,  35},  // window
        {44,  80,  130}, // bookshelf
        {23,  197, 62},  // blinds
        {155, 108, 249}, // ceiling
        {100, 124, 51},  // tv
        {221, 223, 147}, // box
        {161, 66,  179}  // person
};

// map class vals to RGB, and scale to input size
cv::Mat mapVals2RGB(cv::Mat val_mat) {
    cv::Mat color = cv::Mat::zeros(OUTPUT_H, OUTPUT_W, CV_8UC3);
    int *a;
    for (int h = 0; h < OUTPUT_H; ++h) {
        for (int w = 0; w < OUTPUT_W; ++w) {
            a = label_colors[val_mat.at<int>(h, w)];
            color.at<cv::Vec3b>(h, w) = {(uchar) a[2], (uchar) a[1], (uchar) a[0]};
        }
    }
    cv::resize(color, color, cv::Size(), 8, 8, cv::INTER_NEAREST); // scale to input size
    return color;
}

int main(int argc, char *argv[]) {
    // 1.create a TensorRT model from the onnx model and serialize it to a stream {trtModelStream}
    std::cout << "load model.." << std::endl;
    IHostMemory *trtModelStream{nullptr};
    onnxToTRTModel("bisenet.onnx", 1, trtModelStream);
    assert(trtModelStream != nullptr);
    std::cout << "load done!" << std::endl;

    // 2.deserialize the engine {trtModelStream} and get engine_execution_context
    IRuntime *runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine *engine = runtime->deserializeCudaEngine(trtModelStream->data(), trtModelStream->size(), nullptr);
    assert(engine != nullptr);
    trtModelStream->destroy();
    IExecutionContext *context = engine->createExecutionContext();
    assert(context != nullptr);

    // 3.define input & output
    float input[INPUT_C * INPUT_H * INPUT_W]; // C,H,W
    float output[OUTPUT_C * OUTPUT_H * OUTPUT_W];

    // 4.init ros node
    ros::init(argc, argv, "kinect_bisenet");
    ros::NodeHandle n;

    ros::ServiceClient client = n.serviceClient<kinect::RGBD_Image>("get_rgbd_image");
    kinect::RGBD_Image srv;
    srv.request.start = static_cast<unsigned char>(true);

    cv_bridge::CvImagePtr cv_ptr_rgb;
    // cv_bridge::CvImagePtr cv_ptr_depth;

    cv::Mat val_out, color_out;
    int seq = 0;
    while (ros::ok()) {
        if (client.call(srv)) {
            seq = srv.response.rgb.header.seq;
            if (seq > 0) {
                ROS_INFO("get RGBG seg: %d", seq);
                // 1.get RGBD img
                cv_ptr_rgb = cv_bridge::toCvCopy(srv.response.rgb); // cvt ros::sensor_msgs/Image to cv::Mat
                // cv_ptr_depth = cv_bridge::toCvCopy(srv.response.depth);

                // 2.trt model infer
                // preproces
                preprocess(cv_ptr_rgb->image, input);
                // infer
                doInference(*context, input, output, 1); // batch=1
                // postpress
                val_out = getClassVals(output);
                color_out = mapVals2RGB(val_out);
                cv::resize(color_out, color_out, cv::Size(INPUT_W, INPUT_H), cv::INTER_NEAREST);

                // 3.show
                cv::imshow("rgb", cv_ptr_rgb->image);
                cv::imshow("seg", color_out);
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
    // destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
    return 0;
}
