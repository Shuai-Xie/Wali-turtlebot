cmake_minimum_required(VERSION 2.8.3)
project(model)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

# catkin cmake
set(catkin_DIR /opt/ros/kinetic/share/catkin/cmake)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
        roscpp
        rospy
        std_msgs
        cv_bridge
        kinect
        )

find_package(OpenCV REQUIRED)

catkin_package(
        #  INCLUDE_DIRS include
        #  LIBRARIES model
        #  CATKIN_DEPENDS roscpp rospy std_msgs
        #  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
        include
        ${catkin_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        /usr/local/cuda-9.0/targets/aarch64-linux/include # <cuda_runtime_api.h>
        /usr/include/aarch64-linux-gnu # NvInfer.h, NvOnnxConfig.h, NvOnnxParser.h
)

add_executable(kinect_bisenet src/kinect_bisenet.cpp)

## Add cmake target dependencies of the executable
## same as for the library above
add_dependencies(kinect_bisenet
        ${${PROJECT_NAME}_EXPORTED_TARGETS}
        ${catkin_EXPORTED_TARGETS}
        )

## Specify libraries to link a library or executable target against
target_link_libraries(kinect_bisenet
        ${catkin_LIBRARIES}
        ${OpenCV_LIBRARIES}
        libcudart.so
        libnvinfer.so
        libnvparsers.so # onnx parse..
        )
