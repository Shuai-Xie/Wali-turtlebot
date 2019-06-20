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
        stereo
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

# bise_RGBD
add_executable(kinect_bisenet src/kinect_bisenet.cpp) # kinect
add_executable(stereo_bisenet src/stereo_bisenet.cpp) # stereo

# stereo
add_dependencies(stereo_bisenet
        ${${PROJECT_NAME}_EXPORTED_TARGETS}
        ${catkin_EXPORTED_TARGETS}
        )
target_link_libraries(stereo_bisenet
        ${catkin_LIBRARIES}
        ${OpenCV_LIBRARIES}
        libcudart.so
        libnvinfer.so
        libnvparsers.so # onnx parse..
        )

# kinect
add_dependencies(kinect_bisenet
        ${${PROJECT_NAME}_EXPORTED_TARGETS}
        ${catkin_EXPORTED_TARGETS}
        )
target_link_libraries(kinect_bisenet
        ${catkin_LIBRARIES}
        ${OpenCV_LIBRARIES}
        libcudart.so
        libnvinfer.so
        libnvparsers.so # onnx parse..
        )

# bise_cx3
add_executable(kinect_cx3 src/kinect_cx3.cpp) # kinect
add_executable(stereo_cx3 src/stereo_cx3.cpp) # stereo

# stereo
add_dependencies(kinect_cx3
        ${${PROJECT_NAME}_EXPORTED_TARGETS}
        ${catkin_EXPORTED_TARGETS}
        )
target_link_libraries(kinect_cx3
        ${catkin_LIBRARIES}
        ${OpenCV_LIBRARIES}
        libcudart.so
        libnvinfer.so
        libnvparsers.so # onnx parse..
        )

# kinect
add_dependencies(stereo_cx3
        ${${PROJECT_NAME}_EXPORTED_TARGETS}
        ${catkin_EXPORTED_TARGETS}
        )
target_link_libraries(stereo_cx3
        ${catkin_LIBRARIES}
        ${OpenCV_LIBRARIES}
        libcudart.so
        libnvinfer.so
        libnvparsers.so # onnx parse..
        )