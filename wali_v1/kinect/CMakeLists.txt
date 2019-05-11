cmake_minimum_required(VERSION 2.8.3)
project(kinect)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
        roscpp
        rospy
        std_msgs
        sensor_msgs
        message_generation # for cmd add_service_files()
        )

## Generate services in the 'srv' folder
add_service_files(
        FILES
        RGBD_Image.srv
)

## Generate added messages and services with any dependencies listed here
generate_messages(
        DEPENDENCIES
        std_msgs
        sensor_msgs # use Image
)

## !! DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
        #  INCLUDE_DIRS include
        #  LIBRARIES kinect
        CATKIN_DEPENDS roscpp rospy std_msgs sensor_msgs message_runtime # for msg generation
        #  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
        ${catkin_INCLUDE_DIRS}
)