cmake_minimum_required(VERSION 2.8.3)
project(stereo)

## Compile as C++11, supported in ROS Kinetic and newer
#set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/home/nvidia/users/why/tools/opencv/src/opencv-3.4.0/build/")

add_compile_options(-std=c++11)
SET(CMAKE_SYSTEM_PROCESSOR arm)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -std=c++11  -pthread -Wl,--no-as-needed -g -D_REENTRANT -w")
set(CMAKE_BUILD_TYPE Release)


find_package(catkin REQUIRED COMPONENTS
        roscpp
        rospy
        std_msgs
        sensor_msgs
        message_generation
        cv_bridge
        kinect  # use its srv
        )

set(CMAKE_PREFIX_PATH "/home/nvidia/users/why/tools/opencv/src/opencv-3.4.0/build/")


set(OpenCV_DIR "/home/nvidia/users/why/tools/opencv/src/opencv-3.4.0/build/")
find_package(OpenCV REQUIRED CONFIG)
find_package(PkgConfig REQUIRED)

# generate RGBD_Image.msg
add_message_files(
        FILES
        RGBD_Image.msg
)

## Generate added messages and services with any dependencies listed here
generate_messages(
        DEPENDENCIES
        std_msgs
        sensor_msgs # use Image
)

catkin_package(
        #  INCLUDE_DIRS include
        #  LIBRARIES stereo
        CATKIN_DEPENDS roscpp rospy std_msgs sensor_msgs message_runtime
        #  DEPENDS system_lib
)

###########
## Build ##
###########

set(Camera_libelas include/stereo)

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
        ${catkin_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        # stereo camera
        ${Camera_libelas_SRC_DIR}
)

# sources
FILE(GLOB Camera_libelas_SRC_FILES "include/stereo/*.cpp")

# Declare a C++ executable
# With catkin_make all packages are built within a single CMake context
# The recommended prefix ensures that target names across packages don't collide
#add_executable(stereo_rgbd_puber ${Camera_libelas_SRC_FILES} src/stereo_rgbd_puber.cpp)
add_executable(stereo_rgbd_puber ${Camera_libelas_SRC_FILES} src/stereo_rgbd_puber_1sendT.cpp)
add_executable(stereo_rgbd_suber src/stereo_rgbd_suber.cpp)
add_executable(stereo_rgbd_server ${Camera_libelas_SRC_FILES} src/stereo_rgbd_server.cpp)
add_executable(stereo_rgbd_client src/stereo_rgbd_client.cpp)

LINK_DIRECTORIES("/home/nvidia/users/why/tools/opencv/src/opencv-3.4.0/build/lib/")

# rgbd_server
add_dependencies(stereo_rgbd_server
        ${${PROJECT_NAME}_EXPORTED_TARGETS}
        ${catkin_EXPORTED_TARGETS}
        )
target_link_libraries(stereo_rgbd_server
        ${catkin_LIBRARIES}
        ${OpenCV_LIBRARIES}
        )

# rgbd_client
add_dependencies(stereo_rgbd_client
        ${${PROJECT_NAME}_EXPORTED_TARGETS}
        ${catkin_EXPORTED_TARGETS}
        )
target_link_libraries(stereo_rgbd_client
        ${catkin_LIBRARIES}
        ${OpenCV_LIBRARIES}
        )

# rgbd_puber
add_dependencies(stereo_rgbd_puber # need this!, or undefined reference to ros!
        ${${PROJECT_NAME}_EXPORTED_TARGETS}
        ${catkin_EXPORTED_TARGETS}
        )
target_link_libraries(stereo_rgbd_puber
        ${catkin_LIBRARIES}
        ${OpenCV_LIBRARIES}
        #        -lopencv_core
        #        -lopencv_highgui
        #        -lopencv_video
        #        -lopencv_imgproc
        #        -lopencv_videoio
        #        -lopencv_calib3d
        )

# rgbd_suber
add_dependencies(stereo_rgbd_suber
        ${${PROJECT_NAME}_EXPORTED_TARGETS}
        ${catkin_EXPORTED_TARGETS}
        )
target_link_libraries(stereo_rgbd_suber
        ${catkin_LIBRARIES}
        ${OpenCV_LIBRARIES}
        )