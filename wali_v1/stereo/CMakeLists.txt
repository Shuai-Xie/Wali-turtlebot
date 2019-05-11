cmake_minimum_required(VERSION 2.8.3)
project(stereo)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
        roscpp
        rospy
        std_msgs
        sensor_msgs
        message_generation
        cv_bridge
        )

find_package(OpenCV REQUIRED)


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

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
        ${catkin_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
)

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
add_executable(stereo_rgbd_puber src/stereo_rgbd_puber.cpp)
add_executable(stereo_rgbd_suber src/stereo_rgbd_suber.cpp)

## Add cmake target dependencies of the executable
## same as for the library above

# rgbd_puber
add_dependencies(stereo_rgbd_puber
        ${${PROJECT_NAME}_EXPORTED_TARGETS}
        ${catkin_EXPORTED_TARGETS}
        )
target_link_libraries(stereo_rgbd_puber
        ${catkin_LIBRARIES}
        ${OpenCV_LIBRARIES}
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