
# Wali-turtlebot
Wali turtlebot is a self-driving turtlebot, using **Scene Segmentation** on **RGBD** data to make **Path Planning**.

![auto_drive](assets/auto_drive.png)

## 1. Hardware
- Turtlebot2
- Jetson TX2
- HIKVISION wireless camera (stereo)
- Microsoft Kinect v1 (not intense, not outdoor)
- Intel RealSense R200
- HiSilicon970 (arm)

## 2. Wali ROS System Architecture
- ROS Nodes communication (Service, Topic)
- Turtlebot motion control using rospy (forward, left, right, back, and **smoothly speed up**)

![wali_arc](assets/wali_arc.png)

## 3. BiSeNet-RGBD Architecture

![bisenet_rgbd](assets/bisenet_rgbd.png)

BiSeNet-RGBD is trained on [Princeton SUN-RGBD dataset](http://rgbd.cs.princeton.edu/). Now it can predict 37 class, we'll annotate some specific classes in our practical scenario using labelme in the future. 

![SUNRGB_37_label_map.png](assets/SUNRGB_37_label_map.png)

**Test scenes:**

- indoor

![indoor](assets/indoor.png)

- outdoor

![outdoor](assets/outdoor.png)

- stereo: https://www.bilibili.com/video/av44357263/