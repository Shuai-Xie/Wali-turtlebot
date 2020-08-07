
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

### BiSeNet-Wali

Compress BiSeNet with Structure Knowledge Distillation for Real-time image segmentation on wali-TX2.

- Code: https://github.com/Shuai-Xie/BiSeNet-wali

### Test scenes

- indoor

![indoor](assets/indoor.png)

- outdoor

![outdoor](assets/outdoor.png)

- stereo: https://www.bilibili.com/video/av44357263/



### Road Test

未设置匀变速运动，速度突变卡顿

- indoor: https://www.bilibili.com/video/av44314807/

添加匀变速运动之后，卡顿问题解决
- in1：https://www.bilibili.com/video/av44315156/
- in2：https://www.bilibili.com/video/av44315266/
- out1：https://www.bilibili.com/video/av44315515/
- out2：https://www.bilibili.com/video/av44315582/