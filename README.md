
## Wali_TX2 - 2019.05.11

部署 Wali_ROS_system 到 Jetson TX2，基于 **ROS + tb2 + tensorRT + kinect/stereo** 实时 RGBD 场景分割。
- 实时获取 Kinect/RealSense RGBD，自定义 `RGBD_Image.srv` 实现
- 实时获取 双目相机 Stereo RGBD，自定义 `RGBD_Image.msg` 实现
- 使用 tensorRT 构建 BiSeNet C++ 模型推理 engine
- 构建 RGBD 图像发布节点 (kinect, stereo)
- 构建 BiSeNet C++ 模型推理节点
- Demo 演示：https://www.bilibili.com/video/av51854928/

![wali_TX2](assets/wali_TX2.png)

---

## Wali_depth - 2019.02.20

只利用相机 **Depth** 信息指导小车移动方向。

**1. 思路：**

```py
while True:
	Depth 中心模拟一个与小车同大小的 ROI;

	# 判断是否需要转弯
	if ROI_AVG_Depth > SAFE_MIN_DISTANCE:
		小车保持原来方向直走;

	else:
		缓慢减速, 小车停下;

		# begin search new direction
		Depth 横向模拟一组 ROI，计算 ROI_AVG_Depth_Max;

		if ROI_AVG_Depth_Max > SAFE_MIN_DISTANCE:
			选择 ROI_AVG_Depth_Max 对应的 ROI 中心作为前进方向;
			旋转小车直到 相机中心 与 ROI 中心重合;
	  		缓慢加速，小车重新启动;
	  	else:
	  		No way to Go!
	  		break
		# end search
```


**2. Road Test：**

① 未设置匀变速运动，速度突变卡顿
- indoor: https://www.bilibili.com/video/av44314807/

② 添加匀变速运动之后，卡顿问题解决
- in1：https://www.bilibili.com/video/av44315156/
- in2：https://www.bilibili.com/video/av44315266/
- out1：https://www.bilibili.com/video/av44315515/
- out2：https://www.bilibili.com/video/av44315582/

---

## Wali-turtlebot
Wali turtlebot is a self-driving turtlebot, using **Scene Segmentation** on **RGBD** data to make **Path Planning**.

![auto_drive](assets/auto_drive.png)

### 1. Hardware
- Turtlebot2
- Jetson TX2
- HIKVISION wireless camera (stereo)
- Microsoft Kinect v1 (not intense, not outdoor)
- Intel RealSense R200
- HiSilicon970 (arm)

### 2. Wali ROS System Architecture
- ROS Nodes communication (Service, Topic)
- Turtlebot motion control using rospy (forward, left, right, back, and **smoothly speed up**)

![wali_arc](assets/wali_arc.png)

### 3. BiSeNet-RGBD Architecture

![bisenet_rgbd](assets/bisenet_rgbd.png)

BiSeNet-RGBD is trained on [Princeton SUN-RGBD dataset](http://rgbd.cs.princeton.edu/). Now it can predict 37 class, we'll annotate some specific classes in our practical scenario using labelme in the future. 

![SUNRGB_37_label_map.png](assets/SUNRGB_37_label_map.png)

**Test scenes:**

- indoor

![indoor](assets/indoor.png)

- outdoor

![outdoor](assets/outdoor.png)

- stereo: https://www.bilibili.com/video/av44357263/