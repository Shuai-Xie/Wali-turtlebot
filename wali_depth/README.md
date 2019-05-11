## Wali_depth

只利用相机 **Depth** 信息指导小车移动方向。

### 思路

```py
while True:
	模拟一个与小车同大小的 ROI;

	if ROI_AVG_Depth > SAFE_MIN_DISTANCE:
		小车保持原来方向直走;

	else
		缓慢减速, 小车停下;

		# begin search new direction
		使用 Depth 横向模拟一组 ROI;
		计算这组 ROI 中 AVG_Depth_Max;
		if AVG_Depth_Max > SAFE_MIN_DISTANCE;
			选择 AVG_Depth_Max 对应的 ROI 中心作为前进方向;
			旋转小车直到 相机中心 与 ROI 中心重合;
			缓慢加速，小车重新启动;
		else
			No way to Go!
			break;
		# end search
```

备注：
- **横向滑动是因为小车移动是二维的，只要左右转即可**
- 两中心重合旋转角计算：**根据相机内参和原相机中心深度值计算旋转角**