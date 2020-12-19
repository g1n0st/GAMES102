
# 作业8报告

2020/12/19 余畅 电子科技大学

### 问题

+ 实现平面点集CVT的Lloyd算法

### 实现

+ 在给定的正方形区域内随机生成若干采样点
+ 生成这些点的Voronoi剖分
+ 计算每个剖分的重心，将采样点的位置更新到该重心 
+ 迭代步骤2和3

这次作业使用Python完成，求Voronoi剖分使用的是`scipy`库中的`scipy.spatial.Voronoi`来完成，并使用`scipy.spatial.voronoi_plot_2d`和`matplotlib.pyplot`实现了剖分的可视化。结果如下：

在平面上使用`numpy.random.random` 随机采样得到的点集和剖分：

![Figure1](Figure/Figure_1.png)



迭代1次后，结果已经有了极大的改善：

![Figure2](Figure/Figure_2.png)



迭代2次：

![Figure3](Figure/Figure_3.png)



迭代3次：

![Figure4](Figure/Figure_4.png)



另一组100个随机采样点的点集如下：

![Figure10](Figure/Figure_10.png)



迭代10次后：

![Figure11](Figure/Figure_11.png)



迭代20次后：

![Figure12](Figure/Figure_12.png)



迭代30次后：

![Figure13](Figure/Figure_13.png)

更多结果见 `Figure` 和 `Video` 文件夹。

by g1n0st

2020/12/19