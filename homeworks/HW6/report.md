# 作业6报告

2020/12/5 余畅 电子科技大学

### 问题

实现极小曲面的局部法

- 寻找非封闭三角网格曲面的边界

- 每个顶点更新坐标

- 迭代给定次数或迭代至收敛

### 实现

这次作业主要的难点在于安装和配置 Utopia 框架以及熟悉各种功能。歪一下楼，Utopia线性代数库的设计很有启发意义，有空准备重写一下自己的线性代数库。

使用的主要公式即上课ppt中的公式：

$P_{new}=P_{old}+\lambda H(P_{old})\mathbf{n}(P_{old})$

$H\mathbf{n}=\frac{1}{4A}(\text{cot}\alpha_j+\text{cot}\beta_j)(\mathbf{P}-\mathbf{Q}_j)$

但是这样写得到的结果并不正确，多次迭代以后图形无法收敛，公式(2)中 $(\mathbf{P}-\mathbf{Q}_j)$ 得到的结果和 $\mathbf{P}$ 点法线同向，那么 $H\mathbf{n}$ 的不断迭代则会使得 $\mathbf{P}$ 点越来越突出，即该点的曲率越来越大。个人认为公式(1)应写作 $P_{new}=P_{old}-\lambda H(P_{old})\mathbf{n}(P_{old})$ ，实际代码也证实了该结果。

### 结果

![Figure_1](C:\Users\yc\Desktop\hw6\Figure\Figure_1.png)

原始的人脸图像，颜色为法线。

![Figure_0](C:\Users\yc\Desktop\hw6\Figure\Figure_0.png)

$\lambda=0.1$，迭代100次。（着色为顶点迭代移动方向）

![Figure_2](C:\Users\yc\Desktop\hw6\Figure\Figure_2.png)

$\lambda=0.1$，迭代500次。（着色为顶点迭代移动方向）

![Figure_3](C:\Users\yc\Desktop\hw6\Figure\Figure_3.png)

$\lambda=0.1$，迭代1000次。（着色为顶点迭代移动方向）

![Figure_4](C:\Users\yc\Desktop\hw6\Figure\Figure_4.png)

迭代至收敛的结果（着色为正常表面法线）

![Figure_5](C:\Users\yc\Desktop\hw6\Figure\Figure_5.png)

bunny head模型（着色为正常法线）

![Figure_6](C:\Users\yc\Desktop\hw6\Figure\Figure_6.png)

$\lambda=0.1$，迭代10次。（着色为顶点迭代移动方向）

![Figure_7](C:\Users\yc\Desktop\hw6\Figure\Figure_7.png)

$\lambda=0.1$，迭代200次。（着色为顶点迭代移动方向）

![Figure_8](C:\Users\yc\Desktop\hw6\Figure\Figure_8.png)

$\lambda=1.0$，迭代200次。（着色为顶点迭代移动方向）

![Figure_9](C:\Users\yc\Desktop\hw6\Figure\Figure_9.png)

$\lambda=0.5$，迭代200次。（着色为顶点迭代移动方向）

![Figure_10](C:\Users\yc\Desktop\hw6\Figure\Figure_10.png)

$\lambda=0.1$，迭代1000次。（着色为顶点迭代移动方向）

![Figure_11](C:\Users\yc\Desktop\hw6\Figure\Figure_11.png)

$\lambda=1.0$，迭代1000次。（着色为顶点迭代移动方向）

![Figure_12](C:\Users\yc\Desktop\hw6\Figure\Figure_12.png)

迭代至几乎收敛的结果（着色为正常表面法线）

### 结论

$\lambda$ 的大小影响着收敛的效率，但是不能选取过大。

### 潜在BUG汇报

在使用 Utopia 框架的过程中，中间迭代消耗单帧的时间过长，fps降至1~3，但是极小曲面求解结束以后，fps仍没有恢复至原来的水平，整个框架仍处于卡顿的状态中，只能重启恢复。



