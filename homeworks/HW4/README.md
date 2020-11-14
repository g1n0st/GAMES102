# 作业4报告

2020/11/14 余畅 电子科技大学

### 问题

模仿 PowerPoint 写一个曲线设计与编辑工具 

输入有序点列（型值点），实时生成分段的三次样条曲线

### 实现

这次作业由于需要考虑实时的GUI系统，所以使用了Taichi平台实现。

考虑用户输入的 $n+1$ 个数据点，首先需要使用上一次作业的方法进行参数化，本次作业使用的参数化方法为 Centripetal parameterization，得到 $t_i = \sqrt{ \lvert \lvert p_{i+1}-p_{i} \lvert \lvert}$。然后将数据结点和指定的首位端点条件代入矩阵方程，本次作业使用的是自然边界条件，即样条首尾两端没有受到弯曲的力，也即 $S^{'''}=0$。

然后进行矩阵求解，求得二次微分值 $m_i$ （其中 $m_0=0,m_n=0$），由于该矩阵为稀疏矩阵且特别的，是三对角矩阵，使用 `Thomas Algorithm` 算法求解，参考了https://www.cnblogs.com/xpvincent/archive/2013/01/25/2877411.html

最后计算每段样条曲线的系数：

$xa_i=x_i$

$xb_i=\frac{x_{i+1}-x_i}{t_i}-\frac{t_i}{2}m_i-\frac{t_i}{6}(m_{i+1}-m_i)$

$xc_i=\frac{m_i}{2}$

$xd_i=\frac{m_{i+1}-m_i}{6t_i}$

$ya_i=y_i$

$yb_i=\frac{y_{i+1}-y_i}{t_i}-\frac{t_i}{2}m_i-\frac{t_i}{6}(m_{i+1}-m_i)$

$yc_i=\frac{m_i}{2}$

$yd_i=\frac{m_{i+1}-m_i}{6t_i}$

其中 $i=0,1,...,n-1$

在窗口图形显示阶段中，对三次样条进行 $t$ 参数均匀采用，在每个子区间 $t_i \le t \le t_{i+1}$ 中，方程为：

$x(t)=xa_i+xb_i(t-t_i)+xc_i(t-t_i)^2+xd_i(t-t_i)^3$

$y(t)=ya_i+yb_i(t-t_i)+yc_i(t-t_i)^2+yd_i(t-t_i)^3$

### 结果

见同目录下 `record.mp4`