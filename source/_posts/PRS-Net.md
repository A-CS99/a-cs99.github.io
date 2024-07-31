---
title: PRS-Net笔记
date: 2024-07-06 14:14:01
tags: 
    - PRS-Net
    - CNN
    - Symmetry Detection
    - Thesis Note
categories: 
    - 论文复现
---

> 论文原文：[PRS-Net: Planar Reflective Symmetry Detection Network](https://arxiv.org/abs/1910.06511v6)
# 1 导论

**平面反射对称（Planar Reflective Symmetry，PRS）** 是最常见、最重要的对称类型

## 1.1 过去的方法
- **主成分分析法（Principal Component Analysis，PCA）**
  - 适用于简单情况
  - 无法处理不与任何主轴正交的对称平面
  - 对微小的几何变化也十分敏感
- **空间采样方法（Spatial Sampling）**
  - 比PCA更鲁棒（robust）
  - 必须对潜在目标进行采样
  - 结果依赖采样的随机性
- 各种**需要监督的神经网络方法**
  - 当前没有无监督方法能够检测对称平面

## 1.2 本文的方法

### 1.2.1 基本模型
**无监督的深度卷积神经网络（Unsupervised Deep CNN，Convolutional Neural Network）**

- 确定全局模型特征，自动检测全局平面反射对称

- 输入：体素化的形状模型
- 输出：反射面（reflection planes）和旋转轴（rotation axes）

### 1.2.2 损失函数
- 新型**对称距离损失（Symmetry Distance Loss）**
  - 测量给定潜在对称平面的几何与对称的偏离
- **规范化损失（Regularization Loss）**
  - 避免生成重复的对称平面

## 1.3 本文方法的优势
- 检测结果更加可信、更加准确
- 比传统方法快数百倍，可达到实时性能
- 对噪声和错误输入更加鲁棒

# 2 相关工作
> 本文主要关注对PRSNet的理解，此处略过对称检测领域相关工作的介绍

# 3 网络架构
{% asset_img Network.png PRS-Net %}
## 3.1 CNN网络整体结构
  - 5个3D卷积层
    - 卷积核大小（kernel size）为3
    - 填充（padding）为1
    - 步幅（stride）为1
    - 每层后使用最大值池化（Max Pooling），卷积核大小为2
    - 激活函数使用Leaky ReLU
  - 全连接层
## 3.2 输入层
  $32\times32\times32$的体素（由对称形状体素化得到）
## 3.3 输出层
  无监督学习，输出根据损失函数自动优化
  - 三个对称平面的参数
    - 隐式表示（$\bold{P_i}=(\vec{n}_i,d_i),\quad\vec{n}_i^{(1)}x+\vec{n}_i^{(2)}y+\vec{n}_i^{(3)}z+d_i=0$）
    - 法向量初始化为$\vec{n}_1=(1,0,0),\vec{n}_2=(0,1,0),\vec{n}_3=(0,0,1)$
  - 三个对称轴的参数
    - 训练过程中用四元数（quaternion）表示，训练完成转换为轴角表示
    - 四元数表示$\bold{R_i}=cos(\theta/2)+sin(\theta/2)(\vec{v}_i^{(1)}\bold i+\vec{v}_i^{(2)}\bold j+\vec{v}_i^{(3)}\bold k)$
    - 旋转轴初始化为$\vec{v}_1=(1,0,0),\vec{v}_2=(0,1,0),\vec{v}_3=(0,0,1)$
    - 旋转角度初始化为$\theta=\pi$

# 4 损失函数
## 4.1 对称距离损失（Symmetry Distance Loss）
惩罚对称距离较远（即对称后偏离较大）的对称面/旋转轴
  - 第一步：从输入形状$O$上选取$N$个样本点构成点集$Q$，基于这个点集进行后续计算
  - 第二步：计算点集$Q$上任意一点$\bold{q_k}$对称/旋转后得到的点$\bold{q'_k}$
  {% asset_img symmetry_point.png 500 100 对称点 %}
  {% asset_img rotation_point.png 500 100 旋转点 %}
  - 第三步：计算每个点$\bold{q'_k}$到形状$O$的最短距离$D_k$
  {% asset_img shortest_distance.png 300 60 最短距离 %}
  - 第四步：对所有对称平面的最短距离$\hat{D}_k$和旋转轴的最短距离$\tilde{D}_k$求和，得到对称距离损失$L_{sd}$
  {% asset_img Lsd.png 400 100 对称距离损失 %}
## 4.2 规范化损失（Regularization Loss）
  惩罚接近平行的对称面/旋转轴
  - 第一步：定义对称面法向量/旋转轴的单位方向矩阵$M_1,M_2$
    {% asset_img M1.png 300 40 单位方向矩阵 %}
    - 例如对于对称面法向量/旋转轴
      $$\vec{n}_1=(1,0,0),\quad \vec{n}_2=(0,1,0),\quad \vec{n}_3=(0,0,1)$$
      $$\vec{v}_1=(0,0,1),\quad \vec{v}_2=(0,1,0),\quad \vec{v}_3=(1,0,0)$$
    - 则单位方向矩阵为
      $$M_1=\begin{bmatrix}1&0&0\\0&1&0\\0&0&1\end{bmatrix}\quad M_2=\begin{bmatrix}0&0&1\\0&1&0\\1&0&0\end{bmatrix}$$
  - 第二步：计算矩阵$A$和$B$
    当$M_1,M_2$为正交矩阵时，$A,B$分别为零矩阵
    {% asset_img AB.png 200 80 矩阵AB %}
  - 第三步：计算规范化损失，即$A,B$矩阵的$Frobenius$范数之和
    {% asset_img FNorm.png 500 100 Frobenius范数 %}
## 4.3 综合损失函数
$$
\Large L=L_{sd}+w_rL_r
$$
其中$w_r$是规范化损失在综合损失中的所占权重，训练时自行定义

# 5 验证
现实中的形状，往往不是恰有3个对称面/旋转轴，因此验证阶段需要去除的重复或不够好的对称面/旋转轴
## 5.1 重复对称面/旋转轴的去除
  当对称面的二面角/旋转轴的夹角小于阈值（文中为$\pi/6$）时，去除对称距离误差（SDE）更大的一个
## 5.2 较差对称面/旋转轴的去除
  当某个对称面/旋转轴的对称距离误差（SDE）大于阈值（文中为$4\times 10^{-4}$）时，去除该对称面/旋转轴

# 6 实验
## 6.1 数据集
  - **ShapeNet**：包含55个类别的形状，共约51300个3D模型
  - 选取**80%** 的数据作为**训练集**，**20%** 的数据作为**测试集**
## 6.2 训练集数据增强
  - ShapeNet数据集中的形状通常轴对齐，且部分种类的模型较少
  - 采用**数据增强**的方法，对每个模型进行**随机旋转**，在每个种类上得到4000个模型用于训练
## 6.3 测试集人工验证
  - 人工验证并筛选出**存在对称面/旋转轴的形状**作为测试集
  - 人工获取测试集形状的**真实对称面/旋转轴**，与模型输出进行比较
  - 真实误差（Ground Truth Error，GTE）定义为：
    {% asset_img GTE.png 500 40 GTE %}
## 6.4 体素化预处理
  - 表面采样**1000**个点
  - 生成体素化输入的大小为$32\times32\times32$
## 6.5 训练参数
  - **批大小**为$b=32$
  - **学习率**为$l_r=0.01$
  - **规范化损失权重**为$w_r=25$
  - **Adam**优化器
# 7 结果和对比、参数选择等
> 详细内容请参考原文[PRS-Net: Planar Reflective Symmetry Detection Network](https://arxiv.org/abs/1910.06511v6)