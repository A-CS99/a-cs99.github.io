---
title: PRS-Net笔记
date: 2024-07-06 14:14:01
tags: 
    - PRS-Net
    - CNN
    - Symmetry Detection
    - Paper
categories: 
    - 论文笔记
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
  - 第一步：从输入形状$O$上选取$N$个样本点构成点集$Q$，基于这个点集进行后续计算
  - 第二步：计算点集$Q$上任意一点$\bold{q_k}$对称/旋转后得到的点$\bold{q'_k}$
  {% asset_img symmetry_point.png 500 100 对称点 %}
  {% asset_img rotation_point.png 500 100 旋转点 %}
  - 第三步：计算每个点$\bold{q'_k}$到形状$O$的最短距离$D_k$
  {% asset_img shortest_distance.png 300 60 最短距离 %}
  - 第四步：对所有对称平面的最短距离$\hat{D}_k$和旋转轴的最短距离$\tilde{D}_k$求和，得到对称距离损失$L_{sd}$
  {% asset_img Lsd.png 400 100 对称距离损失 %}
## 4.2 规范化损失（Regularization Loss）
  - 第一步：定义