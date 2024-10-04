---
title: Games101-Transformation
date: 2024-10-04 22:31:29
tags:
    - Games101
    - 3D Graphics
    - Transformation
categories:
    - 三维图形学
---

> 课程链接：[GAMES101-现代计算机图形学入门-闫令琪](https://www.bilibili.com/video/BV1X7411F744)

# 1 课程内容

- **2D/3D变换（Transformation）**：将物体从一个坐标系变换到另一个坐标系的过程
  - 两种线性变换：缩放、旋转、反射
  - 一种非线性变换：平移
- **MVP(Model, View, Project)变换**：将物体从模型坐标系变换到屏幕坐标系的过程
  - **Model**：模型坐标系
  - **View**：观察坐标系
  - **Project**：投影坐标系

# 2 2D变换
## 2.1 线性变换
**什么是线性变换？**
  $$
    x' = ax + by \\ y' = cx + dy\\
    \begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} a & b \\ c & d \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix}\\
    \begin{bmatrix} x' \\ y' \end{bmatrix} = M \begin{bmatrix} x \\ y \end{bmatrix}
  $$
- 基变换理解：
  将坐标$\begin{bmatrix} x \\ y \end{bmatrix}$看作是在以$\begin{bmatrix} a \\ c \end{bmatrix}$和$\begin{bmatrix} b \\ d \end{bmatrix}$为基向量的坐标系中的坐标，通过线性变换$M$将向量变换到新的基向量$\begin{bmatrix} 1 \\ 0 \end{bmatrix}$和$\begin{bmatrix} 0 \\ 1 \end{bmatrix}$上，得到新的坐标$\begin{bmatrix} x' \\ y' \end{bmatrix}$
- 线性变换的性质：
  - **封闭性**：线性变换的结果仍然是向量空间中的向量
  - **可逆性**：线性变换的逆变换也是线性变换
  - **保持原点**：线性变换的原点仍然是原点
  - **保持比例**：线性变换的比例关系不变
  - **保持角度**：线性变换的角度不变
- 左乘实现多次线性变换：
  例如：依次进行$M_1, M_2, \dots, M_n$ n次线性变换，则有：
  $$
    \begin{bmatrix} x' \\ y' \end{bmatrix} = M_n \dots M_2M_1\begin{bmatrix} x \\ y \end{bmatrix}
  $$
### 2.1.1 缩放变换
**缩放矩阵**：$S(s_x, s_y) = \begin{bmatrix} s_x & 0 \\ 0 & s_y \end{bmatrix}$
左乘缩放矩阵实现缩放变换：
  $$
    \begin{bmatrix} x' \\ y' \end{bmatrix} = S(s_x, s_y) \begin{bmatrix} x \\ y \end{bmatrix}
  $$

### 2.1.2 旋转变换
**旋转矩阵**：$R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$
左乘旋转矩阵实现旋转变换：
  $$
    \begin{bmatrix} x' \\ y' \end{bmatrix} = R(\theta) \begin{bmatrix} x \\ y \end{bmatrix}
  $$
特别地，旋转矩阵的逆矩阵为：
  $$
    R^{-1}(\theta) = R(-\theta) = \begin{bmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{bmatrix}
  $$
且旋转矩阵的转置矩阵为：
  $$
    R^T(\theta) = \begin{bmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{bmatrix} = R^{-1}(\theta)
  $$
因此，旋转矩阵是**正交矩阵**，即$R^T(\theta)R(\theta) = I$，其中$I$为单位矩阵

### 2.1.3 反射变换
**反射矩阵**：$F_x = \begin{bmatrix} -1 & 0 \\ 0 & 1 \end{bmatrix}$
左乘反射矩阵实现反射变换：
  $$
    \begin{bmatrix} x' \\ y' \end{bmatrix} = F_x \begin{bmatrix} x \\ y \end{bmatrix}
  $$

## 2.2 非线性变换
**什么是非线性变换？**
  $$
    x' = ax + by + t_x \\ y' = cx + dy + t_y\\
    \begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} a & b \\ c & d \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} + \begin{bmatrix} t_x \\ t_y \end{bmatrix}\\
    \begin{bmatrix} x' \\ y' \end{bmatrix} = M \begin{bmatrix} x \\ y \end{bmatrix} + T
  $$
- 基于非线性变换的平移变换：
  - 平移矩阵：$T = \begin{bmatrix} t_x \\ t_y \end{bmatrix}$
- 非线性变换带来的问题：
  - 无法通过矩阵乘法实现多次变换
  - 无法通过矩阵求逆实现逆变换

### 2.2.1 齐次坐标（Homogeneous Coordinates）
**为什么引入齐次坐标？**
- 解决非线性变换的两个问题
  - 可以将非线性变换转化为线性变换
  - 可以将多次变换转化为矩阵乘法

**齐次坐标的形式**：
- 坐标点：$\begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$
- 向量：$\begin{bmatrix} x \\ y \\ 0 \end{bmatrix}$

**齐次坐标的运算**：
- 坐标点 + 坐标点 = 坐标点
- 坐标点 - 坐标点 = 向量
- 坐标点 + 向量 = 坐标点
- 向量 +/- 向量 = 向量

### 2.2.2 齐次坐标下的变换
**齐次坐标下的线性变换**：
  $$
    M = \begin{bmatrix} a & b & 0 \\ c & d & 0 \\ 0 & 0 & 1 \end{bmatrix}\\
    \begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = M \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}
  $$

**齐次坐标下的非线性变换**：
  $$
    M = \begin{bmatrix} a & b & t_x \\ c & d & t_y \\ 0 & 0 & 1 \end{bmatrix}\\
    \begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = M \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}
  $$

**齐次坐标下的多次变换**：
左乘，依次进行$M_1, M_2, \dots, M_n$ n次变换，则有：
  $$
    \begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = M_n \dots M_2M_1\begin{bmatrix} x \\ y \\ 1 \end{bmatrix}
  $$

# 3 3D变换
## 3.1 3D齐次坐标
**3D齐次坐标的形式**：
- 坐标点：$\begin{bmatrix} x \\ y \\ z \\ 1 \end{bmatrix}$
- 向量：$\begin{bmatrix} x \\ y \\ z \\ 0 \end{bmatrix}$

## 3.2 3D变换
**3D线性变换**：
  $$
    M = \begin{bmatrix} a & b & c & 0 \\ d & e & f & 0 \\ g & h & i & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}\\
    \begin{bmatrix} x' \\ y' \\ z' \\ 1 \end{bmatrix} = M \begin{bmatrix} x \\ y \\ z \\ 1 \end{bmatrix}
  $$

**3D非线性变换**：
  $$
    M = \begin{bmatrix} a & b & c & t_x \\ d & e & f & t_y \\ g & h & i & t_z \\ 0 & 0 & 0 & 1 \end{bmatrix}\\
    \begin{bmatrix} x' \\ y' \\ z' \\ 1 \end{bmatrix} = M \begin{bmatrix} x \\ y \\ z \\ 1 \end{bmatrix}
  $$

**特别地，对于3D旋转变换**：
- 旋转矩阵：
  - $R_x(\theta) = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & \cos\theta & -\sin\theta & 0 \\ 0 & \sin\theta & \cos\theta & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$
  - $R_y(\theta) = \begin{bmatrix} \cos\theta & 0 & \sin\theta & 0 \\ 0 & 1 & 0 & 0 \\ -\sin\theta & 0 & \cos\theta & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$
  - $R_z(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta & 0 & 0 \\ \sin\theta & \cos\theta & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$

注意到， 3D旋转矩阵中$R_y$的副对角线上两个$sin\theta$符号相反，是因为对于右手系，$y$轴的正方向为$z$叉乘$x$得到
