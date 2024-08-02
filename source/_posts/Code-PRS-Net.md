---
title: PRS-Net误差和网络的代码实现
date: 2024-07-31 16:08:33
tags:
    - PRS-Net
    - CNN
    - Symmetry Detection
    - PyTorch
    - 3D Graphics
categories:
    - 论文复现
    - 三维图形学
    - Python程序
---

> 论文原文：[PRS-Net: Planar Reflective Symmetry Detection Network](https://arxiv.org/abs/1910.06511v6)

# 1 本文目的
探讨如何使用PyTorch实现PRS-Net的网络结构，以及如何计算误差

# 2 数据集及环境
- ShapeNetCore.v2经过体素化的模型
- CUDA 12.3

# 3 误差计算
误差计算实现中均使用PyTorch的mathematical functions进行张量计算，以便于在GPU上加速和进行反向传播
## 3.1 对称距离损失（Symmetry Distance Loss）
惩罚对称距离较远（即对称后偏离较大）的对称面/旋转轴
### 3.1.1 计算对称、旋转后的点云
关于待验证的对称面和对称轴，对原始样本点云进行对称和旋转变换
```python
def quat2Rmatrix(quaternion: torch.Tensor):
    '''
    将四元数转换为旋转矩阵
    @param quaternion: 旋转四元数 1x4
    @return rotation_matrix: 旋转矩阵 3x3
    '''
    w, x, y, z = quaternion
    rotation_matrix = torch.stack([
        torch.stack([1-2*y**2-2*z**2, 2*x*y-2*w*z, 2*x*z+2*w*y]),
        torch.stack([2*x*y+2*w*z, 1-2*x**2-2*z**2, 2*y*z-2*w*x]),
        torch.stack([2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x**2-2*y**2])
    ])
    return rotation_matrix

def symmetry_points(points: torch.Tensor, plane: torch.Tensor):
    '''
    对称化点云
    @param points: 点云 3xN
    @param plane: 对称平面 1x4
    @return sym_points: 对称化点云 3xN
    '''
    # 1. 截取对称平面的单位法向量 1x3
    plane_normal = plane[:3]
    d_value = plane[3]
    L2_norm = torch.norm(plane_normal)
    unit_normal = plane_normal / L2_norm
    # 2. 计算对称面到点的距离 1xN
    dist2plane = torch.sub(torch.matmul(unit_normal, points), d_value)
    # 3. 计算差值向量（两倍的距离）
    t_unit_normal = torch.unsqueeze(unit_normal, dim=1)
    sub_vector = torch.mul(t_unit_normal, dist2plane)
    # 4. 对称化点云
    sym_points = torch.sub(points, sub_vector, alpha=2)
    return sym_points

def rotate_points(points: torch.Tensor, quaternion: torch.Tensor):
    '''
    旋转点云
    @param points: 点云 3xN
    @param quaternion: 旋转四元数 1x4
    @return rotated_points: 旋转后的点云 3xN
    '''
    # 1. 计算旋转矩阵
    rotation_matrix = quat2Rmatrix(quaternion)
    # 2. 点云旋转
    rotated_points = torch.matmul(rotation_matrix, points)
    return rotated_points
```

### 3.1.2 计算对称距离
对称距离损失为原始样本点云到变换后点云的最短距离之和
将对称面和旋转轴的对称距离损失相加，得到总的对称距离损失
```python
def shortest_distance(points: torch.Tensor, sample: torch.Tensor):
    '''
    计算点到点云的最短距离
    @param points: 点云 3xN
    @param sample: 采样点 3x1
    @return distance: 最短距离 1xM
    '''
    # 1. 计算所有样本点到点云间的距离
    all_distance = torch.norm(torch.sub(points, torch.unsqueeze(sample, dim=1)), dim=0)
    # 2. 计算最短距离
    shortest_distance = torch.min(all_distance)
    return shortest_distance

def plane_sde_loss(plane: torch.Tensor, samples: torch.Tensor, points: torch.Tensor, *, device: torch.device):
    '''
    对称距离损失函数
    @param plane: 预测的对称平面 1x4
    @param samples: 采样点 3xM
    @param points: 模型所有顶点 3xN
    @param device: 设备
    @return plane_loss: 对称距离损失
    '''
    # 计算对称平面的对称距离损失SDE
    plane_loss = 0
    # 1. 计算对称后的采样点
    sym_samples = symmetry_points(samples, plane)
    # 2. 计算采样点到形状的最短距离
    for sample in sym_samples.transpose(0, 1):
        dist = shortest_distance(points, sample)
        # 3. 累加损失
        plane_loss += dist
    return plane_loss

def quat_sde_loss(quaternion: torch.Tensor, samples: torch.Tensor, points: torch.Tensor, *, device: torch.device):
    '''
    对称距离损失函数
    @param quaternion: 预测的旋转轴 1x4
    @param samples: 采样点 3xM
    @param points: 模型所有顶点 3xN
    @param device: 设备
    @return quaternion_loss: 对称距离损失
    '''
    # 计算旋转轴的对称距离损失SDE
    quaternion_loss = 0
    # 1. 计算旋转后的采样点
    rotated_samples = rotate_points(samples, quaternion)
    # 2. 计算采样点到形状的最短距离
    for sample in rotated_samples.transpose(0, 1):
        dist = shortest_distance(points, sample)
        # 3. 累加损失
        quaternion_loss += dist
    return quaternion_loss

def SDELoss(planes: torch.Tensor, quaternions: torch.Tensor, samples: torch.Tensor, points: torch.Tensor, *, device: torch.device):
    '''
    对称距离损失函数
    @param planes: 预测的对称平面 3x4
    @param quaternions: 预测的旋转轴 3x4
    @param samples: 采样点 3xM
    @param points: 模型所有顶点 3xN
    @param device: 设备
    @return sde_loss: 对称距离损失
    '''
    # 1. 计算对称平面的对称距离损失SDE
    plane_loss = 0
    for plane in planes:
        plane_loss += plane_sde_loss(plane, samples, points, device=device)
    
    # 2. 计算旋转轴的对称距离损失SDE
    quaternion_loss = 0
    for quat in quaternions:
        quaternion_loss += quat_sde_loss(quat, samples, points, device=device)
    
    # 三、计算总损失
    sde_loss = plane_loss + quaternion_loss
    return sde_loss
```

## 3.2 规范化损失（Regularization Loss）
惩罚接近平行的对称面/旋转轴，避免生成重复的对称平面
```python
def quat2axisangle(quaternion: torch.Tensor):
    '''
    将四元数转换为旋转轴和旋转角
    @param quaternion: 旋转四元数 1x4
    @return axis: 旋转轴 1x3
    @return angle: 旋转角 1x1
    '''
    w, x, y, z = quaternion
    half_angle = torch.acos(w)
    angle = torch.mul(2, half_angle)
    if half_angle == 0:
        axis = torch.tensor([0, 0, 0])
    else:
        axis = torch.div(torch.stack([x, y, z]), torch.sin(half_angle))
    return axis, angle

def RegLoss(planes, quaternions, *, device: torch.device):
    '''
    正则化损失函数
    @param planes: 预测的对称平面 3x4
    @param quaternions: 预测的旋转轴 3x4
    @param device: 设备
    @return reg_loss: 正则化损失
    '''
    # 1. 计算对称平面法向量的单位向量
    plane_normals = planes[:, :3]
    unit_plane_normals = F.normalize(plane_normals, p=2, dim=1)
    # 2. 计算旋转轴的单位向量
    rotate_axes = []
    for quat in quaternions:
        axis, _ = quat2axisangle(quat)
        rotate_axes.append(axis)
    rotate_axes = torch.stack(rotate_axes)
    unit_rotate_axes = F.normalize(rotate_axes, p=2, dim=1)
    # 3. 计算正交性损失
    planes_matrixA = torch.sub(torch.matmul(unit_plane_normals, unit_plane_normals.transpose(0, 1)), torch.eye(3, device=device))
    axes_matrixB = torch.sub(torch.matmul(unit_rotate_axes, unit_rotate_axes.transpose(0, 1)), torch.eye(3, device=device))
    # 4. 计算F2范数平方的和
    reg_loss = torch.add(torch.sum(torch.pow(planes_matrixA, 2)), torch.sum(torch.pow(axes_matrixB, 2)))
    return reg_loss
```

# 4 CNN网络实现
## 4.1 网络结构
{% asset_img Network.png PRS-Net %}
- 5个3D卷积层
  - 卷积核大小（kernel size）为3
  - 填充（padding）为1
  - 步幅（stride）为1
  - 每层后使用最大值池化（Max Pooling），卷积核大小为2
  - 激活函数使用Leaky ReLU
- 全连接层
## 4.2 代码实现
```python
"""
实现CNN网络结构：
    - 5个3D卷积层
        - 卷积核大小（kernel size）为3
        - 填充（padding）为1
        - 步幅（stride）为1
        - 每层后使用最大值池化（Max Pooling），卷积核大小为2
        - 激活函数使用Leaky ReLU
    - 全连接层
输入：
    32x32x32像素的3D体素数据
输出：
    3个4参数隐式表示的对称平面，3个4参数轴角表示的旋转轴
"""
import torch
import torch.nn as nn

class PRS_Net(nn.Module):
    def __init__(self):
        super().__init__()
        # 5个3D卷积层配置
        self.conv1 = nn.Sequential(nn.Conv3d(1, 4, 3, padding=1), nn.MaxPool3d(2, 2), nn.LeakyReLU())
        self.conv2 = nn.Sequential(nn.Conv3d(4, 8, 3, padding=1), nn.MaxPool3d(2, 2), nn.LeakyReLU())
        self.conv3 = nn.Sequential(nn.Conv3d(8, 16, 3, padding=1), nn.MaxPool3d(2, 2), nn.LeakyReLU())
        self.conv4 = nn.Sequential(nn.Conv3d(16, 32, 3, padding=1), nn.MaxPool3d(2, 2), nn.LeakyReLU())
        self.conv5 = nn.Sequential(nn.Conv3d(32, 64, 3, padding=1), nn.MaxPool3d(2, 2), nn.LeakyReLU())
        # 全连接层配置
        self.plane_seqs = nn.ModuleList([
            nn.Sequential(nn.Linear(64, 32), nn.LeakyReLU(), nn.Linear(32, 16), nn.LeakyReLU(), nn.Linear(16, 4)),
            nn.Sequential(nn.Linear(64, 32), nn.LeakyReLU(), nn.Linear(32, 16), nn.LeakyReLU(), nn.Linear(16, 4)),
            nn.Sequential(nn.Linear(64, 32), nn.LeakyReLU(), nn.Linear(32, 16), nn.LeakyReLU(), nn.Linear(16, 4))
        ])
        self.quat_seqs = nn.ModuleList([
            nn.Sequential(nn.Linear(64, 32), nn.LeakyReLU(), nn.Linear(32, 16), nn.LeakyReLU(), nn.Linear(16, 4)),
            nn.Sequential(nn.Linear(64, 32), nn.LeakyReLU(), nn.Linear(32, 16), nn.LeakyReLU(), nn.Linear(16, 4)),
            nn.Sequential(nn.Linear(64, 32), nn.LeakyReLU(), nn.Linear(32, 16), nn.LeakyReLU(), nn.Linear(16, 4))
        ])

    def conv_layers(self, input):
        '''
        5个3D卷积层实现
        @param input: 输入数据[batch_size, 1, 32, 32, 32]
        @return output: 输出数据[batch_size, 64, 1, 1, 1]
        '''
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        output = self.conv5(x)
        return output
    
    def full_connect(self, input):
        '''
        全连接层实现
        @param input: 输入数据[batch_size, 64, 1, 1, 1]
        @return planes: 3个4参数隐式表示的对称平面[batch_size, 3, 4]
        @return quaternions: 3个4参数轴角表示的旋转轴[batch_size, 3, 4]
        '''
        planes = []
        quaternions = []
        for i in range(3):
            x = input.view(input.size(0), -1)
            plane = self.plane_seqs[i](x)
            quaternion = self.quat_seqs[i](x)
            planes.append(plane)
            quaternions.append(quaternion)
        planes = torch.stack(planes, dim=1)
        quaternions = torch.stack(quaternions, dim=1)
        return planes, quaternions

    def forward(self, input):
        # 前向传播
        x = self.conv_layers(input)
        output = self.full_connect(x)
        return output
```

# 5 训练与测试
## 5.1 配置文件
将静态参数保存在配置文件`config.yml`中，方便调整参数
```yaml
basic:
  dataset_path: './datasets/'
  save_path: './results/'
gpu:
  core_num: 1
hyperparameters:
  epochs: 5
  batch_size: 32
  learning_rate: 0.01
  reg_weight: 25
test:
  dihedral_angle_bound: 0.5235987756
  sde_bound: 4e-4
```

## 5.2 数据集准备
```python
"""
从mat文件中读取数据
"""
import scipy.io as sio
import numpy as np
import torch
import os
from torch.utils.data import Dataset

def parseMat(mat_path):
    """
    从mat文件中读取数据
    :param mat_path: mat文件路径
    :return: 数据
    """
    data = sio.loadmat(mat_path)
    result = {}
    for key in data.keys():
        if '__' not in key:
            result[key] = data[key]
    return result

class RawData:
    '''
    原始数据类
    '''
    def __init__(self, data):
        self.data = data
        self.volumn = torch.tensor(data['volumn'], dtype=torch.float32)
        self.vertices = torch.tensor(data['volumn_vertices'], dtype=torch.float32)
        self.samples = torch.tensor(data['volumn_samples'], dtype=torch.float32)
        self.faces = torch.tensor(data['faces'], dtype=torch.float32)
        self.rotate_axisangle = torch.tensor(data['rotate_axisangle'], dtype=torch.float32)
        self.voxel_centers = torch.tensor(data['voxel_centers'], dtype=torch.float32)
    
    def __getitem__(self, key):
        return self.data[key]
    
    def __setitem__(self, key, value):
        pass

    def volumn(self):
        return self.volumn
    
    def vertices(self):
        return self.vertices
    
    def samples(self):
        return self.samples
    
    def faces(self):
        return self.faces
    
    def rotate_axisangle(self):
        return self.rotate_axisangle
    
    def voxel_centers(self):
        return self.voxel_centers

class ShapeNetData:
    """
    基于ShapeNet数据集的数据加载类
    """
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.dir_list = os.listdir(self.root_dir)
    
    def __len__(self):
        return len(self.dir_list)
    
    def __getitem__(self, idx):
        mat_path = os.path.join(self.root_dir, self.dir_list[idx])
        data = parseMat(mat_path)
        return data

class ShapeNetDataset(Dataset):
    '''
    体素模型的数据加载类
    '''
    def __init__(self, root_dir):
        self.data = ShapeNetData(root_dir)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        volumn = torch.tensor(data['volumn'], dtype=torch.float32)
        volumn = volumn.expand(1, 32, 32, 32)
        return volumn
    
    def raw_data(self, idx):
        raw_data = self.data[idx]
        return RawData(raw_data)


if __name__ == '__main__':
    # 模块测试代码
    from torch.utils.data import DataLoader
    root_dir = './datasets/train/'
    dataset = ShapeNetDataset(root_dir)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    for i, data in enumerate(dataloader):
        print(data.shape)
        break
```

## 5.3 训练
```python
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.loss import SumLoss
from utils.network import PRS_Net
from utils.data import ShapeNetDataset

def train():
    # 配置参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open('./config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    data_path = config['basic']['dataset_path'] + 'train/'
    save_path = config['basic']['save_path']
    # 配置超参数
    hyperparameters = config['hyperparameters']
    epochs = hyperparameters['epochs']
    batch_size = hyperparameters['batch_size']
    learning_rate = hyperparameters['learning_rate']
    reg_wight = hyperparameters['reg_weight']


    # 加载数据
    dataset = ShapeNetDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 加载网络
    net = PRS_Net().to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # 训练
    for epoch in range(epochs):
        print('epoch: ', epoch + 1)
        running_loss = 0.0
        for i, data in enumerate(dataloader):
            print('batch: ', i + 1)
            inputs = data.to(device)
            optimizer.zero_grad()
            planes_batch, quaternions_batch = net(inputs)

            for batch in range(batch_size):
                planes = planes_batch[batch].clone().detach().requires_grad_(True)
                quaternions = quaternions_batch[batch].clone().detach().requires_grad_(True)
                model_idx = i * batch_size + batch
                samples = dataset.raw_data(model_idx).samples.to(device)
                vertices = dataset.raw_data(model_idx).vertices.to(device)
                loss = SumLoss(planes, quaternions, samples, vertices, reg_wight, device=device)
                loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print('loss: ', loss.item())
            if i % 200 == 199:
                print('%d-[%5d, %5d] avg_loss: %.3f' % (epoch + 1, i - 198, i + 1, running_loss / 200))
                running_loss = 0.0
    print('Finished Training')
    torch.save(net.state_dict(), save_path + 'PRS-Net.pth')

if __name__ == '__main__':
    train()
```

## 5.4 测试
```python
import yaml
import torch
from torch.utils.data import DataLoader
from utils.data import ShapeNetDataset
from utils.loss import plane_sde_loss, quat_sde_loss

def test():
    # 配置参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open('./config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    data_path = config['basic']['dataset_path'] + 'test/'
    save_path = config['basic']['save_path']
    sde_bound = config['test']['sde_bound']
    dihedral_angle_bound = config['test']['dihedral_angle_bound']
    
    # 加载模型
    net = torch.load(save_path + 'PRS-Net.pth').to(device)
    net.eval()

    # 加载数据
    dataset = ShapeNetDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 测试
    for i, data in enumerate(dataloader):
        inputs = data.to(device)
        planes, quaternions = net(inputs)
        samples = dataset.raw_data(i).samples.to(device)
        vertices = dataset.raw_data(i).vertices.to(device)
        planes, quaternions = check_invalid(planes, quaternions, samples, vertices, bound=sde_bound, device=device)
        planes, quaternions = check_repeat(planes, quaternions, samples, vertices, bound=dihedral_angle_bound, device=device)
        print('planes:', planes, sep='\n')
        print('quaternions:', quaternions, sep='\n')
        if i == 0:
            break

def check_invalid(planes, quaternions, samples, points, *, bound, device):
    '''
    检查无效的对称平面和旋转轴
    @param planes: 对称平面列表 3x4
    @param quaternions: 旋转轴列表 3x4
    @param samples: 采样点 3xM
    @param points: 模型所有顶点 3xN
    @param device: 设备
    @return planes: 有效的对称平面列表
    @return quaternions: 有效的旋转轴列表
    '''
    # 去除SDE大于阈值的对称平面和旋转轴
    for i in range(len(planes)):
        if plane_sde_loss(planes[i], samples, points, device=device) > bound:
            planes.pop(i)
    for i in range(len(quaternions)):
        if quat_sde_loss(quaternions[i], samples, points, device=device) > bound:
            quaternions.pop(i)
    return planes, quaternions

def check_repeat(planes, quaternions, samples, points, *, bound, device):
    '''
    检查重复的对称平面和旋转轴
    @param planes: 对称平面列表 3x4
    @param quaternions: 旋转轴列表 3x4
    @param device: 设备
    @return planes: 去重后的对称平面列表
    @return quaternions: 去重后的旋转轴列表
    '''
    # 两个法向量的夹角小于阈值则认为是同一个平面，去除SDE较大的平面
    for i in range(len(planes)):
        for j in range(i + 1, len(planes)):
            if dihedral_angle(planes[i], planes[j]) < bound:
                if plane_sde_loss(planes[i], samples, points, device=device) < plane_sde_loss(planes[j], samples, points, device=device):
                    planes.pop(j)
                else:
                    planes.pop(i)
    # 两个旋转轴的夹角小于阈值则认为是同一个旋转轴，去除SDE较大的旋转轴
    for i in range(len(quaternions)):
        for j in range(i + 1, len(quaternions)):
            if dihedral_angle(quaternions[i], quaternions[j]) < bound:
                if quat_sde_loss(quaternions[i], samples, points, device=device) < quat_sde_loss(quaternions[j], samples, points, device=device):
                    quaternions.pop(j)
                else:
                    quaternions.pop(i)
    return planes, quaternions

def dihedral_angle(v1, v2):
    '''
    计算两个向量的二面角
    @param v1: 向量1 1x3
    @param v2: 向量2 1x3
    @return angle: 二面角
    '''
    angle = torch.acos(torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2)))
    return angle

if __name__ == '__main__':
    test()
```