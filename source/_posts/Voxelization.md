---
title: PRS-Net中OBJ格式的体素化处理
date: 2024-07-23 23:25:15
tags:
    - PRS-Net
    - Voxelization
    - MATLAB
    - 3D Graphics
categories:
    - 论文复现
    - 三维图形学
    - MATLAB程序
---

> 论文原文：[PRS-Net: Planar Reflective Symmetry Detection Network](https://arxiv.org/abs/1910.06511v6)

# 1 本文目的
探讨如何使用MATLAB将OBJ格式的三维模型体素化，以便于PRS-Net的输入

# 2 数据集及环境
- ShapeNetCore.v2中的模型（OBJ格式，基于数据量需要对基础模型进行旋转变换）
- MATLAB R2022a


# 3 体素化效果
体素化前后的模型对比如下：
  - 未体素化的模型：
  {% asset_img raw_model.png 440 330 体素化前 %}
  - 体素化后的模型：
  {% asset_img eg_volumn.png 440 330 体素化后 %}

# 4 实现思路
> 将三维模型转换为三维体素网格

## 4.1 读取OBJ文件
使用readOBJ函数读取OBJ文件，获取模型的顶点和三角面信息（体素化使用）
```matlab
% 读取obj文件的顶点和三角面
% % vertices中元素为点坐标：[x,y,z]
% % faces中元素为面的三个点序号: [v1,v2,v3]
[vertices, faces, ~] = readOBJ(obj_path);
```
同时使用meshlpsampling函数从OBJ表面随机选取一定数量的点作为采样点（无监督训练时计算误差使用）
```matlab
% 从obj表面随机选取sample_num个点
% % surf_samples中元素为点坐标: [x,y,z]
[surf_samples, ~] = meshlpsampling(obj_path, sample_num);
```

## 4.2 旋转变换增加模型数量
为了增加数据集的多样性，对基础模型进行旋转变换，生成更多的模型
配置每个种类的目标模型数，从而计算平均到每个模型所需的模型变种数
```matlab
% 配置每个种类的目标模型数 (数量不足时对原模型做变换)
models_need = 4000;
% ...
% 计算平均到每个模型所需的模型变种数
models_need_each = ceil(models_need / length(model_ids));
```
对每个模型的顶点进行旋转变换，生成更多的模型
```matlab
% 生成随机的规范化旋转轴
axis = rand(1,3);
axis = axis/norm(axis);
% 生成随机的旋转角度
angle = rand(1)*2*pi;
% 合成轴角表示
rotate_axisangle = [axis, angle];
% 生成旋转矩阵
rotate_matrix = axang2rotm(rotate_axisangle);
% 旋转所有顶点 (三角面只和顶点序号相关，无需变换)
% % 取顶点矩阵的转置，以和旋转矩阵相乘
r_vertices = rotate_matrix*vertices';
r_surf_samples = rotate_matrix*surf_samples;
```

## 4.3 顶点坐标对应到体素坐标
OBJ原始模型的顶点坐标是规范化的，坐标值在[-0.5,0.5]之间
需要将其转换到体素坐标系，使得坐标值在[0,volumn_size]之间
```matlab
% 将顶点的规范化坐标系平移缩放到体素对应的位置
volumn_vertices = volumn_size*(r_vertices + 0.5) + 0.5;
volumn_samples = volumn_size*(r_surf_samples + 0.5) + 0.5;
```
## 4.4 将三角面的索引形式转换为顶点坐标
OBJ模型的三角面是由三个顶点的索引组成的，将其转换为顶点坐标以便后续体素化处理
```matlab
function volumn = model2volumn(model,volumn_size)
    % volumn_size必须为整数
    volumn_size = round(volumn_size);
    volumn_size = [volumn_size, volumn_size, volumn_size];
    % 按列划分顶点和三角面
    vertices = struct();
    vertices.x = double(model.vertices(1,:));
    vertices.y = double(model.vertices(2,:));
    vertices.z = double(model.vertices(3,:));

    faces = struct();
    faces.v1 = double(model.faces(:,1));
    faces.v2 = double(model.faces(:,2));
    faces.v3 = double(model.faces(:,3));

    % 得到double类型的顶点和三角面的顶点坐标
    % % double_vertices为N×3的矩阵
    % % face_vertices为结构体{'v1': N×3, 'v2': N×3, 'v3': N×3}
    % % 上述N×3矩阵结构相同，每行为点坐标[x,y,z]
    double_vertices = [vertices.x',vertices.y',vertices.z'];
    face_vertices = struct();
    face_vertices.v1 = double_vertices(faces.v1,:);
    face_vertices.v2 = double_vertices(faces.v2,:);
    face_vertices.v3 = double_vertices(faces.v3,:);

    % 将volumn_size转换为double，以便处理
    volumn_size = double(volumn_size);
    % 体素化处理
    volumn = process_model(face_vertices, volumn_size);
end
```

## 4.5 细分三角面
将三角面细分为更小的三角面，使得每个三角面的边长不超过0.5个体素
从而得到足够精细的体素化模型
```matlab
function face_vertices_splited = split_faces(face_vertices)
    splited_faces = struct('v1',[],'v2',[],'v3',[]);
    while(~isempty(face_vertices.v1))
        disp(['remain faces: ', num2str(length(face_vertices.v1))]);
        % 距离矩阵dist，每列分别代表一个边距离[(v1-v2),(v2-v3),(v1-v3)]，初始化为全零矩阵
        dist = zeros(length(face_vertices.v1(:,1)),3);
        % 分别计算x^2，y^2，z^2并求和，得到距离
        for i=1:3
            dist(:,1) = dist(:,1) + (face_vertices.v1(:,i)-face_vertices.v2(:,i)).^2;
            dist(:,2) = dist(:,2) + (face_vertices.v2(:,i)-face_vertices.v3(:,i)).^2;
            dist(:,3) = dist(:,3) + (face_vertices.v1(:,i)-face_vertices.v3(:,i)).^2;
        end
        % 得到三边最大值，构成列向量
        maxdist = max(dist,[],2);
        % 得到最大边长大于0.5个体素的三角面，在can_split列向量对应行标记为1
        can_split = maxdist > 0.5;
        splited_faces.v1 = [splited_faces.v1;face_vertices.v1(~can_split,:)];
        splited_faces.v2 = face_vertices.v2(~can_split,:);
        splited_faces.v3 = face_vertices.v3(~can_split,:);
        % 更新face_vertices和dist，去除无法再分割的三角面
        face_vertices_temp = face_vertices;
        face_vertices_temp.v1 = face_vertices.v1(can_split,:);
        face_vertices_temp.v2 = face_vertices.v2(can_split,:);
        face_vertices_temp.v3 = face_vertices.v3(can_split,:);
        face_vertices = face_vertices_temp;
        dist = dist(can_split,:);
        % 计算仍能分割的三角面的三边中点
        mid_vertices = struct();
        for i=1:3
            mid_vertices.v12(:,i) = (face_vertices.v1(:,i)+face_vertices.v2(:,i))/2;
            mid_vertices.v23(:,i) = (face_vertices.v2(:,i)+face_vertices.v3(:,i))/2;
            mid_vertices.v13(:,i) = (face_vertices.v1(:,i)+face_vertices.v3(:,i))/2;
        end
        % 计算得到各边最长的三角面
        max_v12 = [dist(:,1) >= dist(:,2) & dist(:,1) >= dist(:,3)];
        max_v23 = [dist(:,2) >= dist(:,1) & dist(:,2) >= dist(:,3) & ~max_v12];
        max_v13 = [dist(:,3) >= dist(:,1) & dist(:,3) >= dist(:,2) & ~max_v12 & ~max_v23];
        % 左替换最长边结点为中点
        left_split = face_vertices;
        left_split.v1(max_v12,:) = mid_vertices.v12(max_v12,:);
        left_split.v2(max_v23,:) = mid_vertices.v23(max_v23,:);
        left_split.v1(max_v13,:) = mid_vertices.v13(max_v13,:);
        % 右替换最长边结点为中点
        right_split = face_vertices;
        right_split.v2(max_v12,:) = mid_vertices.v12(max_v12,:);
        right_split.v3(max_v23,:) = mid_vertices.v23(max_v23,:);
        right_split.v3(max_v13,:) = mid_vertices.v13(max_v13,:);
        % 拼接左右替换
        face_vertices.v1 = [left_split.v1;right_split.v1];
        face_vertices.v2 = [left_split.v2;right_split.v2];
        face_vertices.v3 = [left_split.v3;right_split.v3];
    end
    face_vertices_splited = splited_faces;
end
```

## 4.6 根据细分后的顶点坐标生成体素
根据细分后的顶点坐标生成体素，将体素坐标转换为线性索引（提高性能）
从而赋值给体素矩阵，得到体素化后的模型
```matlab
function volumn = process_model(face_vertices, volumn_size)
    % 生成空体素
    volumn = false(volumn_size);
    % 细分三角面
    face_vertices_splited = split_faces(face_vertices);
    % 拼接顶点坐标矩阵，并去除重复行
    splited_vertices = [face_vertices_splited.v1;face_vertices_splited.v2;face_vertices_splited.v3];
    splited_vertices = unique(splited_vertices, 'rows', 'stable');
    % 得到点坐标对应的体素坐标
    voxel_vertices = [floor(splited_vertices(:,1))+1,floor(splited_vertices(:,2))+1,floor(splited_vertices(:,3))+1];
    voxel_vertices = unique(voxel_vertices, 'rows', 'stable');
    % 分别对 voxel_vertices 的每一列应用边界限制
    voxel_vertices(:,1) = max(min(voxel_vertices(:,1), volumn_size(1)), 1);
    voxel_vertices(:,2) = max(min(voxel_vertices(:,2), volumn_size(2)), 1);
    voxel_vertices(:,3) = max(min(voxel_vertices(:,3), volumn_size(3)), 1);
    % 将向量转换为线性索引，利用索引为对应体素位赋值
    linear_indices = sub2ind(size(volumn), voxel_vertices(:,1), voxel_vertices(:,2), voxel_vertices(:,3));
    volumn(linear_indices) = true;
end
```

## 4.7 保存体素化结果
得到体素化后的模型，可以将其保存为MAT文件，以便PRS-Net的输入
```matlab
% 将顶点和三角面组合为结构体
model = struct();
model.vertices = volumn_vertices;
model.faces = faces;
volumn = model2volumn(model, volumn_size);
% 将所需变量保存
save_name = [save_path, '/', model_ids{j}, '_r', num2str(k), '.mat'];
save(save_name, "volumn", "volumn_vertices", "volumn_samples", "faces", "rotate_axisangle", "voxel_centers");
```
