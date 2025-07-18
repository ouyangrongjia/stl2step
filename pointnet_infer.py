# 模型导入
"""
- 功能：加载PointNet分割模型，并实现点云边缘点预测。
"""
import torch
import numpy as np
from pointnet.pointnet import model as pointnet_model
from config import MODEL_PATH, MAX_POINTS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    # 创建PointNetDenseCls模型实例，用于分割（k=2表示两类：边缘和非边缘）
    model = pointnet_model.PointNetDenseCls(k=2, feature_transform=False).to(device)
    # 设置为评估模式
    model.eval()
    return model

def predict_edge_points(model, points: np.ndarray):
    # points = points[:MAX_POINTS]
    # points = torch.tensor(points, dtype=torch.float32).unsqueeze(0).transpose(2, 1)
    # with torch.no_grad():
    #     pred = model(points)  # [1, num_classes]
    #     pred_label = pred.max(1)[1].item()
    # return pred_label  # 这里只是分类，后面扩展为逐点分类用 PointNetSeg

    # 将numpy点云数组转换为PyTorch张量，并放到设备上
    points_tensor = torch.tensor(points, dtype=torch.float32).to(device)
    # 调整张量维度：原始点云形状为[N, 3] -> 转为[1, 3, N]（因为模型要求batch在前，通道在中，点数在后）
    points_tensor = points_tensor.unsqueeze(0).transpose(2, 1)  # [1, 3, N]
    # 不计算梯度
    with torch.no_grad():
        # 模型返回三个值：logits（分割结果），trans（空间变换矩阵），trans_feat（特征变换矩阵）
        # 我们只需要logits，形状为[1, N, 2]
        logits, _, _ = model(points_tensor)  # 解包三个返回值
        # 在最后一个维度（类别维度）上取argmax，得到每个点的预测标签（0或1）
        # 然后取第一个batch（因为只有一个batch）得到一维数组[N]
        preds = logits.argmax(dim=-1)[0]  # 取第一个batch [N]

    # 将预测结果转回CPU并转为numpy数组，并转换为布尔类型（边缘点为True，非边缘点为False）
    return preds.cpu().numpy().astype(bool)

