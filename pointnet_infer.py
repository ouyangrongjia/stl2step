# 模型导入
"""
- 功能：加载PointNet分割模型，并实现点云边缘点预测。
"""
import torch
import numpy as np
# from pointnet.pointnet import model as pointnet_model
from config import MODEL_PATH, MAX_POINTS
from Pointnet_Pointnet2_pytorch.models import pointnet2_sem_seg_msg as pointnet_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    # 创建PointNetDenseCls模型实例，用于分割（k=2表示两类：边缘和非边缘）
    # model = pointnet_model.PointNetDenseCls(k=2, feature_transform=False).to(device)
    model = pointnet_model.get_model(num_classes=2).to(device)
    # 设置为评估模式
    model.eval()
    return model

def prepare_pointnet2_input(points: np.ndarray) -> torch.Tensor:
    """
    将 Nx3 点转换为 Bx9xN 的 PointNet++ 输入（补零填充）
    """
    N = points.shape[0]
    xyz = points.T  # (3, N)
    fake_feats = np.zeros((6, N), dtype=np.float32)  # 用0占位附加特征
    full_feats = np.concatenate([xyz, fake_feats], axis=0)  # (9, N)
    return torch.tensor(full_feats, dtype=torch.float32).unsqueeze(0)  # (1, 9, N)

def predict_edge_points(model, points: np.ndarray):
    # points_tensor = torch.tensor(points, dtype=torch.float32).to(device)
    # # 调整张量维度：原始点云形状为[N, 3] -> 转为[1, 3, N]（因为模型要求batch在前，通道在中，点数在后）
    # points_tensor = points_tensor.unsqueeze(0).transpose(2, 1)  # [1, 3, N]
    input_tensor = prepare_pointnet2_input(points).to(device)
    # 不计算梯度
    with torch.no_grad():
        logits, _ = model(input_tensor) # 输出: [1, N, 2]
        # 在最后一个维度（类别维度）上取argmax，得到每个点的预测标签（0或1）
        # 然后取第一个batch（因为只有一个batch）得到一维数组[N]
        preds = logits.argmax(dim=-1)[0]  # 取第一个batch [N]

    # 将预测结果转回CPU并转为numpy数组，并转换为布尔类型（边缘点为True，非边缘点为False）
    return preds.cpu().numpy().astype(bool)

