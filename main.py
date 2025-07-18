from stl_reader import load_stl
from pointnet_infer import load_model, predict_edge_points
from step_writer import mark_edge_points_on_shape, export_step
from config import STL_PATH, OUTPUT_STEP_PATH
from visualize_edges import visualize_points
from edge_graph import build_edge_graph
from wire_builder import build_connected_edges
from OCC.Core.BRep import BRep_Builder
from OCC.Core.TopoDS import TopoDS_Compound
import torch
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_SAMPLED_POINTS = 50000
def clean_points(points):
    points = np.unique(points, axis=0)
    points = np.array(points, dtype=np.float32)
    assert points.ndim == 2 and points.shape[1] == 3, "点数据必须为 Nx3 的 float32 数组"
    return points

def main():
    # 加载STL点云与几何体
    shape, points = load_stl(STL_PATH)
    # 去除重复顶点
    points = clean_points(points)
    print(f"去除重复顶点后，剩余点数：{len(points)}")
    print(f"加载STL点云与几何体完成，读取点数：{len(points)}")

    # 若去重后点数仍超限，则进行随机采样
    if len(points) > MAX_SAMPLED_POINTS:
        print(f"点数超过{MAX_SAMPLED_POINTS}，进行随机采样...")
        rng = np.random.default_rng()
        sampled_indices = rng.choice(len(points), size=MAX_SAMPLED_POINTS, replace=False)
        points = points[sampled_indices]
        print(f"采样后最终处理点数：{len(points)}")

    model = load_model().to(device)
    print(f"模型加载完成: {type(model).__name__}")

    # 边缘点识别
    # 设置批大小（避免一次性处理整个点云导致显存溢出）
    batch_size = 100000
    # 初始化边缘标签数组，全为False（非边缘）
    edge_labels = np.zeros(len(points), dtype=bool)
    # 分批处理点云
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        # 预测当前批次，得到布尔数组（边缘点为True）
        edge_labels[i:i + batch_size] = predict_edge_points(model, batch)

    # 从点云中提取边缘点
    edge_points = points[edge_labels]
    edge_points = clean_points(edge_points)

    print(f"边缘点识别完成，共 {len(edge_points)} 个")

    # 将边缘点标记到原始几何体上，形成复合体 打点并导出STEP文件
    shape_with_edges = mark_edge_points_on_shape(shape, edge_points)

    # # 构建边缘线段几何体
    # print("构建边缘点连线结构...")
    # edge_indices = build_edge_graph(edge_points, k=5)
    # edge_lines = build_connected_edges(edge_points, edge_indices)
    #
    # # 合并几何体和边线
    # print("合并几何体和边线，准备导出STEP...")
    # builder = BRep_Builder()
    # compound = TopoDS_Compound()
    # builder.MakeCompound(compound)
    # builder.Add(compound, shape)  # 原始几何体
    # builder.Add(compound, edge_lines)  # 边缘线段集合

    # 导出
    success = export_step(shape=shape_with_edges, output_path=OUTPUT_STEP_PATH, unit="MM", schema="AP214")
    print("STEP文件 导出成功" if success else "STEP文件 导出失败")

    # 边缘点可视化
    print("启动边缘点可视化窗口......")
    visualize_points(points, edge_labels, title="PointNet Predict Result")

if __name__ == '__main__':
    main()
