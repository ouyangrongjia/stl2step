from stl_reader import load_stl
from pointnet_infer import load_model, predict_edge_points
from step_writer import mark_edge_points_on_shape, export_step
from config import STL_PATH, OUTPUT_STEP_PATH
from visualize_edges import visualize_points
from edge_graph import build_edge_graph
from wire_builder import build_connected_edges
from OCC.Core.BRep import BRep_Builder
from OCC.Core.TopoDS import TopoDS_Compound
from shape_processor import heal_shape
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
    # 1. 加载STL点云与几何体
    raw_shape, points = load_stl(STL_PATH)
    points = clean_points(points)
    print(f"去除重复顶点后，剩余点数：{len(points)}")

    if len(points) > MAX_SAMPLED_POINTS:
        rng = np.random.default_rng()
        sampled_indices = rng.choice(len(points), size=MAX_SAMPLED_POINTS, replace=False)
        points = points[sampled_indices]
        print(f"点数超限，已随机采样至 {len(points)} 个点。")

    # 几何优化步骤 对从STL加载的原始几何体进行缝合优化
    print("开始对原始STL几何体进行缝合优化...")
    healed_shape = heal_shape(raw_shape, tolerance=0.03)

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

    # 构建边缘线段几何体
    print("开始构建并优化边缘连接网络...")
    # 这里的 k 和 max_dist_factor 是关键参数，需要根据模型调整
    edge_indices = build_edge_graph(edge_points, k=2, max_dist_factor=2.0)
    edge_lines_compound = build_connected_edges(edge_points, edge_indices)

    # 合并几何体和边线
    print("正在合并几何体与边缘线...")
    final_compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(final_compound)

    # 添加优化后的主几何体
    if not healed_shape.IsNull():
        builder.Add(final_compound, healed_shape)

    # 添加优化后的边缘线集合
    if not edge_lines_compound.IsNull():
        builder.Add(final_compound, edge_lines_compound)

    # 导出
    success = export_step(shape=final_compound, output_path=OUTPUT_STEP_PATH, unit="MM", schema="AP214")
    print("STEP文件 导出成功" if success else "STEP文件 导出失败")

    # 边缘点可视化
    print("启动边缘点可视化窗口......")
    visualize_points(
        points=points,
        labels=edge_labels,
        title="边缘点预测结果",
        sample_size=3000,  # 降低显示点数，提高性能
        edge_color='red',
        non_edge_color='gray',
        save_path="vis_result.png",
        show=True
    )


if __name__ == '__main__':
    main()
