# 边缘点可视化
"""
- 功能：使用matplotlib可视化点云，并高亮显示边缘点。
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from stl import mesh
# 设置环境变量，避免OpenMP冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# 加载 STL 点云
def load_stl_pointcloud(stl_path):
    stl_mesh = mesh.Mesh.from_file(stl_path)
    points = stl_mesh.vectors.reshape(-1, 3)
    return points

# 可视化点云 + 边缘标签（1 为边缘点）
def visualize_points(points, labels=None, title="边缘点可视化", sample_size=1024):
    # 创建3D图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 如果没有提供标签，则全部视为非边缘点
    if labels is None:
        labels = np.zeros(len(points))  # 默认全部非边缘

    # 下采样展示数量如果点太多，则下采样
    if len(points) > sample_size:
        idx = np.random.choice(len(points), sample_size, replace=False)
        points = points[idx]
        labels = np.array(labels)[idx]

    # 根据标签创建颜色和大小数组
    # 使用矢量化操作，比循环更快
    colors = np.where(labels == 1, 'red', 'lightgray')
    sizes = np.where(labels == 1, 4, 1)

    # 一次性绘制所有点
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c=colors,
               s=sizes)

    # 设置坐标轴标签和标题
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)

    # 手动创建图例
    # 因为只调用了一次scatter，需要手动创建图例项
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='边缘点',
               markerfacecolor='red', markersize=5),
        Line2D([0], [0], marker='o', color='w', label='非边缘点',
               markerfacecolor='lightgray', markersize=3)
    ]
    ax.legend(handles=legend_elements)

    plt.tight_layout()
    plt.savefig("out.png")
    plt.show()
    # # 分组绘制 将点分为边缘点和非边缘点
    # edge_pts = points[labels == 1]
    # non_edge_pts = points[labels == 0]
    # # 绘制非边缘点（浅灰色，小点）
    # ax.scatter(non_edge_pts[:, 0], non_edge_pts[:, 1], non_edge_pts[:, 2],
    #            c='lightgray', s=1, label='非边缘点')
    # # 绘制边缘点（红色，稍大的点）
    # ax.scatter(edge_pts[:, 0], edge_pts[:, 1], edge_pts[:, 2],
    #            c='red', s=4, label='边缘点')
    # # 设置坐标轴标签和标题
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_zlabel("Z")
    # ax.set_title(title)
    # ax.legend() # 显示图例
    # plt.tight_layout()
    # plt.show()
    # plt.savefig("out.png")