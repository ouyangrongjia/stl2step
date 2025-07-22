import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def visualize_points(points,
                     labels=None,
                     title="边缘点可视化",
                     sample_size=2048,
                     edge_color='red',
                     non_edge_color='lightgray',
                     edge_size=4,
                     non_edge_size=1,
                     show=True,
                     save_path="out.png"):
    """
    可视化点云及边缘点识别结果

    参数说明：
    - points: (N, 3) 点云坐标
    - labels: (N,) bool 或 int，1为边缘点，0为非边缘点
    - sample_size: 限制最大显示点数（加快渲染）
    - edge_color: 边缘点颜色
    - non_edge_color: 非边缘点颜色
    - edge_size: 边缘点大小
    - non_edge_size: 非边缘点大小
    - show: 是否显示图像
    - save_path: 保存图片路径（默认 out.png）
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if labels is None:
        labels = np.zeros(len(points))  # 默认全为非边缘

    # 下采样，加速可视化
    if len(points) > sample_size:
        idx = np.random.choice(len(points), sample_size, replace=False)
        points = points[idx]
        labels = np.array(labels)[idx]

    # 分组绘制
    edge_pts = points[labels == 1]
    non_edge_pts = points[labels == 0]

    if len(non_edge_pts) > 0:
        ax.scatter(non_edge_pts[:, 0], non_edge_pts[:, 1], non_edge_pts[:, 2],
                   c=non_edge_color, s=non_edge_size, label='非边缘点')

    if len(edge_pts) > 0:
        ax.scatter(edge_pts[:, 0], edge_pts[:, 1], edge_pts[:, 2],
                   c=edge_color, s=edge_size, label='边缘点')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    # 保存图像
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"可视化图像保存成功：{save_path}")

    if show:
        plt.show()

    plt.close(fig)
