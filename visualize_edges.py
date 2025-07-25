# visualize_edges.py
import os
import numpy as np
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def visualize_points(points,
                     labels=None,
                     title="模型轮廓与识别出的边缘点",
                     sample_size=5000,  # 此参数在新逻辑中被忽略，但保留以兼容旧调用
                     edge_color='red',
                     non_edge_color='lightgray',
                     edge_size=15,
                     non_edge_size=2,
                     show=True,
                     save_path="vis_result.png"):
    """
    可视化点云及边缘点识别结果。
    绘制完整的模型轮廓作为背景，然后高亮显示识别出的边缘点。
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    if labels is None:
        labels = np.zeros(len(points), dtype=bool)

    # 绘制完整的输入点云(points_for_model)作为背景轮廓
    if len(points) > 0:
        print(f"可视化: 正在绘制 {len(points)} 个点作为完整模型背景...")
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                   c=non_edge_color,
                   s=non_edge_size,
                   label='完整模型轮廓')

    # 提取边缘点
    edge_pts = points[labels == 1]

    # 边缘点覆盖在背景之上，用不同颜色和大小高亮显示
    if len(edge_pts) > 0:
        print(f"可视化: 正在高亮显示 {len(edge_pts)} 个识别出的边缘点...")
        ax.scatter(edge_pts[:, 0], edge_pts[:, 1], edge_pts[:, 2],
                   c=edge_color,
                   s=edge_size,
                   label=f'识别出的边缘点 ({len(edge_pts)}个)',
                   depthshade=False)  # 让红点更突出，不受深度影响而变暗

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.legend()
    ax.set_aspect('auto')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"可视化图像保存成功：{save_path}")

    if show:
        plt.show()
    plt.close(fig)
