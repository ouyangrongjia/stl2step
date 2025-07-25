# edge_graph.py
import numpy as np
from scipy.spatial import KDTree


def build_edge_graph(points: np.ndarray, k: int = 5, max_dist_factor: float = 2.5) -> list:
    """
    构建边缘点连接图，并根据距离进行筛选。

    :param points: np.ndarray (N, 3)，边缘点坐标。
    :param k: 每个点连接k个邻居。
    :param max_dist_factor: 用于计算最大允许连接距离的因子。
    :return: List of (i, j) 元组，表示连接的点索引。
    """
    if len(points) < 2:
        return []

    tree = KDTree(points)

    # 计算一个合理的平均距离作为参考 查询每个点到其最近邻的距离，并取平均值
    distances, _ = tree.query(points, k=2)
    # distances[:, 1] 包含了每个点到其最近一个邻居的距离
    if distances.shape[1] > 1:
        mean_dist = np.mean(distances[:, 1])
        max_dist = mean_dist * max_dist_factor
        print(f"根据邻近点分析，设定最大连接距离为: {max_dist:.4f}")
    else:
        # 如果点太少，无法计算有意义的平均距离，则设置一个默认值或跳过
        max_dist = np.inf

    edges = set()  # 使用集合来自动处理重复边
    for i, pt in enumerate(points):
        # 查询 k+1 个最近的点（包含点本身）
        dists, idxs = tree.query(pt, k=k + 1)

        # 遍历邻居点（从索引1开始，跳过点本身）
        for j, dist in zip(idxs[1:], dists[1:]):
            # 距离筛选：只连接距离在合理范围内的点
            if dist < max_dist:
                # 保证边的方向唯一性 (i, j) 其中 i < j
                if i < j:
                    edges.add((i, j))
                else:
                    edges.add((j, i))

    print(f"边缘网络构建完成：从潜在的连接中筛选出 {len(edges)} 条有效边。")
    return list(edges)