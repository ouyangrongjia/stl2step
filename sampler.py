# sampler.py
import numpy as np
import open3d as o3d


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """
    使用 Open3D 对点云进行体素下采样，以实现点云密度的均匀化。

    :param points: 输入的 Nx3 numpy 点云数组。
    :param voxel_size: 体素网格的大小。这个值决定了采样后的点云密度，
                       值越小，保留的点越多，细节更丰富；
                       值越大，点云越稀疏，计算速度越快。
    :return: 经过体素下采样后的点云数组。
    """
    # 将numpy数组转换为Open3D的PointCloud对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 执行核心的体素下采样算法
    downsampled_pcd = pcd.voxel_down_sample(voxel_size)

    print(f"体素下采样：从 {len(points)} 个点 -> {len(downsampled_pcd.points)} 个点 (体素大小: {voxel_size})")

    # 将Open3D的结果转换回numpy数组并返回
    return np.asarray(downsampled_pcd.points)


def sample_points(points: np.ndarray,
                  voxel_size: float,
                  max_points: int) -> np.ndarray:
    """
    对点云进行两步采样：首先是体素下采样，然后是随机下采样。
    这是一个健壮的组合，既能保留模型的整体结构，又能确保点数不超过上限。

    :param points: 原始点云数组。
    :param voxel_size: 体素下采样的体素大小。
    :param max_points: 最终允许的最大点数。
    :return: 最终采样完成的点云数组。
    """
    # 体素下采样，使点云分布均匀
    sampled_points = voxel_downsample(points, voxel_size)

    # 检查点数是否仍然超限，如果超限则进行随机采样作为补充
    if len(sampled_points) > max_points:
        print(f"体素采样后点数 ({len(sampled_points)}) 仍超限，将随机采样至 {max_points} 点。")
        rng = np.random.default_rng()
        sampled_indices = rng.choice(len(sampled_points), size=max_points, replace=False)
        sampled_points = sampled_points[sampled_indices]

    return sampled_points
