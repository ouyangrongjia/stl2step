# mesh_processor.py (已添加法线计算)
import open3d as o3d
from stl import mesh as np_mesh
import numpy as np
import os


def simplify_stl(input_path: str, output_path: str, target_reduction_factor: float = 0.5):
    """
    使用 Open3D 简化 STL 网格模型。
    更新：增加了在保存前重新计算法线的步骤。
    """
    if not (0 < target_reduction_factor <= 1.0):
        raise ValueError("target_reduction_factor 必须在 (0, 1] 范围内。")

    print(f"正在使用兼容模式加载网格: {input_path}")

    try:
        m = np_mesh.Mesh.from_file(input_path)
    except Exception as e:
        print(f"错误: 使用 numpy-stl 加载 '{input_path}' 失败: {e}")
        return

    vertices = m.vectors.reshape(-1, 3)
    triangles = np.arange(len(vertices)).reshape(-1, 3)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    mesh.remove_duplicated_vertices()

    initial_triangles = len(mesh.triangles)
    if initial_triangles == 0:
        print("警告：输入网格没有三角面片，无法简化。")
        o3d.io.write_triangle_mesh(output_path, mesh)
        return

    print(f"加载成功，原始三角面片数量: {initial_triangles}")

    target_triangles = int(initial_triangles * target_reduction_factor)
    print(f"目标三角面片数量: {target_triangles} (简化率: {target_reduction_factor})")

    simplified_mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangles)

    print("正在清理和修复简化后的网格...")
    simplified_mesh.remove_degenerate_triangles()
    simplified_mesh.remove_duplicated_triangles()
    simplified_mesh.remove_duplicated_vertices()
    simplified_mesh.remove_non_manifold_edges()
    print("网格清理完成。")

    # ==================== 新增的法线计算步骤 ====================
    print("正在为修复后的网格计算法线...")
    simplified_mesh.compute_vertex_normals()
    print("法线计算完成。")
    # ========================================================

    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 现在写入STL文件将会成功
    o3d.io.write_triangle_mesh(output_path, simplified_mesh)
    print(f"修复并简化后的网格已保存到: {output_path}")