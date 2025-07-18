# STL文件读取器
"""
- 功能：读取STL文件，返回两个部分：
    OCC几何体（用于后续的STEP导出）
    点云数据（numpy数组，用于PointNet模型预测）
"""
from stl import mesh
from OCC.Extend.DataExchange import read_stl_file
import numpy as np

def load_stl(stl_path):
    # 使用OCC读取STL文件，得到TopoDS_Shape几何体
    occ_shape = read_stl_file(stl_path)
    # 使用numpy-stl库读取STL文件，得到点云数据（每个三角形三个顶点的坐标）
    stl_mesh = mesh.Mesh.from_file(stl_path)
    # 将三角形网格展开为点云（每个三角形三个顶点，所以点云数量是三角形数量的3倍）
    points = stl_mesh.vectors.reshape(-1, 3)
    return occ_shape, points
