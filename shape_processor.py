# shape_processor.py
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Sewing
from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Core.ShapeUpgrade import ShapeUpgrade_UnifySameDomain


def heal_shape(shape: TopoDS_Shape, tolerance: float = 1e-3) -> TopoDS_Shape:
    """
    使用 BRepBuilderAPI_Sewing 尝试缝合几何体中的所有面。
    这能将离散的面片（来自STL）整合成一个或多个连续的壳（Shell），
    有助于优化拓扑结构并减小文件大小。

    :param shape: 原始的 TopoDS_Shape，通常是包含大量独立面片的 Compound。
    :param tolerance: 缝合容差，用于判断两个边是否可以被认为是重合的。
    :return: 缝合后的 TopoDS_Shape。
    """
    if shape.IsNull():
        print("警告：输入的待缝合形状为空。")
        return shape

    sewer = BRepBuilderAPI_Sewing(tolerance)
    sewer.Add(shape)
    sewer.Perform()

    # 获取缝合后的结果
    sewed_shape = sewer.SewedShape()

    if sewed_shape.IsNull():
        print("错误：缝合操作失败，返回了空的形状。将返回原始形状。")
        return shape

    print(f"几何体缝合完成。")

    print("正在尝试合并共面的面片以优化几何体...")
    unifier = ShapeUpgrade_UnifySameDomain(sewed_shape)
    unifier.Build()
    unified_shape = unifier.Shape()
    print("共面合并完成。")

    return unified_shape
