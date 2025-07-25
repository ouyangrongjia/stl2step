# step_writer.py
"""
- 功能：将边缘点作为顶点标注到原始几何体上，形成复合体，然后导出为STEP文件。
"""
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.Interface import Interface_Static
from OCC.Core.TopTools import TopTools_ListOfShape
from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Shape
from OCC.Core.gp import gp_Pnt
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
from OCC.Core.BRep import BRep_Builder


def mark_edge_points_on_shape(base_shape: TopoDS_Shape, edge_points: list) -> TopoDS_Compound:
    """
    将边缘点作为顶点标注到原始几何体上，并返回复合体 Compound（shape + 点）
    """
    # 创建一个复合体（Compound），用于存放原始几何体和所有边缘点顶点
    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)
    # 添加原始几何体
    if not base_shape.IsNull():
        builder.Add(compound, base_shape)
    else:
        print("警告：原始形状为空")

    # 遍历每个边缘点
    for pt in edge_points:
        try:
            # 确保是3维坐标且为数值类型, 将点的三个坐标转换为浮点数
            x, y, z = map(float, pt[:3])  # 只取前三个维度并转为float
            # 创建点（gp_Pnt）
            pnt = gp_Pnt(x, y, z)
            # 创建顶点（TopoDS_Vertex）
            vertex = BRepBuilderAPI_MakeVertex(pnt).Vertex()
            # 将顶点添加到复合体
            builder.Add(compound, vertex)
        except (TypeError, IndexError) as e:
            print(f"跳过无效点坐标 {pt}: {str(e)}")
            continue

    return compound



def export_step(shape: TopoDS_Shape, output_path: str, unit: str = "MM", schema: str = "AP214") -> bool:
    """
    导出 STEP 文件，支持单位设置（MM/INCH）与 STEP 压缩标准（AP203/AP214）

    :param shape: TopoDS_Shape 或 TopoDS_Compound
    :param output_path: 输出文件路径，.step 或 .stp
    :param unit: 单位设置，"MM" 或 "INCH"
    :param schema: STEP 输出压缩标准，"AP203" 或 "AP214"
    :return: 是否成功
    """
    # === 设置导出参数 ===
    # 设置STEP导出的单位和模式
    unit = unit.upper()
    schema = schema.upper()

    if unit not in ["MM", "INCH"]:
        raise ValueError(f"单位设置错误：{unit}，必须是 'MM' 或 'INCH'")

    if schema not in ["AP203", "AP214"]:
        raise ValueError(f"STEP 标准错误：{schema}，必须是 'AP203' 或 'AP214'")

    # 设置导出单位与压缩格式
    Interface_Static.SetCVal("xstep.cascade.unit", unit)
    Interface_Static.SetCVal("write.step.schema", schema)

    # 可选精度压缩优化（默认启用）
    Interface_Static.SetIVal("write.step.occ.precision.mode", 1)
    Interface_Static.SetRVal("write.precision.val", 0.001)

    writer = STEPControl_Writer()
    status = writer.Transfer(shape, STEPControl_AsIs)

    if status != IFSelect_RetDone:
        print("STEP 写入失败：模型传输失败")
        return False

    write_status = writer.Write(output_path)
    if write_status == IFSelect_RetDone:
        print(f"STEP 文件导出成功：{output_path}")
        return True
    else:
        print("STEP 文件导出失败")
        return False