# wire_builder.py
import numpy as np
from OCC.Core.gp import gp_Pnt
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
from OCC.Core.BRep import BRep_Builder
from OCC.Core.TopoDS import TopoDS_Compound

def build_connected_edges(points: np.ndarray, edges: list) -> TopoDS_Compound:
    """
    将边缘点连线构建为TopoDS_Edge集合，返回为Compound（用于STEP导出）
    """
    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)

    for i, j in edges:
        pt1 = gp_Pnt(*map(float, points[i]))
        pt2 = gp_Pnt(*map(float, points[j]))
        edge = BRepBuilderAPI_MakeEdge(pt1, pt2).Edge()
        builder.Add(compound, edge)

    return compound
