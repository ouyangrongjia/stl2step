# STL to STEP Converter with Edge Detection (via PointNet++)

本项目旨在实现一个工业级 STL 模型转换器，支持将三角网格格式（.stl）转换为包含边缘点标注的 STEP 格式（.step），并结合深度学习模型（PointNet++）自动识别几何边缘。

## 📌 项目亮点

- ✅ 支持 STL（ASCII/Binary） → STEP（AP203/AP214）
- ✅ 结合 PointNet++ 进行边缘点识别
- ✅ 利用 OpenCascade 生成几何体与边缘线段（实现中）
- ✅ 支持边缘点可视化（3D 图）
- ✅ 支持批量处理、大文件解析（1GB STL）（实现中）

---

## 📂 项目结构
.
├── main.py # 主入口，执行 STL → STEP 全流程
├── config.py # 参数配置（路径、单位、输出等）
├── stl_reader.py # STL 文件加载与几何体解析
├── pointnet_infer.py # PointNet++ 推理模块（边缘点分类）
├── step_writer.py # STEP 文件生成与边缘点写入
├── visualize_edges.py # 3D 可视化边缘点
├── Pointnet_Pointnet2_pytorch/
│ ├── pointnet2_sem_seg_msg.py # PointNet++ 语义分割模型
│ └── pointnet2_utils.py # PointNet++ 网络组件
├── requirements.txt # 依赖文件


---

## 🚀 使用方法

### 1. 安装依赖

建议使用虚拟环境：

```bash
pip install -r requirements.txt

### 2. 配置路径（config.py）
STL_PATH = "data/your_model.stl"
OUTPUT_STEP_PATH = "output/your_model_with_edges.step"
MODEL_PATH = "weights/pointnet2_edge_model.pth"

### 3. 运行主程序
python main.py

