# 热图绘制只需要 MambaMIL。原 builder 依赖 MUSK/timm 等训练期重型库，
# 这里不导入，避免运行热图时被迫加载无关的 encoder 权重。
from .MambaMIL import MambaMIL