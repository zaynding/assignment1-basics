import torch
import torch.nn as nn
from cs336_basics.linear import Linear

"""
d_model: 输入特征维度
d_ff: 中间特征维度, 默认为 8/3 * d_model
SwiGLU 的计算公式为：
    gate = SiLU(W1 * x) = (W1 * x) * sigmoid(W1 * x)
    up = W3 * x
    output = W2 * (gate 逐元素相乘 up)
其中 W1, W2, W3 是可学习的线性变换矩阵
SwiGLU 的特点是使用了门控机制来控制信息流动，能够提高模型的表达能力和非线性特征。
"""
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None, device=None, dtype=None):
        super().__init__()
        # 1. 未指定 d_ff 时，按 8/3 比例计算并对齐为 64 的倍数
        if d_ff is None:
            d_ff = int(8 * d_model / 3)
            d_ff = 64 * ((d_ff + 63) // 64)

        self.d_model = d_model
        self.d_ff = d_ff

        # 2. 定义 3 个无偏置 Linear 模块
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)  # Gate
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)  # Down
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)  # Up

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 3. 计算 Gate 分支: SiLU(W1 * x) = (W1 * x) * sigmoid(W1 * x)
        gate = self.w1(x)
        gate = gate * torch.sigmoid(gate)

        # 4. 计算 Up 分支: W3 * x
        up = self.w3(x)

        # 5. 点乘门控融合，并通过 Down 投影还原为 d_model 维度
        return self.w2(gate * up)
    
    
