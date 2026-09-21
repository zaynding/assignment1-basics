import torch
import torch.nn as nn

import torch
import torch.nn as nn

"""
输入 x：(batch_size, sequence_length, d_model),每一个 Token 都有一个 d_model 维的向量表示
RMSNorm 的计算公式为：
    x_norm = x / sqrt(mean(x^2) + eps)
    output = g * x_norm
其中 g 是可学习的增益参数，一维，形状为 (d_model,)，即每一个特征维度都有一个独立的增益参数。
特征通道的语义定义是全局统一的，所以不同 Token 的相同特征维度共用一个 g 权重。
RMSNorm 的特点是只对最后一个维度进行归一化，适用于 Transformer 模型中的注意力机制和前馈网络。
"""
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        # 1. 增益参数 g 初始化为全 1，形状为 (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        # 2. 上采样至 float32 防止计算平方均值时数值溢出
        x_fp32 = x.to(torch.float32)

        # 3. 计算均方根倒数: 1 / sqrt(mean(x^2) + eps)
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        rsqrt = torch.rsqrt(variance + self.eps)
        # 4. 对输入 x 进行归一化
        x_norm = x_fp32 * rsqrt

        # 5. 乘上可学习参数 g（广播机制的“从右向左对齐”），并下采样还原回输入的数据类型
        return (x_norm * self.weight).to(input_dtype)
    