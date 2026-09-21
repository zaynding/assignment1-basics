import torch
import torch.nn as nn
from einops import rearrange
"""
RotaryPositionalEmbedding 是一种旋转位置编码方法 ，用于在 Transformer 模型中引入位置信息。
它通过对输入的特征向量进行二维旋转变换，使得模型能够感知序列中 token 的相对位置关系。
它完全根据 dk 的维度大小和 Token 的个数看怎么旋转 底频是多少
把原输入乘 cos 大矩阵，把相邻奇数和偶数（比如（0，1），（2，3））两两正交对应的矩阵乘 sin 构成的大矩阵
 特征维度 Index：决定“角速度 or 频率”
 Token 序列 Index：充当时间t,固定的角速度旋转了多少次
参数:
    d_k: 输入特征维度，必须为偶数，因为每两个相邻的特征维度组成一个二维旋转平面。
    max_seq_len: 序列的最大长度，用于预计算旋转角度。
    theta: 控制旋转频率的超参数，默认值为 10000.0。
计算公式:
    对于每个二维旋转平面，计算底频:
        theta_k = theta^(-2k / d_k), k = 0, 1, ..., d_k/2 - 1
    对于每个 token 的位置 pos，计算旋转角度:
        angle(pos, k) = pos * theta_k
    然后将输入特征向量 x 的每两个相邻维度 (x_1, x_2) 进行旋转变换:
        y_1 = x_1 * cos(angle) - x_2 * sin(angle)
        y_2 = x_1 * sin(angle) + x_2 * cos(angle)
    最终输出的特征向量 y 与输入 x 具有相同的形状。  
"""
class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        d_k: int,
        max_seq_len: int,
        theta: float = 10000.0,
        device=None,
    ):
        super().__init__()
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.theta = theta

        # 1. 计算每个二维旋转平面的底频: theta_k = theta^(-2k / d_k), k = 0, 1, ..., d_k/2 - 1
        freq_indices = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)
        inv_freq = 1.0 / (theta ** (freq_indices / d_k))  # 形状: (d_k // 2,)

        # 2. 生成序列所有位置的索引: [0, 1, ..., max_seq_len - 1]
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)

        # 3. 外积生成所有位置的角度矩阵: (max_seq_len, d_k // 2)
        angles = torch.outer(positions, inv_freq)

        # 4. 预计算 cos 和 sin，并注册为不参与持久化保存的 Buffer
        self.register_buffer("cos_cached", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cached", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # 输入 x 形状: (..., seq_len, d_k)
        # token_positions 形状: (..., seq_len) 或 (seq_len,)
        input_dtype = x.dtype

        # 1. 根据 token_positions 索引切片出当前 token 对应的 cos 与 sin
        # 切片后形状: (..., seq_len, d_k // 2)
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        # 2. 奇偶相邻元素两两分组: 拆分为 (..., seq_len, d_k // 2, 2)
        x_paired = rearrange(x.float(), "... (d two) -> ... d two", two=2)
        x0 = x_paired[..., 0]  # 偶数索引分量 x_1
        x1 = x_paired[..., 1]  # 奇数索引分量 x_2

        # 若包含注意力头维度 (如 x 是 (b, h, s, d) 而 cos 是 (b, s, d))，自动对齐维度
        if cos.ndim < x0.ndim and cos.ndim == 3 and x0.ndim == 4:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

        # 3. 执行 2D 旋转公式
        y0 = x0 * cos - x1 * sin
        y1 = x0 * sin + x1 * cos

        # 4. 重新组合回原本形状并还原数据类型
        y_paired = torch.stack([y0, y1], dim=-1)
        y = rearrange(y_paired, "... d two -> ... (d two)")
        return y.to(input_dtype)
