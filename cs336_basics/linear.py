import math
import torch
import torch.nn as nn
import torch.nn.init as init
from einops import einsum

# Linear 只关心特征维度,in_features, out_features 为输入和输出的特征维度, 其他维度不关心
class Linear(nn.Module):
    def __init__(self,in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 定义权重矩阵，无偏置bias ( x * W^T )
        self.weight = nn.Parameter(torch.empty((out_features, in_features), 
                                               device=device, 
                                               dtype=dtype))
        
        # 初始化权重矩阵,截断正态分布 sigma = sqrt(2 / (in_features + out_features))
        # 截断在 [-3*sigma, 3*sigma] 之间
        sigma = math.sqrt(2 / (in_features + out_features))
        init.trunc_normal_(self.weight, mean=0.0, std=sigma, a=-3.0 * sigma, b=3.0 * sigma)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_features)
        # 输出: (..., out_features)
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
        