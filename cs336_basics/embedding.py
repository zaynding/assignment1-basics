import torch
import torch.nn as nn
import torch.nn.init as init
from einops import einsum

# 本类是 transformer 的第一层，负责将输入的 token id 映射为 embedding 向量
# embedding向量包含了 token 的语义信息, 维度为 embedding_dim
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # 1. 构造词向量查找表权重矩阵
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), 
                                               device=device, 
                                               dtype=dtype))
        
        # 2. 初始化权重矩阵,截断正态分布 std = 1, 截断在 [-3, 3] 之间
        init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # 3. 根据 token id 查找对应的 embedding 向量，tokenids是多少，就取第几行的向量
        # PyTorch 的高级索引机制:当用整数张量索引一个矩阵时，输出的形状就是输入索引张量的形状拼接上被索引对象的末尾维度。
        return self.weight[token_ids]  
    