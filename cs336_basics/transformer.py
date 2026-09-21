import torch
import torch.nn as nn
import math
from einops import einsum,rearrange
from cs336_basics.rope import RotaryPositionalEmbedding
from cs336_basics.linear import Linear
from cs336_basics.swiglu import SwiGLU
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.embedding import Embedding

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    数值稳定的 Softmax 实现。
    为了防止指数计算溢出 (inf)，在求指数前先减去该维度的最大值。
    """
    # 1. 沿着指定的 dim 找到最大值。
    # 使用 keepdim=True 保证输出维度与 x 匹配，以便正确进行广播减法
    x_max = torch.max(x, dim=dim, keepdim=True).values
    
    # 2. 所有元素减去最大值（防止数值溢出）
    x_shifted = x - x_max
    
    # 3. 计算指数
    x_exp = torch.exp(x_shifted)
    
    # 4. 归一化：除以指数的总和，得到最终的概率分布
    return x_exp / torch.sum(x_exp, dim=dim, keepdim=True)

"""
    缩放点积注意力
    q: 查询张量，形状为 (..., sequence_length, d_k)
    k: 键张量，形状为 (..., sequence_length, d_k)
    v: 值张量，形状为 (..., sequence_length, d_v)
    mask: 可选的掩码张量，形状为 (..., sequence_length, sequence_length)，用于屏蔽不需要关注的部分
    返回: （..., sequence_length, d_v）形状的注意力输出张量
"""
def scaled_dot_product_attention(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor, 
    mask: torch.Tensor = None
) -> torch.Tensor:
    # 提取特征维度d_k，返回的是特征维度的长度。用于缩放
    d_k = q.size(-1)
    
    # 1. 计算Q K的点积，得到注意力分数，并缩放
    # i是查询序列长度，j是键序列长度，d_k是特征维度
    scores = einsum(q, k, " ... i d_k, ... j d_k -> ... i j") / math.sqrt(d_k)
    
    # 2. 如果提供了掩码，则将掩码应用到分数上
    if mask is not None:
        # 将掩码中为0的位置设置为负无穷大，以便在softmax中被忽略
        scores = scores.masked_fill(mask == 0, float("-inf"))
    
    # 3. 对分数应用softmax，得到注意力权重
    attention_weights = softmax(scores, dim=-1)
    
    # 4. 使用注意力权重对值张量进行加权求和，得到最终的注意力输出
    output = einsum(attention_weights, v, " ... i j, ... j d_v -> ... i d_v")
    
    return output


    """
    多头自注意力机制
    """
class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        max_seq_len: int = 2048, 
        rope_theta: float = 10000.0, 
        device=None, 
        dtype=None
    ):
        super().__init__()
        self.num_heads = num_heads
        
        # 头的特征维度 d_k = d_v = d_model / h
        self.d_k = d_model // num_heads
        
        # 四个无偏置的线性投影层
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        
        # 实例化 RoPE 模块
        self.rope = RotaryPositionalEmbedding(
            d_k=self.d_k, 
            max_seq_len=max_seq_len, 
            theta=rope_theta, 
            device=device
        )
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        seq_len = x.shape[-2]
        
        # 1. 线性投影并拆分多头
        q = rearrange(self.q_proj(x), "... s (h d) -> ... h s d", h=self.num_heads)
        k = rearrange(self.k_proj(x), "... s (h d) -> ... h s d", h=self.num_heads)
        v = rearrange(self.v_proj(x), "... s (h d) -> ... h s d", h=self.num_heads)
        
        # 2. 动态应用 RoPE：只有传入了 token_positions 才旋转
        if token_positions is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # 3. 构建因果掩码
        mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device)
        mask = torch.tril(mask)

        # 4. 传入掩码并计算注意力分数 
        attn_out = scaled_dot_product_attention(q, k, v, mask=mask)

        # 5. 拼接所有的注意力头，再经过最后的输出线性投影
        attn_out = rearrange(attn_out, "... h s d -> ... s (h d)")
        return self.output_proj(attn_out)
    
        """
        最小注意力块：包括归一化、多头自注意力和前馈网络
        """
class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int = 2048,
        rope_theta: float = 10000.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        # 与权重字典中的 ln1, attn, ln2, ffn 一一对应
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            device=device,
            dtype=dtype,
        )
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        # Pre-norm 残差结构
        if token_positions is None:
            token_positions = torch.arange(x.shape[-2], device=x.device)
        x = x + self.attn(self.ln1(x), token_positions=token_positions)
        x = x + self.ffn(self.ln2(x))
        return x
    

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff

        # 1. 词嵌入层 (参数键名: token_embeddings.weight, 形状: (vocab_size, d_model))
        self.token_embeddings = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
        )

        # 2. 堆叠 num_layers 个 TransformerBlock (参数键名前缀: layers.{i})
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                rope_theta=rope_theta,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ])

        # 3. 最终层归一化 (参数键名: ln_final.weight, 形状: (d_model,))
        self.ln_final = RMSNorm(d_model=d_model, device=device, dtype=dtype)

        # 4. 语言模型输出头 (参数键名: lm_head.weight, 形状: (vocab_size, d_model))
        self.lm_head = Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device,
            dtype=dtype,
        )

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        # in_indices: (batch_size, sequence_length)
        seq_len = in_indices.shape[-1]

        # 生成 RoPE 需要的位置序列索引 [0, 1, ..., seq_len - 1]
        token_positions = torch.arange(seq_len, device=in_indices.device)

        # 1. 词嵌入查找: (batch_size, sequence_length, d_model)
        h = self.token_embeddings(in_indices)

        # 2. 依次通过各层 TransformerBlock
        for layer in self.layers:
            h = layer(h, token_positions=token_positions)

        # 3. 最终层归一化
        h = self.ln_final(h)

        # 4. 投影到词表维度输出未归一化的 logits: (batch_size, sequence_length, vocab_size)
        logits = self.lm_head(h)
        return logits