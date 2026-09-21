import torch

def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    数值稳定的交叉熵损失函数 (Log-Sum-Exp 技巧)。
    
    Args:
        inputs: 形状为 (..., vocab_size) 的预测 logits
        targets: 形状为 (...) 的真实 token ID 标签
    Returns:
        标量张量 (所有 token 损失的平均值)
    """
    # 1. 取出最后一维 (vocab_size) 的最大值，保持维度用于广播
    m = torch.max(inputs, dim=-1, keepdim=True).values
    
    # 2. 减去最大值防溢出
    shifted_inputs = inputs - m
    
    # 3. 计算 log-sum-exp 部分: m + log(sum(exp(shifted_inputs)))
    # 注意: shifted_inputs.exp().sum(dim=-1, keepdim=True) 避免了数值上溢
    lse = torch.log(torch.sum(torch.exp(shifted_inputs), dim=-1, keepdim=True)) + m
    
    # 4. 提取真实标签 y 对应的 logits: inputs[..., y]
    # targets 需要 unsqueeze(-1) 以便与 inputs 最后一维对齐提取
    targets_expanded = targets.unsqueeze(-1)
    target_logits = torch.gather(inputs, dim=-1, index=targets_expanded)
    
    # 5. 单个样本位置的 loss: - target_logits + lse
    loss = -target_logits + lse
    
    # 6. 对所有样本和序列位置求平均值并返回标量
    return loss.mean()