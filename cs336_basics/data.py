import numpy as np
import torch

"""
    输入序列 (Inputs)：取连续的 m 个 token，即 x[i : i + m]。   
    目标序列 (Targets)：取向后偏移一位的 m 个 token，即 x[i + 1 : i + m + 1]。
    模型在第 k 步的任务就是预测目标序列中对应的下一个 token。
"""
def get_batch(
    x: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    从一维 token 数组中随机采样 batch_size 个序列，返回输入和对应的预测目标。
    
    Args:
        x: 存储一维 token ID 的 numpy 数组
        batch_size: 批大小
        context_length: 上下文长度 (m)
        device: 数据放置的目标设备 (如 'cpu', 'cuda:0', 'mps')
        
    Returns:
        inputs: 形状为 (batch_size, context_length) 的整数张量
        targets: 形状为 (batch_size, context_length) 的整数张量
    """
    # 1. 计算合法的起始索引上限 (保证 i + context_length <= len(x) - 1)
    max_start_idx = len(x) - context_length
    
    # 2. 随机采样 batch_size 个起始位置
    start_indices = np.random.randint(0, max_start_idx, size=batch_size)
    
    # 3. 构造 inputs 和 targets 切片
    inputs_np = np.stack([x[i : i + context_length] for i in start_indices])
    targets_np = np.stack([x[i + 1 : i + context_length + 1] for i in start_indices])
    
    # 4. 转为 PyTorch Tensor，设置 dtype=torch.long 并移至指定设备
    inputs = torch.tensor(inputs_np, dtype=torch.long, device=device)
    targets = torch.tensor(targets_np, dtype=torch.long, device=device)
    
    return inputs, targets