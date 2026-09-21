from collections.abc import Iterable
import torch

def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter], 
    max_l2_norm: float, 
    eps: float = 1e-6
) -> None:
    """
    计算所有参数梯度的全局 L2 范数，并在超过 max_norm 时就地 (in-place) 进行缩放。

    Args:
        parameters: 可迭代的 Parameter 对象集合
        max_norm: 允许的最大全局 L2 范数 M
        eps: 防止除零的小常数，默认 1e-6
    """
    # 1. 过滤出存在梯度的有效参数
    params_with_grad = [p for p in parameters if p.grad is not None]
    if len(params_with_grad) == 0:
        return

    # 2. 累计所有参数梯度的平方和，防止不同设备/类型影响，统一累加
    total_norm_sq = 0.0
    for p in params_with_grad:
        param_norm_sq = torch.sum(p.grad.detach() ** 2)
        total_norm_sq += param_norm_sq.item()

    # 3. 开根号求得全局 L2 范数
    total_norm = total_norm_sq ** 0.5

    # 4. 如果全局范数超过了上限，执行就地缩放
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for p in params_with_grad:
            # 必须就地修改 (in-place)
            p.grad.detach().mul_(scale)