import math
from collections.abc import Callable
import torch
from torch.optim import Optimizer

# weight_decay: 权重衰减系数 (lambda),就是 regularization 中的惩罚项的强度系数
class AdamW(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    # self.step只是一个更新的过程，已经做完了前向传播算出结果、loss 函数，和反向传播算好了每一步的梯度，现在只需要进行梯度下降
    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")
                # 每一个p都有一个对应的状态字典，里面存储了t、m、v等信息
                state = self.state[p]

                # 1. 状态初始化：步数 t、一阶动量 m、二阶动量 v
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                m = state["m"]
                v = state["v"]

                # 2. 迭代步数递增 (讲义要求 t 从 1 开始)
                state["t"] += 1
                t = state["t"]

                # 3. 计算步数 t 校正后的步长 alpha_t，防止第一次爆炸。这里做了一个近似
                alpha_t = lr * math.sqrt(1.0 - beta2 ** t) / (1.0 - beta1 ** t)

                # 4. 执行解耦权重衰减 (讲义公式 Line 8: theta <- theta - alpha * lambda * theta)
                if weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)

                # 5. 更新一阶矩估计 m (讲义公式 Line 9: m <- beta1 * m + (1 - beta1) * g)
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                # 6. 更新二阶矩估计 v (讲义公式 Line 10: v <- beta2 * v + (1 - beta2) * g^2)
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # 7. 应用动量更新权重 (讲义公式 Line 11: theta <- theta - alpha_t * m / (sqrt(v) + eps))
                denom = torch.sqrt(v).add_(eps)
                p.data.addcdiv_(m, denom, value=-alpha_t)

        return loss
    
    

def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    计算第 it 步时的学习率 (带预热的余弦衰减)。
    
    Args:
        it: 当前迭代步数 t
        max_learning_rate: 最大学习率 alpha_max
        min_learning_rate: 最小学习率 alpha_min
        warmup_iters: 预热总步数 T_w
        cosine_cycle_iters: 余弦衰减截止步数 T_c
    """
    # 1. 预热阶段: t < T_w
    if it < warmup_iters:
        return (it / warmup_iters) * max_learning_rate

    # 2. 退火后阶段: t > T_c
    if it > cosine_cycle_iters:
        return min_learning_rate

    # 3. 余弦衰减阶段: T_w <= t <= T_c
    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    cosine_decay = 0.5 * (1.0 + math.cos(progress * math.pi))
    return min_learning_rate + cosine_decay * (max_learning_rate - min_learning_rate)