import torch
import torch.nn as nn


def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    """
    对概率分布进行 Top-p (Nucleus) 截断并采样。
    
    Args:
        probs: 形状为 (..., vocab_size) 的归一化概率分布
        p: 累积概率阈值 (0.0 < p <= 1.0)
    Returns:
        采样出的 token ID (整数标量或张量)
    """
    if p >= 1.0:
        return torch.multinomial(probs, num_samples=1)

    # 1. 降序排序
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    
    # 2. 计算累积概率
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # 3. 构造掩码: 剔除累积和超过 p 的尾部 token
    # 向右平移 1 位以确保第一个超过 p 的 token 依然被包含在核集合中
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    # 4. 将被剔除的位置概率置为 0，并重新归一化
    sorted_probs = sorted_probs.masked_fill(sorted_indices_to_remove, 0.0)
    sorted_probs = sorted_probs / torch.sum(sorted_probs, dim=-1, keepdim=True)

    # 5. 从截断后的分布中采样索引
    sampled_sorted_idx = torch.multinomial(sorted_probs, num_samples=1)

    # 6. 还原回词表中的原始 token ID
    next_token = torch.gather(sorted_indices, dim=-1, index=sampled_sorted_idx)
    return next_token


@torch.no_grad()
def decode(
    model: nn.Module,
    prompt_tokens: list[int] | torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
) -> list[int]:
    """
    自回归文本生成解码函数。
    
    Args:
        model: 训练好的 TransformerLM
        prompt_tokens: 初始输入的提示词 Token ID 列表
        max_new_tokens: 最多生成的新 token 数量
        temperature: 温度系数 (tau)
        top_p: Nucleus 采样阈值 (p)
        eos_token_id: 终止符号 ID (如 <|endoftext|>)，若生成该 ID 则提前结束
    """
    model.eval()

    if isinstance(prompt_tokens, list):
        generated = torch.tensor([prompt_tokens], dtype=torch.long)
    else:
        generated = prompt_tokens.clone()

    device = next(model.parameters()).device
    generated = generated.to(device)

    for _ in range(max_new_tokens):
        # 如果当前序列长度超过了模型的最大上下文长度，则截取最后 context_length 个 token
        if hasattr(model, "context_length") and generated.shape[1] > model.context_length:
            input_tokens = generated[:, -model.context_length :]
        else:
            input_tokens = generated

        # 1. 前向传播获取 logits: (1, seq_len, vocab_size)
        logits = model(input_tokens)

        # 2. 提取最后一个位置的 logits: (1, vocab_size)
        last_logits = logits[:, -1, :]

        # 3. 应用温度系数缩放
        if temperature > 0.0:
            last_logits = last_logits / temperature
            probs = torch.softmax(last_logits, dim=-1)
            # 4. Top-p 采样
            next_token = sample_top_p(probs, top_p)
        else:
            # temperature=0 时退化为贪婪解码
            next_token = torch.argmax(last_logits, dim=-1, keepdim=True)

        # 5. 拼接到已生成的序列后
        generated = torch.cat([generated, next_token], dim=1)

        # 6. 若遇到终止符则提前结束生成
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

    return generated[0].tolist()