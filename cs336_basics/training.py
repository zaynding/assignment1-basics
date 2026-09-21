import argparse
import math
import os
import time
import numpy as np
import torch

from cs336_basics.checkpointing import save_checkpoint
from cs336_basics.data import get_batch
from cs336_basics.decoding import decode
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.gradient_clipping import gradient_clipping
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule
from cs336_basics.transformer import TransformerLM


def parse_args():
    parser = argparse.ArgumentParser(description="CS336 Transformer LM 预训练脚本")
    # 路径与环境配置
    parser.add_argument("--train_data_path", type=str, default="data/TinyStories-train.npy", help="训练集 .bin 或 .npy 路径")
    parser.add_argument("--val_data_path", type=str, default="data/TinyStories-valid.npy", help="验证集 .bin 或 .npy 路径")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="检查点存放路径")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    
    # 讲义 7.2.1 推荐的 TinyStories 基准架构参数
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    # 训练循环与超参数
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_iters", type=int, default=5000, help="总训练迭代步数")
    parser.add_argument("--max_lr", type=float, default=5e-4)
    parser.add_argument("--min_lr", type=float, default=5e-5)
    parser.add_argument("--warmup_iters", type=int, default=500)
    parser.add_argument("--max_norm", type=float, default=1.0, help="梯度裁剪上限")
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)

    # 监控与评估
    parser.add_argument("--eval_interval", type=int, default=200, help="多少步评估一次验证集")
    parser.add_argument("--eval_iters", type=int, default=20, help="评估时抽取的批次数")
    parser.add_argument("--save_interval", type=int, default=1000, help="保存检查点的步数间隔")
    parser.add_argument("--log_interval", type=int, default=20, help="打印日志步数间隔")
    return parser.parse_args()


@torch.no_grad()
def evaluate_loss(model, data, batch_size, context_length, device, eval_iters):
    model.eval()
    losses = []
    for _ in range(eval_iters):
        inputs, targets = get_batch(data, batch_size, context_length, device)
        logits = model(inputs)
        loss = cross_entropy(logits, targets)
        losses.append(loss.item())
    model.train()
    mean_loss = float(np.mean(losses))
    perplexity = math.exp(mean_loss)
    return mean_loss, perplexity


def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = torch.device(args.device)
    print(f" 运行设备: {device}")

    # 1. 使用 np.memmap 零拷贝加载词元数据 (假设数据存储为 uint16)
    train_data = np.load(args.train_data_path, mmap_mode="r")
    val_data = np.load(args.val_data_path, mmap_mode="r")
    print(f" 训练集 Token 数: {len(train_data):,}, 验证集 Token 数: {len(val_data):,}")

    # 2. 实例化模型
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f" 模型总参数量: {total_params:,}")

    # 3. 实例化 AdamW 优化器
    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    # 4. 训练主循环
    model.train()
    start_time = time.time()

    for it in range(1, args.max_iters + 1):
        # 4.1 计算并动态设置当前步的学习率
        lr = get_lr_cosine_schedule(
            it=it,
            max_learning_rate=args.max_lr,
            min_learning_rate=args.min_lr,
            warmup_iters=args.warmup_iters,
            cosine_cycle_iters=args.max_iters,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # 4.2 采样数据、前向计算与反向传播
        inputs, targets = get_batch(train_data, args.batch_size, args.context_length, device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = cross_entropy(logits, targets)
        loss.backward()

        # 4.3 梯度裁剪并更新权重
        gradient_clipping(model.parameters(), max_l2_norm=args.max_norm)
        optimizer.step()

        # 4.4 周期性打印训练日志
        if it % args.log_interval == 0:
            elapsed = time.time() - start_time
            print(f"Step {it:5d}/{args.max_iters} | Train Loss: {loss.item():.4f} | LR: {lr:.6f} | Elapsed: {elapsed:.1f}s")

        # 4.5 周期性评估与生成演示
        if it % args.eval_interval == 0 or it == args.max_iters:
            val_loss, val_ppl = evaluate_loss(
                model, val_data, args.batch_size, args.context_length, device, args.eval_iters
            )
            print(f"\n [Validation Step {it}] Val Loss: {val_loss:.4f} | Val Perplexity: {val_ppl:.2f}")

            # 简易文本解码生成采样 (验证生成连贯性)
            sample_prompt = [train_data[0]]  # 以训练集第一个 token 作为 prompt
            gen_tokens = decode(
                model, prompt_tokens=sample_prompt, max_new_tokens=30, temperature=0.8, top_p=0.9
            )
            print(f" [Generated Sample (Tokens)]: {gen_tokens[:15]}...\n")

        # 4.6 周期性保存断点
        if it % args.save_interval == 0 or it == args.max_iters:
            ckpt_path = os.path.join(args.checkpoint_dir, f"checkpoint_step_{it}.pt")
            save_checkpoint(model, optimizer, it, ckpt_path)
            print(f"💾 检查点已保存: {ckpt_path}")

    print(" 预训练完成！")


if __name__ == "__main__":
    main()