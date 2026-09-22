import argparse
import glob
import math
import os
import re
import matplotlib
# 设置为非交互式后端，避免无 GUI 环境报错
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from cs336_basics.data import get_batch
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.transformer import TransformerLM


def parse_args():
    parser = argparse.ArgumentParser(description="扫描并评测所有 Checkpoint，绘制验证集损失与困惑度曲线")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="检查点目录")
    parser.add_argument("--val_data_path", type=str, default="./data/TinyStories-valid.npy", help="验证集 .npy 路径")
    parser.add_argument("--output_image", type=str, default="./checkpoints/val_loss_curve.png", help="输出折线图路径")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_iters", type=int, default=30, help="每个检查点评估抽取的 Batch 数")
    parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

    # 模型架构配置（与 10,000 词表 TinyStories 基准完全一致）
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    return parser.parse_args()


def extract_step_number(filepath: str) -> int:
    """从文件名（如 checkpoint_step_1000.pt）中精准提取数字步数"""
    numbers = re.findall(r"\d+", os.path.basename(filepath))
    return int(numbers[-1]) if numbers else 0


@torch.no_grad()
def evaluate_checkpoint(model, val_data, batch_size, context_length, device, eval_iters):
    model.eval()
    losses = []
    for _ in range(eval_iters):
        inputs, targets = get_batch(val_data, batch_size, context_length, device)
        logits = model(inputs)
        loss = cross_entropy(logits, targets)
        losses.append(loss.item())
    
    mean_loss = float(np.mean(losses))
    perplexity = math.exp(mean_loss)
    return mean_loss, perplexity


def plot_metrics(steps, val_losses, perplexities, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 1. 绘制 Validation Loss
    ax1.plot(steps, val_losses, color="#1f77b4", marker="o", linewidth=2, markersize=6, label="Val Loss")
    min_loss_idx = int(np.argmin(val_losses))
    ax1.scatter(steps[min_loss_idx], val_losses[min_loss_idx], color="red", s=100, zorder=5)
    ax1.annotate(
        f"Min: {val_losses[min_loss_idx]:.4f}\n(Step {steps[min_loss_idx]})",
        xy=(steps[min_loss_idx], val_losses[min_loss_idx]),
        xytext=(0, 15),
        textcoords="offset points",
        ha="center",
        color="red",
        fontweight="bold",
    )
    ax1.set_title("TinyStories Validation Loss Curve", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Training Steps", fontsize=11)
    ax1.set_ylabel("Cross Entropy Loss", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    # 2. 绘制 Perplexity
    ax2.plot(steps, perplexities, color="#ff7f0e", marker="s", linewidth=2, markersize=6, label="Perplexity")
    ax2.set_title("Validation Perplexity Curve", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Training Steps", fontsize=11)
    ax2.set_ylabel("Perplexity (exp(Loss))", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"\n📊 评测折线图已导出至: {save_path}")


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"🔧 评测设备: {device}")

    # 1. 扫描并按物理步数严格排序检查点
    ckpt_pattern = os.path.join(args.checkpoint_dir, "*.pt")
    ckpt_files = glob.glob(ckpt_pattern)
    if not ckpt_files:
        raise FileNotFoundError(f"在目录 {args.checkpoint_dir} 中未找到任何 .pt 权重文件！")

    ckpt_files.sort(key=extract_step_number)
    print(f"🔎 找到 {len(ckpt_files)} 个检查点:")
    for f in ckpt_files:
        print(f"   - {os.path.basename(f)} (Step {extract_step_number(f)})")

    # 2. 内存映射加载验证集
    print(f"📦 加载验证集: {args.val_data_path}")
    val_data = np.load(args.val_data_path, mmap_mode="r")

    # 3. 初始化模型结构（复用同一个模型实例）
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

    steps = []
    val_losses = []
    perplexities = []

    # 4. 遍历评估每一个 Checkpoint
    print("\n🚀 开始逐个 Checkpoint 运行验证评测...")
    print("-" * 55)
    print(f"{'Checkpoint 文件':<28} | {'Val Loss':<10} | {'Perplexity':<10}")
    print("-" * 55)

    for ckpt_path in ckpt_files:
        step = extract_step_number(ckpt_path)
        checkpoint = torch.load(ckpt_path, map_location=device)

        # 兼容只存 model 权重或存了包含 "model" 键的 checkpoint 结构
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        model.load_state_dict(state_dict)

        loss, ppl = evaluate_checkpoint(
            model=model,
            val_data=val_data,
            batch_size=args.batch_size,
            context_length=args.context_length,
            device=device,
            eval_iters=args.eval_iters,
        )

        steps.append(step)
        val_losses.append(loss)
        perplexities.append(ppl)

        filename = os.path.basename(ckpt_path)
        print(f"{filename:<28} | {loss:<10.4f} | {ppl:<10.2f}")

    print("-" * 55)

    # 5. 导出曲线图
    plot_metrics(steps, val_losses, perplexities, args.output_image)


if __name__ == "__main__":
    main()