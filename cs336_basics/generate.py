import argparse
import torch

from cs336_basics.decoding import decode
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transformer import TransformerLM


def parse_args():
    parser = argparse.ArgumentParser(description="使用训练好的 Transformer LM 生成文本")
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoints/checkpoint_step_5000.pt")
    parser.add_argument("--vocab_path", type=str, default="./vocab.pkl", help="BPE 词表路径")
    parser.add_argument("--merges_path", type=str, default="./merges.pkl", help="BPE 合并规则路径")
    parser.add_argument("--prompt", type=str, default="Once upon a time, there was a little")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8, help="采样温度 (0.0 为贪婪解码, 0.7~0.8 流畅且有创造力)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus 采样阈值")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    # 模型架构超参数 (需与训练时严格一致)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # 1. 加载分词器
    tokenizer = Tokenizer.from_files(
        vocab_filepath=args.vocab_path,
        merges_filepath=args.merges_path,
        special_tokens=["<|endoftext|>"],
    )

    # 2. 实例化并加载模型权重
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        device=device,
    ).to(device)

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print(f"✅ 成功加载第 {checkpoint.get('iteration', 'unknown')} 步的权重")

    # 3. 编码 Prompt
    prompt_tokens = tokenizer.encode(args.prompt)
    print(f"\nPrompt: \"{args.prompt}\"")
    print(f"Prompt Tokens: {prompt_tokens}")

    # 获取 <|endoftext|> 的 token id
    eos_token_id = tokenizer.encode("<|endoftext|>")[0] if "<|endoftext|>" in tokenizer.vocab.values() else None

    # 4. 执行自回归解码
    generated_tokens = decode(
        model=model,
        prompt_tokens=prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token_id=eos_token_id,
    )

    # 5. 解码为可读文本
    generated_text = tokenizer.decode(generated_tokens)
    print("\n" + "=" * 40 + " 故事生成结果 " + "=" * 40)
    print(generated_text)
    print("=" * 94)


if __name__ == "__main__":
    main()