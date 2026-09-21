import os
import numpy as np
from cs336_basics.tokenizer import Tokenizer

def tokenize_and_save(
    input_txt_path: str, 
    vocab_path: str, 
    merges_path: str, 
    output_npy_path: str, 
    special_tokens: list[str]
):
    print(f"1. 正在加载分词器: {vocab_path} 和 {merges_path} ...")
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens=special_tokens)

    # 批次读取行生成器：既避免一次性撑爆内存，又防止把词语/特殊标记硬生生截断
    def safe_line_batch_generator(filepath, batch_lines=2000):
        with open(filepath, "r", encoding="utf-8") as f:
            batch = []
            for line in f:
                batch.append(line)
                if len(batch) >= batch_lines:
                    yield "".join(batch)
                    batch = []
            if batch:
                yield "".join(batch)

    print(f"2. 开始分词: {input_txt_path} ...")
    token_ids_iter = tokenizer.encode_iterable(safe_line_batch_generator(input_txt_path))
    
    # 转换为 uint16 格式的 NumPy 数组
    token_ids_array = np.fromiter(token_ids_iter, dtype=np.uint16)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_npy_path) or ".", exist_ok=True)
    
    print(f"3. 正在保存 {len(token_ids_array):,} 个 Tokens 至 {output_npy_path} ...")
    np.save(output_npy_path, token_ids_array)
    print("保存完成！\n")

    # ================= 效果与指标验证 =================
    print("=" * 20 + " 分词效果验证报告 " + "=" * 20)
    # 1. 计算原始文件大小与压缩率 (对应讲义 2.7 思考题)
    raw_bytes_size = os.path.getsize(input_txt_path)
    total_tokens = len(token_ids_array)
    compression_ratio = raw_bytes_size / total_tokens if total_tokens > 0 else 0
    
    print(f"原始文本大小: {raw_bytes_size / (1024 * 1024):.2f} MB")
    print(f"总 Token 数量: {total_tokens:,}")
    print(f"分词压缩率:   {compression_ratio:.3f} bytes/token (正常在 3.5 ~ 4.5 之间)")
    print(f"最大 Token ID: {token_ids_array.max()} (不超过 9999 且安全落入 uint16)")

    # 2. 抽样反向解码，确认无损还原
    sample_ids = token_ids_array[:50].tolist()
    decoded_sample = tokenizer.decode(sample_ids)
    print("\n抽样前 50 个 Token 解码预览:")
    print("-" * 50)
    print(decoded_sample)
    print("-" * 50)

if __name__ == "__main__":
    SPECIAL_TOKENS = ["<|endoftext|>"]
    VOCAB_PATH = "vocab.pkl"
    MERGES_PATH = "merges.pkl"
    
    # 路径根据你本地的文件名确认
    INPUT_FILE = "/Users/alex/项目/research/cs336/TinyStoriesV2-GPT4-train.txt"
    OUTPUT_FILE = "data/TinyStories-train.npy"
    
    tokenize_and_save(
        input_txt_path=INPUT_FILE,
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
        output_npy_path=OUTPUT_FILE,
        special_tokens=SPECIAL_TOKENS
    )