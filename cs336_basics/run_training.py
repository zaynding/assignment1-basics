import time
import pickle
from cs336_basics.bpe import train_bpe

def main():
    # 讲义要求：TinyStories 词表大小设为 10000，且必须包含特殊 Token
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    
    # 替换为你下载的 TinyStories 训练集或验证集路径
    # 建议先用验证集 (TinyStoriesV2-GPT4-valid.txt) 跑，文件小，验证快[cite: 1]
    input_path = "/Users/alex/项目/research/cs336/TinyStoriesV2-GPT4-train.txt" 
    
    print(f"开始训练 BPE 分词器 (目标词表大小: {vocab_size})...")
    start_time = time.time()
    
    # 执行你辛辛苦苦写好的核心算法
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    
    end_time = time.time()
    print(f"训练完成！耗时: {end_time - start_time:.2f} 秒")
    
    # 将训练产物序列化保存，供 Tokenizer 和后续预处理使用
    print("正在保存 vocab.pkl 和 merges.pkl...")
    with open("vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open("merges.pkl", "wb") as f:
        pickle.dump(merges, f)
        
    print("保存成功！你现在拥有了自己的分词规则手册。")
    print(f"词表最大 ID: {max(vocab.keys())}, merges 数量: {len(merges)}")

if __name__ == "__main__":
    main()