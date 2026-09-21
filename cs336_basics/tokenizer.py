import regex as re
import pickle
from collections.abc import Iterable, Iterator


class Tokenizer:
    # 讲义指定的预分词正则表达式
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab.copy()
        self.merges = merges
        self.special_tokens = special_tokens or []
        
        # 1. 建立特殊 Token 集合与正则 Pattern
        self.special_tokens_set = set(self.special_tokens)
        if self.special_tokens:
            # 使用捕获组 ()，这样 re.split 切分后，特殊 token 本身也会保留在结果列表中
            sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
            escaped = [re.escape(st) for st in sorted_tokens]
            self.split_pattern = re.compile(f"({'|'.join(escaped)})")
        else:
            self.split_pattern = None

        # 2. 动态追加特殊 Token 到词表 (如果还不在的话)
        max_id = max(self.vocab.keys()) if self.vocab else 255
        self.special_tokens_to_id = {}
        for st in self.special_tokens:
            st_bytes = st.encode("utf-8")
            # 检查是否已在 vocab 中
            existing_id = None
            for vid, vbytes in self.vocab.items():
                if vbytes == st_bytes:
                    existing_id = vid
                    break
            
            if existing_id is not None:
                self.special_tokens_to_id[st] = existing_id
            else:
                max_id += 1
                self.vocab[max_id] = st_bytes
                self.special_tokens_to_id[st] = max_id

        # 3. 建立反向词表，实现 O(1) 查 ID
        self.bytes_to_id = {v: k for k, v in self.vocab.items()}
        
        # 4. 建立合并规则优先级索引，序号越小优先级越高
        self.merges_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        
        # 提前编译 GPT-2 正则提速
        self.compiled_pat = re.compile(self.PAT)

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        """
        从文件加载 Tokenizer。
        由于字典键为 int，值为 bytes，直接使用 pickle 是最方便且原生的序列化方式。
        """
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
            
        return cls(vocab, merges, special_tokens)

    def _encode_chunk(self, chunk_bytes: bytes) -> list[int]:
        """核心 BPE 合并逻辑：处理不含特殊 Token 的纯字节块"""
        # 初始化单字节列表
        word = [bytes([b]) for b in chunk_bytes]
        
        while len(word) >= 2:
            # 找到当前 word 中优先级最高（rank 最小）的相邻字节对
            min_rank = float("inf")
            best_pair = None
            
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                rank = self.merges_ranks.get(pair, float("inf"))
                if rank < min_rank:
                    min_rank = rank
                    best_pair = pair
                    
            # 如果找不到任何在 merges 里登记过的对，停止合并
            if min_rank == float("inf"):
                break
                
            # 从左到右贪心合并
            first, second = best_pair
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
            
        return [self.bytes_to_id[b] for b in word]

    def encode(self, text: str) -> list[int]:
        """将完整文本编码为 Token ID 序列"""
        ids = []
        
        # 1. 按照特殊 Token 切分文本
        if self.split_pattern:
            chunks = self.split_pattern.split(text)
        else:
            chunks = [text]
            
        # 2. 遍历切分好的片段
        for chunk in chunks:
            if not chunk:
                continue
                
            # 若是特殊 Token，直接换算为 ID
            if chunk in self.special_tokens_set:
                ids.append(self.special_tokens_to_id[chunk])
            else:
                # 否则使用 GPT-2 正则匹配普通词，并对每个词应用 BPE 编码
                for match in self.compiled_pat.finditer(chunk):
                    piece = match.group(0)
                    chunk_bytes = piece.encode("utf-8")
                    ids.extend(self._encode_chunk(chunk_bytes))
                    
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """迭代生成器：用于极低内存占用处理大型文件"""
        for chunk in iterable:
            for token_id in self.encode(chunk):
                yield token_id

    def decode(self, ids: list[int]) -> str:
        """根据 ID 解码回文本，自动替换非法字节序列"""
        byte_chunks = [self.vocab[i] for i in ids]
        merged_bytes = b"".join(byte_chunks)
        # errors='replace' 保证遇到截断的特殊字符时不会报错，而是替换为 \uFFFD
        return merged_bytes.decode("utf-8", errors="replace")