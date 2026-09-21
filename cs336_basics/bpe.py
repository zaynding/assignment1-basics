import regex as re
from collections import defaultdict

# 讲义指定的 GPT-2 预分词正则表达式
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# 输出Token ID 和 字节串 的映射，以及分词规则（有顺序）
def train_bpe(
    input_path : str,
    vocab_size : int,
    special_tokens : list[str],
) -> tuple[dict[int,bytes],list[tuple[bytes,bytes]]]:
    # 初始化基础词表（0-255字节映射）
    vocab : dict[int,bytes] = {i: bytes([i]) for i in range(256)}
    next_id = 256
    
    # 1. 注册特殊 Token 到 vocab 中
    for st in special_tokens:
        # 将 str 转为 bytes
        st_bytes = st.encode("utf-8")
        vocab[next_id] = st_bytes
        next_id += 1
        
    # 2. 读取文本（utf-8)并对特殊 Token 做硬隔离切分
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
        
    if special_tokens:
        sorted_tokens = sorted(special_tokens, key=len, reverse=True)
        # 构建正则表达式模式，匹配特殊 Token
        special_pattern = "|".join(re.escape(st) for st in sorted_tokens)
        # 使用正则表达式进行切分，保留特殊 Token
        text_chunks = re.split(special_pattern, text)
    else:
        text_chunks = [text]
        
    # 3. 统计分词词频（word_freqs)
    word_freqs: dict[tuple[bytes, ...], int] = defaultdict(int)
    compiled_pat = re.compile(PAT)
    
    for chunk in text_chunks:
        if not chunk:
            continue
        for match in compiled_pat.finditer(chunk):
            piece = match.group(0)
            token_bytes = piece.encode("utf-8")
            # 将每个字节切分成独立的 tuple 项
            # word 是键，不可变所以要转成 tuple
            word = tuple(bytes([b]) for b in token_bytes)
            word_freqs[word] += 1
            
    # 4. 构建对频字典（字节对出现频率）和倒排字典索引(根据字节对查找包含它的词)
    pair_freqs: dict[tuple[bytes,bytes], int] = defaultdict(int)
    pair_to_words: dict[tuple[bytes,bytes], set[tuple[bytes,...]]] = defaultdict(set)
    
    for word, freq in word_freqs.items():
        # 遍历 word 中的相邻字节对
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_freqs[pair] += freq
            pair_to_words[pair].add(word)
    
    # 5. 循环合并，直到达到 vocab_size
    merges: list[tuple[bytes, bytes]] = []
    
    
    # 按照pair要求，合并频率最高的字节对
    def merge_word(word:tuple[bytes,...], pair:tuple[bytes,bytes]) -> tuple[bytes,...]:
        """将 word 中的 pair 合并为一个新字节"""
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                # 合并字节对
                new_word.append(pair[0] + pair[1])
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return tuple(new_word)
    
    while len(vocab) < vocab_size and pair_freqs:
        # 平局打破：按 (频次, 字节对字典序) 选最大值
        best_pair = max(pair_freqs.keys(), key=lambda p: (pair_freqs[p], p))

        # 记录本次合并
        merges.append(best_pair)
        new_token = best_pair[0] + best_pair[1]
        vocab[next_id] = new_token
        next_id += 1

        # 针对包含该 pair 的旧词执行增量更新
        '''
        总结一下
        先在对所有涉及到这个 pair 的旧单词在词频中删除，对频要减去这个词的次数，并把这个pair从倒排索引中删除；
        然后生成新词，在词频中更新，在对频中更新，倒排索引中更新。后两个对频和倒排顺序可以换
        '''
        words_to_update = list(pair_to_words[best_pair])
        for old_word in words_to_update:
            # .pop 返回旧词的频次并从字典中删除
            count = word_freqs.pop(old_word)

            # 1) 扣减旧词此前贡献的所有 pair 频次
            for i in range(len(old_word) - 1):
                p = (old_word[i], old_word[i + 1])
                pair_freqs[p] -= count
                if pair_freqs[p] <= 0:
                    del pair_freqs[p]
                pair_to_words[p].discard(old_word)

            # 2) 生成合并后的新词
            new_word = merge_word(old_word, best_pair)
            word_freqs[new_word] += count

            # 3) 增量计入新词产生的所有 pair 频次
            for i in range(len(new_word) - 1):
                p = (new_word[i], new_word[i + 1])
                pair_freqs[p] += count
                pair_to_words[p].add(new_word)

    return vocab, merges
    