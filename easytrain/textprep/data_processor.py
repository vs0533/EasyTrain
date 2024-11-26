import re
from .tokenizer_loader import tokenizer_loader


def process_data_with_limited_length_and_sliding(
    dataset,
    tokenizer_name,
    max_length,
    output_file,
    save_as_tokens=False,  # 保存为 Token 或句子
    overlap=2,  # 滑动窗口的重叠数
):
    tokenizer = tokenizer_loader(tokenizer_name)

    def split_sentences(text):
        """按标点符号切分句子"""
        if not isinstance(text, str):
            raise TypeError("expected string or bytes-like object")
        return re.findall(r"[^,，.。！？;；\n]+[，。！？;；]?", text)

    def save_chunk(f, chunk):
        """保存一个块到文件"""
        if chunk:
            if save_as_tokens:
                f.write(" ".join(chunk) + "\n")  # 保存 Token
            else:
                f.write("".join(chunk) + "\n")  # 保存句子

    with open(output_file, "w") as f:
        for example in dataset:
            text = example["text"]
            if not isinstance(text, str):
                text = str(text)
            text = text.strip()
            if not text:
                continue

            # 1. 切分句子
            sentences = split_sentences(text)
            if not sentences:
                continue

            if save_as_tokens:
                # === Token 级别处理 ===
                tokens = tokenizer.encode("".join(sentences), add_special_tokens=False)
                idx = 0
                while idx < len(tokens):
                    chunk = tokens[idx : idx + max_length]  # 限制每行最大长度
                    save_chunk(f, list(map(str, chunk)))
                    idx += max_length - overlap  # 滑动窗口
            else:
                # === 句子级别处理 ===
                idx = 0
                current_chunk = []
                current_length = 0

                while idx < len(sentences):
                    sentence = sentences[idx]
                    sentence_length = len(
                        tokenizer.encode(sentence, add_special_tokens=False)
                    )

                    if current_length + sentence_length > max_length:
                        # 当前块已满，保存并开始新块
                        save_chunk(f, current_chunk)
                        current_chunk = []
                        current_length = 0

                    current_chunk.append(sentence)
                    current_length += sentence_length
                    idx += 1

                # 保存最后一块
                if current_chunk:
                    save_chunk(f, current_chunk)
