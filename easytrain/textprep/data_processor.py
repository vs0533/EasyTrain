import re
from .tokenizer_loader import tokenizer_loader


def process_data_to_fixed_length(
    dataset,
    tokenizer_name,
    max_length,
    output_file,
    discard_overlength=False,
    save_as_tokens=False,
):
    """
    处理数据集，将超长句子截断或丢弃，并尽量拼接短句使块接近 max_length。

    Args:
        dataset (Dataset): 数据集，每个样本包含 `text` 字段。
        tokenizer_name (str): 分词器名称。
        max_length (int): 最大 Token 长度。
        output_file (str): 输出文件路径。
        discard_overlength (bool): 是否丢弃超长句子（默认截断保存）。
        save_as_tokens (bool): 是否将输出保存为 token 序列（默认 True）。

    Returns:
        None
    """
    tokenizer = tokenizer_loader(tokenizer_name)
    current_chunk = []
    current_length = 0

    def split_sentences(text):
        """按标点符号切分句子"""
        if not isinstance(text, str):
            raise TypeError("expected string or bytes-like object")
        # 使用正则表达式匹配句子和分隔符
        return re.findall(r"[^,，.。！？\n]+[，。！？]?", text)

    with open(output_file, "w") as f, open(
        "truncated_sentences.txt", "w"
    ) as trunc_file:
        for example in dataset:
            text = example["text"]
            if not isinstance(text, str):
                text = str(text)  # 转换为字符串
            text = text.strip()
            sentences = split_sentences(text)

            for sentence in sentences:
                if not sentence.strip():  # 跳过空句子
                    continue

                tokens = tokenizer.encode(sentence, add_special_tokens=False)

                # 超长句子的处理
                if len(tokens) > max_length:
                    if discard_overlength:
                        # 丢弃超长句子
                        print(f"丢弃超长句子: {sentence}")
                        continue
                    else:
                        # 截断超长句子
                        trunc_file.write(f"Original: {sentence}\n")
                        tokens = tokens[: max_length - 1] + [tokenizer.eos_token_id]
                        trunc_file.write(f"Truncated: {tokens}\n")
                        print(f"截断超长句子: {sentence} -> {tokens}")

                # 如果当前块超长，保存当前块
                while current_length + len(tokens) > max_length:
                    # 计算可以填充的 token 数量
                    space_left = max_length - current_length

                    # 保存当前块
                    if current_chunk:
                        if save_as_tokens:
                            f.write(" ".join(current_chunk) + "\n")
                        else:
                            f.write("".join(current_chunk) + "\n")
                        current_chunk = []
                        current_length = 0
                    else:
                        # 当前块为空，直接截断当前句子
                        trunc_file.write(f"Original (forced truncate): {sentence}\n")
                        if save_as_tokens:
                            f.write(" ".join(map(str, tokens[:space_left])) + "\n")
                        else:
                            f.write(tokenizer.decode(tokens[:space_left]) + "\n")
                        tokens = tokens[space_left:]

                # 添加到当前块
                if save_as_tokens:
                    current_chunk.extend(map(str, tokens))
                else:
                    current_chunk.append(sentence)
                current_length += len(tokens)

        # 保存最后的块
        if current_chunk:
            if save_as_tokens:
                f.write(" ".join(current_chunk) + "\n")
            else:
                f.write("".join(current_chunk) + "\n")
