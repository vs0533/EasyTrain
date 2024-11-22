"""
处理数据
"""

from transformers import AutoTokenizer


def process_data_to_fixed_length(dataset, tokenizer_name, max_length, output_file):
    """
    按句子切分 + 拼接，保存为固定长度。

    Args:
        dataset (Dataset): 数据集。
        tokenizer_name (str): 分词器名称。
        max_length (int): 最大 Token 长度。
        output_file (str): 输出文件路径。

    Returns:
        None
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    current_chunk = []
    current_length = 0

    with open(output_file, "w") as f:
        for example in dataset:
            text = example["text"]
            tokens = tokenizer.encode(text, add_special_tokens=False)
            if current_length + len(tokens) + 2 > max_length:
                f.write(" ".join(current_chunk) + "\n")
                current_chunk = []
                current_length = 0

            current_chunk.append(text)
            current_length += len(tokens)

        if current_chunk:
            f.write(" ".join(current_chunk) + "\n")


def tokenize_and_save(dataset, tokenizer, output_file):
    """
    对数据进行 Token 化并保存。

    Args:
        dataset (Dataset): 数据集。
        tokenizer (AutoTokenizer): 分词器。
        output_file (str): 输出文件路径。

    Returns:
        None
    """
    with open(output_file, "w") as f:
        for example in dataset:
            text = example["text"]
            encoded = tokenizer(text, truncation=True, max_length=512)
            f.write(f"{encoded['input_ids']}\n")
