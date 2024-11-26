"""
    针对单独的json文件的tokenizer和hf格式的tokenizer都可以加载
"""

from transformers import PreTrainedTokenizerFast, AutoTokenizer
from ..utils import isfile


def tokenizer_loader(tokenizer_path):
    """
    加载分词器。

    Args:
        tokenizer_path (str): 分词器路径。

    Returns:
        Tokenizer: 分词器对象。
    """
    tokenizer = None
    if not isfile(tokenizer_path):
        print("加载HF格式预训练分词器...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        print("加载JSON格式预训练分词器...")
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file="tokenizer_Checkpoint/interim_1160.json"
        )
    return tokenizer
