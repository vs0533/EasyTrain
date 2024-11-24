"""
    训练分词器
"""

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from transformers import PreTrainedTokenizerFast
import os

from easytrain import config


def train_tokenizer(
    dataset,
    output_dir,
    vocab_size=30000,
    special_tokens_ext=None,
    pretrained_tokenizer_path=None,
    batch_size=1000,
):
    """
    使用流式数据训练分词器。

    Args:
        dataset (Dataset): 流式数据集。
        output_dir (str): 保存分词器的目录。
        vocab_size (int): 词汇表大小。
        special_tokens (list): 特殊标记列表。
        pretrained_tokenizer_path (str): 预训练分词器路径。

    Returns:
        None
    """
    special_tokens = [
        config.BOS_TOKEN,
        config.EOS_TOKEN,
        config.UNK_TOKEN,
        config.PAD_TOKEN,
        config.MASK_TOKEN,
    ]
    if special_tokens_ext is not None:
        special_tokens.extend(special_tokens_ext)

    # 初始化分词器
    tokenizer = None
    if pretrained_tokenizer_path is not None:
        # hf格式的分词器
        if os.path.isdir(pretrained_tokenizer_path):
            print("加载HF格式预训练分词器...")
            hf_tokenizer = PreTrainedTokenizerFast.from_pretrained(
                pretrained_tokenizer_path
            )
            tokenizer = (
                hf_tokenizer.backend_tokenizer
            )  # 这是tokenizers库的Tokenizer对象
        # json格式的分词器
        elif os.path.isfile(pretrained_tokenizer_path):
            print("加载JSON格式预训练分词器...")
            tokenizer = Tokenizer.from_file(pretrained_tokenizer_path)
    else:
        print("初始化新分词器...")
        tokenizer = Tokenizer(BPE())

    tokenizer.pre_tokenizer = ByteLevel()
    print("Special Tokens:", special_tokens)
    # 定义训练器
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)

    # 定义迭代器
    def batch_iterator(batch_size=batch_size):
        buffer = []
        for example in dataset:
            if "text" not in example or example["text"] is None:
                print("Dataset examples must have a 'text' field.")
                continue
            buffer.append(example["text"])
            if len(buffer) >= batch_size:
                yield buffer
                buffer = []
        if buffer:  # 输出剩余的缓冲数据
            yield buffer

    # 定义迭代器：非批量版本
    def sample_iterator():
        for example in dataset:
            if "text" not in example or example["text"] is None:
                print("Dataset examples must have a 'text' field.")
                continue
                # raise KeyError("Dataset examples must have a 'text' field.")
            # print(f"Processing example: {example['text'][:25]}...")  # 调试
            yield example["text"].strip()

    # 调试：检查数据是否正确加载
    print("Checking dataset samples...")
    for sample in dataset.take(1):
        print(sample)

    # 训练分词器 兼容批量训练
    if batch_size is None:
        tokenizer.train_from_iterator(sample_iterator(), trainer)
    else:
        print("开始增量训练...")
        for batch in batch_iterator(batch_size):
            tokenizer.train_from_iterator(batch, trainer)

    if output_dir is None:
        raise ValueError("output_dir must be specified")

    # 确保目录存在
    os.makedirs(output_dir, exist_ok=True)

    if os.path.isfile(output_dir) and output_dir.endswith(".json"):
        tokenizer.save(output_dir)
        return
    elif os.path.isdir(output_dir):
        Warning(
            f"选择了保存为hf格式的分词器，注意修改PreTrainedTokenizer的[特殊标记]相关参数"
        )
        hf_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
            mask_token="<mask>",
        )
        if special_tokens_ext is not None:
            hf_tokenizer.add_special_tokens(special_tokens_ext)
        hf_tokenizer.save_pretrained(output_dir)
