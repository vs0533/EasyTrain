"""
    训练分词器
"""

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel,Split,Sequence
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from transformers import PreTrainedTokenizerFast
import os

from easytrain import config
from easytrain.utils import isfile


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
        special_tokens_ext (list): 扩展的特殊标记列表。
        pretrained_tokenizer_path (str): 预训练分词器路径。

    Returns:
        None
    """
    
    if output_dir is None:
        raise ValueError("保存目录不能为空！")
    
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
        else:
            print("加载JSON格式预训练分词器...")
            tokenizer = Tokenizer.from_file(pretrained_tokenizer_path)
    else:
        print("初始化新分词器...")
        tokenizer = Tokenizer(BPE())
        
    # 确保预处理器和解码器存在
    if tokenizer.pre_tokenizer is None:
        # 使用 Split 和 ByteLevel 预处理器，并组合它们
        tokenizer.pre_tokenizer = Sequence([
            Split(
                pattern=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",  # 正则表达式
                behavior="isolated",  # isolated 行为
                invert=False
            ),
            ByteLevel(add_prefix_space=False, trim_offsets=False, use_regex=False)
        ])
        
    if tokenizer.decoder is None:
        tokenizer.decoder = ByteLevelDecoder()

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
        print("开始分批训练...")
        batch_ctr = 0
        save_interval = 1  # 指定间隔批次
        for batch in batch_iterator(batch_size):
            batch_ctr += 1
            print(f"批次 {batch_ctr}...")
            tokenizer.train_from_iterator(batch, trainer)
            if save_interval is not None and save_interval > 0 and batch_ctr % save_interval == 0:
                interim_output_dir = os.path.join("tokenizer_Checkpoint", f"interim_{batch_ctr}.json")
                print(f"保存中间分词器到 {interim_output_dir}...")
                save_tokenizer(tokenizer, interim_output_dir, special_tokens_ext)

    save_tokenizer(tokenizer, output_dir, special_tokens_ext)


# 保存分词器
def save_tokenizer(tokenizer, output_dir,special_tokens_ext=None):
    """
    保存分词器。

    Args:
        tokenizer (Tokenizer): 分词器。
        output_dir (str): 保存目录。

    Returns:
        None
    """
    if isfile(output_dir) and output_dir.endswith(".json"):
        print(f"保存Json分词器{output_dir}...")
        directory = os.path.dirname(output_dir)
        if directory != "":
            os.makedirs(directory, exist_ok=True)
        tokenizer.save(output_dir)
        return
    else:
        # 确保目录存在
        os.makedirs(output_dir, exist_ok=True)
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
