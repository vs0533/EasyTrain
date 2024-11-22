# 配置文件
DATA_FILE = "data.txt"  # 输入文件路径
OUTPUT_DIR = "data_chunks"  # 分块保存目录
TOKENIZER_JSON_FILE = "my_tokenizer.json"  # 分词器配置文件
CHUNK_SIZE = 100000  # 每个分块行数
SAMPLE_RATE = 0.001  # 随机采样比例
MAX_SAMPLES = 100000  # 最大采样数量
TOKENIZER_DIR_OR_FILE = "bert-base-uncased"  # 分词器名称

# 分词器配置
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
MASK_TOKEN = "<mask>"
