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


# 模型配置
HIDDEN_SIZE = 1024  # 隐藏层维度
NUM_ATTENTION_HEADS = 16  # 注意力头数
NUM_HIDDEN_LAYERS = 24  # 隐藏层层数
N_INNER = 4096  # 前馈网络维度
MAX_POSITION_EMBEDDINGS = 1024  # 最大位置编码数量

# 训练参数
TRAIN_OUTPUT_DIR = "./check_point"  # 检查点输出目录
TRAIN_FINISHED_MODEL_DIR = "./finished_model"  # 训练完成模型保存目录
EVAL_STEPS = 1000000  # 每X步评估一次
TRAIN_BATCH_SIZE = 48  # 训练 batch size
GRADIENT_ACCUMULATION_STEPS = 128  # 梯度累积
TRAIN_EPOCHS = 10  # 训练轮数
SAVE_STEPS = 20  # 每X步保存一次模型
LOGGING_STEPS = 1  # 每X步记录日志
LEARNING_RATE = 4e-5  # 学习率
WARMUP_STEPS = 10000  # warmup 步数; 0 表示不使用 warmup; 通常是 0.1 * 训练步数; 学习率会从0，经过指定的步数[warmup_steps]线性增长到设定值[learning_rate]
LR_SCHEDULER_TYPE = "linear"  # 学习率调度器类型
DEEPSPEED = "0"  # deepspeed 使用的zero级别
