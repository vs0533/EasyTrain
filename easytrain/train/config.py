from transformers import TrainingArguments
from ..config import (
    TRAIN_OUTPUT_DIR,
    EVAL_STEPS,
    TRAIN_BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    TRAIN_EPOCHS,
    SAVE_STEPS,
    LOGGING_STEPS,
    LEARNING_RATE,
    WARMUP_STEPS,
    LR_SCHEDULER_TYPE,
    DEEPSPEED,
)


# 设置训练参数
training_args = TrainingArguments(
    remove_unused_columns=False,  # 忽略数据集中的额外列
    output_dir=TRAIN_OUTPUT_DIR,  # 输出目录
    # eval_strategy="steps",  # 每隔一定步骤进行评估
    eval_strategy="no",  # 不进行评估
    eval_steps=EVAL_STEPS,  # 每X步评估一次
    # logging_dir="./logs", # 日志目录
    per_device_train_batch_size=TRAIN_BATCH_SIZE,  # 每个设备的 batch size
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,  # 梯度累积
    max_grad_norm=1.0,  # 适当的梯度裁剪值
    num_train_epochs=TRAIN_EPOCHS,  # 训练轮数
    save_steps=SAVE_STEPS,  # 每500步保存一次模型
    logging_steps=LOGGING_STEPS,  # 每10步记录日志
    learning_rate=LEARNING_RATE,  # 学习率
    fp16=True,  # 混合精度训练
    # max_steps=10,  # 根据数据量调整 max_steps
    report_to="wandb",  # 将日志发送到 wandb
    log_level="debug",  # 设置日志级别为 debug
    warmup_steps=WARMUP_STEPS,  # warmup 步数
    lr_scheduler_type=LR_SCHEDULER_TYPE,  # 学习率调度器类型
    # deepspeed="deepspeed_config_0.json",
)
