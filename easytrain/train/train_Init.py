"""
    为模型训练提供初始化
"""

import wandb
import torch
from transformers import DataCollatorForLanguageModeling


def wandb_init(
    project="EasyTrain", name="my_unique_run_name_4090x8", training_args=None
):
    """
    初始化 wandb

    Args:
        project (str): 项目名称
        name (str): 运行名称
        training_args (TrainingArguments): 训练参数

    """
    wandb.init(
        project=project,
        entity="byron0533",
        name=name,
        config={
            "learning_rate": training_args.learning_rate,
            "batch_size": training_args.per_device_train_batch_size,
            "epochs": training_args.num_train_epochs,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
        },
    )


def collate_fn_custome(batch, tokenizer=None):
    """
    将数据批次转换为模型输入

    Args:
        batch (list): 数据批次
    """
    # 直接堆叠 input_ids 和 attention_mask
    input_ids = torch.stack([sample["input_ids"] for sample in batch])
    attention_mask = torch.stack([sample["attention_mask"] for sample in batch])

    #     # 将 input_ids 转换为文本
    #     input_text = [tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]

    #     # 打印输入的 input_ids 和其对应的文本
    #     print("Input IDs (Batch):")
    #     print(input_ids)
    #     print("Input Texts (Batch):")
    #     print(input_text)

    # 创建 labels，右移一位
    labels = input_ids.clone()  # 克隆 input_ids 作为 labels
    labels[:, :-1] = input_ids[:, 1:]  # 将 labels 向左移一位
    labels[:, -1] = tokenizer.eos_token_id  # 最后一个位置设置为 eos_token_id
    # labels = input_ids

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def collate_fn_Std(tokenizer=None):
    """
    将数据批次转换为模型输入

    Args:
        batch (list): 数据批次
    """
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False  # 对于 GPT，mlm=False，因为它是自回归模型
    )
    return data_collator
