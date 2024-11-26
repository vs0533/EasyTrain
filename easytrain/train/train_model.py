"""
    万事俱备，只欠训练
"""

import torch
import random

from .GptConfig import MyGPTConfig
from .train_Init import wandb_init, collate_fn_custome, collate_fn_Std
from transformers import GPT2LMHeadModel, Trainer
from .config import training_args
from ..config import (
    TRAIN_FINISHED_MODEL_DIR,
    HIDDEN_SIZE,
    NUM_ATTENTION_HEADS,
    NUM_HIDDEN_LAYERS,
    N_INNER,
    MAX_POSITION_EMBEDDINGS,
)


class MyGPT2LMHeadModel(GPT2LMHeadModel):
    def __init__(self, config: MyGPTConfig):
        super().__init__(config)


class TrainModel:
    def __init__(self, tokenizer=None, state_dict_path=None, wandb_run_name=None):
        self.tokenizer = tokenizer
        self.mode_config = self._initialize_model_config()
        self.state_dict_path = state_dict_path
        self.training_args = training_args
        self.wandb_run_name = wandb_run_name
        self.collate_fn = collate_fn_Std(tokenizer)
        self.model = self._initialize_model()

    def _initialize_model(self):
        """初始化模型"""
        model = MyGPT2LMHeadModel(self.mode_config)
        if self.state_dict_path is not None:
            model.load_state_dict(torch.load(self.state_dict_path))
            print("加载模型参数成功！")
        return model

    def _initialize_model_config(self):
        """初始化模型配置"""
        configGpt = MyGPTConfig(
            hidden_size=HIDDEN_SIZE,  # 隐藏层维度
            num_attention_heads=NUM_ATTENTION_HEADS,  # 注意力头数
            num_hidden_layers=NUM_HIDDEN_LAYERS,  # 隐藏层层数
            n_inner=N_INNER,  # 前馈网络维度
            max_position_embeddings=MAX_POSITION_EMBEDDINGS,  # 最大位置编码数量
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            unk_token_id=self.tokenizer.unk_token_id,
            mask_token_id=self.tokenizer.mask_token_id,
        )
        return configGpt

    def start_train(self, train_ds=None, val_ds=None):
        """开始训练"""
        if self.wandb_run_name is not None:
            wandb_run_name = f"{wandb_run_name}_{''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=2))}"
            wandb_init(name=wandb_run_name, training_args=self.training_args)
        print("开始训练")
        if self.training_args.eval_strategy is not None:
            trainer = Trainer(
                model=self.model,
                args=self.training_args,
                data_collator=self.collate_fn,
                train_dataset=train_ds,
                eval_dataset=val_ds,
            )
        else:
            trainer = Trainer(
                model=self.model,
                args=self.training_args,
                data_collator=self.collate_fn,
                train_dataset=train_ds,
            )
        trainer.train()
        print("训练完成")
        trainer.save_model(TRAIN_FINISHED_MODEL_DIR)
        print(f"模型已保存到 {TRAIN_FINISHED_MODEL_DIR}")
