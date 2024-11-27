from easytrain import load_dataset_custom, tokenizer_loader, TrainModel
from datasets import load_dataset


def main():
    # 初始化数据集
    data_files = {
        # "wiki": "./dataset_origin/wiki_chinese/data/*.parquet",
        # "sex1": "./sex1.txt",
        # "sex2": "./sex2.txt",
        "train": "./wiki_token_ds/*.txt"
    }
    dataset = load_dataset_custom(data_files=data_files, streaming=True)

    # # 将 train 数据集拆分为 train 和 validation
    # split_dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
    # # 查看拆分结果
    # train_dataset = split_dataset["train"]
    # validation_dataset = split_dataset["test"]

    dataset = dataset.shuffle(seed=42)
    # dataset = dataset.skip(4750000)
    # dataset = dataset.take(3000)

    # ====== 加载数据 ======
    def tokenize_line(examples):
        # examples["text"] 是一个包含多个样本文本的列表
        tokens = [list(map(int, text.split())) for text in examples["text"]]
        return {"input_ids": tokens}

    # 使用 batched=True 时，Hugging Face 会自动分批
    dataset = dataset.map(tokenize_line, batched=True, remove_columns=["text"])

    tokenizer = tokenizer_loader("tokenizer")

    # for example in dataset:
    #     print(tokenizer.decode(example["input_ids"]))

    # ====== 训练模型 ======
    trainModel = TrainModel(tokenizer=tokenizer)
    trainModel.start_train(train_ds=dataset)


if __name__ == "__main__":
    main()
