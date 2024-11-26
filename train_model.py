from easytrain import load_dataset_custom, tokenizer_loader, TrainModel
from datasets import load_dataset


def main():
    # 初始化数据集
    data_files = {
        # "wiki": "./dataset_origin/wiki_chinese/data/*.parquet",
        # "sex1": "./sex1.txt",
        "sex2": "./sex2.txt",
    }
    dataset = load_dataset_custom(data_files=data_files, streaming=False)
    # dataset = dataset.shuffle(seed=42)
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

    # for example in dataset.take(2):
    #     tokenline = tokenize_line(example)
    #     print(tokenline)
    #     print(tokenizer.decode(tokenline["input_ids"]))

    # count = sum(1 for _ in dataset)
    # print(count)
    # for example in dataset.take(1):
    #     print(example["text"])
    # print(dataset.features)
    # print(dataset.info)
    # 训练分词器
    # train_tokenizer(
    #     dataset,
    #     output_dir="tokenizer",
    #     vocab_size=30000,
    #     batch_size=10000,
    #     pretrained_tokenizer_path="tokenizer_Checkpoint/interim_1160.json",
    # )


if __name__ == "__main__":
    main()
