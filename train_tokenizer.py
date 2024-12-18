from easytrain import load_dataset_custom, train_tokenizer, process_data_to_fixed_length
from datasets import load_dataset
from easytrain import utils


def main():
    # 初始化数据集
    data_files = {
        # "wiki": "./dataset_origin/wiki_chinese/data/*.parquet",
        "sex1": "./dataset_origin/sex1/*.json",
        # "sex2": "./dataset_origin/sex2/modified_erotic_literature_collection.csv",
    }
    dataset = load_dataset_custom(
        data_files=data_files,
        streaming=False,
        field_mapping={"sex2": "output"},
    )
    dataset = dataset.shuffle(seed=42)
    # dataset = dataset.skip(4750000)
    # dataset = dataset.take(3000)

    # count = sum(1 for _ in dataset)
    # print(count)
    # for example in dataset.take(1):
    #     print(example["text"])
    # print(dataset.features)
    # print(dataset.info)

    # 训练分词器
    train_tokenizer(
        dataset,
        output_dir="tokenizer",
        vocab_size=30000,
        batch_size=10000,
        pretrained_tokenizer_path="tokenizer_Checkpoint/interim_1160.json",
    )


if __name__ == "__main__":
    main()
