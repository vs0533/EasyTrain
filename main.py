from easytrain import load_dataset_custom, train_tokenizer
from datasets import load_dataset


def main():
    # 1. 配置参数
    data_files = {
        # "wiki": "./dataset_origin/wiki_chinese/data/*.parquet",
        "sex1": "./dataset_origin/sex1/*.json",
        "sex2": "./dataset_origin/sex2/modified_erotic_literature_collection.csv",
    }  # 示例数据文件
    dataset = load_dataset_custom(
        data_files=data_files,
        streaming=False,
        field_mapping={"sex2": "output"},
    )

    dataset = dataset.shuffle(seed=42)

    # count = sum(1 for _ in dataset)
    # print(count)
    # for example in dataset.take(1):
    #     print(example["text"])
    # print(dataset.features)
    # print(dataset.info)
    # 训练分词器
    train_tokenizer(dataset, output_dir="./tokenizer", vocab_size=30000)


if __name__ == "__main__":
    main()
