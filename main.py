from easytrain import load_dataset_custom, train_tokenizer
from datasets import load_dataset
from easytrain import utils


def main():
    # 1. 配置参数
    data_files = {
        "wiki": "./dataset_origin/wiki_chinese/data/*.parquet",
        # "sex1": "./dataset_origin/sex1/*.json",
        # "sex2": "./dataset_origin/sex2/modified_erotic_literature_collection.csv",
    }  # 示例数据文件
    dataset = load_dataset_custom(
        data_files=data_files,
        streaming=True,
        field_mapping={"sex2": "output"},
    )
    
    dataset = dataset.shuffle(seed=42)
    # dataset = dataset.skip(18000)
    dataset = dataset.take(3000)

    # count = sum(1 for _ in dataset)
    # print(count)
    # for example in dataset.take(1):
    #     print(example["text"])
    # print(dataset.features)
    # print(dataset.info)
    # 训练分词器
    train_tokenizer(
        dataset, output_dir="tokenizer", vocab_size=30000, batch_size=3000,pretrained_tokenizer_path="tokenizer2"
    )


if __name__ == "__main__":
    main()
