from easytrain import load_dataset_custom, train_tokenizer
from datasets import load_dataset


def main():
    # 1. 配置参数
    data_files = {"train": "./origin_db/sex_txt/*.json"}  # 示例数据文件
    dataset = load_dataset_custom(data_files=data_files)
    # 训练分词器
    train_tokenizer(dataset, output_dir="./my_tokenizer", vocab_size=30000)


if __name__ == "__main__":
    main()
