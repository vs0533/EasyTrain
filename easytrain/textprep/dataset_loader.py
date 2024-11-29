"""
    负责加载数据集，支持流式加载
"""

from datasets import load_dataset, concatenate_datasets
import datasets


def load_dataset_custom(data_files, streaming=True, field_mapping=None):
    """
    加载数据集，支持多种文件类型并合并。

    Args:
        data_files (dict): 包含文件路径的字典，例如：
                           {"train": "./data/*.json", "val": "./data/*.parquet"}
        streaming (bool): 是否流式加载。

    Returns:
        Dataset or IterableDataset: 合并后的数据集。
    """
    datasets = []

    for name, file_pattern in data_files.items():
        # 获取扩展名
        file_ext = file_pattern.split(".")[-1]
        if file_ext == "txt":
            file_ext = "text"

        print(f"正在加载数据集:{name}")
        dataset = load_dataset(
            file_ext, data_files={name: file_pattern}, streaming=streaming,cache_dir="./.cache"
        )
        print(f"数据集加载完成:{name}:\n{dataset}")
        print("=====================================")

        # 如果提供了field_mapping参数，则根据映射关系重命名字段
        if field_mapping is not None:
            if name in field_mapping:
                print(f"重命名字段:{field_mapping[name]} -> 'text'")
                # dataset = dataset.rename_column(field_mapping[name], "text")
                # 可能在流式加载中第一次调用时无法识别字段，因此使用map
                dataset = dataset.map(lambda x: {"text": x[field_mapping[name]]})

        # 添加到数据集列表
        datasets.append(dataset[name])

    # 如果只有一个数据集直接返回，否则合并
    if len(datasets) == 1:
        print("只有一个数据集，直接返回")
        return datasets[0]
    print("合并数据集")
    result = concatenate_datasets(datasets)
    print("数据集合并完成...")
    return result
