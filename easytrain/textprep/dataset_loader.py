"""
    负责加载数据集，支持流式加载
"""

from datasets import load_dataset, concatenate_datasets


def load_dataset_custom(data_files, streaming=True):
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

        dataset = load_dataset(
            file_ext, data_files={name: file_pattern}, streaming=streaming
        )

        # 添加到数据集列表
        datasets.append(dataset[name])

    # 如果只有一个数据集直接返回，否则合并
    if len(datasets) == 1:
        return datasets[0]
    return concatenate_datasets(datasets)
