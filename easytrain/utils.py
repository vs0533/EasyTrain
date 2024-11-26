"""
    工具集
"""

import os


# 判断不存在的路径是否是文件
def isfile(path: str):
    base = os.path.basename(path)
    suffix = os.path.splitext(base)[1]
    return True if suffix != "" and "." in suffix else False
