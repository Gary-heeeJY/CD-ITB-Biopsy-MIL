import torch
import numpy as np

def get_tokenizer():
    """返回一个简单占位 tokenizer（仅供导入使用）。

    注意：此为最小占位实现，若要进行文本相关的推理/生成，请替换为官方 tokenizer。
    """
    def _tokenizer(text):
        return text
    return _tokenizer


def tokenize(texts, max_length: int = 77):
    """把输入文本转换为整数 id 张量的占位实现。

    该实现返回全 0 的 LongTensor，形状为 (batch, max_length)，以保证依赖此函数的代码不会因导入失败而崩溃。
    如果需要真实的文本编码，请用官方 tokenizer 替换此文件或安装相应依赖。
    """
    if isinstance(texts, str):
        texts = [texts]
    batch_size = len(texts)
    arr = np.zeros((batch_size, max_length), dtype=np.int64)
    return torch.from_numpy(arr)
