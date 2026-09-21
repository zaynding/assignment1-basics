import os
import typing
import torch
import torch.nn as nn
from torch.optim import Optimizer

"""
    一个完整的训练检查点（Checkpoint）必须保存能够无损恢复训练状态的三大要素：   
    模型权重：model.state_dict()。   
    优化器状态：optimizer.state_dict()。像 AdamW 这类包含一阶矩与二阶矩动量历史的优化器，如果不恢复其内部状态，恢复训练后的第一步会导致步长计算失准。   
    当前步数：当前的迭代次数 iteration，用于继续驱动余弦退火学习率调度器与数据游标。 
"""
def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
) -> None:
    """
    将模型状态、优化器状态以及当前迭代步数序列化保存至目标路径或文件对象。
    """
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: nn.Module,
    optimizer: Optimizer,
) -> int:
    """
    从目标路径或文件对象加载检查点，恢复模型和优化器状态，并返回保存的迭代步数。
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["iteration"])