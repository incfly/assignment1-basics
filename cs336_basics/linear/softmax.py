import torch
from torch import Tensor


def softmax(t: Tensor, dim: int) -> Tensor:
    shifted = t - torch.max(t, dim=dim, keepdim=True).values
    exp = torch.exp(shifted)
    return exp / torch.sum(exp, dim=dim, keepdim=True)
