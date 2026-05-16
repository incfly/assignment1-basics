import math
from collections.abc import Iterable

import torch


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate

    if it > cosine_cycle_iters:
        return min_learning_rate

    cosine_progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    cosine_weight = 0.5 * (1 + math.cos(cosine_progress * math.pi))
    return min_learning_rate + cosine_weight * (max_learning_rate - min_learning_rate)


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    parameters = list(parameters)
    grads = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not grads:
        return

    total_norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(grad, ord=2) for grad in grads]), ord=2)
    clip_coef = max_l2_norm / (total_norm + eps)
    clip_coef = torch.clamp(clip_coef, max=1.0)

    for grad in grads:
        grad.mul_(clip_coef)
