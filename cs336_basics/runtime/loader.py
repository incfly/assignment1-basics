import numpy as np
import numpy.typing as npt
import torch


def get_batch(
    dataset: npt.NDArray[np.integer],
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample input token windows and next-token targets.

    Args:
        dataset: 1D numpy array of token IDs.
        batch_size: Number of sampled sequences.
        context_length: Number of input tokens per sequence.
        device: PyTorch device string, e.g. "cpu" or "cuda:0".

    Returns:
        Two LongTensors on `device`, both shaped `(batch_size, context_length)`.
        The first tensor is input token IDs.
        The second tensor is the same windows shifted one token to the right.

    Example:
        dataset = np.arange(10)
        batch_size = 2
        context_length = 3

        If the sampled start indices are [2, 5], then:

        x = tensor([
            [2, 3, 4],
            [5, 6, 7],
        ])

        y = tensor([
            [3, 4, 5],
            [6, 7, 8],
        ])
    """
    xs = []
    ys = []
    for i in range(batch_size):
        xs.append(dataset[i: i+context_length])
        ys.append(dataset[i+1: i+1+context_length])
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long) 
