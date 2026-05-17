from pathlib import Path

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
    max_start = len(dataset) - context_length
    starts = np.random.randint(0, max_start, size=batch_size)

    xs = []
    ys = []
    for start in starts:
        xs.append(dataset[start : start + context_length])
        ys.append(dataset[start + 1 : start + context_length + 1])

    return (
        torch.tensor(np.array(xs), dtype=torch.long, device=device),
        torch.tensor(np.array(ys), dtype=torch.long, device=device),
    )


def open_token_memmap(path: str | Path, dtype: str = "uint16") -> np.memmap:
    """Open a raw token-id binary file as a 1D numpy memmap."""
    path = Path(path)
    itemsize = np.dtype(dtype).itemsize
    size_bytes = path.stat().st_size
    if size_bytes % itemsize != 0:
        raise ValueError(f"{path} has {size_bytes} bytes, not divisible by dtype {dtype}")
    return np.memmap(path, dtype=dtype, mode="r", shape=(size_bytes // itemsize,))
