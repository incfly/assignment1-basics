import torch
import torch.nn as nn


class RoPE(nn.Module):
    cos: torch.Tensor
    sin: torch.Tensor

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("RoPE requires an even embedding dimension.")

        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        dim_indices = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)
        inv_freq = theta ** (-dim_indices / d_k)
        angles = positions[:, None] * inv_freq[None, :]

        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        token_positions = token_positions.to(device=x.device, dtype=torch.long)
        if torch.any(token_positions >= self.max_seq_len):
            raise ValueError("token_positions contains positions beyond max_seq_len.")

        cos = self.cos[token_positions].to(dtype=x.dtype)
        sin = self.sin[token_positions].to(dtype=x.dtype)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        rotated = torch.empty_like(x)
        rotated[..., 0::2] = x_even * cos - x_odd * sin
        rotated[..., 1::2] = x_even * sin + x_odd * cos
        return rotated
