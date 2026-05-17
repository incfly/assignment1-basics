import torch
import torch.nn as nn

from cs336_basics.model.linear import Linear

# This is feedforward layer!
# vanila: x -> W1 -> ReLU -> w2, ReLU for non linear part. spicy up the model.
# otherwise Wx = W1(W2(x)). not adding power.
# 
# x -> W1 -> activation SiLU, _with gate, using W3_ -> W2 linear layer
# Why this works?
# "We offer no explanation as to why these architectures seem to work; we attribute their success, as
# all else, to divine benevolence"
# What does better mean? AI: loss. Maybe lots of ablation experiments...
class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if d_ff is None:
            d_ff = self._default_d_ff(d_model)

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    @staticmethod
    def _default_d_ff(d_model: int) -> int:
        multiple_of = 64
        hidden_dim = int(8 * d_model / 3)
        return multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1_x = self.w1(x)
        silu = w1_x * torch.sigmoid(w1_x)
        return self.w2(silu * self.w3(x))
