from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from cs336_basics.model.linear import Linear
from cs336_basics.model.rope import RoPE
from cs336_basics.model.softmax import scaled_dot_product_attention


class MultiheadSelfAttention(nn.Module):
    # This module owns the real q/k/v/o parameters.
    # The adapter passes reference weights from tests and copies them here.
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = None
        if max_seq_len is not None and theta is not None:
            self.rope = RoPE(theta=theta, d_k=self.d_head, max_seq_len=max_seq_len, device=device)

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        *batch_dims, sequence_length, _ = x.shape

        # Project first, then split the projected q/k/v features into heads.
        # This is the packed version of the modular Head code from part6.
        q = self._split_heads(self.q_proj(x), batch_dims, sequence_length)
        k = self._split_heads(self.k_proj(x), batch_dims, sequence_length)
        v = self._split_heads(self.v_proj(x), batch_dims, sequence_length)

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(sequence_length, device=x.device)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        mask = torch.tril(
            torch.ones(sequence_length, sequence_length, device=x.device, dtype=torch.bool)
        )
        attn = scaled_dot_product_attention(q, k, v, mask)
        # Concatenate heads, then use output_proj to mix them back into the model stream.
        attn = attn.transpose(-3, -2).contiguous()
        attn = attn.reshape(*batch_dims, sequence_length, self.d_model)
        # We choose to attn dim same as the embedding. somehow for the efficiency? not fully get.
        # Could be different in theory as we can project from x -> y arbitrarily.
        return self.output_proj(attn)

    def _split_heads(self, x: Tensor, batch_dims: list[int], sequence_length: int) -> Tensor:
        x = x.reshape(*batch_dims, sequence_length, self.num_heads, self.d_head)
        return x.transpose(-3, -2)
