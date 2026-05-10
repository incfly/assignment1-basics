import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


class Linear(nn.Module):

    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        # I used to use in, out. AI: wrong, just not convention :( 
        # But `layer.weight.data = weights` in the adapter does make sense.
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features, device=device, dtype=dtype)
        )

    def forward(
        self,
        x: Float[Tensor, " ... d_in"],
    ) -> Float[Tensor, " ... d_out"]:
        return x @ self.weight.T
