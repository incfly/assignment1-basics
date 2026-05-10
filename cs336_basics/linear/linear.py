import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


class Linear(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        sigma = (2 / (in_features + out_features)) ** 0.5
        # I used to use in, out. AI: wrong, just not convention :(
        # But `layer.weight.data = weights` in the adapter does make sense.

        # TODO: this init is from the slides. Check later how it helps with training.
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=sigma, a=-3 * sigma, b=3 * sigma)

    def forward(
        self,
        x: Float[Tensor, " ... d_in"],
    ) -> Float[Tensor, " ... d_out"]:
        return x @ self.weight.T
