import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        # not init first as trunc_normal below fill them anyway.
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    # Int[Tensor, " ..."], jaxtyping annotation, a Tensor data type is int, any dimension.
    # Float ... d_model is similar, any dimension but last one must be d_model.
    # TODO: learn more about 
    def forward(self, token_ids: Int[Tensor, " ..."]) -> Float[Tensor, " ... d_model"]:
        return self.weight[token_ids]
