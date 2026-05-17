import torch
from jaxtyping import Float, Int
from torch import Tensor


def softmax(t: Tensor, dim: int) -> Tensor:
    shifted = t - torch.max(t, dim=dim, keepdim=True).values
    exp = torch.exp(shifted)
    return exp / torch.sum(exp, dim=dim, keepdim=True)


def cross_entropy(
    inputs: Float[Tensor, " ... vocab_size"],
    targets: Int[Tensor, " ..."],
) -> Float[Tensor, ""]:
    # Shift logits so exp(...) stays finite, without changing softmax probabilities.
    shifted = inputs - torch.max(inputs, dim=-1, keepdim=True).values

    # Add a final singleton dimension so targets can index the vocab dimension.
    target_indices = targets.unsqueeze(-1)

    # Pick the correct-class logit along the final vocab dimension.
    target_logits = shifted.gather(
        dim=-1,
        index=target_indices,
    )

    # Remove the singleton vocab dimension added for gather.
    # [ [1] [2] ] -> [1, 2]
    target_logits = target_logits.squeeze(-1)

    # Compute log(sum(exp(logits))) across vocab for each batch position.
    log_sum_exp = torch.log(torch.sum(torch.exp(shifted), dim=-1))

    # Cross entropy is logsumexp(logits) minus the correct-class logit.
    # The target side cancels log(exp(target_logit)) to just target_logit.
    # The log_sum_exp side cannot cancel because log is outside a sum.
    return (log_sum_exp - target_logits).mean()


def scaled_dot_product_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / (d_k**0.5)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    return softmax(scores, dim=-1) @ V
