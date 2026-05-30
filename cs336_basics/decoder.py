from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor


def sample_next_token(logits: Tensor, temperature: float = 1.0, top_p: float = 1.0) -> int:
    if logits.ndim != 1:
        raise ValueError(f"logits must be 1D, got shape={tuple(logits.shape)}")
    if top_p <= 0.0 or top_p > 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")

    if temperature <= 0.0:
        return int(torch.argmax(logits).item())

    probs = torch.softmax(logits / temperature, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_ids = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        keep = cumulative <= top_p
        keep[0] = True
        over_threshold = torch.nonzero(cumulative >= top_p)
        if over_threshold.numel() > 0:
            keep[: int(over_threshold[0].item()) + 1] = True

        sorted_probs = sorted_probs * keep
        sorted_probs = sorted_probs / sorted_probs.sum()
        sample = torch.multinomial(sorted_probs, num_samples=1)
        return int(sorted_ids[sample].item())

    return int(torch.multinomial(probs, num_samples=1).item())


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    context_length: int,
    eos_id: int | None = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
    device: str | torch.device | None = None,
) -> list[int]:
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be non-negative, got {max_new_tokens}")
    if context_length <= 0:
        raise ValueError(f"context_length must be positive, got {context_length}")

    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)

    model.eval()
    ids = list(prompt_ids)
    drop_initial_eos = False
    if not ids:
        if eos_id is None:
            raise ValueError("empty prompts require eos_id so generation has a first token")
        ids = [eos_id]
        drop_initial_eos = True

    for _ in range(max_new_tokens):
        context = ids[-context_length:]
        x = torch.tensor([context], dtype=torch.long, device=device)
        logits = model(x)[0, -1]
        next_id = sample_next_token(logits, temperature=temperature, top_p=top_p)
        if eos_id is not None and next_id == eos_id:
            break
        ids.append(next_id)

    return ids[1:] if drop_initial_eos else ids
