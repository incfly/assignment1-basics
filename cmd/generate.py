from __future__ import annotations

import argparse
import tomllib
from pathlib import Path
from typing import Any

import torch

from cs336_basics.bpe.encode import Encoder
from cs336_basics.decoder import generate
from cs336_basics.model import TransformerLM
from cs336_basics.runtime.device import default_device


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"[{name}] must be a table")
    return value


def _build_model(config: dict[str, Any], device: str) -> TransformerLM:
    model_cfg = _section(config, "model")
    return TransformerLM(
        vocab_size=int(model_cfg["vocab_size"]),
        context_length=int(model_cfg["context_length"]),
        d_model=int(model_cfg["d_model"]),
        num_layers=int(model_cfg["num_layers"]),
        num_heads=int(model_cfg["num_heads"]),
        d_ff=int(model_cfg["d_ff"]),
        rope_theta=float(model_cfg.get("rope_theta", 10_000.0)),
        device=torch.device(device),
    )


def _resolve_config_relative(path: str, config_path: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else config_path.parent / p


def _resolve_existing_path(path: str, config_path: Path) -> Path:
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p
    config_relative = config_path.parent / p
    return config_relative if config_relative.exists() else p


def _resolve_override_path(path: str, config_path: Path, from_cli: bool) -> Path:
    if from_cli:
        return _resolve_existing_path(path, config_path)
    return _resolve_config_relative(path, config_path)


def _infer_tokenizer_files(train_path: Path) -> tuple[Path, Path]:
    base = train_path
    if base.name.endswith("-tokenized.bin"):
        base = base.with_name(base.name[: -len("-tokenized.bin")])
    return base.with_name(base.name + "-vocab.json"), base.with_name(base.name + "-merge.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--prompt")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open("rb") as f:
        config = tomllib.load(f)

    data_cfg = _section(config, "data")
    model_cfg = _section(config, "model")
    gen_cfg = _section(config, "generation")
    checkpoint = args.checkpoint if args.checkpoint is not None else gen_cfg.get("checkpoint")
    if checkpoint is None:
        raise ValueError("[generation].checkpoint is required")

    device = str(args.device if args.device is not None else gen_cfg.get("device") or default_device())
    seed = args.seed if args.seed is not None else gen_cfg.get("seed")
    if seed is not None:
        torch.manual_seed(int(seed))

    model = _build_model(config, device)
    checkpoint_path = _resolve_override_path(str(checkpoint), config_path, args.checkpoint is not None)
    payload = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(payload["model"] if isinstance(payload, dict) and "model" in payload else payload)
    model.to(device)

    train_path = _resolve_existing_path(str(data_cfg["train_path"]), config_path)
    vocab_path, merges_path = _infer_tokenizer_files(train_path)
    tokenizer = Encoder.from_files(str(vocab_path), str(merges_path), special_tokens=["<|endoftext|>"])

    prompt = str(args.prompt if args.prompt is not None else gen_cfg.get("prompt", ""))
    prompt_ids = tokenizer.encode(prompt)
    output_ids = generate(
        model=model,
        prompt_ids=prompt_ids,
        max_new_tokens=int(args.max_new_tokens if args.max_new_tokens is not None else gen_cfg.get("max_new_tokens", 100)),
        context_length=int(model_cfg["context_length"]),
        eos_id=tokenizer.special_token_to_id.get("<|endoftext|>"),
        temperature=float(args.temperature if args.temperature is not None else gen_cfg.get("temperature", 1.0)),
        top_p=float(args.top_p if args.top_p is not None else gen_cfg.get("top_p", 1.0)),
        device=device,
    )
    print(tokenizer.decode(output_ids))


if __name__ == "__main__":
    main()
