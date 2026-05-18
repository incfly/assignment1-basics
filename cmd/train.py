from __future__ import annotations

import argparse
import json
import shutil
import time
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cs336_basics.checkpoint import load_checkpoint, save_checkpoint
from cs336_basics.model import TransformerLM, cross_entropy
from cs336_basics.model.adam import AdamW
from cs336_basics.optim import get_lr_cosine_schedule, gradient_clipping
from cs336_basics.runtime.device import default_device
from cs336_basics.runtime.loader import get_batch, open_token_memmap


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


def _set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def _make_run_dir(train_cfg: dict[str, Any], config_path: Path) -> Path:
    run_root = Path(train_cfg.get("run_root", "runs"))
    run_name = train_cfg.get("run_name") or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = run_root / str(run_name)
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(config_path, run_dir / "config.toml")
    return run_dir


def _write_metric(metrics_path: Path, row: dict[str, Any]) -> None:
    with metrics_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


@torch.no_grad()
def _evaluate(
    model: TransformerLM,
    valid_data,
    batch_size: int,
    context_length: int,
    device: str,
    eval_iters: int,
) -> float:
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(valid_data, batch_size, context_length, device)
        losses.append(float(cross_entropy(model(x), y).item()))
    model.train()
    return sum(losses) / len(losses)


def train(config: dict[str, Any], config_path: Path) -> None:
    data_cfg = _section(config, "data")
    model_cfg = _section(config, "model")
    opt_cfg = _section(config, "optimizer")
    train_cfg = _section(config, "training")

    device = str(train_cfg.get("device") or default_device())
    torch.manual_seed(int(train_cfg.get("seed", 0)))
    np.random.seed(int(train_cfg.get("seed", 0)))

    train_data = open_token_memmap(data_cfg["train_path"], dtype=str(data_cfg.get("dtype", "uint16")))
    valid_path = data_cfg.get("valid_path")
    valid_data = open_token_memmap(valid_path, dtype=str(data_cfg.get("dtype", "uint16"))) if valid_path else None

    model = _build_model(config, device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(opt_cfg["lr"]),
        weight_decay=float(opt_cfg.get("weight_decay", 0.0)),
        betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
        eps=float(opt_cfg.get("eps", 1e-8)),
    )

    iteration = 0
    resume_from = train_cfg.get("resume_from")
    if resume_from:
        iteration = load_checkpoint(resume_from, model, optimizer, map_location=device)
        print(f"resumed iteration={iteration} from {resume_from}")

    batch_size = int(train_cfg["batch_size"])
    context_length = int(model_cfg["context_length"])
    max_iters = int(train_cfg["max_iters"])
    log_every = int(train_cfg.get("log_every", 10))
    eval_every = int(train_cfg.get("eval_every", 100))
    eval_iters = int(train_cfg.get("eval_iters", 10))
    checkpoint_every = int(train_cfg.get("checkpoint_every", 0))
    run_dir = _make_run_dir(train_cfg, config_path)
    metrics_path = run_dir / "metrics.jsonl"
    checkpoint_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints"))
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = run_dir / checkpoint_dir
    max_grad_norm = train_cfg.get("max_grad_norm")
    tokens_per_iteration = batch_size * context_length
    print(f"run_dir={run_dir}")

    start = time.time()
    model.train()
    while iteration < max_iters:
        lr = get_lr_cosine_schedule(
            it=iteration,
            max_learning_rate=float(opt_cfg["lr"]),
            min_learning_rate=float(opt_cfg.get("min_lr", 0.0)),
            warmup_iters=int(opt_cfg.get("warmup_iters", 0)),
            cosine_cycle_iters=int(opt_cfg.get("cosine_cycle_iters", max_iters)),
        )
        _set_lr(optimizer, lr)

        x, y = get_batch(train_data, batch_size, context_length, device)
        optimizer.zero_grad()
        loss = cross_entropy(model(x), y)
        loss.backward()
        if max_grad_norm is not None:
            gradient_clipping(model.parameters(), float(max_grad_norm))
        optimizer.step()
        iteration += 1

        if iteration % log_every == 0:
            elapsed = time.time() - start
            iter_sec = elapsed / max(iteration, 1)
            tokens_seen = iteration * tokens_per_iteration
            print(f"iter={iteration} train_loss={loss.item():.4f} lr={lr:.6g} elapsed={elapsed:.1f}s")
            _write_metric(
                metrics_path,
                {
                    "iter": iteration,
                    "split": "train",
                    "loss": float(loss.item()),
                    "lr": lr,
                    "elapsed_sec": elapsed,
                    "iter_sec": iter_sec,
                    "tokens_seen": tokens_seen,
                    "batch_size": batch_size,
                    "context_length": context_length,
                    "device": device,
                },
            )

        if valid_data is not None and iteration % eval_every == 0:
            elapsed = time.time() - start
            val_loss = _evaluate(model, valid_data, batch_size, context_length, device, eval_iters)
            print(f"iter={iteration} val_loss={val_loss:.4f}")
            _write_metric(
                metrics_path,
                {
                    "iter": iteration,
                    "split": "valid",
                    "loss": val_loss,
                    "lr": lr,
                    "elapsed_sec": elapsed,
                    "iter_sec": elapsed / max(iteration, 1),
                    "tokens_seen": iteration * tokens_per_iteration,
                    "batch_size": batch_size,
                    "context_length": context_length,
                    "eval_iters": eval_iters,
                    "device": device,
                },
            )

        if checkpoint_every and iteration % checkpoint_every == 0:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            path = checkpoint_dir / f"ckpt_{iteration}.pt"
            save_checkpoint(model, optimizer, iteration, path)
            print(f"saved checkpoint={path}")
            _write_metric(
                metrics_path,
                {
                    "iter": iteration,
                    "event": "checkpoint",
                    "checkpoint": str(path),
                    "elapsed_sec": time.time() - start,
                },
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open("rb") as f:
        config = tomllib.load(f)
    train(config, config_path)


if __name__ == "__main__":
    main()
