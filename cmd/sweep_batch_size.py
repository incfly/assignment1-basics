from __future__ import annotations

import argparse
import copy
import sys
import tempfile
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import train  # noqa: E402


def _toml_value(value: Any) -> str:
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    return str(value)


def _write_toml(config: dict[str, Any], path: Path) -> None:
    lines = []
    for section, values in config.items():
        if not isinstance(values, dict):
            lines.append(f"{section} = {_toml_value(values)}")
            continue
        lines.append(f"[{section}]")
        for key, value in values.items():
            lines.append(f"{key} = {_toml_value(value)}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _batch_name(batch_size: int) -> str:
    return f"batch-{batch_size}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[4, 8, 16, 32, 64, 96])
    parser.add_argument("--target-tokens", type=int, default=24_576_000)
    parser.add_argument("--lr", type=float, default=0.0015)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-iters", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--eval-iters", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open("rb") as f:
        base_config = tomllib.load(f)

    model_cfg = base_config.get("model", {})
    context_length = int(model_cfg["context_length"])
    run_root = args.run_root or f"runs/batch-size-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    for batch_size in args.batch_sizes:
        if batch_size <= 0:
            raise ValueError(f"batch size must be positive: {batch_size}")
        tokens_per_iter = batch_size * context_length
        max_iters = args.target_tokens // tokens_per_iter
        if max_iters <= 0:
            raise ValueError(f"target_tokens={args.target_tokens} is too small for batch_size={batch_size}")
        if args.target_tokens % tokens_per_iter:
            actual_tokens = max_iters * tokens_per_iter
            print(
                f"batch_size={batch_size} does not divide target_tokens exactly; "
                f"using max_iters={max_iters} for {actual_tokens} tokens"
            )

        config = copy.deepcopy(base_config)
        opt_cfg = config.setdefault("optimizer", {})
        train_cfg = config.setdefault("training", {})

        opt_cfg["lr"] = args.lr
        opt_cfg["min_lr"] = args.lr * args.min_lr_ratio
        opt_cfg["warmup_iters"] = args.warmup_iters
        opt_cfg["cosine_cycle_iters"] = max_iters

        train_cfg["batch_size"] = batch_size
        train_cfg["max_iters"] = max_iters
        train_cfg["checkpoint_every"] = args.checkpoint_every
        train_cfg["run_root"] = run_root
        train_cfg["run_name"] = _batch_name(batch_size)
        train_cfg.pop("resume_from", None)
        if args.device is not None:
            train_cfg["device"] = args.device
        if args.eval_every is not None:
            train_cfg["eval_every"] = args.eval_every
        if args.eval_iters is not None:
            train_cfg["eval_iters"] = args.eval_iters

        with tempfile.TemporaryDirectory() as tmp_dir:
            trial_config = Path(tmp_dir) / "config.toml"
            _write_toml(config, trial_config)
            print(f"starting batch_size={batch_size} max_iters={max_iters} run_root={run_root}")
            try:
                train(config, trial_config)
            except RuntimeError as exc:
                print(f"failed batch_size={batch_size}: {exc}")


if __name__ == "__main__":
    main()
