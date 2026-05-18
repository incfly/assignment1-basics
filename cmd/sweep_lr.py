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


def _lr_name(lr: float) -> str:
    name = f"{lr:.8g}".replace(".", "p").replace("-", "m").replace("+", "")
    return f"lr-{name}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--lrs", type=float, nargs="+", default=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2])
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--warmup-iters", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--eval-iters", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume-from", default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open("rb") as f:
        base_config = tomllib.load(f)

    run_root = args.run_root or f"runs/lr-sweep-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    for lr in args.lrs:
        config = copy.deepcopy(base_config)
        opt_cfg = config.setdefault("optimizer", {})
        train_cfg = config.setdefault("training", {})

        opt_cfg["lr"] = lr
        opt_cfg["min_lr"] = lr * args.min_lr_ratio
        if args.max_iters is not None:
            train_cfg["max_iters"] = args.max_iters
            opt_cfg["cosine_cycle_iters"] = args.max_iters
        if args.warmup_iters is not None:
            opt_cfg["warmup_iters"] = args.warmup_iters
        if args.eval_every is not None:
            train_cfg["eval_every"] = args.eval_every
        if args.eval_iters is not None:
            train_cfg["eval_iters"] = args.eval_iters
        if args.device is not None:
            train_cfg["device"] = args.device

        train_cfg["checkpoint_every"] = args.checkpoint_every
        train_cfg["run_root"] = run_root
        train_cfg["run_name"] = _lr_name(lr)
        if args.resume_from:
            train_cfg["resume_from"] = args.resume_from
        else:
            train_cfg.pop("resume_from", None)

        with tempfile.TemporaryDirectory() as tmp_dir:
            trial_config = Path(tmp_dir) / "config.toml"
            _write_toml(config, trial_config)
            print(f"starting lr={lr:g} run_root={run_root}")
            try:
                train(config, trial_config)
            except RuntimeError as exc:
                print(f"failed lr={lr:g}: {exc}")


if __name__ == "__main__":
    main()
