import argparse
import csv
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import numpy.typing as npt
import psutil
import torch

DEFAULT_TOKENIZED_PATH = Path("../cs336-data/tinystory/TinyStories-train.txt-tokenized.bin")
DEFAULT_MEMORY_CSV = Path("runtime-memory.csv")

PHASES = ("startup", "mmap_open", "load_copy", "iterate", "done")


def get_batch(
    dataset: npt.NDArray[np.integer],
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample input token windows and next-token targets.

    Args:
        dataset: 1D numpy array of token IDs.
        batch_size: Number of sampled sequences.
        context_length: Number of input tokens per sequence.
        device: PyTorch device string, e.g. "cpu" or "cuda:0".

    Returns:
        Two LongTensors on `device`, both shaped `(batch_size, context_length)`.
        The first tensor is input token IDs.
        The second tensor is the same windows shifted one token to the right.

    Example:
        dataset = np.arange(10)
        batch_size = 2
        context_length = 3

        If the sampled start indices are [2, 5], then:

        x = tensor([
            [2, 3, 4],
            [5, 6, 7],
        ])

        y = tensor([
            [3, 4, 5],
            [6, 7, 8],
        ])
    """
    max_start = len(dataset) - context_length
    starts = np.random.randint(0, max_start, size=batch_size)

    xs = []
    ys = []
    for start in starts:
        xs.append(dataset[start : start + context_length])
        ys.append(dataset[start + 1 : start + context_length + 1])

    return (
        torch.tensor(np.array(xs), dtype=torch.long, device=device),
        torch.tensor(np.array(ys), dtype=torch.long, device=device),
    )


# what is `np.memmap` abstaction? 
# bundled mmap formed array. np interpret as list of integer.
def open_token_memmap(path: str | Path, dtype: str = "uint16") -> np.memmap:
    """Open a raw token-id binary file as a 1D numpy memmap."""
    path = Path(path)
    itemsize = np.dtype(dtype).itemsize
    size_bytes = path.stat().st_size
    if size_bytes % itemsize != 0:
        raise ValueError(f"{path} has {size_bytes} bytes, not divisible by dtype {dtype}")
    return np.memmap(path, dtype=dtype, mode="r", shape=(size_bytes // itemsize,))


def load_token_file(path: str | Path, dtype: str = "uint16") -> npt.NDArray[np.integer]:
    """Read a raw token-id binary file into a normal numpy array."""
    return np.fromfile(Path(path), dtype=dtype)


def _memory_row(process: psutil.Process) -> dict[str, int | float]:
    info = process.memory_info()
    try:
        full_info = process.memory_full_info()
    except (psutil.AccessDenied, psutil.Error):
        full_info = info
    return {
        "rss_bytes": int(info.rss),
        "vms_bytes": int(info.vms),
        "shared_bytes": int(getattr(info, "shared", 0)),
        "uss_bytes": int(getattr(full_info, "uss", 0)),
        "pss_bytes": int(getattr(full_info, "pss", 0)),
        "cpu_percent": float(process.cpu_percent(interval=None)),
    }


def _memory_sampler(
    pid: int,
    csv_path: str,
    interval_s: float,
    stop_event: mp.Event,
    ready_event: mp.Event,
    phase_value: mp.Value,
    run_id: str,
    mode: str,
) -> None:
    process = psutil.Process(pid)
    process.cpu_percent(interval=None)
    started = time.monotonic()
    fields = [
        "run_id",
        "mode",
        "elapsed_s",
        "phase",
        "rss_bytes",
        "vms_bytes",
        "shared_bytes",
        "uss_bytes",
        "pss_bytes",
        "cpu_percent",
    ]

    csv_file = Path(csv_path)
    needs_header = not csv_file.exists() or csv_file.stat().st_size == 0
    with csv_file.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if needs_header:
            writer.writeheader()
        ready_event.set()
        while True:
            phase_index = int(phase_value.value)
            writer.writerow(
                {
                    "run_id": run_id,
                    "mode": mode,
                    "elapsed_s": f"{time.monotonic() - started:.6f}",
                    "phase": PHASES[phase_index],
                    **_memory_row(process),
                }
            )
            f.flush()
            if stop_event.wait(interval_s):
                break


def _let_sampler_tick(interval_s: float) -> None:
    time.sleep(min(max(interval_s, 0.01), 0.25))


def run_memory_probe(
    input_path: str | Path = DEFAULT_TOKENIZED_PATH,
    csv_path: str | Path = DEFAULT_MEMORY_CSV,
    mode: str = "mmap",
    dtype: str = "uint16",
    interval_s: float = 0.25,
    iterations: int = 100,
    batch_size: int = 64,
    context_length: int = 256,
    device: str = "cpu",
) -> int:
    """Run a tiny mmap/load workload while a subprocess samples process memory."""
    if mode not in {"mmap", "load"}:
        raise ValueError(f"mode must be 'mmap' or 'load', got {mode!r}")

    ctx = mp.get_context("spawn")
    stop_event = ctx.Event()
    ready_event = ctx.Event()
    phase_value = ctx.Value("i", 0)
    run_id = f"{mode}-{int(time.time())}-{os.getpid()}"
    sampler = ctx.Process(
        target=_memory_sampler,
        args=(os.getpid(), str(csv_path), interval_s, stop_event, ready_event, phase_value, run_id, mode),
        daemon=True,
    )

    checksum = 0
    sampler.start()
    ready_event.wait(timeout=max(2.0, interval_s * 4))
    try:
        if mode == "mmap":
            phase_value.value = PHASES.index("mmap_open")
            _let_sampler_tick(interval_s)
            dataset = open_token_memmap(input_path, dtype=dtype)
            checksum += int(dataset[0])
        else:
            phase_value.value = PHASES.index("load_copy")
            _let_sampler_tick(interval_s)
            dataset = load_token_file(input_path, dtype=dtype)
            checksum += int(dataset[0]) + int(dataset[-1])

        phase_value.value = PHASES.index("iterate")
        _let_sampler_tick(interval_s)
        for _ in range(iterations):
            x, y = get_batch(dataset, batch_size, context_length, device)
            checksum += int(x[0, 0].item()) + int(y[-1, -1].item())

        phase_value.value = PHASES.index("done")
        return checksum
    finally:
        time.sleep(interval_s)
        stop_event.set()
        sampler.join(timeout=max(2.0, interval_s * 4))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe mmap token loading memory usage.")
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_TOKENIZED_PATH)
    parser.add_argument("--csv", type=Path, default=DEFAULT_MEMORY_CSV)
    parser.add_argument("--mode", choices=("mmap", "load", "both"), default="both")
    parser.add_argument("--dtype", default="uint16")
    parser.add_argument("--interval", type=float, default=0.25)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    modes = ("mmap", "load") if args.mode == "both" else (args.mode,)
    for mode in modes:
        checksum = run_memory_probe(
            input_path=args.input,
            csv_path=args.csv,
            mode=mode,
            dtype=args.dtype,
            interval_s=args.interval,
            iterations=args.iterations,
            batch_size=args.batch_size,
            context_length=args.context_length,
            device=args.device,
        )
        print(f"{mode}: wrote {args.csv} checksum={checksum}")


if __name__ == "__main__":
    main()
