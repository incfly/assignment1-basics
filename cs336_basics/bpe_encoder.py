import argparse
import json
import logging
import os
from pathlib import Path
import regex as re
import time
from collections import OrderedDict
from collections.abc import Iterable, Iterator
from concurrent.futures import Future, ProcessPoolExecutor, wait, FIRST_COMPLETED
from typing import TYPE_CHECKING

from cs336_basics.pretoken import PY_PRETOKEN_PATTERN, RegexMode

if TYPE_CHECKING:
    from cs336_basics.bpe_merge import ExternalMerges, ExternalVocab

LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False

_WORKER_ENCODER: "Encoder | None" = None


def _load_vocab(vocab_filepath: str) -> dict[int, bytes]:
    payload = json.loads(Path(vocab_filepath).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return {int(row["token_id"]): bytes.fromhex(row["hex"]) for row in payload}
    raise ValueError(f"unsupported vocab file format: {vocab_filepath}")


def _load_merges(merges_filepath: str) -> list[tuple[bytes, bytes]]:
    text = Path(merges_filepath).read_text(encoding="utf-8")
    payload = json.loads(text)
    if isinstance(payload, list):
        return [
            (bytes.fromhex(row["left_hex"]), bytes.fromhex(row["right_hex"]))
            for row in payload
        ]
    raise ValueError(f"unsupported merges file format: {merges_filepath}")


def _parse_size(s: str) -> int:
    units = {"k": 10**3, "m": 10**6, "g": 10**9, "ki": 2**10, "mi": 2**20, "gi": 2**30}
    s = s.strip().lower()
    for suffix, mult in sorted(units.items(), key=lambda kv: -len(kv[0])):
        if s.endswith(suffix):
            return int(float(s[: -len(suffix)]) * mult)
    return int(s)


def _default_data_root() -> Path:
    vm_root = Path("/mnt/disks/openweb-data/cs336-data")
    if vm_root.exists():
        return vm_root
    return Path(os.environ.get("DATA_ROOT", Path(__file__).resolve().parents[1] / "../cs336-data")).resolve()


def _default_encoder_workers() -> int:
    return max(1, os.cpu_count() or 1)


def _resolve_input_path(input_arg: str | None, root: Path, default_name: str = "owt_train.txt") -> Path:
    if input_arg is None:
        return root / default_name
    path = Path(input_arg)
    return path if path.is_absolute() else root / path


def _resolve_tokenizer_files(
    input_path: Path,
    vocab_arg: str | None,
    merges_arg: str | None,
) -> tuple[Path, Path]:
    vocab_path = Path(vocab_arg) if vocab_arg else input_path.with_name(input_path.name + "-vocab.json")
    merges_path = Path(merges_arg) if merges_arg else input_path.with_name(input_path.name + "-merge.json")
    return vocab_path, merges_path


class Encoder:
    def __init__(
        self,
        vocab: "ExternalVocab",
        merge: "ExternalMerges",
        special_tokens: list[str] | None = None,
        regex_mode: RegexMode = "cpp",
        pretoken_cache_size: int = 500_000,
    ):
        self.vocab = vocab
        self.merge = merge
        self.regex_mode = regex_mode
        self.pretoken_cache_size = pretoken_cache_size
        self.pretoken_cache: OrderedDict[str, tuple[int, ...]] = OrderedDict()
        self.pretoken_cache_hits = 0
        self.pretoken_cache_misses = 0
        self.special_tokens = special_tokens or []
        existing_tokens = set(vocab.values())
        for special_token in self.special_tokens:
            token_bytes = special_token.encode("utf-8")
            if token_bytes not in existing_tokens:
                vocab[len(vocab)] = token_bytes
                existing_tokens.add(token_bytes)
        self.token_to_id = {token: token_id for token_id, token in vocab.items()}
        self.special_token_to_id = {
            special_token: self.token_to_id[special_token.encode("utf-8")]
            for special_token in self.special_tokens
        }
        self.merge_rank = {pair: rank for rank, pair in enumerate(merge)}
        self.special_pattern = (
            re.compile("|".join(re.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True)))
            if self.special_tokens
            else None
        )
        self.pretoken_pattern = re.compile(PY_PRETOKEN_PATTERN) if regex_mode == "py" else None
        self.cpp_pretokenize = None
        if regex_mode == "cpp":
            try:
                from re2_demo import pretokenize as re2_pretokenize
            except ImportError as exc:
                raise RuntimeError(
                    "regex_mode='cpp' requires building re2_demo first via "
                    "`./scripts/bootstrap_re2_linux.sh` and `PYTHON_BIN=python3 ./scripts/build_re2_demo_linux.sh`."
                ) from exc
            self.cpp_pretokenize = re2_pretokenize
        elif regex_mode != "py":
            raise ValueError(f"unsupported regex_mode={regex_mode!r}")

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
        pretoken_cache_size: int = 500_000,
    ) -> "Encoder":
        vocab = _load_vocab(vocab_filepath)
        return cls(
            vocab,
            _load_merges(merges_filepath),
            special_tokens=special_tokens,
            pretoken_cache_size=pretoken_cache_size,
        )

    def _encode_pretoken(self, pretoken: str) -> tuple[int, ...]:
        if self.pretoken_cache_size <= 0:
            return tuple(self.encode_pretoken_bytes(pretoken.encode("utf-8")))

        cached = self.pretoken_cache.get(pretoken)
        if cached is not None:
            self.pretoken_cache_hits += 1
            self.pretoken_cache.move_to_end(pretoken)
            return cached

        self.pretoken_cache_misses += 1
        encoded = tuple(self.encode_pretoken_bytes(pretoken.encode("utf-8")))
        self.pretoken_cache[pretoken] = encoded
        if len(self.pretoken_cache) > self.pretoken_cache_size:
            self.pretoken_cache.popitem(last=False)
        return encoded

    def cache_stats(self) -> dict[str, int | float]:
        total = self.pretoken_cache_hits + self.pretoken_cache_misses
        return {
            "pretoken_cache_size": self.pretoken_cache_size,
            "pretoken_cache_entries": len(self.pretoken_cache),
            "pretoken_cache_hits": self.pretoken_cache_hits,
            "pretoken_cache_misses": self.pretoken_cache_misses,
            "pretoken_cache_hit_rate": self.pretoken_cache_hits / total if total else 0.0,
        }

    # List of the merge is a long list. 
    # Token itself is short.
    # Naive approach is to iterate O(n)
    # better approach might be build up the pair of the token of current word.
    # Then look up from the merge list. get all the index and take the smallest one.
    # O(len(pre-token word)) per merge round.
    def encode_pretoken_bytes(self, b: bytes) -> list[int]:
        parts = [bytes([byte]) for byte in b]
        if not parts:
            return []

        while len(parts) > 1:
            best_rank = None
            best_pair = None
            for i in range(len(parts) - 1):
                pair = (parts[i], parts[i + 1])
                rank = self.merge_rank.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_pair = pair
            if best_pair is None:
                break

            merged = []
            i = 0
            while i < len(parts):
                if i < len(parts) - 1 and (parts[i], parts[i + 1]) == best_pair:
                    merged.append(parts[i] + parts[i + 1])
                    i += 2
                else:
                    merged.append(parts[i])
                    i += 1
            parts = merged

        return [self.token_to_id[token] for token in parts]

    def _encode_text_without_special(self, text: str) -> list[int]:
        ids: list[int] = []
        if self.regex_mode == "cpp":
            assert self.cpp_pretokenize is not None
            pretokens = self.cpp_pretokenize(text)
        else:
            assert self.pretoken_pattern is not None
            pretokens = self.pretoken_pattern.findall(text)
        for pretoken in pretokens:
            ids.extend(self._encode_pretoken(pretoken))
        return ids

    def encode(self, text: str) -> list[int]:
        if self.special_pattern is None:
            return self._encode_text_without_special(text)

        ids: list[int] = []
        pos = 0
        for match in self.special_pattern.finditer(text):
            ids.extend(self._encode_text_without_special(text[pos:match.start()]))
            ids.append(self.special_token_to_id[match.group(0)])
            pos = match.end()
        ids.extend(self._encode_text_without_special(text[pos:]))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[token_id] for token_id in ids).decode("utf-8", errors="replace")


def _init_encode_worker(
    vocab_path: str,
    merges_path: str,
    special_tokens: list[str],
    pretoken_cache_size: int,
) -> None:
    global _WORKER_ENCODER
    _WORKER_ENCODER = Encoder.from_files(
        vocab_path,
        merges_path,
        special_tokens=special_tokens,
        pretoken_cache_size=pretoken_cache_size,
    )


def _worker_cache_delta(before_hits: int, before_misses: int) -> dict[str, int | float]:
    assert _WORKER_ENCODER is not None
    stats = _WORKER_ENCODER.cache_stats()
    return {
        "worker_pid": os.getpid(),
        "pretoken_cache_size": stats["pretoken_cache_size"],
        "pretoken_cache_entries": stats["pretoken_cache_entries"],
        "pretoken_cache_hits": int(stats["pretoken_cache_hits"]) - before_hits,
        "pretoken_cache_misses": int(stats["pretoken_cache_misses"]) - before_misses,
    }


def _encode_chunk_to_part(task: tuple[int, bytes, str]) -> tuple[int, int, int, str, dict[str, int | float]]:
    idx, data, part_path = task
    assert _WORKER_ENCODER is not None
    import numpy as np

    before_hits = _WORKER_ENCODER.pretoken_cache_hits
    before_misses = _WORKER_ENCODER.pretoken_cache_misses
    ids = _WORKER_ENCODER.encode(data.decode("utf-8", errors="replace"))
    np.asarray(ids, dtype=np.uint16).tofile(part_path)
    return idx, len(data), len(ids), part_path, _worker_cache_delta(before_hits, before_misses)


def _read_chunk_tasks(input_path: Path, target_bytes: int, chunk_bytes: int) -> Iterator[tuple[int, bytes]]:
    separator = b"<|endoftext|>"
    seen = 0
    idx = 0
    carry = b""
    current_parts: list[bytes] = []
    current_size = 0

    def flush_current() -> bytes:
        nonlocal current_parts, current_size
        block = b"".join(current_parts)
        current_parts = []
        current_size = 0
        return block

    with input_path.open("rb") as f:
        while seen < target_bytes:
            data = f.read(min(chunk_bytes, target_bytes - seen))
            if not data:
                break
            seen += len(data)
            pieces = (carry + data).split(separator)
            carry = pieces[-1]
            for piece in pieces[:-1]:
                segment = piece + separator
                if current_size and current_size + len(segment) > chunk_bytes:
                    yield idx, flush_current()
                    idx += 1
                current_parts.append(segment)
                current_size += len(segment)

    if carry:
        if current_size and current_size + len(carry) > chunk_bytes:
            yield idx, flush_current()
            idx += 1
        current_parts.append(carry)
        current_size += len(carry)
    if current_parts:
        yield idx, flush_current()


def _merge_cache_stats(stats: Iterable[dict[str, int | float]], cache_size: int) -> dict[str, int | float]:
    hits = 0
    misses = 0
    worker_entries: dict[int, int] = {}
    for row in stats:
        hits += int(row["pretoken_cache_hits"])
        misses += int(row["pretoken_cache_misses"])
        worker_pid = int(row.get("worker_pid", 0))
        worker_entries[worker_pid] = max(worker_entries.get(worker_pid, 0), int(row["pretoken_cache_entries"]))
    total = hits + misses
    return {
        "pretoken_cache_size": cache_size,
        "pretoken_cache_entries": sum(worker_entries.values()),
        "pretoken_cache_hits": hits,
        "pretoken_cache_misses": misses,
        "pretoken_cache_hit_rate": hits / total if total else 0.0,
    }


def _submit_next_part(
    pool: ProcessPoolExecutor,
    task_iter: Iterator[tuple[int, bytes]],
    part_dir: Path,
    pending: set[Future[tuple[int, int, int, str, dict[str, int | float]]]],
) -> bool:
    try:
        idx, data = next(task_iter)
    except StopIteration:
        return False
    pending.add(pool.submit(_encode_chunk_to_part, (idx, data, str(part_dir / f"part-{idx:08d}.bin"))))
    return True


def tokenize_file(
    input_path: Path,
    output_path: Path,
    vocab_path: Path,
    merges_path: Path,
    target_bytes: int,
    chunk_bytes: int,
    encoder_workers: int,
    pretoken_cache_size: int,
) -> dict[str, float | int | str]:
    # File-level "tokenization" is encoding raw text into uint16 token IDs for NN training.
    import numpy as np

    start = time.perf_counter()
    if encoder_workers <= 1:
        tokenizer = Encoder.from_files(
            str(vocab_path),
            str(merges_path),
            special_tokens=["<|endoftext|>"],
            pretoken_cache_size=pretoken_cache_size,
        )
        seen = 0
        tokens = 0
        with output_path.open("wb") as out:
            for _, data in _read_chunk_tasks(input_path, target_bytes, chunk_bytes):
                ids = tokenizer.encode(data.decode("utf-8", errors="replace"))
                np.asarray(ids, dtype=np.uint16).tofile(out)
                seen += len(data)
                tokens += len(ids)
                elapsed = time.perf_counter() - start
                LOGGER.info(
                    "tokenize progress bytes=%s/%s tokens=%s elapsed=%.2fs bytes_per_second=%.2f",
                    seen,
                    target_bytes,
                    tokens,
                    elapsed,
                    seen / elapsed if elapsed else 0.0,
                )
        elapsed = time.perf_counter() - start
        return {
            "input": str(input_path),
            "output": str(output_path),
            "bytes": seen,
            "tokens": tokens,
            "seconds": elapsed,
            "bytes_per_second": seen / elapsed if elapsed else 0.0,
            "encoder_workers": encoder_workers,
            **tokenizer.cache_stats(),
        }

    part_dir = output_path.with_name(output_path.name + ".parts")
    part_dir.mkdir(parents=True, exist_ok=True)
    for old_part in part_dir.glob("part-*.bin"):
        old_part.unlink()

    seen = 0
    tokens = 0
    parts: dict[int, str] = {}
    cache_stats: list[dict[str, int | float]] = []
    with ProcessPoolExecutor(
        max_workers=encoder_workers,
        initializer=_init_encode_worker,
        initargs=(str(vocab_path), str(merges_path), ["<|endoftext|>"], pretoken_cache_size),
    ) as pool:
        task_iter = _read_chunk_tasks(input_path, target_bytes, chunk_bytes)
        pending: set[Future[tuple[int, int, int, str, dict[str, int | float]]]] = set()
        for _ in range(encoder_workers * 2):
            if not _submit_next_part(pool, task_iter, part_dir, pending):
                break
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                idx, byte_count, token_count, part_path, stats = future.result()
                _submit_next_part(pool, task_iter, part_dir, pending)
                seen += byte_count
                tokens += token_count
                parts[idx] = part_path
                cache_stats.append(stats)
                elapsed = time.perf_counter() - start
                LOGGER.info(
                    "parallel tokenize progress chunks=%s bytes=%s/%s tokens=%s elapsed=%.2fs bytes_per_second=%.2f",
                    len(parts),
                    seen,
                    target_bytes,
                    tokens,
                    elapsed,
                    seen / elapsed if elapsed else 0.0,
                )

    with output_path.open("wb") as out:
        for idx in sorted(parts):
            with Path(parts[idx]).open("rb") as part:
                out.write(part.read())
    for part_path in parts.values():
        Path(part_path).unlink()
    part_dir.rmdir()

    elapsed = time.perf_counter() - start
    return {
        "input": str(input_path),
        "output": str(output_path),
        "bytes": seen,
        "tokens": tokens,
        "seconds": elapsed,
        "bytes_per_second": seen / elapsed if elapsed else 0.0,
        "encoder_workers": encoder_workers,
        **_merge_cache_stats(cache_stats, pretoken_cache_size),
    }


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", nargs="?")
    parser.add_argument("--data-root", default=str(_default_data_root()))
    parser.add_argument("--input", dest="input_flag")
    parser.add_argument("--output")
    parser.add_argument("--vocab-file", "--vocab", dest="vocab_file")
    parser.add_argument("--merge-file", "--merges", dest="merge_file")
    parser.add_argument("--sample-size", "--bytes", dest="sample_size")
    parser.add_argument("--chunk-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--pretoken-cache-size", type=int, default=500_000)
    parser.add_argument("--encoder-worker", type=int, default=_default_encoder_workers())
    parser.add_argument("--serial", action="store_true")
    parser.add_argument("--out")
    args = parser.parse_args()

    root = Path(args.data_root)
    input_path = _resolve_input_path(args.input_flag or args.input_path, root)
    vocab_path, merges_path = _resolve_tokenizer_files(input_path, args.vocab_file, args.merge_file)
    output_path = Path(args.output) if args.output else input_path.with_name(input_path.name + "-tokenized.bin")
    target_bytes = _parse_size(args.sample_size) if args.sample_size else input_path.stat().st_size
    result = tokenize_file(
        input_path,
        output_path,
        vocab_path,
        merges_path,
        target_bytes,
        args.chunk_bytes,
        1 if args.serial else args.encoder_worker,
        args.pretoken_cache_size,
    )
    payload = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    _main()
