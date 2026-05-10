import argparse
import json
import logging
import os
from pathlib import Path
import regex as re
import time
from collections import OrderedDict
from collections.abc import Iterable, Iterator
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


def _iter_docs(path: Path, chunk_chars: int = 8 * 1024 * 1024) -> Iterator[str]:
    carry = ""
    with path.open("r", encoding="utf-8", errors="replace") as f:
        while True:
            chunk = f.read(chunk_chars)
            if not chunk:
                break
            parts = (carry + chunk).split("<|endoftext|>")
            yield from (p for p in parts[:-1] if p)
            carry = parts[-1]
    if carry:
        yield carry


def _sample_docs(path: Path, count: int) -> list[str]:
    docs: list[str] = []
    for doc in _iter_docs(path):
        docs.append(doc)
        if len(docs) >= count:
            break
    return docs


def _ratio(tokenizer: "Encoder", docs: list[str]) -> dict[str, float | int]:
    byte_count = sum(len(doc.encode("utf-8")) for doc in docs)
    token_count = sum(len(tokenizer.encode(doc)) for doc in docs)
    return {"bytes": byte_count, "tokens": token_count, "bytes_per_token": byte_count / token_count}


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


def benchmark_file(
    tokenizer: Encoder,
    input_path: Path,
    target_bytes: int,
    chunk_bytes: int,
) -> dict[str, float | int | str]:
    seen = 0
    tokens = 0
    start = time.perf_counter()
    with input_path.open("rb") as f:
        while seen < target_bytes:
            data = f.read(min(chunk_bytes, target_bytes - seen))
            if not data:
                break
            ids = tokenizer.encode(data.decode("utf-8", errors="replace"))
            seen += len(data)
            tokens += len(ids)
            elapsed = time.perf_counter() - start
            LOGGER.info(
                "benchmark progress bytes=%s/%s tokens=%s elapsed=%.2fs bytes_per_second=%.2f",
                seen,
                target_bytes,
                tokens,
                elapsed,
                seen / elapsed if elapsed else 0.0,
            )
    elapsed = time.perf_counter() - start
    return {
        "input": str(input_path),
        "bytes": seen,
        "tokens": tokens,
        "seconds": elapsed,
        "bytes_per_second": seen / elapsed if elapsed else 0.0,
        **tokenizer.cache_stats(),
    }


def _main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    ans = sub.add_parser("answers")
    ans.add_argument("--data-root", default=str(_default_data_root()))
    ans.add_argument("--sample-count", type=int, default=10)
    ans.add_argument("--tiny-vocab-file", "--tiny-vocab", dest="tiny_vocab_file")
    ans.add_argument("--tiny-merge-file", "--tiny-merges", dest="tiny_merge_file")
    ans.add_argument("--owt-vocab-file", "--owt-vocab", dest="owt_vocab_file")
    ans.add_argument("--owt-merge-file", "--owt-merges", dest="owt_merge_file")
    ans.add_argument("--out")

    bench = sub.add_parser("benchmark")
    bench.add_argument("--data-root", default=str(_default_data_root()))
    bench.add_argument("--input")
    bench.add_argument("--vocab-file", "--vocab", dest="vocab_file")
    bench.add_argument("--merge-file", "--merges", dest="merge_file")
    bench.add_argument("--sample-size", "--bytes", dest="sample_size", default="128m")
    bench.add_argument("--chunk-bytes", type=int, default=8 * 1024 * 1024)
    bench.add_argument("--pretoken-cache-size", type=int, default=500_000)
    bench.add_argument("--out")
    args = parser.parse_args()

    root = Path(args.data_root)
    if args.cmd == "answers":
        tiny_vocab = Path(args.tiny_vocab_file) if args.tiny_vocab_file else root / "TinyStories-train.txt-vocab.json"
        tiny_merges = Path(args.tiny_merge_file) if args.tiny_merge_file else root / "TinyStories-train.txt-merge.json"
        owt_vocab = Path(args.owt_vocab_file) if args.owt_vocab_file else root / "owt_train.txt-vocab.json"
        owt_merges = Path(args.owt_merge_file) if args.owt_merge_file else root / "owt_train.txt-merge.json"
        tiny_tok = Encoder.from_files(str(tiny_vocab), str(tiny_merges), special_tokens=["<|endoftext|>"])
        owt_tok = Encoder.from_files(str(owt_vocab), str(owt_merges), special_tokens=["<|endoftext|>"])
        tiny_docs = _sample_docs(root / "TinyStories-train.txt", args.sample_count)
        owt_docs = _sample_docs(root / "owt_train.txt", args.sample_count)
        result = {
            "tiny_docs_tiny_tokenizer": _ratio(tiny_tok, tiny_docs),
            "owt_docs_owt_tokenizer": _ratio(owt_tok, owt_docs),
            "owt_docs_tiny_tokenizer": _ratio(tiny_tok, owt_docs),
        }
        payload = json.dumps(result, indent=2)
        if args.out:
            Path(args.out).write_text(payload + "\n", encoding="utf-8")
        print(payload)
    elif args.cmd == "benchmark":
        input_path = Path(args.input) if args.input else root / "owt_train.txt"
        vocab_path = Path(args.vocab_file) if args.vocab_file else root / "owt_train.txt-vocab.json"
        merges_path = Path(args.merge_file) if args.merge_file else root / "owt_train.txt-merge.json"
        tokenizer = Encoder.from_files(
            str(vocab_path),
            str(merges_path),
            special_tokens=["<|endoftext|>"],
            pretoken_cache_size=args.pretoken_cache_size,
        )
        result = benchmark_file(tokenizer, input_path, _parse_size(args.sample_size), args.chunk_bytes)
        payload = json.dumps(result, indent=2)
        if args.out:
            Path(args.out).write_text(payload + "\n", encoding="utf-8")
        print(payload)


if __name__ == "__main__":
    _main()
