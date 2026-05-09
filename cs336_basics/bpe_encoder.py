import json
from pathlib import Path
import regex as re
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING

from cs336_basics.pretoken import PY_PRETOKEN_PATTERN, RegexMode

if TYPE_CHECKING:
    from cs336_basics.bpe_merge import ExternalMerges, ExternalVocab


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


class Encoder:
    def __init__(
        self,
        vocab: "ExternalVocab",
        merge: "ExternalMerges",
        special_tokens: list[str] | None = None,
        regex_mode: RegexMode = "cpp",
    ):
        self.vocab = vocab
        self.merge = merge
        self.regex_mode = regex_mode
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
    ) -> "Encoder":
        vocab = _load_vocab(vocab_filepath)
        return cls(vocab, _load_merges(merges_filepath), special_tokens=special_tokens)

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
            ids.extend(self.encode_pretoken_bytes(pretoken.encode("utf-8")))
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
