from pathlib import Path

import pytest

from cs336_basics.bpe_merge import bpe_merge


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "test"


def _to_bytes(key: object) -> bytes:
    if isinstance(key, bytes):
        return key
    return bytes(key)


def _render(vocab, byte_pair_map) -> str:
    lines = ["vocab:"]
    for token_id, token in sorted(vocab.items()):
        lines.append(f"{token_id}: {token.decode('utf-8', errors='replace')}")

    lines.append("pairs:")
    normalized_pairs = {_to_bytes(key): info.freq for key, info in byte_pair_map.items()}
    for token, freq in sorted(normalized_pairs.items()):
        lines.append(f"{token.decode('utf-8', errors='replace')}: {freq}")
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize(
    ("input_name", "output_name", "merge_times"),
    [
        ("bpe1.txt", "bpe1-out.txt", 0),
        ("bpe2.txt", "bpe2-out.txt", 1),
        ("bpe3.txt", "bpe3-out.txt", 1),
        ("bpe4.txt", "bpe4-out.txt", 1),
        ("bpe5.txt", "bpe5-out.txt", 2),
    ],
)
def test_bpe_merge_fixtures(input_name: str, output_name: str, merge_times: int) -> None:
    # These fixtures describe the intended BPE state after each round.
    # Some cases are expected to fail until merge() handles non-overlapping
    # replacement and updates pairs on both sides of the merged token.
    vocab, byte_pair_map = bpe_merge(str(DATA_DIR / input_name), merge_times=merge_times)
    expected = (DATA_DIR / output_name).read_text(encoding="utf-8")
    assert _render(vocab, byte_pair_map) == expected
