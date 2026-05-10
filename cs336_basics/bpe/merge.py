import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# To test, run
# uv run pytest /Users/incfly/workspace/github.com/incfly/assignment1-basics/tests/test_bpe_merge.py -q
try:
    from cs336_basics.bpe.pair_heap import PairHeap, PairKey
    from cs336_basics.bpe.pretoken import init_token_freqmap, FrequencyMap
except ImportError:
    from pair_heap import PairHeap, PairKey
    from pretoken import init_token_freqmap

LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False


def _default_pretoken_workers() -> int:
    return max(1, os.cpu_count() or 1)


def _default_pretoken_chunks(pretoken_workers: int) -> int:
    return max(1, 4 * pretoken_workers)


class Vocab:
    def __init__(self):
        self.toint: dict[bytes, int] = {bytes([i]): i for i in range(256)}
        self.toword: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.next_ind = 257
        self.merges: list[Merge] = []

    def add_word(self, b: bytes) -> int:
        token_id = self.toint.get(b)
        if token_id is not None:
            return token_id

        token_id = self.next_ind
        self.next_ind += 1
        self.toint[b] = token_id
        self.toword[token_id] = b
        return token_id

    def add_merge(self, left_id: int, right_id: int) -> int:
        merged = self.toword[left_id] + self.toword[right_id]
        return self.add_word(merged)

    def items(self):
        return (
            (token_id, token)
            for token_id, token in self.toword.items()
            if token_id >= 256
        )



@dataclass
class Node:
    token: int
    # the frequency where this node resides in. i.e. the word.
    freq: int
    prev: Optional["Node"] = None
    next: Optional["Node"] = None
    alive: bool = True


class PairInfo:
    """
    (int_of_token_a, int_of_token_b)
    occurred in lots of places.
    """
    def __init__(self, t1:int, t2:int):
        self.t1 = t1
        self.t2 = t2
        self.freq = 0
        self.records = []
    
    def record(self, n:Node):
        self.records.append(n)
        self.freq += n.freq


# One node per byte.
def init_word_nodes_list(word: bytes, freq: int) -> Node:
    if not word:
        raise ValueError("word must not be empty")

    head = Node(token=word[0], freq=freq)
    prev = head
    for token in word[1:]:
        cur = Node(token=token, freq=freq, prev=prev)
        prev.next = cur
        prev = cur
    return head


type AllPairs = dict[(int,int), PairInfo]
type Merge = tuple[int, int]
type BytePairs = dict[bytes, PairInfo]

# Structure required by the assignement.
type ExternalMerges = list[tuple[bytes, bytes]]
type ExternalVocab = dict[int, bytes]

# Cannot use set. set you can't modify the element itself.
# Same in C++, value are constant.
def init_pair_info(heads: list[Node]) -> AllPairs:
    start_time = time.perf_counter()
    all_pairs : AllPairs = {}
    total_pairs = 0
    total_heads = len(heads)
    for head_index, head in enumerate(heads, start=1):
        ptr = head
        while ptr.next is not None:
            total_pairs += 1
            t1 = ptr.token
            t2 = ptr.next.token
            pair_key = (t1, t2)
            pair_info = all_pairs.get(pair_key)
            if pair_info is None:
                all_pairs[pair_key] = PairInfo(t1, t2)
                pair_info = all_pairs[pair_key]
            pair_info.record(ptr)
            ptr = ptr.next
        if head_index % 1_000_000 == 0:
            LOGGER.info(
                "init_pair_info progress heads=%s/%s pair_records=%s unique_pairs=%s elapsed=%.2fs",
                head_index,
                total_heads,
                total_pairs,
                len(all_pairs),
                time.perf_counter() - start_time,
            )
    LOGGER.info(
        "init_pair_info finish heads=%s pair_records=%s unique_pairs=%s elapsed=%.2fs",
        total_heads,
        total_pairs,
        len(all_pairs),
        time.perf_counter() - start_time,
    )
    return all_pairs


def decrement_pair(all_pairs: AllPairs, left: Node, right: Node) -> PairKey | None:
    pair_key = (left.token, right.token)
    pair_info = all_pairs.get(pair_key)
    if pair_info is None:
        return None
    pair_info.freq -= left.freq
    if pair_info.freq <= 0:
        del all_pairs[pair_key]
    return pair_key


def increment_pair(all_pairs: AllPairs, left: Node, right: Node) -> PairKey:
    pair_key = (left.token, right.token)
    pair_info = all_pairs.get(pair_key)
    if pair_info is None:
        pair_info = PairInfo(left.token, right.token)
        all_pairs[pair_key] = pair_info
    pair_info.record(left)
    return pair_key


def decrement_neighbor_pairs(all_pairs: AllPairs, left: Optional[Node], t1: Node, t2: Node, right: Optional[Node]) -> set[PairKey]:
    changed = set()
    if left is not None and left.alive:
        changed_key = decrement_pair(all_pairs, left, t1)
        if changed_key is not None:
            changed.add(changed_key)
    changed_key = decrement_pair(all_pairs, t1, t2)
    if changed_key is not None:
        changed.add(changed_key)
    if right is not None and right.alive:
        changed_key = decrement_pair(all_pairs, t2, right)
        if changed_key is not None:
            changed.add(changed_key)
    return changed


def increment_neighbor_pairs(all_pairs: AllPairs, left: Optional[Node], merged: Node, right: Optional[Node]) -> set[PairKey]:
    changed = set()
    if left is not None and left.alive:
        changed.add(increment_pair(all_pairs, left, merged))
    if right is not None and right.alive:
        changed.add(increment_pair(all_pairs, merged, right))
    return changed


def merge_two_nodes(token_id: int, t1: Node, t2: Node) -> Node:
    right = t2.next
    t1.token = token_id
    t1.next = right
    if right is not None:
        right.prev = t1

    t2.alive = False
    t2.prev = None
    t2.next = None
    return t1


def merge(token_id: int, pair: PairInfo, all_pairs: AllPairs) -> set[PairKey]:
    # a, b, c, d => a, bc, d
    changed = set()
    for t1 in pair.records:
        if not t1.alive:
            continue
        t2 = t1.next
        # aaa, => a, aa. then a[0]a[1], a[1] is not alive.
        if t2 is None or not t2.alive:
            continue
        if t1.token != pair.t1 or t2.token != pair.t2:
            continue

        left = t1.prev
        right = t2.next

        changed.update(decrement_neighbor_pairs(all_pairs, left, t1, t2, right))
        merged = merge_two_nodes(token_id, t1, t2)
        changed.update(increment_neighbor_pairs(all_pairs, left, merged, right))
    return changed


def export_pairs(all_pairs: AllPairs, vocab: Vocab) -> BytePairs:
    out: BytePairs = {}
    for (left_id, right_id), info in all_pairs.items():
        out[vocab.toword[left_id] + vocab.toword[right_id]] = info
    return out


def pair_sort_key(item: tuple[tuple[int, int], PairInfo], vocab: Vocab) -> tuple[int, bytes, bytes]:
    (left_id, right_id), info = item
    return info.freq, vocab.toword[left_id], vocab.toword[right_id]


def export_train_result(vocab: Vocab, special_tokens: list[str]) -> tuple[ExternalVocab, ExternalMerges]:
    external_vocab: ExternalVocab = {}
    external_merges: ExternalMerges = []

    next_token_id = 0
    for special_token in special_tokens:
        external_vocab[next_token_id] = special_token.encode("utf-8")
        next_token_id += 1

    for byte in range(256):
        external_vocab[next_token_id] = bytes([byte])
        next_token_id += 1

    for left_id, right_id in vocab.merges:
        left = vocab.toword[left_id]
        right = vocab.toword[right_id]
        external_merges.append((left, right))
        external_vocab[next_token_id] = left + right
        next_token_id += 1

    return external_vocab, external_merges


def _artifact_paths(input_path: str) -> tuple[Path, Path]:
    input_file = Path(input_path)
    return (
        input_file.parent / f"{input_file.name}-vocab.json",
        input_file.parent / f"{input_file.name}-merge.json",
    )


def _persist_train_result(
    input_path: str,
    vocab: ExternalVocab,
    merges: ExternalMerges,
) -> tuple[Path, Path]:
    vocab_path, merge_path = _artifact_paths(input_path)
    vocab_payload = [
        {"token_id": token_id, "hex": token.hex(), "repr": repr(token)}
        for token_id, token in sorted(vocab.items())
    ]
    merge_payload = [
        {
            "left_hex": left.hex(),
            "right_hex": right.hex(),
            "left_repr": repr(left),
            "right_repr": repr(right),
        }
        for left, right in merges
    ]

    vocab_path.write_text(json.dumps(vocab_payload, indent=2) + "\n", encoding="utf-8")
    merge_path.write_text(json.dumps(merge_payload, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("persisted vocab=%s merge=%s", vocab_path, merge_path)
    return vocab_path, merge_path

# TODO: PairInfo.records is append-only, so while the current validation protects
# correctness better, the structure will still accumulate stale records and may get inefficient.
def bpe_merge(
    filename: str,
    vocab_size: int | None = None,
    merge_times: int | None = None,
    pretoken_workers: int | None = None,
    pretoken_chunks: int | None = None,
    pretoken_profile_dir: str | None = None,
    regex_mode: str = "cpp",
    special_token: bytes = b"<|endoftext|>",
) -> tuple[Vocab, BytePairs]:
    start_time = time.perf_counter()
    if pretoken_workers is None:
        pretoken_workers = _default_pretoken_workers()
    if pretoken_chunks is None:
        pretoken_chunks = _default_pretoken_chunks(pretoken_workers)

    if merge_times is None:
        if vocab_size is None:
            merge_times = 1
        else:
            merge_times = max(0, vocab_size - 256)

    LOGGER.info(
        "bpe_merge start filename=%s vocab_size=%s merge_times=%s pretoken_workers=%s pretoken_chunks=%s regex_mode=%s pretoken_profile_dir=%s special_token=%r",
        filename,
        vocab_size,
        merge_times,
        pretoken_workers,
        pretoken_chunks,
        regex_mode,
        pretoken_profile_dir,
        special_token,
    )

    freq_map = init_token_freqmap(
        filename,
        desired_num_chunks=pretoken_chunks,
        num_workers=pretoken_workers,
        profile_dir=pretoken_profile_dir,
        regex_mode=regex_mode,
        special_token=special_token,
    )
    vocab = Vocab()
    heads = []
    build_heads_start = time.perf_counter()
    total_heads = len(freq_map)
    total_word_bytes = 0
    LOGGER.info("bpe_merge build_heads start unique_tokens=%s", total_heads)
    for head_index, (word, freq) in enumerate(freq_map.items(), start=1):
        heads.append(init_word_nodes_list(word, freq))
        total_word_bytes += len(word)
        if head_index % 1_000_000 == 0:
            LOGGER.info(
                "bpe_merge build_heads progress heads=%s/%s total_word_bytes=%s elapsed=%.2fs",
                head_index,
                total_heads,
                total_word_bytes,
                time.perf_counter() - build_heads_start,
            )
    LOGGER.info(
        "bpe_merge build_heads finish heads=%s total_word_bytes=%s elapsed=%.2fs",
        total_heads,
        total_word_bytes,
        time.perf_counter() - build_heads_start,
    )
    LOGGER.info("bpe_merge init_pair_info start heads=%s", len(heads))
    all_pairs = init_pair_info(heads)
    pair_heap = PairHeap(all_pairs, vocab.toword)
    for i in range(merge_times):
        if not all_pairs:
            break
        # Break frequency ties by preferring the lexicographically larger byte pair.
        find_max_start = time.perf_counter()
        largest_pair = pair_heap.pop_best()
        if largest_pair is None:
            break
        LOGGER.info(
            "bpe_merge find_max finish iteration=%s/%s unique_pairs=%s elapsed=%.4fs",
            i + 1,
            merge_times,
            len(all_pairs),
            time.perf_counter() - find_max_start,
        )
        vocab.merges.append((largest_pair.t1, largest_pair.t2))
        token_id = vocab.add_merge(largest_pair.t1, largest_pair.t2)
        pair_heap.push_many(merge(token_id, largest_pair, all_pairs))
        completed = i + 1
        if completed % 100 == 0:
            LOGGER.info(
                "bpe_merge progress filename=%s merges_completed=%s/%s vocab_size=%s remaining_pairs=%s",
                filename,
                completed,
                merge_times,
                len(vocab.toword),
                len(all_pairs),
            )
    LOGGER.info(
        "bpe_merge finish filename=%s merges_completed=%s requested_merges=%s vocab_size=%s elapsed=%.2fs",
        filename,
        len(vocab.merges),
        merge_times,
        len(vocab.toword),
        time.perf_counter() - start_time,
    )
    return vocab, export_pairs(all_pairs, vocab)


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    pretoken_workers: int | None = None,
    pretoken_chunks: int | None = None,
    pretoken_profile_dir: str | None = None,
    regex_mode: str = "cpp",
) -> tuple[ExternalVocab, ExternalMerges]:
    if pretoken_workers is None:
        pretoken_workers = _default_pretoken_workers()
    if pretoken_chunks is None:
        pretoken_chunks = _default_pretoken_chunks(pretoken_workers)

    merge_vocab_size = max(256, vocab_size - len(special_tokens))
    split_special_token = (
        special_tokens[0].encode("utf-8")
        if special_tokens
        else b"<|endoftext|>"
    )

    vocab, _ = bpe_merge(
        input_path,
        vocab_size=merge_vocab_size,
        pretoken_workers=pretoken_workers,
        pretoken_chunks=pretoken_chunks,
        pretoken_profile_dir=pretoken_profile_dir,
        regex_mode=regex_mode,
        special_token=split_special_token,
    )
    exported_vocab, exported_merges = export_train_result(vocab, special_tokens)
    _persist_train_result(input_path, exported_vocab, exported_merges)
    return exported_vocab, exported_merges

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("--vocab-size", type=int, default=5)
    parser.add_argument("--pretoken-worker", type=int, default=_default_pretoken_workers())
    parser.add_argument("--pretoken-profile-dir")
    parser.add_argument("--regex-mode", choices=["py", "cpp"], default="cpp")
    args = parser.parse_args()
    vocab, merges = train_bpe(
        input_path=args.input_file,
        vocab_size=args.vocab_size,
        pretoken_workers=args.pretoken_worker,
        pretoken_profile_dir=args.pretoken_profile_dir,
        regex_mode=args.regex_mode,
        special_tokens=["<|endoftext|>"])
