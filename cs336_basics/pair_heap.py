import heapq
from dataclasses import dataclass
from typing import Any, Mapping


PairKey = tuple[int, int]


@dataclass(frozen=True)
class _Entry:
    priority: tuple[int, bytes, bytes]
    pair_key: PairKey

    def __lt__(self, other: "_Entry") -> bool:
        return self.priority > other.priority


class PairHeap:
    def __init__(self, all_pairs: Mapping[PairKey, Any], token_bytes: Mapping[int, bytes]):
        self.all_pairs = all_pairs
        self.token_bytes = token_bytes
        self.heap: list[_Entry] = []
        for pair_key in all_pairs:
            self.push(pair_key)

    def _priority(self, pair_info: Any) -> tuple[int, bytes, bytes]:
        return (
            pair_info.freq,
            self.token_bytes[pair_info.t1],
            self.token_bytes[pair_info.t2],
        )

    def push(self, pair_key: PairKey) -> None:
        pair_info = self.all_pairs.get(pair_key)
        if pair_info is not None and pair_info.freq > 0:
            heapq.heappush(self.heap, _Entry(self._priority(pair_info), pair_key))

    def push_many(self, pair_keys: set[PairKey]) -> None:
        for pair_key in pair_keys:
            self.push(pair_key)

    def pop_best(self) -> Any | None:
        # Lazy invalidation: updates only push a fresh priority snapshot.
        # Old heap entries are ignored here when they no longer match all_pairs.
        while self.heap:
            entry = heapq.heappop(self.heap)
            pair_info = self.all_pairs.get(entry.pair_key)
            if pair_info is None or pair_info.freq <= 0:
                continue
            if entry.priority == self._priority(pair_info):
                return pair_info
            self.push(entry.pair_key)
        return None
