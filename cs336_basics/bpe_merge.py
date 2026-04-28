from dataclasses import dataclass, field
from typing import Optional

# To test, run
# uv run pytest /Users/incfly/workspace/github.com/incfly/assignment1-basics/tests/test_bpe_merge.py -q
try:
    from cs336_basics.pretoken import init_token_freqmap, FrequencyMap
except ImportError:
    from pretoken import init_token_freqmap


type Vocab = dict[int, bytes]    


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

# Cannot use set. set you can't modify the element itself.
# Same in C++, value are constant.
def init_pair_info(heads: list[Node]) -> AllPairs:
    all_pairs : AllPairs = {}
    for head in heads:
        ptr = head
        while ptr.next is not None:
            t1 = ptr.token
            t2 = ptr.next.token
            pair_key = (t1, t2)
            pair_info = all_pairs.get(pair_key)
            if pair_info is None:
                all_pairs[pair_key] = PairInfo(t1, t2)
                pair_info = all_pairs[pair_key]
            pair_info.record(ptr)
            ptr = ptr.next
    return all_pairs


def decrement_pair(all_pairs: AllPairs, left: Node, right: Node):
    pair_key = (left.token, right.token)
    pair_info = all_pairs.get(pair_key)
    if pair_info is None:
        return
    pair_info.freq -= left.freq
    if pair_info.freq <= 0:
        del all_pairs[pair_key]


def increment_pair(all_pairs: AllPairs, left: Node, right: Node):
    pair_key = (left.token, right.token)
    pair_info = all_pairs.get(pair_key)
    if pair_info is None:
        pair_info = PairInfo(left.token, right.token)
        all_pairs[pair_key] = pair_info
    pair_info.record(left)


def merge(token_id: int, pair: PairInfo, all_pairs: AllPairs):
    # a, b, c, d => a, bc, d
    for t1 in pair.records:
        if not t1.alive:
            continue
        t2 = t1.next
        if t2 is None or not t2.alive:
            continue
        if t1.token != pair.t1 or t2.token != pair.t2:
            continue

        left = t1.prev
        right = t2.next

        # decrementing a, bc.
        if left is not None and left.alive:
            decrement_pair(all_pairs, left, t1)
        # decrement b, c itself.
        decrement_pair(all_pairs, t1, t2)
        # decrementing bc, d
        if right is not None and right.alive:
            decrement_pair(all_pairs, t2, right)

        # merge operation.
        t1.token = token_id
        t1.next = right
        if right is not None:
            right.prev = t1

        t2.alive = False
        t2.prev = None
        t2.next = None

        # why checking alive? because bpe pair gen itself is not considering overlapping or not
        # it can aa, it can invalidate ab(second a).

        # increment a, bc; bc, d.
        if left is not None and left.alive:
            increment_pair(all_pairs, left, t1)
        if right is not None and right.alive:
            increment_pair(all_pairs, t1, right)


def bpe_merge(filename: str, merge_times=1):
    freq_map = init_token_freqmap(filename)
    heads = []
    for word, freq in freq_map.items():
        heads.append(init_word_nodes_list(word, freq))
    all_pairs = init_pair_info(heads)
    next_token_id = 257
    for i in range(merge_times):
        # TODO: ineffiency for scanning.
        largest_pair = max(all_pairs.values(), key=lambda pair: pair.freq)
        merge(next_token_id, largest_pair, all_pairs)
        next_token_id += 1


# STOP HERE. REST is old impl

type BytePair = tuple[int, int]


@dataclass
class WordDict:
    toint: dict[bytes, int] = field(default_factory=dict) # diff with {}? 
    toword: dict[int, bytes] = field(default_factory=dict)

@dataclass
class Index:
    word_id: int
    self_ind: int
    self_len: int

@dataclass
class MergeInfo:
    freq: int = 0
    # how many bytes total we have, so we know what to look forward.
    len: int = 0
    # [1] means the token itself, beginning.
    # token itself, for both before and after. mistake: only after.
    # TODO: fix this.
    indices: list[tuple[int,int]] = field(default_factory=list)
    indexv2: list[Index] = field(default_factory=list)


class BytePairMap(dict[BytePair, MergeInfo]):
    def max_pair(self) -> BytePair:
        return max(self, key=lambda pair: self[pair].freq)
    
    def decrement(self, bp: BytePair, freq: int):
        if self.get(bp) is None:
            print("panic!")
            return
        self[bp].freq -= freq
        if self[bp].freq == 0:
            del self[bp]
            print('delete key')


def init_worddict(freq_map: FrequencyMap) -> WordDict:
    wd : WordDict = WordDict()
    word_ind = 1
    for token, _ in freq_map.items():
        wd.toword[word_ind] = token
        wd.toint[token] = word_ind
        word_ind += 1
    return wd


if __name__ == "__main__":
    bpe_merge("./data/tiny-1000.txt")
