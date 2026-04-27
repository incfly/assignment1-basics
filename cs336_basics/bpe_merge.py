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
def init_word_nodes_list(word: bytes) -> Node:
    pass

type AllPairs = dict[(int,int), PairInfo]

# Cannot use set. set you can't modify the element itself.
# Same in C++, value are constant.
def init_pair_info(heads: list[Node]) -> AllPairs:
    all_pairs : AllPairs = {}
    return all_pairs


def merge(pair: PairInfo):
    pass


def bpe_merge(filename: str, merge_times=1):
    freq_map = init_token_freqmap(filename)
    heads = []
    for word, freq in freq_map.items():
        heads.append(init_word_nodes_list(word))
    all_pairs = init_pair_info(heads)
    for i in range(merge_times):
        # TODO: ineffiency for scanning.
        largest_pair = max(all_pairs, key=lambda p: p.freq)
        merge(largest_pair)


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
