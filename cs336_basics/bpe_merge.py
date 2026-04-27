from dataclasses import dataclass, field

# To test, run
# uv run pytest /Users/incfly/workspace/github.com/incfly/assignment1-basics/tests/test_bpe_merge.py -q
try:
    from cs336_basics.pretoken import init_token_freqmap, FrequencyMap
except ImportError:
    from pretoken import init_token_freqmap


type Vocab = dict[int, bytes]
type BytePair = tuple[int, int]


@dataclass
class WordDict:
    toint: dict[bytes, int] = field(default_factory=dict) # diff with {}? 
    toword: dict[int, bytes] = field(default_factory=dict)

def resolve_token(vocab: Vocab, b: bytes) -> bytes:
    best: bytes | None = None
    for token_bytes in vocab.values():
        if b.startswith(token_bytes):
            if best is None or len(token_bytes) > len(best):
                best = token_bytes
    if best is None:
        raise ValueError(f"No token in vocab matches prefix of {b!r}")
    return best

@dataclass
class Index:
    word_id: int
    self_ind: int
    self_len: int

    def destroyed_pair(self, wd:WordDict, vocab: Vocab) -> bytes:
        word = wd.toword[self.word_id]
        start = self.self_ind
        tok = resolve_token(vocab, word[start + self.self_len:])
        total_len = self.self_len + len(tok)
        return word[start : start + total_len]

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

def merge(byte_pair_map : BytePairMap, 
        freq_map : FrequencyMap,
        wdd:WordDict,
        vocab: Vocab,
        next_vocab_int: int) -> int:
    max_bytes_key = byte_pair_map.max_pair()
    max_bytes = bytes(max_bytes_key)

    print('Current iteration BPE max frequency token list is: ', max_bytes.decode("utf-8", errors="replace"))

    vocab[next_vocab_int] = max_bytes
    next_vocab_int += 1
    pair_info = byte_pair_map[max_bytes_key]
    freq = pair_info.freq
    # First remove the ones get disappeared due to new representation.
    for ind in pair_info.indexv2:
        # the last one byte in the sequence is getting distroyed
        destroyed = ind.destroyed_pair(wdd, vocab)
        byte_pair_map.decrement(destroyed, freq)

    # Updating the max_byte_pair keys.
    
    for ind in pair_info.indexv2:
        # Word.
        word = wdd.toword[ind.word_id]
        after = ind.self_ind+1
        if after >= len(word):
            continue
        new_bytes = max_bytes + bytes([word[after]])
        new_info = byte_pair_map.get(new_bytes, MergeInfo())

        # ab,c in the word fabc, we see fabc lots of
        new_info.freq += freq_map[word]

        # Index is the same.
        new_ind = ind
        new_ind.self_len += 1
        new_info.indexv2.append(new_ind)
        byte_pair_map[new_bytes] = new_info
    return next_vocab_int


def bpe_merge(filename: str, merge_times=1) -> tuple[Vocab, BytePairMap]:
    vocab: Vocab = {}
    next_vocab_int = 257
    freq_map = init_token_freqmap(filename)
    wdd = init_worddict(freq_map)

    byte_pair_map = BytePairMap()
    for token, pretoken_freq in freq_map.items():
        for i, pair in enumerate(zip(token[:-1], token[1:])):
            info = byte_pair_map.get(pair)
            if info is None:
                info = MergeInfo()
                byte_pair_map[pair] = info
            info.freq += pretoken_freq
            # NOTE: how about the indices? need word + index of byte / char. word we can't store the str, but use an int.
            # That's why word_ind for.
            next = i+2
            if next >= len(token):
                next = -1
            word_id = wdd.toint[token]
            info.indexv2.append(Index(word_id=word_id, self_ind=i, self_len=2))
    if not byte_pair_map:
        return vocab, byte_pair_map

    for _ in range(merge_times):
        next_vocab_int = merge(byte_pair_map, freq_map, wdd, vocab, next_vocab_int)

    return vocab, byte_pair_map


if __name__ == "__main__":
    bpe_merge("./data/tiny-1000.txt")
