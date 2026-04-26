from dataclasses import dataclass, field

try:
    from cs336_basics.pretoken import init_token_freqmap, FrequencyMap
except ImportError:
    from pretoken import init_token_freqmap

type Vocab = dict[int, bytes]
type BytePair = tuple[int, int]

@dataclass
class MergeInfo:
    freq: int = 0
    # how many bytes total we have, so we know what to look forward.
    len: int = 0
    # [1] means the token itself, beginning.
    # token itself, for both before and after. mistake: only after.
    # TODO: fix this.
    indices: list[tuple[int,int]] = field(default_factory=list)

@dataclass
class WordDict:
    toint: dict[bytes, int] = field(default_factory=dict) # diff with {}? 
    toword: dict[int, bytes] = field(default_factory=dict)


def init_worddict(freq_map: FrequencyMap):
    wd : WordDict = WordDict()
    word_ind = 1
    for token, _ in freq_map.items():
        wd.toword[word_ind] = token
        wd.toint[token] = word_ind
        word_ind += 1
    return wd


def bpe_merge(filename: str) -> tuple[Vocab, dict[BytePair, MergeInfo]]:
    vocab: Vocab = {}
    next_vocab_int = 257
    freq_map = init_token_freqmap(filename)
    wdd = init_worddict(freq_map)

    byte_pair: dict[BytePair, MergeInfo] = {}
    for token, pretoken_freq in freq_map.items():
        for i, pair in enumerate(zip(token[:-1], token[1:])):
            info = byte_pair.get(pair)
            if info is None:
                info = MergeInfo()
                byte_pair[pair] = info
            info.freq += pretoken_freq
            # NOTE: how about the indices? need word + index of byte / char. word we can't store the str, but use an int.
            # That's why word_ind for.
            next = i+2
            if next >= len(token):
                next = -1
            info.indices.append((wdd.toint[token],next))
    if not byte_pair:
        return vocab, byte_pair

    max_bytes_key = max(byte_pair, key=lambda pair: byte_pair[pair].freq)
    max_bytes = bytes(max_bytes_key)
    print('Current iteration BPE max frequency token list is: ', max_bytes.decode("utf-8", errors="replace"))

    vocab[next_vocab_int] = max_bytes
    next_vocab_int += 1

    # Updating the max_byte_pair keys.
    cur_info = byte_pair[max_bytes_key]
    for ind in cur_info.indices:
        if ind[1] == -1:
            continue
        # Word.
        word_id = ind[0]
        word = wdd.toword[ind[0]]

        next_byte = word[ind[1]]
        new_byte_ind = ind[1] + 1
        if new_byte_ind >= len(word):
            new_byte_ind = -1

        new_bytes = max_bytes + bytes([next_byte])
        new_info = byte_pair.get(new_bytes, MergeInfo())

        # ab,c in the word fabc, we see fabc lots of
        new_info.freq += freq_map[word]

        new_info.indices.append((word_id, new_byte_ind))
        byte_pair[new_bytes] = new_info

    return vocab, byte_pair


if __name__ == "__main__":
    bpe_merge("./data/tiny-1000.txt")
