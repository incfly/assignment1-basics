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
    # Placeholder for future occurrence bookkeeping.
    indices: list[tuple[int,int]] = field(default_factory=list)


def freqmap_to_tokenmap(freq_map: FrequencyMap):
    out = dict[int, int]
    word_ind = 1
    for token, _ in freq_map:
        out[token] = word_ind
        word_ind += 1
    return word_ind


def bpe_merge(filename: str = "./data/tiny-1000.txt") -> tuple[Vocab, dict[BytePair, MergeInfo]]:
    vocab: Vocab = {}
    next_vocab_int = 257

    freq_map = init_token_freqmap(filename)
    word_ind = freqmap_to_tokenmap(freq_map)

    byte_pair: dict[BytePair, MergeInfo] = {}
    for token, _ in freq_map.items():
        for i, pair in enumerate(zip(token[:-1], token[1:])):
            info = byte_pair.get(pair, MergeInfo())
            info.freq += 1
            # NOTE: how about the indices? need word + index of byte / char. word we can't store the str, but use an int.
            # That's why word_ind for.
            next = i+2
            if next >= len(token):
                next = -1
            info.indices.append((word_ind[token],next))
    if not byte_pair:
        return vocab, byte_pair

    max_byte_pair_key = max(byte_pair, key=lambda pair: byte_pair[pair].freq)
    max_byte_pair = bytes(max_byte_pair_key)
    print('Current iteration BPE max frequency token list is: ', max_byte_pair.decode("utf-8", errors="replace"))

    vocab[next_vocab_int] = max_byte_pair
    next_vocab_int += 1

    # Updating the max_byte_pair keys.
    cur_info = byte_pair[max_byte_pair_key]
    for ind in cur_info.indices:
        if ind[1] == -1:
            continue
        word = word_ind[ind[0]]
        next_byte = word[ind[1]]
        new_byte_ind = ind[1]
        if new_byte_ind == len(word):
            new_byte_ind = -1
        new_byte_key = max_byte_pair_key
        new_byte_key.append(bytes(next_byte))

        new_info = byte_pair.get(new_byte_key, MergeInfo())
        new_info.freq += 1
        new_info.indices.append((word_ind, next_byte))
        byte_pair[new_byte_key] = new_info

    return vocab, byte_pair


if __name__ == "__main__":
    bpe_merge()
