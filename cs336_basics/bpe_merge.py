from dataclasses import dataclass, field

try:
    from cs336_basics.pretoken import init_token_freqmap
except ImportError:
    from pretoken import init_token_freqmap

type Vocab = dict[int, bytes]
type BytePair = tuple[int, int]


@dataclass
class MergeInfo:
    freq: int = 0
    # Placeholder for future occurrence bookkeeping.
    indices: list[int] = field(default_factory=list)


def bpe_merge(filename: str = "./data/tiny-1000.txt") -> tuple[Vocab, dict[BytePair, MergeInfo]]:
    vocab: Vocab = {}
    next_vocab_int = 257

    freq_map = init_token_freqmap(filename)
    byte_pair: dict[BytePair, MergeInfo] = {}
    for token, count in freq_map.items():
        for pair in zip(token[:-1], token[1:]):
            info = byte_pair.get(pair, MergeInfo())
            info.freq += 1
            # TODO: how about the indices? need index + global str? or maybe index the token first and the two int?

    if not byte_pair:
        return vocab, byte_pair

    max_byte_pair_key = max(byte_pair, key=lambda pair: byte_pair[pair].freq)
    max_byte_pair = bytes(max_byte_pair_key)
    print(max_byte_pair.decode("utf-8", errors="replace"))

    vocab[next_vocab_int] = max_byte_pair
    next_vocab_int += 1
    return vocab, byte_pair


if __name__ == "__main__":
    bpe_merge()
