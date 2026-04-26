from pretoken import init_token_freqmap, FrequencyMap

from itertools import islice


def bpe_merge():
    freq_map = init_token_freqmap("./data/tiny-1000.txt")
    # Also the same type, even though the key is always sized 2. But it's ok.
    byte_pair: FrequencyMap = {}
    for token, count in freq_map.items():
        for b in zip(token[:-1], token[1:]):
            byte_pair[b] = byte_pair.get(b, 0) + 1

    # get the highest byte pair.
    max_byte_pair = bytes(max(byte_pair, key=byte_pair.get))
    print(max_byte_pair.decode('utf-8'))
    

if __name__ == "__main__":
    bpe_merge()