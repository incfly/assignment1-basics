from pretoken import init_token_freqmap, FrequencyMap

def bpe_merge():
    freq_map = init_token_freqmap("./data/tiny-1000.txt")
    print(f"freq_map size {len(freq_map)}")

if __name__ == "__main__":
    bpe_merge()