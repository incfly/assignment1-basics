class Encoder:

    def __init__(self, vocab : ExternalVocab, merge : ExternalMerges):
        self.vocab = vocab
        self.merge = merge
    

    # List of the merge is a long list. 
    # Token itself is short.
    # Naive approach is to iterate O(n)
    # better approach might be build up the pair of the token of current word.
    # Then look up from the merge list. get all the index and take the smallest one.
    # O(len(pre-token word))
    def encode(self, b : str) -> list[int]:
        return []


