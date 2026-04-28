# BPE

pre token, for efficiency
    avoid scanning the corupus again and again
    re.findall() find all pre-token ` and`, ` text`
    count frequency;

impl details
    py GIL, using compute pool, worker, merge output

then BPE merge within pre-token
    not across boundary, `william shakespear` would never be merged
    for efficiency, only look at the dict.

merge algo crucial
    MergeInfo where storing index for faster merge iteration
    Record index of the thing itself, so update both before and after
        `ad`, so `bad` and `advertise` both udpate `ad`.
        not just after.
    non overlapping merge   
        `aaaa` -> `aa` `aa` but should now be `aa, aa` token, produce `aa, aa` together as a new token
        but not `aa, a` as next token. as that does not make sense.
        need special care to fix this.
    not only needs to add but also delete
        `abab`, `ab:2, ba:1`
        `ab,ab`, no `ba` anymore! my buggy has this
        but not necessarily complete remove. `ba` can still exists
            only delete when frequency is 0.
    get next destroyed is tricky because not just next byte
        can be `ab<foo>` the whole `b<foo>` used to be toegether.
        using longest sequence is wrong! bpe is split not just
        [e.g.](https://pastila.nl/?007427fd/c242125bc050b22791155a95d2dcd6d2#UdW4BCiyZeIkG29/cxNiRA==GCM)

what to do, doubly linked list.
    https://chatgpt.com/c/69ef0184-bde8-83ea-9824-ee1e23290d3f


half vibe coding,
    psedyo code with comment. ask what can be wrong.
```python
def merge(token_id: int, pair: PairInfo, all_pairs: AllPairs):
    # imagine, a, b, c, d, => a, bc, d.
    # we are deleting ab and cd, first.
    # for all pairinfo.records we do this:
    # TODO: identify ab and cd.
    # then reduce frequency by node.freq
    # then add bc frequncy by node.freq
    # then we add frequency of `bc, d` and `a, bc`
    pass
```

## TODO

>  When encoding text, it’s often desirable to treat some strings as “special tokens” that should
2Note that the original BPE formulation R. Sennrich et al. [3] specifies the inclusion of an end-of-word token. We do not
add an end-of-word-token when training byte-level BPE models because all bytes (including whitespace and punctuation)
are included in the model’s vocabulary. Since we’re explicitly representing spaces and punctuation, the learned BPE merges
will naturally reflect these word boundaries.

Not sure how this would be represented in the impl.
