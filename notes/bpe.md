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

## TODO

>  When encoding text, it’s often desirable to treat some strings as “special tokens” that should
2Note that the original BPE formulation R. Sennrich et al. [3] specifies the inclusion of an end-of-word token. We do not
add an end-of-word-token when training byte-level BPE models because all bytes (including whitespace and punctuation)
are included in the model’s vocabulary. Since we’re explicitly representing spaces and punctuation, the learned BPE merges
will naturally reflect these word boundaries.

Not sure how this would be represented in the impl.
