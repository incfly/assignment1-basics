import cProfile
import sys
import time
import os
import logging
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path
from typing import BinaryIO

import regex as re

type ChunkBounds = tuple[int, int]
type ChunkBoundsList = list[ChunkBounds]
type FrequencyMap = dict[bytes, int]

data: bytes
split_token: bytes = b"<|endoftext|>"
worker_profile_dir: str | None = None

LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def _init_shared_string(
    s: bytes,
    special_token: bytes,
    profile_dir: str | None,
) -> None:
    '''
    Init the worker with entire training data corpus. Not the best but ok.
    NOTE: Good to know, python async io only helps with io concurrency not compute.
    global mem sharing data, with compute pool (diff process). This is to work around GIL issue.
    '''
    global data
    global split_token
    global worker_profile_dir
    data = s
    split_token = special_token
    worker_profile_dir = profile_dir


def _get_file_chunk_bounds(
    filename: str,
    desired_num_chunks: int = 4,
    special_token: bytes = b"<|endoftext|>",
) -> tuple[bytes, ChunkBoundsList]:
    with open(filename, "rb") as f:
        boundaries = find_chunk_boundaries(f, desired_num_chunks, special_token)
        f.seek(0)
        data = f.read()
    return data, list(zip(boundaries[:-1], boundaries[1:]))

def _pretoken_worker(bounds: ChunkBounds) -> FrequencyMap:
    '''
    NOTE: Worker takes a slice of boundaries. working on them in a separate process.
    We first split each byte chunk on the special token, then decode each piece to
    text and run the GPT-style regex pre-tokenizer. The output frequency map is
    keyed by each matched pre-token encoded as UTF-8 bytes, which is the right
    representation for byte-level BPE training.
    '''
    start, end = bounds
    share = data[start:end]
    pid = os.getpid()
    preview = share[:80].decode("utf-8", errors="ignore").replace("\n", "\\n")
    LOGGER.debug(
        "worker %s processing bytes[%s:%s] %r",
        pid,
        start,
        end,
        preview,
    )

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    profiler = cProfile.Profile() if worker_profile_dir is not None else None
    if profiler is not None:
        profiler.enable()

    try:
        out: defaultdict[bytes, int] = defaultdict(int)
        for doc in share.split(split_token):
            doc_text = doc.decode("utf-8", errors="ignore")
            for t in re.findall(PAT, doc_text):
                out[t.encode("utf-8")] += 1
        return dict(out)
    finally:
        if profiler is not None:
            profiler.disable()
            profile_path = Path(worker_profile_dir) / f"pretoken-worker-{pid}-{start}-{end}.prof"
            profiler.dump_stats(profile_path)


def _pretoken_with_pool(
    shared_data: bytes,
    bounds_list: ChunkBoundsList,
    special_token: bytes,
    num_workers: int,
    profile_dir: str | None,
) -> FrequencyMap:
    '''
    NOTE: map reduce fashion. worker emit frequency map, main process consolidate into merged dict.
    '''
    ctx = mp.get_context("fork")
    if profile_dir is not None:
        Path(profile_dir).mkdir(parents=True, exist_ok=True)
    with ctx.Pool(
        num_workers,
        initializer=_init_shared_string,
        initargs=(shared_data, special_token, profile_dir),
    ) as p:
        results = p.map(_pretoken_worker, bounds_list)
        merged: FrequencyMap = {}
        for d in results:
            for k, v in d.items():
                merged[k] = merged.get(k, 0) + v
        return merged


def init_token_freqmap(
    filename: str,
    desired_num_chunks: int = 4,
    num_workers: int = 4,
    profile_dir: str | None = None,
    special_token: bytes = b"<|endoftext|>",
) -> FrequencyMap:
    start_time = time.perf_counter()
    LOGGER.info(
        "init_token_freqmap start filename=%s requested_chunks=%s requested_workers=%s",
        filename,
        desired_num_chunks,
        num_workers,
    )
    shared_data, bounds_list = _get_file_chunk_bounds(
        filename=filename,
        desired_num_chunks=desired_num_chunks,
        special_token=special_token,
    )
    freq_map = _pretoken_with_pool(
        shared_data,
        bounds_list,
        special_token,
        num_workers,
        profile_dir,
    )
    LOGGER.info(
        "init_token_freqmap finish filename=%s actual_chunks=%s workers=%s unique_tokens=%s elapsed=%.2fs",
        filename,
        len(bounds_list),
        num_workers,
        len(freq_map),
        time.perf_counter() - start_time,
    )
    return freq_map


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} <filename>")

    merged = init_token_freqmap(sys.argv[1])
    for k, v in sorted(merged.items(), key=lambda kv: kv[1], reverse=True):
        print(f"{k!r} -> {v}")
