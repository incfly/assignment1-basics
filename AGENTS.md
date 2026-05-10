# Agent Notes

This repo keeps large datasets outside the checkout. Default local data lives at:

```bash
../cs336-data
```

On the GCP training VM, use:

```bash
/mnt/disks/openweb-data/cs336-data
```

Initialize datasets with:

```bash
DATA_ROOT=/mnt/disks/openweb-data/cs336-data ./scripts/init_data.sh
```

The data init script downloads TinyStories plus Stanford's smaller CS336 OpenWebText sample (`owt_train.txt.gz` and `owt_valid.txt.gz`). It does not recreate the full 37 GB Skylion OpenWebText dump.

## Linux Setup

On a fresh Ubuntu/Debian VM:

```bash
./scripts/setup_linux.sh
```

To also initialize data during setup:

```bash
DATA_ROOT=/mnt/disks/openweb-data/cs336-data INSTALL_DATA=1 ./scripts/setup_linux.sh
```

The setup script installs apt dependencies, installs `uv` if missing, runs `uv sync`, builds RE2, builds the Python `cs336_basics.re_cpp._re_cpp` extension, and smoke-tests BPE if `tiny-1000.txt` exists.

## BPE Defaults

`cs336_basics/bpe/merge.py` defaults to:

```text
pretoken_workers = os.cpu_count()
pretoken_chunks = 4 * pretoken_workers
regex_mode = cpp
```

The CLI intentionally exposes only `--pretoken-worker`; chunk count is derived from the worker count.

`bpe-perf.sh` intentionally keeps a hardcoded default:

```bash
PRETOKEN_WORKER=8
```

Do not change that script to auto-infer workers unless explicitly requested.

## RE2 Notes

Linux uses:

```bash
./scripts/bootstrap_re2_linux.sh
./scripts/build_re_cpp_linux.sh
```

`bootstrap_re2_linux.sh` pins RE2 to `2023-03-01` because Debian 12's packaged Abseil is too old for newer RE2 tags. It also builds RE2 with `CMAKE_POSITION_INDEPENDENT_CODE=ON` so Python can link the native extension.

## VM Sync

Minimal code sync uses:

```bash
./scripts/rsync_vm.sh push
./scripts/rsync_vm.sh pull
```

`.rsyncignore` excludes data, virtualenvs, build outputs, `.git`, local RE2 checkout, native `.so` files, tarballs, and profiler output.

Current VM convention:

```text
instance: bpe-trainer-64
zone: us-west1-b
repo: /home/incfly/assignment1-basics
data: /mnt/disks/openweb-data/cs336-data
```
