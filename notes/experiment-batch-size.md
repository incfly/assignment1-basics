# Batch Size Experiment Handoff

This note is for a future Codex session to run the batch-size experiment on RunPod.

## Context

Previous LR sweeps used:

```text
batch_size = 32
context_length = 256
max_iters = 3000
tokens = 32 * 256 * 3000 = 24,576,000
device = cuda
dtype = fp32
```

Best observed peak LR was `1.5e-3` from `runs/lr-refine-20260518-062114/lr-2e-03/config.toml`.
That folder name is misleading from an old bug; the config says `lr = 0.0015`.

Use fixed LR first:

```text
optimizer.lr = 0.0015
optimizer.min_lr = 0.00015
optimizer.warmup_iters = 100
optimizer.cosine_cycle_iters = max_iters for each batch
training.device = "cuda"
training.checkpoint_every = 0
```

## Batch Sizes

Use a fixed token budget matching the LR sweep.

```text
target_tokens = 24,576,000
context_length = 256
max_iters = target_tokens / (batch_size * context_length)
```

Planned runs:

```text
B=4    max_iters=24000
B=8    max_iters=12000
B=16   max_iters=6000
B=32   max_iters=3000
B=64   max_iters=1500
B=96   max_iters=1000
```

Optional smoke only:

```text
B=128  max_iters=750
```

RTX 4070 12 GB may be tight at `128`; run one step before committing.

## Required Code Step

Add a minimal batch sweep script, probably `cmd/sweep_batch_size.py`.

It should mirror `cmd/sweep_lr.py`, but override:

```text
training.batch_size
training.max_iters
optimizer.lr
optimizer.min_lr
optimizer.cosine_cycle_iters
training.run_root
training.run_name = batch-<B>
training.resume_from removed
```

It should accept:

```bash
--batch-sizes 4 8 16 32 64 96
--target-tokens 24576000
--lr 0.0015
--min-lr-ratio 0.1
--device cuda
```

Keep it simple. No W&B. Use existing `cmd/train.py` metrics logging.

## RunPod Workflow

Preferred RunPod target for this experiment:

```text
First choice: RTX 4070, if available.
Similar small-GPU fallback: NVIDIA L4 in US-MO-2.
Other close fallbacks: RTX A4000, RTX 4000 Ada, RTX A5000.
```

As of 2026-05-30, RTX 4070 was not rentable from RunPod even when requested
directly. An RTX A4000 rented successfully, but landed in Sweden and the
TinyStories tokenized train bin upload was too slow. A US L4 in `US-MO-2`
rented successfully and is a reasonable 24 GB fallback for batch-size 16.

For best practical availability from the US, try regions in this order:

```text
US-CA-2  broad workstation/high-end catalog; try RTX 4000 Ada or RTX A5000
US-IL-1  broad workstation catalog; try RTX A5000 or RTX 4090 if needed
US-MO-2  L4 available; good small-GPU fallback
US-GA-1  RTX A4000 advertised, but rental may be stale
US-KS-2  broad high-end catalog if cost is less important
```

If those fail, try `EU-RO-1`, `EUR-IS-1`, or `EUR-IS-2` for small GPU
availability. Avoid distant regions for this data-heavy workflow unless no US
region can rent a suitable GPU; SSH upload of the 1.08 GB train bin can dominate
the experiment.

Create a fallback pod explicitly, for example:

```bash
./scripts/runpod.py create \
  --gpu-id "NVIDIA L4" \
  --cloud-type SECURE \
  --data-center-ids US-MO-2
```

TinyStories is staged in GCS:

```text
gs://cs336-artifacts/lab1/tinystory
```

On a pod, prefer pulling TinyStories from GCS instead of uploading over SSH:

```bash
mkdir -p /workspace/cs336-data/tinystory
gcloud storage rsync gs://cs336-artifacts/lab1/tinystory /workspace/cs336-data/tinystory --recursive
```

Prepare pod:

```bash
./scripts/runpod.py setup
```

Launch in tmux on the pod:

```bash
./scripts/runpod.py ssh
cd /workspace/assignment1-basics
RUN_ID="batch-size-$(date +%Y%m%d-%H%M%S)"
SESSION="${RUN_ID//-/_}"
mkdir -p logs
printf "%s\n" "$RUN_ID" > logs/latest_batch_size.txt
tmux new-session -d -s "$SESSION" \
  "cd /workspace/assignment1-basics && PYTHONUNBUFFERED=1 PYTHONPATH=. python3 -u cmd/sweep_batch_size.py -c cs336_basics/config.toml --batch-sizes 4 8 16 32 64 96 --target-tokens 24576000 --lr 0.0015 --device cuda --run-root runs/$RUN_ID 2>&1 | tee logs/$RUN_ID.log"
```

Check progress:

```bash
tail -f logs/$RUN_ID.log
tmux ls
find runs/$RUN_ID -name metrics.jsonl -print
```

Pull artifacts:

```bash
./scripts/runpod.py pull --artifacts-only --run-id <RUN_ID>
```

Stop pod immediately:

```bash
./scripts/runpod.py stop
```

## Evaluation

Plot two views:

```text
validation loss vs tokens_seen
validation loss vs elapsed_sec
```

Also report:

```text
best validation loss
final validation loss
tokens/sec = run_tokens_seen / elapsed_sec
whether training looks noisy or unstable
```

Interpretation target:

```text
Small batches may learn with noisier gradients and poor GPU utilization.
Large batches may be faster per token, but can generalize worse or need LR retuning.
If a large batch is worse, rerun a small LR sweep around it before declaring it bad.
```
