# TinyStories LR Tuning Plan

Goal: compare peak learning rates for the same TinyStories Transformer run and choose the LR with the best validation loss curve under a fixed token budget.

We sweep only peak LR first because the schedule is cosine decay. For each trial, set `optimizer.lr` to the candidate peak, `optimizer.min_lr = lr / 10`, warm up for 100 steps, then decay to min LR by `max_iters`. This makes each candidate a complete short training run with the same compute budget.

Candidates:

```text
1e-4, 3e-4, 1e-3, 3e-3, 1e-2
```

Run from scratch for each LR. Do not resume from a checkpoint during the sweep. Use CUDA, TinyStories tokenized train/valid bins, `max_iters=3000`, `eval_every=100`, `eval_iters=20`, and `checkpoint_every=0` to save disk.

RunPod entry point:

```bash
cd /workspace/assignment1-basics
PYTHONPATH=. python3 cmd/sweep_lr.py -c cs336_basics/config.toml \
  --lrs 1e-4 3e-4 1e-3 3e-3 1e-2 \
  --max-iters 3000 --warmup-iters 100 \
  --eval-every 100 --eval-iters 20 \
  --checkpoint-every 0 --device cuda \
  --run-root runs/lr-sweep-$(date +%Y%m%d-%H%M%S)
```

Evaluate with `metrics.jsonl` per run. Primary comparison is final validation loss. Also check whether a run diverged, how quickly validation drops, and whether validation begins rising. Plot curves with:

```bash
PYTHONPATH=. python3 cmd/plot_metrics.py --runs-root runs/<sweep-dir>
```
