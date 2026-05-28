# CLAUDE.md — IBD autoresearch / evo optimization

## Experiment log (EXPERIMENTS.md)

After every `evo run`, append an entry to `EXPERIMENTS.md` in the project root. Use this template:

```markdown
## <exp_id> — <short title>
**Status:** IMPROVEMENT | NO_IMPROVEMENT | FAILED
**Score:** <score> AUROC  (parent: <parent_score>)
**Parent:** <parent_exp_id>
**What changed:** <one or two sentences — what file, what mechanism, what values>
**Notes:** <anything surprising, failure modes, what to try next>
```

- `IMPROVEMENT` = score > parent + 0.03
- `NO_IMPROVEMENT` = ran but score ≤ parent + 0.03
- `FAILED` = benchmark errored or timed out

The orchestrator also updates pending entries once results are known.

## Conda environment

Correct path: `/group/glastonbury/conda_envs/lazyslide.v0.9.3`

Common mistake: the path has been entered incorrectly before as `/groups/glastonbury/conda_env/lazyslide.v0.9.3` (plural "groups", singular "conda_env"). Always use `conda run -p /group/glastonbury/conda_envs/lazyslide.v0.9.3`.

## Benchmark command stdout contract

evo expects stdout to be a single JSON object (the score). `evaluate.py` prints training logs + JSON to stdout, which breaks evo's parser. The benchmark command must redirect evaluate.py stdout to stderr and then cat the result file:

```bash
python {worktree}/evaluate.py --worktree {worktree} --out {worktree}/.evo_result.json >&2 && cat {worktree}/.evo_result.json
```

Training output lands in `benchmark_err.log`; `benchmark.log` contains only the clean JSON result. This is already set in `evo_setup.sh`.

## After workspace reset: re-set the runtime prefix

**Critical:** Deleting `.evo/run_*` directories and resetting `meta.json` wipes the runtime prefix from the run config. After any workspace reset, always re-run:

```bash
evo config runtime set --prefix "conda run -p /group/glastonbury/conda_envs/lazyslide.v0.9.3"
```

Without this, `evaluate.py` runs under the system Python (no torch → `ModuleNotFoundError: No module named 'torch'`). Verify with `evo config runtime show`.

## Running `evo run`

`evo run <exp_id>` is long-running — it trains the model for `EVO_MAX_EPOCHS` epochs (default 15), which takes ~10–20 minutes on GPU. Key rules:

- **Never stack calls.** Before running, check `ps aux | grep "evo run"`. If a process is already alive, wait for it or kill it first — don't launch another one.
- **Track progress via log files**, not CLI stdout. The benchmark log is at:
  `.evo/run_0001/experiments/<exp_id>/attempts/<n>/benchmark.log`
  and errors at `benchmark_err.log`.
- **Use a long timeout.** Always pass `--timeout 7200` to `evo run` (2 hours). The default is 30 min which is not enough for 10+ epochs (~40 min per run). Use `run_in_background=true` and poll the log file.
- **`max_attempts` is 3.** Each failed `evo run` consumes one attempt. If an experiment has used all 3, discard it and create a new one with `evo new`.

## Baseline setup

The evo workspace needs a committed baseline before the frontier opens and subagents can run. Without it, `evo frontier` returns an empty list and `/evo:optimize` cannot proceed.

To check: `evo status` — look for `committed >= 1`. If `committed=0`, the baseline hasn't run yet.

To run the baseline: `evo run exp_0001` (or whatever the active experiment is). Wait for it to commit before spawning subagents.

## Environment variables

Set in `.evo/benchmark.env`:
- `EVO_FEATURE_ROOT` — path to extracted Virchow2 hierarchical features
- `EVO_SPLIT_CSV` — fold CSV for train/val split
- `EVO_MAX_EPOCHS` — proxy training epochs per evo iteration (default 15)
- `EVO_NUM_WORKERS` — dataloader workers (default 4)

## Files in scope for editing

- `src/models/Discriminator.py` — MoE architecture
- `src/training/trainer.py` — Lightning trainer, loss, optimizer
- `src/utils/implementations.py` — experimental scratchpad

Do **not** edit: `data/dataset.py`, `evaluate.py`, `gate.sh`, `tests/`.

## Optimization targets

Metric: macro AUROC on val set. Baseline ≈ 0.75, target > 0.85. Discard experiments within ±0.03 of parent (noise floor).

Promising directions (not yet tried):
- Contrastive auxiliary loss between UC/CD expert activations
- Contrastive MIL
- Optimizer / LR / scheduler tuning
- Activation functions
- Skip connections, number of layers
- Different aggregation strategies for patch → slide prediction
