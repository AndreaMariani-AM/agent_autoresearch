#!/usr/bin/env bash
# evo_setup.sh — one-shot evo workspace initialisation for MammothMIL
#
# Run once from your project root:
#   bash /path/to/evo_setup.sh
#
# Prerequisites:
#   - Claude Code with the evo plugin installed
#       /plugin marketplace add evo-hq/evo
#       /plugin install evo@evo-hq-evo
#   - Python 3.12+, git, uv on PATH
#   - evaluate.py and gate.sh committed to your repo (evo forks from HEAD)
#   - GPU available on this machine (nvidia-smi should show a device)
# ─────────────────────────────────────────────────────────────────────────────

eval "$(mamba shell hook --shell bash)"
mamba activate /group/glastonbury/conda_envs/installer_tools

set -euo pipefail

PROJECT_ROOT="${PWD}"
# The single --target passed to evo init; real multi-file scope is in project.md
TARGET_FILE="src/models/Discriminator.py"

# ── Edit these paths ──────────────────────────────────────────────────────────
FEATURE_ROOT="/group/glastonbury/andrea/projects/IBD/IBD_predictive_model/data/features/extracted_features/hierarchical_embeddings/virchow2_128_hierarchical_full_cohorts"
SPLIT_CSV="/group/glastonbury/andrea/projects/IBD/IBD_predictive_model/data/folds/discriminator_folds/fold_0_predictions.csv"

# Proxy-run epochs per evo iteration — keep low for fast hill-climbing.
# Raise to 50 (or set EVO_FULL_EVAL=1) for a final validation run.
EVO_MAX_EPOCHS=15

# ─────────────────────────────────────────────────────────────────────────────

echo "=== evo workspace setup ==="
echo "  project root : $PROJECT_ROOT"
echo "  target file  : $TARGET_FILE"
echo ""

# Sanity check: evaluate.py and gate.sh must be committed (evo forks from HEAD)
for f in evaluate.py gate.sh; do
    git ls-files --error-unmatch "$f" 2>/dev/null || {
        echo "ERROR: $f is not committed to git. Commit it first — evo forks from HEAD." >&2
        exit 1
    }
done

# Sanity check: GPU present
nvidia-smi -L 2>/dev/null || {
    echo "WARNING: no GPU detected (nvidia-smi failed). evaluate.py will run on CPU." >&2
}

cd "$PROJECT_ROOT"

# ── Initialise evo workspace ──────────────────────────────────────────────────
# Benchmark: python runs evaluate.py directly from the worktree.
# Gate:      bash runs gate.sh from the worktree (CPU, < 30 s).
# Metric:    val_auroc_macro — the extra field we write alongside "score".
evo init \
    --target  "$TARGET_FILE" \
    --benchmark "python {worktree}/evaluate.py --worktree {worktree} --out {worktree}/.evo_result.json" \
    --gate      "bash {worktree}/gate.sh {worktree}" \
    --metric    "max" \
    --host      "claude-code"

# ── Runtime environment ───────────────────────────────────────────────────────
# Write a dotenv file and load it so every benchmark/gate process inherits
# these variables automatically.
cat > .evo/benchmark.env << ENVEOF
EVO_FEATURE_ROOT=$FEATURE_ROOT
EVO_SPLIT_CSV=$SPLIT_CSV
EVO_MAX_EPOCHS=$EVO_MAX_EPOCHS
EVO_NUM_WORKERS=4
ENVEOF

evo env load --all .evo/benchmark.env

# ── project.md — the agent's authoritative scope document ────────────────────
cat > .evo/project.md << 'MDEOF'
## Project
IBD subtype classifier — UC vs CD — using MammothMIL (Mixture of Experts).

## Score
Macro AUROC over val set. Higher is better. 0.5 = random, baseline ≈ 0.75, target > 0.85.

## Files in scope (agents may edit these)
- `src/models/Discriminator.py`    — top-level MoE architecture
- `src/training/trainer.py`        — Lightning trainer with loss function, optimizer and training logic
- `src/utils/implementations.py`   — scratchpad for notes and experimental code
- `scripts/train_discriminatorMIL.py`   — human training entry point (reference only; do not modify)

## Files out of scope (do not modify)
- `data/dataset.py`     — data loading / splits (changing this invalidates comparison)
- `evaluate.py`                  — benchmark script
- `gate.sh`                      — gate script
- `tests/`

## Known failure modes — avoid these
- Only set `use_amp=True` if you know the target GPU supports it.

## Benchmark determinism
Sampling-based — variance expected across runs. Seed is fixed per worktree
via EVO_SEED (default 42) but stochastic training means ±0.01 AUROC noise
is normal. Discard experiments that differ by < 0.03 from parent.

## Future experiment candidates
- Contrastive auxiliary loss between UC/CD expert activations
- Implementation of Contrstive MIL
- All optimizers and hyperparameters
- All activation functions
- All architectural details: skip connections, number of layers, etc.
- Different architectures
- Different ways to aggregated patch features to slide level prediction
MDEOF

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo ""
echo "  1. Validate baseline (run from Claude Code or terminal):"
echo "       evo run --timeout 3600"
echo "     This trains one proxy run and confirms the result file is written."
echo ""
echo "  2. Start optimising (open Claude Code, then):"
echo "       /evo:optimize subagents=2 budget=5 stall=4"
echo "     Use subagents=2 initially; increase to 3–5 once the loop is stable."
echo ""
echo "  3. Monitor:"
echo "       evo status"
echo "       evo dashboard   (browser UI on port 8080)"
echo ""
echo "  4. Promote a good experiment to main:"
echo "       evo show        # find the best exp_XXXX"
echo "       git checkout exp_XXXX"
