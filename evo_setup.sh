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

set -euo pipefail

PROJECT_ROOT="${PWD}"
# The single --target passed to evo init; real multi-file scope is in project.md
TARGET_FILE="hierarchical_mil/model.py"

# ── Edit these paths ──────────────────────────────────────────────────────────
DATA_ROOT="/data/ibd_wsi"
FEATURE_ROOT="/data/ibd_wsi/features"
SPLIT_CSV="$DATA_ROOT/splits/train_val_test.csv"

# Proxy-run epochs per evo iteration — keep low for fast hill-climbing.
# Raise to 50 (or set EVO_FULL_EVAL=1) for a final validation run.
EVO_MAX_EPOCHS=10

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
    --metric    "val_auroc_macro"

# ── Runtime environment ───────────────────────────────────────────────────────
# Injected into every benchmark and gate process automatically.
evo env set EVO_DATA_ROOT    "$DATA_ROOT"
evo env set EVO_FEATURE_ROOT "$FEATURE_ROOT"
evo env set EVO_SPLIT_CSV    "$SPLIT_CSV"
evo env set EVO_MAX_EPOCHS   "$EVO_MAX_EPOCHS"
evo env set EVO_NUM_WORKERS  "4"

# ── project.md — the agent's authoritative scope document ────────────────────
cat > .evo/project.md << 'MDEOF'
## Project
IBD subtype classifier — UC vs CD — using MammothMIL (Mixture of Experts).

## Score
Macro AUROC over val set. Higher is better. 0.5 = random, baseline ≈ 0.72, target > 0.80.

## Files in scope (agents may edit these)
- `hierarchical_mil/model.py`    — top-level MoE, expert routing, gating
- `hierarchical_mil/experts.py`  — CellExpert, PatchExpert, RegionExpert
- `hierarchical_mil/fusion.py`   — fusion head / additive combination logic
- `hierarchical_mil/loss.py`     — loss terms, label smoothing, class weights

## Files out of scope (do not modify)
- `hierarchical_mil/data.py`     — data loading / splits (changing this invalidates comparison)
- `hierarchical_mil/trainer.py`  — training loop, optimizer, AMP config
- `evaluate.py`                  — benchmark script
- `gate.sh`                      — gate script
- `tests/`

## Known failure modes — avoid these
- `pooling="sum"` inflates logit magnitudes proportional to patch count →
  val_loss explodes, gate rejects on magnitude > 50. Use `pooling="mean"`.
- CellExpert must have `warmup_epochs` before it contributes signal; don't
  remove the warmup guard even if shortening the training run.
- `use_amp=False` is required on V100s — no bfloat16, and `scatter_add_`
  needs float32. Only set `use_amp=True` if you know the target GPU supports it.
- Fusion MLP co-adapts on the train/val split and hurts generalisation.
  Prefer additive combination with learned gating weights over a deep MLP.
- Train/val asymmetry in scale dropout caused diverging val loss in earlier
  runs — ensure any new dropout/noise layers are disabled in eval mode.

## Benchmark determinism
Sampling-based — variance expected across runs. Seed is fixed per worktree
via EVO_SEED (default 42) but stochastic training means ±0.01 AUROC noise
is normal. Discard experiments that differ by < 0.01 from parent.

## Future experiment candidates
- Attention-based pooling over cell tokens (replace mean pooling in CellExpert)
- Curriculum learning rate schedule tied to CellExpert warmup phase
- Contrastive auxiliary loss between UC/CD expert activations
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
