#!/usr/bin/env bash
# gate.sh — evo gate for MammothMIL
#
# Runs on the same machine immediately after benchmark passes,
# before evo commits the experiment. Must complete in < 30 s.
#
# Configure in evo:
#   evo config set gate "bash {worktree}/gate.sh {worktree}"
#
# Exit 0 = passed. Non-zero = discard this experiment.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
WORKTREE="${1:?usage: gate.sh <worktree_path>}"
RESULT="$WORKTREE/.evo_result.json"

# ── 1. Result file + score range ─────────────────────────────────────────────
[[ -f "$RESULT" ]] || { echo "[gate] FAIL: result file missing" >&2; exit 1; }

python3 - << PYEOF
import json, sys
with open("$RESULT") as f:
    d = json.load(f)
score = d.get("score")
if score is None:
    sys.exit("[gate] FAIL: missing 'score' field")
try:
    s = float(score)
except (ValueError, TypeError):
    sys.exit(f"[gate] FAIL: score not numeric: {score!r}")
if s < 0.45:
    sys.exit(f"[gate] FAIL: score={s:.4f} below 0.45 (worse than random)")
print(f"[gate] score={s:.4f}  OK")
PYEOF

# ── 2. Import check ───────────────────────────────────────────────────────────
python3 - << PYEOF
import sys
sys.path.insert(0, "$WORKTREE")
sys.path.insert(0, "$WORKTREE/src")
try:
    from models.Discriminator import MammothNet
    from models.experts_MIL import ExpertOutput
    print("[gate] imports OK")
except Exception as exc:
    raise SystemExit(f"[gate] FAIL: import error: {exc}")
PYEOF

# ── 3. CPU forward-pass: shape, NaN, logit magnitude ─────────────────────────
python3 - << PYEOF
import sys, torch
sys.path.insert(0, "$WORKTREE")
sys.path.insert(0, "$WORKTREE/src")
from models.Discriminator import MammothNet

model = MammothNet(num_classes=2)
model.eval()

N_PATCHES  = 4
N_FEATURES = 2560  # Virchow2 CLS+mean embedding dim
N_CLASSES  = 2     # UC / CD

with torch.no_grad():
    feats  = torch.randn(1, N_PATCHES, N_FEATURES)
    output = model(feats)   # returns ExpertOutput
    logits = output.logits  # shape (N_CLASSES,)

assert logits.shape[-1] == N_CLASSES, \
    f"[gate] FAIL: expected logits [..., {N_CLASSES}], got {tuple(logits.shape)}"

if torch.isnan(logits).any():
    raise SystemExit("[gate] FAIL: NaN in logits")
if torch.isinf(logits).any():
    raise SystemExit("[gate] FAIL: Inf in logits")

mag = logits.abs().max().item()
if mag > 50.0:
    raise SystemExit(
        f"[gate] FAIL: logit magnitude={mag:.1f} > 50 — "
        "check pooling='sum' or unnormalised attention weights"
    )

print(f"[gate] forward pass OK  shape={tuple(logits.shape)}  max_abs={mag:.3f}")
PYEOF

echo "[gate] all checks passed"
exit 0
