"""
evaluate.py — GPU benchmark for HierarchicalMIL autoresearch with evo.

Called directly by evo as the benchmark command (no Slurm wrapper needed):
  evo config set benchmark "python {worktree}/evaluate.py --worktree {worktree} --out {worktree}/.evo_result.json"

The only contract with evo:
  - Write {"score": <float>} to --out on success.
  - Exit 0 on success, non-zero on failure (evo marks FAILED and retries).

Score metric: macro AUROC over {UC, CD}. Higher is better, robust to class
imbalance, and avoids the logit-inflation trap that makes val_loss unreliable.

Fast proxy mode: EVO_MAX_EPOCHS (default 10) keeps each iteration cheap.
Set EVO_MAX_EPOCHS=50 or EVO_FULL_EVAL=1 for a full validation run.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import traceback
from pathlib import Path


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--worktree", required=True, type=Path,
                   help="Absolute path to the evo experiment worktree")
    p.add_argument("--out", required=True, type=Path,
                   help="Path to write the JSON result file")
    return p.parse_args()


# ── Dynamic import from the worktree ─────────────────────────────────────────
# evo checks out each subagent's edits into a fresh worktree. Inserting it
# at the front of sys.path means `import hierarchical_mil` resolves from
# that experiment's code, not from the installed/main-branch copy.

def activate_worktree(worktree: Path) -> None:
    wt_str = str(worktree)
    if wt_str not in sys.path:
        sys.path.insert(0, wt_str)


# ── Config ────────────────────────────────────────────────────────────────────

def load_experiment_config(worktree: Path) -> dict:
    """
    Priority: evo_config.json (evo overrides) > config.json (project config).
    Adapt keys to your actual config schema.
    """
    for fname in ("evo_config.json", "config.json"):
        p = worktree / fname
        if p.exists():
            with p.open() as f:
                return json.load(f)
    return {}


def build_train_config(cfg: dict, worktree: Path) -> dict:
    max_epochs = int(os.getenv("EVO_MAX_EPOCHS", cfg.get("max_epochs", 10)))
    full_eval  = os.getenv("EVO_FULL_EVAL", "0") == "1"

    data_root    = Path(os.getenv("EVO_DATA_ROOT",    cfg.get("data_root",    "/data/ibd_wsi")))
    feature_root = Path(os.getenv("EVO_FEATURE_ROOT", cfg.get("feature_root", "/data/ibd_wsi/features")))
    split_csv    = Path(os.getenv("EVO_SPLIT_CSV",    cfg.get("split_csv",    str(data_root / "splits/train_val_test.csv"))))

    cell_expert_warmup = int(cfg.get("cell_expert_warmup_epochs", 2))
    # Ensure CellExpert fires at least once before we measure
    if max_epochs <= cell_expert_warmup and not full_eval:
        max_epochs = cell_expert_warmup + 2

    return {
        "worktree":           worktree,
        "data_root":          data_root,
        "feature_root":       feature_root,
        "split_csv":          split_csv,
        "max_epochs":         max_epochs,
        "batch_size":         int(cfg.get("batch_size", 8)),
        "lr":                 float(cfg.get("lr", 1e-4)),
        "cell_expert_warmup": cell_expert_warmup,
        "use_amp":            cfg.get("use_amp", False),   # set True if your GPU supports bf16
        "num_workers":        int(os.getenv("EVO_NUM_WORKERS", "4")),
        "seed":               int(cfg.get("seed", 42)),
        "checkpoint_dir":     worktree / ".evo_checkpoints",
        "full_eval":          full_eval,
    }


# ── Training + evaluation ─────────────────────────────────────────────────────

def run_training(cfg: dict) -> dict:
    import torch
    import numpy as np
    from sklearn.metrics import roc_auc_score, f1_score

    # Dynamic imports from the worktree — adjust paths to your module layout
    try:
        model_mod   = importlib.import_module("hierarchical_mil.model")
        trainer_mod = importlib.import_module("hierarchical_mil.trainer")
        data_mod    = importlib.import_module("hierarchical_mil.data")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Import failed from worktree {cfg['worktree']}. "
            f"Check module layout. Error: {exc}"
        ) from exc

    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[evaluate] device={device}  epochs={cfg['max_epochs']}", flush=True)

    # Data — adapt to your actual DataModule / DataLoader interface
    train_loader, val_loader = data_mod.get_loaders(
        split_csv=cfg["split_csv"],
        feature_root=cfg["feature_root"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        seed=cfg["seed"],
    )

    # Model — adapt to your HierarchicalMIL constructor signature
    model = model_mod.HierarchicalMIL(
        cell_expert_warmup_epochs=cfg["cell_expert_warmup"],
    ).to(device)

    # Trainer
    trainer = trainer_mod.Trainer(
        model=model,
        device=device,
        lr=cfg["lr"],
        use_amp=cfg["use_amp"],
        checkpoint_dir=cfg["checkpoint_dir"],
    )

    best_auroc    = 0.0
    best_val_loss = float("inf")

    for epoch in range(1, cfg["max_epochs"] + 1):
        train_loss = trainer.train_epoch(train_loader, epoch=epoch)
        val_loss, logits, labels = trainer.val_epoch(val_loader)

        logits_np = logits.cpu().numpy()   # (N, 2)
        labels_np = labels.cpu().numpy()   # (N,) in {0, 1}

        try:
            auroc = roc_auc_score(labels_np, logits_np, multi_class="ovr", average="macro")
        except ValueError:
            auroc = 0.5   # single class in batch — too early to measure

        print(f"[evaluate] epoch={epoch:03d}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  auroc={auroc:.4f}", flush=True)

        if auroc > best_auroc:
            best_auroc    = auroc
            best_val_loss = val_loss
            trainer.save_checkpoint("best.pt")

    # Re-evaluate best checkpoint for final per-class numbers
    trainer.load_checkpoint("best.pt")
    _, logits, labels = trainer.val_epoch(val_loader)
    logits_np = logits.cpu().numpy()
    labels_np = labels.cpu().numpy()
    preds     = logits_np.argmax(axis=1)

    f1_macro = f1_score(labels_np, preds, average="macro", zero_division=0)
    try:
        auroc_uc = roc_auc_score((labels_np == 0).astype(int), logits_np[:, 0])
        auroc_cd = roc_auc_score((labels_np == 1).astype(int), logits_np[:, 1])
    except ValueError:
        auroc_uc = auroc_cd = float("nan")

    return {
        "val_auroc_macro": float(best_auroc),
        "val_auroc_uc":    float(auroc_uc),
        "val_auroc_cd":    float(auroc_cd),
        "val_loss":        float(best_val_loss),
        "f1_macro":        float(f1_macro),
        "epochs_run":      cfg["max_epochs"],
    }


# ── Sanity checks (run before committing result) ──────────────────────────────

def sanity_checks(metrics: dict) -> None:
    auroc = metrics["val_auroc_macro"]
    if not (0.0 <= auroc <= 1.0):
        raise ValueError(f"AUROC out of range: {auroc}")
    if metrics["val_loss"] != metrics["val_loss"]:
        raise ValueError("val_loss is NaN — likely logit explosion")
    if metrics["val_loss"] > 20.0:
        raise ValueError(
            f"val_loss={metrics['val_loss']:.2f} suspiciously large — "
            "check pooling='sum' or logit magnitude"
        )


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args    = parse_args()
    worktree: Path = args.worktree.resolve()
    out: Path      = args.out.resolve()

    out.parent.mkdir(parents=True, exist_ok=True)
    activate_worktree(worktree)

    experiment_cfg = load_experiment_config(worktree)
    train_cfg      = build_train_config(experiment_cfg, worktree)

    print(f"[evaluate] worktree={worktree}", flush=True)
    print(f"[evaluate] config={json.dumps({k: str(v) for k, v in train_cfg.items() if k != 'worktree'}, indent=2)}", flush=True)

    try:
        metrics = run_training(train_cfg)
        sanity_checks(metrics)
    except Exception:
        # Write a partial result for evo's error trace, then exit non-zero
        out.write_text(json.dumps({"score": 0.0, "error": traceback.format_exc()}, indent=2))
        raise

    result = {"score": metrics["val_auroc_macro"], **metrics}

    tmp = out.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(result, indent=2))
    tmp.replace(out)   # atomic rename — avoids evo reading a partial file

    print(f"[evaluate] result written → {out}")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
