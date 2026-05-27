"""
evaluate.py — GPU benchmark for MammothMIL autoresearch with evo.

Called directly by evo as the benchmark command (no Slurm wrapper needed):
  evo config set benchmark "python {worktree}/evaluate.py --worktree {worktree} --out {worktree}/.evo_result.json"

The only contract with evo:
  - Write {"score": <float>} to --out on success.
  - Exit 0 on success, non-zero on failure (evo marks FAILED and retries).

Score metric: macro AUROC over {UC, CD} via MammothTrainer's val_AUROC
(multiclass, torchmetrics). Higher is better, robust to class imbalance.

Training uses n_classes=2 + CrossEntropyLoss throughout.

Fast proxy mode: EVO_MAX_EPOCHS (default 10) keeps each iteration cheap.
Set EVO_MAX_EPOCHS=50 or EVO_FULL_EVAL=1 for a full validation run.
"""

from __future__ import annotations

import argparse
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
# evo checks out each subagent's edits into a fresh worktree. Inserting both
# the root and src/ subdirectory means `from src.X import Y` and the legacy
# `from models.X import Y` style used inside trainer/Discriminator both resolve
# from that experiment's code, not from the installed/main-branch copy.

def activate_worktree(worktree: Path) -> None:
    for p in (str(worktree), str(worktree / "src")):
        if p not in sys.path:
            sys.path.insert(0, p)


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

    feature_root = Path(os.getenv("EVO_FEATURE_ROOT", cfg.get("feature_root", "/data/ibd_wsi/features")))
    split_csv    = Path(os.getenv("EVO_SPLIT_CSV",    cfg.get("split_csv",    "/data/ibd_wsi/splits/fold_0_predictions.csv")))

    return {
        "worktree":       worktree,
        "feature_root":   feature_root,
        "split_csv":      split_csv,
        "max_epochs":     max_epochs,
        "lr":             float(cfg.get("lr", 1e-4)),
        "weight_decay":   float(cfg.get("weight_decay", 1e-5)),
        "moe_args":       cfg.get("moe_args", {}),
        "use_amp":        bool(cfg.get("use_amp", False)),
        "num_workers":    int(os.getenv("EVO_NUM_WORKERS", "4")),
        "seed":           int(cfg.get("seed", 42)),
        "checkpoint_dir": worktree / ".evo_checkpoints",
        "full_eval":      full_eval,
    }


# ── Training + evaluation ─────────────────────────────────────────────────────

def run_training(cfg: dict) -> dict:
    import torch
    import lightning as L
    from lightning.pytorch.callbacks import ModelCheckpoint
    from torch.utils.data import DataLoader
    from src.training.trainer import MammothTrainer
    from src.data.dataset import MILDataset

    L.seed_everything(cfg["seed"], workers=True)

    device_info = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[evaluate] device={device_info}  epochs={cfg['max_epochs']}", flush=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_dataset = MILDataset(
        csv_path=str(cfg["split_csv"]),
        representation_dir=str(cfg["feature_root"]),
        split="train",
        use_discriminator=True,
    )
    val_dataset = MILDataset(
        csv_path=str(cfg["split_csv"]),
        representation_dir=str(cfg["feature_root"]),
        split="val",
        use_discriminator=True,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = MammothTrainer(
        n_classes=2,
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        moe_args=cfg["moe_args"],
    )

    # ── Checkpoint ────────────────────────────────────────────────────────────
    ckpt_dir = cfg["checkpoint_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        monitor="val_AUROC",
        mode="max",
        save_top_k=1,
        filename="best",
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    precision = "16-mixed" if cfg["use_amp"] else "32-true"

    trainer = L.Trainer(
        max_epochs=cfg["max_epochs"],
        accelerator="auto",
        devices=1,
        precision=precision,
        callbacks=[ckpt_cb],
        logger=False,
        enable_checkpointing=True,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
    )

    trainer.fit(model, train_loader, val_loader)

    # ── Extract metrics from best checkpoint ──────────────────────────────────
    best_path = ckpt_cb.best_model_path
    if best_path:
        best_ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        val_auroc = float(best_ckpt.get("val_AUROC", float("nan")))
        val_loss  = float(best_ckpt.get("val_loss",  float("nan")))
        f1_macro  = float(best_ckpt.get("val_F1",    float("nan")))
    else:
        # No improvement recorded — fall back to last epoch values
        val_auroc = float(model.current_val_AUROC)
        val_loss  = float(model.current_val_loss)
        f1_macro  = float(model.current_val_F1)

    return {
        "val_auroc_macro": val_auroc,
        "val_auroc_uc":    float("nan"),   # requires a separate predict pass
        "val_auroc_cd":    float("nan"),
        "val_loss":        val_loss,
        "f1_macro":        f1_macro,
        "epochs_run":      cfg["max_epochs"],
    }


# ── Sanity checks (run before committing result) ──────────────────────────────

def sanity_checks(metrics: dict) -> None:
    auroc = metrics["val_auroc_macro"]
    if not (0.0 <= auroc <= 1.0):
        raise ValueError(f"AUROC out of range: {auroc}")
    if metrics["val_loss"] != metrics["val_loss"]:
        raise ValueError("val_loss is NaN — likely logit explosion")
    if metrics["val_loss"] > 5.0:
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
    print(f"[evaluate] config={json.dumps({k: str(v) for k, v in train_cfg.items() if k not in ('worktree', 'moe_args')}, indent=2)}", flush=True)

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
