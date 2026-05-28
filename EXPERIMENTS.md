# Experiment Log

Tracks all evo experiments. Agents append an entry after each run.
Status tags: `IMPROVEMENT` · `NO_IMPROVEMENT` · `FAILED`

Baseline: **0.768 macro AUROC** (exp_0003, 10 epochs)
Target: **> 0.85**

---

## exp_0003 — Baseline
**Status:** BASELINE
**Score:** 0.768 AUROC
**Parent:** root
**What changed:** Unmodified MammothMIL MoE classifier. 10 epochs, lr=0.0001, weight_decay=1e-05, AdamW, no scheduler. Additive MIL aggregation.
**Notes:** val_auroc_uc and val_auroc_cd are NaN (torchmetrics multiclass issue — use macro AUROC only). val_loss=0.576 after 10 epochs suggests room for convergence improvement.

---

## exp_0004 — Warmup + Cosine LR Schedule
**Status:** IMPROVEMENT
**Score:** 0.784 AUROC (parent: 0.768, Δ=+0.016)
**Parent:** exp_0003
**What changed:** `trainer.py` — replaced fixed LR with `SequentialLR`: 1-epoch linear warmup (lr/10 → lr) then cosine annealing (lr → lr/100) over remaining epochs. `evo_config.json` — weight_decay increased 10× from 1e-05 to 1e-04.
**Notes:** New best. val_loss increased (0.576 → 0.686) despite AUROC gain — LR schedule may be too aggressive for 10 epochs. Good candidate for further tuning: longer warmup, softer cosine floor, or different weight_decay.

---

## exp_0005 — ABMIL Gated Attention Aggregation
**Status:** NO_IMPROVEMENT
**Score:** 0.776 AUROC (parent: 0.768, Δ=+0.008 — within 0.01 noise floor)
**Parent:** exp_0003
**What changed:** `src/models/Discriminator.py` — added `ABMILAggregator` class replacing `ClassConditionalAdditiveMIL`. Gated attention → softmax-weighted bag embedding → linear classifier. Decouples patch selection from classification. Dropout=0.25, key_dim=128.
**Notes:** AUROC marginally up (+0.008) but val_loss increased 0.576 → 0.678, suggesting the attention aggregator may not be converging as well. delta below noise floor — discard. Could revisit with better initialisation or longer training.

---

## exp_0006 — Softer Cosine Floor + 2-Epoch Warmup
**Status:** NO_IMPROVEMENT
**Score:** 0.7845 AUROC (parent: 0.7841, Δ=+0.0003)
**Parent:** exp_0004
**What changed:** `trainer.py` — extended linear warmup to 2 epochs, softened cosine annealing floor from lr/100 to lr/10. Warmup epochs set to max(2, 20% of max_epochs).
**Notes:** Essentially flat vs parent (Δ=+0.0003, well below noise floor). LR schedule tuning direction is now saturated — both variations (exp_0004→exp_0006 and exp_0004→exp_0007) confirm no further gain from schedule adjustments alone at 10 epochs. Need a more substantial change.

---

## exp_0007 — ABMIL Aggregation + Warmup/Cosine LR (stacked on exp_0004)
**Status:** NO_IMPROVEMENT
**Score:** 0.789 AUROC (parent: 0.784, Δ=+0.005)
**Parent:** exp_0004
**What changed:** `src/models/Discriminator.py` — added `ABMILAggregator` class and switched `MammothNet` from `ClassConditionalAdditiveMIL` to `ABMILAggregator`. Inherits exp_0004's warmup+cosine LR schedule and weight_decay=1e-4 unchanged.
**Notes:** Small positive movement (+0.005) but below the 0.01 threshold. val_loss improved vs exp_0004 (0.663 vs 0.686), suggesting ABMIL convergence does benefit from the warmer LR schedule. The additive synergy hypothesis is not confirmed at 10 epochs — ABMIL alone on exp_0003 gave +0.008, and stacking on exp_0004 gives only +0.005 over that higher base. Both changes individually and jointly fall below the noise floor. May require more epochs or architectural changes to break past 0.80.

---

## exp_0008 — Label Smoothing 0.1 on CrossEntropyLoss
**Status:** NO_IMPROVEMENT
**Score:** 0.7862 AUROC (parent: 0.7889, Δ=-0.0027)
**Parent:** exp_0007
**What changed:** `src/training/trainer.py` — changed `nn.CrossEntropyLoss()` to `nn.CrossEntropyLoss(label_smoothing=0.1)` in `MammothTrainer.__init__` for the multi-class branch.
**Notes:** Slight regression vs parent (Δ=-0.0027, well within noise floor). val_loss=0.649; label smoothing did not reduce overconfidence enough to improve AUROC. Only 10 epochs ran, so the regularization effect may not have had time to manifest. Not harmful but no clear benefit at this epoch budget — label smoothing alone is insufficient to break past the 0.79 plateau.

---

## exp_0009 — Higher Base LR 3e-4 with Warmup+Cosine
**Status:** NO_IMPROVEMENT
**Score:** 0.7780 AUROC (parent: 0.7889, Δ=−0.011)
**Parent:** exp_0007
**What changed:** `evo_config.json` — lr increased from 1e-4 to 3e-4. Warmup+cosine LR schedule and ABMIL aggregation inherited unchanged from exp_0007.
**Notes:** Score regressed by −0.011 vs parent. A 3× higher base LR appears too aggressive for this ABMIL architecture at 10 epochs — val_loss also increased (0.683 vs 0.663). LR direction is now confirmed: the optimal is at or below 1e-4, not above. Next priority: explore architectural changes (skip connections, deeper/wider experts) or contrastive auxiliary losses rather than LR tuning.

---
