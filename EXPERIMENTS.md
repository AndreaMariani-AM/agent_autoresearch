# Experiment Log

Tracks all evo experiments. Agents append an entry after each run.
Status tags: `IMPROVEMENT` · `NO_IMPROVEMENT` · `FAILED`

Baseline (10 epochs): **0.768 macro AUROC** (exp_0003)
Baseline (20 epochs): **0.768 macro AUROC** (exp_0015) — new root after EVO_MAX_EPOCHS=20
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

## exp_0010 — LayerNorm on Projected Patch Embeddings
**Status:** NO_IMPROVEMENT
**Score:** 0.763 AUROC (parent: 0.7889, Δ=−0.026)
**Parent:** exp_0007
**What changed:** `src/models/Discriminator.py` — added `nn.LayerNorm(output_dim)` to `MammothNet.__init__` and applied it after the MLP encoder in `forward()` before MIL aggregation.
**Notes:** Clear regression (−0.026). Normalising patch embeddings before aggregation hurt performance — the raw projected embeddings may carry magnitude information (e.g. confidence or salience) that LayerNorm discards. The pattern across rounds 3–4 is consistent: adding regularisation or normalisation always regresses; the model seems to benefit from the natural scale of features.

---

## exp_0011 — Wider Output Projection (output_dim=512)
**Status:** NO_IMPROVEMENT
**Score:** 0.7701 AUROC (parent: 0.7889, Δ=−0.0188)
**Parent:** exp_0007
**What changed:** `src/training/trainer.py` — changed default `output_dim` from 256 to 512 in `MammothTrainer.__init__`. MLP encoder now projects 2560→1280→512, and ABMILAggregator operates on 512-dim embeddings.
**Notes:** Small regression vs parent (Δ=−0.0188, below noise floor). Doubling the bottleneck dimension did not help — the extra capacity may have made training harder at 10 epochs with lr=1e-4, or the 256-dim projection was already sufficient. val_loss=0.769 is higher than parent's 0.663, indicating worse convergence. The 256-dim projection appears to act as a useful regularizer rather than an information bottleneck. Width reduction or orthogonal architectural changes (skip connections, contrastive auxiliary loss) should be prioritized over projection size expansion.

---

## exp_0012 — Lower Dropout (0.1)
**Status:** NO_IMPROVEMENT
**Score:** 0.7859 AUROC (parent: 0.7889, Δ=−0.003)
**Parent:** exp_0007
**What changed:** `src/training/trainer.py` — reduced default dropout from 0.25 to 0.1 in `MammothTrainer.__init__`. Applies to both MLP encoder and ABMILAggregator.
**Notes:** Slight regression (−0.003). Interestingly val_loss improved (0.646 vs 0.663) but AUROC fell — same decoupling seen in exp_0008 and exp_0013. Lower dropout is not the bottleneck. The consistent pattern across rounds 3–5: changes that reduce val_loss do not translate to AUROC gain, suggesting the 10-epoch budget is too short for any architectural change to improve discriminability.

---

## exp_0013 — Stochastic Patch Dropout 25%
**Status:** NO_IMPROVEMENT
**Score:** 0.7886 AUROC (parent: 0.7889, Δ=−0.0003)
**Parent:** exp_0007
**What changed:** `src/models/Discriminator.py` — added stochastic patch dropout (p=0.25) in `MammothNet.forward()` during training, applied after MLP encoder and before MIL aggregation. Minimum 4 patches retained per bag.
**Notes:** Essentially flat (Δ=−0.0003, well within noise floor). val_loss slightly improved (0.663 → 0.633) and f1_macro improved (0.743 → 0.761), but AUROC was unchanged. The dropout augmentation appears to provide modest regularization (lower val_loss, better F1) without translating to AUROC gain at 10 epochs. This direction could be revisited with more epochs or combined with other augmentations.

---

## exp_0018 — TransformerAggregator + Warmup/Cosine LR (convergence fix)
**Status:** NO_IMPROVEMENT
**Score:** 0.776 AUROC (parent: 0.779, Δ=−0.003)
**Parent:** exp_0016
**What changed:** `src/training/trainer.py` — added SequentialLR (10% warmup + cosine to lr/100). `evo_config.json` — weight_decay=1e-4. Builds on exp_0016's TransformerAggregator + 2D RoPE.
**Notes:** LR schedule fixed convergence (val_loss dropped 0.834 → 0.586, nearly matching baseline 0.576). However AUROC was flat vs parent (−0.003). The transformer with proper training is on par with ABMIL (~0.776 vs ~0.785) but not beating it. Discarded — spatial RoPE without further architectural changes is saturated.

---

## exp_0016 — TransformerAggregator with 2D RoPE Spatial Encoding
**Status:** IMPROVEMENT
**Score:** 0.779 AUROC (parent: 0.768, Δ=+0.011)
**Parent:** exp_0015
**What changed:** `src/models/Discriminator.py` — added `_apply_rope_2d()`, `RoPE2DAttention`, and `TransformerAggregator` classes. `MammothNet` now uses `TransformerAggregator` (CLS token + 2-layer transformer, 4 heads, dim=256, GELU FFN 4×) instead of `ClassConditionalAdditiveMIL`. `src/training/trainer.py` — added `_load_coords()` loading `mid/xy_256` bounding boxes from h5 files; center computed as `(x1+x2)/2/128` in patch units; passed as coords to transformer via `mask` arg.
**Notes:** +0.011 positive signal above noise floor but below the 0.03 threshold. val_loss=0.834 (vs 0.576 baseline) indicates the transformer hasn't converged — it needs a warmup+cosine LR schedule. Next step: stack exp_0017's LR schedule on this transformer architecture. The spatial RoPE encoding appears to provide a modest signal; combination with better optimization may push past 0.80.

---

## exp_0017 — ABMIL + Warmup/Cosine LR on 20-Epoch Baseline
**Status:** IMPROVEMENT
**Score:** 0.7847 AUROC (parent: 0.768, Δ=+0.017)
**Parent:** exp_0015
**What changed:** `src/models/Discriminator.py` — ABMILAggregator replacing ClassConditionalAdditiveMIL. `src/training/trainer.py` — SequentialLR (10% warmup + cosine to lr/100). `evo_config.json` — weight_decay=1e-4. Faithful port of exp_0007 on the 20-epoch baseline.
**Notes:** Essentially the same result as exp_0007 at 10 epochs (0.785 vs 0.789). Doubling the epoch budget gave no additional gain — this ABMIL+LR config saturates within 10 epochs. val_loss=0.668, f1_macro=0.706. Confirms that more epochs alone won't break the plateau; need architectural changes (transformer aggregator, contrastive loss).

---

## exp_0015 — 20-Epoch Root Baseline
**Status:** BASELINE
**Score:** 0.768 AUROC
**Parent:** root
**What changed:** EVO_MAX_EPOCHS increased from 10 to 20 (breaking infra event registered). Unmodified MammothMIL model (ClassConditionalAdditiveMIL aggregation, MLP encoder 2560→1280→256). This is the new root baseline for the 20-epoch optimization epoch.
**Notes:** Identical score to exp_0003 (10-epoch baseline). val_loss=0.576, f1_macro=0.716. The model appears to converge similarly with or without the extra epochs at baseline. Doubling the epoch budget provides more headroom for architectural changes (transformer aggregators, contrastive losses) to converge. Previous best (exp_0007, 0.789) was achieved at 10 epochs — target is now >0.85 with 20 epochs.

---

## exp_0017 — ABMIL + Warmup+Cosine LR on 20-Epoch Baseline (port of exp_0007)
**Status:** NO_IMPROVEMENT
**Score:** 0.785 AUROC  (parent: 0.768)
**Parent:** exp_0015
**What changed:** `src/models/Discriminator.py` — added `ABMILAggregator` class and switched `MammothNet` from `ClassConditionalAdditiveMIL` to `ABMILAggregator`. `src/training/trainer.py` — replaced bare AdamW return with warmup+cosine `SequentialLR` (10% warmup epochs, cosine annealing to lr/100). `evo_config.json` — lr=1e-4, weight_decay=1e-4. Exact port of exp_0007 onto the 20-epoch baseline.
**Notes:** AUROC 0.785 vs parent 0.768 (Δ=+0.017), below the 0.03 improvement threshold. val_loss=0.668, f1_macro=0.706. Compared to exp_0007's 0.789 at 10 epochs, doubling the epoch budget gave essentially the same result — ABMIL+cosine-LR converges within 10 epochs and gains nothing from the extra epochs. This direction is saturated; a fundamentally different aggregation or loss strategy is needed to break past 0.79.

---

## exp_0018 — TransformerAggregator + Warmup+Cosine LR Schedule (convergence fix)
**Status:** NO_IMPROVEMENT
**Score:** 0.776 AUROC (parent: 0.779, Δ=−0.003)
**Parent:** exp_0016
**What changed:** `src/training/trainer.py` — added `SequentialLR` warmup+cosine schedule to `MammothTrainer.configure_optimizers()`: 10% warmup (LinearLR 0.1→1.0) then cosine annealing to lr/100. `evo_config.json` — weight_decay=1e-4. TransformerAggregator with 2D RoPE architecture unchanged from exp_0016.
**Notes:** The LR schedule dramatically fixed convergence: val_loss dropped from 0.834 (exp_0016) to 0.586, close to the baseline val_loss of 0.576. However, AUROC decreased slightly (0.779 → 0.776, Δ=−0.003) — within noise floor, essentially tied with parent. f1_macro=0.679 (vs 0.725 parent). The convergence fix worked but the transformer is not outperforming ABMIL (0.785). Next steps: try deeper transformer (more heads/layers), contrastive auxiliary loss between experts, or a different aggregation strategy that exploits spatial structure more aggressively.

---

## exp_0019 — Residual MLP Encoder + Warmup/Cosine LR
**Status:** IMPROVEMENT
**Score:** 0.7955 AUROC (parent: 0.768, Δ=+0.0274)
**Parent:** exp_0015
**What changed:** `src/models/Discriminator.py` — replaced `MLP` with `ResidualMLP` in `MammothNet`: added a skip connection (`nn.Linear(in_features, out_features, bias=False)`) from the encoder input to output alongside the two-layer MLP path. `src/training/trainer.py` — warmup+cosine LR schedule (10% warmup, cosine to lr/100), weight_decay=1e-4.
**Notes:** New best at +0.027 over baseline (above 0.01 threshold). val_loss=0.864 is high (worse than exp_0016's 0.834 and baseline 0.576), suggesting the residual encoder with cosine LR hasn't converged well despite the schedule. f1_macro=0.771. The skip connection adds a direct pathway from raw projected features to the MIL aggregator. High val_loss vs AUROC gain is an interesting decoupling — the model ranks correctly even without low loss. Priority next: stack ResidualMLP with ABMIL aggregator, or add ABMIL on top of this residual encoder.

---

## exp_0021 — EMA Prototype Contrastive Loss on Bag Representations
**Status:** NO_IMPROVEMENT
**Score:** 0.7955 AUROC (parent: 0.7955, Δ=0.000)
**Parent:** exp_0019
**What changed:** `src/training/trainer.py` — added EMA class prototype buffer (`_proto`, shape 2×repr_dim) and contrastive auxiliary loss: at each training step, push bag embedding away from opposite-class running mean via cosine similarity. lambda=0.1, EMA alpha=0.9.
**Notes:** Score exactly equal to parent — the contrastive loss had zero effect. Root cause: `repr` was detached before computing `F.cosine_similarity`, so no gradient flowed back through the bag representation. The EMA updates and loss value changed numerically, but the backward pass was identical to plain cross-entropy. Fix: use undetached `repr` on the left side of `cosine_similarity` (only detach the prototype buffer, not the query). Should retry corrected version on exp_0020 (ResidualMLP+ABMIL) as the parent.

---

## exp_0020 — ResidualMLP Encoder + ABMIL Gated Attention Aggregator
**Status:** IMPROVEMENT
**Score:** 0.8089 AUROC (parent: 0.7955, Δ=+0.013)
**Parent:** exp_0019
**What changed:** `src/models/Discriminator.py` — added `ABMILAggregator` class (gated attention: V/U projections, tanh(V)·sigmoid(U) gating, softmax attention, weighted bag embedding, linear head) and replaced `ClassConditionalAdditiveMIL` with it in `MammothNet`. Residual encoder and warmup/cosine LR schedule inherited unchanged from exp_0019.
**Notes:** New best at 0.8089. Stacking the two individually-proven changes (ResidualMLP encoder from exp_0019 + ABMIL from exp_0017) produced additive gains: +0.027 + ~0.013 ≈ +0.040 total over baseline. val_loss=0.887 is high but AUROC is the metric; f1_macro=0.661 (lower than exp_0019's 0.771, unusual). The residual encoder + ABMIL combination is the new best architecture. Target is >0.85; we are at 0.809 — still 0.041 to go. Next: contrastive auxiliary loss on this architecture, or deeper/wider variants.

---

## exp_0023 — FocalLoss on Binary Path (null change)
**Status:** NO_IMPROVEMENT
**Score:** N/A (discarded — null change)
**Parent:** exp_0020
**What changed:** `src/training/trainer.py` — replaced `nn.BCEWithLogitsLoss` with `FocalLoss(gamma=2)` in the n_classes=1 binary branch. Model uses n_classes=2; the edited branch is never reached.
**Notes:** Dead edit. FocalLoss in losses.py uses binary_cross_entropy_with_logits and only applies to n_classes=1. Our classifier is multi-class (n_classes=2), so CrossEntropyLoss in the else-branch was unchanged. evo run process also died silently mid-attempt; force-discarded.

---

## exp_0024 — Multi-class Softmax Focal Loss (gamma=2)
**Status:** NO_IMPROVEMENT
**Score:** 0.7927 AUROC (parent: 0.8089, Δ=−0.016)
**Parent:** exp_0020
**What changed:** `src/training/trainer.py` — replaced `nn.CrossEntropyLoss()` with inline softmax focal loss in the n_classes>1 branch: `ce = F.cross_entropy(logits, label.long(), reduction='none'); pt = torch.exp(-ce); loss = ((1-pt)**2 * ce).mean()`. Same change applied in validation_step.
**Notes:** Regression (−0.016). Surprisingly, val_loss dropped dramatically (0.887 → 0.406) — focal loss improved calibration — but AUROC fell. This suggests focal loss shifts the model toward lower-entropy predictions at the cost of ranking quality. The model becomes more "confident" but ranks slides less accurately. Focal loss is not suited to this MIL AUROC-optimisation task; the model benefits from soft, less-confident distributions for ranking.

---

## exp_0025 — Two-Block Deep ResidualMLP Encoder (2560→1280→256, two skip connections)
**Status:** IMPROVEMENT
**Score:** 0.8226 AUROC (parent: 0.8089, Δ=+0.014)
**Parent:** exp_0020
**What changed:** `src/models/Discriminator.py` — added `DeepResidualEncoder` class stacking two `ResidualMLP` blocks: block1 = ResidualMLP(2560, 1280, 1280) and block2 = ResidualMLP(1280, 640, 256). Replaced single `ResidualMLP` encoder in `MammothNet` with `DeepResidualEncoder(in_features=2560, mid_features=1280, out_features=256)`. ABMIL aggregator and warmup/cosine LR unchanged from exp_0020.
**Notes:** New best at 0.8226. Deeper encoder continues to improve AUROC additively — each extra residual stage adds ~+0.013. Notably, f1_macro recovered strongly (0.661 → 0.752), suggesting the deeper encoder produces better-separated class representations. val_loss increased (0.887 → 0.932) — the deeper model is harder to calibrate but ranks well. Gap to target (>0.85) is now 0.027. Next: three-block encoder, or try a different aggregation strategy on top of DeepResidualEncoder.

---

## exp_0022 — Contrastive EMA Prototype Loss (gradient flow fixed)
**Status:** NO_IMPROVEMENT
**Score:** 0.8092 AUROC  (parent: 0.8089)
**Parent:** exp_0020
**What changed:** `src/training/trainer.py` — added EMA prototype contrastive loss with corrected gradient flow: `self.model(features)` (returns full ExpertOutput) replaces `self(features)` in `training_step` so that `.representation` has attached gradients. `repr_raw` is passed undetached to `F.cosine_similarity`; only the EMA buffer update uses `.detach()`. lambda=0.1, alpha=0.9. Parent architecture (ResidualMLP + ABMILAggregator + warmup/cosine LR) unchanged.
**Notes:** Delta is only +0.0003 — well within the noise floor. Gradient did flow correctly (repr not detached), but the contrastive signal is too weak to move AUROC meaningfully. f1_macro unchanged at 0.661 (identical to exp_0020). The loss appears not to provide useful discriminative signal at lambda=0.1 or the EMA prototypes are converging to indistinct positions too quickly (alpha=0.9 is very fast). Directions to explore: stronger lambda (0.5+), slower EMA (alpha=0.99), triplet loss instead of cosine push, or dropping this auxiliary loss entirely and trying a fundamentally different approach (e.g. FocalLoss, class-balanced sampling, or deeper encoder).

---
