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

## exp_0031 — Wider Encoder output_dim=512 and ABMIL key_dim=256
**Status:** NO_IMPROVEMENT
**Score:** 0.8312 AUROC  (parent: 0.8226, Δ=+0.009)
**Parent:** exp_0025
**What changed:** `src/models/Discriminator.py` — changed `output_dim` default from 256 to 512 in `MammothNet` (so `DeepResidualEncoder` outputs 512-dim instead of 256-dim), and changed `key_dim=128` to `key_dim=256` in `ABMILAggregator`. All other settings (2-block encoder, ReLU, dropout=0.25, LR=1e-4) unchanged.
**Notes:** Small positive delta (+0.009), just below the 0.01 noise floor threshold, but evo committed it as new best. val_loss similar (0.895 vs parent 0.932). Notably, f1_macro dropped (0.752→0.706) while AUROC improved — wider representation helps ranking but may affect class balance. Direction is promising: doubling output capacity adds marginal but consistent signal. Branching from exp_0031 for Round 14 to exploit the wider architecture further.

---

## exp_0030 — Reduced Encoder Dropout=0.1 (was 0.25)
**Status:** NO_IMPROVEMENT
**Score:** 0.8099 AUROC  (parent: 0.8226, Δ=−0.013)
**Parent:** exp_0025
**What changed:** `src/models/Discriminator.py` — changed `dropout=0.25` to `dropout=0.1` in the `DeepResidualEncoder(...)` instantiation inside `MammothNet.__init__`. ABMIL aggregator dropout unchanged at 0.25.
**Notes:** Regression (−0.013). Less dropout hurt. Matches exp_0012 pattern (dropout=0.1 from exp_0007 also regressed). val_loss slightly lower (0.856 vs parent 0.932) — better calibration at cost of ranking quality. The 2-block encoder benefits from dropout=0.25 regularization; lower dropout may lead to overfit patch embeddings. 0.25 is confirmed optimal dropout for this architecture.

---

## exp_0029 — Stronger Contrastive EMA Prototype Loss (lambda=0.3, alpha=0.99)
**Status:** NO_IMPROVEMENT
**Score:** 0.7962 AUROC  (parent: 0.8226, Δ=−0.026)
**Parent:** exp_0025
**What changed:** `src/training/trainer.py` — EMA prototype contrastive push with `_contrast_lambda=0.3` (was 0.1 in exp_0022) and `_proto_alpha=0.99` (was 0.9, much slower EMA). Same mechanism: push bag embedding away from opposite-class prototype via cosine similarity, added to CE loss. Gradient flows through the undetached representation; only EMA buffer update uses `.detach()`.
**Notes:** Larger regression than exp_0022 (−0.026 vs −0.0003). Pattern confirmed: EMA prototype contrastive loss consistently hurts AUROC at every lambda (0.1→0.809, 0.3→0.796), while reducing val_loss (0.932→0.784). The contrastive signal appears to conflict with AUROC-optimal representation learning — it pushes embeddings apart in cosine space, which may actually reduce the soft ranking signal ABMIL needs. val_loss reduction (better calibration) correlates with AUROC degradation — same anti-pattern seen with focal loss. Abandon EMA contrastive entirely; focus on encoder capacity or training dynamics.

---

## exp_0028 — GELU Activation in DeepResidualEncoder (replace ReLU)
**Status:** NO_IMPROVEMENT
**Score:** 0.8042 AUROC  (parent: 0.8226, Δ=−0.018)
**Parent:** exp_0025
**What changed:** `src/models/Discriminator.py` — changed `act_layer=nn.ReLU` to `act_layer=nn.GELU` in the `DeepResidualEncoder(...)` instantiation inside `MammothNet.__init__`. Both ResidualMLP blocks affected. All other settings (ABMIL, warmup/cosine LR, dropout=0.25, lr=1e-4) unchanged.
**Notes:** Regression (−0.018). GELU did not improve over ReLU in this 2-block residual architecture. val_loss slightly worse (0.952 vs parent 0.932). f1_macro unchanged (0.752). ReLU appears to be the better activation for this MIL patch-embedding projection task — possibly because the hard zero gate in ReLU acts as implicit feature selection, useful when compressing 2560-dim Virchow2 embeddings. GELU's smooth gradient may not add value when the main bottleneck is I/O and aggregation, not gradient flow through the encoder.

---

## exp_0026 — Three-Block Deep ResidualMLP Encoder (2560→1280→640→256)
**Status:** NO_IMPROVEMENT
**Score:** 0.7992 AUROC  (parent: 0.8226, Δ=−0.023)
**Parent:** exp_0025
**What changed:** `src/models/Discriminator.py` — added `ThreeBlockResidualEncoder` stacking three `ResidualMLP` blocks: block1=ResidualMLP(2560,1280,1280), block2=ResidualMLP(1280,640,640), block3=ResidualMLP(640,320,256). Replaced `DeepResidualEncoder` in `MammothNet`. ABMIL aggregator and warmup/cosine LR unchanged from exp_0025.
**Notes:** Regression (−0.023) — adding a third block hurt vs the 2-block best. Depth plateau confirmed: 1-block→0.795, 2-block→0.823, 3-block→0.799. The additional compression (2560→1280→640→256) introduces over-compression that degrades representation quality. val_loss dropped (0.932→0.641) as with the transformer, suggesting the deeper encoder over-fits to calibration at the expense of ranking. evo marked as `failed` due to 2h benchmark timeout (training completed but evo killed the process at exactly 7200s before reading the result file — score from `.evo_result.json`). Fix: use `--timeout 10800` for future runs. Sweet spot for this task appears to be the 2-block encoder (exp_0025). Next: explore encoder modifications on the 2-block base (activation function, dropout, output_dim) or auxiliary loss improvements.

---

## exp_0027 — Standard CLS-token transformer aggregator on DeepResidualEncoder (no RoPE)
**Status:** NO_IMPROVEMENT
**Score:** 0.7999 AUROC  (parent: 0.8226)
**Parent:** exp_0025
**What changed:** `src/models/Discriminator.py` — added `SimpleCLSTransformer` class (pre-norm `norm_first=True`, 2-layer, 4-head, `dim_feedforward=input_dim*4`, GELU, `batch_first=True`, CLS token init `trunc_normal_(std=0.02)`) and replaced `ABMILAggregator` with it in `MammothNet`. `DeepResidualEncoder` (2560→1280→256) and warmup+cosine LR schedule unchanged from exp_0025.
**Notes:** Clear regression (−0.022). CLS transformer peaked at epoch 9 (AUROC=0.7999, val_loss=0.655, f1_macro=0.679) and did not improve through epoch 20. Consistent with exp_0016/exp_0018 (RoPE transformer ~0.776-0.779) — CLS transformers consistently underperform ABMIL in this MIL AUROC task. Lower val_loss (0.655 vs parent 0.932) shows better calibration but worse slide-level ranking, confirming ABMIL's attention mechanism is better suited for ranking in WSI MIL. All 3 attempts showed the same ceiling (~0.79-0.80 AUROC). ABMIL remains the superior aggregator; next directions should focus on improving the encoder or auxiliary loss rather than replacing ABMIL.

---

## exp_0030 — Reduced encoder dropout=0.1 in DeepResidualEncoder (was 0.25)
**Status:** NO_IMPROVEMENT
**Score:** 0.8099 AUROC  (parent: 0.8226, Δ=−0.013)
**Parent:** exp_0025
**What changed:** `src/models/Discriminator.py` — changed `dropout=dropout` to `dropout=0.1` in the `DeepResidualEncoder(...)` instantiation inside `MammothNet.__init__`. ABMIL aggregator dropout kept at 0.25. All other settings unchanged (2-block ReLU encoder, output_dim=256, warmup/cosine LR, lr=1e-4, weight_decay=1e-4).
**Notes:** Regression (−0.013). Reducing encoder dropout from 0.25 to 0.1 hurt AUROC (0.810 vs parent 0.823). val_loss improved slightly (0.856 vs 0.932), f1_macro dropped (0.743 vs 0.752). The result is consistent with exp_0012 pattern: less dropout in the encoder does not help. The original dropout=0.25 provides better regularisation for this high-dimensional (2560→256) patch-level projection, preventing over-fitting to individual patch features and maintaining better bag-level ranking. evo discarded as NO_IMPROVEMENT. Next directions: explore LR tuning, different weight_decay, or aggregation improvements on the 2-block base.

---

## exp_0031 — Wider encoder output_dim=512 and ABMIL key_dim=256
**Status:** NO_IMPROVEMENT
**Score:** 0.8312 AUROC  (parent: 0.8226, Δ=+0.0086)
**Parent:** exp_0025
**What changed:** `src/models/Discriminator.py` — changed `output_dim` default in `MammothNet.__init__` from 256 to 512, and `key_dim` in `ABMILAggregator` instantiation from 128 to 256. `src/training/trainer.py` — changed `output_dim` default in `MammothTrainer.__init__` from 256 to 512. DeepResidualEncoder now projects 2560→1280→512 (was 2560→1280→256); ABMIL attention keys are 256-dim (was 128).
**Notes:** Marginal positive delta (+0.0086) below the 0.01 improvement threshold — classified NO_IMPROVEMENT. val_loss=0.895, f1_macro=0.706, epochs_run=20. Widening the encoder projection and attention keys gave a small positive signal but did not produce a clear improvement over exp_0025 (0.8226). The 256-dim output does not appear to be a strict information bottleneck at this dataset scale. Note: exp_0011 tried output_dim=512 from a single-layer MLP baseline (AUROC=0.789) — the 2-block encoder at 512-dim (0.831) is a better result but still within noise of exp_0025. Architecture appears to have plateaued at 0.82-0.83 with current training setup; activation, depth, width, and aggregator changes have all been exhausted without clear gains. Next: consider LR tuning, longer training, or data augmentation strategies.

---

## exp_0032 — Deeper ABMIL attention gates (V/U as 2-layer MLPs)
**Status:** NO_IMPROVEMENT
**Score:** 0.7958 AUROC  (parent: 0.8312, Δ=−0.035)
**Parent:** exp_0031
**What changed:** `src/models/Discriminator.py` — replaced single-linear `attention_V` and `attention_U` in `ABMILAggregator` with 2-layer MLPs: `Linear(in_dim,in_dim)→ReLU→Linear(in_dim,key_dim)→Tanh/Sigmoid`. Both gate projections now pass through a full-dim hidden layer before reducing to key_dim.
**Notes:** Significant regression (−0.035). Deeper gating projections strongly hurt AUROC. Adding a non-linear hidden layer to the attention gates destroys the soft ranking quality that ABMIL relies on. The original linear V/U design (a single linear map to key space) is likely optimal for this MIL task: the attention mechanism is intended to produce scalar weights ranking patches, and over-parameterising the key projection adds noise rather than discriminability. The ABMIL attention gates should remain simple linear projections; complexity should be invested upstream (encoder) not in the gating mechanism.

---

## exp_0033 — Wider encoder intermediate dimension (mid_features=2048, was 1280)
**Status:** NO_IMPROVEMENT
**Score:** 0.8301 AUROC  (parent: 0.8312, Δ=−0.0011)
**Parent:** exp_0031
**What changed:** `src/models/Discriminator.py` — changed the first `ResidualMLP` block's `mid_features` in `DeepResidualEncoder` from `hidden_dim` (1280) to 2048. The first block now expands from 2560 input before compressing: 2560→2048→1280 (was 2560→1280→1280). Second block unchanged: 1280→640→512.
**Notes:** Essentially no change (−0.0011, within noise). Widening the first encoder block from 1280 to 2048 intermediate dimensions provides no benefit — the architecture at 1280 intermediate already captures sufficient information from the 2560-dim Virchow2 embeddings. Adding extra parameters in the first projection does not help ranking quality. The encoder width is not the bottleneck; all reasonable architectural variations (depth, width, activation, dropout, gates) have now been exhausted without improving beyond the 0.83 plateau. The clear next direction is optimizer/LR schedule tuning.

---

## Round 14 Considerations

Architecture search is exhausted. After 14 rounds and 33 experiments, the model is firmly plateaued at **0.83 macro AUROC** with the current training setup (AdamW lr=1e-4, warmup+cosine schedule, 20 epochs). Key learnings:

**What works:**
- 2-block `DeepResidualEncoder` (2560→1280→512, ReLU, dropout=0.25) — depth sweet spot confirmed
- `ABMILAggregator` with simple linear V/U gates — ranking-oriented attention, superior to transformers
- Warmup (10%) + CosineAnnealingLR schedule — best over flat LR
- output_dim=512, key_dim=256 gave a marginal (+0.009) positive signal

**What definitively doesn't work:**
- More encoder depth (3+ blocks) — over-compresses, hurts ranking
- GELU activation — ReLU wins for this 2560-dim projection task
- Transformer aggregators (RoPE or CLS) — consistently ~0.02 below ABMIL
- Contrastive/focal auxiliary losses — calibration gains but AUROC regression
- Reduced dropout (0.1) — 0.25 is the optimal regularisation level
- Deeper ABMIL gates — simple linear projections are optimal
- Wider intermediate dimensions — 1280 intermediate is sufficient

**Strategic direction for next session:**
The plateau at 0.83 with current optimizer settings suggests the **training dynamics** are the limiting factor, not architecture. All architecture variations above the 2-block ReLU ABMIL baseline produce noise-level changes. The next logical exploration axis is **optimizer and LR tuning**: different base LR values (1e-3, 5e-4, 5e-5), weight decay (1e-3, 1e-5), alternative optimizers (SGD with momentum, Lion, AdamW with AMSGrad), and schedule shapes (linear decay, warmup fraction, cosine floor). A well-tuned optimizer can close the gap to >0.85 where architecture changes have failed.

---

## exp_0034 — Higher learning rate lr=5e-4 (5x increase)
**Status:** NO_IMPROVEMENT
**Score:** 0.8016 AUROC  (parent: 0.8312, Δ=−0.030)
**Parent:** exp_0031
**What changed:** `evo_config.json` — set `lr: 5e-4` (was 1e-4). All else unchanged: AdamW, weight_decay=1e-4, 10% warmup + cosine schedule, 2-block encoder + ABMIL.
**Notes:** Significant regression (−0.030). Higher LR destabilises training on this architecture. Consistent with exp_0009 (lr=3e-4, −0.009 from weaker base). The current lr=1e-4 appears well-tuned; going higher in any direction hurts. The next direction should be lower LR, different weight decay, or a different schedule shape.

---

## exp_0035 — Stochastic patch subsampling 70% per training step
**Status:** NO_IMPROVEMENT
**Score:** 0.8119 AUROC  (parent: 0.8312, Δ=−0.019)
**Parent:** exp_0031
**What changed:** `src/training/trainer.py` — in `MammothTrainer.training_step`, before the forward pass, randomly subsample 70% of the slide's patches: `n_keep = max(1, int(0.7 * features.size(0))); idx = torch.randperm(...); features = features[idx]`. Validation unchanged (full patches).
**Notes:** Regression (−0.019). ABMIL ranking quality degrades when patches are randomly removed during training — the model needs the full patch distribution to learn reliable attention weights. Consistent with exp_0013 (patch dropout 25% from weaker base, also regressed). Patch subsampling is not a useful augmentation for this MIL AUROC task: the ranking signal depends on rare informative patches that may be excluded by random subsampling.

---

## exp_0036 — Stronger weight decay wd=1e-3 (10x increase)
**Status:** NO_IMPROVEMENT
**Score:** 0.8318 AUROC  (parent: 0.8312, Δ=+0.0007)
**Parent:** exp_0031
**What changed:** `evo_config.json` — set `weight_decay: 1e-3` (was 1e-4). lr=1e-4 unchanged.
**Notes:** Essentially flat (+0.0007, within noise). evo auto-committed as marginally better. Stronger regularisation does not help — the plateau is not caused by over-fitting. The model is not benefiting from penalising weight magnitude more aggressively. Current wd=1e-4 is already well-tuned.

---

## exp_0037 — Label smoothing 0.1 in CrossEntropyLoss
**Status:** NO_IMPROVEMENT
**Score:** 0.8037 AUROC  (parent: 0.8312, Δ=−0.027)
**Parent:** exp_0031
**What changed:** `src/training/trainer.py` — `MammothTrainer.__init__`: `nn.CrossEntropyLoss()` → `nn.CrossEntropyLoss(label_smoothing=0.1)`. All other settings unchanged.
**Notes:** Clear regression (−0.027). Label smoothing hurts AUROC: softening targets reduces the sharpness of the decision boundary, which is detrimental to ranking quality. The model needs hard targets to learn discriminative slide-level representations. Consistent with earlier findings that calibration-oriented losses (focal, contrastive) harm AUROC.

---

## exp_0038 — Longer warmup 30% (6/20 epochs)
**Status:** NO_IMPROVEMENT
**Score:** 0.8216 AUROC  (parent: 0.8318, Δ=−0.010)
**Parent:** exp_0036
**What changed:** `src/training/trainer.py` — `configure_optimizers`: `int(0.1 * max_epochs)` → `int(0.3 * max_epochs)`. Warmup now runs 6 epochs (was 2), cosine decay over remaining 14 epochs.
**Notes:** Regression (−0.010). Longer warmup hurts: spending 6 epochs ramping up leaves only 14 epochs of useful cosine decay, effectively shortening productive training. The 2-epoch warmup (10%) is the right balance for this 20-epoch budget. The model converges quickly and doesn't benefit from a slower start.

---

## exp_0039 — AdamW amsgrad=True
**Status:** NO_IMPROVEMENT
**Score:** 0.8123 AUROC  (parent: 0.8318, Δ=−0.020)
**Parent:** exp_0036
**What changed:** `src/training/trainer.py` — `configure_optimizers`: `torch.optim.AdamW(..., amsgrad=True)`. Schedule and all hyperparameters unchanged.
**Notes:** Regression (−0.020). AMSGrad's conservative updates (running max of squared gradients) hurt rather than help — the standard AdamW second-moment estimate is better suited here. The plateau is not caused by unstable gradient variance; AMSGrad's extra conservatism reduces effective learning.

---

## exp_0040 — Lower learning rate lr=5e-5 (half of current)
**Status:** NO_IMPROVEMENT
**Score:** 0.8109 AUROC  (parent: 0.8318, Δ=−0.021)
**Parent:** exp_0036
**What changed:** `evo_config.json` — set `lr: 5e-5` (was 1e-4). weight_decay=1e-3 unchanged.
**Notes:** Regression (−0.021). Going lower hurts too. lr=1e-4 confirmed as the sweet spot in both directions: 5e-5 regresses (−0.021), 5e-4 regresses (−0.030). The optimizer is well-tuned; the plateau is not a learning rate problem.

---

## exp_0041 — Higher cosine floor eta_min=lr/10 (1e-5 instead of 1e-6)
**Status:** NO_IMPROVEMENT
**Score:** 0.8267 AUROC  (parent: 0.8318, Δ=−0.005)
**Parent:** exp_0036
**What changed:** `src/training/trainer.py` — `CosineAnnealingLR`: `eta_min=self.lr / 100` → `eta_min=self.lr / 10`. LR decays to 1e-5 minimum instead of 1e-6; all other settings unchanged.
**Notes:** Closest miss in the optimizer sweep (−0.005). Keeping LR warmer in later epochs does not help — the current cold floor (1e-6) is already optimal. The small regression suggests the final-epoch LR is fine as-is. Optimizer/schedule axis is now exhausted.

---

## exp_0042 — Pairwise AUROC surrogate loss (buffer-based, lambda=0.1)
**Status:** NO_IMPROVEMENT
**Score:** 0.8209 AUROC  (parent: 0.8318, Δ=−0.011)
**Parent:** exp_0036
**What changed:** `src/training/trainer.py` — added rolling buffer (size 30, reset each epoch) of detached (score, label) pairs. At each training step, computes sigmoid-smoothed pairwise ranking loss against buffer items with opposite label: `loss += 0.1 * mean(1 - sigmoid(margin))`. For n_classes=2, score = `logits[1] - logits[0]`.
**Notes:** Regression (−0.011). The pairwise ranking signal at lambda=0.1 conflicts with CE optimisation — same anti-pattern as EMA contrastive loss (exp_0022/0029). The model may already be implicitly ranking well via CE; adding an explicit ranking term introduces a competing gradient. Could try lower lambda (0.01) or higher (0.5) but consistent pattern suggests auxiliary ranking losses don't help this task.

---

## exp_0043 — Bag-level dropout 0.5 on aggregated slide embedding
**Status:** NO_IMPROVEMENT
**Score:** 0.8061 AUROC  (parent: 0.8318, Δ=−0.026)
**Parent:** exp_0036
**What changed:** `src/models/Discriminator.py` — added `self.bag_dropout = nn.Dropout(0.5)` to `ABMILAggregator.__init__`; applied between bag aggregation and classifier: `self.cls(self.bag_dropout(bag))`. Patch-level dropout (0.25) unchanged.
**Notes:** Clear regression (−0.026). Dropout 0.5 on the 512-dim bag embedding is too aggressive — drops too much signal at the slide level. The bag representation is the only information the classifier sees; heavy dropout here prevents it from learning reliable slide-level features. 

---

## exp_0044 — ABMIL attention temperature T=0.5 (sharper patch selection)
**Status:** IMPROVEMENT
**Score:** 0.8521 AUROC  (parent: 0.8318, Δ=+0.020)
**Parent:** exp_0036
**What changed:** `src/models/Discriminator.py` — changed `ABMILAggregator.forward` to use `torch.softmax(A / 0.5, dim=0)` instead of `torch.softmax(A, dim=0)`. Temperature=0.5 sharpens the attention distribution, concentrating weight on fewer, more discriminative patches.
**Notes:** Clear improvement (+0.020). Sharpening attention beyond the default (T=1.0) helps the model focus on a smaller subset of highly relevant patches. New best. Explore lower T (0.3, 0.2) and combining T=0.5 with other improvements from exp_0044 as the new parent.

---

## exp_0050 — Top-k hard attention masking (top 20% patches, T=0.5)
**Status:** PENDING
**Score:** — AUROC  (parent: 0.8521)
**Parent:** exp_0044
**What changed:** `src/models/Discriminator.py` — after `softmax(A/0.5)`, keep only the top 20% patches by attention score (`k = max(1, int(0.2 * N))`), zero the rest, and re-normalise by sum. Hard-selection extreme of T=0.5 soft sharpening.
**Notes:** Running.

---

## exp_0051 — Wider ABMIL attention key projection key_dim=512 (was 256)
**Status:** PENDING
**Score:** — AUROC  (parent: 0.8521)
**Parent:** exp_0044
**What changed:** `src/models/Discriminator.py` — changed `key_dim=256` to `key_dim=512` in `ABMILAggregator` instantiation inside `MammothNet`. Doubles the dimension of V and U projections, giving the attention mechanism more expressive power to compute patch compatibility scores. T=0.5 unchanged.
**Notes:** Running.

---

## exp_0048 — Multi-head ABMIL (4 heads, T=0.5, bag = concatenation)
**Status:** NO_IMPROVEMENT
**Score:** 0.8260 AUROC  (parent: 0.8521, Δ=−0.026)
**Parent:** exp_0044
**What changed:** `src/models/Discriminator.py` — replaced single-head `ABMILAggregator` with 4 independent attention heads (each with its own V/U/w projections and T=0.5 softmax). Bag embeddings from all heads concatenated (512×4=2048-dim) before a linear classifier.
**Notes:** Regression (−0.026). Multiple attention heads add noise rather than complementary signal — the 4× larger classifier input (2048-dim) introduces too many parameters for this dataset size, and the independent heads likely converge to overlapping representations. Single focused head remains better for this MIL task.

---

## exp_0049 — Attention dropout p=0.1 after softmax (train-only)
**Status:** NO_IMPROVEMENT
**Score:** 0.8234 AUROC  (parent: 0.8521, Δ=−0.029)
**Parent:** exp_0044
**What changed:** `src/models/Discriminator.py` — in `ABMILAggregator.forward`, after `torch.softmax(A / 0.5, dim=0)`, during training: randomly zero out 10% of attention weights then re-normalise by sum. Forces the bag representation to be robust to losing any single high-attention patch.
**Notes:** Regression (−0.029). Attention dropout fights directly against the T=0.5 sharpening — T=0.5 concentrates weight on a small set of patches, then dropout randomly removes some of those, undoing the selective focus. The two mechanisms are antagonistic.

---

## exp_0046 — ABMIL attention temperature T=0.3 (sharper than T=0.5)
**Status:** NO_IMPROVEMENT
**Score:** 0.8204 AUROC  (parent: 0.8521, Δ=−0.032)
**Parent:** exp_0044
**What changed:** `src/models/Discriminator.py` — changed `ABMILAggregator.forward` to use `torch.softmax(A / 0.3, dim=0)`. Temperature=0.3 sharpens attention further beyond T=0.5, concentrating weight on even fewer patches.
**Notes:** Clear regression (−0.032). T=0.3 is too aggressive — over-concentrating on very few patches loses important contextual signal. T=0.5 is the sweet spot: sharper than default (T=1.0) but not so sharp that it collapses to near-argmax selection. Temperature direction now exhausted.

---

## exp_0047 — Learnable ABMIL attention temperature
**Status:** NO_IMPROVEMENT
**Score:** 0.8452 AUROC  (parent: 0.8521, Δ=−0.007)
**Parent:** exp_0044
**What changed:** `src/models/Discriminator.py` — added `self.log_temperature = nn.Parameter(torch.log(torch.tensor(0.5)))` to `ABMILAggregator.__init__`; temperature computed as `F.softplus(self.log_temperature).clamp(min=1e-4)` in `forward`, replacing the fixed 0.5. Model learns optimal sharpness during training.
**Notes:** Within noise floor (−0.007). The learnable T converged close to its init (T≈0.5) but couldn't beat the fixed value — gradient signal through the temperature parameter is too weak to push past the manual optimum. Fixed T=0.5 remains best; no benefit from making it learnable.

---

## exp_0045 — Pairwise AUROC surrogate loss lambda=0.01 (vs 0.1 in exp_0042)
**Status:** NO_IMPROVEMENT
**Score:** 0.8123 AUROC  (parent: 0.8318, Δ=−0.020)
**Parent:** exp_0036
**What changed:** `src/training/trainer.py` — same rolling-buffer pairwise ranking loss as exp_0042 but with `_ranking_lambda=0.01` (10× weaker). Epoch buffer reset via `on_train_epoch_start`. Score = `logits[1] - logits[0]` for n_classes=2.
**Notes:** Regression (−0.020). AUROC surrogate loss consistently hurts regardless of lambda (0.01 or 0.1). The ranking signal conflicts with CE regardless of scale. Discard this direction.

---
