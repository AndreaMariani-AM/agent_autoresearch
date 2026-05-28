## Project
WSI classifier

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
via EVO_SEED (default 42) but stochastic training means ±0.005 AUROC noise
is normal. Discard experiments that differ by < 0.01 from parent.

## Future experiment candidates
- Different architectures for feature encoding. Right now is either a MLP of Mammoth (Mixture of Experts)for each patch. We can try different encoder architectures
- Different ways to aggregated patch features to slide level prediction, right now is additive mil where we sum all contributions to create logits. We should explore different ways to aggregate to slide level
# - Implementation of Contrstive MIL 
# - All optimizers and hyperparameters
# - All activation functions
# - All architectural details: skip connections, number of layers, etc.
