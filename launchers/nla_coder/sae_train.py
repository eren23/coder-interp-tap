"""Phase-2.1 SAE training launcher for nla_qwen2_5_coder_1_5b.

SCAFFOLD ONLY. Real implementation goes here when Phase 0 smoke is green.

Intended pipeline:
  1. Stream ~100M code tokens from CommitPackFT (Python + TS subsets).
  2. Forward through Qwen2.5-Coder-1.5B, hook layer 6 residuals.
  3. Train TopK SAE via SAELens (fallback: EleutherAI Sparsify) with
     d_model=1536, d_sae=24576, L0=50, aux_loss_coef=0.0625.
  4. Periodic eval: reconstruction MSE on held-out 1M tokens, dead-feature
     fraction, top-k feature stability across checkpoints.
  5. Checkpoint to W&B every CHECKPOINT_INTERVAL steps; resume-aware.

For now: defer to the smoke runner.
"""

from __future__ import annotations

from launchers.nla_coder.smoke import main


if __name__ == "__main__":
    raise SystemExit(main())
