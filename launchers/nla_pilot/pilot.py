"""Phase-1 pilot launcher for nla_qwen3_5_2b_pilot.

SCAFFOLD ONLY. Real implementation goes here when Phase 0 smoke is green.

Intended pipeline:
  1. residual_dump   — forward 200K tokens through Qwen3.5-2B, hook layer 12,
                       cache (token_id, residual) to data/qwen3_5_2b_residuals_layer12.h5
  2. feature_extract — load Qwen-Scope SAE for layer 12, compute TopK feature
                       activations, cache to data/qwen3_5_2b_features_layer12.h5
  3. synthetic_desc  — for each of 32K features, generate ≤25-word description
                       via Claude Sonnet over top-firing tokens. Cost-gated.
  4. av_train        — train AV (residual + feature_idx → description) for
                       AV_TRAIN_STEPS steps, with periodic eval + checkpoints
  5. eval_pass       — manual quality rubric (50 features, 1-5 scale),
                       BERTScore distinctiveness, round-trip cosine

For now: import the smoke runner so pod plumbing still works on a `pilot` variant
until the real loop lands.
"""

from __future__ import annotations

from launchers.nla_pilot.smoke import main


if __name__ == "__main__":
    raise SystemExit(main())
