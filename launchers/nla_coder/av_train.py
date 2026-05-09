"""Phase-2.2 AV training launcher for nla_qwen2_5_coder_1_5b.

SCAFFOLD ONLY. Real implementation goes here when Phase 2.1 SAE
training is producing usable feature activations.

Intended pipeline:
  1. Load saved Phase-2.1 SAE checkpoint.
  2. For each of d_sae features, generate synthetic NL description via
     Claude Sonnet over top-firing tokens. Cap by DESCRIPTION_BUDGET_USD.
  3. Train AV (residual + feature_idx → description) for AV_TRAIN_STEPS,
     with W&B periodic eval (BERTScore distinctiveness, round-trip cosine).
  4. Final manual rubric pass: rate 50 random feature descriptions 1-5.

For now: defer to the smoke runner.
"""

from __future__ import annotations

from launchers.nla_coder.smoke import main


if __name__ == "__main__":
    raise SystemExit(main())
