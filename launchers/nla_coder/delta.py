"""Phase-3 LoRA delta study launcher for nla_qwen2_5_coder_1_5b.

SCAFFOLD ONLY. Real implementation goes here once Phase 2 (SAE + AV) ships.

Intended pipeline:
  1. LoRA-tune Qwen2.5-Coder-1.5B on a small fine-tune signal dataset
     (synthetic preference pairs OR external preference data SCP'd in via
     Crucible local_files:). TRL DPOTrainer + Unsloth, QLoRA rank=8, alpha=16.
  2. Re-extract residuals on DELTA_HOLDOUT_TOKENS held-out tokens through
     both baseline and tuned model.
  3. Apply the same (frozen) Phase-2.1 SAE to both → (baseline_features,
     tuned_features) triples.
  4. Compute the 5 feature-diff metrics:
       - feature-level cosine drift
       - AV description edit distance / BERTScore
       - intent-category coverage shift
       - top-k feature stability (Jaccard)
       - feature-firing KL divergence
  5. W&B dashboard panel per metric + table of top features by drift.

For now: defer to the smoke runner.
"""

from __future__ import annotations

from launchers.nla_coder.smoke import main


if __name__ == "__main__":
    raise SystemExit(main())
