# Evaluation

Reusable eval tools for the NLA / SAE pipeline.

## Planned

| Plugin | Purpose | Status |
|---|---|---|
| `feature_descriptions` | Score AV-generated feature descriptions: BERTScore distinctiveness across features (target ≤0.5 mean), round-trip cosine via frozen text-encoder (target ≥0.4), manual quality rubric (1–5 scale, target ≥3.5 average). | Planned |
| `feature_diff` | Compare feature activations between two model checkpoints on the same token sample. Emits the 5 metrics: feature-level cosine drift, AV description edit distance / BERTScore, intent-category coverage shift, top-k feature stability (Jaccard), feature-firing KL divergence. | Planned |
| `intent_probe` | Linear probe over SAE features → intent categories (refactor, bug-fix, feature, perf, docs, test, infra, other). 200 hand-labeled features as the training set; held-out 50 for accuracy. Target ≥40% on 8-class classification (vs 12.5% chance). | Planned |
