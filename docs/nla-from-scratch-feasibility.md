# NLA-from-Scratch on a Coder Model — Feasibility Note

This is a feasibility analysis, not an implementation plan. Decision deferred until we have a personalized model worth measuring.

## Why this question matters

The kitft NLA system gives natural-language descriptions of activation vectors via an Activation Verbalizer (AV) and an Activation Reconstructor (AR). Released checkpoints exist only for four (model, layer) pairs (Qwen2.5-7B-Instruct L20, Gemma-3-12B-IT L32, Gemma-3-27B-IT L41, Llama-3.3-70B-Instruct L53). To use NLA on a *coder-tuned* model — Qwen2.5-Coder-1.5B/7B, Yi-Coder, etc. — you have to train your own AV+AR pair from scratch.

The eventual goal of this research line is "a personalized coder that understands me." NLA is one of several measurement tools, not a primary deliverable. This document documents the cost and complexity of training NLA on a coder model so we can make an informed yes/no decision later.

## Recipe (per kitft README + nla_meta.yaml in released checkpoints)

The AV is itself a fine-tune of the same base model. Training is two stages, each with two sub-models trained jointly:

| Stage | What | Cost shape |
|---|---|---|
| Activation collection | Forward N tokens through the base model, dump residual stream at the chosen layer, save as parquet/HDF5 | ~$0–10 (small GPU) |
| Synthetic-description generation | Prompt a strong LLM with top-firing contexts of activation clusters, get back NL descriptions of what those clusters represent | $50–500 LLM API |
| **AR SFT (critic)** | Fine-tune the AR (a copy of the base model) on (description → reconstructed-vector) pairs. Loss: MSE on raw activations. | 2× H100-80GB |
| **AV SFT (actor)** | Fine-tune the AV (another copy of the base model) on (vector + injection-char → description) pairs. Loss: next-token NLL with the activation injected at the marked position. | 2× H100-80GB |
| **AV RL + AR supervised (joint)** | GRPO on the AV with reward = −normalized-MSE round-trip via the AR. AR continues supervised in parallel. Stopping criterion: ~75% FVE round-trip. | **2 nodes × 8 H100s = 16 GPUs** |

Wall-clock times not published. Typical RL fine-tunes at 7B + 16 GPUs are 1–3 days. 1.5B should be ~5× faster. Smaller batch sizes scale differently; budget conservatively.

## Math — Qwen2.5-Coder-1.5B AV+AR

**Activation collection.** ~100M tokens through Qwen2.5-Coder-1.5B at layer 6.
- Forward time: ~50ms/sequence × ~500 tokens/sequence = ~10ms/token = 1000s for 100K sequences. Call it 30 minutes on a 4090.
- Disk: 100M tokens × 1536 dims × 4 bytes (fp32) = ~600 GB. Cut to ~150 GB with bf16. Subsample heavily; in practice the kitft pipeline likely uses ~10M activations sampled from a larger corpus.
- **~$0.20 in compute, ~$0.50 in HF transfer.**

**Synthetic-description generation.** ~1M activations clustered into ~100K activation-cluster centroids; describe each cluster via a strong LLM.
- 100K calls × ~1000 tokens (round-trip) at $0.0002/1K (DeepSeek V3 via OpenRouter) = **~$20**.
- At Claude Sonnet rates (~$0.018/call for 1K tokens): ~$1800. Avoid.
- **Pick OpenRouter cheap models for this stage: ~$20–50.**

**AR SFT (critic)** on Qwen2.5-Coder-1.5B (5× cheaper than 7B):
- 2× H100-80GB × 12–24 hours wall = 24–48 H100-hours.
- Spot rate ~$2/H100/hr → **$48–96**.

**AV SFT (actor)** — same shape: **$48–96**.

**Joint RL phase** (GRPO + supervised AR):
- 16 H100s × ~24–36 hours wall (1.5B is ~5× faster than 7B; assume 7B takes 1–2 days, so 1.5B is ~12–24 hours).
- 384–576 H100-hours.
- Spot rate $2/H100/hr → **$770–1150**.

**Total: ~$900–1400** in raw compute + ~$20–50 in API + ~3–5 days wall (most of it RL).

## Math — Qwen2.5-Coder-7B AV+AR

Scaling 1.5B numbers ~5×:
- AR SFT + AV SFT: ~$500.
- RL phase: ~$4000.
- Synthetic-data gen: ~$50.
- **Total: ~$4500–6000** + ~5–7 days wall.

## Math — alternative measurement paths (for comparison)

| Path | Cost | Wall | What you get |
|---|---|---|---|
| **Use existing kitft Qwen2.5-7B-Instruct AV (non-coder)** | $0 training, ~$0.50/run | minutes | Verbalizes general-instruct features. Distribution mismatch if you point it at coder activations. |
| **SAE + LLM-API feature explainer** (what the `feature_explainer` project does) | ~$2–10 per run via OpenRouter | ~1h | Per-feature NL descriptions. Works on any model where you can train an SAE. Cheaper than NLA, less round-trip-faithful. |
| **NLA on Qwen2.5-Coder-1.5B** | ~$1000–1500 + 3–5 days | days | Round-trip-faithful verbalization on a coder model. Real instrument for Track-C delta studies. |
| **NLA on Qwen2.5-Coder-7B** | ~$5000 + 5–7 days | days | Same, on the production-target base for Track B. |

## Engineering complexity

The kitft training stack is real ops complexity. Expect to spend 3–5 ideal-days getting it green even with budget for compute. Not script-and-go.

- **Ray-orchestrated RL training** (FSDP2 backend for 1.5B/7B/12B; Megatron for 70B). Configuring Ray correctly across 2 nodes × 8 GPUs is 1–2 days of work.
- **sglang server** for the AR critic during RL — must be running on the same cluster, addressable from the actor process. Network setup, port mapping, health checks.
- **Activation-injection protocol** must be exactly right (we already have the meta.yaml format from a released checkpoint, so this is the easiest part).
- **Datagen pipeline** (kitft `nla.datagen`) — must be ported / understood.
- **Hyperparameter sweeps** at this scale are expensive. Probably need 2–3 attempts before getting a viable training run.
- **Failure modes**: AV collapses to generic descriptions; reward hacking via the AR; AR saturation; injection-position validation drifting on tokenizer changes.

For a solo engineer, plan ~1 ideal-week for engineering + ~1 ideal-week of paid compute for the actual training, **per model**.

## Decision criteria — when to commit to NLA-from-scratch

Don't commit until ALL of these are true:

1. **You have a personalized coder model that's measurably different from the baseline.** Track A's preference logger has produced ≥1K real preference pairs. Track B's DPO-tuned model exists and has shown organicity gains on kai-bench.
2. **The 5-metric SAE + LLM-API feature-diff measurement is insufficient.** I.e., you've run that pipeline on baseline-vs-tuned and the descriptions don't capture what changed in a way that's actionable.
3. **You've identified a specific scientific question that needs round-trip fidelity.** Without round-trip fidelity (the AR's job), AV descriptions are unverifiable. Round-trip is what makes NLA different from "ask Claude to describe these features."
4. **You have ~$1500 (1.5B) or ~$5000 (7B) compute budget approved.** Plus 1–2 ideal-weeks of engineering time.

If even one of these is false, default to the SAE + OpenRouter-LLM feature explainer. It's 100× cheaper and gives ~70% of the interpretability signal.

## Recommendation

**Defer NLA-from-scratch until at least Track A's preference logger has been running for 4+ weeks AND a Track-B DPO model exists.** Until then:

- Use the existing `nla_real_qwen2_5_7b` project for any general-instruct verbalization needs.
- Use the `feature_explainer` project (SAE + DeepSeek/Llama via OpenRouter) for coder-feature interpretation. Costs ~$2–10 per pass, gives concrete coder-feature names, scales to thousands of features.
- Train the SAE that the explainer needs at the `full` variant (20K steps, ~2h on 4090) and freeze that SAE as the canonical feature dictionary for everything downstream.
- Re-evaluate NLA-from-scratch once you have a *measurable* personalized model and a specific question that requires round-trip fidelity.

When you do commit to NLA, start with **Qwen2.5-Coder-1.5B**: lower cost, faster wall-clock, lets you exercise the training infra before scaling to 7B.

## Open questions

- Does kitft's training code support custom base models cleanly, or is it tied to specific tokenizer/architecture assumptions?
- What does the synthetic-description dataset look like for a *code* model? (Anthropic's published examples are general-text.)
- Can the activation-collection step be reused across SAE training and NLA training? (Probably yes if we're careful about activation-distribution overlap.)
- Does the AV's quality on out-of-distribution activations (e.g., Qwen2.5-Coder activations fed into the Qwen2.5-Instruct AV) degrade gracefully or catastrophically? Quick experiment available; useful preliminary data point.
