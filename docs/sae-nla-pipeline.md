# SAE/NLA Pipeline — Experiment Plan

## Goal

Stand up an NLA-like measurement pipeline (verbalize feature activations, compare across model checkpoints) that's **usable, not perfect**. Three phases, gated.

## Why this path

Comparing a fine-tuned coder model against its baseline via NLA-verbalized features at 7B scale is expensive (training NLAs on 7B is ~100–200 GPU-h, and there's no pre-trained SAE for coder models yet). Two simplifications make the work tractable:

1. **Use a pre-trained SAE on a smaller base model first** — Qwen-Scope's residual SAEs on Qwen3.5-2B base ([HuggingFace](https://huggingface.co/Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50)) are off-the-shelf, all 24 layers, TopK with L0=50, width 32K. Validates the verbalize-and-diff pipeline without training an SAE.
2. **Train a smaller SAE on a coder-tuned model** — Qwen2.5-Coder-1.5B (smallest Qwen2.5-Coder variant, hidden=1536, 12 layers). Train one TopK SAE on layer 6, then port the verbalizer.

The aspiration is a working delta study (baseline vs LoRA-tuned) on Qwen2.5-Coder-1.5B. If a phase fails, the next phase is automatically deferred.

## Phase 0 — infrastructure (1–2 days)

- **Crucible tap scaffold** — this repo. Project YAMLs in `projects/`, callback skeleton in `callbacks/`, launcher stubs in `launchers/`. Already in place after this commit.
- **W&B project** — `coder-interp-pilot` under entity `eren23`. One project; runs distinguished by `WANDB_RUN_NAME`.
- **RunPod template via Crucible** — 1× RTX 4090 spot, 40GB container disk, 40GB volume disk, CUDA 12.4, torch ≥2.5, Python 3.10. Mirrors the shape used by `diff_xyz_runpod_smoke.yaml` in the community tap.
- **Smoke test** — `crucible run_project nla_qwen3_5_2b_pilot --variant smoke` runs for 5 minutes, downloads Qwen3.5-2B (≈4GB) and one SAE layer (≈260MB), captures 1K residuals, runs 50 AV training steps, logs to W&B. Validates plumbing only.

**Exit criterion**: smoke run finishes green, W&B run page shows ≥1 logged metric and ≥1 logged sample.

## Phase 1 — NLA pilot on Qwen3.5-2B + Qwen-Scope SAE (3–5 days)

**Why first**: pre-trained SAE means zero SAE-training cost. Validates AV training and feature-diff machinery end-to-end on a base (non-coder) model. If AV training fails here, the framing is wrong; no point porting to a coder model.

### Steps

1. **Activation dump** (`data_adapters/residual_dump/`, planned).
   - Sample 200K tokens stratified across The Stack v2 (code) and open-web math.
   - Forward through Qwen3.5-2B; hook layer 12 residual.
   - Dump (token_id, residual ∈ ℝ^2048) to `data/qwen3_5_2b_residuals_layer12.h5`.
   - Time: ~2h on RTX 4090.

2. **Feature extraction**.
   - Load `Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50` checkpoint for layer 12.
   - For each residual: `pre_acts = residual @ W_enc.T + b_enc; topk_vals, topk_idx = pre_acts.topk(50)`.
   - Cache to `data/qwen3_5_2b_features_layer12.h5`.
   - Time: ~30min.

3. **AV (Activation Verbalizer) training** (`architectures/nla_av/`, planned).
   - Architecture: small encoder (residual + selected feature index → text tokens). Treat as a conditioned language model: input = residual concatenated with one-hot of selected feature; output = NL description token sequence.
   - **Synthetic descriptions**: for each of the 32K features, pick top-firing tokens (e.g., top 20 contexts with highest feature activation). Prompt Claude Sonnet to describe the feature in ≤25 words. Cost: ~32K × $0.001 ≈ $32.
   - Loss: NLL on description tokens.
   - **Frozen AR** in Phase 1 — skip the Activation Reconstructor. Round-trip cosine measured later as eval, not as training signal. Reduces complexity.
   - Time: ~6h on RTX 4090 with batch 64, ~20K steps.

4. **Evaluation** (`evaluation/feature_descriptions/`, planned).
   - **Quality probe**: 50 random feature descriptions, manually rated 1–5 ("does this match the top-firing tokens?"). Target ≥3.5 average.
   - **Distinctiveness**: BERTScore between descriptions of different features should be low (target ≤0.5 mean). High BERTScore = AV is collapsing to generic descriptions.
   - **Round-trip cosine** (sanity): use a frozen text-encoder (e.g., bge-small-en) as a stand-in AR; embed the AV description and compute cosine vs an embedding of the top-firing tokens. Target ≥0.4.

5. **W&B logging**.
   - Train: NLL loss every 100 steps.
   - Eval: BERTScore distinctiveness, cosine round-trip every 1000 steps.
   - Artifacts: AV checkpoint every 5000 steps; sample 20 feature descriptions every 1000 steps as a W&B Table.

### Phase 1 exit criteria

- AV NLL converges (not flat, not diverging).
- Manual quality rating ≥3.5/5 on 50 features.
- BERTScore distinctiveness ≤0.5.

If AV training fails: simplify to **feature-description retrieval** (no autoencoder framing — just generate descriptions once via Claude, store as a lookup table, retrieve during diff studies). Salvages the measurement story.

## Phase 2 — port to Qwen2.5-Coder-1.5B (1–2 weeks)

**Why coder model**: the eventual subject is a fine-tuned coder. Phase 1 validated the pipeline on a base model; Phase 2 validates it on a coder-tuned model.

### Steps

1. **SAE training** (`architectures/topk_sae_trainer/`, planned).
   - No pre-trained SAE for coder models. Train a TopK SAE matching Qwen-Scope's recipe: d_model=1536, d_sae=24576 (16×), L0=50.
   - Layer: middle (6 of 12).
   - Data: 100M code tokens — CommitPackFT (Python + TS subsets).
   - Loss: MSE on TopK reconstruction + auxiliary loss on dead features (per Anthropic's recipe in [transformer-circuits.pub/2024/scaling-monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/)).
   - Framework choice: try **SAELens** (https://github.com/jbloomAus/SAELens) first; if instability, fall back to EleutherAI's **Sparsify** (https://github.com/EleutherAI/sparsify).
   - Time: ~12h on 1× A100 spot ($1/hr × 12 = $12), or ~18h on 4090.

2. **Activation dump** for AV training (analogous to Phase 1).

3. **AV training** — same shape as Phase 1, with coder-specific synthetic descriptions (Claude prompted to describe code-relevant features: "what code pattern does this feature fire on?").

4. **Code-specific probe** (`evaluation/intent_probe/`, planned).
   - Label 200 features by hand: which intent category does the feature relate to (refactor, feature, bug-fix, perf, docs, test, infra, other)?
   - Train a linear probe: feature_idx → intent_category. Target ≥40% accuracy across 8 classes (vs 12.5% chance).
   - Probes the assumption that coder-SAE features carry intent-relevant information.

### Phase 2 exit criteria

- SAE reconstruction loss plateaus at <10% of input variance.
- Dead-feature fraction <20% (per SAELens diagnostics).
- AV quality rating ≥3.5/5.
- Linear probe ≥40% on intent-category classification.

If SAE training collapses: drop to a **smaller width** (d_sae=8192, expansion=5×) and accept lower interpretability for stability. If even that fails, freeze the project at Phase 1 scope (we still have a working pipeline on the base model).

## Phase 3 — LoRA delta study (1–2 weeks)

**Pre-requisite**: a small set of fine-tune signal pairs. Synthetic pairs from a code corpus (`(model_proposal, ground_truth_commit)`) are a credible starter dataset — we don't need *real* preference data for the diff to be measurable; we just need *some* fine-tune signal that produces a measurable internal change.

### Steps

1. **LoRA-tune Qwen2.5-Coder-1.5B** on 100–500 synthetic preference pairs via TRL `DPOTrainer` + Unsloth, QLoRA rank=8, alpha=16.
   - Time: ~2h on 4090.
   - Dataset is small; expect mild gain or wash on test-pass rate. Goal is *measurable change*, not improvement.

2. **Re-extract features**.
   - Run the same activation dump on 5K held-out tokens through both baseline and tuned model.
   - Apply the same SAE (frozen — same encoder for both, so the diff is purely activation-level).
   - Cache (token_id, baseline_features, tuned_features) triples.

3. **The 5 metrics** (`evaluation/feature_diff/`, planned).
   - Feature-level cosine drift (per-feature, then aggregate).
   - AV description edit distance / BERTScore (compare AV outputs on baseline-tokens vs tuned-tokens for the same feature index).
   - Intent-category coverage shift (per Phase 2 probe — does the tuned model fire on different intent categories?).
   - Top-k feature stability (Jaccard over highly-active feature sets).
   - Feature-firing KL divergence on activation magnitude distributions.

4. **W&B dashboard** with one panel per metric, plus a sortable table of "top features by drift".

### Phase 3 exit criteria

- Each of the 5 metrics has a numeric value.
- ≥3 of 5 metrics show statistically meaningful difference between baseline and tuned (use paired bootstrap CI over 5K tokens).
- Qualitative inspection: top-10 most-drifted features have AV descriptions that intuitively match what the fine-tune signal might change.

## Compute / cost estimate

| Phase | Hardware | Hours | Cost (spot) |
|---|---|---|---|
| 0 (smoke) | 4090 spot | 0.5 | ~$0.30 |
| 1 (Qwen3.5-2B AV) | 4090 spot | 8–12 | ~$5–8 |
| 1.5 (synthetic descriptions, Claude) | n/a | n/a | ~$32 |
| 2.1 (Qwen2.5-Coder SAE) | A100 spot | 12 | ~$12 |
| 2.2 (Qwen2.5-Coder AV) | 4090 spot | 8 | ~$5 |
| 2.3 (synthetic + intent probe labels) | n/a | n/a | ~$15 |
| 3 (LoRA + 5-metric diff) | 4090 spot | 4 | ~$2 |
| **Total** | | ~36 | **~$70 + Claude API** |

Fits a "weekend project" budget. Numbers are rough; expect 2× over-run.

## What "usable" means

- Pipeline runs end-to-end on 1 GPU within 24h wall time per phase.
- W&B logs are auditable: loss curves, eval metrics, sample feature descriptions every N steps.
- Validation checkpoint every ~10% of training; resumable.
- Each phase gates the next.
- Deferring or downgrading a phase (e.g., dropping AV in favor of feature-description retrieval) does not block downstream phases — measurement story stays intact.

## Failure modes & mitigations (cheat sheet)

| Failure | Detection | Mitigation |
|---|---|---|
| AV NLL diverges | Phase 1, step ≥2K | Smaller batch; increase synthetic-description data |
| AV produces generic descriptions | BERTScore distinctiveness >0.6 | Sharper prompt; harder negative sampling |
| AV quality rating <3 | Manual eval | Skip AV; use feature-description retrieval lookup |
| SAE collapses (Phase 2) | Reconstruction loss flat at >50% variance | Reduce expansion to 5×; switch SAELens → Sparsify |
| SAE has >50% dead features | SAELens diagnostics | Lower learning rate; longer warmup; auxiliary loss weight up |
| Intent probe <30% | Phase 2 evaluation | Add commit-cluster context as input; or accept it (probe is bonus, not gate) |
| LoRA delta is washed out | Phase 3, all 5 metrics within noise | Train LoRA longer; or use a more aggressive fine-tune signal (full SFT on a 100-example synthetic corpus) |
| RunPod 4090 OOM during AV training | OOM trace | Drop batch to 16; or use 4bit base model + LoRA |

## Concrete first action (this week)

1. **Push this repo** to GitHub: `git init && git remote add origin git@github.com:eren23/coder-interp-tap.git`. Push to main.
2. **Add the tap to Crucible**: `crucible tap add https://github.com/eren23/coder-interp-tap && crucible tap sync coder-interp-tap`.
3. **Implement Phase 0 smoke**: fill in `architectures/nla_av/` (skeleton only — synthetic forward pass returning loss=1.0) so the smoke project YAML can run end-to-end. Confirm RunPod provisioning + W&B logging.
4. **Defer real Phase 1 implementation** until Phase 0 smoke is green.

## Open questions

1. **Layer choice for Qwen3.5-2B SAE** — pilot uses layer 12 (mid). Layer 18 (later) might encode more semantic features but is also more LoRA-affected when we get to Phase 3. Worth doing a layer-by-layer sweep in Phase 1 once AV training is stable.
2. **Synthetic description budget** — $32 is a guess; if descriptions per feature need >1 try (refinement loop), it may climb to $100+. Cap with a budget gate.
3. **Frozen AR vs full NLA** — Phase 1 intentionally skips AR training. If NLA fidelity matters for the Phase 3 delta study, AR training has to happen in Phase 2 or later. Defer the decision until Phase 1 results are in.
4. **Which fine-tune signal feeds Phase 3** — synthetic git-history pairs are a credible default. Anything that produces a measurable parameter delta works for the diff study.
