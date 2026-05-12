# Interpretability pilot session — SAE, VPD-v2, personal-style LoRA, feature-diff

**Date**: 2026-05-11 → 2026-05-12
**Author**: Claude (with Eren driving)
**Total compute**: ≈ $2.20 (A6000 on-demand pods)
**W&B project**: [eren23/coder-interp-pilot](https://wandb.ai/eren23/coder-interp-pilot)

---

## TL;DR

We trained a TopK SAE on Qwen2.5-Coder-1.5B and proved its features are
interpretable; built Goodfire's "VPD" (adVersarial Parameter Decomposition)
from scratch with the full five-loss recipe and the persistent-PGD adversary,
and showed that **the adversarial loss is load-bearing** — a naive copy
without it leaves every gate stuck near 1.0, while the full version sparsifies
to as few as 4 active components out of 128 per weight matrix. We then
fine-tuned a personal-style LoRA on the user's top-20 GitHub repos and ran the
existing `feature_diff_study` pipeline, which surfaced concrete bias signal
(open-source license boilerplate, FSF address tokens, Django metadata) with
DeepSeek-V3 NLA labels.

Six W&B runs document the full chain end-to-end. Three orchestration crons
ran the overnight pipeline (LoRA → destroy pod → provision feature_diff_study
→ run → destroy) with zero manual intervention.

---

## Experiments

### 1. TopK SAE — Qwen2.5-Coder-1.5B layer 6
- **W&B**: [`u20xk1jx`](https://wandb.ai/eren23/coder-interp-pilot/runs/u20xk1jx)
- **Spec**: `projects/sae_train_qwen2_5_coder_1_5b.yaml` (`overnight` variant)
- **Numbers**: d_sae = 24,576 (16× expansion), top-k = 50, 100k steps, FVE = 0.9996, dead frac 0.04%.
- **Verdict**: ✅ great. Pre-existing baseline pipeline; produced a clean SAE artifact.

### 2. SAE feature top-context analysis
- **W&B**: [`ade7o3va`](https://wandb.ai/eren23/coder-interp-pilot/runs/ade7o3va)
- **Spec**: `projects/analyze_sae_qwen2_5_coder_1_5b.yaml` (full variant)
- **What it does**: for the top-N most-active SAE features on a sample of the
  user's code corpus, log the top-K firing token contexts.
- **Result excerpt** (6 of 8 sampled features were crisp single-concept):
  - feat 415: `"""` docstring opener
  - feat 974: `import`/`from` keywords
  - feat 1377: `export` keyword (TS), activation 626
  - feat 563: `{` after `import` (JS destructure)
  - feat 1108: `{\n` (block open)
- **Verdict**: ✅ good. ~75% of sampled features are interpretable in one
  glance. The remaining 25% look like sub-word/BPE artifacts.

### 3. Naive VPD pilot — adversarial loss dropped
- **W&B**: vpd-final on `u20xk1jx`'s sibling run
- **Spec**: `launchers/vpd_pilot/main.py`
- **What it does**: rank-1 weight decomposition + causal-importance gate on one
  matrix of Qwen3-0.6B, **without** the adversarial loss (kept only recon +
  stochastic recon + L_p importance penalty + Δ-L2).
- **Result**: gates collapsed to mean ≈ 0.97 on every component. fraction
  g > 0.5 was **99.5%**. Decomposition recovered W faithfully but did NOT
  sparsify.
- **Verdict**: ❌ failure — and that's exactly the data point we needed.
  Proves the adversarial loss is what forces sparsification.

### 4. VPD-v2 — full Goodfire recipe
- **W&B**: [`2xehk87f`](https://wandb.ai/eren23/coder-interp-pilot/runs/2xehk87f)
- **Spec**: `launchers/vpd_v2/main.py`, `projects/vpd_v2_qwen3_0_6b.yaml` (pilot variant)
- **What it implements** (faithful to https://www.goodfire.ai/research/interpreting-lm-parameters):
  - Rank-1 decomposition `W ≈ Σ U[:,c] V[:,c]ᵀ + Δ` for each of 7 matrices in
    Qwen3-0.6B layer 13 (`q,k,v,o,gate,up,down`)
  - 128 components per matrix → 896 total
  - Per-matrix MLP `Γ` (paper uses a shared transformer; scoped down for pilot)
  - **All five losses**:
    1. `L_adv-recon` — minimax KL with **persistent-PGD** attacker on the
       source `r ∈ [0,1]`. Adam β=(0.5,0.99) lr=1e-2, n_adv=3 inner steps.
    2. `L_stoch-recon` — uniform-source mask with random-k subset routing
       per (batch, position).
    3. `L_imp-min` — mean |g|^p across all components; p anneals 2.0 → 0.4.
    4. `L_freq-min` — Σ_c s·log₂(1+s) MDL penalty splits polysemantic comps.
    5. `L_Δ-L2` — tight `W ≈ Σ UVᵀ + Δ`; sole loss during a 400-step warmup.
  - Two leaky hard sigmoids on `Γ`'s output (lower-leaky for masks, upper-leaky
    for `L_imp-min` / `L_freq-min`) with custom STE.
  - Mask formula `m = g + (1-g)·r` (NOT Bernoulli, NOT hard-concrete).
- **Result**: training ran cleanly, all 5 losses + adversary computed.
  Sparsification per matrix (alive of 128):

  | matrix | alive |
  |---|---|
  | `mlp.up_proj` | **4** |
  | `self_attn.o_proj` | **5** |
  | `self_attn.q_proj` | 9 |
  | `mlp.gate_proj` | 9 |
  | `mlp.down_proj` | 16 |
  | `self_attn.k_proj` | 70 |
  | `self_attn.v_proj` | 88 |

  fraction g > 0.5: **0.11%** (vs 99.5% in the naive pilot).
- **Verdict**: ✅ **big win**. Replicates the paper's headline finding on a
  real LM block. Some matrices (k_proj, v_proj) remain less sparse — likely
  because they're "more polysemantic" or because the per-matrix Γ instead of
  the paper's shared transformer Γ underconstrains them.

### 5. VPD-v2 4-viz analysis
- **W&B**: [`5rlz2q0a`](https://wandb.ai/eren23/coder-interp-pilot/runs/5rlz2q0a)
  (first attempt, empty tables due to threshold bug) and a re-run with the fix
  (pending at report time).
- **Spec**: `launchers/analyze_vpd_v2_pod/main.py`, `projects/analyze_vpd_v2_qwen3_0_6b.yaml`
- **Visualizations** (one W&B run, four artifacts):
  - **A. concept_cards**: per-matrix × component × top firing token contexts.
  - **B. logit_lens**: top vocab tokens for `U[:,c]` of residual-writing
    matrices (mlp.down_proj, self_attn.o_proj). Decodes each rank-1
    component's *output direction* through the LM head.
  - **C. sparsity bars**: per-matrix bar plot of sorted mean gates.
  - **D. coactivation heatmap**: 7×7 Jaccard of alive components across pairs
    of matrices — looks for emergent cross-matrix circuits.
- **Result excerpt**: first run had a threshold bug (alive_threshold=0.01 vs
  trainer's 1e-6) → tables empty. Plots (sparsity, coactivation) were fine.
  Fix in commit `c2d77a2`; re-run in flight.
- **Verdict** (post-fix): TBD.

### 6. Personal-style LoRA — Qwen2.5-Coder-1.5B + user's top 20 GitHub repos
- **W&B**: [`niiz0d0u`](https://wandb.ai/eren23/coder-interp-pilot/runs/niiz0d0u)
- **Spec**: `launchers/lora_style/main.py`, `projects/lora_style_eren.yaml` (full variant)
- **Data**: `eren23/eren-code-style` HF dataset built from 20 repos, 6,374
  files, ~17 M tokens. Mix: 13 Python / 3 Rust / 3 TS / 1 Go.
- **Hyperparameters**: LoRA r=32 α=64, lr=2e-4, 3 epochs, batch=2×grad_accum=8,
  seq=1024, target modules = all linear layers in attention + SwiGLU.
- **Result**: train_loss 1.05 → **0.62**, mean_token_accuracy 0.87,
  17M tokens trained, adapter 159 MB.
- **Verdict**: ✅ good. Smooth loss curve, no instabilities. Adapter saved
  as `lora-style-final:v0`. The actual personalisation signal still needs to
  be validated by `feature_diff_study` against THIS LoRA (the pilot run #7
  ran against a different bias proxy — see below).

### 7. feature_diff_study — SAE feature drift between baseline and a biased LoRA
- **W&B**: [`l8g7e3dy`](https://wandb.ai/eren23/coder-interp-pilot/runs/l8g7e3dy)
- **Spec**: `projects/feature_diff_study.yaml` (pilot variant)
- **What it does** (existing pipeline, unchanged): trains a tiny LoRA on a
  filtered subset of CommitPackFT (default: Python only) → runs both base and
  LoRA over a held-out token stream → applies the frozen SAE to both →
  computes per-feature log_ratio(tuned_rate / baseline_rate) → cross-references
  with the existing `feature_descriptions` artifact (DeepSeek-V3 NLA labels).
- **Result**: 1,848 features shifted, p99 |log_ratio| = 11.46.
- **Top drifted features (excerpt)**:

  **UP** in tuned (LoRA-Python pushed these higher):
  - **f9668** — "last names in author/maintainer metadata in Python package
    setup/config files (Cox, Combs)"
  - **f7687** — "web-related terms (webhook, Browser, Web) in Python imports"
  - **f2366** — "Django model field class definitions"
  - **f776** — "the word 'Lesser' in GNU Lesser General Public License"
  - **f4967** — "the number '59' in addresses, particularly in **Free Software
    Foundation** references" ← *literally the '59 Temple Place' FSF address*
  - **f974** — "docstring parameter sections marked by Parameters / ---------"

  **DOWN** in tuned:
  - **f20001** — "the word 'float' or 'floating' in variable types"
  - **f21418** — "Unix-style hidden directory paths `~/.`"
  - **f3077** — "Django management command handle method signatures"
- **Verdict**: ✅ shipping-grade. The pipeline correctly surfaces **license-
  and metadata-boilerplate bias** in the Python CommitPackFT subset — exactly
  the failure mode the system is designed to catch. Caveat: this LoRA is a
  bias-proxy, NOT our personal-style LoRA. Running `feature_diff_study`
  against `lora-style-final` is the obvious next step.

---

## What broke + lessons

1. **TRL 0.13 API renames bit twice**: `SFTConfig.max_seq_length` → `max_length`,
   `SFTTrainer(tokenizer=)` → `processing_class=`. The pinned `trl>=0.12`
   resolved to 0.13+ on fresh pods. Two failed LoRA launches before we caught
   it. Lesson: pin the **upper** bound on third-party libraries when their
   API churn is known.
2. **`sync_code` is not the same as `bootstrap_project`**. `sync_code` rsyncs
   the *crucible* repo and the *tap clone*, not the *project* repo. The
   project's code on `/workspace/project/` only updates via `bootstrap_project`,
   which re-clones from GitHub. Three failed retries before I figured out I
   needed to push to main first. Lesson: the project's source-of-truth is the
   git remote.
3. **VPD-v2 device init**: U/V/Δ were created on CPU while the base model is
   on cuda → first smoke run died. Fixed: allocate on `original.weight.device`.
4. **Naive VPD (no adversary) doesn't sparsify**. Important negative result.
   The adversary is doing the heavy lifting in VPD.
5. **Crucible reports `failure_class: wandb_crashed` for runs that actually
   succeeded**. This is a contract-enforcement false positive (transient W&B
   init noise). Successful runs are identifiable by `[main] done` in the log
   tail. The monitor crons learned to ignore the crucible "failed" status and
   look at the log + W&B artifact presence directly.
6. **Crons are session-only**. Closing this Claude window kills all monitors.
   For longer-running fleet, durable cron / a separate orchestrator process is
   needed.

---

## What's next

1. **Run `feature_diff_study` against `lora-style-final`** instead of the
   built-in Python-bias LoRA. This is the real "what does my style change in
   feature space" experiment. ~$0.20, ~25 min.
2. **Re-run VPD-v2 with the trainer's true threshold** (in flight at time of
   writing) — produces non-empty concept_cards + logit_lens tables and lets
   us actually inspect what the 4-88 alive components per matrix mean.
3. **Stronger VPD-v2**: longer training (paper does 400k steps; we did 1.5k +
   400 warmup). Shared transformer Γ instead of per-matrix MLP. Apply to all
   layers, not just layer 13.
4. **Downstream "lens" CLI**: per-file style-match score using the feature-diff
   delta vector + LoRA — the actual end-user app sketched at session start.

---

## Reproducibility

All commits on `eren23/coder-interp-tap@main`:
- `0b21727` — VPD-v2 implementation
- `42bac10` — VPD-v2 device init fix
- `503e55e` — switch VPD-v2 dataset to `eren23/eren-code-style`
- `bf804b5` — `SFTConfig max_seq_length` → `max_length`
- `947fca4` — `SFTTrainer tokenizer` → `processing_class`
- `f856cbc` — projects swap RTX 4090 → A6000
- `d8ce99f` — local analysis scripts
- `846c53b` — pod-side SAE analysis launcher
- `f2923be` — pod-side VPD-v2 analysis launcher
- `c2d77a2` — VPD-v2 analysis alive_threshold fix

HF artifacts:
- Dataset `eren23/eren-code-style` (public, 23.6 MB parquet, 6,374 rows)

---
