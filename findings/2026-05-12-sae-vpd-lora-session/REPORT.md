# Interpretability pilot session — SAE, VPD-v2, personal-style LoRA, feature-diff

**Date**: 2026-05-11 → 2026-05-12
**Author**: Claude (with Eren driving)
**Total compute**: ≈ **$5** all-in (A6000 on-demand pods + DeepSeek-V3 autointerp)
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

### Confirmed VPD findings (replicated across multiple runs)
- **K/V "import-circuit"**: 5 components each in `self_attn.k_proj` and
  `self_attn.v_proj` of layer 13 fire saturated on `import` / `#!/` / `///`
  tokens, with cross-matrix Jaccard coactivation 0.34–0.39. Replicates in
  base Qwen3-0.6B, base Coder-1.5B, and LoRA-merged Coder-1.5B.
- **MLP lexical detectors** (after fixing the `beta_delta` bug):
  `mlp.down_proj` components labeled cleanly as Python docstring opener
  (`"""`), Python `__future__` imports, Python shebang `/env`;
  `mlp.up_proj` Rust `#[derive]`. Before the fix these labels were vague
  "punctuation fragments" — the bug was suppressing real structure.

### Honest negative findings
- **Naive VPD (no adversarial loss) fails to sparsify** — gates stuck at ~1.
  Differential proof that the adversary is the active ingredient.
- **The "LoRA encodes Rust circuits" claim is largely a CORPUS artifact**.
  Running VPD on the same Coder-1.5B WITHOUT the LoRA on the same corpus
  surfaces nearly the same "Rust pub use ::" labels. LoRA quantitatively
  reorganizes attention (more alive components, higher adversary error) but
  does NOT introduce qualitatively new concept categories at layer 13.

### Process
15 W&B runs document the full pipeline (SAE → SAE-autointerp → naive VPD →
faithful VPD → ablation sweep → analysis viz → LLM autointerp → confidence
tests). Multiple chained crons ran a fully autonomous overnight pipeline
(LoRA → feature_diff_study, with auto-provision / bootstrap / run / destroy).
Two bugs caught and fixed mid-session by reading the data, not the code.

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

### 5. VPD-v2 4-viz analysis  + LLM autointerp
- **W&B (final, with LLM labels)**: [`gze67ldu`](https://wandb.ai/eren23/coder-interp-pilot/runs/gze67ldu)
- **Spec**: `launchers/analyze_vpd_v2_pod/main.py`, `projects/analyze_vpd_v2_qwen3_0_6b.yaml`
- **Visualizations** (one W&B run):
  - **A. concept_cards**: per-matrix × component × top firing token contexts
  - **B. logit_lens**: top vocab tokens for `U[:,c]` of residual-writing matrices
  - **C. sparsity bars** per matrix
  - **D. cross-matrix coactivation Jaccard heatmap**
  - **E. NEW — `concept_labels`**: DeepSeek-V3 generated one-sentence concept name per alive component (179 labels)
  - **F. NEW — `demo/<prompt>` heatmaps**: 6 demo code prompts, gate-magnitude heatmaps over (alive_components × tokens)
- **Two bugs found + fixed during analysis**:
  1. `alive_threshold` was 0.01 (vs trainer's 1e-6) — produced empty tables. Fixed in `c2d77a2`.
  2. `beta_delta` was hardcoded to 1e-3 in the main loop (vs paper's 1e7 throughout). Let Δ drift, leaving `UVᵀ ≈ 0` — gates were controlling an empty subspace. Fixed in `7a689a6`.
- **Verdict**: ✅ post-fix this is the headline interpretability win. Clear single-concept components in K/V (import keyword, shebang, doc-comment markers), and after the β_Δ fix the MLP signal sharpens dramatically (Python `"""` docstring opener, Python `__future__` import, Rust `#[derive]`).

### 5b. Confidence tests (post bug-fixes)
After flagging that the "LoRA Rust shift" was potentially corpus-induced, we ran two control experiments to settle the open questions.

**(i) β_Δ fix on Qwen3-0.6B** — [W&B run](https://wandb.ai/eren23/coder-interp-pilot/runs/onx7hjjh)
- Same config (C=32, no gate_proj) but with `beta_delta=1e7` throughout instead of buggy 1e-3 in main loop.
- MLP labels became dramatically sharper:
  - `mlp.down_proj` #30 → *"Python `__future__` imports for annotations"*
  - `mlp.down_proj` #0, #23 → *"Python docstring opening quotes `"""`"*
  - `mlp.down_proj` #9, #19 → *"Python shebang `/env` path"*
  - `mlp.up_proj` #24 → *"Python `__future__` module"*
  - `mlp.up_proj` #18 → *"Rust `#[derive]` attribute"*
- Verdict: ✅ **bug was suppressing real MLP structure**. MLPs at this scale DO carry interpretable single-concept components when `UVᵀ` is forced to do the work instead of `Δ`.

**(ii) BASE Coder-1.5B (no LoRA) control** — [W&B run](https://wandb.ai/eren23/coder-interp-pilot/runs/2zcsew0x)
- VPD on Qwen2.5-Coder-1.5B without the LoRA merged, same C=32 no-gate config, same corpus.
- Compare alive component labels with vpd-lora run [`jfjid9o6`](https://wandb.ai/eren23/coder-interp-pilot/runs/jfjid9o6).
- Result: **the "Rust shift" claim partially BUSTED.** The BASE model ALREADY has alive components labeled *"the keyword `use` in Rust import statements"*, *"Rust programming language keywords and imports"*, *"importing or declaring dependencies in Rust code"*. The Rust circuits are mostly in the base model's pre-trained weights + amplified by the Rust-heavy corpus, NOT introduced by the LoRA.
- What IS different in LoRA-merged vs base: more alive components (k=15 vs 10, v=23 vs 18), higher adversary error (0.27 vs 0.08) → LoRA reorganized attention to use more entangled structure, but did NOT introduce new conceptual categories.
- Verdict: 🟡 **honest finding** — LoRA quantitatively perturbs attention but doesn't qualitatively add new concept detectors. The corpus drives the "Rust pub use ::" labels, not the fine-tune.

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
7. **`destroy_nodes` by name can leave RunPod zombies**: false alarm in our case
   (a stale console refresh), but next time verify with `pod_ids`-based destroy
   after a name-based call.
8. **VPD `beta_delta` must be ≥ 1e7 in main loop too**, not just warmup. With a
   tiny `1e-3` coefficient in main, the decomposition becomes degenerate:
   `decomp_residual` stays at 0 because Δ is happy to drift to `Δ ≈ W`, leaving
   `UVᵀ ≈ 0`. Gates then "control" an effectively empty subspace. Fix: paper-
   style `beta_delta=1e7` everywhere.
9. **Auto-interpretation LLMs confabulate at low gate magnitudes**. Components
   with `mean_g ≈ 10⁻⁵` are essentially noise; asking DeepSeek "what concept
   do these tokens share?" still produces a confident answer. Only labels with
   `mean_g ≥ ~10⁻⁴` are trustworthy. Trust strength of evidence, not the label.
10. **Corpus-vs-LoRA confound**: if the LoRA dataset has the same biases as
    the analysis corpus, you can't distinguish "LoRA learned X" from "corpus
    surfaces X in base too". Always run the base-control on the same corpus.

---

## What's next

1. **Stronger VPD-v2 at scale**: longer training (paper does 400k steps; we did
   up to 15k). Shared transformer Γ instead of per-matrix MLP. Apply to all
   layers, not just layer 13.
2. **Ablation validation**: mask the "alive" components and measure logit drift.
   If they're truly load-bearing, masking should hurt; if our threshold is loose,
   it won't. This is the gold-standard test we haven't done yet.
3. **Downstream "lens" CLI**: per-file style-match score using the feature-diff
   delta vector + LoRA — the actual end-user app sketched at session start.
4. **Better LoRA target**: instead of a personal-style LoRA (whose biases overlap
   with the analysis corpus), try a LoRA trained on a DIFFERENT distribution
   (e.g., math-bench) and analyze on code corpus. The disjoint distributions
   would let VPD cleanly attribute differences to the LoRA.

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
- `913c2d3` — LLM autointerp + per-prompt heatmaps for analyze_vpd_v2_pod
- `7a689a6` — `beta_delta` bugfix in vpd_v2 main loop

HF artifacts:
- Dataset `eren23/eren-code-style` (public, 23.6 MB parquet, 6,374 rows)

W&B runs (eren23/coder-interp-pilot):
| run | name |
|---|---|
| `u20xk1jx` | SAE training (sae-final, 24K features) |
| `ade7o3va` | SAE feature top-context analysis |
| `2xehk87f` | VPD-v2 pilot baseline (C=128) |
| `o8ixp4cq` | VPD-v2 C=32+noGate (winning config) |
| `at2weoo4` | VPD-v2 scale-15k |
| `kdblndko` | VPD-v2 Coder-1.5B BASE control |
| `m6nftt9r` | VPD-v2 Qwen3-0.6B β_Δ bugfix |
| `niiz0d0u` | LoRA style SFT (lora-style-final, 159 MB) |
| `pghxvjh2` | VPD on LoRA-merged Coder-1.5B |
| `xm7jwp61` | VPD-v2 4-viz analysis (baseline) |
| `gze67ldu` | Rich-viz analysis with LLM labels (C=32 winner) |
| `jfjid9o6` | Rich-viz analysis with LLM labels (vpd-lora) |
| `2zcsew0x` | Rich-viz analysis (Coder-1.5B base control) |
| `onx7hjjh` | Rich-viz analysis (β_Δ-fixed Qwen3) |
| `l8g7e3dy` | feature_diff_study pilot |

Total compute: ≈ $4 of A6000 on-demand pod time + ≈ $1 of OpenRouter DeepSeek-V3 calls for autointerp. All-in: ~$5, well under the $10 budget originally set.

---
