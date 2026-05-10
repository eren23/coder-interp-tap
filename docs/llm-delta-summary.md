# LLM-Explains-Delta — Macro-Bias Summarization

A postprocess layer for `feature_diff_study` that compresses a per-feature drift table into a 2–3 paragraph natural-language summary of *what the LoRA actually did*.

## TL;DR

The current `feature_diff_study` produces a W&B table of ~500–2500 changed features, each with a name like *"pytorch.nn import"*. Reading it is a chore. The LLM-explains-delta layer takes the top-K rows, feeds them to an LLM, and gets back English like:

> *"This LoRA shifts the model toward PyTorch-style ML code: deep-learning imports, tensor manipulation patterns, concise inline comments. Away from verbose Java-style docstrings and enterprise OOP boilerplate. Surprise: it also amplifies type-hint usage on parameters but not on returns."*

Same DeepSeek V3 / OpenRouter endpoint we already use for `feature_explainer`. Adds ~$0.005 per delta-study run.

## The three abstraction layers

```
                            ┌────────────────────────────────────┐
   raw residual stream      │   1536-d float vectors             │
   (Qwen2.5-Coder-1.5B L6)  │   inscrutable                      │
                            └─────────────┬──────────────────────┘
                                          │  SAE encode (k=50 of 24,576)
                                          ▼
                            ┌────────────────────────────────────┐
   SAE atomic features      │   F#1234 fires at activation 0.7   │
   (this layer is what we   │   F#5678 fires at activation 1.2   │
    have today)             │   ... + named via DeepSeek V3      │
                            └─────────────┬──────────────────────┘
                                          │  feature_diff_study:
                                          │  log_ratio(tuned/baseline)
                                          ▼
                            ┌────────────────────────────────────┐
   per-feature drift        │   feature_idx | log_ratio | name   │
                            │   F#1234     | +2.30     | "pytorch.nn import"
                            │   F#5678     | +1.80     | "tensor reshape"
                            │   ... (500–2500 rows)              │
                            └─────────────┬──────────────────────┘
                                          │  THIS DOC: top-K → LLM
                                          ▼
                            ┌────────────────────────────────────┐
   macro-bias summary       │   "This LoRA biases toward PyTorch │
   (NEW)                    │    style ML code... away from Java │
                            │    docstrings... surprise: ..."    │
                            └────────────────────────────────────┘
```

The new layer is the bottom box. Two compressions stack: SAE turns vectors into atoms, LLM turns atoms into a sentence.

## Pipeline

```
       feature_diff_study run completes
                  │
                  │  W&B Table: (feature_idx, log_ratio, name, ...)
                  ▼
       ┌────────────────────────┐
       │  Sort by |log_ratio|   │
       │  Take top 30 UP        │
       │  Take top 30 DOWN      │
       └────────────┬───────────┘
                    │
                    │  60-row context
                    ▼
       ┌────────────────────────┐
       │  LLM prompt template   │
       │  (see "Prompt" below)  │
       └────────────┬───────────┘
                    │
                    │  one HTTP call to OpenRouter
                    ▼
       ┌────────────────────────┐
       │  DeepSeek V3 chat      │
       └────────────┬───────────┘
                    │
                    │  2–3 paragraph summary
                    ▼
       ┌────────────────────────────────┐
       │  Append to W&B run description │
       │  Log as W&B Table column too   │
       └────────────────────────────────┘
```

## Prompt template

```
You are analyzing what a LoRA fine-tune did to a code language model.
Below is a list of "features" — each feature is a recurring concept
the base model already encoded. We measured firing-rate change after
fine-tuning. UP means the tuned model fires this concept more than
the baseline; DOWN means less.

UP after tune (top 30 by |log_ratio|):
  F#1234  +2.30  "pytorch.nn import"
  F#5678  +1.80  "tensor reshape patterns"
  F#9012  +1.62  "tensor.cuda() call"
  ...

DOWN after tune (top 30):
  F#3456  -2.10  "verbose Java-style docstrings"
  F#7890  -1.95  "enterprise getter/setter"
  ...

In 2–3 short paragraphs:
1. What style/domain/skill is this LoRA biasing TOWARD?
2. What is it biasing AWAY from?
3. Anything surprising or unexpected — features that don't fit the
   obvious story?

Be specific. Reference feature names. No hedging.
```

## Concrete example

**Input** (excerpt from a hypothetical `BIAS_FILTER=pytorch_only` run):

```
UP after tune (top 30):
F#0421 +2.61 torch.nn.functional import
F#7732 +2.43 tensor reshape (.view, .reshape)
F#1108 +2.18 .cuda() / .to(device) call
F#2299 +1.95 torch.optim.Adam
F#4488 +1.71 nn.Module subclass definition
F#0033 +1.65 numpy.array literal
F#9001 +1.58 dataclass decorator
...

DOWN after tune (top 30):
F#5512 -2.40 javadoc-style /** comment block
F#3344 -2.15 "private static final" pattern
F#8821 -1.98 try/catch with NullPointerException
F#1772 -1.77 verbose interface definition
F#4400 -1.50 enterprise factory pattern
...
```

**LLM output:**

> The LoRA biases the model strongly toward modern PyTorch ML code: it amplifies torch.nn imports, tensor manipulation idioms (reshape, .view, .cuda), optimizer usage (Adam), and the `nn.Module` subclass pattern. Numpy literal usage and dataclass decorators also rise — typical co-occurrences in research-style ML scripts.
>
> It pushes away from Java enterprise boilerplate: javadoc comment blocks, "private static final" constants, NullPointerException handling, verbose interface definitions, and factory-pattern idioms. This is a clean signal — the SFT data looked like Python ML code and the tuned model has internalized it.
>
> One surprise: dataclass decorators (F#9001, +1.58) come up despite not being a typical PyTorch pattern. Possible explanation: the SFT corpus included config-heavy training scripts where dataclasses are used for hyperparameter containers. Worth checking whether this generalizes or is corpus-specific.

That's ~80 words of useful signal vs the raw 60-row table.

## Where it slots into the codebase

```
launchers/feature_diff_study/
├── main.py                    ← existing: trains LoRA, computes diff, logs W&B
└── summarize_delta.py         ← NEW: postprocess
```

`summarize_delta.py` is invoked at end of `main.py`:

```python
# in launchers/feature_diff_study/main.py, after the W&B Table is logged:
from summarize_delta import summarize_via_llm
summary = summarize_via_llm(
    diff_rows=top_k_rows_up + top_k_rows_down,
    llm_endpoint=os.environ.get("LLM_ENDPOINT_URL"),
    api_key=os.environ.get("LLM_API_KEY"),
)
wandb.run.notes = summary           # appears on W&B Run page
wandb.log({"delta_summary": summary})
```

The `LLM_ENDPOINT_URL` / `LLM_API_KEY` env vars are already plumbed through the launcher (used by `feature_explainer/main.py`).

## Why not run the LLM-summary inline at training time?

We could. Two reasons to keep it as a postprocess:

1. **Cheap retry.** If the prompt template needs tweaking, we re-run summarize on the existing W&B Table without re-running the GPU-hours of LoRA + activation capture.
2. **Same script can compare multiple delta studies.** A second mode: take 5 different bias runs (random Python, PyTorch, Karpathy, ggerganov, eren23) and feed all 5 top-K tables to the LLM with prompt *"summarize each, then compare and contrast."* That's where the personal-style signal becomes legible.

## Two W&B tables — extreme vs dense-shifted

`feature_diff_study/main.py` logs two parallel views of the same diff:

| Table key | Ranking | Purpose |
|---|---|---|
| `diff/top_features` | top-K by `\|log2_ratio\|` | Extreme view — features that flipped on/off. Dominated by `rb=0` or `rt=0` (one-sided sparse). Useful for "what got introduced/eliminated." |
| `diff/top_features_dense` | top-K by `min(rate_baseline, rate_tuned) × \|log2_ratio\|` | **What the LLM summarizer consumes.** Surfaces features that fired meaningfully in BOTH models AND shifted. The actual personalization signal. |

The LLM-explains-delta postprocess uses `diff/top_features_dense`. The
extreme table is kept for human inspection — sometimes "this feature
went from 0 firings to 16 firings" is the interesting finding, but
that's a different question than "what changed in the model's
day-to-day behavior."

## Rare-event filter — what the LLM actually sees

The raw top-K-by-|log2_ratio| view is dominated by **divide-by-near-zero
outliers**: features that barely fired in baseline (rb≈0.0000) and a
handful of times in tuned (rt≈0.0008). These produce log2 ratios of
+18 to +20 — *technically* a million-fold change, but only a handful of
firings out of ~20K tokens. Letting them dominate the prompt mistakes
"feature was introduced" for "feature was amplified by a million."

The fix is a pre-LLM filter: keep only rows where **both** baseline and
tuned fired at rate ≥ `DELTA_SUMMARY_MIN_RATE` (default `0.001`,
i.e. ≥20 firings each in a 20K-token holdout). This surfaces the dense,
genuinely-changed features instead of the rare-event tail.

If filtering leaves fewer than `DELTA_SUMMARY_MIN_ROWS` (default 5)
in either direction, the script falls back to the unfiltered ranking
with a note in the prompt telling the LLM to read huge log2 values as
"introduced/eliminated" rather than "amplified."

Env var knobs:
- `DELTA_SUMMARY_MIN_RATE` — minimum firing rate per side (default 0.001)
- `DELTA_SUMMARY_MIN_ROWS` — minimum rows per direction before fallback (default 5)
- `DELTA_SUMMARY_TOP_N` — how many UP and DOWN rows to keep after filter (default 30)

## Cost

- Per single delta-study summary: ~60 rows × ~30 tokens each = ~1800 input tokens + ~300 output. DeepSeek V3 via OpenRouter at ~$0.0003/1k in / $0.001/1k out → **~$0.005**.
- Per cross-bias comparison (5 runs at once): ~9k in + ~1k out → **~$0.005**.
- Negligible relative to the $1–3/run GPU cost of feature_diff_study.

## Worked example — what we train on, what we detect

Before the LLM-explains-delta layer can do anything, the upstream
`feature_diff_study` has to actually *do* something to the model. Here's
the concrete shape, with all model and dataset names spelled out so this
section reads as a recipe.

### The base model and the SAE we read it through

```
base model         : Qwen/Qwen2.5-Coder-1.5B           (HF, Apache 2.0)
read at            : layer 6 residual stream            (1536-d float vectors)
SAE                : custom TopK, k=50, W=24,576 features
                     trained on bigcode/commitpackft (python slice)
                     log_artifact: sae-final  (W&B coder-interp-pilot)
feature names      : 24,532 of 24,576 features auto-named via
                     deepseek/deepseek-chat (DeepSeek V3) on OpenRouter
                     log_artifact: feature-descriptions
LoRA adapter       : rank=8 alpha=16 lr=1e-4  (peft + transformers)
delta-summary LLM  : deepseek/deepseek-chat (same)
```

### The four bias modes we actually run

The launcher has one knob — `BIAS_*` env vars — and produces one of four
training distributions. Same code path each time, just different inputs.

#### 1. Random Python (the noise-floor / null baseline)
```yaml
overrides:
  BIAS_FILTER_KEY:    "lang"
  BIAS_FILTER_VALUE:  "python"     # bigcode/commitpackft 'python' subset
  # no regex, no github user
  N_LORA_EXAMPLES:    "500"
  LORA_TRAIN_STEPS:   "500"
```
Trains on a uniform sample of CommitPackFT Python commits. Used as the
"what does *any* fine-tune look like" baseline. Headline: median
|log2_ratio|=0.033, ~2,400 features measurably changed.

#### 2. PyTorch-bias regex (a tight stylistic slice)
```yaml
overrides:
  BIAS_FILTER_VALUE:  "python"
  BIAS_CONTENT_REGEX: "import torch|torch\\.|nn\\.Module|\\.cuda\\(\\)"
```
Same population, but only commits whose content matches the PyTorch regex.
Detected: features for `pytorch.nn` imports, tensor reshape patterns,
optimizer usage all rise; Java-style boilerplate features fall.

#### 3. Cross-language holdout (a sanity-check on what features *are*)
```yaml
overrides:
  BIAS_FILTER_VALUE:  "python"      # train bias on Python
  HOLDOUT_LANG:       "rust"        # measure activations on Rust
```
Trains on Python, measures the diff on a Rust held-out stream. Confirms
the SAE's "Python-import" features are language-specific (they
don't fire on Rust). Headline: median |log2_ratio| ≈ 0.000 — the diff
machinery isn't producing noise where it shouldn't.

#### 4. Personal-codebase mode (the actual personalization signal)
```yaml
overrides:
  BIAS_GITHUB_USER:        "karpathy"      # or ggerganov, eren23, ...
  BIAS_GITHUB_EXTENSIONS:  ".py,.ts,.tsx,.js,.rs,.go,.md"
```
Skips CommitPackFT entirely. Clones the user's public repos
(`build_bias_dataset_from_github` in `main.py`), samples files matching
the extension list, trains LoRA on those. This is the mode that produces
the per-author macro-bias readout — *"karpathy biases toward small clean
training scripts and tensor manipulation"* vs *"ggerganov biases toward
low-level C/CUDA kernels"*.

The `BIAS_GITHUB_USER` mode is what the LLM-explains-delta layer is
ultimately for: each author run produces one paragraph of English saying
how that author's code shifted the model. Five author runs side-by-side
gives the personalization-claim test in plain language.

### What we detect, end to end

```
       train bias                            measure on holdout
   ┌──────────────────┐                     ┌─────────────────────┐
   │ commitpackft     │                     │ commitpackft        │
   │  python  OR      │                     │  python (default)   │
   │ regex slice      │  →  LoRA-SFT  →     │  OR rust (cross-    │
   │  OR              │     (peft)          │      lang test)     │
   │ github user repos│                     └──────────┬──────────┘
   └──────────────────┘                                │
                                                       ▼
   ┌─────────────────┐         ┌────────────────┐   forward N tokens
   │ baseline Qwen   │         │ tuned Qwen     │   through both
   │ Coder-1.5B      │         │ (LoRA merged)  │
   └────────┬────────┘         └────────┬───────┘
            │                           │
            └─────────  L6 residuals  ──┘
                          │
                          ▼
                  ┌─────────────────┐
                  │ frozen SAE      │  k=50 of 24,576
                  └────────┬────────┘
                           │
                  per-feature firing rate (baseline, tuned)
                           │
                           ▼
                  log2_ratio = log2(rate_tuned / rate_baseline)
                           │
                           ▼
                  top-K by |log2_ratio| + named description
                           │
                           ▼
                  W&B Table: rate_b, rate_t, log2_ratio, name
                           │
                           ▼
                  summarize_delta.py  →  macro-bias paragraph
                           │
                           ▼
                  wandb.run.notes
```

What we **detect** at each layer:
- **Per-feature**: which named concepts (out of ~24K) the LoRA
  amplified or suppressed.
- **Aggregate**: how many features changed substantially
  (`|log2_ratio| > 0.5`), the median and p99 of |log2_ratio| —
  i.e. how *coherent* the bias is.
- **Macro**: a 2-3 paragraph English readout of the style/domain shift.
- **Cross-bias contrast**: same metrics produced for each author or
  data slice, comparable side-by-side.

What we **follow** across runs:
- The four standard W&B metric panels (`diff/n_features_with_changes`,
  `diff/median_abs_log_ratio`, `diff/p99_abs_log_ratio`,
  `diff/baseline_tokens`).
- The top-K table per run.
- The new `wandb.summary["delta_summary"]` paragraph.

## What this enables

Once `summarize_delta.py` ships:

1. Every `feature_diff_study` W&B run has a human readable headline of what the LoRA did. No more squinting at log-ratio tables.
2. **The personal-coding-model claim becomes testable in plain English.** "Does Karpathy bias differ from ggerganov bias differ from eren23 bias?" is answered by reading three short paragraphs side by side.
3. Sets up the eventual NLA postprocess (see the [TODO note on training Coder-1.5B AV+AR](https://github.com/eren23/coder-interp-tap)). The LLM-summary layer and the NLA verbalization layer are complementary: SAE feature names answer "*which named concept changed*", NLA captions answer "*what does the activation feel like at this decision point*". Both can feed the same macro-summary prompt.

## Status

Not yet implemented. Ships next. ~150 LOC, ~30 min of dev time, then a sanity-check call against the random-Python delta study run we already have on W&B.
