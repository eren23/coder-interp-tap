# coder-interp-tap

A [Crucible](https://github.com/eren23/parameter-golf_dev) tap for **coder-model interpretability**: TopK Sparse Autoencoder training, NLA-style activation verbalizers, and feature-diff metrics for measuring how fine-tuning changes a model's internal features.

The pipeline runs on RunPod via Crucible, logs to W&B, and is deliberately scoped for "usable, not perfect" — single-GPU budget, off-the-shelf base models, off-the-shelf pre-trained SAEs where available.

## Tap layout

| | Count | What |
|---|:---:|---|
| [`projects/`](projects/) | 2 | Project YAMLs: `nla_qwen3_5_2b_pilot.yaml`, `nla_qwen2_5_coder_1_5b.yaml`. |
| [`architectures/`](architectures/) | 0 | SAE / NLA model code (planned: TopK SAE wrapper, AV+AR pair). |
| [`callbacks/`](callbacks/) | 1 | `wandb_periodic_validation` skeleton — periodic eval to W&B every N steps. |
| [`data_adapters/`](data_adapters/) | 0 | Activation capture pipelines (planned: residual-stream HDF5 dumper). |
| [`evaluation/`](evaluation/) | 0 | Feature-diff metrics (planned: cosine drift, BERTScore on AV descriptions, top-k Jaccard, KL on firing). |
| [`launchers/`](launchers/) | 6 | Stub launchers per project variant. Smoke variants are runnable end-to-end and validate pod plumbing; the pilot/training variants are scaffolds for follow-up implementation. |
| [`findings/`](findings/) | 0 | Documented experiment findings — populated as runs complete. |
| [`examples/`](examples/) | 0 | Example notebooks / scripts. |

## Quick start

```bash
crucible tap add https://github.com/eren23/coder-interp-tap
crucible tap sync coder-interp-tap
crucible run_project nla_qwen3_5_2b_pilot --variant smoke
```

## What this tap is for

Two concrete pipelines, both runnable as Crucible projects:

1. **Phase 1 — pilot on Qwen3.5-2B base + Qwen-Scope pre-trained TopK SAE.** Validates the activation-dump → feature-extract → AV-train → quality-probe pipeline using an off-the-shelf SAE ([Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50](https://huggingface.co/Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50), W=32K, L0=50, all 24 layers). Cheap (~$5 + Claude API for synthetic feature descriptions).

2. **Phase 2/3 — port to Qwen2.5-Coder-1.5B + LoRA delta study.** Trains a fresh TopK SAE on a coder-tuned model (no pre-trained coder-SAE exists), trains an AV verbalizer on those features, then runs a LoRA delta study using the 5 feature-diff metrics. ~$20 GPU + Claude API.

See [`docs/sae-nla-pipeline.md`](docs/sae-nla-pipeline.md) for the full plan.

## Why a separate tap

This work uses different base models (Qwen3.5-2B base, Qwen2.5-Coder-1.5B) and different libraries (SAELens or Sparsify, anthropic SDK, sentence-transformers) than the existing `crucible-community-tap`. Keeping it separate avoids bloating that tap with research-specific dependencies.

## License

Code is MIT unless a plugin's `plugin.yaml` says otherwise.
