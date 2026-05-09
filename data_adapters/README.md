# Data adapters

Activation capture and dataset shaping plugins.

## Planned

| Plugin | Purpose | Status |
|---|---|---|
| `residual_dump` | Forward arbitrary token sequences through a base model and dump residual-stream activations at a chosen layer to HDF5. Inputs: HuggingFace model id, token corpus, layer idx. Outputs: HDF5 with (token_id, residual ∈ ℝ^d). | Planned |
| `commitpackft_streamer` | Stream code tokens from the CommitPackFT dataset, deduplicated against any test-set we use for eval. Yields tokens compatible with `residual_dump`. | Planned |
| `personal_git_streamer` | Stream tokens from the user's personal git history (loaded via `git log -p` parsed to file/diff/message tuples). Sanitizes secrets. | Planned |

## Schema convention

All HDF5 dumps follow the schema declared in [`../DATA_REGISTRY.yaml`](../DATA_REGISTRY.yaml). Adding a new dataset = adding both a `data_adapters/` plugin and a `DATA_REGISTRY.yaml` entry.
