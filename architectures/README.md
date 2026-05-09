# Architectures

Crucible model architecture plugins for the NLA / SAE pipeline.

## Planned

| Plugin | Purpose | Status |
|---|---|---|
| `topk_sae` | TopK Sparse Autoencoder. Encoder + decoder with TopK activation; auxiliary dead-feature loss. Matches Qwen-Scope's recipe (W=32K, L0=50). Used for both loading pre-trained SAEs and training new ones. | Planned |
| `nla_av` | NLA Activation Verbalizer. Conditional LM mapping (residual, feature_idx) → NL description tokens. Frozen base text decoder + small adapter. | Planned |
| `nla_ar` | NLA Activation Reconstructor. Frozen text-encoder (e.g. bge-small-en) → linear projection back to feature space. Used as a frozen scoring head in Phase 1; trainable in Phase 2+ if NLA fidelity matters. | Planned |

## Adding a new architecture

```
architectures/<name>/
├── plugin.yaml          # manifest
├── <name>.py            # Python module (entrypoint per plugin.yaml)
└── README.md            # what it does, key hyperparams, references
```

Reference an existing architecture: see [`code_wm/`](https://github.com/eren23/crucible-community-tap/tree/main/architectures/code_wm) in the community tap.
