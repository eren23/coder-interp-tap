# What does a fine-tune actually learn?

Source for the Dria blog post on personal-style LoRA interpretability.

Self-contained HTML deliverable — open `article.html` in any browser. No build
step. No CDN dependencies at runtime. All widget data is inlined.

## Layout

```
dria-fine-tune-interp-post/
├── article.html               # the deliverable
├── meta.json                  # CMS metadata (title, slug, OG, authors, tags)
├── README.md                  # this file
├── widgets/
│   ├── components-browser.html    # Widget 1 — paginated alive-component browser
│   └── attribution-graph.html     # Widget 2 — multi-layer attribution graph
├── data/
│   ├── attribution_bundle.json    # 8 prompts × {base, lora} IG outputs (inlined into Widget 2)
│   ├── attribution/               # raw per-prompt JSONs as published to W&B
│   │   ├── xkwsi2qd/              #   base run (attribution-graphs-base:v0)
│   │   └── oy2hhprr/              #   lora run (attribution-graphs-lora:v0)
│   ├── explorer.json              # filtered alive-component table for Widget 1
│   ├── showcase_components.json   # six hand-picked components used in the article body
│   ├── sweep_manifest.json        # 16 W&B sweep runs (8 layers × {base, lora})
│   ├── feature_diff.json          # SAE drift table (200 features) — sourced from feature_diff_study
│   ├── coactivation.json          # auxiliary K↔V Jaccard data
│   └── prompt_heatmap.json        # auxiliary per-prompt gate ribbons
└── scripts/
    └── export_widget_data.py      # one-shot W&B → JSON exporter (idempotent)
```

## Reproducing the data

```bash
# from the repo root
python3 dria-fine-tune-interp-post/scripts/export_widget_data.py
```

The exporter pulls every blob from W&B project `eren23/coder-interp-pilot` and
writes byte-identical JSON modulo timestamps. Requires a `WANDB_API_KEY`.

## W&B runs referenced

| run id | role |
|---|---|
| `u20xk1jx` | SAE training (24K features) |
| `l8g7e3dy` | SAE feature drift study |
| `gze67ldu` | VPD-v2 winner analysis |
| `jfjid9o6` | VPD on LoRA-merged Coder-1.5B (single-layer L13) |
| `2zcsew0x` | VPD on BASE Coder-1.5B control (single-layer L13) |
| `niiz0d0u` | LoRA-style-final adapter |
| `tv0p4dr8`, `vpbhnnxw`, `yctbiwgm`, `aqh1gseb`, `tp64wyw6`, `xgj2p5og`, `20c9qn68`, `cki3d3fm` | 8-layer base sweep |
| `lzsdkvyi`, `hhazy118`, `62nybek7`, `ngx5wy4c`, `7f766bvf`, `sj37sykr`, `drqswaa0`, `5t8vmd74` | 8-layer LoRA sweep |
| `xkwsi2qd` | attribution-graphs-base IG run |
| `oy2hhprr` | attribution-graphs-lora IG run |

## Limitations

See the "Honest limitations" section of `article.html` for the full list. Short
version: small model (1.5B), short VPD schedule (1.5k steps vs the paper's 400k),
single-individual case study, no causal ablation validation.
