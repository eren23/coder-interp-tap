# Callbacks

Training callbacks for the NLA / SAE pipeline.

## Plugins

| Plugin | Purpose | Status |
|---|---|---|
| [`wandb_periodic_validation/`](wandb_periodic_validation/) | Periodic validation runner that logs metrics + sample artifacts (feature descriptions, AV NLL on a held-out set) to W&B every `EVAL_INTERVAL` steps. | Skeleton |

## Adding a new callback

```
callbacks/<name>/
├── plugin.yaml          # manifest
├── <name>.py            # callback class (subclass crucible.Callback)
└── README.md            # what it logs, when it fires
```
