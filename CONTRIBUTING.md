# Contributing

This is a personal research tap. External contributions are welcome but the bar is high since the experiment direction is opinionated.

## Adding a plugin

1. Pick the right plugin type directory: `architectures/`, `callbacks/`, `data_adapters/`, `evaluation/`.
2. Create a subdirectory named after the plugin.
3. Add a `plugin.yaml` manifest:
   ```yaml
   name: my_plugin
   type: callback   # one of: architecture, callback, data_adapter, evaluation, ...
   version: 0.1.0
   description: One-line description.
   author: your-handle
   entrypoint: my_plugin.MyPluginClass
   dependencies:
     - torch>=2.5
     - wandb
   ```
4. Add the implementation file(s) alongside the manifest.
5. Test via `crucible run_project <project_yaml> --variant smoke` before opening a PR.

## Adding a project

1. Drop a YAML directly into `projects/` (no subdirectory).
2. Follow the schema used by `nla_qwen3_5_2b_pilot.yaml` — pod spec, install, train command, env_set, env_forward.
3. Always include a `smoke` variant that runs in <10 minutes for cheap iteration.
4. Reference plugins by name; let Crucible resolve them from this tap.

## Style

- Project YAMLs: comments above each non-obvious field. `env_forward` denylist (per Crucible) blocks RUNPOD_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, HF_TOKEN, etc. — put those in `.env.runpod.local` instead.
- Python plugins: type hints, no global state, no implicit dependencies on other plugins.
- README updates: bump the count in the table at the top whenever a plugin is added.
