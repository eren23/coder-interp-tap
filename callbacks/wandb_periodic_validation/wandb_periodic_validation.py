"""Skeleton periodic-validation callback for W&B logging.

Drops into a Crucible training loop. Every `eval_interval` steps it runs a
validation pass and logs scalar metrics and a sample table to W&B. Every
`checkpoint_interval` steps it saves the model state.

This is a SKELETON. Wire `run_validation` to your actual eval logic when
the AV / SAE training loops exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import torch


@dataclass
class WandbPeriodicValidationConfig:
    eval_interval: int = 1000
    checkpoint_interval: int = 5000
    num_eval_samples: int = 200
    log_table_size: int = 20
    checkpoint_dir: str = "checkpoints"


class WandbPeriodicValidation:
    """Periodic validation runner. Hook into a training loop's `on_step_end`."""

    def __init__(
        self,
        config: WandbPeriodicValidationConfig,
        validation_fn: Callable[[int], dict[str, Any]],
        sample_fn: Callable[[int], Iterable[dict[str, Any]]] | None = None,
    ) -> None:
        self.config = config
        self.validation_fn = validation_fn
        self.sample_fn = sample_fn
        self._wandb = None

    def _wandb_module(self):
        if self._wandb is None:
            import wandb

            self._wandb = wandb
        return self._wandb

    def on_step_end(self, step: int, model: torch.nn.Module) -> None:
        if step > 0 and step % self.config.eval_interval == 0:
            self._run_validation(step)
        if (
            self.config.checkpoint_interval > 0
            and step > 0
            and step % self.config.checkpoint_interval == 0
        ):
            self._save_checkpoint(step, model)

    def _run_validation(self, step: int) -> None:
        wandb = self._wandb_module()
        metrics = self.validation_fn(step)
        wandb.log({f"val/{k}": v for k, v in metrics.items()}, step=step)

        if self.sample_fn is not None:
            rows = list(self.sample_fn(step))[: self.config.log_table_size]
            if rows:
                columns = sorted({k for row in rows for k in row.keys()})
                table = wandb.Table(columns=columns)
                for row in rows:
                    table.add_data(*[row.get(c) for c in columns])
                wandb.log({"val/samples": table}, step=step)

    def _save_checkpoint(self, step: int, model: torch.nn.Module) -> None:
        out_dir = Path(self.config.checkpoint_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"step_{step:08d}.pt"
        torch.save({"step": step, "model_state": model.state_dict()}, path)

        wandb = self._wandb_module()
        artifact = wandb.Artifact(
            name=f"checkpoint-step-{step}",
            type="model",
            metadata={"step": step},
        )
        artifact.add_file(str(path))
        wandb.log_artifact(artifact)
