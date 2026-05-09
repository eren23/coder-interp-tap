"""LoRA + DPO fine-tune launcher — STUB.

This launcher validates pod plumbing (HF cache layout, W&B init, model
download path) but does NOT yet run the actual TRL DPOTrainer. Real
implementation goes here when you start Track B in earnest.

Outline of the real pipeline (TODO):
  1. Build a small synthetic preference dataset:
       - sample N_PREFERENCE_PAIRS (prompt, ground_truth_commit) tuples
         from CommitPackFT (or load from local kai-overlay if available)
       - generate `chosen` and `rejected` responses by:
           * chosen   = the ground-truth diff from CommitPackFT
           * rejected = a random other commit's diff (low-quality mismatch)
         OR for richer signal, use the base model's own greedy completion
         as the rejected sample, ground truth as chosen.
  2. Load BASE_MODEL with peft.PeftModel.from_pretrained or QLoRA
     (USE_4BIT_BASE=1 toggles BitsAndBytesConfig 4-bit quant).
  3. trl.DPOTrainer with rank=LORA_RANK, alpha=LORA_ALPHA, beta=DPO_BETA,
     lr=DPO_LR, batch=DPO_BATCH_SIZE, grad_accum=DPO_GRAD_ACCUM,
     max_steps=DPO_TRAIN_STEPS.
  4. W&B: train/dpo_loss, train/policy_logps, train/reference_logps,
     train/rewards/chosen, train/rewards/rejected, train/rewards/accuracies.
  5. Save the LoRA adapter to /workspace/project/lora_adapter and log as a
     W&B Artifact.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional


def _env(name: str, default: Optional[str] = None) -> str:
    val = os.environ.get(name, default)
    if val is None:
        raise RuntimeError(f"required env var not set: {name}")
    return val


@dataclass
class Cfg:
    base_model: str
    use_4bit: bool
    lora_rank: int
    lora_alpha: int
    dpo_beta: float
    dpo_lr: float
    dpo_batch: int
    dpo_grad_accum: int
    dpo_train_steps: int
    n_pairs: int
    dataset_id: str
    dataset_subset: str
    run_name: str
    wandb_project: str
    wandb_entity: Optional[str]


def load_cfg() -> Cfg:
    return Cfg(
        base_model=_env("BASE_MODEL"),
        use_4bit=_env("USE_4BIT_BASE", "0") == "1",
        lora_rank=int(_env("LORA_RANK", "8")),
        lora_alpha=int(_env("LORA_ALPHA", "16")),
        dpo_beta=float(_env("DPO_BETA", "0.1")),
        dpo_lr=float(_env("DPO_LR", "5e-5")),
        dpo_batch=int(_env("DPO_BATCH_SIZE", "4")),
        dpo_grad_accum=int(_env("DPO_GRAD_ACCUM", "4")),
        dpo_train_steps=int(_env("DPO_TRAIN_STEPS", "10")),
        n_pairs=int(_env("N_PREFERENCE_PAIRS", "100")),
        dataset_id=_env("DATASET_ID", "bigcode/commitpackft"),
        dataset_subset=_env("DATASET_SUBSET", "python"),
        run_name=_env("WANDB_RUN_NAME", f"lora-dpo-{int(time.time())}"),
        wandb_project=_env("WANDB_PROJECT", "coder-interp-pilot"),
        wandb_entity=os.environ.get("WANDB_ENTITY"),
    )


def main() -> int:
    cfg = load_cfg()
    print(f"[stub] cfg: {cfg}", flush=True)

    try:
        import wandb
    except ImportError:
        print("[stub] wandb not installed; skipping W&B init", flush=True)
        wandb = None

    if wandb is not None:
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=cfg.run_name,
            config={
                "phase": "scaffold",
                "base_model": cfg.base_model,
                "lora_rank": cfg.lora_rank,
                "lora_alpha": cfg.lora_alpha,
                "dpo_beta": cfg.dpo_beta,
                "dpo_lr": cfg.dpo_lr,
                "dpo_train_steps": cfg.dpo_train_steps,
                "use_4bit": cfg.use_4bit,
            },
        )

    print(
        "[stub] LoRA + DPO trainer is NOT YET IMPLEMENTED.",
        flush=True,
    )
    print(
        "[stub] Real pipeline outlined in the module docstring. Implement and re-run.",
        flush=True,
    )
    print("[stub] Sanity: imports load, env parses, W&B init succeeded.", flush=True)

    if wandb is not None:
        wandb.log({"stub/phase": 0, "stub/n_pairs_planned": cfg.n_pairs})
        wandb.finish()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
