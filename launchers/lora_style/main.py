"""LoRA SFT on the user's personal-code-style corpus.

Pipeline:
  1. Load BASE_MODEL (Qwen2.5-Coder-1.5B by default) in bf16 on cuda.
  2. Load a HF dataset of code files (DATASET_ID, schema from
     scripts/build_eren_style_corpus.py: repo/path/lang/content).
  3. Chunk into fixed-window samples with a small file-path prompt prefix
     so the model learns "in this style/repo, write code like X".
  4. Wrap base model in a LoRA adapter (peft) and run SFTTrainer.
  5. Save the LoRA adapter under /workspace/project/lora_adapter and log
     as a W&B Artifact (name: lora-style-final / lora-style-step-N).

Downstream consumer: launchers/feature_diff_study/main.py with
LORA_WANDB_ARTIFACT=lora-style-final:latest and the existing sae-final
from project sae_train_qwen2_5_coder_1_5b.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import wandb
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer


def _env(name: str, default: Optional[str] = None) -> str:
    val = os.environ.get(name, default)
    if val is None:
        raise RuntimeError(f"required env var not set: {name}")
    return val


@dataclass
class Cfg:
    base_model: str
    dataset_id: str
    dataset_split: str
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    lr: float
    epochs: float
    max_steps: int
    batch_size: int
    grad_accum: int
    warmup_steps: int
    seq_len: int
    workspace: Path
    run_name: str
    wandb_project: str
    wandb_entity: Optional[str]


def load_cfg() -> Cfg:
    return Cfg(
        base_model=_env("BASE_MODEL"),
        dataset_id=_env("DATASET_ID", "eren23/eren-code-style"),
        dataset_split=_env("DATASET_SPLIT", "train"),
        lora_rank=int(_env("LORA_RANK", "32")),
        lora_alpha=int(_env("LORA_ALPHA", "64")),
        lora_dropout=float(_env("LORA_DROPOUT", "0.05")),
        lr=float(_env("LR", "2e-4")),
        epochs=float(_env("EPOCHS", "3")),
        max_steps=int(_env("MAX_STEPS", "-1")),
        batch_size=int(_env("BATCH_SIZE", "2")),
        grad_accum=int(_env("GRAD_ACCUM", "8")),
        warmup_steps=int(_env("WARMUP_STEPS", "50")),
        seq_len=int(_env("SEQ_LEN", "1024")),
        workspace=Path(_env("WORKSPACE_DIR", "/workspace/project")),
        run_name=_env("WANDB_RUN_NAME", f"lora-style-{int(time.time())}"),
        wandb_project=_env("WANDB_PROJECT", "coder-interp-pilot"),
        wandb_entity=os.environ.get("WANDB_ENTITY"),
    )


def format_record(rec: dict) -> str:
    """Prepend a small file-path/lang header so each sample carries context."""
    header = f"# repo: {rec['repo']}\n# path: {rec['path']}\n# lang: {rec['lang']}\n"
    return header + rec["content"]


def main() -> int:
    cfg = load_cfg()
    print(f"[main] cfg: {cfg}", flush=True)

    print(f"[ds] loading {cfg.dataset_id}:{cfg.dataset_split}", flush=True)
    ds = load_dataset(cfg.dataset_id, split=cfg.dataset_split)
    print(f"[ds] rows={len(ds)} cols={ds.column_names}", flush=True)
    ds = ds.map(
        lambda r: {"text": format_record(r)},
        remove_columns=[c for c in ds.column_names if c != "text"],
    )

    print(f"[base] loading {cfg.base_model}", flush=True)
    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.gradient_checkpointing_enable()

    peft_cfg = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=cfg.run_name,
        config={
            "base_model": cfg.base_model,
            "dataset_id": cfg.dataset_id,
            "lora_rank": cfg.lora_rank,
            "lora_alpha": cfg.lora_alpha,
            "lora_dropout": cfg.lora_dropout,
            "lr": cfg.lr,
            "epochs": cfg.epochs,
            "max_steps": cfg.max_steps,
            "batch_size": cfg.batch_size,
            "grad_accum": cfg.grad_accum,
            "seq_len": cfg.seq_len,
        },
    )

    sft_cfg = SFTConfig(
        output_dir=str(cfg.workspace / "lora_style_out"),
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr,
        num_train_epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        warmup_steps=cfg.warmup_steps,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        # TRL >= 0.13 renamed max_seq_length -> max_length; we use the new name
        # since the project YAML pins trl>=0.12 which can resolve to 0.13+.
        max_length=cfg.seq_len,
        packing=True,
        report_to=["wandb"],
        run_name=cfg.run_name,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=ds,
        # TRL >= 0.13 renamed `tokenizer` -> `processing_class`.
        processing_class=tok,
    )

    print("[train] starting SFT", flush=True)
    trainer.train()

    # Save final LoRA adapter only (not full base weights).
    adapter_dir = cfg.workspace / "lora_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir))
    print(f"[save] adapter -> {adapter_dir}", flush=True)

    artifact = wandb.Artifact("lora-style-final", type="model")
    artifact.add_dir(str(adapter_dir))
    wandb.log_artifact(artifact)
    wandb.finish()
    print("[main] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
