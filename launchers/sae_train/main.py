"""TopK Sparse Autoencoder trainer over a HuggingFace base model's
residual stream.

Pipeline:
  1. Load base model (BASE_MODEL) in bf16 on cuda.
  2. Stream documents from a HF dataset (DATASET_ID / DATASET_SUBSET) and
     tokenize on the fly to SEQ_LEN.
  3. Forward CAPTURE_BATCH_SIZE sequences at a time; capture residuals
     at hidden_states[SAE_LAYER] for valid (non-pad) positions.
  4. Train a TopK SAE (d_in=D_MODEL, d_sae=D_SAE, top-k=TOPK) on the
     captured activations:
       - Adam, lr=SAE_LR
       - Loss = MSE(reconstruction, input) + AUX_COEF × dead-feature regularizer
       - Pre-bias decoder: encode (x - b_dec)
       - TopK activation
  5. Log to W&B every LOG_INTERVAL steps:
       - train/recon_loss
       - train/fraction_var_explained
       - train/dead_fraction (running)
       - train/buffer_size
  6. Save a checkpoint + W&B artifact every CHECKPOINT_INTERVAL steps.

Bespoke trainer (no SAELens dep) so it works directly on stock HF models
without TransformerLens compatibility concerns.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def _env(name: str, default: Optional[str] = None) -> str:
    val = os.environ.get(name, default)
    if val is None:
        raise RuntimeError(f"required env var not set: {name}")
    return val


@dataclass
class Cfg:
    base_model: str
    layer: int
    d_model: int
    d_sae: int
    topk: int
    aux_coef: float
    lr: float
    sae_batch: int
    capture_batch: int
    seq_len: int
    train_steps: int
    log_interval: int
    eval_interval: int
    checkpoint_interval: int
    dataset_id: str
    dataset_subset: str
    workspace: Path
    run_name: str
    wandb_project: str
    wandb_entity: Optional[str]


def load_cfg() -> Cfg:
    return Cfg(
        base_model=_env("BASE_MODEL"),
        layer=int(_env("SAE_LAYER")),
        d_model=int(_env("D_MODEL")),
        d_sae=int(_env("D_SAE")),
        topk=int(_env("TOPK", "50")),
        aux_coef=float(_env("SAE_AUX_LOSS_COEF", "0.0625")),
        lr=float(_env("SAE_LR", "3e-4")),
        sae_batch=int(_env("SAE_BATCH_SIZE", "4096")),
        capture_batch=int(_env("CAPTURE_BATCH_SIZE", "8")),
        seq_len=int(_env("SEQ_LEN", "512")),
        train_steps=int(_env("SAE_TRAIN_STEPS", "1000")),
        log_interval=int(_env("LOG_INTERVAL", "20")),
        eval_interval=int(_env("EVAL_INTERVAL", "200")),
        checkpoint_interval=int(_env("CHECKPOINT_INTERVAL", "1000")),
        dataset_id=_env("DATASET_ID", "bigcode/commitpackft"),
        dataset_subset=_env("DATASET_SUBSET", "python"),
        workspace=Path("/workspace/project"),
        run_name=_env("WANDB_RUN_NAME", f"sae-{int(time.time())}"),
        wandb_project=_env("WANDB_PROJECT", "coder-interp-pilot"),
        wandb_entity=os.environ.get("WANDB_ENTITY"),
    )


class TopKSAE(nn.Module):
    """Encoder + decoder with TopK activation. Pre-bias on encoder input."""

    def __init__(self, d_model: int, d_sae: int, topk: int):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.topk = topk
        # Kaiming-ish init.
        self.W_enc = nn.Parameter(torch.randn(d_model, d_sae) / (d_model ** 0.5))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.randn(d_sae, d_model) / (d_sae ** 0.5))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    def encode(self, x: torch.Tensor):
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc  # (B, d_sae)
        topk_val, topk_idx = pre.topk(self.topk, dim=-1)
        sparse = torch.zeros_like(pre)
        sparse.scatter_(-1, topk_idx, topk_val)
        return sparse, pre

    def decode(self, h: torch.Tensor):
        return h @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor):
        h, pre = self.encode(x)
        recon = self.decode(h)
        return recon, h, pre


def stream_activations(
    cfg: Cfg, model, tok, dataset_iter
) -> Iterator[torch.Tensor]:
    """Yield (N, d_model) activation tensors from sampled documents."""
    while True:
        batch_texts: list[str] = []
        for _ in range(cfg.capture_batch):
            try:
                sample = next(dataset_iter)
            except StopIteration:
                return
            text = sample.get("content") or sample.get("text") or sample.get("new_contents")
            if text:
                batch_texts.append(text)
        if not batch_texts:
            continue
        enc = tok(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.seq_len,
        ).to("cuda")
        with torch.inference_mode():
            out = model(**enc, output_hidden_states=True)
        hs = out.hidden_states[cfg.layer]  # (B, T, d)
        mask = enc.attention_mask.bool()
        # gather only non-pad positions
        flat = hs[mask]  # (N, d)
        yield flat.float()


def save_checkpoint(cfg: Cfg, sae: TopKSAE, step: int) -> Path:
    cfg.workspace.mkdir(parents=True, exist_ok=True)
    ckpt = cfg.workspace / f"sae_step_{step:06d}.pt"
    torch.save(
        {
            "step": step,
            "state_dict": sae.state_dict(),
            "config": {
                "d_model": cfg.d_model,
                "d_sae": cfg.d_sae,
                "topk": cfg.topk,
                "base_model": cfg.base_model,
                "layer": cfg.layer,
            },
        },
        ckpt,
    )
    return ckpt


def main() -> int:
    cfg = load_cfg()
    print(f"[main] cfg: {cfg}", flush=True)

    print(f"[base] loading {cfg.base_model} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )

    # Multi-subset support: DATASET_SUBSET="python,cpp,c" interleaves three
    # subsets so the SAE sees a multi-language activation distribution. Use
    # for training a coder SAE that needs to recognize C/C++ as well as
    # Python (e.g. for ggerganov-style author delta studies).
    subsets = [s.strip() for s in cfg.dataset_subset.split(",") if s.strip()]
    print(f"[ds] streaming {cfg.dataset_id} subsets={subsets}", flush=True)
    if len(subsets) == 1:
        ds = load_dataset(
            cfg.dataset_id,
            subsets[0],
            streaming=True,
            split="train",
            trust_remote_code=True,
        )
    else:
        from datasets import interleave_datasets
        sub_dss = [
            load_dataset(
                cfg.dataset_id,
                s,
                streaming=True,
                split="train",
                trust_remote_code=True,
            )
            for s in subsets
        ]
        ds = interleave_datasets(sub_dss, stopping_strategy="all_exhausted")
    ds_iter = iter(ds.shuffle(seed=42, buffer_size=1000))

    print(
        f"[sae] init TopKSAE d_model={cfg.d_model} d_sae={cfg.d_sae} k={cfg.topk}",
        flush=True,
    )
    sae = TopKSAE(cfg.d_model, cfg.d_sae, cfg.topk).cuda()
    opt = torch.optim.Adam(sae.parameters(), lr=cfg.lr)

    wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=cfg.run_name,
        config={
            "base_model": cfg.base_model,
            "layer": cfg.layer,
            "d_model": cfg.d_model,
            "d_sae": cfg.d_sae,
            "topk": cfg.topk,
            "lr": cfg.lr,
            "train_steps": cfg.train_steps,
            "sae_batch": cfg.sae_batch,
            "capture_batch": cfg.capture_batch,
            "seq_len": cfg.seq_len,
            "dataset_id": cfg.dataset_id,
            "dataset_subset": cfg.dataset_subset,
        },
    )

    activation_buffer = torch.empty(0, cfg.d_model, device="cuda")
    feature_fire_count = torch.zeros(cfg.d_sae, device="cuda")

    activation_stream = stream_activations(cfg, model, tok, ds_iter)
    last_log_t = time.time()
    step = 0
    while step < cfg.train_steps:
        # Refill buffer.
        while activation_buffer.shape[0] < cfg.sae_batch:
            try:
                new_acts = next(activation_stream)
            except StopIteration:
                # Re-iter the dataset if exhausted (streaming should be infinite for HF).
                ds_iter = iter(ds.shuffle(seed=int(time.time()), buffer_size=1000))
                activation_stream = stream_activations(cfg, model, tok, ds_iter)
                continue
            activation_buffer = torch.cat([activation_buffer, new_acts.cuda()], dim=0)

        batch = activation_buffer[: cfg.sae_batch]
        activation_buffer = activation_buffer[cfg.sae_batch :]

        recon, h, pre = sae(batch)
        recon_loss = F.mse_loss(recon, batch)

        # Aux loss: penalize features that haven't fired in a while via L2 on
        # their decoder rows scaled by inverse-firing-rate. Light touch.
        with torch.no_grad():
            firing_now = (h.abs() > 1e-6).any(dim=0).float()
            feature_fire_count.mul_(0.99).add_(firing_now)
            dead_mask = (feature_fire_count < 0.05).float()  # never-fires after warmup
        aux_loss = (sae.W_dec * dead_mask.unsqueeze(-1)).pow(2).sum() / max(
            dead_mask.sum().item(), 1.0
        )
        loss = recon_loss + cfg.aux_coef * aux_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % cfg.log_interval == 0:
            with torch.no_grad():
                input_var = batch.var()
                resid_var = (batch - recon).var()
                fve = (1 - resid_var / max(input_var.item(), 1e-8)).item()
                dead_fraction = (feature_fire_count < 0.05).float().mean().item()
            wandb.log(
                {
                    "train/recon_loss": float(recon_loss.item()),
                    "train/aux_loss": float(aux_loss.item()),
                    "train/total_loss": float(loss.item()),
                    "train/fraction_var_explained": float(fve),
                    "train/dead_fraction": float(dead_fraction),
                    "train/buffer_size": int(activation_buffer.shape[0]),
                },
                step=step,
            )
            elapsed = time.time() - last_log_t
            print(
                f"[step {step:>6}/{cfg.train_steps}] "
                f"recon={recon_loss.item():.4f} fve={fve:.3f} "
                f"dead={dead_fraction:.3f} buf={activation_buffer.shape[0]} "
                f"({elapsed:.1f}s/{cfg.log_interval} steps)",
                flush=True,
            )
            last_log_t = time.time()

        if (
            cfg.checkpoint_interval > 0
            and step > 0
            and step % cfg.checkpoint_interval == 0
        ):
            ckpt = save_checkpoint(cfg, sae, step)
            print(f"[ckpt] saved {ckpt}", flush=True)
            artifact = wandb.Artifact(f"sae-step-{step}", type="model")
            artifact.add_file(str(ckpt))
            wandb.log_artifact(artifact)

        step += 1

    # Final save.
    final = save_checkpoint(cfg, sae, step)
    print(f"[final] saved {final}", flush=True)
    artifact = wandb.Artifact("sae-final", type="model")
    artifact.add_file(str(final))
    wandb.log_artifact(artifact)
    wandb.finish()
    print("[main] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
