"""Feature-diff delta study — end-to-end demo of the Track-C measurement pipeline.

Pipeline (all phases run in sequence on one pod):
  1. Load base Qwen2.5-Coder-1.5B.
  2. Build a small biased SFT dataset from a CommitPackFT slice.
  3. LoRA-SFT for LORA_TRAIN_STEPS steps. Save adapter.
  4. Forward HOLDOUT_TOKENS held-out tokens through BOTH the baseline and
     the merged LoRA model, capture L6 hidden states.
  5. Apply the frozen SAE to both. Per-feature: firing rate, mean active
     magnitude.
  6. Compute log-ratio of firing rates. Pick top-K features by |log-ratio|.
  7. Look up each top feature's NL description from the feature_descriptions
     W&B artifact.
  8. Log a wandb.Table.

This is intentionally a single-pass linear script for clarity. It loads
two model copies sequentially when possible to fit on a 4090.
"""

from __future__ import annotations

import gc
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _env(name: str, default: Optional[str] = None) -> str:
    val = os.environ.get(name, default)
    if val is None:
        raise RuntimeError(f"required env var not set: {name}")
    return val


@dataclass
class Cfg:
    base_model: str
    sae_layer: int
    d_model: int
    d_sae: int
    topk: int
    sae_wandb_artifact: str
    descriptions_wandb_artifact: str
    lora_rank: int
    lora_alpha: int
    lora_lr: float
    lora_batch: int
    lora_grad_accum: int
    lora_max_seq_len: int
    n_lora_examples: int
    lora_train_steps: int
    bias_filter_key: str
    bias_filter_value: str
    holdout_tokens: int
    top_k_features: int
    capture_batch_size: int
    seq_len: int
    workspace: Path
    run_name: str
    wandb_project: str
    wandb_entity: Optional[str]


def load_cfg() -> Cfg:
    return Cfg(
        base_model=_env("BASE_MODEL"),
        sae_layer=int(_env("SAE_LAYER")),
        d_model=int(_env("D_MODEL")),
        d_sae=int(_env("D_SAE")),
        topk=int(_env("TOPK", "50")),
        sae_wandb_artifact=_env("SAE_WANDB_ARTIFACT", "sae-final:latest"),
        descriptions_wandb_artifact=_env(
            "DESCRIPTIONS_WANDB_ARTIFACT", "feature-descriptions:latest"
        ),
        lora_rank=int(_env("LORA_RANK", "8")),
        lora_alpha=int(_env("LORA_ALPHA", "16")),
        lora_lr=float(_env("LORA_LR", "1e-4")),
        lora_batch=int(_env("LORA_BATCH_SIZE", "2")),
        lora_grad_accum=int(_env("LORA_GRAD_ACCUM", "4")),
        lora_max_seq_len=int(_env("LORA_MAX_SEQ_LEN", "512")),
        n_lora_examples=int(_env("N_LORA_EXAMPLES", "100")),
        lora_train_steps=int(_env("LORA_TRAIN_STEPS", "200")),
        bias_filter_key=_env("BIAS_FILTER_KEY", "lang"),
        bias_filter_value=_env("BIAS_FILTER_VALUE", "Python"),
        holdout_tokens=int(_env("HOLDOUT_TOKENS", "10000")),
        top_k_features=int(_env("TOP_K_FEATURES", "100")),
        capture_batch_size=int(_env("CAPTURE_BATCH_SIZE", "8")),
        seq_len=int(_env("SEQ_LEN", "512")),
        workspace=Path("/workspace/project"),
        run_name=_env("WANDB_RUN_NAME", f"feature-diff-{int(time.time())}"),
        wandb_project=_env("WANDB_PROJECT", "coder-interp-pilot"),
        wandb_entity=os.environ.get("WANDB_ENTITY"),
    )


# ---------------------------------------------------------------------------
# SAE
# ---------------------------------------------------------------------------


def make_sae(cfg: Cfg):
    import torch
    import torch.nn as nn

    class TopKSAE(nn.Module):
        def __init__(self, d_model: int, d_sae: int, topk: int):
            super().__init__()
            self.d_model = d_model
            self.d_sae = d_sae
            self.topk = topk
            self.W_enc = nn.Parameter(torch.empty(d_model, d_sae))
            self.b_enc = nn.Parameter(torch.empty(d_sae))
            self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
            self.b_dec = nn.Parameter(torch.empty(d_model))

        def encode(self, x: torch.Tensor):
            pre = (x - self.b_dec) @ self.W_enc + self.b_enc
            topk_val, topk_idx = pre.topk(self.topk, dim=-1)
            sparse = torch.zeros_like(pre)
            sparse.scatter_(-1, topk_idx, topk_val)
            return sparse

    sae = TopKSAE(cfg.d_model, cfg.d_sae, cfg.topk).cuda()
    sae.requires_grad_(False)
    return sae


def load_sae(cfg: Cfg, sae) -> None:
    """Download SAE artifact from W&B and load into the SAE module."""
    import torch
    import wandb

    api = wandb.Api()
    art = api.artifact(
        f"{cfg.wandb_entity}/{cfg.wandb_project}/{cfg.sae_wandb_artifact}"
    )
    download_dir = Path(art.download(root=str(cfg.workspace / "sae_artifact")))
    pts = list(download_dir.glob("*.pt"))
    if not pts:
        raise RuntimeError(f"no .pt in {download_dir}")
    state = torch.load(pts[0], map_location="cuda", weights_only=False)
    sae.load_state_dict(state["state_dict"])
    print(f"[sae] loaded step={state.get('step')} from {pts[0]}", flush=True)


def load_descriptions(cfg: Cfg) -> dict[int, str]:
    import wandb

    api = wandb.Api()
    art = api.artifact(
        f"{cfg.wandb_entity}/{cfg.wandb_project}/{cfg.descriptions_wandb_artifact}"
    )
    download_dir = Path(art.download(root=str(cfg.workspace / "descriptions_artifact")))
    files = list(download_dir.glob("*.json"))
    if not files:
        raise RuntimeError(f"no descriptions JSON in {download_dir}")
    raw = json.loads(files[0].read_text())
    out: dict[int, str] = {}
    for k, v in raw.items():
        try:
            out[int(k)] = v.get("description", "") if isinstance(v, dict) else str(v)
        except Exception:
            pass
    print(f"[desc] loaded {len(out)} feature descriptions", flush=True)
    return out


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def build_bias_dataset(cfg: Cfg, tok):
    """Stream commitpackft and pull N samples that match the bias filter."""
    from datasets import load_dataset

    print(
        f"[data] streaming commitpackft, filter {cfg.bias_filter_key}={cfg.bias_filter_value}",
        flush=True,
    )
    ds = load_dataset(
        "bigcode/commitpackft",
        cfg.bias_filter_value if cfg.bias_filter_key == "lang" else None,
        streaming=True,
        split="train",
        trust_remote_code=True,
    )

    samples: list[str] = []
    for sample in ds:
        text = sample.get("new_contents") or sample.get("content") or sample.get("text")
        if not text:
            continue
        samples.append(text[: cfg.lora_max_seq_len * 4])  # rough char cap
        if len(samples) >= cfg.n_lora_examples:
            break

    print(f"[data] gathered {len(samples)} bias examples", flush=True)
    return samples


def held_out_texts(cfg: Cfg, n: int) -> list[str]:
    """Pull n texts NOT in the bias-filtered slice (use a different seed)."""
    from datasets import load_dataset

    ds = load_dataset(
        "bigcode/commitpackft",
        "Python",
        streaming=True,
        split="train",
        trust_remote_code=True,
    )
    ds = ds.shuffle(seed=2026, buffer_size=2000)
    out: list[str] = []
    for sample in ds:
        text = sample.get("new_contents") or sample.get("content") or sample.get("text")
        if text:
            out.append(text[: cfg.seq_len * 4])
        if len(out) >= n:
            break
    return out


# ---------------------------------------------------------------------------
# LoRA SFT
# ---------------------------------------------------------------------------


def lora_finetune(cfg: Cfg, base_model, tok, samples) -> Path:
    """LoRA-SFT on `samples`, save adapter, return adapter dir."""
    import torch
    import wandb
    from peft import LoraConfig, get_peft_model

    lora_cfg = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    base_model = get_peft_model(base_model, lora_cfg)
    base_model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        [p for p in base_model.parameters() if p.requires_grad], lr=cfg.lora_lr
    )

    base_model.train()
    step = 0
    micro = 0
    sample_idx = 0
    while step < cfg.lora_train_steps:
        batch = []
        for _ in range(cfg.lora_batch):
            batch.append(samples[sample_idx % len(samples)])
            sample_idx += 1
        enc = tok(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.lora_max_seq_len,
        ).to("cuda")
        labels = enc.input_ids.clone()
        labels[enc.attention_mask == 0] = -100
        out = base_model(**enc, labels=labels)
        loss = out.loss / cfg.lora_grad_accum
        loss.backward()
        micro += 1
        if micro >= cfg.lora_grad_accum:
            optimizer.step()
            optimizer.zero_grad()
            micro = 0
            step += 1
            if step % 10 == 0:
                wandb.log({"lora/loss": float(out.loss.item())}, step=step)
                print(f"[lora step {step}/{cfg.lora_train_steps}] loss={out.loss.item():.4f}", flush=True)

    adapter_dir = cfg.workspace / "lora_adapter"
    base_model.save_pretrained(str(adapter_dir))
    print(f"[lora] saved adapter to {adapter_dir}", flush=True)
    return adapter_dir


# ---------------------------------------------------------------------------
# Activation capture
# ---------------------------------------------------------------------------


def capture_features(cfg: Cfg, model, tok, sae, texts: list[str]) -> tuple:
    """Forward `texts` through model, encode through SAE, accumulate
    per-feature firing count + summed magnitude. Returns (firings, sums, total_tokens)."""
    import torch

    firings = torch.zeros(cfg.d_sae, device="cuda")
    sums = torch.zeros(cfg.d_sae, device="cuda")
    total_tokens = 0
    last_print = time.time()

    i = 0
    while i < len(texts):
        batch = texts[i : i + cfg.capture_batch_size]
        i += len(batch)
        enc = tok(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.seq_len,
        ).to("cuda")
        with torch.inference_mode():
            out = model(**enc, output_hidden_states=True)
        hs = out.hidden_states[cfg.sae_layer]
        attn = enc.attention_mask.bool()
        flat = hs[attn].reshape(-1, cfg.d_model).float()
        if flat.shape[0] == 0:
            continue
        with torch.inference_mode():
            sparse = sae.encode(flat)  # (N, d_sae)
        nonzero = (sparse > 1e-6).float()
        firings += nonzero.sum(dim=0)
        sums += sparse.clamp(min=0).sum(dim=0)
        total_tokens += flat.shape[0]
        if total_tokens >= cfg.holdout_tokens:
            break
        if time.time() - last_print > 30:
            print(f"[capture] {total_tokens}/{cfg.holdout_tokens} tokens", flush=True)
            last_print = time.time()

    return firings, sums, total_tokens


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    cfg = load_cfg()
    print(f"[main] cfg base={cfg.base_model} layer={cfg.sae_layer} d_sae={cfg.d_sae}", flush=True)

    import torch
    import wandb
    from transformers import AutoModelForCausalLM, AutoTokenizer

    wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=cfg.run_name,
        config={
            "base_model": cfg.base_model,
            "sae_layer": cfg.sae_layer,
            "d_sae": cfg.d_sae,
            "lora_rank": cfg.lora_rank,
            "lora_alpha": cfg.lora_alpha,
            "lora_train_steps": cfg.lora_train_steps,
            "n_lora_examples": cfg.n_lora_examples,
            "holdout_tokens": cfg.holdout_tokens,
            "bias_filter_key": cfg.bias_filter_key,
            "bias_filter_value": cfg.bias_filter_value,
        },
    )

    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"[base] loading {cfg.base_model} ...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )

    # Load SAE + descriptions before training.
    sae = make_sae(cfg)
    load_sae(cfg, sae)
    descriptions = load_descriptions(cfg)

    bias_samples = build_bias_dataset(cfg, tok)

    # Phase 1: LoRA-SFT
    adapter_dir = lora_finetune(cfg, base, tok, bias_samples)

    # Phase 2: capture baseline activations.
    # Reload baseline model fresh (the LoRA-tuned one is the same instance, mutated).
    print("[capture] reloading clean base for baseline pass ...", flush=True)
    del base
    gc.collect()
    torch.cuda.empty_cache()

    base = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    holdout = held_out_texts(cfg, n=max(64, cfg.holdout_tokens // cfg.seq_len))
    print(f"[capture] baseline forward over {len(holdout)} held-out texts", flush=True)
    fire_b, sum_b, tok_b = capture_features(cfg, base, tok, sae, holdout)
    del base
    gc.collect()
    torch.cuda.empty_cache()

    # Phase 3: capture tuned activations.
    print(f"[capture] loading base+LoRA adapter from {adapter_dir}", flush=True)
    from peft import PeftModel

    base = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    tuned = PeftModel.from_pretrained(base, str(adapter_dir))
    tuned = tuned.merge_and_unload()  # flatten to a normal HF model
    print(f"[capture] tuned forward over {len(holdout)} held-out texts", flush=True)
    fire_t, sum_t, tok_t = capture_features(cfg, tuned, tok, sae, holdout)
    del tuned, base
    gc.collect()
    torch.cuda.empty_cache()

    # Phase 4: compute deltas.
    rate_b = (fire_b / max(tok_b, 1)).clamp(min=1e-9)
    rate_t = (fire_t / max(tok_t, 1)).clamp(min=1e-9)
    log_ratio = (rate_t.log() - rate_b.log())  # ln-scale
    log_ratio_abs = log_ratio.abs()
    mean_b = (sum_b / fire_b.clamp(min=1)).cpu()
    mean_t = (sum_t / fire_t.clamp(min=1)).cpu()

    top_idx = log_ratio_abs.topk(cfg.top_k_features).indices.cpu().tolist()

    # Phase 5: log W&B Table.
    columns = [
        "feature_idx",
        "rate_baseline",
        "rate_tuned",
        "log_ratio_e",
        "log2_ratio",
        "mean_act_baseline",
        "mean_act_tuned",
        "description",
    ]
    table = wandb.Table(columns=columns)
    print(f"[diff] top {cfg.top_k_features} features by |log-ratio|:", flush=True)
    for i, idx in enumerate(top_idx):
        rb = float(rate_b[idx].item())
        rt = float(rate_t[idx].item())
        lr_e = float(log_ratio[idx].item())
        lr_2 = lr_e / math.log(2)
        mb = float(mean_b[idx].item())
        mt = float(mean_t[idx].item())
        desc = descriptions.get(idx, "")
        table.add_data(int(idx), rb, rt, lr_e, lr_2, mb, mt, desc)
        if i < 25:
            arrow = "↑" if lr_e > 0 else "↓"
            print(
                f"  feat {idx:>5} {arrow} log2={lr_2:+.2f}  "
                f"rb={rb:.4f} rt={rt:.4f}  desc: {desc[:120]}",
                flush=True,
            )

    wandb.log({"diff/top_features": table})
    wandb.log(
        {
            "diff/n_features_with_changes": int((log_ratio_abs > 0.5).sum().item()),
            "diff/baseline_tokens": tok_b,
            "diff/tuned_tokens": tok_t,
            "diff/median_abs_log_ratio": float(log_ratio_abs.median().item()),
            "diff/p99_abs_log_ratio": float(log_ratio_abs.quantile(0.99).item()),
        }
    )
    wandb.finish()

    print("[main] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
