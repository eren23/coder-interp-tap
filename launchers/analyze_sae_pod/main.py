"""SAE interpretability analysis launcher (pod-side).

Runs the SAE feature top-context analysis on a GPU pod so we don't have
to fight MPS / CPU model-load issues locally. Designed to be invoked by
projects/analyze_sae_qwen2_5_coder_1_5b.yaml.

Pipeline:
  1. Download a previously-trained SAE artifact (default sae-final from
     the coder-interp-pilot project) via W&B.
  2. Load the base Qwen2.5-Coder-1.5B model on cuda in bf16.
  3. Stream the user's personal-code-style HF dataset.
  4. Pass-1: compute per-feature fire rate; pick top N features.
  5. Pass-2: for those features, collect top-K firing token contexts.
  6. Log a W&B Table {feature, fire_rate, activation, context} plus
     per-feature fire-rate / mean-activation plots.

This is the GPU-pod counterpart to scripts/analyze_sae.py.
"""

from __future__ import annotations

import heapq
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import wandb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def _env(name: str, default: Optional[str] = None) -> str:
    val = os.environ.get(name, default)
    if val is None:
        raise RuntimeError(f"required env var not set: {name}")
    return val


class TopKSAE(nn.Module):
    def __init__(self, d_model: int, d_sae: int, topk: int):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.topk = topk
        self.W_enc = nn.Parameter(torch.randn(d_model, d_sae) / (d_model ** 0.5))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.randn(d_sae, d_model) / (d_sae ** 0.5))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    def encode(self, x: torch.Tensor):
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc
        topk_val, topk_idx = pre.topk(self.topk, dim=-1)
        sparse = torch.zeros_like(pre)
        sparse.scatter_(-1, topk_idx, topk_val)
        return sparse


@dataclass
class Cfg:
    base_model: str
    layer: int
    source_wandb_run: str      # full path: entity/project/run_id (where SAE lives)
    artifact_name: str         # e.g. sae-final
    dataset_id: str
    dataset_split: str
    num_samples: int
    seq_len: int
    num_features: int
    top_k_contexts: int
    ctx: int
    workspace: Path
    run_name: str
    wandb_project: str
    wandb_entity: Optional[str]


def load_cfg() -> Cfg:
    return Cfg(
        base_model=_env("BASE_MODEL", "Qwen/Qwen2.5-Coder-1.5B"),
        layer=int(_env("SAE_LAYER", "6")),
        source_wandb_run=_env("SOURCE_WANDB_RUN", "eren23/coder-interp-pilot/u20xk1jx"),
        artifact_name=_env("ARTIFACT_NAME", "sae-final"),
        dataset_id=_env("DATASET_ID", "eren23/eren-code-style"),
        dataset_split=_env("DATASET_SPLIT", "train"),
        num_samples=int(_env("NUM_SAMPLES", "200")),
        seq_len=int(_env("SEQ_LEN", "256")),
        num_features=int(_env("NUM_FEATURES", "16")),
        top_k_contexts=int(_env("TOP_K_CONTEXTS", "8")),
        ctx=int(_env("CONTEXT_WINDOW", "10")),
        workspace=Path(_env("WORKSPACE_DIR", "/workspace/project")),
        run_name=_env("WANDB_RUN_NAME", f"sae-analysis-{int(time.time())}"),
        wandb_project=_env("WANDB_PROJECT", "coder-interp-pilot"),
        wandb_entity=os.environ.get("WANDB_ENTITY"),
    )


def fetch_artifact(source_run: str, artifact_name: str, out_dir: Path) -> Path:
    api = wandb.Api()
    run = api.run(source_run)
    arts = [a for a in run.logged_artifacts() if a.name.startswith(artifact_name)]
    if not arts:
        raise RuntimeError(f"no artifact starting with {artifact_name} on {source_run}")
    art = arts[-1]
    print(f"[wandb] downloading {art.name} ({art.size/1e6:.2f} MB)", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    art.download(root=str(out_dir))
    return next(out_dir.glob("*.pt"))


def main() -> int:
    cfg = load_cfg()
    print(f"[main] cfg: {cfg}", flush=True)

    ckpt_dir = cfg.workspace / "sae_ckpt"
    ckpt = fetch_artifact(cfg.source_wandb_run, cfg.artifact_name, ckpt_dir)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    sae_cfg = state["config"]
    print(f"[sae] cfg={sae_cfg}", flush=True)
    sae = TopKSAE(sae_cfg["d_model"], sae_cfg["d_sae"], sae_cfg["topk"]).cuda()
    sae.load_state_dict(state["state_dict"])
    sae.train(False)

    print(f"[base] loading {cfg.base_model}", flush=True)
    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, torch_dtype=torch.bfloat16, device_map="cuda",
    )
    model.train(False)
    for p in model.parameters():
        p.requires_grad_(False)

    wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=cfg.run_name,
        config={
            "base_model": cfg.base_model,
            "layer": cfg.layer,
            "source_wandb_run": cfg.source_wandb_run,
            "artifact_name": cfg.artifact_name,
            "dataset_id": cfg.dataset_id,
            "num_samples": cfg.num_samples,
            "seq_len": cfg.seq_len,
            "num_features": cfg.num_features,
            "top_k_contexts": cfg.top_k_contexts,
            "d_sae": sae_cfg["d_sae"],
            "topk": sae_cfg["topk"],
        },
    )

    print(f"[ds] loading {cfg.dataset_id}:{cfg.dataset_split}", flush=True)
    ds = load_dataset(cfg.dataset_id, split=cfg.dataset_split)
    print(f"[ds] rows={len(ds)} cols={ds.column_names}", flush=True)
    rows = list(ds.shuffle(seed=42).select(range(min(cfg.num_samples, len(ds)))))

    captured = {}
    def hook(_m, _i, out):
        captured["hs"] = out[0].detach() if isinstance(out, tuple) else out.detach()
    layer_mod = model.model.layers[cfg.layer]
    handle = layer_mod.register_forward_hook(hook)

    # Pass 1: per-feature fire rate.
    D = sae_cfg["d_sae"]
    fire_count = torch.zeros(D, device="cuda")
    total_pos = 0
    for sid, rec in enumerate(rows):
        text = rec.get("content") or rec.get("text") or ""
        if not text:
            continue
        enc = tok(text, return_tensors="pt", truncation=True, max_length=cfg.seq_len).to("cuda")
        with torch.inference_mode():
            model(**enc, use_cache=False)
        hs = captured["hs"][0].float()
        with torch.inference_mode():
            sparse = sae.encode(hs)
        fired = (sparse.abs() > 1e-6).any(dim=0).float()
        fire_count += fired
        total_pos += hs.shape[0]
        if (sid + 1) % 20 == 0:
            print(f"[pass1] {sid+1}/{len(rows)} (tokens={total_pos})", flush=True)
    fire_rate = (fire_count / max(len(rows), 1)).cpu()
    chosen = fire_rate.topk(cfg.num_features).indices.tolist()
    print(
        f"[pass1] done. tokens={total_pos}, chosen {cfg.num_features} features: {chosen}",
        flush=True,
    )

    # Pass 2: top contexts for the chosen features.
    heaps: dict[int, list[tuple[float, int, int]]] = {f: [] for f in chosen}
    sample_token_strings: list[list[str]] = []
    for sid, rec in enumerate(rows):
        text = rec.get("content") or rec.get("text") or ""
        if not text:
            sample_token_strings.append([])
            continue
        enc = tok(text, return_tensors="pt", truncation=True, max_length=cfg.seq_len).to("cuda")
        ids = enc.input_ids[0].tolist()
        toks = [tok.decode([i]) for i in ids]
        sample_token_strings.append(toks)
        with torch.inference_mode():
            model(**enc, use_cache=False)
        hs = captured["hs"][0].float()
        with torch.inference_mode():
            sparse = sae.encode(hs)
        for f in chosen:
            col = sparse[:, f]
            for pos, v in enumerate(col.tolist()):
                if v <= 0:
                    continue
                if len(heaps[f]) < cfg.top_k_contexts:
                    heapq.heappush(heaps[f], (v, sid, pos))
                else:
                    if v > heaps[f][0][0]:
                        heapq.heapreplace(heaps[f], (v, sid, pos))
        if (sid + 1) % 20 == 0:
            print(f"[pass2] {sid+1}/{len(rows)}", flush=True)
    handle.remove()

    # Build W&B table + plots.
    feature_table = wandb.Table(
        columns=["feature", "rank", "fire_rate", "activation", "repo", "path", "context"]
    )
    for rank, f in enumerate(chosen):
        items = sorted(heaps[f], reverse=True)
        for v, sid, pos in items:
            toks = sample_token_strings[sid]
            lo = max(0, pos - cfg.ctx)
            hi = min(len(toks), pos + cfg.ctx + 1)
            ctx_str = (
                "".join(toks[lo:pos])
                + " ⟦" + toks[pos] + "⟧ "
                + "".join(toks[pos + 1 : hi])
            ).replace("\n", "\\n")
            feature_table.add_data(
                int(f), rank, float(fire_rate[f].item()), float(v),
                rows[sid].get("repo", ""), rows[sid].get("path", ""),
                ctx_str,
            )
    wandb.log({"feature_top_contexts": feature_table})

    # Per-feature fire rate plot for the WHOLE D, sorted.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(D), sorted(fire_rate.tolist(), reverse=True))
    ax.set_xlabel("feature (sorted by fire rate)")
    ax.set_ylabel("fire rate")
    ax.set_title(f"SAE feature fire rate (top {D} features, {total_pos} tokens)")
    p1 = cfg.workspace / "fire_rate.png"
    fig.tight_layout()
    fig.savefig(p1, dpi=120)
    plt.close(fig)
    wandb.log({"fire_rate_plot": wandb.Image(str(p1))})

    # Tokens-per-active-feature histogram (sparsity).
    nonzero_per_token: list[int] = []
    for rec in rows[: min(50, len(rows))]:
        text = rec.get("content") or rec.get("text") or ""
        if not text:
            continue
        enc = tok(text, return_tensors="pt", truncation=True, max_length=cfg.seq_len).to("cuda")
        with torch.inference_mode():
            model(**enc, use_cache=False)
        hs = captured["hs"][0].float()
        with torch.inference_mode():
            sparse = sae.encode(hs)
        per_tok = (sparse.abs() > 1e-6).sum(dim=1).cpu().tolist()
        nonzero_per_token.extend(per_tok)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(nonzero_per_token, bins=40)
    ax.set_xlabel("active features per token (top-K SAE)")
    ax.set_ylabel("token count")
    ax.set_title("SAE per-token activity (should center around K)")
    p2 = cfg.workspace / "active_per_token.png"
    fig.tight_layout()
    fig.savefig(p2, dpi=120)
    plt.close(fig)
    wandb.log({"active_per_token_plot": wandb.Image(str(p2))})

    # Persist a JSON dump too.
    summary = {
        "source_wandb_run": cfg.source_wandb_run,
        "artifact_name": cfg.artifact_name,
        "n_samples": len(rows),
        "n_tokens": int(total_pos),
        "n_features_inspected": cfg.num_features,
        "chosen_features": [int(f) for f in chosen],
        "fire_rate": {int(f): float(fire_rate[f].item()) for f in chosen},
    }
    out_path = cfg.workspace / "sae_analysis_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    art = wandb.Artifact("sae-analysis-summary", type="analysis")
    art.add_file(str(out_path))
    art.add_file(str(p1))
    art.add_file(str(p2))
    wandb.log_artifact(art)

    wandb.finish()
    print("[main] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
