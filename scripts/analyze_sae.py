"""Quick local top-context analysis for an SAE artifact.

Goal: prove the SAE's features are interpretable by showing, for a
handful of high-firing features, their top-K firing contexts on a real
code stream — WITHOUT running the full LLM-labeling pipeline.

Loads the W&B SAE artifact, runs Qwen2.5-Coder-1.5B on local code from
the user's style corpus JSONL, captures residuals at SAE_LAYER, computes
sparse encodings, and for each chosen feature prints the top-K tokens
where it fires hardest with surrounding context.
"""

from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class TopKSAE(nn.Module):
    """Same layout as launchers/sae_train/main.py."""

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


def fetch_artifact(wandb_run: str, artifact: str, out_dir: Path) -> Path:
    import wandb
    api = wandb.Api()
    run = api.run(wandb_run)
    arts = [a for a in run.logged_artifacts() if a.name.startswith(artifact.split(":")[0])]
    if not arts:
        raise RuntimeError(f"no artifact matching {artifact}")
    art = arts[-1]
    print(f"[wandb] downloading {art.name} ({art.size/1e6:.2f} MB)", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    art.download(root=str(out_dir))
    return next(out_dir.glob("*.pt"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wandb-run", required=True)
    ap.add_argument("--artifact", default="sae-final")
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-Coder-1.5B")
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--local-jsonl", required=True)
    ap.add_argument("--num-samples", type=int, default=40,
                    help="Number of code files to process")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--num-features", type=int, default=8,
                    help="Number of features to inspect (top by fire rate)")
    ap.add_argument("--top-k-contexts", type=int, default=5)
    ap.add_argument("--ctx", type=int, default=8,
                    help="Tokens of context each side of firing token")
    ap.add_argument("--out-dir", default="/tmp/sae_analysis")
    cfg = ap.parse_args()

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = fetch_artifact(cfg.wandb_run, cfg.artifact, out_dir / "ckpt")
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    sae_cfg = state["config"]
    print(f"[sae] cfg={sae_cfg}", flush=True)
    sae = TopKSAE(sae_cfg["d_model"], sae_cfg["d_sae"], sae_cfg["topk"])
    sae.load_state_dict(state["state_dict"])

    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32
    sae = sae.to(device)
    sae.train(False)

    print(f"[base] loading {cfg.base_model} on {device}/{dtype}", flush=True)
    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, torch_dtype=dtype, device_map=device,
    )
    model.train(False)
    for p in model.parameters():
        p.requires_grad_(False)

    # Per-feature heap of top contexts: heap of (-activation, sample_id, pos)
    # plus sample storage for token-string lookup later.
    samples: list[list[str]] = []
    # Track activations: store sparse (feature_id -> heap of negatives) but we
    # don't know which features matter yet. Two-pass approach is too expensive,
    # so single-pass: keep per-feature heap of size top_k_contexts for ALL features.

    D = sae_cfg["d_sae"]
    # For efficiency: only track features after a warmup sweep determined which
    # features fire at all. We do a quick sweep first to find candidates.

    # ---- Pass 1: count firing rates ----
    fire_count = torch.zeros(D)
    total_pos = 0
    rows = []
    with open(cfg.local_jsonl) as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= cfg.num_samples:
                break

    print(f"[ds] loaded {len(rows)} files from JSONL", flush=True)

    captured = {}
    def hook(_m, _i, out):
        captured["hs"] = out[0].detach() if isinstance(out, tuple) else out.detach()

    layer_mod = model.model.layers[cfg.layer]
    handle = layer_mod.register_forward_hook(hook)

    sample_texts: list[list[str]] = []  # token strings per sample
    per_token_top: list[tuple[float, int, int, int]] = []
    # Will build (-act, sample_id, pos, feature_id) heaps PER feature in pass 2.

    # Pass 1: just compute firing counts to pick top features
    for sid, rec in enumerate(rows):
        text = rec.get("content") or rec.get("text") or ""
        if not text:
            continue
        enc = tok(text, return_tensors="pt", truncation=True, max_length=cfg.seq_len).to(device)
        with torch.inference_mode():
            model(**enc, use_cache=False)
        hs = captured["hs"][0].float()  # (T, d)
        with torch.inference_mode():
            sparse = sae.encode(hs.to(device))
        fired = (sparse.abs() > 1e-6).any(dim=0).float().cpu()
        fire_count += fired
        total_pos += hs.shape[0]
        if (sid + 1) % 10 == 0:
            print(f"[pass1] {sid+1}/{len(rows)}", flush=True)

    fire_rate = fire_count / max(len(rows), 1)
    chosen = fire_rate.topk(cfg.num_features).indices.tolist()
    print(
        f"[pass1] done. tokens={total_pos}, chosen {cfg.num_features} features: {chosen}",
        flush=True,
    )

    # ---- Pass 2: for the chosen features only, gather top contexts ----
    heaps: dict[int, list[tuple[float, int, int]]] = {f: [] for f in chosen}
    sample_token_strings: list[list[str]] = []
    for sid, rec in enumerate(rows):
        text = rec.get("content") or rec.get("text") or ""
        if not text:
            sample_token_strings.append([])
            continue
        enc = tok(text, return_tensors="pt", truncation=True, max_length=cfg.seq_len).to(device)
        ids = enc.input_ids[0].tolist()
        toks = [tok.decode([i]) for i in ids]
        sample_token_strings.append(toks)
        with torch.inference_mode():
            model(**enc, use_cache=False)
        hs = captured["hs"][0].float().to(device)
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
        if (sid + 1) % 10 == 0:
            print(f"[pass2] {sid+1}/{len(rows)}", flush=True)
    handle.remove()

    # Pretty print + save
    out: dict = {
        "wandb_run": cfg.wandb_run,
        "artifact": cfg.artifact,
        "n_samples": len(rows),
        "n_tokens": int(total_pos),
        "top_features_by_fire_rate": [
            {"feature": int(f), "fire_rate": float(fire_rate[f].item())}
            for f in chosen
        ],
        "feature_top_contexts": {},
    }
    for f in chosen:
        items = sorted(heaps[f], reverse=True)
        rows_out = []
        for v, sid, pos in items:
            toks = sample_token_strings[sid]
            lo = max(0, pos - cfg.ctx)
            hi = min(len(toks), pos + cfg.ctx + 1)
            ctx = (
                "".join(toks[lo:pos])
                + " ⟦" + toks[pos] + "⟧ "
                + "".join(toks[pos + 1 : hi])
            )
            rows_out.append({
                "activation": round(v, 3),
                "context": ctx.replace("\n", "\\n"),
            })
        out["feature_top_contexts"][str(f)] = rows_out

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False), flush=True)
    print(f"[summary] wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
