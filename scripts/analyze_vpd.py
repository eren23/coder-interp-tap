"""Post-hoc analysis of a trained VPD decomposition.

Downloads the W&B `vpd-final` artifact, loads the rank-1 components
and importance MLP, runs the base model over a streaming sample of
CommitPackFT, captures gates g(x_t) for every token, then reports:

  - Gate-sparsity histogram (alive components per token, where
    "alive" = g_c > THRESHOLD).
  - Per-component fire-rate: fraction of tokens where g_c > THRESHOLD.
  - Top-K firing token contexts per most-fired component.
  - Saves three PNGs to OUT_DIR.

Usage:
    python3 scripts/analyze_vpd.py \
        --wandb-run eren23/coder-interp-pilot/643sif07 \
        --artifact vpd-final:latest \
        --base-model Qwen/Qwen3-0.6B \
        --layer 13 --target mlp.up_proj \
        --num-samples 200 \
        --out-dir /tmp/vpd_analysis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


class VPDDecomposition(nn.Module):
    """Mirror of the trainer's class so we can load the state_dict."""

    def __init__(self, d_in: int, d_out: int, C: int, importance_hidden: int):
        super().__init__()
        self.U = nn.Parameter(torch.zeros(d_out, C))
        self.V = nn.Parameter(torch.zeros(d_in, C))
        self.importance = nn.Sequential(
            nn.Linear(d_in, importance_hidden),
            nn.GELU(),
            nn.Linear(importance_hidden, C),
        )

    def gates(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.importance(x))


def fetch_artifact(wandb_run: str, artifact: str, out_dir: Path) -> Path:
    import wandb
    api = wandb.Api()
    run = api.run(wandb_run)
    arts = [a for a in run.logged_artifacts() if a.name.startswith(artifact.split(":")[0])]
    if not arts:
        raise RuntimeError(f"no artifact matching {artifact} on run {wandb_run}")
    art = arts[-1]
    print(f"[wandb] downloading {art.name} ({art.size/1e6:.2f} MB)", flush=True)
    download_dir = out_dir / "vpd_ckpt"
    download_dir.mkdir(parents=True, exist_ok=True)
    art.download(root=str(download_dir))
    pt_files = list(download_dir.glob("*.pt"))
    if not pt_files:
        raise RuntimeError(f"no .pt file in artifact {art.name}")
    return pt_files[0]


def resolve_target(model: nn.Module, layer: int, path: str) -> nn.Linear:
    block = model.model.layers[layer]
    obj: nn.Module = block
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def collect_gates(
    cfg,
    model,
    tok,
    target: nn.Linear,
    vpd: VPDDecomposition,
    threshold: float,
) -> dict:
    """Run base model over samples, gather gates + token strings."""
    captured: dict = {}

    def hook(_m, inp, _o):
        captured["x"] = inp[0].detach()

    handle = target.register_forward_hook(hook)
    if cfg.local_jsonl:
        import json as _json
        def _gen():
            while True:
                with open(cfg.local_jsonl) as fh:
                    for line in fh:
                        yield _json.loads(line)
        ds_iter = _gen()
    else:
        ds = load_dataset(
            cfg.dataset, cfg.subset, streaming=True, split="train",
        )
        ds_iter = iter(ds.shuffle(seed=0, buffer_size=200))

    all_gates: list[torch.Tensor] = []
    all_tokens: list[list[str]] = []
    all_sample_id: list[int] = []
    all_pos: list[int] = []

    total = 0
    sample_id = 0
    while total < cfg.num_samples:
        batch_texts: list[str] = []
        for _ in range(cfg.batch):
            try:
                rec = next(ds_iter)
            except StopIteration:
                break
            t = rec.get("content") or rec.get("text") or rec.get("new_contents")
            if t:
                batch_texts.append(t)
        if not batch_texts:
            break

        device = next(model.parameters()).device
        enc = tok(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.seq_len,
        ).to(device)
        with torch.inference_mode():
            model(**enc, use_cache=False)
        x = captured["x"].float()  # (B, T, d_in)
        mask = enc.attention_mask.bool()  # (B, T)

        # Token strings per position (for context lookup later).
        for b in range(enc.input_ids.shape[0]):
            ids = enc.input_ids[b]
            valid = mask[b]
            ids_v = ids[valid].tolist()
            toks = [tok.decode([i]) for i in ids_v]
            xb = x[b][valid]              # (T_valid, d_in)
            with torch.inference_mode():
                gb = vpd.gates(xb)         # (T_valid, C)
            all_gates.append(gb.cpu())
            all_tokens.append(toks)
            all_sample_id.extend([sample_id] * len(toks))
            all_pos.extend(list(range(len(toks))))
            sample_id += 1
            total += len(toks)

        print(f"[capture] tokens so far: {total}/{cfg.num_samples}", flush=True)

    handle.remove()
    G = torch.cat(all_gates, dim=0)  # (N_tokens, C)
    print(f"[capture] G shape: {tuple(G.shape)}", flush=True)
    return {
        "gates": G,
        "tokens": all_tokens,        # list-of-list-of-str (per sample)
        "sample_id": all_sample_id,  # length N_tokens
        "pos": all_pos,              # length N_tokens
    }


def plot_and_report(out_dir: Path, G: torch.Tensor, info: dict, threshold: float) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N, C = G.shape

    # 1) Per-component fire rate.
    fire = (G > threshold).float().mean(dim=0).numpy()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(C), sorted(fire, reverse=True))
    ax.set_xlabel("component (sorted by fire rate)")
    ax.set_ylabel(f"fraction of tokens with g > {threshold}")
    ax.set_title(f"VPD per-component fire rate (N={N} tokens)")
    p1 = out_dir / "fire_rate.png"
    fig.tight_layout()
    fig.savefig(p1, dpi=150)
    plt.close(fig)

    # 2) Per-token alive count histogram.
    alive_per_tok = (G > threshold).sum(dim=1).numpy()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(alive_per_tok, bins=min(C, 50))
    ax.set_xlabel(f"components with g > {threshold} per token")
    ax.set_ylabel("token count")
    ax.set_title(f"VPD gate sparsity per token (N={N})")
    p2 = out_dir / "alive_per_token.png"
    fig.tight_layout()
    fig.savefig(p2, dpi=150)
    plt.close(fig)

    # 3) Mean activation per component (continuous; not thresholded).
    fig, ax = plt.subplots(figsize=(10, 4))
    g_mean_per_c = G.mean(dim=0).numpy()
    ax.bar(range(C), sorted(g_mean_per_c, reverse=True))
    ax.set_xlabel("component (sorted by mean g)")
    ax.set_ylabel("mean g(x)")
    ax.set_title("VPD per-component mean gate activation")
    p3 = out_dir / "mean_activation.png"
    fig.tight_layout()
    fig.savefig(p3, dpi=150)
    plt.close(fig)

    # 4) Top-K firing contexts per top-N most-active components.
    top_components = torch.tensor(fire).topk(min(8, C)).indices.tolist()
    top_contexts: dict[int, list[dict]] = {}
    K = 5
    CTX = 6
    for c in top_components:
        vals, idxs = G[:, c].topk(min(K, N))
        rows = []
        for v, i in zip(vals.tolist(), idxs.tolist()):
            sid = info["sample_id"][i]
            pos = info["pos"][i]
            toks = info["tokens"][sid]
            ctx_lo = max(0, pos - CTX)
            ctx_hi = min(len(toks), pos + CTX + 1)
            ctx = (
                "".join(toks[ctx_lo:pos])
                + " ⟦" + toks[pos] + "⟧ "
                + "".join(toks[pos + 1 : ctx_hi])
            )
            rows.append({"g": round(v, 3), "ctx": ctx.replace("\n", "\\n")})
        top_contexts[c] = rows

    summary = {
        "n_tokens": int(N),
        "n_components": int(C),
        "threshold": threshold,
        "global_mean_g": float(G.mean()),
        "global_pct_g_above_thresh": float(((G > threshold).float().mean()) * 100),
        "alive_per_token_mean": float(alive_per_tok.mean()),
        "alive_per_token_p5": float(sorted(alive_per_tok.tolist())[int(0.05 * N)]),
        "alive_per_token_p95": float(sorted(alive_per_tok.tolist())[int(0.95 * N)]),
        "top_components_by_fire_rate": top_components,
        "top_contexts": {str(k): v for k, v in top_contexts.items()},
        "plots": {"fire_rate": str(p1), "alive_per_token": str(p2), "mean_activation": str(p3)},
    }
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wandb-run", required=True)
    ap.add_argument("--artifact", default="vpd-final:latest")
    ap.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--layer", type=int, default=13)
    ap.add_argument("--target", default="mlp.up_proj")
    ap.add_argument("--num-samples", type=int, default=2000,
                    help="Number of tokens to gather")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--dataset", default="bigcode/commitpackft")
    ap.add_argument("--subset", default="python")
    ap.add_argument(
        "--local-jsonl", default=None,
        help="Path to a local JSONL of {content|text} rows to bypass HF datasets.",
    )
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out-dir", default="/tmp/vpd_analysis")
    cfg = ap.parse_args()

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = fetch_artifact(cfg.wandb_run, cfg.artifact, out_dir)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"[ckpt] loaded {ckpt_path.name}; cfg={state['config']}", flush=True)

    cfg_ckpt = state["config"]
    C = cfg_ckpt["num_components"]
    importance_hidden = cfg_ckpt["importance_hidden"]
    d_in = cfg_ckpt["d_in"]
    d_out = cfg_ckpt["d_out"]

    vpd = VPDDecomposition(d_in, d_out, C, importance_hidden)
    vpd.load_state_dict(state["state_dict"])
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32
    vpd = vpd.to(device)
    vpd.train(False)

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

    target = resolve_target(model, cfg.layer, cfg.target)
    print(f"[target] resolved layer={cfg.layer} path={cfg.target}", flush=True)

    info = collect_gates(cfg, model, tok, target, vpd, cfg.threshold)
    summary = plot_and_report(out_dir, info["gates"], info, cfg.threshold)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[summary] wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
