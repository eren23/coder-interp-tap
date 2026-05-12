"""Pod-side VPD-v2 interpretability visualizations.

Produces four artifacts inside one W&B run:

  (A) concept_cards:           wandb.Table  matrix x component x top firing contexts
  (B) logit_lens:              wandb.Table  matrix x component x top vocab tokens
                               (only for matrices that write to the residual
                               stream: mlp.down_proj and self_attn.o_proj)
  (C) per_matrix_sparsity:     bar plot per matrix of sorted mean-gate values
  (D) coactivation_heatmap:    Jaccard similarity matrix between adjacent
                               matrices' alive components

Inputs:
  * SOURCE_WANDB_RUN  - the run whose `vpd-v2-final` artifact we analyze
  * BASE_MODEL        - Qwen3-0.6B
  * DATASET_ID        - eren23/eren-code-style by default
  * Tunables: NUM_SAMPLES, SEQ_LEN, GATE_THRESHOLD, TOP_K_CONTEXTS, TOP_K_VOCAB.
"""

from __future__ import annotations

import heapq
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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


# Short code snippets the demo heatmaps visualize gates over.
DEMO_PROMPTS = {
    "py_import": "from dataclasses import dataclass\nimport asyncio\n",
    "py_docstring": '"""\nConfiguration management for Hermes Agent.\n"""\n',
    "ts_export": 'export { Button, type ButtonProps } from "./Button";\n',
    "ts_import_destruct": 'import { describe, it, expect } from "vitest";\n',
    "rust_impl": "impl Drop for NoGradGuard {\n    fn drop(&mut self) {\n",
    "py_django_model": "class Profile(models.Model):\n    user = models.OneToOneField(\n",
}


def run_llm_autointerp(cfg, alive_index, G, sample_tokens, pos_records,
                        max_components_per_matrix: int = 32):
    """For each alive component, send its top firing contexts to DeepSeek-V3
    via OpenRouter and ask for a one-sentence concept label. Returns
    {(matrix_key, comp_idx): label_str}.

    Skipped silently if OPENROUTER_API_KEY is not set.
    """
    import json
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[E] no OPENROUTER_API_KEY / OPENAI_API_KEY — skipping LLM autointerp",
              flush=True)
        return {}
    api_base = os.environ.get("LLM_API_BASE", "https://openrouter.ai/api/v1")
    model_name = os.environ.get("LLM_MODEL", "deepseek/deepseek-chat")

    import asyncio, httpx
    semaphore = asyncio.Semaphore(int(os.environ.get("LLM_CONCURRENCY", "6")))

    async def call_one(client: httpx.AsyncClient, prompt: str) -> str:
        async with semaphore:
            try:
                r = await client.post(
                    f"{api_base}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": model_name,
                        "messages": [
                            {"role": "system",
                             "content": "You name what a single neural-net subcomponent detects."
                                        " Reply with ONE short sentence (<15 words) describing"
                                        " the concept all the highlighted tokens share."},
                            {"role": "user", "content": prompt},
                        ],
                        "max_tokens": 60,
                        "temperature": 0.0,
                    },
                    timeout=60.0,
                )
                r.raise_for_status()
                data = r.json()
                return data["choices"][0]["message"]["content"].strip()
            except Exception as exc:
                return f"[ERR: {type(exc).__name__}]"

    # Build prompts for each (matrix, component) — cap components per matrix.
    tasks_meta: list[tuple[str, int]] = []
    prompts: list[str] = []
    for key, comps in alive_index.items():
        comps_use = comps[:max_components_per_matrix]
        for c in comps_use:
            col = G[key][:, c]
            top_vals, top_idx = col.topk(min(cfg.top_k_contexts, col.shape[0]))
            ctx_lines = []
            for v, i in zip(top_vals.tolist(), top_idx.tolist()):
                if v <= 0:
                    continue
                sid_, pos_ = pos_records[i]
                toks = sample_tokens[sid_]
                lo = max(0, pos_ - cfg.ctx_window)
                hi = min(len(toks), pos_ + cfg.ctx_window + 1)
                ctx = (
                    "".join(toks[lo:pos_])
                    + " ⟦" + toks[pos_] + "⟧ "
                    + "".join(toks[pos_+1:hi])
                ).replace("\n", "↵")
                ctx_lines.append(f"  - act={v:.3f}: {ctx}")
            if not ctx_lines:
                continue
            user_msg = (
                f"Subcomponent fires hardest on these token contexts "
                f"(the firing token is wrapped in ⟦ ⟧):\n"
                + "\n".join(ctx_lines)
                + "\n\nWhat single concept do the highlighted tokens share?"
            )
            tasks_meta.append((key, int(c)))
            prompts.append(user_msg)

    if not prompts:
        return {}

    async def run_all():
        async with httpx.AsyncClient() as client:
            return await asyncio.gather(*(call_one(client, p) for p in prompts))
    labels_list = asyncio.run(run_all())
    return {meta: lab for meta, lab in zip(tasks_meta, labels_list)}


# ---------------------------------------------------------------------------
# Mirror minimal pieces of the trainer's modules (just enough to load
# state_dict). We do NOT need MaskedLinear/forward-hook plumbing here; we
# only need to recompute g(x) given saved Gamma weights.

class GammaPerMatrix(nn.Module):
    def __init__(self, d_in: int, num_components: int, hidden: int):
        super().__init__()
        self.norm = nn.RMSNorm(d_in)
        self.fc1 = nn.Linear(d_in, hidden)
        self.fc2 = nn.Linear(hidden, num_components)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x.float())
        h = F.gelu(self.fc1(h))
        return self.fc2(h)


def upper_leaky_sigmoid(z: torch.Tensor, alpha: float = 0.01) -> torch.Tensor:
    return torch.clamp(z, 0.0, 1.0) + alpha * F.relu(z - 1.0)


# ---------------------------------------------------------------------------
# Config

@dataclass
class Cfg:
    source_wandb_run: str
    artifact_name: str
    base_model: str
    dataset_id: str
    dataset_split: str
    num_samples: int
    seq_len: int
    gate_threshold: float
    alive_threshold: float
    top_k_contexts: int
    top_k_vocab: int
    ctx_window: int
    workspace: Path
    run_name: str
    wandb_project: str
    wandb_entity: Optional[str]


def load_cfg() -> Cfg:
    return Cfg(
        source_wandb_run=_env("SOURCE_WANDB_RUN", "eren23/coder-interp-pilot/2xehk87f"),
        artifact_name=_env("ARTIFACT_NAME", "vpd-v2-final"),
        base_model=_env("BASE_MODEL", "Qwen/Qwen3-0.6B"),
        dataset_id=_env("DATASET_ID", "eren23/eren-code-style"),
        dataset_split=_env("DATASET_SPLIT", "train"),
        num_samples=int(_env("NUM_SAMPLES", "120")),
        seq_len=int(_env("SEQ_LEN", "128")),
        gate_threshold=float(_env("GATE_THRESHOLD", "0.1")),
        # Trainer uses 1e-6 to flag alive components. Match it; the previous
        # 0.01 default was 10000x too strict and zeroed out every alive list.
        alive_threshold=float(_env("ALIVE_THRESHOLD", "1e-6")),
        top_k_contexts=int(_env("TOP_K_CONTEXTS", "6")),
        top_k_vocab=int(_env("TOP_K_VOCAB", "20")),
        ctx_window=int(_env("CTX_WINDOW", "8")),
        workspace=Path(_env("WORKSPACE_DIR", "/workspace/project")),
        run_name=_env("WANDB_RUN_NAME", f"vpd-v2-analysis-{int(time.time())}"),
        wandb_project=_env("WANDB_PROJECT", "coder-interp-pilot"),
        wandb_entity=os.environ.get("WANDB_ENTITY"),
    )


# ---------------------------------------------------------------------------
# Pod-side path helpers

def _resolve(module: nn.Module, dotted: str) -> nn.Module:
    obj: nn.Module = module
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def fetch_artifact(source_run: str, artifact_name: str, out_dir: Path) -> Path:
    api = wandb.Api()
    run = api.run(source_run)
    arts = [a for a in run.logged_artifacts() if a.name.startswith(artifact_name)]
    if not arts:
        raise RuntimeError(f"no artifact starting with {artifact_name}")
    art = arts[-1]
    print(f"[wandb] download {art.name} ({art.size/1e6:.2f} MB)", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    art.download(root=str(out_dir))
    return next(out_dir.glob("*.pt"))


# ---------------------------------------------------------------------------
# Main

def main() -> int:
    cfg = load_cfg()
    print(f"[main] cfg: {cfg}", flush=True)

    ckpt = fetch_artifact(cfg.source_wandb_run, cfg.artifact_name,
                          cfg.workspace / "vpd_v2_ckpt")
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    art_cfg = state["config"]
    layer = art_cfg["layer"]
    targets = art_cfg["target_modules"]
    C = art_cfg["num_components"]
    gamma_hidden = art_cfg["gamma_hidden"]
    print(
        f"[ckpt] layer={layer} targets={targets} C={C} gamma_hidden={gamma_hidden}",
        flush=True,
    )

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

    block = model.model.layers[layer]

    # Build per-matrix Gamma + load weights, and stash U, V tensors for later.
    gammas: dict[str, GammaPerMatrix] = {}
    Us: dict[str, torch.Tensor] = {}
    Vs: dict[str, torch.Tensor] = {}
    Ds: dict[str, torch.Tensor] = {}
    matrix_dims: dict[str, tuple[int, int]] = {}
    for target in targets:
        key = target.replace(".", "_")
        orig: nn.Linear = _resolve(block, target)  # type: ignore[assignment]
        d_in = orig.in_features
        d_out = orig.out_features
        matrix_dims[key] = (d_out, d_in)
        g = GammaPerMatrix(d_in, C, gamma_hidden).cuda()
        g.load_state_dict(state["gammas"][key])
        g.train(False)
        gammas[key] = g
        Us[key] = state["masked"][key]["U"].cuda()
        Vs[key] = state["masked"][key]["V"].cuda()
        Ds[key] = state["masked"][key]["Delta"].cuda()

    # W&B init
    wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=cfg.run_name,
        config={
            "source_wandb_run": cfg.source_wandb_run,
            "artifact_name": cfg.artifact_name,
            "base_model": cfg.base_model,
            "layer": layer,
            "targets": targets,
            "num_components": C,
            "num_samples": cfg.num_samples,
            "seq_len": cfg.seq_len,
            "gate_threshold": cfg.gate_threshold,
        },
    )

    # ---------- Gather data: stream from HF, capture pre-weight x per matrix ----------
    print(f"[ds] loading {cfg.dataset_id}", flush=True)
    ds = load_dataset(cfg.dataset_id, split=cfg.dataset_split, streaming=True)
    ds_iter = iter(ds.shuffle(seed=42, buffer_size=200))

    captured: dict[str, torch.Tensor] = {}
    handles = []
    for target in targets:
        key = target.replace(".", "_")
        mod = _resolve(block, target)

        def _make_hook(k: str):
            def _hook(_m, inp):
                captured[k] = inp[0].detach()
            return _hook

        h = mod.register_forward_pre_hook(_make_hook(key))
        handles.append(h)

    sample_tokens: list[list[str]] = []   # per-sample token strings
    sample_meta: list[tuple[str, str]] = []  # (repo, path) per sample
    # Per-key gate matrices: list of (N_per_sample, C) tensors -> later cat
    gates_chunks: dict[str, list[torch.Tensor]] = {k: [] for k in gammas}
    # Per-key (sample_id, position) record  parallel to flattened gates
    pos_records: list[tuple[int, int]] = []
    # We compute gates for ALL matrices on the same set of tokens to keep things
    # aligned across matrices.

    n_total = 0
    sid = 0
    while sid < cfg.num_samples:
        try:
            rec = next(ds_iter)
        except StopIteration:
            break
        text = rec.get("content") or rec.get("text") or ""
        if not text:
            continue
        enc = tok(text, return_tensors="pt", truncation=True, max_length=cfg.seq_len).to("cuda")
        ids = enc.input_ids[0].tolist()
        toks = [tok.decode([i]) for i in ids]
        sample_tokens.append(toks)
        sample_meta.append((rec.get("repo", ""), rec.get("path", "")))
        with torch.inference_mode():
            model(**enc, use_cache=False)
        for key, gamma in gammas.items():
            x = captured[key][0]  # (T, d_in)
            with torch.inference_mode():
                z = gamma(x)
                g = upper_leaky_sigmoid(z, 0.01)  # (T, C)
            gates_chunks[key].append(g.cpu())
        T = enc.input_ids.shape[1]
        for pos in range(T):
            pos_records.append((sid, pos))
        n_total += T
        sid += 1
        if sid % 20 == 0:
            print(f"[capture] {sid}/{cfg.num_samples} samples, {n_total} tokens", flush=True)

    for h in handles:
        h.remove()

    if sid == 0:
        raise RuntimeError("no samples collected")
    print(f"[capture] done: {sid} samples, {n_total} total tokens", flush=True)

    # Concatenated gate matrices, shape (n_total, C) per key.
    G: dict[str, torch.Tensor] = {k: torch.cat(gates_chunks[k], dim=0) for k in gates_chunks}

    # ---------- (C) Per-matrix sparsity bar plots ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_paths: dict[str, str] = {}
    for key, Gk in G.items():
        mean_g = Gk.mean(dim=0).numpy()
        sorted_g = sorted(mean_g, reverse=True)
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.bar(range(C), sorted_g)
        alive = int(((mean_g > cfg.alive_threshold)).sum())
        ax.set_xlabel(f"component (sorted by mean gate)  —  {alive}/{C} alive")
        ax.set_ylabel("mean g(x)")
        ax.set_title(f"{key}  per-component mean gate  (n_tokens={n_total})")
        p = cfg.workspace / f"sparsity_{key}.png"
        fig.tight_layout()
        fig.savefig(p, dpi=120)
        plt.close(fig)
        plot_paths[key] = str(p)
        wandb.log({f"sparsity/{key}": wandb.Image(str(p))})

    # ---------- (A) Concept cards: top-firing contexts per alive component ----------
    concept_table = wandb.Table(
        columns=["matrix", "component", "rank", "activation",
                 "mean_gate", "repo", "path", "context"]
    )
    alive_index: dict[str, list[int]] = {}
    for key, Gk in G.items():
        mean_g = Gk.mean(dim=0)
        alive = (mean_g > cfg.alive_threshold).nonzero(as_tuple=True)[0].tolist()
        alive_index[key] = alive
        print(f"[alive] {key}: {len(alive)} components", flush=True)
        for c in alive:
            col = Gk[:, c]
            top_vals, top_idx = col.topk(min(cfg.top_k_contexts, col.shape[0]))
            for rank, (v, i) in enumerate(zip(top_vals.tolist(), top_idx.tolist())):
                sid_, pos_ = pos_records[i]
                toks = sample_tokens[sid_]
                lo = max(0, pos_ - cfg.ctx_window)
                hi = min(len(toks), pos_ + cfg.ctx_window + 1)
                ctx = (
                    "".join(toks[lo:pos_])
                    + " ⟦" + toks[pos_] + "⟧ "
                    + "".join(toks[pos_ + 1 : hi])
                ).replace("\n", "\\n")
                repo, path = sample_meta[sid_]
                concept_table.add_data(
                    key, int(c), rank, float(v),
                    float(mean_g[c].item()), repo, path, ctx,
                )
    wandb.log({"concept_cards": concept_table})
    print(f"[A] concept_cards table: {concept_table.columns}", flush=True)

    # ---------- (B) Logit-lens for matrices that write to residual stream ----------
    # For Qwen3 architecture: mlp.down_proj and self_attn.o_proj output is added
    # back to residual. Decode their U[:, c] direction through the LM head.
    lm_head_weight = model.lm_head.weight  # (vocab, d_model)
    vocab_size = lm_head_weight.shape[0]
    logit_lens_table = wandb.Table(
        columns=["matrix", "component", "mean_gate", "top_tokens", "top_logits"]
    )
    write_residual = {"mlp_down_proj", "self_attn_o_proj"}
    for key in [k for k in G.keys() if k in write_residual]:
        U = Us[key]            # (d_out, C); for these matrices d_out == d_model
        for c in alive_index[key]:
            direction = U[:, c].float()  # (d_model,)
            logits = lm_head_weight.float() @ direction  # (vocab,)
            top_v, top_i = logits.topk(cfg.top_k_vocab)
            tokens_str = [tok.decode([i]) for i in top_i.tolist()]
            logit_lens_table.add_data(
                key, int(c), float(G[key].mean(dim=0)[c].item()),
                " | ".join(repr(t) for t in tokens_str),
                ",".join(f"{v:.3f}" for v in top_v.tolist()),
            )
    wandb.log({"logit_lens": logit_lens_table})
    print(f"[B] logit_lens table: {logit_lens_table.columns}", flush=True)

    # ---------- (D) Cross-matrix coactivation heatmap ----------
    # Compute pairwise Jaccard between matrices on per-token >threshold gates.
    # Visualize as a 7x7 heatmap of mean Jaccard across alive component pairs.
    keys_list = list(G.keys())
    # Build a (n_total, C_total) bool matrix of "above-threshold" gates per matrix
    fire_above: dict[str, torch.Tensor] = {
        k: (G[k] > cfg.gate_threshold) for k in keys_list
    }
    # For each ordered pair (i, j), compute mean over alive_i, alive_j of
    # |fire_i,c & fire_j,c'| / |fire_i,c | fire_j,c'|.
    L = len(keys_list)
    M = torch.zeros(L, L)
    for i, ki in enumerate(keys_list):
        ai = alive_index[ki][: min(24, len(alive_index[ki]))]
        if not ai:
            continue
        fi = fire_above[ki][:, ai]            # (N, |ai|)
        for j, kj in enumerate(keys_list):
            aj = alive_index[kj][: min(24, len(alive_index[kj]))]
            if not aj:
                continue
            fj = fire_above[kj][:, aj]        # (N, |aj|)
            # Intersection: |ai| x |aj| matrix of per-pair AND counts
            inter = (fi.float().T @ fj.float())  # (|ai|, |aj|)
            ci = fi.float().sum(dim=0).unsqueeze(1)  # (|ai|, 1)
            cj = fj.float().sum(dim=0).unsqueeze(0)  # (1, |aj|)
            union = ci + cj - inter + 1e-6
            jacc = inter / union
            M[i, j] = jacc.mean().item()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(M.numpy(), vmin=0, vmax=float(M.max().item() + 1e-6), cmap="viridis")
    ax.set_xticks(range(L))
    ax.set_yticks(range(L))
    ax.set_xticklabels(keys_list, rotation=45, ha="right")
    ax.set_yticklabels(keys_list)
    ax.set_title("Cross-matrix coactivation (mean Jaccard of alive comps)")
    for ii in range(L):
        for jj in range(L):
            ax.text(
                jj, ii, f"{M[ii, jj].item():.2f}",
                ha="center", va="center", fontsize=7,
                color=("white" if M[ii, jj].item() < 0.5 * M.max().item() else "black"),
            )
    fig.colorbar(im, ax=ax)
    p_co = cfg.workspace / "coactivation.png"
    fig.tight_layout()
    fig.savefig(p_co, dpi=140)
    plt.close(fig)
    wandb.log({"coactivation/heatmap": wandb.Image(str(p_co))})
    print(f"[D] coactivation heatmap saved", flush=True)

    # ---------- (E) LLM auto-interp: concept label per alive component ----------
    if os.environ.get("ENABLE_LLM_AUTOINTERP", "1") == "1":
        try:
            print("[E] LLM autointerp starting", flush=True)
            llm_labels = run_llm_autointerp(cfg, alive_index, G, sample_tokens, pos_records)
            label_table = wandb.Table(
                columns=["matrix", "component", "mean_gate", "n_contexts",
                         "concept_label", "top_context"]
            )
            for key, comps in alive_index.items():
                for c in comps:
                    label = llm_labels.get((key, c), "")
                    # quick top context preview
                    col = G[key][:, c]
                    if col.numel() == 0:
                        continue
                    _, i = col.topk(1)
                    sid_, pos_ = pos_records[i.item()]
                    toks = sample_tokens[sid_]
                    lo = max(0, pos_ - 6)
                    hi = min(len(toks), pos_ + 7)
                    top_ctx = (
                        "".join(toks[lo:pos_]) + " ⟦" + toks[pos_] + "⟧ "
                        + "".join(toks[pos_+1:hi])
                    ).replace("\n", "\\n")
                    label_table.add_data(
                        key, int(c), float(G[key].mean(dim=0)[c].item()),
                        cfg.top_k_contexts, label, top_ctx,
                    )
            wandb.log({"concept_labels": label_table})
            print(f"[E] concept_labels logged: {len(llm_labels)} labels", flush=True)
        except Exception as exc:
            print(f"[E] LLM autointerp FAILED: {exc}", flush=True)

    # ---------- (F) Demo prompt heatmaps ----------
    if os.environ.get("ENABLE_DEMO_HEATMAPS", "1") == "1":
        try:
            print("[F] rendering demo prompt heatmaps", flush=True)
            handles2 = []
            captured2: dict[str, torch.Tensor] = {}
            for target in targets:
                key = target.replace(".", "_")
                mod = _resolve(block, target)

                def _make_hook(k: str):
                    def _hook(_m, inp):
                        captured2[k] = inp[0].detach()
                    return _hook
                handles2.append(mod.register_forward_pre_hook(_make_hook(key)))
            for prompt_name, prompt_text in DEMO_PROMPTS.items():
                enc = tok(prompt_text, return_tensors="pt", truncation=True,
                          max_length=64).to("cuda")
                ids = enc.input_ids[0].tolist()
                toks_p = [tok.decode([i]) for i in ids]
                with torch.inference_mode():
                    model(**enc, use_cache=False)
                # Build a (alive_total, T) gate matrix stacked across matrices
                stacked: list[tuple[str, int, torch.Tensor]] = []
                for key in alive_index:
                    if not alive_index[key]:
                        continue
                    x_p = captured2[key][0]
                    with torch.inference_mode():
                        z_p = gammas[key](x_p)
                        g_p = upper_leaky_sigmoid(z_p, 0.01)  # (T, C)
                    for c in alive_index[key]:
                        stacked.append((key, c, g_p[:, c].cpu()))
                if not stacked:
                    continue
                heat = torch.stack([t for _, _, t in stacked], dim=0).numpy()
                row_labels = [f"{k}#{c}" for k, c, _ in stacked]
                fig, ax = plt.subplots(figsize=(max(8, len(toks_p)*0.55),
                                                max(5, len(stacked)*0.12)))
                im = ax.imshow(heat, aspect="auto", cmap="viridis",
                                vmin=0, vmax=max(heat.max(), 1e-6))
                ax.set_xticks(range(len(toks_p)))
                ax.set_xticklabels(toks_p, rotation=45, ha="right", fontsize=7)
                ax.set_yticks(range(len(row_labels)))
                ax.set_yticklabels(row_labels, fontsize=6)
                ax.set_title(f"Per-token gates  —  prompt: {prompt_name}")
                fig.colorbar(im, ax=ax)
                p_demo = cfg.workspace / f"demo_{prompt_name}.png"
                fig.tight_layout()
                fig.savefig(p_demo, dpi=130)
                plt.close(fig)
                wandb.log({f"demo/{prompt_name}": wandb.Image(str(p_demo))})
            for h in handles2:
                h.remove()
            print("[F] demo heatmaps logged", flush=True)
        except Exception as exc:
            print(f"[F] demo heatmaps FAILED: {exc}", flush=True)

    # ---------- Summary JSON ----------
    summary = {
        "source_wandb_run": cfg.source_wandb_run,
        "n_samples": sid,
        "n_tokens": n_total,
        "alive_counts": {k: len(v) for k, v in alive_index.items()},
        "alive_indexes": {k: v for k, v in alive_index.items()},
        "config": art_cfg,
    }
    summary_path = cfg.workspace / "vpd_v2_analysis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    art = wandb.Artifact("vpd-v2-analysis", type="analysis")
    art.add_file(str(summary_path))
    art.add_file(str(p_co))
    for p in plot_paths.values():
        art.add_file(p)
    wandb.log_artifact(art)

    wandb.finish()
    print("[main] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
