"""Feature explainer — auto-interpret SAE features via an LLM.

For each SAE feature, collect its top-K firing contexts from a streaming
corpus, then ask an OpenAI-compatible LLM (default: DeepSeek V3 via
OpenRouter) to describe the concept the feature represents. Output goes
to a JSON file + a W&B Table.

Hot-paths:
  - Forward Qwen2.5-Coder-1.5B in bf16, capture residual at SAE_LAYER.
  - SAE encode (TopK), keep features that fire above threshold.
  - For each (token_idx, feature_idx, strength) above threshold, push
    onto the per-feature heap of top firing contexts.
  - At the end, batch the LLM calls with asyncio + httpx.
"""

from __future__ import annotations

import asyncio
import heapq
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx
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
    sae_layer: int
    d_model: int
    d_sae: int
    topk: int
    sae_local_path: str
    sae_wandb_artifact: Optional[str]
    context_window: int
    top_contexts_per_feature: int
    min_contexts: int
    dataset_id: str
    dataset_subset: str
    seq_len: int
    capture_batch_size: int
    num_features_to_explain: int
    num_tokens_to_stream: int
    llm_api_base: str
    llm_model: str
    llm_api_key: str
    llm_concurrency: int
    llm_max_retries: int
    llm_max_tokens: int
    llm_temperature: float
    workspace: Path
    run_name: str
    wandb_project: str
    wandb_entity: Optional[str]


def load_cfg() -> Cfg:
    api_key = (
        os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("LLM_API_KEY")
        or ""
    )
    if not api_key:
        raise RuntimeError(
            "no LLM key — set OPENROUTER_API_KEY (or LLM_API_KEY) in env or "
            ".env.runpod.local"
        )
    return Cfg(
        base_model=_env("BASE_MODEL"),
        sae_layer=int(_env("SAE_LAYER")),
        d_model=int(_env("D_MODEL")),
        d_sae=int(_env("D_SAE")),
        topk=int(_env("TOPK", "50")),
        sae_local_path=_env("SAE_LOCAL_PATH", "/workspace/project/sae_final.pt"),
        sae_wandb_artifact=os.environ.get("SAE_WANDB_ARTIFACT"),
        context_window=int(_env("CONTEXT_WINDOW", "20")),
        top_contexts_per_feature=int(_env("TOP_CONTEXTS_PER_FEATURE", "10")),
        min_contexts=int(_env("MIN_CONTEXTS", "3")),
        dataset_id=_env("DATASET_ID", "bigcode/commitpackft"),
        dataset_subset=_env("DATASET_SUBSET", "python"),
        seq_len=int(_env("SEQ_LEN", "512")),
        capture_batch_size=int(_env("CAPTURE_BATCH_SIZE", "8")),
        num_features_to_explain=int(_env("NUM_FEATURES_TO_EXPLAIN", "50")),
        num_tokens_to_stream=int(_env("NUM_TOKENS_TO_STREAM", "100000")),
        llm_api_base=_env("LLM_API_BASE", "https://openrouter.ai/api/v1"),
        llm_model=_env("LLM_MODEL", "deepseek/deepseek-chat"),
        llm_api_key=api_key,
        llm_concurrency=int(_env("LLM_CONCURRENCY", "8")),
        llm_max_retries=int(_env("LLM_MAX_RETRIES", "3")),
        llm_max_tokens=int(_env("LLM_MAX_TOKENS", "80")),
        llm_temperature=float(_env("LLM_TEMPERATURE", "0.3")),
        workspace=Path("/workspace/project"),
        run_name=_env("WANDB_RUN_NAME", f"feature-explainer-{int(time.time())}"),
        wandb_project=_env("WANDB_PROJECT", "coder-interp-pilot"),
        wandb_entity=os.environ.get("WANDB_ENTITY"),
    )


# ---------------------------------------------------------------------------
# SAE
# ---------------------------------------------------------------------------


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
        return sparse, topk_val, topk_idx


def load_sae(cfg: Cfg) -> TopKSAE:
    local = Path(cfg.sae_local_path)
    if not local.exists() and cfg.sae_wandb_artifact:
        print(f"[sae] local not found, fetching W&B artifact {cfg.sae_wandb_artifact}", flush=True)
        api = wandb.Api()
        artifact = api.artifact(
            f"{cfg.wandb_entity}/{cfg.wandb_project}/{cfg.sae_wandb_artifact}"
        )
        download_dir = Path(artifact.download(root=str(cfg.workspace / "sae_artifact")))
        # Pick the .pt file.
        pts = list(download_dir.glob("*.pt"))
        if not pts:
            raise RuntimeError(f"no .pt in artifact {download_dir}")
        local = pts[0]
        print(f"[sae] downloaded to {local}", flush=True)
    if not local.exists():
        raise RuntimeError(f"SAE checkpoint not found at {local}")
    state = torch.load(local, map_location="cuda", weights_only=False)
    sae = TopKSAE(cfg.d_model, cfg.d_sae, cfg.topk).cuda()
    sae.load_state_dict(state["state_dict"])
    print(
        f"[sae] loaded step={state.get('step')} from {local}",
        flush=True,
    )
    return sae


# ---------------------------------------------------------------------------
# Top-context collection
# ---------------------------------------------------------------------------


def collect_top_contexts(
    cfg: Cfg, sae: TopKSAE, base_model, tok, dataset_iter
) -> dict[int, list[tuple[float, str]]]:
    """For each feature, return up to top_contexts_per_feature (strength,
    context_text) tuples ranked by firing strength."""
    # Heap per feature: min-heap of (strength, context).
    heaps: dict[int, list[tuple[float, str]]] = {}
    tokens_seen = 0
    last_print_t = time.time()

    while tokens_seen < cfg.num_tokens_to_stream:
        batch_texts: list[str] = []
        for _ in range(cfg.capture_batch_size):
            try:
                sample = next(dataset_iter)
            except StopIteration:
                return heaps
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
            out = base_model(**enc, output_hidden_states=True)
        hs = out.hidden_states[cfg.sae_layer]  # (B, T, d)

        B, T, _ = hs.shape
        attn = enc.attention_mask.bool()
        ids = enc.input_ids

        # Encode every position in batch (flatten for efficiency).
        flat = hs[attn].reshape(-1, cfg.d_model).float()
        if flat.shape[0] == 0:
            continue
        with torch.inference_mode():
            _, topk_val, topk_idx = sae.encode(flat)  # (N_flat, k)

        # Map flat-index back to (b, t).
        b_arr, t_arr = torch.where(attn)  # both (N_flat,)
        b_arr = b_arr.cpu().tolist()
        t_arr = t_arr.cpu().tolist()
        ids_cpu = ids.cpu()

        topk_val_cpu = topk_val.cpu()
        topk_idx_cpu = topk_idx.cpu()

        for n in range(flat.shape[0]):
            b = b_arr[n]
            t = t_arr[n]
            seq = ids_cpu[b]
            lo = max(0, t - cfg.context_window)
            hi = min(int(attn[b].sum().item()), t + cfg.context_window + 1)
            window = seq[lo:hi].tolist()
            firing_token = seq[t].item()
            try:
                window_text = tok.decode(window, skip_special_tokens=True)
            except Exception:
                window_text = ""
            try:
                fire_text = tok.decode([firing_token], skip_special_tokens=True)
            except Exception:
                fire_text = ""
            # Mark the firing token in-context with ⟦…⟧.
            decoded_pre = tok.decode(seq[lo:t].tolist(), skip_special_tokens=True)
            decoded_post = tok.decode(seq[t + 1 : hi].tolist(), skip_special_tokens=True)
            marked = f"{decoded_pre}⟦{fire_text}⟧{decoded_post}"

            for k in range(cfg.topk):
                feat_idx = int(topk_idx_cpu[n, k].item())
                strength = float(topk_val_cpu[n, k].item())
                heap = heaps.setdefault(feat_idx, [])
                if len(heap) < cfg.top_contexts_per_feature:
                    heapq.heappush(heap, (strength, marked))
                elif strength > heap[0][0]:
                    heapq.heapreplace(heap, (strength, marked))

        tokens_seen += int(attn.sum().item())
        if time.time() - last_print_t > 30:
            n_features = len(heaps)
            n_features_with_min = sum(
                1 for h in heaps.values() if len(h) >= cfg.min_contexts
            )
            print(
                f"[collect] tokens={tokens_seen}/{cfg.num_tokens_to_stream} "
                f"features_seen={n_features} ready={n_features_with_min}",
                flush=True,
            )
            last_print_t = time.time()

    return heaps


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------


def build_prompt(contexts: list[tuple[float, str]]) -> str:
    sorted_ctx = sorted(contexts, key=lambda x: -x[0])
    lines = ["You are an interpretability researcher analyzing a neural-network feature."]
    lines.append("")
    lines.append(
        "The following are the contexts where this feature fires most strongly. "
        "The firing token in each context is marked with ⟦TOKEN⟧."
    )
    lines.append("")
    for i, (strength, text) in enumerate(sorted_ctx, 1):
        snippet = text.replace("\n", " ")[:400]
        lines.append(f"Context {i} (strength={strength:.2f}): {snippet}")
    lines.append("")
    lines.append(
        "Describe in 15–25 words what concept, code pattern, or syntactic role this "
        "feature represents. Be specific. Don't hedge. Output ONLY the description."
    )
    return "\n".join(lines)


async def call_llm(
    cfg: Cfg,
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    feature_idx: int,
    contexts: list[tuple[float, str]],
) -> tuple[int, str]:
    prompt = build_prompt(contexts)
    payload = {
        "model": cfg.llm_model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": cfg.llm_max_tokens,
        "temperature": cfg.llm_temperature,
    }
    headers = {
        "Authorization": f"Bearer {cfg.llm_api_key}",
        "HTTP-Referer": "https://github.com/eren23/coder-interp-tap",
        "X-Title": "coder-interp-tap feature explainer",
    }
    last_err = None
    for attempt in range(cfg.llm_max_retries):
        try:
            async with sem:
                resp = await client.post(
                    f"{cfg.llm_api_base}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=60.0,
                )
            if resp.status_code != 200:
                last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
                if resp.status_code == 429:
                    await asyncio.sleep(2 ** attempt)
                    continue
                if 500 <= resp.status_code < 600:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return feature_idx, f"<error: {last_err}>"
            data = resp.json()
            text = data["choices"][0]["message"]["content"].strip()
            return feature_idx, text
        except Exception as e:  # noqa
            last_err = repr(e)
            await asyncio.sleep(2 ** attempt)
    return feature_idx, f"<error: {last_err}>"


async def explain_features(
    cfg: Cfg, heaps: dict[int, list[tuple[float, str]]]
) -> dict[int, str]:
    candidates = [
        (idx, ctx)
        for idx, ctx in heaps.items()
        if len(ctx) >= cfg.min_contexts
    ]
    candidates.sort(key=lambda p: -sum(s for s, _ in p[1]))
    candidates = candidates[: cfg.num_features_to_explain]
    print(
        f"[llm] explaining {len(candidates)} features via {cfg.llm_model} @ {cfg.llm_api_base}",
        flush=True,
    )

    sem = asyncio.Semaphore(cfg.llm_concurrency)
    descriptions: dict[int, str] = {}
    completed = 0
    last_print_t = time.time()
    async with httpx.AsyncClient(http2=True, timeout=60) as client:
        tasks = [
            asyncio.create_task(call_llm(cfg, client, sem, idx, ctx))
            for idx, ctx in candidates
        ]
        for task in asyncio.as_completed(tasks):
            feat_idx, desc = await task
            descriptions[feat_idx] = desc
            completed += 1
            if time.time() - last_print_t > 30 or completed == len(candidates):
                preview = (desc[:80] + "…") if len(desc) > 80 else desc
                print(
                    f"[llm] {completed}/{len(candidates)} feat={feat_idx}: {preview}",
                    flush=True,
                )
                last_print_t = time.time()
    return descriptions


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    cfg = load_cfg()
    print(f"[main] cfg: model={cfg.llm_model} api_base={cfg.llm_api_base} features={cfg.num_features_to_explain}", flush=True)

    print(f"[base] loading {cfg.base_model} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )

    sae = load_sae(cfg)

    print(f"[ds] streaming {cfg.dataset_id}:{cfg.dataset_subset}", flush=True)
    ds = load_dataset(
        cfg.dataset_id,
        cfg.dataset_subset,
        streaming=True,
        split="train",
        trust_remote_code=True,
    )
    ds_iter = iter(ds.shuffle(seed=42, buffer_size=1000))

    heaps = collect_top_contexts(cfg, sae, base, tok, ds_iter)
    print(f"[collect] done; {len(heaps)} features had at least 1 firing", flush=True)

    # Free base before LLM calls.
    del base, sae
    import gc as _gc
    _gc.collect()
    torch.cuda.empty_cache()

    descriptions = asyncio.run(explain_features(cfg, heaps))

    out_path = cfg.workspace / "feature_descriptions.json"
    serializable = {
        str(k): {
            "description": descriptions.get(k, ""),
            "top_contexts": [
                {"strength": s, "text": t}
                for s, t in sorted(heaps.get(k, []), key=lambda p: -p[0])
            ],
        }
        for k in descriptions
    }
    out_path.write_text(json.dumps(serializable, indent=2))
    print(f"[save] wrote {len(descriptions)} descriptions to {out_path}", flush=True)

    wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=cfg.run_name,
        config={
            "base_model": cfg.base_model,
            "sae_layer": cfg.sae_layer,
            "d_sae": cfg.d_sae,
            "topk": cfg.topk,
            "llm_model": cfg.llm_model,
            "llm_api_base": cfg.llm_api_base,
            "num_features_to_explain": cfg.num_features_to_explain,
            "num_tokens_to_stream": cfg.num_tokens_to_stream,
            "context_window": cfg.context_window,
            "top_contexts_per_feature": cfg.top_contexts_per_feature,
        },
    )
    columns = ["feature_idx", "n_contexts", "top_strength", "description", "top_context_preview"]
    table = wandb.Table(columns=columns)
    for feat_idx, desc in descriptions.items():
        ctx = sorted(heaps.get(feat_idx, []), key=lambda p: -p[0])
        top_strength = ctx[0][0] if ctx else 0.0
        preview = ctx[0][1][:300] if ctx else ""
        table.add_data(int(feat_idx), len(ctx), float(top_strength), desc, preview)
    wandb.log({"explainer/features": table})
    wandb.log({"explainer/n_described": len(descriptions)})

    artifact = wandb.Artifact("feature-descriptions", type="explanations")
    artifact.add_file(str(out_path))
    wandb.log_artifact(artifact)
    wandb.finish()

    print("[main] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
