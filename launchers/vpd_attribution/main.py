"""Multi-layer VPD attribution graph — compute per-component, signed,
per-token attribution to a target-logit for the Dria interpretability post.

Loads every trained vpd-v2-final checkpoint from the eight-layer sweep
(0/4/8/12/16/20/24/27) on Qwen2.5-Coder-1.5B for either the BASE model
(SKIP_LORA_MERGE=1) or the LoRA-merged model. Installs MaskedLinears +
Gammas at all 8 layers simultaneously, then for each demo prompt:

  1.  Forward at the operating gate g(x) -> capture per-(layer, matrix, token,
      component) gate magnitudes.
  2.  Integrated gradients on the gate values: K steps from g=0 to g=1
      (mask channel scaled linearly). For each component c, accumulate
      d(logit_target)/d(g_c) integrated along the path. Multiply by g_c to
      get the signed attribution share.
  3.  Edges: for each adjacent (layer L row -> layer L' row), set edge weight
      = mean over tokens of |attr_A| * |attr_B| * sign(attr_A * attr_B), where
      A is a source component and B is a downstream component. Threshold +
      top-K to keep the graph readable.

Outputs JSON per (prompt, model):
  {
    "prompt": "...",
    "tokens": [...str...],
    "target_idx": int,            # last input token's logits drive prediction
    "predicted_top": [{"token", "prob"}],
    "nodes": [
      {"id", "layer", "matrix", "component", "mean_gate",
       "attribution_per_token": [floats len T],
       "total_attribution": float}
    ],
    "edges": [{"src","dst","weight","sign"}],
  }

Designed for one A6000 pod, ~10-20 min wall for 10 prompts x 2 models.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT = "eren23/coder-interp-pilot"
LAYERS = [0, 4, 8, 12, 16, 20, 24, 27]


# ---------------------------------------------------------------------------
# VPD primitives — mirror launchers/vpd_lora/main.py so checkpoints load
# unchanged. Kept small & inline (no shared import) to avoid coupling.

class MaskedLinear(nn.Module):
    def __init__(self, original: nn.Linear, num_components: int):
        super().__init__()
        self.d_out, self.d_in = original.weight.shape
        self.C = num_components
        for p in original.parameters():
            p.requires_grad_(False)
        self.original_weight = original.weight
        self.original_bias = original.bias
        dev = original.weight.device
        std_u = 1.0 / math.sqrt(self.d_out)
        std_v = 1.0 / math.sqrt(self.d_in)
        self.U = nn.Parameter(torch.randn(self.d_out, num_components, device=dev) * std_u)
        self.V = nn.Parameter(torch.randn(self.d_in, num_components, device=dev) * std_v)
        with torch.no_grad():
            delta_init = original.weight.detach().float() - self.U @ self.V.T
        self.Delta = nn.Parameter(delta_init)
        self.current_m: Optional[torch.Tensor] = None

    def reset_mask(self) -> None:
        self.current_m = None

    def set_mask(self, m: torch.Tensor) -> None:
        self.current_m = m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.current_m is None:
            return F.linear(x, self.original_weight, self.original_bias)
        m = self.current_m
        proj = x.float() @ self.V
        scaled = proj * m
        y_uv = scaled @ self.U.T
        y_delta = x.float() @ self.Delta.T
        y = y_uv + y_delta
        if self.original_bias is not None:
            y = y + self.original_bias
        return y.to(x.dtype)


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


def _resolve(module: nn.Module, dotted: str) -> nn.Module:
    obj = module
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def _replace(parent: nn.Module, dotted: str, new_mod: nn.Module) -> None:
    parts = dotted.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_mod)


# ---------------------------------------------------------------------------
# Multi-layer system: install MaskedLinears + Gammas at all 8 trained layers.

@dataclass
class LayerCfg:
    layer: int
    target_modules: tuple[str, ...]
    num_components: int
    gamma_hidden: int


class MultiLayerVPD(nn.Module):
    def __init__(self, model: nn.Module, per_layer: dict[int, LayerCfg]):
        super().__init__()
        self.model = model
        self.per_layer = per_layer
        self.masked: nn.ModuleDict = nn.ModuleDict()
        self.gammas: nn.ModuleDict = nn.ModuleDict()
        self._captured_x: dict[str, torch.Tensor] = {}

        for layer, cfg in per_layer.items():
            block = model.model.layers[layer]
            for target in cfg.target_modules:
                orig: nn.Linear = _resolve(block, target)
                wrapped = MaskedLinear(orig, cfg.num_components)
                _replace(block, target, wrapped)
                key = self.make_key(layer, target)
                self.masked[key] = wrapped
                self.gammas[key] = GammaPerMatrix(
                    d_in=wrapped.d_in,
                    num_components=cfg.num_components,
                    hidden=cfg.gamma_hidden,
                )

                def _make_hook(k: str):
                    def _hook(_m, inp):
                        self._captured_x[k] = inp[0]
                    return _hook

                wrapped.register_forward_pre_hook(_make_hook(key))

    @staticmethod
    def make_key(layer: int, target: str) -> str:
        return f"L{layer}__{target.replace('.', '_')}"

    @staticmethod
    def parse_key(key: str) -> tuple[int, str]:
        layer_s, matrix = key.split("__", 1)
        return int(layer_s[1:]), matrix

    @property
    def keys(self) -> list[str]:
        return list(self.masked.keys())

    def reset_masks(self) -> None:
        for m in self.masked.values():
            m.reset_mask()

    def compute_gates_pre_sigmoid(self) -> dict[str, torch.Tensor]:
        return {k: self.gammas[k](self._captured_x[k]) for k in self.keys}


def load_checkpoint_into(vpd: MultiLayerVPD, layer: int, ckpt: dict) -> None:
    """Apply a per-layer vpd-v2-final state dict to the matching matrices/gammas
    in the multi-layer system."""
    ml_cfg = ckpt["config"]
    targets = ml_cfg["target_modules"]
    for target in targets:
        key = MultiLayerVPD.make_key(layer, target)
        if key not in vpd.masked:
            raise KeyError(f"layer {layer} matrix {target} not installed (key {key})")
        flat = target.replace(".", "_")
        ml = vpd.masked[key]
        m_state = ckpt["masked"][flat]
        with torch.no_grad():
            ml.U.copy_(m_state["U"].to(ml.U.device, ml.U.dtype))
            ml.V.copy_(m_state["V"].to(ml.V.device, ml.V.dtype))
            ml.Delta.copy_(m_state["Delta"].to(ml.Delta.device, ml.Delta.dtype))
        vpd.gammas[key].load_state_dict(
            {k: v.to(ml.U.device) for k, v in ckpt["gammas"][flat].items()}
        )


# ---------------------------------------------------------------------------
# Forward helpers

def forward_with_mask_scale(
    vpd: MultiLayerVPD,
    input_ids: torch.Tensor,
    g_dict: dict[str, torch.Tensor],
    scale: float,
) -> torch.Tensor:
    """Run the LM with mask m = scale * g, for IG path integration.

    A fresh r=0 mask (no adversarial noise — we want clean attribution to the
    structured gates, not to the random source).
    """
    for k in vpd.keys:
        vpd.masked[k].set_mask(scale * g_dict[k])
    out = vpd.model(input_ids=input_ids, use_cache=False).logits
    vpd.reset_masks()
    return out


def integrated_gradient_attribution(
    vpd: MultiLayerVPD,
    input_ids: torch.Tensor,
    target_pos: int,
    target_token_id: int,
    n_steps: int = 16,
) -> dict[str, torch.Tensor]:
    """Integrated gradients on the gate values.

    Path: scale ∈ [0, 1] in n_steps. Attribution_c per (token,component) =
    mean_{step}  ∂logit_target / ∂g_c * g_c.

    Returns dict[key] -> (T, C) tensor (single batch dimension stripped).
    """
    # Compute the operating gates from the upper-leaky sigmoid on the captured
    # x. We need to do one inference-only forward first to populate _captured_x.
    vpd.reset_masks()
    with torch.no_grad():
        _ = vpd.model(input_ids=input_ids, use_cache=False).logits
    z = vpd.compute_gates_pre_sigmoid()
    g_op = {k: upper_leaky_sigmoid(v.detach()).clamp(0.0, 1.0) for k, v in z.items()}

    # Make g_op require grad — we differentiate the logit w.r.t. it at each step.
    g_param = {k: g.clone().requires_grad_(True) for k, g in g_op.items()}

    # Accumulators.
    accum: dict[str, torch.Tensor] = {
        k: torch.zeros_like(g, dtype=torch.float32, device=g.device)
        for k, g in g_op.items()
    }

    for step in range(n_steps):
        scale = (step + 1) / n_steps
        for k in vpd.keys:
            vpd.masked[k].set_mask(scale * g_param[k])
        logits = vpd.model(input_ids=input_ids, use_cache=False).logits
        # Target logit at target_pos for target_token_id (B=1).
        target_logit = logits[0, target_pos, target_token_id]

        grads = torch.autograd.grad(
            outputs=target_logit, inputs=list(g_param.values()),
            retain_graph=False, create_graph=False,
        )
        for k, g_grad in zip(vpd.keys, grads):
            accum[k] += g_grad.detach().float().squeeze(0)
        vpd.reset_masks()

    # Final attribution = (mean grad along path) * g_op
    attribution: dict[str, torch.Tensor] = {}
    for k in vpd.keys:
        mean_grad = accum[k] / n_steps  # (T, C)
        attribution[k] = (mean_grad * g_op[k].detach().float().squeeze(0))
    return attribution, {k: g_op[k].detach().float().squeeze(0) for k in vpd.keys}


# ---------------------------------------------------------------------------
# Edges: relevance-flow between adjacent (layer, matrix) rows.
#
# We use a token-aligned co-attribution score:
#   edge(A in row R1, B in row R2) = sum_t sign(a_A[t]) * sign(a_B[t]) *
#                                    sqrt(|a_A[t]| * |a_B[t]|)
# weighted positively when both rise/fall together on the same token, negative
# when they oppose. Captures circuit-style "A's signal reads into B" relations
# without needing the (expensive) Jacobian between gates across layers.

def compute_edges(
    attribution: dict[str, torch.Tensor],
    threshold: float = 0.0,
    top_k_per_node: int = 4,
) -> list[dict]:
    """Adjacent-layer edges between alive components."""
    # Group by (layer, matrix).
    by_row: dict[tuple[int, str], list[tuple[str, int]]] = {}
    for key, attr in attribution.items():
        layer, matrix = MultiLayerVPD.parse_key(key)
        # alive = any token has |attr| > eps
        live_mask = attr.abs().max(dim=0).values > 1e-3
        for c in torch.nonzero(live_mask).flatten().tolist():
            by_row.setdefault((layer, matrix), []).append((key, c))

    # Adjacent layers: pair every (Li, *) row with (Lj, *) where Lj is the
    # NEXT trained layer up. We also pair matrices intra-layer in the standard
    # computational order: q->o, k->o, v->o, o->up, up->down.
    intra_layer = [
        ("self_attn_q_proj", "self_attn_o_proj"),
        ("self_attn_k_proj", "self_attn_o_proj"),
        ("self_attn_v_proj", "self_attn_o_proj"),
        ("self_attn_o_proj", "mlp_up_proj"),
        ("mlp_up_proj",      "mlp_down_proj"),
    ]
    layer_list = sorted({layer for layer, _ in by_row.keys()})

    edges = []
    def _emit(src_key, src_c, dst_key, dst_c):
        a = attribution[src_key][:, src_c]   # (T,)
        b = attribution[dst_key][:, dst_c]
        s = torch.sign(a) * torch.sign(b)
        mag = (a.abs() * b.abs()).sqrt()
        w = (s * mag).sum().item()
        if abs(w) < threshold:
            return
        edges.append({
            "src": f"{src_key}:{src_c}",
            "dst": f"{dst_key}:{dst_c}",
            "weight": float(w),
            "sign": "+" if w >= 0 else "-",
        })

    # Intra-layer connections
    for layer in layer_list:
        for src_m, dst_m in intra_layer:
            src_row = by_row.get((layer, src_m), [])
            dst_row = by_row.get((layer, dst_m), [])
            for src_key, src_c in src_row:
                for dst_key, dst_c in dst_row:
                    _emit(src_key, src_c, dst_key, dst_c)

    # Inter-layer (down.mlp at L -> q/k/v at L+1)
    for i, l1 in enumerate(layer_list[:-1]):
        l2 = layer_list[i + 1]
        for src_m in ("mlp_down_proj",):
            for dst_m in ("self_attn_q_proj", "self_attn_k_proj", "self_attn_v_proj"):
                src_row = by_row.get((l1, src_m), [])
                dst_row = by_row.get((l2, dst_m), [])
                for src_key, src_c in src_row:
                    for dst_key, dst_c in dst_row:
                        _emit(src_key, src_c, dst_key, dst_c)

    # Trim to top-K per src node (keep graph readable).
    by_src: dict[str, list[dict]] = {}
    for e in edges:
        by_src.setdefault(e["src"], []).append(e)
    trimmed = []
    for src_id, lst in by_src.items():
        lst.sort(key=lambda x: abs(x["weight"]), reverse=True)
        trimmed.extend(lst[:top_k_per_node])
    return trimmed


# ---------------------------------------------------------------------------
# Main

DEFAULT_PROMPTS = [
    {"id": "rust_pubuse",    "text": "pub use crate::"},
    {"id": "py_dataclasses", "text": "from dataclasses import"},
    {"id": "ts_export",      "text": "export const "},
    {"id": "rust_impl_drop", "text": "impl Drop for "},
    {"id": "py_django",      "text": "class Profile(models.Model):\n    user = models."},
    {"id": "py_async_def",   "text": "async def fetch("},
    {"id": "rust_trait",     "text": "trait Iterator { fn next(&mut self) -> "},
    {"id": "ts_typeof",      "text": "type ButtonProps = "},
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-steps", type=int, default=int(os.environ.get("IG_STEPS", "16")))
    p.add_argument("--max-seq", type=int, default=int(os.environ.get("MAX_SEQ", "32")))
    p.add_argument("--out-dir", type=str, default=os.environ.get("OUT_DIR", "/workspace/project/attribution_out"))
    p.add_argument("--top-k-edges", type=int, default=int(os.environ.get("TOP_K_EDGES", "4")))
    p.add_argument("--prompts-file", type=str, default=os.environ.get("PROMPTS_FILE", ""))
    return p.parse_args()


def load_prompts(path: str) -> list[dict]:
    if not path:
        return DEFAULT_PROMPTS
    return json.loads(Path(path).read_text())


def download_checkpoint(api, run_id: str, workspace: Path) -> Path:
    r = api.run(f"{PROJECT}/{run_id}")
    arts = [a for a in r.logged_artifacts() if a.name.startswith("vpd-v2-final")]
    if not arts:
        raise RuntimeError(f"no vpd-v2-final artifact on {run_id}")
    art = arts[-1]
    dl_dir = workspace / "vpd_ckpts" / run_id
    dl_dir.mkdir(parents=True, exist_ok=True)
    art.download(root=str(dl_dir))
    pts = list(dl_dir.rglob("*.pt"))
    if not pts:
        raise RuntimeError(f"no .pt in {dl_dir}")
    return pts[0]


def build_model(workspace: Path, base_model_id: str, skip_lora: bool):
    print(f"[build] tokenizer {base_model_id}", flush=True)
    tok = AutoTokenizer.from_pretrained(base_model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"[build] base {base_model_id}", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype=torch.bfloat16, device_map="cuda",
    )
    if skip_lora:
        print("[build] BASE mode (no LoRA)", flush=True)
        model = base
    else:
        lora_run = os.environ.get("LORA_WANDB_RUN", f"{PROJECT}/niiz0d0u")
        lora_art_name = os.environ.get("LORA_ARTIFACT", "lora-style-final")
        print(f"[build] downloading LoRA {lora_art_name} from {lora_run}", flush=True)
        api = wandb.Api()
        src = api.run(lora_run)
        arts = [a for a in src.logged_artifacts() if a.name.startswith(lora_art_name)]
        adapter_dir = workspace / "lora_adapter_dl"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        arts[-1].download(root=str(adapter_dir))
        from peft import PeftModel
        peft_model = PeftModel.from_pretrained(base, str(adapter_dir))
        print("[build] merging adapter", flush=True)
        model = peft_model.merge_and_unload()
    model.train(False)
    for p in model.parameters():
        p.requires_grad_(False)
    return tok, model


def main() -> int:
    args = parse_args()
    workspace = Path(os.environ.get("WORKSPACE_DIR", "/workspace/project"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_model_id = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-Coder-1.5B")
    skip_lora = os.environ.get("SKIP_LORA_MERGE", "0") == "1"
    mode = "base" if skip_lora else "lora"
    print(f"[main] mode={mode}", flush=True)

    api = wandb.Api()

    # Pick the 8 sweep runs matching mode.
    SWEEP_IDS = {
        ("base", 0):  "tv0p4dr8", ("base", 4):  "vpbhnnxw", ("base", 8):  "yctbiwgm",
        ("base", 12): "aqh1gseb", ("base", 16): "tp64wyw6", ("base", 20): "xgj2p5og",
        ("base", 24): "20c9qn68", ("base", 27): "cki3d3fm",
        ("lora", 0):  "lzsdkvyi", ("lora", 4):  "hhazy118", ("lora", 8):  "62nybek7",
        ("lora", 12): "ngx5wy4c", ("lora", 16): "7f766bvf", ("lora", 20): "sj37sykr",
        ("lora", 24): "drqswaa0", ("lora", 27): "5t8vmd74",
    }

    # Download all 8 checkpoints in serial (we can parallelize later).
    ckpts: dict[int, dict] = {}
    for layer in LAYERS:
        run_id = SWEEP_IDS[(mode, layer)]
        print(f"[ckpt] L{layer:>2} <- {run_id}", flush=True)
        path = download_checkpoint(api, run_id, workspace)
        ckpts[layer] = torch.load(path, map_location="cpu", weights_only=False)

    # Use the first ckpt to derive shared config (num_components, gamma_hidden, targets).
    sample_cfg = ckpts[LAYERS[0]]["config"]
    targets = tuple(sample_cfg["target_modules"])
    num_components = int(sample_cfg["num_components"])
    gamma_hidden = int(sample_cfg["gamma_hidden"])
    print(
        f"[cfg] targets={targets} C={num_components} gamma_hidden={gamma_hidden}",
        flush=True,
    )

    tok, model = build_model(workspace, base_model_id, skip_lora)

    per_layer = {
        layer: LayerCfg(
            layer=layer, target_modules=targets,
            num_components=num_components, gamma_hidden=gamma_hidden,
        )
        for layer in LAYERS
    }
    vpd = MultiLayerVPD(model, per_layer).cuda()

    for layer in LAYERS:
        load_checkpoint_into(vpd, layer, ckpts[layer])
    print(f"[vpd] loaded {len(vpd.keys)} matrices over {len(LAYERS)} layers", flush=True)

    prompts = load_prompts(args.prompts_file)
    print(f"[main] {len(prompts)} prompts", flush=True)

    for pr in prompts:
        t0 = time.time()
        text = pr["text"]
        enc = tok(text, return_tensors="pt", truncation=True, max_length=args.max_seq)
        input_ids = enc.input_ids.cuda()
        # Last input position predicts the next token; use it as target_pos.
        target_pos = input_ids.shape[1] - 1

        # Greedy decode the top predicted next-token for labeling purposes.
        with torch.no_grad():
            vpd.reset_masks()
            logits = model(input_ids=input_ids, use_cache=False).logits
            probs = F.softmax(logits[0, target_pos], dim=-1)
            topk = probs.topk(5)
            predicted_top = [
                {"token": tok.decode([int(i)]), "prob": float(v)}
                for i, v in zip(topk.indices.tolist(), topk.values.tolist())
            ]
        target_token_id = int(topk.indices[0])
        print(
            f"[ig] {pr['id']:<20s} target='{predicted_top[0]['token']}' "
            f"(p={predicted_top[0]['prob']:.3f})",
            flush=True,
        )

        attribution, g_op = integrated_gradient_attribution(
            vpd, input_ids,
            target_pos=target_pos, target_token_id=target_token_id,
            n_steps=args.n_steps,
        )

        # Build nodes: ALL live components in any layer/matrix.
        nodes = []
        for key, attr in attribution.items():
            layer, matrix = MultiLayerVPD.parse_key(key)
            mg_per_c = g_op[key].mean(dim=0)   # (C,)
            attr_per_c_token = attr.cpu().numpy().tolist()  # T x C list
            attr_total = attr.sum(dim=0)
            live = (attr.abs().max(dim=0).values > 1e-3).cpu().tolist()
            for c in range(len(live)):
                if not live[c]:
                    continue
                nodes.append({
                    "id": f"{key}:{c}",
                    "layer": layer, "matrix": matrix, "component": c,
                    "mean_gate": float(mg_per_c[c].item()),
                    "attribution_per_token": [row[c] for row in attr_per_c_token],
                    "total_attribution": float(attr_total[c].item()),
                })

        edges = compute_edges(attribution, threshold=1e-3, top_k_per_node=args.top_k_edges)
        token_strs = [tok.decode([int(t)]) for t in input_ids[0].tolist()]

        obj = {
            "prompt_id": pr["id"], "prompt": text, "mode": mode,
            "base_model": base_model_id,
            "tokens": token_strs,
            "target_pos": target_pos, "target_token_id": target_token_id,
            "predicted_top": predicted_top,
            "layers": LAYERS, "matrices": list(targets),
            "num_components": num_components,
            "nodes": nodes, "edges": edges,
        }
        out_path = out_dir / f"attribution_{mode}_{pr['id']}.json"
        out_path.write_text(json.dumps(obj))
        sz = out_path.stat().st_size
        print(f"[ig] {pr['id']:<20s} -> {out_path.name} "
              f"({sz/1024:.1f} KB, {len(nodes)} nodes, {len(edges)} edges, "
              f"{time.time()-t0:.1f}s)", flush=True)

    # Bundle and log to W&B for the article exporter to pick up.
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "coder-interp-pilot"),
        entity=os.environ.get("WANDB_ENTITY"),
        name=os.environ.get("WANDB_RUN_NAME", f"vpd-attribution-{mode}"),
        config={
            "mode": mode, "base_model": base_model_id,
            "layers": LAYERS, "matrices": list(targets),
            "num_components": num_components,
            "n_steps": args.n_steps, "n_prompts": len(prompts),
        },
    )
    art = wandb.Artifact(f"attribution-graphs-{mode}", type="analysis")
    for f in sorted(out_dir.glob(f"attribution_{mode}_*.json")):
        art.add_file(str(f))
    wandb.log_artifact(art)
    wandb.finish()

    print("[main] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
