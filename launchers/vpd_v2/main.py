"""Faithful (scoped-down) re-implementation of Goodfire's VPD.

Reference: https://www.goodfire.ai/research/interpreting-lm-parameters

Implements every component of the paper's training recipe:

  * Per-matrix rank-1 decomposition  W ~= sum_c U[:,c] V[:,c]^T + Delta
  * Causal-importance network Gamma producing g(x_t) in [0,1]^C
  * Mask formula     m = g + (1 - g) * r        (NOT Bernoulli/Concrete)
  * Persistent-PGD attack on the *source* r (not on activations / weights)
  * Five training losses:
      L_adv-recon, L_stoch-recon, L_imp-min, L_freq-min, L_Delta-L2
  * Lower-leaky and upper-leaky hard sigmoids with STE on Gamma's output
  * Lp annealing  (p: 2.0 -> 0.4 linearly)
  * Stochastic subset routing  (each (b,t) masks a random k of L matrices)
  * 400-step Delta-L2 warmup phase

Scoped down from the paper (which targets all 24 matrices of a 67M LM with a
shared transformer Gamma) to:
  * a SINGLE transformer block of Qwen3-0.6B  (7 matrices: q,k,v,o,gate,up,down)
  * per-matrix small MLP Gamma  (paper: shared transformer over all matrices)
  * smaller component counts, shorter training

The interpretability question is unchanged: do the gates sparsify so that
specific (token, subcomponent) pairs become semantically coherent?

Outputs:
  * W&B run with loss curves, gate-statistics plots, alive-component count
  * `vpd-v2-final` artifact (state_dict for U, V, Delta, Gamma per matrix)
"""

from __future__ import annotations

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


# ---------------------------------------------------------------------------
# Config

def _env(name: str, default: Optional[str] = None) -> str:
    val = os.environ.get(name, default)
    if val is None:
        raise RuntimeError(f"required env var not set: {name}")
    return val


@dataclass
class Cfg:
    base_model: str
    layer: int
    target_modules: tuple[str, ...]     # e.g. ("self_attn.q_proj", "mlp.up_proj", ...)
    num_components: int                 # per-matrix component count
    gamma_hidden: int

    # Training
    train_steps: int
    warmup_steps: int
    batch_size: int
    seq_len: int
    lr_main: float
    lr_adv: float
    n_adv: int                          # inner-PGD steps per outer step

    # Loss coefficients (paper)
    beta_adv: float                     # 0.5
    beta_stoch: float                   # 0.5
    beta_imp: float                     # 2e-4
    beta_freq: float                    # 1e-4
    beta_delta: float                   # 1e7 (warmup-only by default)

    # Lp annealing
    p_initial: float
    p_final: float

    # Gate STE
    leaky_alpha: float                  # 0.01

    # Reporting
    log_interval: int
    checkpoint_interval: int

    # Data
    dataset_id: str
    dataset_split: str
    workspace: Path
    run_name: str
    wandb_project: str
    wandb_entity: Optional[str]


def load_cfg() -> Cfg:
    targets_csv = _env(
        "VPD_TARGETS",
        "self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj,"
        "mlp.gate_proj,mlp.up_proj,mlp.down_proj",
    )
    return Cfg(
        base_model=_env("BASE_MODEL", "Qwen/Qwen3-0.6B"),
        layer=int(_env("VPD_LAYER", "13")),
        target_modules=tuple(t.strip() for t in targets_csv.split(",") if t.strip()),
        num_components=int(_env("VPD_NUM_COMPONENTS", "128")),
        gamma_hidden=int(_env("VPD_GAMMA_HIDDEN", "256")),
        train_steps=int(_env("VPD_TRAIN_STEPS", "1500")),
        warmup_steps=int(_env("VPD_WARMUP_STEPS", "400")),
        batch_size=int(_env("VPD_BATCH_SIZE", "4")),
        seq_len=int(_env("SEQ_LEN", "128")),
        lr_main=float(_env("VPD_LR_MAIN", "5e-5")),
        lr_adv=float(_env("VPD_LR_ADV", "1e-2")),
        n_adv=int(_env("VPD_N_ADV", "3")),
        beta_adv=float(_env("VPD_BETA_ADV", "0.5")),
        beta_stoch=float(_env("VPD_BETA_STOCH", "0.5")),
        beta_imp=float(_env("VPD_BETA_IMP", "2e-4")),
        beta_freq=float(_env("VPD_BETA_FREQ", "1e-4")),
        beta_delta=float(_env("VPD_BETA_DELTA", "1e7")),
        p_initial=float(_env("VPD_P_INITIAL", "2.0")),
        p_final=float(_env("VPD_P_FINAL", "0.4")),
        leaky_alpha=float(_env("VPD_LEAKY_ALPHA", "0.01")),
        log_interval=int(_env("LOG_INTERVAL", "10")),
        checkpoint_interval=int(_env("CHECKPOINT_INTERVAL", "500")),
        dataset_id=_env("DATASET_ID", "eren23/eren-code-style"),
        dataset_split=_env("DATASET_SPLIT", "train"),
        workspace=Path(_env("WORKSPACE_DIR", "/workspace/project")),
        run_name=_env("WANDB_RUN_NAME", f"vpd-v2-{int(time.time())}"),
        wandb_project=_env("WANDB_PROJECT", "coder-interp-pilot"),
        wandb_entity=os.environ.get("WANDB_ENTITY"),
    )


# ---------------------------------------------------------------------------
# Leaky hard sigmoid variants with STE
#
# `_HardSigmoidLowerLeaky`: forward is clamp(z, 0, 1). Below 0 only negative
# incoming grads pass through with slope alpha (allows mask to "wake up" from
# z<0 but not push further). Above 1, no grad.
#
# `_HardSigmoidUpperLeaky`: forward continues linearly with slope alpha above
# 1; real linear region, not STE. Used for L_imp-min / L_freq-min so the
# importance penalty can keep pushing components past saturation.

class _HardSigmoidLowerLeaky(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.save_for_backward(z)
        ctx.alpha = alpha
        return z.clamp(0.0, 1.0)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (z,) = ctx.saved_tensors
        alpha = ctx.alpha
        grad_z = grad_out.clone()
        below_0 = z < 0
        above_1 = z > 1
        # Below 0: only let through grads that would *increase* z (grad<0 in PyTorch
        # gradient-descent convention since update is z = z - lr*grad).
        # Block positive grads (those would push z further negative).
        mask_block = below_0 & (grad_out > 0)
        mask_leak = below_0 & (grad_out <= 0)
        grad_z[mask_block] = 0
        grad_z[mask_leak] = grad_out[mask_leak] * alpha
        grad_z[above_1] = 0
        return grad_z, None


def lower_leaky_sigmoid(z: torch.Tensor, alpha: float = 0.01) -> torch.Tensor:
    return _HardSigmoidLowerLeaky.apply(z, alpha)


def upper_leaky_sigmoid(z: torch.Tensor, alpha: float = 0.01) -> torch.Tensor:
    """forward(z) = clamp(z, 0, 1) + alpha * relu(z - 1).
    Real (non-STE) function; autograd handles gradients naturally.
    """
    return torch.clamp(z, 0.0, 1.0) + alpha * F.relu(z - 1.0)


# ---------------------------------------------------------------------------
# MaskedLinear: wraps an nn.Linear, computes y = W'(m) @ x  + b
# where  W'(m) = sum_c m_c * U[:,c] V[:,c]^T  +  Delta
#
# `current_m`: (B, T, C) per-token mask tensor set externally per forward.
#   If None, falls back to the original W (used for f(x|W) target forward).

class MaskedLinear(nn.Module):
    def __init__(self, original: nn.Linear, num_components: int):
        super().__init__()
        self.d_out, self.d_in = original.weight.shape
        self.C = num_components
        # Freeze the original linear (we'll only read its weight as the "target").
        for p in original.parameters():
            p.requires_grad_(False)
        self.original_weight = original.weight    # (d_out, d_in)  frozen
        self.original_bias = original.bias        # may be None
        # Trainable rank-1 components. Allocate on the same device as the
        # original linear so we don't get cross-device errors when the model
        # is already on cuda before VPDSystem is built.
        dev = original.weight.device
        std_u = 1.0 / math.sqrt(self.d_out)
        std_v = 1.0 / math.sqrt(self.d_in)
        self.U = nn.Parameter(
            torch.randn(self.d_out, num_components, device=dev) * std_u
        )
        self.V = nn.Parameter(
            torch.randn(self.d_in, num_components, device=dev) * std_v
        )
        # Delta = W - sum U V^T  (kept exact at start; refined during warmup).
        with torch.no_grad():
            uv = self.U @ self.V.T
            delta_init = self.original_weight.detach().float() - uv
        self.Delta = nn.Parameter(delta_init)
        self.current_m: Optional[torch.Tensor] = None  # set per forward

    def reset_mask(self) -> None:
        self.current_m = None

    def set_mask(self, m: torch.Tensor) -> None:
        """m: (B, T, C) per-token mask. C must match self.C."""
        self.current_m = m

    def approx_W(self) -> torch.Tensor:
        return self.U @ self.V.T + self.Delta

    def decomp_residual(self) -> torch.Tensor:
        approx = self.U @ self.V.T + self.Delta
        return ((self.original_weight.float() - approx) ** 2).mean()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.current_m is None:
            return F.linear(x, self.original_weight, self.original_bias)
        m = self.current_m   # (B, T, C)
        proj = x.float() @ self.V                       # (B, T, C)
        scaled = proj * m                                # (B, T, C)
        y_uv = scaled @ self.U.T                         # (B, T, d_out)
        y_delta = x.float() @ self.Delta.T               # (B, T, d_out)
        y = y_uv + y_delta
        if self.original_bias is not None:
            y = y + self.original_bias
        return y.to(x.dtype)


# ---------------------------------------------------------------------------
# Per-matrix Gamma network: small MLP on the RMS-normed input of the matrix.

class GammaPerMatrix(nn.Module):
    def __init__(self, d_in: int, num_components: int, hidden: int):
        super().__init__()
        self.norm = nn.RMSNorm(d_in)
        self.fc1 = nn.Linear(d_in, hidden)
        self.fc2 = nn.Linear(hidden, num_components)
        # Initialize so pre-sigmoid output starts near 0  (paper: g near 0 initially).
        nn.init.zeros_(self.fc2.bias)
        nn.init.normal_(self.fc2.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_in) — returns pre-sigmoid logits z (B, T, C)
        h = self.norm(x.float())
        h = F.gelu(self.fc1(h))
        z = self.fc2(h)
        return z


# ---------------------------------------------------------------------------
# Helpers for navigating the transformer block.

def _resolve(module: nn.Module, dotted: str) -> nn.Module:
    obj: nn.Module = module
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def _replace(parent: nn.Module, dotted: str, new_mod: nn.Module) -> None:
    parts = dotted.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_mod)


# ---------------------------------------------------------------------------
# Setup: install MaskedLinears at all targeted matrices in the chosen block,
# and one Gamma per matrix. The gamma is fed the matrix's input (captured by
# a pre-forward hook on the MaskedLinear itself).

class VPDSystem(nn.Module):
    """Holds the MaskedLinears + per-matrix Gammas. Provides plumbing to set
    masks for forward passes."""

    def __init__(self, model: nn.Module, cfg: Cfg):
        super().__init__()
        self.cfg = cfg
        self.block = model.model.layers[cfg.layer]
        self.masked: nn.ModuleDict = nn.ModuleDict()
        self.gammas: nn.ModuleDict = nn.ModuleDict()
        self._captured_x: dict[str, torch.Tensor] = {}

        for target in cfg.target_modules:
            orig: nn.Linear = _resolve(self.block, target)  # type: ignore[assignment]
            if not isinstance(orig, nn.Linear):
                raise TypeError(f"target {target} is {type(orig)}, expected nn.Linear")
            wrapped = MaskedLinear(orig, cfg.num_components)
            _replace(self.block, target, wrapped)
            # ModuleDict keys must not contain '.'; flatten.
            key = target.replace(".", "_")
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

    @property
    def keys(self) -> list[str]:
        return list(self.masked.keys())

    def reset_masks(self) -> None:
        for ml in self.masked.values():
            ml.reset_mask()  # type: ignore[union-attr]

    def reset_captures(self) -> None:
        self._captured_x.clear()

    def compute_gates(self) -> dict[str, torch.Tensor]:
        """Return pre-sigmoid logits z[key] for every key, computed from the
        captured x of the last forward."""
        return {k: self.gammas[k](self._captured_x[k]) for k in self.keys}

    def total_components(self) -> int:
        return len(self.keys) * self.cfg.num_components

    def decomp_residual(self) -> torch.Tensor:
        return torch.stack([m.decomp_residual() for m in self.masked.values()]).mean()


# ---------------------------------------------------------------------------
# Persistent r_adv buffer (one per matrix, shape (B, T, C)).
# Re-allocated only when batch/seq shape changes.

class AdvSourceBuffer:
    def __init__(self, keys: list[str], device: torch.device):
        self.tensors: dict[str, torch.Tensor] = {}
        self.keys = keys
        self.device = device

    def ensure(self, B: int, T: int, C: int) -> None:
        for k in self.keys:
            t = self.tensors.get(k)
            if t is None or t.shape != (B, T, C):
                self.tensors[k] = (
                    torch.rand(B, T, C, device=self.device).requires_grad_(True)
                )

    def parameters(self) -> list[torch.Tensor]:
        return list(self.tensors.values())

    def clamp_(self) -> None:
        for t in self.tensors.values():
            t.data.clamp_(0.0, 1.0)


# ---------------------------------------------------------------------------
# Loss helpers

def kl_divergence(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    """KL(p || q) summed over vocab, averaged over batch * seq."""
    log_p = F.log_softmax(p_logits.float(), dim=-1)
    log_q = F.log_softmax(q_logits.float(), dim=-1)
    p = log_p.exp()
    kl = (p * (log_p - log_q)).sum(dim=-1)
    return kl.mean()


def importance_minimality_loss(
    gates_upper: dict[str, torch.Tensor], p: float
) -> torch.Tensor:
    """L_imp-min = (1 / (B T)) * sum_{l,c} sum_{b,t} |g|^p / total_comps.
    Mean across batch*seq, sum across components, sum across matrices.
    """
    parts: list[torch.Tensor] = []
    for g in gates_upper.values():
        # g shape (B, T, C). |g|^p per element, mean over B*T, sum over C.
        gv = g.abs().pow(p)
        parts.append(gv.mean(dim=(0, 1)).sum())
    return torch.stack(parts).sum()


def frequency_minimality_loss(
    gates_upper: dict[str, torch.Tensor], p: float, eps: float = 1e-6
) -> torch.Tensor:
    """L_freq-min = (1/(B T)) * sum_{l,c} s^l_c * log2(1 + s^l_c)
    where s^l_c = sum_{b,t} |g^l_{b,t,c}|^p
    """
    parts: list[torch.Tensor] = []
    for g in gates_upper.values():
        B, T, C = g.shape
        s = g.abs().add(eps).pow(p).sum(dim=(0, 1))   # (C,)
        s = s / (B * T)                                # normalize
        parts.append((s * torch.log2(1 + s)).sum())
    return torch.stack(parts).sum()


# ---------------------------------------------------------------------------
# Data utilities

def streaming_text_iter(cfg: Cfg):
    # Default: eren23/eren-code-style (public parquet). For other public
    # datasets, pass DATASET_ID via env.
    ds = load_dataset(
        cfg.dataset_id, split=cfg.dataset_split, streaming=True,
    )
    return iter(ds.shuffle(seed=42, buffer_size=200))


def collect_batch(cfg: Cfg, tok, ds_iter) -> torch.Tensor:
    texts: list[str] = []
    while len(texts) < cfg.batch_size:
        try:
            rec = next(ds_iter)
        except StopIteration:
            return None  # caller handles
        t = rec.get("content") or rec.get("text") or rec.get("new_contents")
        if t:
            texts.append(t)
    enc = tok(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=cfg.seq_len,
    )
    return enc.input_ids.cuda()


# ---------------------------------------------------------------------------
# Forward-pass orchestration: given a base model, run f(x|W) and f(x|W'(m))

def forward_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    out = model(input_ids=input_ids, use_cache=False)
    return out.logits


# ---------------------------------------------------------------------------
# One outer training step

def outer_step(
    cfg: Cfg,
    step: int,
    model,
    vpd: VPDSystem,
    main_opt: torch.optim.Optimizer,
    adv_opt: torch.optim.Optimizer,
    adv_buf: AdvSourceBuffer,
    input_ids: torch.Tensor,
    is_warmup: bool,
    p_now: float,
) -> dict[str, float]:

    B, T = input_ids.shape
    keys = vpd.keys
    C = cfg.num_components
    alpha = cfg.leaky_alpha

    # ---------- TARGET FORWARD ----------
    vpd.reset_masks()
    vpd.reset_captures()
    with torch.inference_mode():
        target_logits = forward_logits(model, input_ids).detach()
    # Now we have captured_x for every matrix.
    z_logits = vpd.compute_gates()  # dict key -> (B,T,C) pre-sigmoid
    g_lower: dict[str, torch.Tensor] = {
        k: lower_leaky_sigmoid(z, alpha) for k, z in z_logits.items()
    }
    g_upper: dict[str, torch.Tensor] = {
        k: upper_leaky_sigmoid(z, alpha) for k, z in z_logits.items()
    }

    # ---------- DELTA-L2 (always logged, used as sole loss in warmup) ----------
    decomp = vpd.decomp_residual()

    if is_warmup:
        loss = cfg.beta_delta * decomp
        main_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in vpd.parameters() if p.requires_grad], max_norm=0.01,
        )
        main_opt.step()
        return {
            "phase": "warmup",
            "loss_total": float(loss.item()),
            "decomp_residual": float(decomp.item()),
        }

    # ---------- ADVERSARIAL INNER LOOP (n_adv - 1 warmup PGD + 1 dual step) ----------
    adv_buf.ensure(B, T, C)

    # Inner warmup PGD updates: detach g, U, V, Delta so only r_adv learns.
    for _ in range(cfg.n_adv - 1):
        masks = {
            k: g_lower[k].detach()
            + (1.0 - g_lower[k].detach()) * adv_buf.tensors[k]
            for k in keys
        }
        for k in keys:
            vpd.masked[k].set_mask(masks[k])
        masked_logits = forward_logits(model, input_ids)
        l_inner = kl_divergence(target_logits, masked_logits)
        adv_opt.zero_grad()
        (-l_inner).backward()
        adv_opt.step()
        adv_buf.clamp_()
        vpd.reset_masks()

    # Final dual step: grads flow to r_adv (ascent) AND to U/V/Gamma/Delta (descent).
    masks = {
        k: g_lower[k] + (1.0 - g_lower[k]) * adv_buf.tensors[k]
        for k in keys
    }
    for k in keys:
        vpd.masked[k].set_mask(masks[k])
    adv_logits = forward_logits(model, input_ids)
    l_adv = kl_divergence(target_logits, adv_logits)
    vpd.reset_masks()

    # ---------- STOCHASTIC LOSS with subset routing ----------
    # For each (b,t) pick a random k in [1, L]; mask only k randomly-chosen matrices.
    L = len(keys)
    k_per_pos = torch.randint(1, L + 1, (B, T), device=input_ids.device)  # (B,T)
    # For each (b,t,l) decide whether matrix l is in the chosen subset.
    # Cheaper approximation: pick a per-position random permutation of [0..L), and
    # include matrices whose rank in permutation < k.
    rand = torch.rand(B, T, L, device=input_ids.device)
    rank = rand.argsort(dim=-1).argsort(dim=-1)  # (B,T,L), values in [0..L-1]
    in_subset = (rank < k_per_pos.unsqueeze(-1))  # (B,T,L) bool

    r_stoch = {k: torch.rand_like(g_lower[k]) for k in keys}
    masks_stoch = {}
    for li, k in enumerate(keys):
        m_full = g_lower[k] + (1.0 - g_lower[k]) * r_stoch[k]
        sub = in_subset[:, :, li].unsqueeze(-1)  # (B,T,1)
        ones = torch.ones_like(m_full)
        masks_stoch[k] = torch.where(sub, m_full, ones)
        # When NOT in subset, mask=1 means the matrix is used unmodified
        # (m=1 → W' = U V^T + Delta which should ~= W).
    for k in keys:
        vpd.masked[k].set_mask(masks_stoch[k])
    stoch_logits = forward_logits(model, input_ids)
    l_stoch = kl_divergence(target_logits, stoch_logits)
    vpd.reset_masks()

    # ---------- IMPORTANCE / FREQUENCY MINIMALITY ----------
    l_imp = importance_minimality_loss(g_upper, p_now)
    l_freq = frequency_minimality_loss(g_upper, p_now)

    # ---------- TOTAL ----------
    loss = (
        cfg.beta_adv * l_adv
        + cfg.beta_stoch * l_stoch
        + cfg.beta_imp * l_imp
        + cfg.beta_freq * l_freq
        # Delta-L2 only in warmup phase; tiny coefficient outside it
        # BUGFIX: was hardcoded 1e-3 — paper uses beta_delta=1e7 throughout
        # to keep Delta tight and force structure into U V^T. Without this the
        # decomposition is degenerate (Delta carries everything, U V is ~0).
        + cfg.beta_delta * decomp
    )
    main_opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for p in vpd.parameters() if p.requires_grad], max_norm=0.01,
    )
    main_opt.step()

    # Stats for logging.
    with torch.no_grad():
        gate_concat = torch.cat([g_lower[k].reshape(-1) for k in keys])
        alive_per_key = {
            f"alive/{k}": int((g_upper[k].mean(dim=(0, 1)) > 1e-6).sum().item())
            for k in keys
        }
        mean_g = float(gate_concat.mean().item())
        frac_g_over_05 = float((gate_concat > 0.5).float().mean().item())
        adv_diff = float((adv_logits.float() - target_logits.float()).abs().mean().item())

    out = {
        "phase": "main",
        "loss_total": float(loss.item()),
        "loss_adv": float(l_adv.item()),
        "loss_stoch": float(l_stoch.item()),
        "loss_imp": float(l_imp.item()),
        "loss_freq": float(l_freq.item()),
        "loss_decomp": float(decomp.item()),
        "p_now": p_now,
        "mean_g": mean_g,
        "frac_g_over_0.5": frac_g_over_05,
        "adv_logits_l1": adv_diff,
    }
    out.update(alive_per_key)
    return out


# ---------------------------------------------------------------------------
# Save / load

def save_checkpoint(cfg: Cfg, vpd: VPDSystem, step: int) -> Path:
    cfg.workspace.mkdir(parents=True, exist_ok=True)
    ckpt = cfg.workspace / f"vpd_v2_step_{step:06d}.pt"
    state = {
        "step": step,
        "config": {
            "base_model": cfg.base_model,
            "layer": cfg.layer,
            "target_modules": list(cfg.target_modules),
            "num_components": cfg.num_components,
            "gamma_hidden": cfg.gamma_hidden,
        },
        "masked": {
            k: {
                "U": vpd.masked[k].U.detach().cpu(),
                "V": vpd.masked[k].V.detach().cpu(),
                "Delta": vpd.masked[k].Delta.detach().cpu(),
            }
            for k in vpd.keys
        },
        "gammas": {k: vpd.gammas[k].state_dict() for k in vpd.keys},
    }
    torch.save(state, ckpt)
    return ckpt


# ---------------------------------------------------------------------------
# Main

def main() -> int:
    cfg = load_cfg()
    print(f"[main] cfg: {cfg}", flush=True)

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

    vpd = VPDSystem(model, cfg).cuda()
    print(
        f"[vpd] installed at layer {cfg.layer}, "
        f"{len(vpd.keys)} matrices, {vpd.total_components()} components total",
        flush=True,
    )

    wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=cfg.run_name,
        config={
            "base_model": cfg.base_model,
            "layer": cfg.layer,
            "target_modules": list(cfg.target_modules),
            "num_components": cfg.num_components,
            "train_steps": cfg.train_steps,
            "warmup_steps": cfg.warmup_steps,
            "batch_size": cfg.batch_size,
            "seq_len": cfg.seq_len,
            "lr_main": cfg.lr_main,
            "lr_adv": cfg.lr_adv,
            "n_adv": cfg.n_adv,
            "beta_adv": cfg.beta_adv,
            "beta_stoch": cfg.beta_stoch,
            "beta_imp": cfg.beta_imp,
            "beta_freq": cfg.beta_freq,
            "p_initial": cfg.p_initial,
            "p_final": cfg.p_final,
        },
    )

    trainable_main = [
        p for n, p in vpd.named_parameters()
        if p.requires_grad and not n.startswith("masked.")
        or "U" in n or "V" in n or "Delta" in n or "gammas" in n
    ]
    main_opt = torch.optim.AdamW(trainable_main, lr=cfg.lr_main, weight_decay=0.0)

    adv_buf = AdvSourceBuffer(vpd.keys, torch.device("cuda"))
    # Adv opt created lazily once buffer is allocated; need to provide a parameter at init.
    # Workaround: lazy init on first step.
    adv_opt: Optional[torch.optim.Optimizer] = None

    ds_iter = streaming_text_iter(cfg)
    last_log_t = time.time()

    total = cfg.warmup_steps + cfg.train_steps
    for step in range(total):
        # Refill batch (re-iter on exhaustion).
        ids = collect_batch(cfg, tok, ds_iter)
        if ids is None:
            ds_iter = streaming_text_iter(cfg)
            ids = collect_batch(cfg, tok, ds_iter)
            if ids is None:
                print("[main] dataset exhausted twice; stopping", flush=True)
                break

        # p annealing across main-phase only.
        if step < cfg.warmup_steps:
            p_now = cfg.p_initial
            is_warmup = True
        else:
            frac = (step - cfg.warmup_steps) / max(cfg.train_steps - 1, 1)
            p_now = cfg.p_initial + (cfg.p_final - cfg.p_initial) * frac
            is_warmup = False
            # Lazy-init adv_opt after we know shapes.
            if adv_opt is None:
                adv_buf.ensure(cfg.batch_size, cfg.seq_len, cfg.num_components)
                adv_opt = torch.optim.Adam(
                    adv_buf.parameters(), lr=cfg.lr_adv, betas=(0.5, 0.99),
                )

        try:
            stats = outer_step(
                cfg=cfg, step=step, model=model, vpd=vpd,
                main_opt=main_opt, adv_opt=adv_opt,  # type: ignore[arg-type]
                adv_buf=adv_buf, input_ids=ids, is_warmup=is_warmup, p_now=p_now,
            )
        except Exception as exc:
            print(f"[step {step}] ERROR: {exc}", flush=True)
            raise

        if step % cfg.log_interval == 0:
            elapsed = time.time() - last_log_t
            print(
                f"[step {step:>5}/{total}] "
                + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                           for k, v in stats.items() if not k.startswith("alive/"))
                + f" ({elapsed:.1f}s/{cfg.log_interval}s)",
                flush=True,
            )
            wandb.log({f"vpd/{k}": v for k, v in stats.items() if not isinstance(v, str)}, step=step)
            last_log_t = time.time()

        if (
            cfg.checkpoint_interval > 0
            and step > 0
            and step % cfg.checkpoint_interval == 0
        ):
            ckpt = save_checkpoint(cfg, vpd, step)
            print(f"[ckpt] {ckpt}", flush=True)
            art = wandb.Artifact(f"vpd-v2-step-{step}", type="model")
            art.add_file(str(ckpt))
            wandb.log_artifact(art)

    final = save_checkpoint(cfg, vpd, total)
    print(f"[final] {final}", flush=True)
    art = wandb.Artifact("vpd-v2-final", type="model")
    art.add_file(str(final))
    wandb.log_artifact(art)
    wandb.finish()
    print("[main] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
