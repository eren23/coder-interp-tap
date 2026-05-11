"""Minimal VPD (adVersarial Parameter Decomposition) pilot.

Decomposes a SINGLE weight matrix W of a base LM into C rank-1
subcomponents plus a causal-importance network g(x). Faithful to the
Goodfire paper's training objective except:

  - we drop the adversarial-reconstruction term (replaced with simple
    stochastic recon — much cheaper, no inner optimization loop)
  - we drop the frequency-minimality and L2 on Delta-output terms
  - we target ONE matrix instead of all 24 weight matrices

This is a feasibility / cost pilot. The point is to verify that
parameter-level decomposition works on a real LM (Qwen3-0.6B) at small
scale and to get a real $/h number for extrapolation to the 1.5B coder
model.

Pipeline:
  1. Load BASE_MODEL on cuda in bf16, freeze all params.
  2. Resolve the target weight by VPD_TARGET path within
     model.model.layers[VPD_LAYER]; e.g. "mlp.up_proj".
  3. Forward-hook the parent module to capture (x_t, W x_t) pairs for
     non-pad token positions.
  4. Init the rank-1 components U (d_out, C) and V (d_in, C) from the
     top-C SVD of W (strong warm start) and the importance MLP from
     random.
  5. Optimize:
       y_hat(x)         = sum_c g_c(x) * U[:,c] * (V[:,c] . x)
       L_recon          = MSE(y_hat, W x)
       L_stoch          = MSE(y_hat_stochastic, W x)   (mask ~ Bern(g))
       L_importance_min = lam_imp * mean(g)
       L_decomp_l2      = lam_decomp * ||W - U V^T||^2
       total = L_recon + L_stoch + L_importance_min + L_decomp_l2
  6. Log to W&B every LOG_INTERVAL steps; save state_dict + artifact at
     the end.
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
    target_path: str           # e.g. "mlp.up_proj"
    num_components: int
    importance_hidden: int
    lr: float
    batch_size: int            # number of token activations per opt step
    capture_batch: int         # number of sequences per forward
    seq_len: int
    train_steps: int
    log_interval: int
    checkpoint_interval: int
    lambda_recon: float
    lambda_stoch: float
    lambda_importance: float
    lambda_decomp: float
    dataset_id: str
    dataset_subset: str
    workspace: Path
    run_name: str
    wandb_project: str
    wandb_entity: Optional[str]


def load_cfg() -> Cfg:
    return Cfg(
        base_model=_env("BASE_MODEL"),
        layer=int(_env("VPD_LAYER")),
        target_path=_env("VPD_TARGET", "mlp.up_proj"),
        num_components=int(_env("VPD_NUM_COMPONENTS", "128")),
        importance_hidden=int(_env("VPD_IMPORTANCE_HIDDEN", "256")),
        lr=float(_env("VPD_LR", "3e-4")),
        batch_size=int(_env("VPD_BATCH_SIZE", "2048")),
        capture_batch=int(_env("CAPTURE_BATCH_SIZE", "8")),
        seq_len=int(_env("SEQ_LEN", "512")),
        train_steps=int(_env("VPD_TRAIN_STEPS", "1000")),
        log_interval=int(_env("LOG_INTERVAL", "20")),
        checkpoint_interval=int(_env("CHECKPOINT_INTERVAL", "500")),
        lambda_recon=float(_env("VPD_LAMBDA_RECON", "1.0")),
        lambda_stoch=float(_env("VPD_LAMBDA_STOCH", "1.0")),
        lambda_importance=float(_env("VPD_LAMBDA_IMPORTANCE", "0.01")),
        lambda_decomp=float(_env("VPD_LAMBDA_DECOMP", "0.1")),
        dataset_id=_env("DATASET_ID", "bigcode/commitpackft"),
        dataset_subset=_env("DATASET_SUBSET", "python"),
        workspace=Path(_env("WORKSPACE_DIR", "/workspace/project")),
        run_name=_env("WANDB_RUN_NAME", f"vpd-{int(time.time())}"),
        wandb_project=_env("WANDB_PROJECT", "coder-interp-pilot"),
        wandb_entity=os.environ.get("WANDB_ENTITY"),
    )


class VPDDecomposition(nn.Module):
    """Rank-1 decomposition of a single weight matrix + causal-importance MLP.

    W is treated as W in R^{d_out, d_in}. Decomposition:
        W ~ U V^T,  U in R^{d_out, C},  V in R^{d_in, C}

    Causal importance g(x) in [0,1]^C is a 2-layer MLP on x in R^{d_in}.
    """

    def __init__(
        self,
        W: torch.Tensor,
        num_components: int,
        importance_hidden: int,
    ):
        super().__init__()
        d_out, d_in = W.shape
        self.d_out = d_out
        self.d_in = d_in
        self.C = num_components

        # Warm-start U and V from the top-C SVD of W.
        with torch.no_grad():
            W_f = W.float()
            U_svd, S, Vh = torch.linalg.svd(W_f, full_matrices=False)
            C_eff = min(num_components, S.shape[0])
            sqrt_s = S[:C_eff].sqrt()
            U_init = U_svd[:, :C_eff] * sqrt_s.unsqueeze(0)  # (d_out, C_eff)
            V_init = Vh[:C_eff, :].T * sqrt_s.unsqueeze(0)   # (d_in, C_eff)
            if C_eff < num_components:
                pad_U = torch.zeros(d_out, num_components - C_eff)
                pad_V = torch.zeros(d_in, num_components - C_eff)
                U_init = torch.cat([U_init, pad_U], dim=1)
                V_init = torch.cat([V_init, pad_V], dim=1)

        self.U = nn.Parameter(U_init)
        self.V = nn.Parameter(V_init)

        self.importance = nn.Sequential(
            nn.Linear(d_in, importance_hidden),
            nn.GELU(),
            nn.Linear(importance_hidden, num_components),
        )

    def gates(self, x: torch.Tensor) -> torch.Tensor:
        """Returns g(x) in [0,1]^{B, C}."""
        return torch.sigmoid(self.importance(x))

    def forward(self, x: torch.Tensor, g: torch.Tensor | None = None):
        """y_hat = sum_c g_c(x) * U[:,c] * (V[:,c] . x).

        x: (B, d_in). Returns y_hat: (B, d_out).
        """
        if g is None:
            g = self.gates(x)
        proj = x @ self.V          # (B, C)
        scaled = proj * g          # (B, C)
        y_hat = scaled @ self.U.T  # (B, d_out)
        return y_hat, g

    def decomp_residual(self, W_target: torch.Tensor) -> torch.Tensor:
        """||W - U V^T||^2_F."""
        approx = self.U @ self.V.T
        return F.mse_loss(approx, W_target)


def resolve_target(model: nn.Module, layer: int, path: str) -> tuple[nn.Linear, str]:
    """Get the nn.Linear at model.model.layers[layer].<path>."""
    block = model.model.layers[layer]
    obj: nn.Module = block
    for part in path.split("."):
        obj = getattr(obj, part)
    if not isinstance(obj, nn.Linear):
        raise TypeError(f"target {path} is {type(obj)}, expected nn.Linear")
    return obj, f"model.layers[{layer}].{path}"


def stream_target_io(
    cfg: Cfg, model, tok, target: nn.Linear, ds_iter
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield (x, Wx) pairs at the target linear from streaming docs."""
    captured: dict[str, torch.Tensor] = {}

    def hook(_mod, inp, out):
        x = inp[0].detach()      # (B, T, d_in)
        captured["x"] = x
        captured["out"] = out.detach()

    handle = target.register_forward_hook(hook)
    try:
        while True:
            batch_texts: list[str] = []
            for _ in range(cfg.capture_batch):
                try:
                    sample = next(ds_iter)
                except StopIteration:
                    return
                text = (
                    sample.get("content")
                    or sample.get("text")
                    or sample.get("new_contents")
                )
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
                model(**enc, use_cache=False)
            x = captured["x"]
            y = captured["out"]
            mask = enc.attention_mask.bool().unsqueeze(-1)
            x_flat = x.masked_select(mask).view(-1, x.shape[-1])
            y_flat = y.masked_select(mask).view(-1, y.shape[-1])
            yield x_flat.float(), y_flat.float()
    finally:
        handle.remove()


def save_checkpoint(cfg: Cfg, vpd: VPDDecomposition, step: int) -> Path:
    cfg.workspace.mkdir(parents=True, exist_ok=True)
    ckpt = cfg.workspace / f"vpd_step_{step:06d}.pt"
    torch.save(
        {
            "step": step,
            "state_dict": vpd.state_dict(),
            "config": {
                "base_model": cfg.base_model,
                "layer": cfg.layer,
                "target_path": cfg.target_path,
                "num_components": cfg.num_components,
                "importance_hidden": cfg.importance_hidden,
                "d_in": vpd.d_in,
                "d_out": vpd.d_out,
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
    model.train(False)  # inference-mode (dropout off); we never train the base
    for p in model.parameters():
        p.requires_grad_(False)

    target, target_name = resolve_target(model, cfg.layer, cfg.target_path)
    W = target.weight.detach().float().cpu()  # (d_out, d_in)
    d_out, d_in = W.shape
    print(
        f"[target] {target_name} W shape={tuple(W.shape)} "
        f"(d_in={d_in}, d_out={d_out})",
        flush=True,
    )

    vpd = VPDDecomposition(
        W=W.cuda(),
        num_components=cfg.num_components,
        importance_hidden=cfg.importance_hidden,
    ).cuda()
    opt = torch.optim.Adam(vpd.parameters(), lr=cfg.lr)

    print(f"[ds] streaming {cfg.dataset_id}:{cfg.dataset_subset}", flush=True)
    ds = load_dataset(
        cfg.dataset_id,
        cfg.dataset_subset,
        streaming=True,
        split="train",
        trust_remote_code=True,
    )
    ds_iter = iter(ds.shuffle(seed=42, buffer_size=1000))

    wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=cfg.run_name,
        config={
            "base_model": cfg.base_model,
            "layer": cfg.layer,
            "target_path": cfg.target_path,
            "num_components": cfg.num_components,
            "importance_hidden": cfg.importance_hidden,
            "d_in": d_in,
            "d_out": d_out,
            "lr": cfg.lr,
            "batch_size": cfg.batch_size,
            "train_steps": cfg.train_steps,
            "lambda_recon": cfg.lambda_recon,
            "lambda_stoch": cfg.lambda_stoch,
            "lambda_importance": cfg.lambda_importance,
            "lambda_decomp": cfg.lambda_decomp,
        },
    )

    W_target = W.cuda()  # frozen reference for decomp_residual

    buf_x = torch.empty(0, d_in, device="cuda")
    buf_y = torch.empty(0, d_out, device="cuda")
    stream = stream_target_io(cfg, model, tok, target, ds_iter)
    last_log_t = time.time()
    step = 0
    while step < cfg.train_steps:
        while buf_x.shape[0] < cfg.batch_size:
            try:
                xn, yn = next(stream)
            except StopIteration:
                ds_iter = iter(ds.shuffle(seed=int(time.time()), buffer_size=1000))
                stream = stream_target_io(cfg, model, tok, target, ds_iter)
                continue
            buf_x = torch.cat([buf_x, xn.cuda()], dim=0)
            buf_y = torch.cat([buf_y, yn.cuda()], dim=0)

        x_batch = buf_x[: cfg.batch_size]
        y_batch = buf_y[: cfg.batch_size]
        buf_x = buf_x[cfg.batch_size :]
        buf_y = buf_y[cfg.batch_size :]

        # Standard recon with soft gates.
        y_hat, g = vpd(x_batch)
        loss_recon = F.mse_loss(y_hat, y_batch)

        # Stochastic recon: hard Bernoulli mask from g (straight-through).
        with torch.no_grad():
            mask = torch.bernoulli(g)
        g_st = mask + (g - g.detach())  # straight-through
        y_hat_stoch, _ = vpd(x_batch, g=g_st)
        loss_stoch = F.mse_loss(y_hat_stoch, y_batch)

        loss_imp = g.mean()
        loss_decomp = vpd.decomp_residual(W_target)

        loss = (
            cfg.lambda_recon * loss_recon
            + cfg.lambda_stoch * loss_stoch
            + cfg.lambda_importance * loss_imp
            + cfg.lambda_decomp * loss_decomp
        )

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % cfg.log_interval == 0:
            with torch.no_grad():
                tgt_var = y_batch.var()
                res_var = (y_batch - y_hat).var()
                fve = (1 - res_var / max(tgt_var.item(), 1e-8)).item()
                mean_g = g.mean().item()
                alive = (g.mean(dim=0) > 0.01).sum().item()
            wandb.log(
                {
                    "vpd/recon_mse": float(loss_recon.item()),
                    "vpd/stoch_mse": float(loss_stoch.item()),
                    "vpd/importance_mean": float(mean_g),
                    "vpd/decomp_residual": float(loss_decomp.item()),
                    "vpd/total_loss": float(loss.item()),
                    "vpd/fraction_var_explained": float(fve),
                    "vpd/alive_components": int(alive),
                    "vpd/buffer_size": int(buf_x.shape[0]),
                },
                step=step,
            )
            elapsed = time.time() - last_log_t
            print(
                f"[step {step:>5}/{cfg.train_steps}] "
                f"recon={loss_recon.item():.4f} stoch={loss_stoch.item():.4f} "
                f"g_mean={mean_g:.3f} alive={alive}/{cfg.num_components} "
                f"fve={fve:.3f} decomp={loss_decomp.item():.4f} "
                f"({elapsed:.1f}s/{cfg.log_interval} steps)",
                flush=True,
            )
            last_log_t = time.time()

        if (
            cfg.checkpoint_interval > 0
            and step > 0
            and step % cfg.checkpoint_interval == 0
        ):
            ckpt = save_checkpoint(cfg, vpd, step)
            print(f"[ckpt] {ckpt}", flush=True)
            artifact = wandb.Artifact(f"vpd-step-{step}", type="model")
            artifact.add_file(str(ckpt))
            wandb.log_artifact(artifact)

        step += 1

    final = save_checkpoint(cfg, vpd, step)
    print(f"[final] {final}", flush=True)
    artifact = wandb.Artifact("vpd-final", type="model")
    artifact.add_file(str(final))
    wandb.log_artifact(artifact)
    wandb.finish()
    print("[main] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
