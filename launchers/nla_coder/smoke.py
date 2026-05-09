"""Phase-0 smoke launcher for nla_qwen2_5_coder_1_5b.

Same pattern as launchers/nla_pilot/smoke.py — validates pod plumbing,
logs a fake training curve to W&B, exits 0.
"""

from __future__ import annotations

import os
import time


def _env(name: str, default: str | None = None) -> str:
    val = os.environ.get(name, default)
    if val is None:
        raise RuntimeError(f"required env var not set: {name}")
    return val


def main() -> int:
    base_model = _env("BASE_MODEL", "Qwen/Qwen2.5-Coder-1.5B")
    sae_layer = _env("SAE_LAYER", "6")
    sae_train_steps = int(_env("SAE_TRAIN_STEPS", "50"))
    av_train_steps = int(_env("AV_TRAIN_STEPS", "50"))
    eval_interval = int(_env("EVAL_INTERVAL", "25"))
    run_name = _env("WANDB_RUN_NAME", f"coder-smoke-{int(time.time())}")

    print("[smoke] config:")
    print(f"  base_model={base_model}")
    print(f"  sae_layer={sae_layer}")
    print(f"  sae_train_steps={sae_train_steps}")
    print(f"  av_train_steps={av_train_steps}")
    print(f"  eval_interval={eval_interval}")
    print(f"  run_name={run_name}")

    try:
        import wandb
    except ImportError:
        print("[smoke] wandb not installed; skipping W&B logging")
        wandb = None

    if wandb is not None:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "coder-interp-pilot"),
            entity=os.environ.get("WANDB_ENTITY"),
            name=run_name,
            config={
                "phase": 0,
                "variant": "smoke",
                "base_model": base_model,
                "sae_layer": int(sae_layer),
            },
        )

    total_steps = sae_train_steps + av_train_steps
    for step in range(total_steps):
        fake_loss = 1.0 / (1.0 + step * 0.04)
        bucket = "sae" if step < sae_train_steps else "av"
        if wandb is not None:
            wandb.log({f"train/{bucket}_loss": fake_loss}, step=step)
        if step % eval_interval == 0:
            if wandb is not None:
                wandb.log(
                    {f"val/{bucket}_proxy_metric": 0.30 + 0.01 * step},
                    step=step,
                )
            print(f"[smoke] step={step} bucket={bucket} loss={fake_loss:.4f}")
        time.sleep(0.05)

    if wandb is not None:
        wandb.finish()

    print("[smoke] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
