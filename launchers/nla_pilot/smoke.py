"""Phase-0 smoke launcher for nla_qwen3_5_2b_pilot.

Validates pod plumbing:
  - reads project env vars
  - logs to W&B
  - sleeps briefly
  - exits 0

NO real training. Replace with the actual Phase-1 pilot loop once the
project YAML's `pilot` variant becomes the focus.
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
    base_model = _env("BASE_MODEL", "Qwen/Qwen3.5-2B")
    sae_repo = _env("SAE_REPO", "Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50")
    sae_layer = _env("SAE_LAYER", "12")
    num_residuals = int(_env("NUM_RESIDUALS", "1000"))
    av_train_steps = int(_env("AV_TRAIN_STEPS", "50"))
    eval_interval = int(_env("EVAL_INTERVAL", "25"))

    run_name = _env("WANDB_RUN_NAME", f"smoke-{int(time.time())}")

    print("[smoke] config:")
    print(f"  base_model={base_model}")
    print(f"  sae_repo={sae_repo}")
    print(f"  sae_layer={sae_layer}")
    print(f"  num_residuals={num_residuals}")
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
                "sae_repo": sae_repo,
                "sae_layer": int(sae_layer),
                "num_residuals": num_residuals,
                "av_train_steps": av_train_steps,
            },
        )

    for step in range(av_train_steps):
        fake_loss = 1.0 / (1.0 + step * 0.05)
        if wandb is not None:
            wandb.log({"train/loss": fake_loss}, step=step)
        if step % eval_interval == 0:
            if wandb is not None:
                wandb.log(
                    {
                        "val/distinctiveness": 0.42 + 0.01 * step,
                        "val/round_trip_cosine": 0.30 + 0.01 * step,
                    },
                    step=step,
                )
            print(f"[smoke] step={step} loss={fake_loss:.4f}")
        time.sleep(0.05)

    if wandb is not None:
        wandb.finish()

    print("[smoke] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
