"""Real NLA pilot launcher.

End-to-end pipeline that produces actual natural-language descriptions
of Qwen2.5-7B-Instruct activation vectors via the kitft NLA Activation
Verbalizer at layer 20.

Pipeline:
  1. Define a small prompt set (NUM_PROMPTS, POSITIONS_PER_PROMPT).
  2. Load Qwen2.5-7B-Instruct, forward each prompt, capture
     hidden_states[NLA_LAYER] at chosen token positions. Free base.
  3. Save (prompt_idx, position, token_text, vector) tuples to parquet.
  4. Clone the kitft NLA repo. Start sglang serving the AV checkpoint.
  5. Wait for sglang readiness (poll /get_model_info).
  6. Run kitft's nla_inference.py against the parquet, capture stdout.
  7. Parse output (descriptions separated by lines of dashes) and align
     with the original (prompt, position, token) tuples.
  8. Log a wandb.Table with (idx, prompt, position, token, description).
  9. Tear down sglang.
"""

from __future__ import annotations

import gc
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _env(name: str, default: str | None = None) -> str:
    val = os.environ.get(name, default)
    if val is None:
        raise RuntimeError(f"required env var not set: {name}")
    return val


@dataclass
class Cfg:
    base_model: str
    av_repo: str
    ar_repo: str
    layer: int
    num_prompts: int
    positions_per_prompt: int
    sglang_port: int
    sglang_boot_timeout_s: int
    max_new_tokens: int
    temperature: float
    workspace: Path
    run_name: str
    wandb_project: str
    wandb_entity: Optional[str]


def load_cfg() -> Cfg:
    return Cfg(
        base_model=_env("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        av_repo=_env("AV_REPO", "kitft/nla-qwen2.5-7b-L20-av"),
        ar_repo=_env("AR_REPO", "kitft/nla-qwen2.5-7b-L20-ar"),
        layer=int(_env("NLA_LAYER", "20")),
        num_prompts=int(_env("NUM_PROMPTS", "3")),
        positions_per_prompt=int(_env("POSITIONS_PER_PROMPT", "1")),
        sglang_port=int(_env("SGLANG_PORT", "30000")),
        sglang_boot_timeout_s=int(_env("SGLANG_BOOT_TIMEOUT_S", "900")),
        max_new_tokens=int(_env("MAX_NEW_TOKENS", "120")),
        temperature=float(_env("TEMPERATURE", "0.5")),
        workspace=Path("/workspace/project"),
        run_name=_env("WANDB_RUN_NAME", f"nla-real-{int(time.time())}"),
        wandb_project=_env("WANDB_PROJECT", "coder-interp-pilot"),
        wandb_entity=os.environ.get("WANDB_ENTITY"),
    )


PROMPTS = [
    "def tokenize_python(source: str) -> list[str]:\n    \"\"\"Tokenize Python source code into lexical tokens.\"\"\"",
    "The prime factorization of 360 is 2^3 * 3^2 * 5.",
    "Climate models predict that average global temperatures will rise",
    "import torch\nimport torch.nn as nn\n\nclass MultiHeadAttention(nn.Module):",
    "The chief difficulty Alice found at first was managing her flamingo:",
    "SELECT customer_id, SUM(total) FROM orders WHERE status = 'shipped'",
    "Theorem: every continuous function on a compact set attains its maximum.",
    "func main() {\n    fmt.Println(\"hello world\")\n}",
    "She ran her hand along the smooth weathered driftwood and remembered",
    "The gradient of f(x, y) = x^2 + y^2 with respect to x is",
    "async function fetchUserData(id) {\n    const res = await fetch(`/users/${id}`)",
    "In quantum mechanics, observables are represented by Hermitian operators.",
    "The judge ruled that the contract was unenforceable because the consideration",
    "let mut counter: u64 = 0;\nfor _ in 0..1000 {\n    counter += 1;",
    "Photosynthesis converts carbon dioxide and water into glucose and oxygen using",
    "// Apply LoRA adapter weights to the base model's attention projection",
    "If x + 2y = 7 and 3x - y = 5, then x equals",
    "The melody descends a perfect fifth and resolves on the tonic.",
    "git rebase --interactive HEAD~5 lets you reorder, squash, or drop",
    "When boundary conditions are periodic, the discrete Fourier transform diagonalizes",
]


def capture_activations(cfg: Cfg) -> Path:
    """Forward prompts through the base model, dump (prompt_idx, position,
    token_text, vector) rows to parquet."""

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import pyarrow as pa
    import pyarrow.parquet as pq

    print(f"[capture] loading base model {cfg.base_model} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )

    chosen_prompts = PROMPTS[: cfg.num_prompts]
    rows_prompt_idx: list[int] = []
    rows_position: list[int] = []
    rows_token: list[str] = []
    rows_prompt: list[str] = []
    rows_vec: list[list[float]] = []

    with torch.inference_mode():
        for p_idx, prompt in enumerate(chosen_prompts):
            ids = tok(prompt, return_tensors="pt").to("cuda")
            out = model(**ids, output_hidden_states=True)
            hs = out.hidden_states[cfg.layer][0]  # (seq_len, d)
            seq_len = hs.shape[0]

            # Pick `positions_per_prompt` positions, biased toward the back
            # so we avoid the BOS / very-early-token region.
            if cfg.positions_per_prompt <= 1 or seq_len <= 2:
                positions = [seq_len - 1]
            else:
                n = min(cfg.positions_per_prompt, seq_len - 1)
                start = max(seq_len // 2, 1)
                stop = seq_len - 1
                step = max((stop - start) // max(n - 1, 1), 1)
                positions = sorted({start + i * step for i in range(n)} | {stop})
                positions = [p for p in positions if 0 <= p < seq_len][:n]

            ids_list = ids.input_ids[0].tolist()
            for pos in positions:
                tok_text = tok.decode([ids_list[pos]])
                vec = hs[pos].float().cpu().tolist()
                rows_prompt_idx.append(p_idx)
                rows_position.append(pos)
                rows_token.append(tok_text)
                rows_prompt.append(prompt)
                rows_vec.append(vec)

            print(
                f"[capture] prompt={p_idx} seq_len={seq_len} positions={positions}",
                flush=True,
            )

    cfg.workspace.mkdir(parents=True, exist_ok=True)
    parquet_path = cfg.workspace / "activations.parquet"

    table = pa.table(
        {
            "activation_vector": rows_vec,
            "prompt_idx": rows_prompt_idx,
            "position": rows_position,
            "token_text": rows_token,
            "prompt": rows_prompt,
        }
    )
    pq.write_table(table, parquet_path)
    print(f"[capture] wrote {len(rows_vec)} rows to {parquet_path}", flush=True)

    del model
    del tok
    gc.collect()
    torch.cuda.empty_cache()

    return parquet_path


def clone_kitft_repo(cfg: Cfg) -> Path:
    repo_dir = cfg.workspace / "nla_repo"
    if repo_dir.exists() and (repo_dir / "nla_inference.py").exists():
        print(f"[clone] kitft repo already present at {repo_dir}", flush=True)
        return repo_dir
    print(f"[clone] cloning kitft repo → {repo_dir}", flush=True)
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/kitft/natural_language_autoencoders",
            str(repo_dir),
        ],
        check=True,
    )
    return repo_dir


def start_sglang(cfg: Cfg) -> subprocess.Popen:
    print(
        f"[sglang] starting on port {cfg.sglang_port} for {cfg.av_repo}",
        flush=True,
    )
    log_path = cfg.workspace / "sglang.log"
    log_fp = open(log_path, "wb")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            cfg.av_repo,
            "--port",
            str(cfg.sglang_port),
            "--tp",
            "1",
            "--mem-fraction-static",
            "0.85",
        ],
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    print(f"[sglang] pid={proc.pid}; log={log_path}", flush=True)
    return proc


def wait_sglang_ready(cfg: Cfg, proc: subprocess.Popen) -> None:
    url = f"http://127.0.0.1:{cfg.sglang_port}/get_model_info"
    deadline = time.time() + cfg.sglang_boot_timeout_s
    last_status_print = 0.0
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"sglang exited with code {proc.returncode}; see "
                f"{cfg.workspace / 'sglang.log'}"
            )
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    print("[sglang] ready", flush=True)
                    return
        except Exception:
            pass
        if time.time() - last_status_print > 30:
            elapsed = int(time.time() - (deadline - cfg.sglang_boot_timeout_s))
            print(f"[sglang] still booting ... t={elapsed}s", flush=True)
            last_status_print = time.time()
        time.sleep(3)
    raise TimeoutError(
        f"sglang did not become ready within {cfg.sglang_boot_timeout_s}s"
    )


def run_nla_inference(cfg: Cfg, repo_dir: Path, parquet_path: Path) -> str:
    n = cfg.num_prompts * cfg.positions_per_prompt
    cmd = [
        sys.executable,
        str(repo_dir / "nla_inference.py"),
        cfg.av_repo,
        "--sglang-url",
        f"http://127.0.0.1:{cfg.sglang_port}",
        "--parquet",
        str(parquet_path),
        "--n",
        str(n),
        "--max-new-tokens",
        str(cfg.max_new_tokens),
        "--temperature",
        str(cfg.temperature),
    ]
    print(f"[infer] running: {' '.join(cmd)}", flush=True)
    out = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
        cwd=str(repo_dir),
    )
    print(
        f"[infer] stdout {len(out.stdout)} chars; stderr {len(out.stderr)} chars",
        flush=True,
    )
    if out.stderr:
        print(f"[infer] stderr tail:\n{out.stderr[-2000:]}", flush=True)
    return out.stdout


def parse_descriptions(stdout: str, expected: int) -> list[str]:
    """nla_inference.py prints one description per vector, separated by
    lines of dashes."""
    chunks = [c.strip() for c in re.split(r"\n[-]{3,}\n?", stdout) if c.strip()]
    if not chunks:
        chunks = [c.strip() for c in stdout.split("\n\n") if c.strip()]
    if len(chunks) > expected:
        chunks = chunks[-expected:]
    while len(chunks) < expected:
        chunks.append("")
    return chunks


def log_to_wandb(cfg: Cfg, parquet_path: Path, descriptions: list[str], stdout: str) -> None:
    import wandb
    import pyarrow.parquet as pq

    df = pq.read_table(parquet_path).to_pandas()

    wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=cfg.run_name,
        config={
            "base_model": cfg.base_model,
            "av_repo": cfg.av_repo,
            "ar_repo": cfg.ar_repo,
            "layer": cfg.layer,
            "num_prompts": cfg.num_prompts,
            "positions_per_prompt": cfg.positions_per_prompt,
            "max_new_tokens": cfg.max_new_tokens,
            "temperature": cfg.temperature,
        },
    )

    columns = ["idx", "prompt_idx", "position", "token_text", "prompt", "av_description"]
    table = wandb.Table(columns=columns)
    for i, row in df.iterrows():
        desc = descriptions[i] if i < len(descriptions) else ""
        table.add_data(
            int(i),
            int(row["prompt_idx"]),
            int(row["position"]),
            str(row["token_text"]),
            str(row["prompt"]),
            desc,
        )
    wandb.log({"nla/verbalizations": table})
    wandb.log(
        {
            "nla/n_described": int(sum(1 for d in descriptions if d.strip())),
            "nla/n_total": int(len(descriptions)),
        }
    )
    raw_path = cfg.workspace / "nla_inference_stdout.txt"
    raw_path.write_text(stdout)
    artifact = wandb.Artifact("nla-inference-stdout", type="raw-output")
    artifact.add_file(str(raw_path))
    wandb.log_artifact(artifact)
    wandb.finish()


def main() -> int:
    cfg = load_cfg()
    print(f"[main] cfg: {cfg}", flush=True)

    parquet_path = capture_activations(cfg)
    repo_dir = clone_kitft_repo(cfg)

    sglang_proc = start_sglang(cfg)
    stdout = ""
    try:
        wait_sglang_ready(cfg, sglang_proc)
        stdout = run_nla_inference(cfg, repo_dir, parquet_path)
        descriptions = parse_descriptions(
            stdout, expected=cfg.num_prompts * cfg.positions_per_prompt
        )
        for i, d in enumerate(descriptions):
            preview = (d[:160] + "…") if len(d) > 160 else d
            print(f"[result] {i}: {preview}", flush=True)
        log_to_wandb(cfg, parquet_path, descriptions, stdout)
    finally:
        try:
            os.killpg(os.getpgid(sglang_proc.pid), signal.SIGTERM)
        except Exception:
            pass
        try:
            sglang_proc.wait(timeout=20)
        except Exception:
            pass

    print("[main] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
