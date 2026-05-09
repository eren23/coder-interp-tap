"""Real NLA pilot launcher (transformers-only, no sglang).

End-to-end pipeline that produces actual natural-language descriptions of
Qwen2.5-7B-Instruct activation vectors via the kitft Activation Verbalizer
at layer 20.

The injection protocol (per kitft's nla_meta.yaml):
  - prompt template wraps the ㈎ injection char inside <concept>...</concept>
  - tokenize the chat-templated prompt
  - find positions where token_id == injection_token_id (149705 for Qwen 7B)
    AND validate left/right neighbor IDs match the sidecar
  - in the embedding tensor produced by the AV's input embedding layer,
    REPLACE the embedding at the injection position with the activation
    vector L2-normalized and rescaled to injection_scale (150.0 for 7B)
  - call model.generate(inputs_embeds=..., attention_mask=...) and decode

Pipeline:
  1. Capture L20 hidden states from Qwen2.5-7B over NUM_PROMPTS prompts.
  2. Free base. Load AV checkpoint + read its nla_meta.yaml for injection
     params (`huggingface_hub.hf_hub_download`).
  3. For each captured activation: build chat-templated prompt, tokenize,
     inject normalized activation, generate, decode, extract <explanation>.
  4. Log W&B Table with (idx, prompt, position, token_text, description).
"""

from __future__ import annotations

import gc
import os
import re
import time
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
    layer: int
    num_prompts: int
    positions_per_prompt: int
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
        layer=int(_env("NLA_LAYER", "20")),
        num_prompts=int(_env("NUM_PROMPTS", "3")),
        positions_per_prompt=int(_env("POSITIONS_PER_PROMPT", "1")),
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
    token_text, vector) rows to parquet. Returns parquet path."""

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import pyarrow as pa
    import pyarrow.parquet as pq

    print(f"[capture] loading base {cfg.base_model} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )

    chosen = PROMPTS[: cfg.num_prompts]
    rows_idx, rows_pos, rows_tok, rows_prompt, rows_vec = [], [], [], [], []

    with torch.inference_mode():
        for p_idx, prompt in enumerate(chosen):
            ids = tok(prompt, return_tensors="pt").to("cuda")
            out = model(**ids, output_hidden_states=True)
            hs = out.hidden_states[cfg.layer][0]  # (seq_len, d)
            seq_len = hs.shape[0]

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
                rows_idx.append(p_idx)
                rows_pos.append(pos)
                rows_tok.append(tok_text)
                rows_prompt.append(prompt)
                rows_vec.append(vec)

            print(
                f"[capture] prompt={p_idx} seq_len={seq_len} positions={positions}",
                flush=True,
            )

    cfg.workspace.mkdir(parents=True, exist_ok=True)
    parquet_path = cfg.workspace / "activations.parquet"
    pq.write_table(
        pa.table(
            {
                "activation_vector": rows_vec,
                "prompt_idx": rows_idx,
                "position": rows_pos,
                "token_text": rows_tok,
                "prompt": rows_prompt,
            }
        ),
        parquet_path,
    )
    print(f"[capture] wrote {len(rows_vec)} rows to {parquet_path}", flush=True)

    del model, tok
    gc.collect()
    import torch as _torch  # noqa
    _torch.cuda.empty_cache()
    return parquet_path


def load_meta(av_repo: str) -> dict:
    """Download nla_meta.yaml from the AV's HF repo + parse."""
    from huggingface_hub import hf_hub_download
    import yaml

    print(f"[av] fetching nla_meta.yaml from {av_repo} ...", flush=True)
    p = hf_hub_download(repo_id=av_repo, filename="nla_meta.yaml")
    with open(p) as f:
        return yaml.safe_load(f)


def verbalize_all(cfg: Cfg, parquet_path: Path) -> list[str]:
    """Load AV; for each activation in parquet, run injection + generate."""

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    activation_vectors = table.column("activation_vector").to_pylist()

    meta = load_meta(cfg.av_repo)
    inj_char = meta["tokens"]["injection_char"]
    inj_id = int(meta["tokens"]["injection_token_id"])
    left_id = int(meta["tokens"]["injection_left_neighbor_id"])
    right_id = int(meta["tokens"]["injection_right_neighbor_id"])
    inj_scale = float(meta["extraction"]["injection_scale"])
    template = meta["prompt_templates"]["av"]
    print(
        f"[av] meta: char={inj_char!r} id={inj_id} L={left_id} R={right_id} scale={inj_scale}",
        flush=True,
    )

    print(f"[av] loading {cfg.av_repo} ...", flush=True)
    av_tok = AutoTokenizer.from_pretrained(cfg.av_repo)
    av = AutoModelForCausalLM.from_pretrained(
        cfg.av_repo,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )

    # Build the chat-templated prompt once. Since chat formatting and the
    # injection-char position are deterministic (don't depend on the actual
    # vector), we tokenize once and reuse.
    user_content = template.format(injection_char=inj_char)
    messages = [{"role": "user", "content": user_content}]
    chat_text = av_tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    enc = av_tok(chat_text, return_tensors="pt").to("cuda")
    input_ids = enc.input_ids[0]  # (T,)
    attn = enc.attention_mask.to("cuda")

    # Validate injection position(s).
    candidate_positions = (input_ids == inj_id).nonzero(as_tuple=True)[0].tolist()
    if not candidate_positions:
        raise RuntimeError(
            f"injection token id {inj_id} not in tokenized prompt; "
            f"first 30 tokens: {input_ids[:30].tolist()}"
        )
    valid = []
    for p in candidate_positions:
        if p == 0 or p == len(input_ids) - 1:
            continue
        if int(input_ids[p - 1]) == left_id and int(input_ids[p + 1]) == right_id:
            valid.append(p)
    if not valid:
        # fall back: take first occurrence; emit a loud warning
        valid = candidate_positions[:1]
        print(
            f"[av] warning: neighbor validation failed at all candidates; "
            f"using first candidate at pos={valid[0]}",
            flush=True,
        )
    inj_pos = valid[0]
    print(f"[av] inj_pos={inj_pos} (T={len(input_ids)})", flush=True)

    # Embed the prompt once. We'll mutate a copy each iteration for the
    # specific activation vector being verbalized.
    embed_layer = av.get_input_embeddings()
    base_embeds = embed_layer(input_ids.unsqueeze(0))  # (1, T, d)

    descriptions: list[str] = []

    pad_id = av_tok.pad_token_id or av_tok.eos_token_id

    with torch.inference_mode():
        for i, vec_list in enumerate(activation_vectors):
            vec = torch.tensor(vec_list, dtype=torch.float32, device="cuda")
            norm = vec.norm()
            if norm < 1e-6:
                print(f"[av] zero-norm vector at row {i}; skipping", flush=True)
                descriptions.append("")
                continue
            scaled = vec * (inj_scale / norm)

            embeds = base_embeds.clone()
            embeds[0, inj_pos] = scaled.to(embeds.dtype)

            gen = av.generate(
                inputs_embeds=embeds,
                attention_mask=attn,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=cfg.temperature > 0,
                temperature=cfg.temperature if cfg.temperature > 0 else 1.0,
                pad_token_id=pad_id,
            )
            # When inputs_embeds is supplied, generate returns ONLY the new
            # tokens, not the prompt — confirmed across HF transformers >=4.43.
            new_text = av_tok.decode(gen[0], skip_special_tokens=True)
            descriptions.append(new_text)
            preview = (new_text[:160] + "…") if len(new_text) > 160 else new_text
            print(f"[av] {i}: {preview}", flush=True)

    return descriptions


_EXP_RE = re.compile(r"<explanation>(.*?)</explanation>", re.DOTALL)


def extract_explanation(raw: str) -> str:
    m = _EXP_RE.search(raw)
    if m:
        return m.group(1).strip()
    return raw.strip()


def log_to_wandb(
    cfg: Cfg,
    parquet_path: Path,
    raw_outputs: list[str],
) -> None:
    import wandb
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    prompt_idx_col = table.column("prompt_idx").to_pylist()
    position_col = table.column("position").to_pylist()
    token_text_col = table.column("token_text").to_pylist()
    prompt_col = table.column("prompt").to_pylist()
    n_rows = table.num_rows

    explanations = [extract_explanation(r) for r in raw_outputs]

    wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=cfg.run_name,
        config={
            "base_model": cfg.base_model,
            "av_repo": cfg.av_repo,
            "layer": cfg.layer,
            "num_prompts": cfg.num_prompts,
            "positions_per_prompt": cfg.positions_per_prompt,
            "max_new_tokens": cfg.max_new_tokens,
            "temperature": cfg.temperature,
        },
    )

    columns = [
        "idx",
        "prompt_idx",
        "position",
        "token_text",
        "prompt",
        "av_explanation",
        "av_raw",
    ]
    wb_table = wandb.Table(columns=columns)
    for i in range(n_rows):
        raw = raw_outputs[i] if i < len(raw_outputs) else ""
        explanation = explanations[i] if i < len(explanations) else ""
        wb_table.add_data(
            int(i),
            int(prompt_idx_col[i]),
            int(position_col[i]),
            str(token_text_col[i]),
            str(prompt_col[i]),
            explanation,
            raw,
        )
    wandb.log({"nla/verbalizations": wb_table})
    wandb.log(
        {
            "nla/n_described": int(sum(1 for e in explanations if e.strip())),
            "nla/n_total": int(len(explanations)),
        }
    )

    raw_path = cfg.workspace / "nla_raw_outputs.txt"
    raw_path.write_text("\n----\n".join(raw_outputs))
    artifact = wandb.Artifact("nla-raw-outputs", type="raw-output")
    artifact.add_file(str(raw_path))
    wandb.log_artifact(artifact)
    wandb.finish()


def main() -> int:
    cfg = load_cfg()
    print(f"[main] cfg: {cfg}", flush=True)

    parquet_path = capture_activations(cfg)
    raw_outputs = verbalize_all(cfg, parquet_path)
    log_to_wandb(cfg, parquet_path, raw_outputs)

    print("[main] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
