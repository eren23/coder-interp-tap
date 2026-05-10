"""LLM-explains-delta postprocess for feature_diff_study.

Compresses the top-K drift table into a 2-3 paragraph natural-language
summary of what the LoRA fine-tune actually did. Same OpenAI-compatible
endpoint pattern as feature_explainer (default DeepSeek V3 via OpenRouter).

Drop-in: imported by main.py at end of run, attached to wandb.run.notes
and logged as a wandb summary string.

See docs/llm-delta-summary.md for the design rationale.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import httpx


@dataclass
class DiffRow:
    feature_idx: int
    log2_ratio: float
    rate_baseline: float
    rate_tuned: float
    description: str


@dataclass
class SummarizerCfg:
    api_base: str
    api_key: str
    model: str
    max_tokens: int
    temperature: float
    top_n_each_direction: int
    timeout_s: float


def _env(key: str, default: str | None = None) -> str:
    val = os.environ.get(key, default)
    if val is None:
        raise KeyError(f"required env var {key} is unset")
    return val


def _load_dotenv_fallback() -> None:
    """Mirror feature_explainer/main.py: read /workspace/project/.env.runpod.local."""
    if os.environ.get("OPENROUTER_API_KEY") or os.environ.get("LLM_API_KEY"):
        return
    candidates = [
        Path("/workspace/project/.env.runpod.local"),
        Path.home() / ".env.runpod.local",
    ]
    for p in candidates:
        if not p.is_file():
            continue
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v
        if os.environ.get("OPENROUTER_API_KEY") or os.environ.get("LLM_API_KEY"):
            return


def load_cfg() -> SummarizerCfg | None:
    _load_dotenv_fallback()
    api_key = (
        os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("LLM_API_KEY")
        or ""
    )
    if not api_key:
        return None
    return SummarizerCfg(
        api_base=os.environ.get("LLM_API_BASE", "https://openrouter.ai/api/v1"),
        api_key=api_key,
        model=os.environ.get("LLM_MODEL", "deepseek/deepseek-chat"),
        max_tokens=int(os.environ.get("DELTA_SUMMARY_MAX_TOKENS", "600")),
        temperature=float(os.environ.get("DELTA_SUMMARY_TEMPERATURE", "0.3")),
        top_n_each_direction=int(os.environ.get("DELTA_SUMMARY_TOP_N", "30")),
        timeout_s=float(os.environ.get("DELTA_SUMMARY_TIMEOUT_S", "120")),
    )


def build_prompt(rows_up: list[DiffRow], rows_down: list[DiffRow]) -> str:
    def fmt_row(r: DiffRow) -> str:
        sign = "+" if r.log2_ratio > 0 else ""
        desc = (r.description or "<no description>").replace("\n", " ")[:160]
        return f"  F#{r.feature_idx:<5} {sign}{r.log2_ratio:+.2f}  rb={r.rate_baseline:.4f} rt={r.rate_tuned:.4f}  \"{desc}\""

    lines = [
        "You are analyzing what a LoRA fine-tune did to a code language model.",
        "Below is a list of \"features\" — each feature is a recurring concept",
        "the base model already encoded. We measured the firing-rate change",
        "after fine-tuning. log2_ratio is log2(tuned_firing_rate / baseline_firing_rate).",
        "UP rows fire MORE in the tuned model. DOWN rows fire LESS.",
        "",
        f"UP after tune (top {len(rows_up)} by |log2_ratio|):",
    ]
    lines.extend(fmt_row(r) for r in rows_up)
    lines.append("")
    lines.append(f"DOWN after tune (top {len(rows_down)} by |log2_ratio|):")
    lines.extend(fmt_row(r) for r in rows_down)
    lines.append("")
    lines.append(
        "In 2-3 short paragraphs:\n"
        "1. What style/domain/skill is this LoRA biasing TOWARD?\n"
        "2. What is it biasing AWAY from?\n"
        "3. Anything surprising — features that don't fit the obvious story?\n"
        "\n"
        "Be specific. Reference feature names. No hedging. No bullet points "
        "in the output — write prose paragraphs."
    )
    return "\n".join(lines)


def split_top_rows(
    rows: list[DiffRow],
    n_each: int,
) -> tuple[list[DiffRow], list[DiffRow]]:
    rows_sorted = sorted(rows, key=lambda r: r.log2_ratio, reverse=True)
    rows_up = [r for r in rows_sorted if r.log2_ratio > 0][:n_each]
    rows_down = list(reversed([r for r in rows_sorted if r.log2_ratio < 0]))[:n_each]
    return rows_up, rows_down


def summarize(rows: list[DiffRow], cfg: SummarizerCfg) -> str:
    rows_up, rows_down = split_top_rows(rows, cfg.top_n_each_direction)
    if not rows_up and not rows_down:
        return "<no drifted features to summarize>"
    prompt = build_prompt(rows_up, rows_down)
    payload = {
        "model": cfg.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": cfg.max_tokens,
        "temperature": cfg.temperature,
    }
    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "HTTP-Referer": "https://github.com/eren23/coder-interp-tap",
        "X-Title": "coder-interp-tap delta summarizer",
    }
    with httpx.Client(timeout=cfg.timeout_s) as client:
        resp = client.post(
            f"{cfg.api_base}/chat/completions",
            json=payload,
            headers=headers,
        )
    if resp.status_code != 200:
        return f"<llm error: HTTP {resp.status_code}: {resp.text[:300]}>"
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def summarize_or_warn(rows: list[DiffRow]) -> str | None:
    """Convenience wrapper: returns None if no API key, else the summary string."""
    cfg = load_cfg()
    if cfg is None:
        print(
            "[summarize] no OPENROUTER_API_KEY / LLM_API_KEY in env; skipping "
            "delta summary. Set the key on the pod to enable.",
            flush=True,
        )
        return None
    print(
        f"[summarize] requesting macro-bias summary via {cfg.model} @ {cfg.api_base}",
        flush=True,
    )
    summary = summarize(rows, cfg)
    print("[summarize] ----- delta summary -----", flush=True)
    print(summary, flush=True)
    print("[summarize] -------------------------", flush=True)
    return summary
