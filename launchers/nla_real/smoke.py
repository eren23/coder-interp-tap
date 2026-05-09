"""Smoke variant for nla_real_qwen2_5_7b — runs the same pipeline with
NUM_PROMPTS / POSITIONS_PER_PROMPT defaulting to 3 / 1 (set in the
project YAML's smoke variant). Just delegates to pilot.py."""

from __future__ import annotations

from launchers.nla_real.pilot import main


if __name__ == "__main__":
    raise SystemExit(main())
