"""Build a personal code-style corpus from the user's top GitHub repos.

Clones a fixed list of repos shallowly, walks for source files in a small
set of extensions, dedupes by sha256, and uploads the result as a HF
dataset.

Output dataset schema:
  - repo:    str   (org/name, e.g. "eren23/sfumato")
  - path:    str   (path within the repo)
  - lang:    str   ("py" | "ts" | "tsx" | "rs" | "go")
  - n_chars: int
  - sha256:  str   (content hash, also used for dedupe)
  - content: str   (UTF-8, truncated to MAX_FILE_CHARS)

Usage (local):
    HUGGINGFACE_HUB_TOKEN=$HF_TOKEN \\
    python3 scripts/build_eren_style_corpus.py \\
        --owner eren23 \\
        --target-hf-repo eren23/eren-code-style

Idempotent: deletes /tmp/eren-style-corpus before each run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

# Top-20 by disk size (filtered to source-bearing repos; TeX papers dropped).
DEFAULT_REPOS: list[str] = [
    "synapse",
    "non_linear_ai_chat",
    "tinyworlds",
    "codewm-paper-public",
    "visual_reps",
    "attocode",
    "sfumato",
    "erenmes",
    "openflipbook",
    "parameter-golf",
    "crucible",
    "attocodepy_swarmtester_9",
    "sqld",
    "geminilivehackathon",
    "open_geo_spy",
    "endlessexplore",
    "ryx",
    "cgol_diffusion",
    "crucible-community-tap",
    "coder-interp-tap",
]

LANG_BY_EXT = {
    ".py": "py",
    ".ts": "ts",
    ".tsx": "tsx",
    ".rs": "rs",
    ".go": "go",
}

# Path fragments that signal generated / vendored / lockfile content.
SKIP_FRAGMENTS = (
    "/node_modules/",
    "/.venv/",
    "/venv/",
    "/__pycache__/",
    "/.git/",
    "/target/",
    "/dist/",
    "/build/",
    "/.next/",
    "/.cache/",
    "/site-packages/",
    "/migrations/",
    ".min.js",
    ".min.css",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "Cargo.lock",
    "go.sum",
)

MAX_FILE_CHARS = 200_000  # ~50K tokens at 4 chars/token, plenty.
MIN_FILE_CHARS = 80       # skip stubs & one-liners


def shallow_clone(owner: str, repo: str, dest: Path) -> bool:
    if dest.exists():
        shutil.rmtree(dest)
    url = f"https://github.com/{owner}/{repo}.git"
    print(f"[clone] {owner}/{repo} -> {dest}", flush=True)
    res = subprocess.run(
        ["git", "clone", "--depth", "1", "--quiet", url, str(dest)],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        print(f"[clone] FAILED {owner}/{repo}: {res.stderr.strip()}", flush=True)
        return False
    return True


def walk_code_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        s = "/" + str(p.relative_to(root)).replace(os.sep, "/")
        if any(frag in s for frag in SKIP_FRAGMENTS):
            continue
        if p.suffix not in LANG_BY_EXT:
            continue
        yield p


def read_text(path: Path) -> str | None:
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in raw[:4096]:  # binary heuristic
        return None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    return text


def collect(owner: str, repos: list[str], work_dir: Path) -> list[dict]:
    work_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    seen: set[str] = set()
    for repo in repos:
        dest = work_dir / repo
        if not shallow_clone(owner, repo, dest):
            continue
        kept = 0
        for f in walk_code_files(dest):
            text = read_text(f)
            if text is None:
                continue
            n = len(text)
            if n < MIN_FILE_CHARS:
                continue
            if n > MAX_FILE_CHARS:
                text = text[:MAX_FILE_CHARS]
                n = len(text)
            sha = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
            if sha in seen:
                continue
            seen.add(sha)
            rel = str(f.relative_to(dest))
            records.append(
                {
                    "repo": f"{owner}/{repo}",
                    "path": rel,
                    "lang": LANG_BY_EXT[f.suffix],
                    "n_chars": n,
                    "sha256": sha,
                    "content": text,
                }
            )
            kept += 1
        print(f"[{repo}] kept {kept} files", flush=True)
    return records


def push_to_hub(records: list[dict], target_repo: str) -> None:
    from datasets import Dataset

    ds = Dataset.from_list(records)
    print(f"[hf] dataset rows={len(ds)} cols={ds.column_names}", flush=True)
    print(f"[hf] push -> {target_repo}", flush=True)
    ds.push_to_hub(target_repo, private=False)
    print("[hf] done", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner", default="eren23")
    ap.add_argument("--repos", nargs="*", default=DEFAULT_REPOS)
    ap.add_argument(
        "--target-hf-repo",
        default="eren23/eren-code-style",
        help="HF dataset repo to push to.",
    )
    ap.add_argument("--work-dir", default="/tmp/eren-style-corpus")
    ap.add_argument("--dry-run", action="store_true",
                    help="Collect and save locally; skip HF push.")
    ap.add_argument("--out-json", default=None,
                    help="Optional local JSONL path for inspection.")
    args = ap.parse_args()

    work_dir = Path(args.work_dir)
    records = collect(args.owner, args.repos, work_dir)

    if not records:
        print("[main] no records collected, aborting", flush=True)
        return 1

    total_chars = sum(r["n_chars"] for r in records)
    by_lang: dict[str, int] = {}
    for r in records:
        by_lang[r["lang"]] = by_lang.get(r["lang"], 0) + r["n_chars"]
    print(
        f"[main] total files={len(records)} total_chars={total_chars:,} "
        f"(~{total_chars // 4:,} approx tokens)",
        flush=True,
    )
    print(f"[main] chars by lang: {by_lang}", flush=True)

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"[main] wrote {out}", flush=True)

    if args.dry_run:
        print("[main] dry-run: skipping HF push", flush=True)
        return 0

    push_to_hub(records, args.target_hf_repo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
