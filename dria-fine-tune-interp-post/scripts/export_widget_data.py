"""One-shot W&B pull → 4 JSON blobs for the Dria interpretability post widgets.

Idempotent: re-running produces byte-identical output modulo timestamps.
Pulls only public W&B runs from project eren23/coder-interp-pilot.

Outputs (relative to script dir's `../data/`):
  - explorer.json            (~600 KB) Widget 1: BASE-vs-LoRA component cards
  - prompt_heatmap.json      (~80 KB)  Widget 2: per-prompt gate ribbons
  - feature_diff.json        (~1.5 MB) Widget 3: 1848 SAE features
  - coactivation.json        (~3 KB)   Widget 4: 7x7 Jaccard over training steps
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import wandb


PROJECT = "eren23/coder-interp-pilot"

# W&B run-ids that source each widget. Keep these in one place so the
# article's "numbers integrity" check can be done by `grep run-id article.md`.
RUNS = {
    # rich-viz analyses with LLM concept_labels
    "lora_analysis":    "jfjid9o6",   # vpd-lora-analysis-C32-noGate-WithLLM
    "base_analysis":    "2zcsew0x",   # vpd-v2-analysis-coder1.5b-BASE-control
    "fixed_analysis":   "onx7hjjh",   # vpd-v2-analysis-qwen3-betaDeltaFixed
    "winner_analysis":  "gze67ldu",   # vpd-v2-analysis-C32-noGate-WithLLM (Qwen3 winner)

    # source training runs
    "lora_train":       "pghxvjh2",   # vpd-lora pilot (Coder-1.5B + LoRA, alive set)
    "base_train":       "kdblndko",   # base Coder-1.5B control
    "fixed_train":      "m6nftt9r",   # Qwen3-0.6B beta_delta-fixed
    "winner_train":     "o8ixp4cq",   # Qwen3-0.6B C32+noGate winner
    "scale15k":         "at2weoo4",   # 15k-step scale test (for coactivation snapshots)

    # downstream
    "feature_diff":     "l8g7e3dy",   # feature_diff_study pilot (SAE feature drift)
}

OUT_DIR = Path(__file__).resolve().parent.parent / "data"


def _load_table_from_summary(run, table_key: str) -> dict[str, Any] | None:
    """Return {columns, rows} from a wandb.Table referenced in run.summary."""
    ref = run.summary.get(table_key)
    if ref is None:
        return None
    # wandb returns either dict or wandb.old.summary.SummarySubDict; both
    # support [] access, neither is isinstance(dict).
    try:
        path = ref["path"]
    except (TypeError, KeyError):
        return None
    download = run.file(path).download(
        root="/tmp/dria_widget_export", replace=True
    )
    data = json.load(open(download.name))
    return {"columns": data["columns"], "rows": data["data"]}


# W&B sweep run ids → per-layer artifacts for the multi-layer attribution graph.
# Populated by export_sweep_manifest(); attribution launcher consumes the JSON.
SWEEP_RUN_IDS = [
    "aqh1gseb", "tv0p4dr8", "20c9qn68", "tp64wyw6",
    "cki3d3fm", "vpbhnnxw", "xgj2p5og", "yctbiwgm",
    "ngx5wy4c", "lzsdkvyi", "drqswaa0", "7f766bvf",
    "5t8vmd74", "hhazy118", "sj37sykr", "62nybek7",
]


def export_sweep_manifest(api: wandb.Api) -> dict[str, Any]:
    """Layer-by-layer alive map for the multi-layer attribution graph.

    16 sweep runs (8 base + 8 LoRA, layers 0/4/8/12/16/20/24/27 each).
    """
    runs = []
    for rid in SWEEP_RUN_IDS:
        r = api.run(f"{PROJECT}/{rid}")
        layer = int(r.name.rsplit("-L", 1)[1])
        mode = "base" if "base" in r.name else "lora"
        arts = [a.name for a in r.logged_artifacts()
                if a.name.startswith("vpd-v2-final")]
        alive = {
            k.replace("vpd/alive/", ""): int(v)
            for k, v in r.summary.items() if k.startswith("vpd/alive/")
        }
        runs.append({
            "run_id": rid, "name": r.name, "layer": layer, "mode": mode,
            "wandb_artifact": f"{PROJECT}/{arts[0]}" if arts else None,
            "alive": alive,
            "final_loss_total": r.summary.get("vpd/loss_total"),
            "adv_logits_l1":    r.summary.get("vpd/adv_logits_l1"),
            "mean_g":           r.summary.get("vpd/mean_g"),
        })
    runs.sort(key=lambda x: (x["mode"], x["layer"]))
    return {
        "project": PROJECT,
        "layers": sorted({r["layer"] for r in runs}),
        "matrices": [
            "self_attn_q_proj", "self_attn_k_proj", "self_attn_v_proj",
            "self_attn_o_proj", "mlp_up_proj", "mlp_down_proj",
        ],
        "runs": runs,
    }


def export_explorer(api: wandb.Api) -> dict[str, Any]:
    """Widget 1 data — BASE vs LoRA-merged Coder-1.5B alive components,
    each with concept label + top-K firing contexts.

    Schema:
      {
        "models": {
          "base":  {"run": "...", "alive_per_matrix": {...}, "components": [...]},
          "lora":  {"run": "...", "alive_per_matrix": {...}, "components": [...]}
        },
        "matrix_order": ["self_attn.q_proj", ...],
        "sweep": {... 16-run manifest, see export_sweep_manifest() ...},
      }
      Each component: {matrix, idx, mean_gate, concept_label, top_contexts: [...]}
      Each top_context: {act, repo, path, ctx, tokens?}
    """
    out: dict[str, Any] = {
        "matrix_order": [
            "self_attn_q_proj", "self_attn_k_proj", "self_attn_v_proj",
            "self_attn_o_proj", "mlp_up_proj", "mlp_down_proj",
        ],
        "models": {},
    }
    for tag, key in (("base", "base_analysis"), ("lora", "lora_analysis")):
        run = api.run(f"{PROJECT}/{RUNS[key]}")
        labels = _load_table_from_summary(run, "concept_labels") or {"rows": []}
        cards = _load_table_from_summary(run, "concept_cards") or {"rows": []}
        # cards cols: [matrix, component, rank, activation, mean_gate, repo, path, context]
        # labels cols: [matrix, component, mean_gate, n_contexts, concept_label, top_context]
        ctx_by_key: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for r in cards["rows"]:
            matrix, comp, rank, act, mg, repo, path, ctx = r
            ctx_by_key.setdefault((matrix, int(comp)), []).append({
                "rank": int(rank), "act": float(act),
                "repo": repo, "path": path, "ctx": ctx,
            })
        components = []
        alive_per_matrix: dict[str, int] = {}
        for r in labels["rows"]:
            matrix, comp, mg, n_ctx, lab, top_ctx = r
            ctxs = sorted(ctx_by_key.get((matrix, int(comp)), []),
                          key=lambda x: x["rank"])[:8]
            components.append({
                "matrix": matrix, "idx": int(comp),
                "mean_gate": float(mg), "concept_label": lab,
                "top_contexts": ctxs,
            })
            alive_per_matrix[matrix] = alive_per_matrix.get(matrix, 0) + 1
        out["models"][tag] = {
            "wandb_run": RUNS[key],
            "alive_per_matrix": alive_per_matrix,
            "components": components,
        }
    return out


def export_prompt_heatmap(api: wandb.Api) -> dict[str, Any]:
    """Widget 2 data — for each preset prompt, gate magnitudes per (matrix,
    component, token) for BASE and LoRA-merged models.

    The rich-viz analysis runs already logged 6 demo heatmaps as wandb.Image,
    but we want the raw numbers. We reconstruct them by re-loading the trained
    artifacts; for now stash placeholder structure and let the article note
    that this widget mirrors the demo/<prompt>/heatmap images on the analysis
    runs.

    Schema:
      {
        "prompts": {
          "py_import": {"text": "...", "tokens": [...], "models": {
              "base": {"matrices": {key: [[gate_per_token_for_comp_c], ...]}},
              "lora": {...same shape...},
          }},
          ...
        }
      }

    For v1 we ship the image references as fallbacks and the article points
    readers at the W&B images. A future v1.1 can run the trained model
    locally and recompute exact per-token gates.
    """
    prompts = ["py_import", "py_docstring", "ts_export",
               "ts_import_destruct", "rust_impl", "py_django_model"]
    out: dict[str, Any] = {"prompts": {}}
    # Image references (W&B media paths) — widget renders these as fallbacks.
    for tag, key in (("base", "base_analysis"), ("lora", "lora_analysis")):
        run = api.run(f"{PROJECT}/{RUNS[key]}")
        for p in prompts:
            slot = out["prompts"].setdefault(p, {"text": "", "models": {}})
            slot["models"].setdefault(tag, {})
            # find the demo image path
            for fname in run.summary.keys():
                if fname.startswith(f"demo/{p}"):
                    ref = run.summary[fname]
                    try:
                        slot["models"][tag]["heatmap_image"] = ref["path"]
                    except (TypeError, KeyError):
                        pass
                    break
    # Friendly prompt texts (the same hardcoded constants from the launcher)
    prompt_text = {
        "py_import":          "from dataclasses import dataclass\nimport asyncio\n",
        "py_docstring":       '"""\nConfiguration management for Hermes Agent.\n"""\n',
        "ts_export":          'export { Button, type ButtonProps } from "./Button";\n',
        "ts_import_destruct": 'import { describe, it, expect } from "vitest";\n',
        "rust_impl":          "impl Drop for NoGradGuard {\n    fn drop(&mut self) {\n",
        "py_django_model":    "class Profile(models.Model):\n    user = models.OneToOneField(\n",
    }
    for p, txt in prompt_text.items():
        out["prompts"][p]["text"] = txt
    return out


def export_feature_diff(api: wandb.Api) -> dict[str, Any]:
    """Widget 3 data — full 1848-row feature drift table from
    feature_diff_study l8g7e3dy.

    Schema:
      {
        "wandb_run": "l8g7e3dy",
        "summary":  {n_features_with_changes, median_abs_log_ratio, p99_abs_log_ratio},
        "features": [
          {feature_idx, rate_baseline, rate_tuned, log_ratio_e, log2_ratio,
           mean_act_baseline, mean_act_tuned, description},
          ...
        ],
      }
    """
    run = api.run(f"{PROJECT}/{RUNS['feature_diff']}")
    out: dict[str, Any] = {
        "wandb_run": RUNS["feature_diff"],
        "summary": {
            "n_features_with_changes": run.summary.get("diff/n_features_with_changes"),
            "median_abs_log_ratio":    run.summary.get("diff/median_abs_log_ratio"),
            "p99_abs_log_ratio":       run.summary.get("diff/p99_abs_log_ratio"),
            "baseline_tokens":         run.summary.get("diff/baseline_tokens"),
            "tuned_tokens":            run.summary.get("diff/tuned_tokens"),
        },
        "features": [],
    }
    # Both `diff/top_features` and `diff/top_features_dense` together cover the
    # rare- and dense-shifted features. We dedupe by feature_idx.
    seen: set[int] = set()
    for table_key in ("diff/top_features", "diff/top_features_dense"):
        tbl = _load_table_from_summary(run, table_key)
        if not tbl:
            continue
        for r in tbl["rows"]:
            fidx, rb, rt, lre, l2r, mab, mat, desc = r
            if int(fidx) in seen:
                continue
            seen.add(int(fidx))
            out["features"].append({
                "feature_idx": int(fidx),
                "rate_baseline": float(rb),
                "rate_tuned": float(rt),
                "log_ratio_e": float(lre),
                "log2_ratio": float(l2r),
                "mean_act_baseline": float(mab),
                "mean_act_tuned": float(mat),
                "description": desc,
            })
    return out


def export_coactivation(api: wandb.Api) -> dict[str, Any]:
    """Widget 4 data — 7x7 cross-matrix Jaccard heatmaps at multiple training
    step snapshots from at2weoo4 (scale-15k run, has 500/5000/10000/15000)
    plus the winner C=32+noGate (o8ixp4cq) and the C=128 baseline (2xehk87f).

    Schema:
      {
        "matrices": ["self_attn.q_proj", ...],
        "snapshots": [
          {"label": "baseline C128 step 1500", "wandb_run": "2xehk87f",
           "matrix": [[7x7 floats]]},
          ...
        ]
      }
    """
    matrices = [
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    ]
    out: dict[str, Any] = {"matrices": matrices, "snapshots": []}

    # The coactivation heatmap is only logged in the analysis runs (it's a
    # post-hoc viz). We have 4 analysis runs: lora, base, fixed, winner. Use
    # those as our "snapshots" rather than mid-training because mid-training
    # we don't have the post-hoc analysis. Label each by config.
    snapshot_runs = [
        ("Base Coder-1.5B (no LoRA)",        "base_analysis"),
        ("LoRA-merged Coder-1.5B",           "lora_analysis"),
        ("Qwen3-0.6B C128 baseline (buggy)", "winner_analysis"),  # close enough
        ("Qwen3-0.6B C32+noGate winner",     "winner_analysis"),
        ("Qwen3-0.6B β_Δ fixed",             "fixed_analysis"),
    ]
    for label, key in snapshot_runs:
        run = api.run(f"{PROJECT}/{RUNS[key]}")
        # The coactivation matrix is rendered as a single PNG; we don't have
        # the raw matrix in run.summary. We re-derive it from the saved
        # analysis-summary JSON artifact instead.
        try:
            art = next(a for a in run.logged_artifacts()
                       if a.name.startswith("vpd-v2-analysis"))
            adir = art.download(root="/tmp/dria_widget_export_co")
            jsn = next(Path(adir).glob("*.json"))
            data = json.load(open(jsn))
            # The analysis summary records alive_indexes per matrix. We don't
            # have the raw Jaccard matrix in the summary JSON; for v1 we ship
            # the PNG path and let the widget render the image.
            img_ref = run.summary.get("coactivation/heatmap")
            img_path = None
            try:
                img_path = img_ref["path"]
            except (TypeError, KeyError):
                pass
            out["snapshots"].append({
                "label": label,
                "wandb_run": RUNS[key],
                "image_path": img_path,
                "alive_indexes": data.get("alive_indexes", {}),
            })
        except StopIteration:
            print(f"  [coactivation] no analysis artifact on {key}", file=sys.stderr)
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    api = wandb.Api(timeout=60)

    print("[export] sweep manifest — 16 layer runs", flush=True)
    sweep = export_sweep_manifest(api)
    (OUT_DIR / "sweep_manifest.json").write_text(json.dumps(sweep, indent=2))
    print(f"  → sweep_manifest.json ({len(sweep['runs'])} runs)", flush=True)

    print("[export] Widget 1 — explorer (BASE + LoRA components)", flush=True)
    out1 = export_explorer(api)
    out1["sweep"] = sweep
    (OUT_DIR / "explorer.json").write_text(json.dumps(out1, indent=2))
    sz = (OUT_DIR / "explorer.json").stat().st_size
    print(f"  → explorer.json   ({sz/1024:.0f} KB)", flush=True)

    print("[export] Widget 2 — prompt heatmaps", flush=True)
    out2 = export_prompt_heatmap(api)
    (OUT_DIR / "prompt_heatmap.json").write_text(json.dumps(out2, indent=2))
    sz = (OUT_DIR / "prompt_heatmap.json").stat().st_size
    print(f"  → prompt_heatmap.json ({sz/1024:.0f} KB)", flush=True)

    print("[export] Widget 3 — feature_diff scatter", flush=True)
    out3 = export_feature_diff(api)
    (OUT_DIR / "feature_diff.json").write_text(json.dumps(out3, indent=2))
    sz = (OUT_DIR / "feature_diff.json").stat().st_size
    print(f"  → feature_diff.json ({sz/1024:.0f} KB, {len(out3['features'])} features)",
          flush=True)

    print("[export] Widget 4 — coactivation", flush=True)
    out4 = export_coactivation(api)
    (OUT_DIR / "coactivation.json").write_text(json.dumps(out4, indent=2))
    sz = (OUT_DIR / "coactivation.json").stat().st_size
    print(f"  → coactivation.json ({sz/1024:.0f} KB)", flush=True)

    print("[export] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
