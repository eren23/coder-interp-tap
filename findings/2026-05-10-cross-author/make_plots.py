"""Generate visualizations from the v1/v2/v3 cross-author delta study.

Pulls live W&B data for the 9 completed runs and produces 4 PNGs in this
directory. Real numbers, no synthesis — change the V1/V2/V3 dicts to point
to a new round of runs to regenerate.

Run:
    cd findings/2026-05-10-cross-author
    python3 make_plots.py
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import wandb

api = wandb.Api()

V1 = {"karpathy": "u0c5r9e8", "ggerganov": "6dlp85f5", "eren23": "2rgnerqg"}
V2 = {"karpathy": "yx9cegt8", "ggerganov": "51hwpvoh", "eren23": "6qxs724s"}
V3 = {"karpathy": "sh025de4", "ggerganov": "74p73nlb", "eren23": "83kopx5a"}


def fetch(version_map):
    out = {}
    for author, run_id in version_map.items():
        run = api.run(f"eren23/coder-interp-pilot/{run_id}")
        s = run.summary
        out[author] = {
            "median_log2": float(s["diff/median_abs_log_ratio"]),
            "n_changes": int(s["diff/n_features_with_changes"]),
            "p99": float(s["diff/p99_abs_log_ratio"]),
            "lora_loss": float(s["lora/loss"]),
        }
    return out


def grouped_bar(out_path, ylabel, key, title, values_v1, values_v2, values_v3, fmt="{:.3f}"):
    authors = list(values_v1.keys())
    x = np.arange(len(authors))
    w = 0.27
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w, [values_v1[a][key] for a in authors], w, label="v1: orig pipeline", color="#cccccc")
    ax.bar(x, [values_v2[a][key] for a in authors], w, label="v2: dense-rank table", color="#7faddb")
    ax.bar(x + w, [values_v3[a][key] for a in authors], w, label="v3: + license filter", color="#2b6cb0")
    ax.set_xticks(x)
    ax.set_xticklabels(authors)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    for i, a in enumerate(authors):
        for j, vmap in enumerate([values_v1, values_v2, values_v3]):
            ax.text(i + (j - 1) * w, vmap[a][key] + (max(vmap[a][key] for vmap in [values_v1, values_v2, values_v3] for a in authors) * 0.02),
                    fmt.format(vmap[a][key]), ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"saved {out_path}")


def log2_distribution_per_author(out_path, version_map, version_label):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, author in zip(axes, version_map):
        run = api.run(f"eren23/coder-interp-pilot/{version_map[author]}")
        tbl_file = next(
            f for f in run.files()
            if "top_features" in f.name and "dense" not in f.name and f.name.endswith(".table.json")
        )
        local = tbl_file.download(root=f"/tmp/v_{author}", replace=True)
        data = json.load(open(local.name))
        cols = data["columns"]
        li = cols.index("log2_ratio")
        log2s = np.array([r[li] for r in data["data"]])
        n_changes = int(run.summary["diff/n_features_with_changes"])
        ax.hist(log2s, bins=40, color="#2b6cb0", edgecolor="white")
        ax.axvline(0, color="black", linestyle="--", alpha=0.5)
        ax.set_title(f"{author} {version_label}\nn_features_changed={n_changes}")
        ax.set_xlabel("log2(rate_tuned / rate_baseline)")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("count of top-K features")
    plt.suptitle(
        f"Distribution of feature-firing log2-ratios per author ({version_label}, top 500 by |log2|)",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {out_path}")


def top_features_bar(out_path, run_id_a, run_id_b, label_a, label_b, n=10, author_label=""):
    def top_n_up(run_id):
        run = api.run(f"eren23/coder-interp-pilot/{run_id}")
        tbl_file = next(
            f for f in run.files()
            if "top_features" in f.name and "dense" not in f.name and f.name.endswith(".table.json")
        )
        local = tbl_file.download(root=f"/tmp/dl_{run_id}", replace=True)
        data = json.load(open(local.name))
        cols = data["columns"]
        li = cols.index("log2_ratio")
        fi = cols.index("feature_idx")
        ds = cols.index("description")
        sorted_rows = sorted(data["data"], key=lambda r: r[li], reverse=True)
        return [(int(r[fi]), float(r[li]), r[ds][:60]) for r in sorted_rows[:n]]

    a_top = top_n_up(run_id_a)
    b_top = top_n_up(run_id_b)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, top, label, color in [
        (axes[0], a_top, label_a, "#cccccc"),
        (axes[1], b_top, label_b, "#2b6cb0"),
    ]:
        feats = [f"F#{f[0]} {f[2]}" for f in top]
        log2s = [f[1] for f in top]
        y = np.arange(len(feats))
        ax.barh(y, log2s, color=color, edgecolor="black")
        ax.set_yticks(y)
        ax.set_yticklabels(feats, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("log2_ratio (higher = more amplified)")
        ax.set_title(f"{author_label} {label}\ntop-{n} UP features")
        ax.grid(axis="x", alpha=0.3)
    plt.suptitle(
        "Same author, same SAE, different ranking + bias filter → different top features",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {out_path}")


def main():
    v1 = fetch(V1)
    v2 = fetch(V2)
    v3 = fetch(V3)
    grouped_bar(
        "methodology_progression.png",
        "median |log2_ratio| of feature firing rates",
        "median_log2",
        "Personalization signal: median feature drift across 3 methodology versions\n"
        "(same OLD pilot SAE, same 3 authors)",
        v1, v2, v3,
    )
    grouped_bar(
        "lora_loss_progression.png",
        "final LoRA SFT loss (lower = better fit on bias data)",
        "lora_loss",
        "LoRA fit quality across methodology versions\n"
        "(license filter forces the LoRA to fit real code, not boilerplate)",
        v1, v2, v3,
        fmt="{:.2f}",
    )
    log2_distribution_per_author("log2_distribution_v3.png", V3, "v3")
    top_features_bar(
        "top_features_v1_vs_v3_karpathy.png",
        V1["karpathy"], V3["karpathy"],
        "v1 (original pipeline)", "v3 (dense-rank + license filter)",
        n=10, author_label="karpathy",
    )


if __name__ == "__main__":
    main()
