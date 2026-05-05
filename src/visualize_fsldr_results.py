import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_jsonl(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSON on line {line_no}: {e}") from e
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", required=True, help="Path to JSONL results file")
    ap.add_argument("--output", "-o", required=True, help="Output PDF path (e.g. plots/ndcg_facets.pdf)")
    args = ap.parse_args()

    df = load_jsonl(args.input)

    # Normalize column name (your data uses token_treshold_upper (typo), keep support for either)
    if "token_treshold_upper" in df.columns and "token_threshold_upper" not in df.columns:
        df = df.rename(columns={"token_treshold_upper": "token_threshold_upper"})

    required = {"ndcg@10", "n_queries", "token_threshold_lower", "token_threshold_upper", "model", "language"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Coerce types
    df["ndcg@10"] = pd.to_numeric(df["ndcg@10"], errors="coerce")
    df["n_queries"] = pd.to_numeric(df["n_queries"], errors="coerce")
    df["token_threshold_lower"] = pd.to_numeric(df["token_threshold_lower"], errors="coerce")
    df["token_threshold_upper"] = pd.to_numeric(df["token_threshold_upper"], errors="coerce")

    # Fill "no upper value" with a value > 32000 (use 32768 by default)
    # (if you want "next bin edge", change this)
    UPPER_SENTINEL = 32768
    df["token_threshold_upper_filled"] = df["token_threshold_upper"].fillna(UPPER_SENTINEL)

    # X-axis is the (lower, upper) pair -> use mid-point as x coordinate
    df["x_mid"] = (df["token_threshold_lower"] + df["token_threshold_upper_filled"]) / 2.0

    # Nice label showing the pair; keep sentinel visible
    def fmt_pair(lo, up, up_is_nan):
        if up_is_nan:
            return f"[{int(lo)}, >32000]"
        return f"[{int(lo)}, {int(up)}]"

    df["bin_label"] = [
        fmt_pair(lo, up, pd.isna(up))
        for lo, up in zip(df["token_threshold_lower"], df["token_threshold_upper"])
    ]

    # Keep only fin/swe (as requested)
    df = df[df["language"].isin(["fin", "swe"])].copy()

    # Order x by numeric midpoint
    order = (
        df[["bin_label", "x_mid"]]
        .drop_duplicates()
        .sort_values("x_mid", ascending=True)["bin_label"]
        .tolist()
    )
    df["bin_label"] = pd.Categorical(df["bin_label"], categories=order, ordered=True)

    sns.set_theme(style="whitegrid", font_scale=1.0)

    # Facet plot: one facet per language
    g = sns.FacetGrid(
        df,
        col="language",
        col_order=["fin", "swe"],
        sharex=True,
        sharey=True,
        height=6.0,     # bigger panels
        aspect=1.6,     # wider panels
        margin_titles=True,
        despine=False,
    )

    def draw_panel(data, **kwargs):
        ax = plt.gca()

        sns.lineplot(
            data=data,
            x="bin_label",
            y="ndcg@10",
            hue="model",
            marker="o",
            linewidth=2.2,
            markersize=7,
            ax=ax,
            legend=False,
        )
    # One n_queries annotation per x-position (bin_label), not per model
        per_x = (
            data.groupby("bin_label", observed=True)
                .agg(n_queries=("n_queries", "first"),
                     y_max=("ndcg@10", "max"))
                .reset_index()
        )

        for _, r in per_x.iterrows():
            ax.annotate(
                f"n={int(r['n_queries'])}",
                (r["bin_label"], r["y_max"]),
                textcoords="offset points",
                xytext=(0, 8),          # put above highest point at that x
                ha="center",
                va="bottom",
                fontsize=8,
                color="black",
            )

        ax.set_xlabel("token thresholds [lower, upper]")
        ax.set_ylabel("ndcg@10")
        ax.tick_params(axis="x", rotation=35)

    g.map_dataframe(draw_panel)

    # Build a single legend (models) for the whole figure
    # Create handles by plotting invisible lines per model (stable ordering)
    models = df["model"].dropna().unique().tolist()
    palette = sns.color_palette(n_colors=len(models))
    handles = []
    labels = []
    for m, c in zip(models, palette):
        h, = g.axes.flat[0].plot([], [], color=c, marker="o", linewidth=1.0)
        handles.append(h)
        labels.append(m)

    g.fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=min(3, len(models)),
        frameon=True,
    )

    g.set_titles(col_template="{col_name}")

    g.fig.tight_layout()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    g.fig.savefig(out_path, format="png", bbox_inches="tight")
    plt.close(g.fig)


if __name__ == "__main__":
    main()