"""Read logs/<run_id>/metrics.jsonl and produce plots under logs/<run_id>/plots/.

Plots:
  - mlp_rms_vs_step.png    : per-layer + cross-layer aggregates (mean / RMSE / MAE) vs step
  - attn_rms_vs_step.png   : same shape for attention output
  - grad_norm_step_<S>.png : one bar chart per sampled step (x = layer, y = grad L2)
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_records(path: Path):
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def safe_array(values):
    """Convert list with None to (filtered_indices, filtered_values)."""
    idx = [i for i, v in enumerate(values) if v is not None]
    vals = [values[i] for i in idx]
    return idx, vals


def per_layer_lines(rows, key, num_layers):
    """Return (steps, [layer_series]). layer_series[i] is list of values aligned with steps (None for missing)."""
    steps = [r["step"] for r in rows]
    layer_series = [[None] * len(rows) for _ in range(num_layers)]
    for ri, r in enumerate(rows):
        vec = r.get(key) or []
        for li, v in enumerate(vec[:num_layers]):
            layer_series[li][ri] = v
    return steps, layer_series


def cross_layer_aggregates(rows, key):
    """For each step, compute mean / RMSE / MAE across non-null layers.
    RMSE here means sqrt(mean(x^2)) of the per-layer values; MAE means mean(|x|).
    Returns (steps, mean, rmse, mae)."""
    steps, mean_, rmse_, mae_ = [], [], [], []
    for r in rows:
        vec = [v for v in (r.get(key) or []) if v is not None]
        if not vec:
            continue
        steps.append(r["step"])
        n = len(vec)
        mean_.append(sum(vec) / n)
        rmse_.append(math.sqrt(sum(v * v for v in vec) / n))
        mae_.append(sum(abs(v) for v in vec) / n)
    return steps, mean_, rmse_, mae_


def plot_per_layer_and_aggregate(rows, key, title, ylabel, out_path, num_layers):
    steps, layer_series = per_layer_lines(rows, key, num_layers)
    agg_steps, mean_, rmse_, mae_ = cross_layer_aggregates(rows, key)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    cmap = plt.get_cmap("viridis")
    for li, series in enumerate(layer_series):
        xs = [steps[i] for i, v in enumerate(series) if v is not None]
        ys = [v for v in series if v is not None]
        if not xs:
            continue
        ax.plot(xs, ys, color=cmap(li / max(1, num_layers - 1)), label=f"L{li}", linewidth=1)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} — per layer")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if agg_steps:
        ax.plot(agg_steps, mean_, label="mean", linewidth=2)
        ax.plot(agg_steps, rmse_, label="RMSE (across layers)", linewidth=2)
        ax.plot(agg_steps, mae_, label="MAE (across layers)", linewidth=2)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} — cross-layer aggregates")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_grad_norm_per_step(rows, out_dir, num_layers):
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in rows:
        vec = r.get("grad_norm") or []
        idx, vals = safe_array(vec[:num_layers])
        if not idx:
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(idx, vals)
        ax.set_xlabel("layer index")
        ax.set_ylabel("grad L2 norm")
        ax.set_title(f"per-layer grad norm @ step {r['step']}")
        ax.set_xticks(list(range(num_layers)))
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(out_dir / f"grad_norm_step_{r['step']:06d}.png", dpi=120)
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="logs/<run_id> directory")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        print(f"no metrics file at {metrics_path}", file=sys.stderr)
        return 1

    rows = load_records(metrics_path)
    if not rows:
        print("metrics file is empty", file=sys.stderr)
        return 1

    num_layers = max(len(r.get("mlp_rms") or []) for r in rows)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_per_layer_and_aggregate(
        rows, "mlp_rms",
        title="FFN output RMS",
        ylabel="RMS",
        out_path=plots_dir / "mlp_rms_vs_step.png",
        num_layers=num_layers,
    )
    plot_per_layer_and_aggregate(
        rows, "attn_rms",
        title="Attention output RMS",
        ylabel="RMS",
        out_path=plots_dir / "attn_rms_vs_step.png",
        num_layers=num_layers,
    )

    plot_grad_norm_per_step(rows, plots_dir / "grad_norm", num_layers)

    print(f"wrote plots to {plots_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
