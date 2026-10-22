import csv
import math
from pathlib import Path

import numpy as np


def make_log_density_bins():
    return [0.0] + [10.0 ** exp for exp in range(-34, 3)] + [float("inf")]


LOG_DENSITY_BINS = make_log_density_bins()
SUMMARY_FIELDS = [
    "baseline",
    "family",
    "checkpoint",
    "num_samples",
    "num_voxels",
    "avg_input_construction_seconds",
    "std_input_construction_seconds",
    "total_input_construction_seconds",
    "avg_model_reconstruction_seconds",
    "std_model_reconstruction_seconds",
    "total_model_reconstruction_seconds",
    "avg_inference_seconds",
    "std_inference_seconds",
    "total_inference_seconds",
    "mse",
    "rmse",
    "mae",
    "pearson",
    "total_electron_error",
    "mae_total_electron_error",
    "rmse_total_electron_error",
    "status",
    "notes",
]
PER_SAMPLE_FIELDS = [
    "sample_id",
    "prediction",
    "input_construction_seconds",
    "model_reconstruction_seconds",
    "inference_seconds",
    "mse",
    "rmse",
    "mae",
    "pearson",
    "total_electron_error",
]


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(path, rows, fieldnames):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary_csv(path, row):
    write_csv(path, [row], SUMMARY_FIELDS)


def pearson_corr(pred, target):
    pred = pred.reshape(-1).astype(np.float64)
    target = target.reshape(-1).astype(np.float64)
    pred_std = pred.std()
    target_std = target.std()
    if pred_std == 0 or target_std == 0:
        return float("nan")
    return float(np.corrcoef(pred, target)[0, 1])


def density_bin_label(lo, hi):
    if lo == 0.0:
        return "1e-34~0"
    if math.isinf(hi):
        return "inf~1e2"
    return f"{hi:.0e}~{lo:.0e}"


def _safe_log_center(lo, hi):
    lo_eff = max(lo, 1e-34)
    hi_eff = max(hi, lo_eff * 10.0)
    return math.sqrt(lo_eff * hi_eff)


def write_error_distribution_plot(path, rows, title, xlim=(1e-34, 1e4), ylim=(0.0, 1.0)):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.ticker import FuncFormatter, LogFormatterMathtext, MaxNLocator

    finite_rows = []
    for row in rows:
        count = int(row["count"])
        bin_lower = float(row["bin_lower"])
        bin_upper = float(row["bin_upper"])
        probability = float(row["probability"])
        if count <= 0 or not np.isfinite(bin_upper):
            continue
        finite_rows.append(
            {
                "count": count,
                "bin_lower": bin_lower,
                "bin_upper": bin_upper,
                "probability": probability,
            }
        )
    if not finite_rows:
        return

    left_edges = np.asarray(
        [max(row["bin_lower"], 1e-34) for row in finite_rows],
        dtype=np.float64,
    )
    right_edges = np.asarray(
        [max(row["bin_upper"], 1e-34) for row in finite_rows],
        dtype=np.float64,
    )
    centers = np.asarray(
        [_safe_log_center(row["bin_lower"], row["bin_upper"]) for row in finite_rows],
        dtype=np.float64,
    )
    probs = np.asarray([row["probability"] for row in finite_rows], dtype=np.float64)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10.5,
            "axes.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig, ax = plt.subplots(figsize=(5.4, 5.4), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "custom_blue_grad",
        ["#C9DDF0", "#7EA6D8", "#3E6FA8"],
    )
    bar_colors = [cmap(i) for i in np.linspace(0.22, 0.95, len(left_edges))]

    line_color = "#A64B5A"
    grid_color = "#DCE2EA"
    spine_color = "#3F4650"
    tick_color = "#2F3640"

    ax.bar(
        left_edges,
        probs,
        width=right_edges - left_edges,
        align="edge",
        color=bar_colors,
        alpha=0.90,
        edgecolor="white",
        linewidth=0.9,
        label="Binned probability",
        zorder=2,
    )
    ax.plot(
        centers,
        probs,
        color=line_color,
        linewidth=2.0,
        marker="o",
        markersize=4.5,
        markerfacecolor="white",
        markeredgecolor=line_color,
        markeredgewidth=1.1,
        label="Trend",
        zorder=3,
    )

    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    ax.set_xlabel("Absolute Error", labelpad=6)
    ax.set_ylabel("Probability", labelpad=6)
    ax.set_title(title, pad=10, fontweight="semibold")

    ax.xaxis.set_major_formatter(LogFormatterMathtext())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.2f}"))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    ax.grid(
        True,
        which="major",
        axis="y",
        color=grid_color,
        linewidth=0.8,
        alpha=0.75,
        linestyle="-",
        zorder=0,
    )
    ax.grid(False, axis="x")

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(spine_color)
    ax.spines["bottom"].set_color(spine_color)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    ax.tick_params(
        axis="both",
        colors=tick_color,
        width=0.9,
        length=4,
        direction="out",
    )

    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index("Binned probability"), labels.index("Trend")]
    ax.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        frameon=False,
        loc="upper right",
        handlelength=2.2,
        borderaxespad=0.4,
        labelspacing=0.6,
    )

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


class SamplewiseMetricsAccumulator:
    def __init__(self, bins=None):
        self.bins = list(bins or LOG_DENSITY_BINS)
        self.n = 0
        self.num_samples = 0
        self.sum_pred = 0.0
        self.sum_target = 0.0
        self.sum_pred_sq = 0.0
        self.sum_target_sq = 0.0
        self.sum_pred_target = 0.0
        self.sum_abs = 0.0
        self.sum_sq = 0.0
        self.sample_rows = []
        self.sample_total_electron_errors = []
        self.sample_input_construction_seconds = []
        self.sample_model_reconstruction_seconds = []
        self.sample_inference_seconds = []
        self.max_abs_error = 0.0
        self.density_bin_stats = [
            {"count": 0, "sum_abs": 0.0, "sum_sq": 0.0}
            for _ in range(len(self.bins) - 1)
        ]
        self.error_hist_counts = np.zeros(len(self.bins) - 1, dtype=np.int64)

    def update_sample(
        self,
        pred,
        target,
        sample_id=None,
        prediction=None,
        voxel_volume=1.0,
        input_construction_seconds=None,
        model_reconstruction_seconds=None,
        inference_seconds=None,
    ):
        pred = np.asarray(pred, dtype=np.float64).reshape(-1)
        target = np.asarray(target, dtype=np.float64).reshape(-1)
        if pred.size == 0:
            return

        diff = pred - target
        abs_diff = np.abs(diff)
        sq_diff = diff ** 2
        n = int(target.size)

        self.num_samples += 1
        self.n += n
        self.sum_pred += float(pred.sum())
        self.sum_target += float(target.sum())
        self.sum_pred_sq += float((pred ** 2).sum())
        self.sum_target_sq += float((target ** 2).sum())
        self.sum_pred_target += float((pred * target).sum())
        self.sum_abs += float(abs_diff.sum())
        self.sum_sq += float(sq_diff.sum())
        self.max_abs_error = max(self.max_abs_error, float(abs_diff.max(initial=0.0)))

        total_electron_error = float(diff.sum() * voxel_volume)
        self.sample_total_electron_errors.append(total_electron_error)
        input_construction_seconds_value = (
            float(input_construction_seconds)
            if input_construction_seconds is not None and np.isfinite(input_construction_seconds)
            else np.nan
        )
        model_reconstruction_seconds_value = (
            float(model_reconstruction_seconds)
            if model_reconstruction_seconds is not None and np.isfinite(model_reconstruction_seconds)
            else np.nan
        )
        inference_seconds_value = (
            float(inference_seconds) if inference_seconds is not None and np.isfinite(inference_seconds) else np.nan
        )
        self.sample_input_construction_seconds.append(input_construction_seconds_value)
        self.sample_model_reconstruction_seconds.append(model_reconstruction_seconds_value)
        self.sample_inference_seconds.append(inference_seconds_value)
        self.sample_rows.append(
            {
                "sample_id": sample_id if sample_id is not None else "",
                "prediction": prediction if prediction is not None else "",
                "input_construction_seconds": input_construction_seconds_value,
                "model_reconstruction_seconds": model_reconstruction_seconds_value,
                "inference_seconds": inference_seconds_value,
                "mse": float(np.mean(sq_diff)),
                "rmse": float(np.sqrt(np.mean(sq_diff))),
                "mae": float(np.mean(abs_diff)),
                "pearson": pearson_corr(pred, target),
                "total_electron_error": total_electron_error,
            }
        )

        for idx, (lo, hi) in enumerate(zip(self.bins[:-1], self.bins[1:])):
            if math.isinf(hi):
                mask = target >= lo
            else:
                mask = (target >= lo) & (target < hi)
            count = int(np.count_nonzero(mask))
            if count == 0:
                continue
            stats = self.density_bin_stats[idx]
            stats["count"] += count
            stats["sum_abs"] += float(abs_diff[mask].sum())
            stats["sum_sq"] += float(sq_diff[mask].sum())

        error_bins = np.asarray(self.bins[:-1], dtype=np.float64)
        error_idx = np.searchsorted(error_bins, abs_diff, side="right") - 1
        error_idx = np.clip(error_idx, 0, len(self.error_hist_counts) - 1)
        counts = np.bincount(error_idx, minlength=len(self.error_hist_counts))
        self.error_hist_counts += counts.astype(np.int64)

    def summary(self, baseline, family, checkpoint="", status="ok", notes=""):
        if self.n == 0:
            return {
                "baseline": baseline,
                "family": family,
                "checkpoint": checkpoint,
                "num_samples": self.num_samples,
                "num_voxels": 0,
                "avg_input_construction_seconds": np.nan,
                "std_input_construction_seconds": np.nan,
                "total_input_construction_seconds": np.nan,
                "avg_model_reconstruction_seconds": np.nan,
                "std_model_reconstruction_seconds": np.nan,
                "total_model_reconstruction_seconds": np.nan,
                "avg_inference_seconds": np.nan,
                "std_inference_seconds": np.nan,
                "total_inference_seconds": np.nan,
                "mse": np.nan,
                "rmse": np.nan,
                "mae": np.nan,
                "pearson": np.nan,
                "total_electron_error": np.nan,
                "mae_total_electron_error": np.nan,
                "rmse_total_electron_error": np.nan,
                "status": status,
                "notes": notes or "no evaluated voxels",
            }

        mse = self.sum_sq / self.n
        pred_mean = self.sum_pred / self.n
        target_mean = self.sum_target / self.n
        cov = self.sum_pred_target - self.n * pred_mean * target_mean
        pred_var = self.sum_pred_sq - self.n * pred_mean * pred_mean
        target_var = self.sum_target_sq - self.n * target_mean * target_mean
        denom = math.sqrt(max(pred_var, 0.0) * max(target_var, 0.0))
        total_electron_errors = np.asarray(self.sample_total_electron_errors, dtype=np.float64)
        input_construction_seconds = np.asarray(self.sample_input_construction_seconds, dtype=np.float64)
        model_reconstruction_seconds = np.asarray(self.sample_model_reconstruction_seconds, dtype=np.float64)
        inference_seconds = np.asarray(self.sample_inference_seconds, dtype=np.float64)
        valid_input = input_construction_seconds[np.isfinite(input_construction_seconds)]
        valid_model = model_reconstruction_seconds[np.isfinite(model_reconstruction_seconds)]
        valid_inference = inference_seconds[np.isfinite(inference_seconds)]
        return {
            "baseline": baseline,
            "family": family,
            "checkpoint": checkpoint,
            "num_samples": self.num_samples,
            "num_voxels": self.n,
            "avg_input_construction_seconds": float(valid_input.mean()) if valid_input.size > 0 else np.nan,
            "std_input_construction_seconds": float(valid_input.std()) if valid_input.size > 0 else np.nan,
            "total_input_construction_seconds": float(valid_input.sum()) if valid_input.size > 0 else np.nan,
            "avg_model_reconstruction_seconds": float(valid_model.mean()) if valid_model.size > 0 else np.nan,
            "std_model_reconstruction_seconds": float(valid_model.std()) if valid_model.size > 0 else np.nan,
            "total_model_reconstruction_seconds": float(valid_model.sum()) if valid_model.size > 0 else np.nan,
            "avg_inference_seconds": float(valid_inference.mean()) if valid_inference.size > 0 else np.nan,
            "std_inference_seconds": float(valid_inference.std()) if valid_inference.size > 0 else np.nan,
            "total_inference_seconds": float(valid_inference.sum()) if valid_inference.size > 0 else np.nan,
            "mse": mse,
            "rmse": math.sqrt(mse),
            "mae": self.sum_abs / self.n,
            "pearson": cov / denom if denom > 0 else np.nan,
            "total_electron_error": float(total_electron_errors.mean()),
            "mae_total_electron_error": float(np.mean(np.abs(total_electron_errors))),
            "rmse_total_electron_error": float(np.sqrt(np.mean(total_electron_errors ** 2))),
            "status": status,
            "notes": notes,
        }

    def density_bin_rows(self, baseline):
        rows = []
        for idx in range(len(self.bins) - 2, -1, -1):
            lo = self.bins[idx]
            hi = self.bins[idx + 1]
            stats = self.density_bin_stats[idx]
            count = stats["count"]
            if count > 0:
                mae = stats["sum_abs"] / count
                rmse = math.sqrt(stats["sum_sq"] / count)
            else:
                mae = np.nan
                rmse = np.nan
            rows.append(
                {
                    "baseline": baseline,
                    "bin_index": len(self.bins) - 2 - idx,
                    "bin": density_bin_label(lo, hi),
                    "density_min": lo,
                    "density_max": hi,
                    "count": count,
                    "mae": mae,
                    "rmse": rmse,
                }
            )
        return rows

    def error_distribution_rows(self):
        rows = []
        total = int(self.error_hist_counts.sum())
        upper_cap = max(self.max_abs_error, 1e2)
        for idx in range(len(self.bins) - 2, -1, -1):
            lo = self.bins[idx]
            hi = self.bins[idx + 1]
            hi_value = upper_cap if math.isinf(hi) else hi
            count = int(self.error_hist_counts[idx])
            rows.append(
                {
                    "bin_index": len(self.bins) - 2 - idx,
                    "bin": density_bin_label(lo, hi),
                    "bin_lower": lo,
                    "bin_upper": hi_value,
                    "count": count,
                    "probability": (count / total) if total > 0 else np.nan,
                }
            )
        return rows
