#!/usr/bin/env python3
"""LOESS validation plots — all figures in one script."""

import os
import sys
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Shared configuration
# ---------------------------------------------------------------------------

INPUT_DIR = "output/visual"
OUTPUT_DIR = "output/fig"

plt.rcParams["svg.fonttype"] = "none"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def check_file(filename):
    """Return whether an input CSV exists, printing a helpful error when it does not."""
    filepath = os.path.join(INPUT_DIR, filename)
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found. Run 'cargo run --bin visual' first.")
        return False
    return True


def get_input_path(filename):
    """Return the absolute path for an input CSV under the visual output directory."""
    return os.path.join(INPUT_DIR, filename)


def get_output_path(filename):
    """Return the absolute path for an output figure under the figure directory."""
    return os.path.join(OUTPUT_DIR, filename)


def save_figure(figure, filename, *, dpi=72, bbox_inches=None):
    """Save a Matplotlib figure as SVG and close it afterwards."""
    output_file = get_output_path(filename)
    save_kwargs = {"format": "svg", "dpi": dpi}
    if bbox_inches is not None:
        save_kwargs["bbox_inches"] = bbox_inches
    figure.savefig(output_file, **save_kwargs)
    print(f"Saved to {output_file}")
    plt.close(figure)


def column_to_numpy(dataframe, column_name):
    """Convert a DataFrame column to a NumPy array for plotting helpers."""
    return np.asarray(dataframe[column_name].to_numpy())


def add_rasterized_colorbar(figure, surface, axis):
    """Attach a colorbar and rasterize its solids when the backend exposes them."""
    colorbar = figure.colorbar(surface, ax=axis, shrink=0.5)
    solids = getattr(colorbar, "solids", None)
    if solids is not None:
        solids.set_rasterized(True)


def plot_noisy_points(axis, x_values, y_values, **plot_kwargs):
    """Plot noisy sample points with the shared marker style used across figures."""
    plot_kwargs.setdefault("color", "gray")
    axis.plot(x_values, y_values, "o", **plot_kwargs)


def plot_true_signal(axis, x_values, y_values, **plot_kwargs):
    """Plot the shared dashed true-signal reference line."""
    plot_kwargs.setdefault("label", "True Signal")
    axis.plot(x_values, y_values, "k--", **plot_kwargs)


# ---------------------------------------------------------------------------
# Foundation plots
# ---------------------------------------------------------------------------


def plot_degree_comparison():
    """Plot the linear-versus-quadratic smoothing comparison."""
    if not check_file("degree_comparison.csv"):
        return
    print("Plotting Degree Comparison...")

    df = pd.read_csv(get_input_path("degree_comparison.csv"))
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax1 = axes[0]
    ax1.plot(
        df["x"],
        df["y_noisy"],
        "o",
        markersize=3,
        alpha=0.4,
        color="gray",
        markeredgewidth=0,
        label="Noisy data",
        rasterized=True,
    )
    plot_true_signal(ax1, df["x"], df["y_true"], lw=1.5, alpha=0.7)
    ax1.plot(df["x"], df["y_linear"], "b-", lw=2, label="LOESS (Linear)")
    ax1.plot(df["x"], df["y_quadratic"], "r-", lw=2, label="LOESS (Quadratic)")
    # Highlight the zoomed peak region shown in the right panel
    ax1.axvspan(-0.6, 0.6, alpha=0.12, color="orange", label="Zoomed region →")
    ax1.set_xlabel("x", fontsize=12)
    ax1.set_ylabel("y", fontsize=12)
    ax1.set_title("LOESS: Linear vs Quadratic — Full View", fontsize=14)
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    mask = (df["x"] > -0.6) & (df["x"] < 0.6)
    ax2.plot(
        df.loc[mask, "x"],
        df.loc[mask, "y_noisy"],
        "o",
        markersize=4,
        alpha=0.5,
        color="gray",
        markeredgewidth=0,
        label="Noisy data",
        rasterized=True,
    )
    plot_true_signal(ax2, df.loc[mask, "x"], df.loc[mask, "y_true"], lw=2)
    ax2.plot(
        df.loc[mask, "x"],
        df.loc[mask, "y_linear"],
        "b-",
        lw=2.5,
        label="LOESS (Linear)",
    )
    ax2.plot(
        df.loc[mask, "x"],
        df.loc[mask, "y_quadratic"],
        "r-",
        lw=2.5,
        label="LOESS (Quadratic)",
    )
    ax2.set_xlabel("x", fontsize=12)
    ax2.set_ylabel("y", fontsize=12)
    ax2.set_title("Peak Region Zoom", fontsize=14)
    ax2.legend(loc="lower center")
    ax2.grid(True, alpha=0.3)

    figure.tight_layout()
    save_figure(figure, "degree_comparison.svg", dpi=72, bbox_inches=None)


def plot_fraction_comparison():
    """Plot the effect of different smoothing fractions."""
    if not check_file("fraction_comparison.csv"):
        return
    print("Plotting Fraction Comparison...")

    df = pd.read_csv(get_input_path("fraction_comparison.csv"))
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))

    fractions = [0.2, 0.5, 0.9]
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    titles = [
        "Small Fraction (0.2)\nCaptures Details, May Overfit",
        "Medium Fraction (0.5)\nBalanced Smoothing",
        "Large Fraction (0.9)\nVery Smooth, May Underfit",
    ]

    for ax, frac, color, title in zip(axes, fractions, colors, titles):
        ax.plot(
            df["x"],
            df["y_noisy"],
            "o",
            markersize=3.5,
            alpha=0.3,
            color="gray",
            markeredgewidth=0,
            label="Noisy data",
            zorder=1,
            rasterized=True,
        )
        ax.plot(
            df["x"],
            df["y_true"],
            "k--",
            lw=1.5,
            label="True signal",
            alpha=0.6,
            zorder=2,
        )
        ax.plot(
            df["x"],
            df[f"y_frac_{frac}"],
            color=color,
            lw=2.5,
            label=f"LOESS (fraction={frac})",
            zorder=3,
        )
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("y", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(loc="upper left", fontsize=10)
        ax.grid(True, alpha=0.3)
        rmse = ((df[f"y_frac_{frac}"] - df["y_true"]) ** 2).mean() ** 0.5
        ax.text(
            0.98,
            0.02,
            f"RMSE: {rmse:.3f}",
            transform=ax.transAxes,
            fontsize=11,
            ha="right",
            va="bottom",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

    figure.suptitle(
        "Effect of Fraction Parameter on LOESS Smoothing",
        fontsize=16,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(figure, "fraction_comparison.svg")


def plot_intervals_comparison():
    """Plot confidence and prediction interval bands around the smooth."""
    if not check_file("intervals_comparison.csv"):
        return
    print("Plotting Intervals Comparison...")

    df = pd.read_csv(get_input_path("intervals_comparison.csv"))
    figure, ax = plt.subplots(figsize=(10, 8))
    x_values = column_to_numpy(df, "x")
    pred_lower = column_to_numpy(df, "pred_lower")
    pred_upper = column_to_numpy(df, "pred_upper")
    conf_lower = column_to_numpy(df, "conf_lower")
    conf_upper = column_to_numpy(df, "conf_upper")

    ax.plot(
        df["x"],
        df["y_noisy"],
        "o",
        markersize=4.5,
        alpha=0.6,
        color="gray",
        markeredgewidth=0,
        label="Noisy Data",
        zorder=1,
    )
    ax.plot(
        df["x"], df["y_true"], "k--", lw=1.5, label="True Signal", alpha=0.7, zorder=2
    )
    ax.plot(df["x"], df["y_smooth"], "k-", lw=2.5, label="LOESS Fit", zorder=5)

    ax.fill_between(
        x_values,
        cast(Any, pred_lower),
        cast(Any, pred_upper),
        alpha=0.2,
        color="#3b82f6",
        label="95% Prediction Interval",
        zorder=3,
    )
    ax.plot(
        df["x"], df["pred_lower"], linestyle="--", color="#1d4ed8", lw=1.5, alpha=0.6
    )
    ax.plot(
        df["x"], df["pred_upper"], linestyle="--", color="#1d4ed8", lw=1.5, alpha=0.6
    )

    ax.fill_between(
        x_values,
        cast(Any, conf_lower),
        cast(Any, conf_upper),
        alpha=0.4,
        color="#22c55e",
        label="95% Confidence Interval",
        zorder=4,
    )
    ax.plot(
        df["x"], df["conf_lower"], linestyle="--", color="#15803d", lw=1.5, alpha=0.8
    )
    ax.plot(
        df["x"], df["conf_upper"], linestyle="--", color="#15803d", lw=1.5, alpha=0.8
    )

    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_title(
        "Uncertainty Decomposition: Confidence vs Prediction Intervals",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    avg_pred_width = (df["pred_upper"] - df["pred_lower"]).mean()
    avg_conf_width = (df["conf_upper"] - df["conf_lower"]).mean()
    footnote_text = (
        f"Avg Width Ratio (Pred/Conf): {avg_pred_width / avg_conf_width:.2f}x\n"
        "Confidence Interval: Uncertainty in mean curve (Green)\n"
        "Prediction Interval: Uncertainty for new observations (Blue)"
    )
    figure.subplots_adjust(bottom=0.15)
    figure.text(
        0.05,
        0.02,
        footnote_text,
        fontsize=11,
        family="monospace",
        va="bottom",
        ha="left",
    )

    save_figure(figure, "intervals_comparison.svg")


def plot_robust_iter_comparison():
    """Plot the effect of robust reweighting in the presence of outliers."""
    if not check_file("robust_iter_comparison.csv"):
        return
    print("Plotting Robustness Comparison...")

    df = pd.read_csv(get_input_path("robust_iter_comparison.csv"))
    figure, ax = plt.subplots(figsize=(12, 7))
    x_values = df["x"]
    noisy_values = df["y_noisy"]
    true_values = df["y_true"]

    plot_noisy_points(
        ax,
        x_values,
        noisy_values,
        markersize=5.5,
        alpha=0.6,
        label="Noisy Data",
        markeredgewidth=0,
        zorder=1,
    )
    plot_true_signal(ax, x_values, true_values, linewidth=1.5, alpha=0.6, zorder=2)
    ax.plot(
        df["x"],
        df["y_non_robust"],
        color="#ef4444",
        lw=2.0,
        alpha=0.9,
        label="Non-Robust (0 iter)",
    )
    ax.plot(
        df["x"],
        df["y_robust"],
        color="#10b981",
        lw=3.0,
        alpha=0.95,
        label="Robust (6 iter)",
    )

    outlier_mask = np.abs(noisy_values - true_values) > 2.0
    ax.scatter(
        df.loc[outlier_mask, "x"],
        df.loc[outlier_mask, "y_noisy"],
        s=80,
        facecolors="none",
        edgecolors="#ef4444",
        lw=1.5,
        label="Outliers",
        zorder=5,
    )

    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_ylim(-10, 25)
    ax.set_title("Impact of Robustness Iterations", fontsize=14, fontweight="bold")
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)

    save_figure(figure, "robust_iter_comparison.svg")


def plot_loess_concept():
    """Visualize the local neighborhood and fit used at one focal point."""
    if not check_file("loess_concept.csv"):
        return
    print("Plotting LOESS Concept...")

    df = pd.read_csv(get_input_path("loess_concept.csv"))
    focus_row = df[df["is_focus"] == 1].iloc[0]
    x0 = focus_row["x"]
    y0_fit = focus_row["y_smooth"]
    neighborhood = df[df["weight"] > 0]

    figure, ax = plt.subplots(figsize=(12, 7))
    ax.scatter(
        df["x"], df["y_noisy"], c="lightgray", s=30, alpha=0.5, label="Other Data"
    )
    ax.plot(
        df["x"], df["y_smooth"], color="black", lw=2, alpha=0.3, label="Global Curve"
    )

    scatter = ax.scatter(
        neighborhood["x"],
        neighborhood["y_noisy"],
        c=neighborhood["weight"],
        cmap="Blues",
        s=60,
        edgecolor="k",
        linewidth=0.5,
        label="Neighborhood",
    )
    ax.plot(
        neighborhood["x"],
        neighborhood["y_local_fit_x0"],
        color="#d97706",
        lw=3,
        label="Local Polynomial",
    )
    ax.scatter(
        [x0],
        [y0_fit],
        s=150,
        facecolor="#d97706",
        edgecolor="white",
        lw=2,
        zorder=10,
        label="Fitted Value",
    )

    ax.set_title(f"How LOESS Works (Focus x={x0:.2f})", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    colorbar = figure.colorbar(scatter, ax=ax)
    colorbar.set_label("Weight", fontsize=10)

    save_figure(figure, "loess_concept.svg")


def plot_multivariate_loess():
    """Plot the true and smoothed multivariate surfaces side by side."""
    if not check_file("multivariate_loess.csv"):
        return
    print("Plotting Multivariate LOESS...")

    df = pd.read_csv(get_input_path("multivariate_loess.csv"))
    n_x = len(df["x"].unique())
    n_y = len(df["y"].unique())
    x_grid = column_to_numpy(df, "x").reshape(n_x, n_y)
    y_grid = column_to_numpy(df, "y").reshape(n_x, n_y)
    z_true_grid = column_to_numpy(df, "z_true").reshape(n_x, n_y)
    z_smooth_grid = column_to_numpy(df, "z_smooth").reshape(n_x, n_y)

    figure = plt.figure(figsize=(10, 5))

    ax1 = figure.add_subplot(1, 2, 1, projection="3d")
    surf1 = ax1.plot_surface(
        x_grid,
        y_grid,
        z_true_grid,
        cmap="viridis",
        alpha=0.9,
        edgecolor="none",
        rasterized=True,
    )
    ax1.set_title("True Surface")
    ax1.locator_params(nbins=4)
    add_rasterized_colorbar(figure, surf1, ax1)

    ax2 = figure.add_subplot(1, 2, 2, projection="3d")
    surf2 = ax2.plot_surface(
        x_grid,
        y_grid,
        z_smooth_grid,
        cmap="magma",
        alpha=0.9,
        edgecolor="none",
        rasterized=True,
    )
    ax2.set_title("LOESS Smoothed")
    ax2.locator_params(nbins=4)
    add_rasterized_colorbar(figure, surf2, ax2)

    figure.tight_layout()
    save_figure(figure, "multivariate_loess.svg", dpi=100)


# ---------------------------------------------------------------------------
# Modeling plots
# ---------------------------------------------------------------------------


def plot_kernel_comparison():
    """Plot the impact of different kernel weight functions."""
    if not check_file("kernel_comparison.csv"):
        return
    print("Plotting Kernel Comparison...")

    df = pd.read_csv(get_input_path("kernel_comparison.csv"))
    figure, ax = plt.subplots(figsize=(10, 6))

    plot_noisy_points(
        ax, df["x"], df["y_noisy"], markersize=3, alpha=0.3, label="Noisy Data"
    )
    plot_true_signal(ax, df["x"], df["y_true"], linewidth=1.5, alpha=0.5)

    configs = [
        ("y_tricube", "Tricube", "-", "#3b82f6", 3.0),
        ("y_gaussian", "Gaussian", "--", "#ef4444", 2.0),
        ("y_uniform", "Uniform", ":", "#10b981", 2.0),
        ("y_cosine", "Cosine", "-", "#f59e0b", 2.0),
        ("y_epanechnikov", "Epanechnikov", "--", "#8b5cf6", 2.0),
        ("y_biweight", "Biweight", ":", "#06b6d4", 2.0),
        ("y_triangle", "Triangle", "-", "#ec4899", 2.0),
    ]
    for col, name, style, color, lw in configs:
        if col in df.columns:
            ax.plot(df["x"], df[col], linestyle=style, lw=lw, label=name, color=color)

    ax.set_title(
        "Impact of Different Kernel (Weight) Functions", fontsize=14, fontweight="bold"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.3)

    save_figure(figure, "kernel_comparison.svg")


def plot_robust_method_comparison():
    """Compare alternative robust weighting methods."""
    if not check_file("robust_method_comparison.csv"):
        return
    print("Plotting Robust Method Comparison...")

    df = pd.read_csv(get_input_path("robust_method_comparison.csv"))
    figure, ax = plt.subplots(figsize=(12, 7))

    plot_noisy_points(
        ax, df["x"], df["y_noisy"], markersize=4, alpha=0.4, label="Data with Outliers"
    )
    plot_true_signal(ax, df["x"], df["y_true"], linewidth=1, alpha=0.5)
    ax.plot(
        df["x"],
        df["y_bisquare"],
        "-",
        lw=2.5,
        label="Bisquare (Soft Downweighting — most bias here)",
        color="#3b82f6",
    )
    ax.plot(
        df["x"],
        df["y_huber"],
        "--",
        lw=2,
        label="Huber (Proportional — intermediate)",
        color="#ef4444",
    )
    ax.plot(
        df["x"],
        df["y_talwar"],
        ":",
        lw=2.5,
        label="Talwar (Hard Cutoff — least bias here)",
        color="#10b981",
    )

    ax.set_title(
        "Outlier Resistance by Robustness Method", fontsize=14, fontweight="bold"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_figure(figure, "robust_method_comparison.svg")


def plot_boundary_policy_comparison():
    """Compare boundary handling policies near the ends of the domain."""
    if not check_file("boundary_comparison.csv"):
        return
    print("Plotting Boundary Policy Comparison...")

    df = pd.read_csv(get_input_path("boundary_comparison.csv"))
    figure, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ax in [ax1, ax2]:
        plot_noisy_points(
            ax, df["x"], df["y_noisy"], markersize=3, alpha=0.3, label="Noisy Data"
        )
        plot_true_signal(ax, df["x"], df["y_true"], linewidth=1, alpha=0.5)
        ax.plot(df["x"], df["y_none"], "-", lw=2, label="NoBoundary", color="#ef4444")
        ax.plot(df["x"], df["y_extend"], "--", lw=2, label="Extend", color="#3b82f6")
        ax.plot(df["x"], df["y_reflect"], "-.", lw=2, label="Reflect", color="#10b981")
        ax.grid(True, alpha=0.3)

    ax1.set_xlim(-0.05, 0.2)
    ax1.set_ylim(0.8, 2.0)
    ax1.set_title("Left Boundary Impact")
    ax2.set_xlim(0.8, 1.05)
    ax2.set_ylim(10, 22)
    ax2.set_title("Right Boundary Impact")
    figure.suptitle(
        "Mitigating Boundary Bias with Reflection and Extension",
        fontsize=16,
        fontweight="bold",
    )
    ax2.legend(loc="lower right")

    save_figure(figure, "boundary_comparison.svg")


def plot_higher_degree_comparison():
    """Plot four local polynomial degrees on non-uniformly spaced data.

    With uniformly-spaced x the normal equations decouple into independent even/odd
    subsystems, making cubic identical to quadratic.  Non-uniform spacing breaks
    the exact cancellation of odd kernel moments so all four degrees genuinely differ.
    Signal: t^3/3 - t + t^4/12  (has both cubic and quartic components).
    """
    if not check_file("higher_degree_comparison.csv"):
        return
    print("Plotting Higher Degree Comparison...")

    df = pd.read_csv(get_input_path("higher_degree_comparison.csv"))
    figure, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    plot_noisy_points(
        ax1, df["x"], df["y_noisy"], markersize=3, alpha=0.25, label="Noisy Data"
    )
    plot_true_signal(ax1, df["x"], df["y_true"], linewidth=1.5, alpha=0.6)

    degree_specs = [
        ("y_linear", "Linear", "-", "#ef4444"),
        ("y_quadratic", "Quadratic", "--", "#f97316"),
        ("y_cubic", "Cubic", "-.", "#3b82f6"),
        ("y_quartic", "Quartic", ":", "#10b981"),
    ]
    for col, label, style, color in degree_specs:
        r = np.sqrt(((df[col] - df["y_true"]) ** 2).mean())
        ax1.plot(
            df["x"], df[col], style, lw=2, label=f"{label} (RMSE={r:.4f})", color=color
        )

    ax1.set_ylabel("y")
    ax1.set_title(
        "Effect of Local Polynomial Degree on LOESS Fit\n"
        "Non-uniform x spacing \u2014 all four degrees genuinely differ",
        fontsize=11,
        fontweight="bold",
    )
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.axhline(0, color="black", lw=1, alpha=0.5)
    for col, label, style, color in degree_specs:
        ax2.plot(
            df["x"],
            df[col] - df["y_true"],
            style,
            lw=1.5,
            label=label,
            color=color,
            alpha=0.85,
        )
    ax2.set_xlabel("x")
    ax2.set_ylabel("Residual")
    ax2.set_title("Residuals vs True Signal", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    figure.tight_layout()
    save_figure(figure, "higher_degree_comparison.svg")


def plot_gap_handling():
    """Plot smoothing across a region with missing observations."""
    if not check_file("gap_handling.csv"):
        return
    print("Plotting Gap Handling...")

    df = pd.read_csv(get_input_path("gap_handling.csv"))
    figure, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        df["x"],
        df["y_noisy"],
        "o",
        markersize=4,
        alpha=0.6,
        color="#3b82f6",
        label="Available Data",
    )
    ax.plot(df["x"], df["y_smooth"], "r-", lw=3, label="LOESS Interpolation")
    ax.set_title(
        "LOESS Gap Handling (Bridging Missing Regions)", fontsize=14, fontweight="bold"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axvspan(4.0, 7.0, color="gray", alpha=0.1, label="Missing Data Region")
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_figure(figure, "gap_handling.svg")


def plot_cv_comparison():
    """Compare cross-validation score curves and their resulting fits."""
    if not check_file("cv_scores.csv") or not check_file("cv_fits.csv"):
        return
    print("Plotting Cross-Validation Comparison...")

    df_scores = pd.read_csv(get_input_path("cv_scores.csv"))
    df_fits = pd.read_csv(get_input_path("cv_fits.csv"))
    figure, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.plot(
        df_scores["fraction"],
        df_scores["loocv_rmse"],
        "o-",
        label="LOOCV RMSE",
        color="#3b82f6",
    )
    ax1.plot(
        df_scores["fraction"],
        df_scores["kfold_rmse"],
        "s--",
        label="5-Fold RMSE",
        color="#ef4444",
    )

    best_loocv_fraction = df_scores.loc[df_scores["loocv_rmse"].idxmin(), "fraction"]
    best_kfold_fraction = df_scores.loc[df_scores["kfold_rmse"].idxmin(), "fraction"]

    ax1.axvline(best_loocv_fraction, color="#3b82f6", alpha=0.3, linestyle="-")
    ax1.axvline(best_kfold_fraction, color="#ef4444", alpha=0.3, linestyle="--")
    ax1.set_title(
        "Bandwidth Selection: CV Score vs Fraction", fontsize=14, fontweight="bold"
    )
    ax1.set_xlabel("Smoothing Fraction")
    ax1.set_ylabel("RMSE")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    plot_noisy_points(
        ax2,
        df_fits["x"],
        df_fits["y_noisy"],
        markersize=3,
        alpha=0.3,
        label="Noisy Data",
    )
    plot_true_signal(ax2, df_fits["x"], df_fits["y_true"], linewidth=1.5, alpha=0.5)
    ax2.plot(
        df_fits["x"],
        df_fits["y_loocv"],
        "-",
        lw=2.5,
        label=f"LOOCV (f={best_loocv_fraction})",
        color="#3b82f6",
    )
    ax2.plot(
        df_fits["x"],
        df_fits["y_kfold"],
        "--",
        lw=2,
        label=f"5-Fold (f={best_kfold_fraction})",
        color="#ef4444",
    )
    ax2.plot(
        df_fits["x"],
        df_fits["y_fixed"],
        ":",
        lw=2,
        label="No CV (f=0.8, Over-smooth)",
        color="#10b981",
    )

    ax2.set_title(
        "Impact of Bandwidth Selection on Fit", fontsize=14, fontweight="bold"
    )
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)

    figure.tight_layout()
    save_figure(figure, "cv_comparison.svg")


def plot_surface_mode_comparison():
    """Compare direct and interpolation surface evaluation modes."""
    if not check_file("surface_mode_comparison.csv"):
        return
    print("Plotting Surface Mode Comparison...")

    df = pd.read_csv(get_input_path("surface_mode_comparison.csv"))
    figure, (ax_fit, ax_res) = plt.subplots(
        1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [3, 2]}
    )

    # --- Left: both fits on the full signal ---
    plot_noisy_points(
        ax_fit, df["x"], df["y_noisy"], markersize=3, alpha=0.3, label="Noisy Data"
    )
    plot_true_signal(ax_fit, df["x"], df["y_true"], linewidth=1, alpha=0.5)
    ax_fit.plot(
        df["x"], df["y_direct"], "-", lw=2.5, label="Direct (exact)", color="#3b82f6"
    )
    ax_fit.plot(
        df["x"],
        df["y_interpolation"],
        "--",
        lw=2,
        label="Interpolation (~9 vertices)",
        color="#ef4444",
    )
    ax_fit.set_title("Both Fits — nearly identical at this scale", fontsize=12)
    ax_fit.legend(fontsize=10)
    ax_fit.grid(True, alpha=0.3)

    # --- Right: residuals Direct − Interpolation ---
    residuals = df["y_direct"] - df["y_interpolation"]
    ax_res.axhline(0, color="gray", lw=1, ls="--")
    ax_res.plot(df["x"], residuals, "-", lw=1.5, color="#f59e0b")
    ax_res.fill_between(df["x"], residuals, 0, alpha=0.3, color="#f59e0b")
    ax_res.set_title(
        f"Residuals: Direct − Interpolation\nmax = {residuals.abs().max():.4f}",
        fontsize=12,
    )
    ax_res.set_xlabel("x")
    ax_res.grid(True, alpha=0.3)

    figure.suptitle(
        "Direct vs Interpolation Surface Mode\n"
        "Interpolation uses ~9 Hermite-cubic vertices instead of 200 local fits — fast but approximate",
        fontsize=13,
        fontweight="bold",
    )
    figure.tight_layout()

    save_figure(figure, "surface_comparison.svg")


def plot_scaling_comparison():
    """Compare robust residual scaling strategies."""
    if not check_file("scaling_comparison.csv"):
        return
    print("Plotting Scaling Method Comparison...")

    df = pd.read_csv(get_input_path("scaling_comparison.csv"))
    figure, ax = plt.subplots(figsize=(10, 6))

    plot_noisy_points(
        ax,
        df["x"],
        df["y_noisy"],
        markersize=4,
        alpha=0.3,
        label="Data (20% moderate +1.5, 20% extreme +6)",
    )
    plot_true_signal(ax, df["x"], df["y_true"], linewidth=1, alpha=0.5)
    ax.plot(
        df["x"],
        df["y_none"],
        ":",
        lw=2,
        label="No Robustness — both tiers bias the fit",
        color="#10b981",
    )
    ax.plot(
        df["x"],
        df["y_mean"],
        "--",
        lw=2,
        label="Mean (MAE) — extreme outliers inflate scale → moderate outliers slip through",
        color="#ef4444",
    )
    ax.plot(
        df["x"],
        df["y_mad"],
        "-.",
        lw=2,
        label="MAD — centers on median residual first, similar to MAR here",
        color="#f59e0b",
    )
    ax.plot(
        df["x"],
        df["y_mar"],
        "-",
        lw=2.5,
        label="MAR — scale anchored to clean noise level → rejects all outliers",
        color="#3b82f6",
    )

    ax.set_title(
        "Robust Scaling: MAR vs MAD vs Mean (MAE) vs None\n"
        "Extreme outliers inflate Mean scale → looser threshold → moderate outliers leak in",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    save_figure(figure, "scaling_comparison.svg")


def plot_zero_weight_comparison():
    """Compare ZeroWeightFallback policies: UseLocalMean, ReturnOriginal, ReturnNone."""
    if not check_file("zero_weight_comparison.csv"):
        return
    print("Plotting Zero Weight Fallback Comparison...")

    df = pd.read_csv(get_input_path("zero_weight_comparison.csv"))
    figure, ax = plt.subplots(figsize=(11, 6))

    # Shade the anomalous zone where fallbacks are triggered
    ax.axvspan(
        4.0,
        6.0,
        alpha=0.07,
        color="red",
        zorder=0,
        label="Anomalous zone",
    )

    plot_noisy_points(
        ax,
        df["x"],
        df["y_noisy"],
        markersize=3,
        alpha=0.25,
        label="Data",
    )
    plot_true_signal(ax, df["x"], df["y_true"], linewidth=1.2, alpha=0.5)

    ax.plot(
        df["x"],
        df["y_local_mean"],
        "-",
        lw=2.5,
        label="UseLocalMean",
        color="#3b82f6",
        zorder=3,
    )
    ax.plot(
        df["x"],
        df["y_return_original"],
        "--",
        lw=2,
        label="ReturnOriginal",
        color="#ef4444",
        zorder=3,
    )
    ax.plot(
        df["x"],
        df["y_return_none"],
        ":",
        lw=2,
        label="ReturnNone",
        color="#f59e0b",
        zorder=3,
    )

    ax.set_title(
        "Zero-Weight Fallback Policies: UseLocalMean / ReturnOriginal / ReturnNone",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    save_figure(figure, "zero_weight_comparison.svg")


def plot_degree_interpolation_comparison():
    """Compare direct and interpolated evaluation across polynomial degrees."""
    if not check_file("degree_interpolation_comparison.csv"):
        return
    print("Plotting Degree Interpolation Comparison...")

    df = pd.read_csv(get_input_path("degree_interpolation_comparison.csv"))
    figure, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    configs = [
        ("Linear", "y_lin_direct", "y_lin_interp", "#3b82f6", "#93c5fd"),
        ("Quadratic", "y_quad_direct", "y_quad_interp", "#ef4444", "#fca5a5"),
        ("Cubic", "y_cubic_direct", "y_cubic_interp", "#8b5cf6", "#c4b5fd"),
        ("Quartic", "y_quartic_direct", "y_quartic_interp", "#10b981", "#6ee7b7"),
    ]

    for ax, (name, direct_col, interp_col, direct_color, interp_color) in zip(
        axes, configs
    ):
        plot_noisy_points(
            ax, df["x"], df["y_noisy"], markersize=3, alpha=0.3, label=None
        )
        plot_true_signal(ax, df["x"], df["y_true"], linewidth=1, alpha=0.5, label=None)
        ax.plot(
            df["x"],
            df[direct_col],
            "-",
            lw=2.5,
            label=f"{name} Direct",
            color=direct_color,
        )
        ax.plot(
            df["x"],
            df[interp_col],
            "--",
            lw=2.5,
            label=f"{name} Interp",
            color=interp_color,
        )
        ax.set_title(f"{name} Degree", fontsize=12, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

    figure.suptitle(
        "Surface Evaluation Fidelity across polynomial Degrees: Direct vs Interpolation",
        fontsize=16,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    save_figure(figure, "degree_interpolation_comparison.svg")


# ---------------------------------------------------------------------------
# Streaming plots
# ---------------------------------------------------------------------------


def plot_merge_comparison():
    """Compare streaming merge strategies."""
    if not check_file("merge_comparison.csv"):
        return
    print("Plotting Streaming Comparison...")

    df = pd.read_csv(get_input_path("merge_comparison.csv"))
    figure, ax = plt.subplots(figsize=(12, 7))

    plot_noisy_points(
        ax, df["x"], df["y_noisy"], markersize=3, alpha=0.2, label="Noisy Data"
    )
    plot_true_signal(ax, df["x"], df["y_true"], linewidth=1.5, alpha=0.6)
    ax.plot(
        df["x"],
        df["y_first"],
        ":",
        lw=2.5,
        label="TakeFirst — hard seam at chunk boundary",
        color="#10b981",
    )
    ax.plot(
        df["x"],
        df["y_average"],
        "--",
        lw=2,
        label="Average — equal blend in overlap zone",
        color="#ef4444",
    )
    ax.plot(
        df["x"],
        df["y_weighted"],
        "-",
        lw=2.5,
        label="WeightedAverage — distance-weighted smooth ramp",
        color="#3b82f6",
    )

    # Mark chunk boundaries
    ymin, ymax = ax.get_ylim()
    for boundary in [150, 300, 450]:
        ax.axvline(boundary, color="gray", linestyle="--", alpha=0.4, lw=1)
        ax.text(
            boundary + 3,
            ymax * 0.97,
            "chunk\nboundary",
            color="gray",
            fontsize=8,
            va="top",
        )

    ax.set_title(
        "Streaming LOESS: Merge Strategy Comparison\n"
        "Signal jumps at each chunk boundary — strategies diverge in the 90-pt overlap zone",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Point Index")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_figure(figure, "merge_comparison.svg")


def plot_online_comparison():
    """Compare online smoothing settings with different window sizes."""
    if not check_file("online_comparison.csv"):
        return
    print("Plotting Online Comparison...")

    df = pd.read_csv(get_input_path("online_comparison.csv"))
    figure, ax = plt.subplots(figsize=(12, 7))

    plot_noisy_points(
        ax, df["x"], df["y_noisy"], markersize=3, alpha=0.3, label="Streaming Data"
    )
    plot_true_signal(ax, df["x"], df["y_true"], linewidth=1.5, alpha=0.6)
    ax.plot(
        df["x"],
        df["y_small_window"],
        "-",
        lw=2.5,
        label="Online (Window=30, Incremental — fast)",
        color="#ef4444",
    )
    ax.plot(
        df["x"],
        df["y_large_window"],
        "-",
        lw=3,
        label="Online (Window=300, Full — smooth but lags)",
        color="#3b82f6",
    )

    ax.set_title(
        "Online LOESS: Incremental Smoothing with Sliding Window",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Point Index")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axvline(200, color="gray", linestyle=":", alpha=0.5)
    ax.text(205, ax.get_ylim()[1] * 0.92, "Shift +2.5", color="gray", fontsize=10)
    ax.axvline(400, color="gray", linestyle=":", alpha=0.5)
    ax.text(405, ax.get_ylim()[1] * 0.92, "Shift −2.0", color="gray", fontsize=10)

    save_figure(figure, "online_comparison.svg")


def plot_adapter_comparison():
    """Plot the effect of automatic convergence across adapter styles."""
    if not check_file("adapter_comparison.csv"):
        return
    print("Plotting Auto-Convergence Comparison...")

    df = pd.read_csv(get_input_path("adapter_comparison.csv"))
    figure, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    adapters = [
        ("Batch", "batch", ax1, "#3b82f6"),
        ("Streaming", "stream", ax2, "#10b981"),
        ("Online", "online", ax3, "#ef4444"),
    ]

    for name, key, ax, base_color in adapters:
        plot_noisy_points(
            ax, df["x"], df["y_noisy"], markersize=2, alpha=0.15, label="Noisy Data"
        )
        plot_true_signal(ax, df["x"], df["y_true"], linewidth=1, alpha=0.4)
        ax.plot(
            df["x"],
            df[f"y_{key}_off"],
            "-",
            lw=2,
            label=f"{name} (Standard)",
            color=base_color,
        )

        saved = df[f"iter_{key}_off"] - df[f"iter_{key}_on"]
        total_saved = int(saved.sum())

        if key == "online":
            subtitle = "Online processing doesn't benefit much from\nearly stopping since each point update is cheap"
        else:
            subtitle = f"{total_saved} iters saved with auto convergence"

        ax.set_title(
            f"{name} Adapter\n{subtitle}",
            fontsize=12,
            fontweight="bold",
        )
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("X")
        if ax == ax1:
            ax.set_ylabel("Y")

    figure.tight_layout()
    save_figure(figure, "adapter_comparison.svg")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def build_plot_targets():
    """Return the stable CLI mapping from target names to plot functions."""
    return {
        "degree": plot_degree_comparison,
        "fraction": plot_fraction_comparison,
        "intervals": plot_intervals_comparison,
        "robustness": plot_robust_iter_comparison,
        "concept": plot_loess_concept,
        "multivariate": plot_multivariate_loess,
        "kernel": plot_kernel_comparison,
        "robust_method": plot_robust_method_comparison,
        "boundary": plot_boundary_policy_comparison,
        "higher_degree": plot_higher_degree_comparison,
        "gap": plot_gap_handling,
        "cv": plot_cv_comparison,
        "surface": plot_surface_mode_comparison,
        "scaling": plot_scaling_comparison,
        "zero_weight": plot_zero_weight_comparison,
        "degree_interp": plot_degree_interpolation_comparison,
        "streaming": plot_merge_comparison,
        "online": plot_online_comparison,
        "auto_converge": plot_adapter_comparison,
    }


def main():
    """Dispatch to one plot target or generate all figures."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_targets = build_plot_targets()
    target_name = sys.argv[1] if len(sys.argv) >= 2 else "all"

    if target_name == "all":
        for plot_target in plot_targets.values():
            plot_target()
        return 0

    plot_target = plot_targets.get(target_name)
    if plot_target is None:
        print(f"Unknown target: {target_name}")
        print("Usage: python3 plot.py [target] (default: all)")
        print(
            "Targets: all, degree, fraction, intervals, robustness, concept, multivariate, "
            "kernel, robust_method, boundary, higher_degree, gap, cv, surface, scaling, "
            "zero_weight, degree_interp, streaming, online, auto_converge"
        )
        return 1

    plot_target()
    return 0


if __name__ == "__main__":
    sys.exit(main())
