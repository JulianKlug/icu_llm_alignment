"""Plotting utilities for consistent figure styling."""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple

# Color palette
COLORS = {
    'primary': '#2E86AB',      # Blue
    'secondary': '#A23B72',    # Magenta
    'tertiary': '#F18F01',     # Orange
    'quaternary': '#C73E1D',   # Red
    'success': '#3A7D44',      # Green
    'neutral': '#6C757D',      # Gray
    'light': '#E9ECEF',        # Light gray
    'dark': '#212529',         # Dark
}

# Color palette for multiple categories
PALETTE = [
    '#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3A7D44',
    '#6C757D', '#9B59B6', '#1ABC9C', '#E74C3C', '#34495E'
]

# Figure sizes
FIGSIZE_SINGLE = (8, 6)
FIGSIZE_WIDE = (12, 6)
FIGSIZE_TALL = (8, 10)
FIGSIZE_SQUARE = (8, 8)


def setup_plotting():
    """Set up consistent plotting style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    })


def save_figure_variants(
    fig: plt.Figure,
    base_name: str,
    output_dir: Path,
    variant_num: int,
    formats: List[str] = ['png', 'pdf']
):
    """
    Save figure in multiple formats.

    Args:
        fig: Matplotlib figure
        base_name: Base filename (without extension)
        output_dir: Output directory
        variant_num: Variant number (1-4)
        formats: List of formats to save
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        filename = f"{base_name}_v{variant_num}.{fmt}"
        fig.savefig(output_dir / filename)

    plt.close(fig)


def create_bar_chart(
    data: dict,
    title: str,
    xlabel: str,
    ylabel: str,
    figsize: Tuple[int, int] = FIGSIZE_WIDE,
    color: str = COLORS['primary'],
    horizontal: bool = False,
    error_bars: Optional[dict] = None
) -> plt.Figure:
    """
    Create a bar chart.

    Args:
        data: Dictionary of {label: value}
        title: Chart title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        color: Bar color
        horizontal: If True, create horizontal bars
        error_bars: Optional dict of {label: (lower_err, upper_err)}

    Returns:
        Matplotlib figure
    """
    setup_plotting()
    fig, ax = plt.subplots(figsize=figsize)

    labels = list(data.keys())
    values = list(data.values())

    if error_bars:
        yerr = [[error_bars[l][0] for l in labels],
                [error_bars[l][1] for l in labels]]
    else:
        yerr = None

    if horizontal:
        bars = ax.barh(labels, values, color=color, xerr=yerr)
        ax.set_xlabel(ylabel)
        ax.set_ylabel(xlabel)
    else:
        bars = ax.bar(labels, values, color=color, yerr=yerr, capsize=3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.xticks(rotation=45, ha='right')

    ax.set_title(title)
    plt.tight_layout()

    return fig


def create_box_plot(
    data: dict,
    title: str,
    xlabel: str,
    ylabel: str,
    figsize: Tuple[int, int] = FIGSIZE_WIDE
) -> plt.Figure:
    """
    Create a box plot.

    Args:
        data: Dictionary of {label: list of values}
        title: Chart title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    setup_plotting()
    fig, ax = plt.subplots(figsize=figsize)

    labels = list(data.keys())
    values = [data[l] for l in labels]

    bp = ax.boxplot(values, labels=labels, patch_artist=True)

    for patch, color in zip(bp['boxes'], PALETTE[:len(labels)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    return fig


def create_violin_plot(
    data: dict,
    title: str,
    xlabel: str,
    ylabel: str,
    figsize: Tuple[int, int] = FIGSIZE_WIDE
) -> plt.Figure:
    """
    Create a violin plot.

    Args:
        data: Dictionary of {label: list of values}
        title: Chart title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    setup_plotting()
    fig, ax = plt.subplots(figsize=figsize)

    labels = list(data.keys())
    values = [data[l] for l in labels]

    parts = ax.violinplot(values, positions=range(len(labels)), showmeans=True, showmedians=True)

    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(PALETTE[i % len(PALETTE)])
        pc.set_alpha(0.7)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.tight_layout()

    return fig


def create_radar_chart(
    data: dict,
    title: str,
    figsize: Tuple[int, int] = FIGSIZE_SQUARE
) -> plt.Figure:
    """
    Create a radar/spider chart.

    Args:
        data: Dictionary of {category: {dimension: value}}
        title: Chart title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    setup_plotting()

    categories = list(data.keys())
    dimensions = list(data[categories[0]].keys())
    n_dims = len(dimensions)

    # Compute angles
    angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    for i, cat in enumerate(categories):
        values = [data[cat][d] for d in dimensions]
        values += values[:1]  # Complete the circle

        ax.plot(angles, values, 'o-', linewidth=2, label=cat, color=PALETTE[i % len(PALETTE)])
        ax.fill(angles, values, alpha=0.25, color=PALETTE[i % len(PALETTE)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimensions, size=9)
    ax.set_title(title)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    plt.tight_layout()

    return fig


def create_heatmap(
    data: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    title: str,
    xlabel: str,
    ylabel: str,
    figsize: Tuple[int, int] = FIGSIZE_SQUARE,
    cmap: str = 'RdYlGn',
    annot: bool = True,
    fmt: str = '.2f',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None
) -> plt.Figure:
    """
    Create a heatmap.

    Args:
        data: 2D numpy array
        row_labels: Labels for rows
        col_labels: Labels for columns
        title: Chart title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        cmap: Colormap name
        annot: Whether to annotate cells
        fmt: Format string for annotations
        vmin, vmax: Color scale limits

    Returns:
        Matplotlib figure
    """
    setup_plotting()
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        data,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        xticklabels=col_labels,
        yticklabels=row_labels,
        ax=ax,
        vmin=vmin,
        vmax=vmax
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    return fig


def create_scatter_plot(
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    figsize: Tuple[int, int] = FIGSIZE_SINGLE,
    add_regression: bool = True,
    color: Optional[np.ndarray] = None,
    cmap: str = 'viridis'
) -> plt.Figure:
    """
    Create a scatter plot with optional regression line.

    Args:
        x: X values
        y: Y values
        title: Chart title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        add_regression: Whether to add regression line
        color: Optional array for point colors
        cmap: Colormap for colored points

    Returns:
        Matplotlib figure
    """
    setup_plotting()
    fig, ax = plt.subplots(figsize=figsize)

    if color is not None:
        scatter = ax.scatter(x, y, c=color, cmap=cmap, alpha=0.6, s=50)
        plt.colorbar(scatter, ax=ax)
    else:
        ax.scatter(x, y, color=COLORS['primary'], alpha=0.6, s=50)

    if add_regression:
        # Remove NaN values
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean, y_clean = x[mask], y[mask]

        if len(x_clean) > 2:
            z = np.polyfit(x_clean, y_clean, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
            ax.plot(x_line, p(x_line), '--', color=COLORS['secondary'], linewidth=2, label='Regression')
            ax.legend()

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.tight_layout()

    return fig


def create_forest_plot(
    estimates: List[float],
    cis: List[Tuple[float, float]],
    labels: List[str],
    title: str,
    xlabel: str,
    figsize: Tuple[int, int] = FIGSIZE_TALL
) -> plt.Figure:
    """
    Create a forest plot for displaying estimates with confidence intervals.

    Args:
        estimates: Point estimates
        cis: List of (lower, upper) confidence intervals
        labels: Labels for each estimate
        title: Chart title
        xlabel: X-axis label
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    setup_plotting()
    fig, ax = plt.subplots(figsize=figsize)

    n = len(estimates)
    y_pos = np.arange(n)

    # Calculate error bars
    xerr_lower = [est - ci[0] for est, ci in zip(estimates, cis)]
    xerr_upper = [ci[1] - est for est, ci in zip(estimates, cis)]

    ax.errorbar(
        estimates, y_pos,
        xerr=[xerr_lower, xerr_upper],
        fmt='o',
        color=COLORS['primary'],
        capsize=5,
        capthick=2,
        markersize=8
    )

    # Add vertical line at 0
    ax.axvline(x=0, color=COLORS['neutral'], linestyle='--', linewidth=1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.invert_yaxis()  # Labels read top-to-bottom
    plt.tight_layout()

    return fig
