"""
Plot WebSocket round-trip-time latency charts from a benchmark CSV.

Produces two SEPARATE figures:
  1. A histogram of rtt_ms, zoomed to the bulk of the distribution, with
     a note in the title for how many samples fall above the zoomed range.
  2. A rolling average of rtt_ms across the whole run.

Usage:
    from plot_latency import plot_latency_charts
    plot_latency_charts("ws_latency_0.csv")

Or from the command line:
    python plot_latency.py ws_latency_0.csv ws_latency_20.csv ws_latency_400.csv
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def plot_latency_charts(csv_path, rolling_window=500, bins=40, save_prefix=None):
    """
    Read a latency benchmark CSV and produce two separate figures:
      - a histogram of rtt_ms across the FULL range (no zoom/cropping),
        with rtt (ms) on the x-axis and sample count on the y-axis
      - a rolling average of rtt_ms over the whole run

    Parameters
    ----------
    csv_path : str
        Path to the CSV file. Must contain an 'rtt_ms' column.
    rolling_window : int
        Window size (in samples) for the rolling average.
    bins : int
        Number of histogram bins across the full data range.
    save_prefix : str, optional
        If given, saves the two figures as "<save_prefix>_hist.png" and
        "<save_prefix>_rolling.png" instead of just showing them.

    Returns
    -------
    (fig_hist, fig_roll) : the two matplotlib Figure objects
    """
    df = pd.read_csv(csv_path)
    rtt = df["rtt_ms"]

    rolling_avg = rtt.rolling(window=rolling_window, min_periods=1).mean()

    # --- Figure 1: Histogram, full range, rtt (ms) on x, samples on y ---
    bin_edges = np.linspace(rtt.min(), rtt.max(), bins + 1)
    fig_hist, ax_hist = plt.subplots(figsize=(11, 6))
    ax_hist.hist(rtt, bins=bin_edges, color="#4472C4", edgecolor="white", linewidth=1.2)
    ax_hist.set_xlim(rtt.min(), rtt.max())
    ax_hist.set_title(
        f"Distribution of round-trip time ({len(rtt):,} samples, full range)"
    )
    ax_hist.set_xlabel("rtt (ms)")
    ax_hist.set_ylabel("samples")
    ax_hist.set_xticks(bin_edges[::2])
    ax_hist.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.2f}"))
    ax_hist.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax_hist.grid(axis="y", color="lightgray", linewidth=0.6)
    ax_hist.set_axisbelow(True)
    ax_hist.spines["top"].set_visible(False)
    ax_hist.spines["right"].set_visible(False)
    fig_hist.autofmt_xdate(rotation=45)
    fig_hist.tight_layout()

    # --- Figure 2: Rolling average across the whole run ---
    fig_roll, ax_roll = plt.subplots(figsize=(11, 5))
    ax_roll.plot(rolling_avg.index, rolling_avg.values, color="#4472C4", linewidth=1)
    ax_roll.fill_between(rolling_avg.index, rolling_avg.values, color="#4472C4", alpha=0.15)
    ax_roll.set_title(f"{rolling_window}-sample rolling average across the run")
    ax_roll.set_xlabel("sample index")
    ax_roll.set_ylabel("rtt (ms)")
    fig_roll.tight_layout()

    if save_prefix:
        hist_path = f"{save_prefix}_hist.png"
        roll_path = f"{save_prefix}_rolling.png"
        fig_hist.savefig(hist_path, dpi=150)
        fig_roll.savefig(roll_path, dpi=150)
        print(f"Saved: {hist_path}")
        print(f"Saved: {roll_path}")
    else:
        plt.show()

    return fig_hist, fig_roll


if __name__ == "__main__":
    paths = sys.argv[1:]
    if not paths:
        print("Usage: python plot_latency.py file1.csv [file2.csv ...]")
        sys.exit(1)

    for path in paths:
        prefix = path.rsplit(".", 1)[0]
        plot_latency_charts(path, save_prefix=prefix)