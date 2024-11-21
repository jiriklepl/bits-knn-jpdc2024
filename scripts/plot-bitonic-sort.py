#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import utils
import glob
import os

# the file name is data/bitonic-sort-HOSTNAME-JOBID.csv or data/buffer-HOSTNAME-JOBID.csv
files = glob.glob("data/bitonic-sort-*-*.csv") + glob.glob("data/buffer-*-*.csv")

def plot(file, hostname, jobid):
    data = pd.read_csv(file, sep=',')
    data = data.loc[(data["iteration"] >= utils.WARMUP) & (data["phase"] == "selection")]
    if "partial-bitonic" in data["algorithm"].unique():
        baseline_name = "baseline"
        data = data.replace({"algorithm": {
            "bits" : "bits",
            "partial-bitonic": baseline_name,
            "partial-bitonic-warp": "warp-shuffle (dynamic)",
            "partial-bitonic-warp-static": "warp-shuffle",
            "partial-bitonic-regs": "sort-in-registers",
        }})
    elif "partial-bitonic-regs" in data["algorithm"].unique():
        baseline_name = "sort-in-registers"
        data = data.replace({"algorithm": {
            "partial-bitonic-regs": baseline_name,
            "bits" : "bits (sort-in-registers + buffer)",
        }})
    else:
        raise ValueError("Unknown data")

    fig, ax = plt.subplots(1, data["query_count"].nunique())

    alg_shapes = dict(zip(data["algorithm"].unique(), zip(utils.SHAPES, utils.COLORS)))

    # number of non-warmup iterations
    counter = 1
    for query_count in data["query_count"].unique():
        query_data = data.loc[data["query_count"] == query_count]
        point_count = query_data["point_count"].unique()

        assert len(point_count) == 1

        point_count = point_count[0]

        ax = plt.subplot(1, data["query_count"].nunique(), counter)
        counter += 1

        # extract baseline - partial sorting with Bitonic sort in shared memory
        baseline = query_data.loc[query_data["algorithm"] == baseline_name]
        assert len(baseline) > 0

        baseline = baseline.groupby(['k'])['time'].mean()

        # compute speed-up for each algorithm
        for alg in query_data["algorithm"].unique():
            time = query_data.loc[query_data["algorithm"] == alg]
            time = time.groupby(['k'])['time'].mean()

            speedup = baseline / time

            print(f"Speed-up for {alg} with {query_count} queries and {point_count} points: {speedup}")

            # plot the speed-up
            ax.errorbar(
                x=query_data["k"].unique().astype(str),
                y=speedup,
                linewidth=1.5,
                capsize=3,
                marker=alg_shapes[alg][0],
                color=alg_shapes[alg][1],
                linestyle=":" if alg == baseline_name else "-",
                label=alg)

        log_point_count = np.round(np.log2(point_count)).astype(int)
        ax.set_xlabel("k " + f"(q={query_count}, n=$2^{{{log_point_count}}}$)")

        xticks = query_data["k"].unique().astype(int)
        log_xticks = np.round(np.log2(xticks)).astype(int)
        ax.set_xticks(ticks = xticks.astype(str))
        ax.set_xticklabels(f"$2^{{{int(x)}}}$" for x in log_xticks)

        ax.grid(alpha=0.4, linestyle="--")
        ax.set_ylim(0)

    fig.supylabel("Speedup", x=0.005, y=0.6)
    fig.set_size_inches(4.5, 3)

    # get size of the x-axis label in figure coordinates
    font_height = ax.xaxis.label.get_window_extent().transformed(fig.transFigure.inverted()).height

    # give enough space for the legend
    handles, labels = ax.get_legend_handles_labels()
    # sort both labels and handles by labels
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    legend = fig.legend(handles, labels, loc='lower center', frameon=False, ncol=len(labels))

    # legend size
    try_height = 1
    while True: # until the legend fits
        try:
            legend_width = legend.get_window_extent().transformed(fig.transFigure.inverted()).width * 4.5
            plots_width = 3 * data["query_count"].nunique()
            fig.set_size_inches(max(plots_width, legend_width),
                                4 + try_height)
            legend_height = legend.get_window_extent().transformed(fig.transFigure.inverted()).height * 1.1

            # adjust the plot to make room for the legend
            fig.subplots_adjust(bottom=0.07 + legend_height + font_height, top=0.97, left=0.02+font_height/2, right=0.995)
        except ValueError:
            print(f"Legend does not fit, trying with {try_height}")
            try_height += .5
            continue
        break

    # create directory if it does not exist
    os.makedirs("plots", exist_ok=True)

    fig.savefig(file.replace("data", "plots").replace("csv", "pdf"))
    plt.close(fig)

for file in files:
    hostname, jobid = file.split(".")[-2].split("-")[-2:]
    try:
        plot(file, hostname, jobid)
    except Exception as e:
        print(f"Failed to plot {file}: {e}")
