#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import utils
import glob
import os

# the file name is data/opt-bitonic-sort-HOSTNAME-JOBID.csv
files = glob.glob("data/opt-bitonic-sort-*-*.csv")

def plot(file, hostname, jobid):
    data = pd.read_csv(file, sep=',')
    data = data.loc[(data["iteration"] >= utils.WARMUP) & (data["phase"] == "selection")]
    data = data.replace({"algorithm": {
        "partial-bitonic": "baseline",
        "partial-bitonic-warp": "warp shuffles",
        "partial-bitonic-warp-static": "static warp shuffles",
        "partial-bitonic-regs": "sort in regs",
    }})

    fig, ax = plt.subplots(1, data["query_count"].nunique())
    # fig.subplots_adjust(bottom=0.1, top=0.95, left=0.2, right=0.8)

    # shapes for the algorithms (smaller markers)
    shapes = ["+", "x", "*", "^", "_"]
    alg_shapes = dict(zip(data["algorithm"].unique(), shapes))

    # number of non-warmup iterations
    counter = 1
    for query_count in data["query_count"].unique():
        query_data = data.loc[data["query_count"] == query_count]

        ax = plt.subplot(1, data["query_count"].nunique(), counter)
        counter += 1

        for block_size in query_data["block_size"].unique():
            data_block = query_data.loc[query_data["block_size"] == block_size]

            # extract baseline - partial sorting with Bitonic sort in shared memory
            # baseline = data_block.loc[data_block["algorithm"] == "baseline"]
            # baseline = baseline.filter(items=["k", "time"])
            # baseline = baseline.groupby(['k'])['time'].mean()

            # compute speed-up for each algorithm
            for alg in data_block["algorithm"].unique():
                time = data_block.loc[data_block["algorithm"] == alg]
                time = time.filter(items=["k", "time"])
                time = time.groupby(['k'])['time'].mean()
                # speedup = baseline / time
                # max_speedup = max(max_speedup, speedup.max())
                print(alg)
                print(max(time))

                # plot the speed-up
                ax.errorbar(
                    x=data_block["k"].unique().astype(str),
                    y=time,
                    linewidth=1.5,
                    capsize=3,
                    marker=alg_shapes[alg],
                    label=alg + f" (block={block_size})")

            ax.set_xlabel("Nearest neighbors --- k\n" + f"({query_count} {'queries' if query_count > 1 else 'query'})")
            ax.grid(alpha=0.4, linestyle="--")
            ax.set_ylim(0)

    fig.set_size_inches(4.5, 3)
    fig.subplots_adjust(bottom=0.37)

    # give enough space for the legend
    handles, labels = ax.get_legend_handles_labels()
    # sort both labels and handles by labels
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    legend = fig.legend(handles, labels, loc='lower center', frameon=False, ncol=2)


    # legend size
    try_height = 1
    while True:
        try:
            legend_width = legend.get_window_extent().transformed(fig.transFigure.inverted()).width * 4.5
            plots_width = 3 * data["query_count"].nunique()
            fig.set_size_inches(max(plots_width, legend_width),
                                6 + try_height)
            legend_size = legend.get_window_extent().transformed(fig.transFigure.inverted())

            # get size of the x-axis label in figure coordinates
            font_height = ax.xaxis.label.get_window_extent().transformed(fig.transFigure.inverted()).height

            # adjust the plot to make room for the legend
            fig.subplots_adjust(bottom=0.05 + legend_size.height + font_height, top=0.95)
        # until the legend fits
        except ValueError:
            try_height += .5
            continue
        break

    # create directory if it does not exist
    os.makedirs("figures", exist_ok=True)

    fig.savefig(f"figures/opt-bitonic-sort-{hostname}-{jobid}.pdf")

    plt.close(fig)

for file in files:
    hostname, jobid = file.split(".")[-2].split("-")[-2:]
    plot(file, hostname, jobid)
