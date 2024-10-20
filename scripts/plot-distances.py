#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils
import glob
import os


# the file name is data/distances-HOSTNAME-JOBID.csv
files = glob.glob("data/distances-*-*.csv")

def plot(file, hostname, jobid):

    data = pd.read_csv(file, sep=',')
    data = data.loc[(data["iteration"] >= utils.WARMUP) & (data["phase"] == "distances")]
    # change labels to human readable strings
    data = data.replace({"algorithm": {
        "baseline-dist": "baseline",
        "magma-dist": "Modified MAGMA kernel",
        "cublas-dist": "cuBLAS GEMM + postprocessing",
        "tiled-dist": "Kuang et al.",
    }})
    data = data.filter(items=["point_count", "query_count", "algorithm", "dim", "time"])

    ROWS=1
    COLS=data["query_count"].nunique()

    #plt.yscale("log")
    plt.grid(alpha=0.4, linestyle="--")

    fig, axes = plt.subplots(nrows=1, ncols=COLS, sharey=True)

    # plot each kernel
    i = 1
    algnums = {}
    for query_count, query_data in data.groupby("query_count"):
        ax = plt.subplot(1, COLS, i)
        ax.grid(alpha=0.4, linestyle="--")
        query_data = query_data.filter(items=["algorithm", "dim", "point_count", "time"])

        point_count = query_data["point_count"].unique()
        dim = query_data["dim"].unique()

        assert len(point_count) == 1

        point_count = point_count[0]

        min_loads = (point_count + query_count) * dim
        min_stores = point_count * query_count

        min_time = (min_loads + min_stores) / utils.MEMORY_FLOAT_THROUGHPUT(hostname)
        oracle_time = (min_loads * 0 + min_stores) / utils.MEMORY_FLOAT_THROUGHPUT(hostname)

        max_throughput = point_count * query_count / min_time
        oracle_throughput = point_count * query_count / oracle_time

        ax.plot(dim.astype(str), max_throughput, label='Throughput limit', color="black", linestyle="--")
        # ax.plot(dim.astype(str), oracle_throughput, label="Oracle throughput", color="red", linestyle="-.")

        i += 1
        for alg, group in query_data.groupby("algorithm"):
            if alg not in algnums:
                algnums[alg] = len(algnums)
            group = group.filter(items=["dim", "point_count", "time"])
            values = group.groupby(["dim", "point_count"]).mean().reset_index()
            ax.errorbar(
                x=values["dim"].astype(str),
                y=values["point_count"] * query_count / values["time"],
                linewidth=1.5,
                capsize=3,
                marker=utils.SHAPES[algnums[alg]],
                color=utils.COLORS[algnums[alg]],
                label=alg)
            
        xticks = query_data["dim"].unique()
        log_x = np.round(np.log2(xticks)).astype(int)
        ax.set_xticks(xticks.astype(str))
        ax.set_xticklabels([f"$2^{{{int(x)}}}$" for x in log_x])

        ax.set_ylim(0, 1.1 * np.max(oracle_throughput))

        log_point_count = np.round(np.log2(point_count)).astype(int)
        ax.set_xlabel("d (q=%d, n=$2^{%d}$)" % (query_count, log_point_count))

    fig.supylabel("Throughput [distances/s]", x=0.01, y=0.6)
    # fig.supxlabel("d", y=0.13)
    fig.set_size_inches(4.4, 3)

    # get size of the x-axis label in figure coordinates
    font_height = ax.xaxis.label.get_window_extent().transformed(fig.transFigure.inverted()).height

    handles, labels = ax.get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc='lower center',  bbox_to_anchor=(0.5, -0.03), frameon=False, ncol=3)

    # legend size
    try_height = 1
    while True: # until the legend fits
        try:
            legend_width = legend.get_window_extent().transformed(fig.transFigure.inverted()).width * 4.5
            plots_width = 3 * data["query_count"].nunique()
            fig.set_size_inches(max(plots_width, legend_width),
                                ROWS*3 + 1 + try_height)
            legend_height = legend.get_window_extent().transformed(fig.transFigure.inverted()).height

            # adjust the plot to make room for the legend
            fig.subplots_adjust(bottom=0.03 + legend_height*1.1 + font_height, top=.96, left=0.02+font_height/2, right=0.98, hspace=0.3)
        except ValueError:
            print(f"Legend does not fit, trying with {try_height}")
            try_height += .5
            continue
        break

    # create directory if it does not exist
    os.makedirs("figures", exist_ok=True)

    fig.savefig(f"figures/distances-{hostname}-{jobid}.pdf")
    plt.close(fig)

for file in files:
    hostname, jobid = file.split(".")[-2].split("-")[-2:]
    try:
        plot(file, hostname, jobid)
    except Exception as e:
        print(f"Failed to plot {file}: {e}")
