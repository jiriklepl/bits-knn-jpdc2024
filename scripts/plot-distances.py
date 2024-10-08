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
    fig, ax = plt.subplots(nrows=2, ncols=2, sharey=True)
    fig.subplots_adjust(top=0.94, bottom=0.22, hspace=0.35, wspace=0.2)

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
    # data = data.loc[data["query_count"].isin(data["query_count"].unique())]

    assert data["query_count"].nunique() == 4, "Expected 4 different query counts"

    #plt.yscale("log")
    plt.grid(alpha=0.4, linestyle="--")

    # plot each kernel
    i = 1
    algnums = {}
    for query_count, query_data in data.groupby("query_count"):
        ax = plt.subplot(2, 2, i)
        ax.set_title("single query" if query_count == 1 else f"{query_count} queries")
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

        ax.set_ylim(0, 1.1 * np.max(oracle_throughput))

        ax2 = ax.twinx()
        ax2.set_ylim([0.0, 110])
        ax2.plot([], [])
        ax2.set_yticks([0, 50, 100])
        ax2.set_yticklabels(["0 %", "50 %", "100 %"], rotation='vertical', verticalalignment='center')


    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',  bbox_to_anchor=(0.5, -0.03), frameon=False, ncol=2)

    fig.supylabel("Throughput [distances/s]")
    fig.supxlabel("Dimension", y=0.13)
    fig.set_size_inches(6, 5)

    # create directory if it does not exist
    os.makedirs("figures", exist_ok=True)

    fig.savefig(f"figures/distances-{hostname}-{jobid}.pgf")
    fig.savefig(f"figures/distances-{hostname}-{jobid}.pdf")

    plt.close(fig)

for file in files:
    hostname, jobid = file.split(".")[-2].split("-")[-2:]
    try:
        plot(file, hostname, jobid)
    except Exception as e:
        print(f"Failed to plot {file}: {e}")
