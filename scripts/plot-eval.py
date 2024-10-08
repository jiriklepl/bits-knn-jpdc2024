#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import utils
import glob
import os

# the file name is data/eval-mq-HOSTNAME-JOBID.csv
files = glob.glob("data/eval-mq-*-*.csv")

def plot(file, hostname, jobid):
    fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
    fig.subplots_adjust(bottom=0.23, top=0.95, right=0.95, hspace=0.2, wspace=0.1)

    data = pd.read_csv(file, sep=',')
    data = data.loc[(data["iteration"] >= utils.WARMUP) &
                    (data["phase"].isin(["distances", "selection"])) &
                    (data["query_count"] == 8192) &
                    (data["dim"].isin([16, 32, 64, 128]))]

    data["k"] = data["k"].astype(int)
    data["dim"] = data["dim"].astype(int)
    data = data.filter(items=["phase", "algorithm", "iteration", "point_count", "query_count", "dim", "k", "time"])

    data = data.replace({"algorithm": {
        "bits": "bits",
    }})

    dist = data.loc[(data["algorithm"] == "magma-dist") & (data["phase"] == "distances")]
    selection = data.loc[(data["algorithm"] != "magma-dist") & (data["phase"] == "selection")]

    # number of non-warmup iterations
    num_iters = int(max(data["iteration"]) - utils.WARMUP + 1)
    point_count = data["point_count"].values[0]
    query_count = data["query_count"].values[0]
    xticks = np.sort(data["k"].unique()).astype(int)[::2]

    i = 1
    for dim in dist["dim"].unique():
        ax = plt.subplot(2, 2, i)
        ax.set_xscale('log', base=10)
        ax.set_yscale('log', base=10)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_title(f"d = {dim}")
        ax.grid(alpha=0.4, linestyle="--")
        ax.set_xticks(xticks)
        i += 1

        # get time to compute distances
        dist_time = dist.loc[dist["dim"] == dim]["time"].to_numpy().reshape(1, num_iters, 1)

        # group selection kernels by algorithm
        for alg, group in selection.loc[selection["algorithm"] != "fused-regs"].groupby("algorithm"):
            sel_time = group["time"].to_numpy().reshape(-1, 1, num_iters)
            total = point_count * query_count / (sel_time + dist_time)
            total_mean = utils.harm_mean(total, axis=(1, 2))
            total_std = utils.harm_std(total, axis=(1, 2))

            ax.errorbar(
                x=group["k"].unique().astype(int),
                y=total_mean,
                yerr=total_std,
                linewidth=1.5,
                capsize=3,
                label=alg,
                marker='.')

        # plot the fused kernel
        fused = selection.loc[(selection["algorithm"] == "fused-regs") & (selection["dim"] == dim)]
        fused_throughput = point_count * query_count / fused["time"]
        fused_throughput = fused_throughput.to_numpy().reshape(-1, num_iters)
        fused_mean = utils.harm_mean(fused_throughput, axis=1)
        fused_std = utils.harm_std(fused_throughput, axis=1)

        ax.errorbar(
            x=fused["k"].unique().astype(int),
            y=fused_mean,
            yerr=fused_std,
            linewidth=1.5,
            capsize=3,
            label="fused kNN",
            marker='.')
        ax.set_xticklabels(ax.get_xticks(), rotation=45)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.01), frameon=False, ncol=2)

    fig.supylabel("Throughput [distances/s, log]")
    fig.supxlabel("Nearest neighbors --- k", y=0.12)
    fig.set_size_inches(6.3, 6)

    # create directory if it does not exist
    os.makedirs("figures", exist_ok=True)

    fig.savefig(f"figures/eval-mq-{hostname}-{jobid}.pgf", bbox_inches='tight')
    fig.savefig(f"figures/eval-mq-{hostname}-{jobid}.pdf", bbox_inches='tight')

    plt.close(fig)

for file in files:
    hostname, jobid = file.split(".")[-2].split("-")[-2:]
    plot(file, hostname, jobid)
