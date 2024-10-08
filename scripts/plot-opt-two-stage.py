#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import utils
import glob
import os

# the file name is data/opt-two-stage-HOSTNAME-JOBID.csv
files = glob.glob("data/opt-two-stage-*-*.csv")

def plot(file, hostname, jobid):
    fig, ax = plt.subplots()

    data = pd.read_csv(file, sep=',')
    data = data.loc[(data["iteration"] >= utils.WARMUP) &
                    (data["phase"].isin(["selection", "postprocessing"]))]

    # get input size
    point_count = data["point_count"].values[0]
    query_count = data["query_count"].values[0]
    # number of non-warmup iterations
    num_iters = max(data["iteration"]) - utils.WARMUP + 1

    # find baseline
    baseline = point_count * query_count / data.loc[(data["algorithm"] == "cub-sort") & (data["phase"] == "selection")]["time"]
    baseline = baseline.mean()

    # filter out the sorting baseline
    data = data.loc[data["algorithm"].isin(["two-stage-sample-select", "sample-select"])]
    # compute throughput
    data["throughput"] = (data["point_count"] * data["query_count"]) / data["time"]

    data["k"] = np.round(data["k"] / point_count * 100).astype(int)
    data = data.loc[data["k"] % 10 == 0]

    # prepare data
    postprocessing = data.loc[data["phase"] == "postprocessing"]
    postprocessing = postprocessing.filter(items=["algorithm", "k", "throughput"])
    postprocessing = postprocessing.sort_values(by=["algorithm", "k"])
    selection = data.loc[data["phase"] == "selection"]
    selection = selection.filter(items=["algorithm", "k", "throughput"])
    selection = selection.sort_values(by=["algorithm", "k"])

    # plot selection + sort
    sort = postprocessing["throughput"].to_numpy().reshape(2, -1, num_iters, 1)
    sel = selection["throughput"].to_numpy().reshape(2, -1, 1, num_iters)
    sel_mean = utils.harm_mean(sel, axis=(2, 3))
    sel_std = utils.harm_std(sel, axis=(2, 3))
    # sum all pairs to get a better estimate
    # sum normalized time (inverse throughput) and invert it to get throughput
    total = 1 / (1 / sel + 1 / sort)
    total_mean = utils.harm_mean(total, axis=(2, 3))
    total_std = utils.harm_std(total, axis=(2, 3))

    labels = selection["k"].unique()

    selection_label = ["single-stage (unsorted)", "two-stage (unsorted)"]
    total_label = ["single-stage (sorted)", "two-stage (sorted)"]

    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.axhline(
        y=baseline,
        color="black",
        linestyle=":",
        label="CUB sort")

    BAR_WIDTH = 4
    for i in range(2):
        bar = ax.bar(
            labels + (i * 2 - 1) * (BAR_WIDTH / 2),
            total_mean[i],
            width=4,
            alpha=1,
            label=total_label[i])
        ax.bar(
            x=labels + (i * 2 - 1) * (BAR_WIDTH / 2),
            height=sel_mean[i] - total_mean[i], # plot the difference
            bottom=total_mean[i],
            width=4,
            color=bar.patches[0].get_facecolor(),
            alpha=0.5,
            label=selection_label[i])

    #plt.grid(alpha=0.4, linestyle="--")
    plt.xticks(labels)
    plt.xlabel("Nearest neighbors --- k [\\% of database size]")
    plt.ylabel("Throughput [pairs/s]")
    plt.legend(frameon=False)
    plt.ylim([0, np.max(sel) * 1.4])
    plt.xlim([0, 100])
    fig.set_size_inches(5, 4)

    # create directory if it does not exist
    os.makedirs("figures", exist_ok=True)

    fig.savefig(f"figures/opt-two-stage-{hostname}-{jobid}.pgf", bbox_inches='tight')
    fig.savefig(f"figures/opt-two-stage-{hostname}-{jobid}.pdf", bbox_inches='tight')

    plt.close(fig)

for file in files:
    hostname, jobid = file.split(".")[-2].split("-")[-2:]
    plot(file, hostname, jobid)
