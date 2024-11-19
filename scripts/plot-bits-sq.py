#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import utils
import glob
import os


# the file name is data/bits-sq-HOSTNAME-JOBID.csv
files = glob.glob("data/bits-sq-*-*.csv")

def plot(file, hostname, jobid):
    data = pd.read_csv(file, sep=',')
    data = data.loc[(data["iteration"] >= utils.WARMUP) &
                    (data["phase"] == "selection")]

    data["throughput"] = data["point_count"] * data["query_count"] / data["time"]

    data = data.loc[data["k"].isin([2048, 1024, 512, 256, 128, 64, 32])]

    # compute % of peak throughput
    best = max(data["throughput"])
    print(best / utils.MEMORY_FLOAT_THROUGHPUT(hostname))

    worse = min(data.loc[(data["k"] == 2048) & (data["deg"] == 64)]["throughput"])
    print(worse / utils.MEMORY_FLOAT_THROUGHPUT(hostname))

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.21)

    # plot the throughput
    for (k, point_count, query_count), group in data.groupby(["k", "point_count", "query_count"]):
        # compute mean throughput
        values = group.filter(items=["deg", "time"])
        values = values.groupby("deg")["time"].mean().reset_index()

        ax.errorbar(
            x=values["deg"].astype(str),
            y=point_count * query_count / values["time"],
            linewidth=1.5,
            marker='.',
            label=k,
            capsize=3)

    ticks = ax.get_yticks().tolist()
    ax.set_yticks(ticks, labels=[int(val / 1e10) for val in ticks])

    fig.set_size_inches(6, 3)

    ax.set_ylim(bottom=0)
    plt.legend(frameon=False, loc="upper left", title="nearest neighbors -- k")
    plt.xlabel("Number of partitions")
    plt.ylabel("Throughput [distances/s] ×$10^{10}$")
    plt.grid(alpha=0.4, linestyle="--")

    # create directory if not exists
    os.makedirs("figures", exist_ok=True)

    fig.savefig(f"figures/bits-sq-{hostname}-{jobid}.pdf")

    plt.close(fig)

for file in files:
    hostname, jobid = file.split(".")[-2].split("-")[-2:]
    plot(file, hostname, jobid)
