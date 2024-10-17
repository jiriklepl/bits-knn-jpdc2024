#!/usr/bin/env python3

import math
import pandas as pd
import matplotlib.pyplot as plt
import utils
import glob
import os

# the file name is data/kselection-sp-HOSTNAME-JOBID.csv
files = glob.glob("data/kselection-sp-*-*.csv")

def plot(file, hostname, jobid):
    fig, ax = plt.subplots(nrows=3, ncols=2, sharey=True)
    fig.subplots_adjust(bottom=0.18, top=0.95, hspace=0.4, wspace=0.1)

    data = pd.read_csv(file, sep=',')
    data = data.loc[(data["iteration"] >= utils.WARMUP) &
                    (data["phase"] == "selection") &
                    (data["algorithm"].isin([
                            "bits",
                        ])) &
                    (data["point_count"] >= 32 * 1024)]
    # compute throughput
    data["throughput"] = data["point_count"] * data["query_count"] / data["time"]

    best = max(data.loc[data["algorithm"] == "bits"]["throughput"])
    print(f"peak throughput: {best / utils.MEMORY_FLOAT_THROUGHPUT(hostname)}")

    # transform labels to human readable strings
    data = data.replace({"algorithm": {
        "bits": "bits (our implementation)",
        "warp-select": "WarpSelect",
        "block-select": "BlockSelect",
    }})

    # compute the maximum throughput that is shown in the plot
    max_throughput = (math.ceil(utils.MEMORY_FLOAT_THROUGHPUT(hostname) / 1e11) - 0.3) * 1e11

    # plot the speed-up
    i = 1
    for n, group in data.groupby("point_count"):
        scaled_n = n // 1024
        group = group.filter(items=["algorithm", "k", "throughput"])
        ax = plt.subplot(3, 2, i)
        ax.set_title(f"n = {scaled_n}k, q = {1024 // scaled_n}k")
        ax.set_ylim([0, max_throughput])
        ax.set_yticks(ax.get_yticks().tolist()) # shut up the warning (other than that, does nothing)
        ax.set_yticklabels([int(val / 1e10) for val in ax.get_yticks().tolist()])
        ax.grid(alpha=0.4, linestyle="--")
        i += 1

        # setup the secondary axis
        ax2 = ax.twinx()
        ax2.set_ylim([0.0, max_throughput / utils.MEMORY_FLOAT_THROUGHPUT(hostname) * 100])
        ax2.plot([], [])

        # set label for the secondary axis
        if scaled_n in [64, 256, 1024]:
            ax2.set_yticks([0, 25, 50, 75, 100])
        else:
            ax2.set_yticks([])

        ax.axhline(
            y=utils.MEMORY_FLOAT_THROUGHPUT(hostname),
            color="black",
            linestyle=":",
            label="Theoretical peak throughput")

        # plot values for given k
        for alg, subgroup in group.groupby("algorithm"):
            subgroup = subgroup.filter(items=["k", "throughput"])
            values = (subgroup.groupby(["k"])["throughput"]
                .agg([utils.harm_mean, utils.harm_std])
                .reset_index())
            # plot speed-up
            ax.errorbar(
                x=values["k"].astype(str),
                y=values["harm_mean"],
                yerr=values["harm_std"],
                linewidth=1.5,
                capsize=3,
                label=alg,
                marker='.')


    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0), frameon=False, ncol=2)

    fig.supylabel("Throughput [distances/s] ×$10^{10}$", x=0.05)
    fig.text(0.97, 0.5, 'Throughput [\\% of peak]', va='center', rotation='vertical', fontsize='large')
    fig.supxlabel("Nearest neighbors --- k", y=0.115)
    fig.set_size_inches(6.3, 7)

    # create directory if it does not exist
    os.makedirs("figures", exist_ok=True)

    fig.savefig(f"figures/kselection-sp-{hostname}-{jobid}.pgf", bbox_inches='tight')
    fig.savefig(f"figures/kselection-sp-{hostname}-{jobid}.pdf", bbox_inches='tight')

    plt.close(fig)

for file in files:
    hostname, jobid = file.split(".")[-2].split("-")[-2:]
    plot(file, hostname, jobid)
