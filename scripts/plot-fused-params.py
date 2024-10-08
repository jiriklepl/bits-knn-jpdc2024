#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import utils
import glob
import os

# the file name is data/fused-params-HOSTNAME-JOBID.csv
files = glob.glob("data/fused-params-*-*.csv")

def plot(file, hostname, jobid):
    fig, ax = plt.subplots(nrows=3, ncols=2, sharey=True)
    fig.subplots_adjust(bottom=0.19, top=0.93, hspace=0.45, wspace=0.1)

    data = pd.read_csv(file, sep=',')
    data = data.loc[(data["iteration"] >= utils.WARMUP) &
                    (data["phase"] == "selection")]
    data["k"] = data["k"].astype(int)
    data["dim"] = data["dim"].astype(int)
    data["iteration"] = data["iteration"].astype(int)
    data["block_size"] = data["block_size"].astype(int)

    config = data["items_per_thread"].str.split(pat=",")
    data["query_regs"] = config.str[0].astype(int)
    data["point_regs"] = config.str[1].astype(int)

    i = 1
    for dim, group in data.groupby("dim"):
        ax = plt.subplot(3, 2, i)
        ax.title.set_text(f"dim = {dim}")
        for (point_count, query_count, query_regs, point_regs, block_size), subgroup in group.groupby(["point_count", "query_count", "query_regs", "point_regs", "block_size"]):
            values = subgroup.groupby("k")["time"].mean().reset_index()
            values["throughput"] = point_count * query_count / values["time"]

            ax.errorbar(
                x=values["k"].astype(str),
                y=values["throughput"],
                linewidth=1.5,
                capsize=3,
                marker='.',
                label=f"rp = {point_regs}, rq = {query_regs}, bs = {block_size}")
        i += 1

    fig.set_size_inches(7, 7)
    fig.supylabel("Throughput [distances/s]", x=0.03)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.03), frameon=False, ncol=2)

    # create directory if it does not exist
    os.makedirs("figures", exist_ok=True)

    fig.savefig(f"figures/fused-params-{hostname}-{jobid}.pgf")
    fig.savefig(f"figures/fused-params-{hostname}-{jobid}.pdf")

    plt.close(fig)

for file in files:
    hostname, jobid = file.split(".")[-2].split("-")[-2:]
    plot(file, hostname, jobid)
