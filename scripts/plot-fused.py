#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import utils
import glob
import os

# the file name is data/fused-HOSTNAME-JOBID.csv
files = glob.glob("data/fused-*-*.csv")

def plot(file, hostname, jobid):
    fig, ax = plt.subplots(nrows=2, ncols=2, sharey=True)
    fig.subplots_adjust(bottom=0.19, top=0.93, hspace=0.45, wspace=0.1)

    xticks = [4, 8, 16, 32, 64, 128]
    data = pd.read_csv(file, sep=',')
    data = data.loc[(data["iteration"] >= utils.WARMUP) &
                    (data["phase"].isin(["selection", "distances"])) &
                    (data["point_count"] == 524288) &
                    (data["query_count"] == 8192) &
                    (data["k"].isin(xticks)) &
                    (data["dim"].isin([8, 16, 32, 64]))]
    data["k"] = data["k"].astype(int)
    data["dim"] = data["dim"].astype(int)
    data["iteration"] = data["iteration"].astype(int)

    num_points = data["point_count"].values[0]
    num_queries = data["query_count"].values[0]

    # compute throughput of the fused kernel
    fused = data.loc[(data["algorithm"] == "fused-regs") &
                    (data["phase"] == "selection")].reset_index()
    fused["throughput"] = num_points * num_queries / fused["time"]

    # plot the fused kernel
    i = 1
    for dim, group in fused.groupby("dim"):
        values = group.groupby("k")["time"].mean().reset_index()
        ax = plt.subplot(2, 2, i)
        ax.set_xscale('log', base=2)
        ax.set_xticks(xticks)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.errorbar(
            x=values["k"].astype(int),
            y=num_points * num_queries / values["time"],
            linewidth=1.5,
            capsize=3,
            marker='.',
            label="fused kernel")
        i += 1

    # compute mean baseline throughput
    baseline = data.loc[data["algorithm"] == "bits"]
    # distance computation time + k-selection time
    baseline = baseline.groupby(["dim", "k", "iteration"])["time"].sum().reset_index()
    baseline["throughput"] = num_queries * num_points / baseline["time"]

    # plot baseline
    i = 1
    for dim, group in baseline.groupby("dim"):
        values = (group.groupby("k")["throughput"]
            .agg([utils.harm_mean, utils.harm_std])
            .reset_index())
        ax = plt.subplot(2, 2, i)
        ax.set_title(f"dimension = {int(dim)}")
        ax.errorbar(
            x=values["k"].astype(int),
            y=values["harm_mean"],
            yerr=values["harm_std"],
            linewidth=1.5,
            capsize=3,
            marker='.',
            label="MAGMA distance + bits")
        ax.hlines(y=values["harm_mean"][0], xmin=min(xticks), xmax=32, linestyle=':', linewidth=1.5, color='C1')
        ax.grid(alpha=0.4, linestyle="--")
        i += 1

        ticks = ax.get_yticks().tolist()
        ax.set_yticks(ticks, labels=[int(val / 1e10) for val in ticks])

    fig.set_size_inches(5.5, 4)
    fig.supylabel("Throughput [distances/s] ×$10^{10}$", x=0.03)
    fig.supxlabel("Nearest neighbors --- k", y=0.07)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.03), frameon=False, ncol=2)

    # create directory if it does not exist
    os.makedirs("figures", exist_ok=True)

    fig.savefig(f"figures/fused-{hostname}-{jobid}.pgf")
    fig.savefig(f"figures/fused-{hostname}-{jobid}.pdf")

    plt.close(fig)

for file in files:
    if file.startswith("data/fused-params") or file.startswith("data/fused-cache"):
        continue

    hostname, jobid = file.split(".")[-2].split("-")[-2:]
    try:
        plot(file, hostname, jobid)
    except Exception as e:
        print(f"Failed to plot {file}: {e}")
        continue
