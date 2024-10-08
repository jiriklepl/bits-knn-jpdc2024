#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils
import glob
import os

# file name is data/fused-HOSTNAME-JOBID.csv
files = glob.glob("data/fused-*-*.csv")

def plot(file, hostname, jobid):
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.3, top=0.9, left=0.2, right=0.8, hspace=0.35, wspace=0.1)

    data = pd.read_csv(file, sep=',')
    data = data.loc[(data["iteration"] >= utils.WARMUP) &
                    (data["phase"].isin(["selection", "distances"])) &
                    (data["dim"] == 16) &
                    (data["k"] == 32)]
    data["k"] = data["k"].astype(int)
    data["dim"] = data["dim"].astype(int)
    data["iteration"] = data["iteration"].astype(int)

    # compute throughput of the fused kernel
    fused = data.loc[(data["algorithm"] == "fused-regs") &
                    (data["phase"] == "selection")].reset_index()
    fused["throughput"] = fused["point_count"] * fused["query_count"] / fused["time"]

    xticks = np.sort(data["query_count"].unique()).astype(int)
    ax.set_xscale('log', base=2)
    ax.set_xlim([xticks[0], xticks[-1]])
    ax.set_xticks(xticks)
    ax.grid(alpha=0.4, linestyle="--")
    ax.set_ylabel("Throughput [distances/s] ×$10^{10}$")
    ax.set_xlabel("Number of queries")

    # plot the fused kernel
    fused_values = fused.groupby("query_count")["throughput"].agg(["mean", "std"]).reset_index()
    ax.errorbar(
        x=fused_values["query_count"].astype(int),
        y=fused_values["mean"],
        yerr=fused_values["std"],
        linewidth=1.5,
        capsize=3,
        marker='.',
        label="fused kernel")

    # compute mean baseline throughput
    baseline = data.loc[(data["algorithm"] == "bits") &
                        (data["phase"].isin(["distances", "selection"]))]
    # distance computation time + k-selection time
    baseline = baseline.groupby(["query_count", "point_count", "iteration"])["time"].sum().reset_index()
    baseline["throughput"] = baseline["query_count"] * baseline["point_count"] / baseline["time"]

    # plot baseline
    values = (baseline.groupby(["query_count", "point_count"])["throughput"]
        .agg([utils.harm_mean, utils.harm_std])
        .reset_index())
    ax.errorbar(
        x=values["query_count"].astype(int),
        y=values["harm_mean"],
        yerr=values["harm_std"],
        linewidth=1.5,
        capsize=3,
        marker='.',
        label="MAGMA distance + bits")

    # values in 5e10 increments that fit values["harm_mean"]
    yticks = np.arange(0, (int(fused_values["mean"].max()) + 10e10) // 5e10 * 5e10, 5e10)
    ax.set_yticks(yticks, labels=[int(val / 1e10) for val in yticks])
    ax.xaxis.set_major_formatter(lambda x, pos: str(int(x / 1024)) + 'k')

    fig.set_size_inches(4, 2.7)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.03), frameon=False, ncol=2)

    # create directory if not exists
    os.makedirs("figures", exist_ok=True)

    fig.savefig(f"figures/fused-config-{hostname}-{jobid}.pgf")
    fig.savefig(f"figures/fused-config-{hostname}-{jobid}.pdf")

    plt.close(fig)

for file in files:
    if file.startswith("data/fused-params") or file.startswith("data/fused-cache"):
        continue

    hostname, jobid = file.split(".")[-2].split("-")[-2:]
    plot(file, hostname, jobid)
