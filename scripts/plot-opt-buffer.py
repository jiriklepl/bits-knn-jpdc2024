#!/usr/bin/env python3

import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import utils
import glob
import os

# the file name is data/opt-buffer-HOSTNAME-JOBID.csv
files = glob.glob("data/opt-buffer-*-*.csv")

def plot(file, hostname, jobid):
    data = pd.read_csv(file, sep=',')
    data = data.loc[(data["iteration"] >= utils.WARMUP) & (data["phase"] == "selection")]

    fig, ax = plt.subplots()
    plt.subplots_adjust(top=0.95, bottom=0.21)

    # number of non-warmup iterations
    num_iters = max(data["iteration"]) - utils.WARMUP + 1

    # compute speed-up for all pairs of samples with the same k
    baseline = data.loc[data["algorithm"] == "partial-bitonic-regs"]
    baseline = baseline.filter(items=["k", "time"])
    baseline = baseline.groupby(['k'])['time'].mean()
    time = data.loc[data["algorithm"] == "bits"]
    time = time.filter(items=["k", "time"])
    time = time.groupby(['k'])['time'].mean()
    speedup = baseline / time

    # plot the speed-up
    ax.errorbar(
        x=data["k"].unique().astype(str),
        y=speedup,
        linewidth=1.5,
        marker='.',
        label="with buffering",
        capsize=3)
    ax.axhline(y=1, color="black", linestyle=":", label="partial sorting")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.set_size_inches(3, 2)

    plt.legend(frameon=False, loc="center left")
    plt.xlabel("Nearest neighbors --- k")
    plt.ylabel("Speed-up")
    plt.grid(alpha=0.4, linestyle="--")
    plt.ylim(ymin=0, ymax=math.ceil(max(speedup) * 1.1))

    # create directory if it does not exist
    os.makedirs("figures", exist_ok=True)

    fig.savefig(f"figures/opt-buffer-{hostname}-{jobid}.pgf")
    fig.savefig(f"figures/opt-buffer-{hostname}-{jobid}.pdf")

    plt.close(fig)

for file in files:
    hostname, jobid = file.split(".")[-2].split("-")[-2:]
    plot(file, hostname, jobid)
