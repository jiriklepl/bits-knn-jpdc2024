#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils
import glob
import os

# the file name is data/parallel-HOSTNAME-JOBID.csv
files = glob.glob("data/parallel-*-*.csv")

def plot(file, hostname, jobid):
    data = pd.read_csv(file, sep=',')
    data = data.loc[(data["iteration"] >= utils.WARMUP) & (data.k.isin([512, 1024, 2048]))]

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.21, left=0.2)

    db_point_count = data['point_count'].iat[0]
    query_count = data['query_count'].iat[0]

    def mean_time(items):
        num_cat = len(items['dim'].unique())
        items_dist = items.loc[items.phase == 'distances']['time'].to_numpy().reshape([num_cat, 1, -1])
        items_sel = items.loc[items.phase == 'selection']['time'].to_numpy().reshape([num_cat, -1, 1])
        total = items_dist + items_sel
        return np.mean(total, axis=(1, 2))

    # compute mean throughput of kNN on the CPU
    for k in data['k'].unique():
        cpu_mean = mean_time(data.loc[(data.algorithm == 'parallel') & (data.k == k)])
        gpu_mean = mean_time(data.loc[(data.algorithm == 'bits') & (data.k == k)])
        speedup = cpu_mean / gpu_mean

        ax.errorbar(
            x=data["dim"].unique().astype(str),
            y=speedup,
            linewidth=1.5,
            capsize=3,
            label=k,
            marker='.')

    fig.set_size_inches(4, 3)

    ax.set_ylim(bottom=0)
    plt.legend(frameon=False, title="Nearest neighbors -- k")
    plt.xlabel("Dimension")
    plt.ylabel("GPU Speed-up")
    plt.grid(alpha=0.4, linestyle="--")

    # create directory if it does not exist
    os.makedirs("figures", exist_ok=True)

    fig.savefig(f"figures/gpu-speedup-{hostname}-{jobid}.pgf")
    fig.savefig(f"figures/gpu-speedup-{hostname}-{jobid}.pdf")

    plt.close(fig)

for file in files:
    hostname, jobid = file.split(".")[-2].split("-")[-2:]
    plot(file, hostname, jobid)
