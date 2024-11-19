#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import utils
import glob
import os

# the file name is data/parallel-HOSTNAME-JOBID.csv
files = glob.glob("data/parallel-*-*.csv")

def plot(file, hostname, jobid):
    data = pd.read_csv(file, sep=',')
    data = data.loc[(data["iteration"] >= utils.WARMUP) & (data.algorithm == 'bits')]

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.21, left=0.2)

    num_dim = len(data['dim'].unique())
    cpu_to_gpu = data.loc[(data.phase == 'transfer-in')].groupby('dim').mean(numeric_only=True)['time']

    for k in [512, 1024, 2048]:
        gpu_to_cpu = data.loc[(data.phase == 'transfer-out') & (data.k == k)].groupby('dim').mean(numeric_only=True)['time']
        trans = cpu_to_gpu + gpu_to_cpu

        comp_dist = data.loc[(data.phase == 'distances')].groupby('dim').mean(numeric_only=True)['time']
        comp_sel = data.loc[(data.phase == 'selection') & (data.k == k)].groupby('dim').mean(numeric_only=True)['time']
        comp = comp_dist + comp_sel

        ax.errorbar(
            x=data["dim"].unique().astype(str),
            y=comp / trans,
            linewidth=1.5,
            capsize=3,
            label=k,
            marker='.')

    fig.set_size_inches(4, 2.5)

    ax.set_ylim(bottom=0)
    plt.legend(frameon=False, title="Nearest neighbors -- k")
    plt.xlabel("Dimension")
    plt.ylabel("Computation / Transfers ratio")
    plt.grid(alpha=0.4, linestyle="--")

    # create directory if it does not exist
    os.makedirs("figures", exist_ok=True)

    fig.savefig(f"figures/gpu-transfers-{hostname}-{jobid}.pdf")

    plt.close(fig)

for file in files:
    hostname, jobid = file.split(".")[-2].split("-")[-2:]
    plot(file, hostname, jobid)
