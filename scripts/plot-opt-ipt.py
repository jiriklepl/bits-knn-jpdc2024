#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils
import glob
import os

# the file name is data/opt-ipt-HOSTNAME-JOBID.csv
files = glob.glob("data/opt-ipt-*-*.csv")

def plot(file, hostname, jobid):
    data = pd.read_csv(file, sep=',')

    K = data["k"].nunique()
    Q = data["query_count"].nunique()

    NUM_ROWS = Q * 3
    NUM_COLS = (K + 2) // 3

    if NUM_COLS == 0 or NUM_ROWS == 0:
        return

    fig, axes = plt.subplots(nrows=NUM_ROWS, ncols=NUM_COLS, sharey="row", sharex="row")
    fig.subplots_adjust(bottom=0.2, top=0.94, right=0.99, wspace=0.1, hspace=0.5)

    data = data.loc[(data["iteration"] >= utils.WARMUP) &
                    (data["phase"] == "selection")]

    # extract items per thread
    data["items_per_thread"] = data["items_per_thread"].apply(lambda x: int(x.split(',')[0]))

    if "deg" in data.columns:
        data_deg = data.loc[data["deg"] > 1]
        data = data.loc[data["deg"] == 1]

        # add each "deg" (1, 2, ...) to "algorithm"
        data_deg["algorithm"] = data_deg["algorithm"] + " (deg=" + data_deg["deg"].astype(str) + ")"

        data = pd.concat([data, data_deg])

    # labels on the bottom and left subplots
    for i in range(NUM_ROWS):
        axes = plt.subplot(NUM_ROWS, NUM_COLS, i * NUM_COLS + 1)
        axes.set_ylabel("time (s)")
    for j in range(NUM_COLS):
        axes = plt.subplot(NUM_ROWS, NUM_COLS, (NUM_ROWS - 1) * NUM_COLS + j + 1)
        axes.set_xticks([4, 8, 12, 16])
        axes.grid(alpha=0.4, linestyle="--")
        axes.set_xlabel("Items per thread")

    # convert pd.Axes to list
    axes = fig.axes

    colors = {}
    i = 0
    for alg, subgroup in data.groupby("algorithm"):
        # generate a color for the algorithm
        color = plt.cm.tab20(i)
        colors[alg] = color
        i += 1

    # plot the speed-up
    for (q, k), group in data.groupby(["query_count", "k"]):
        group = group.filter(items=["algorithm", "items_per_thread", "time"])

        q_index = data["query_count"].unique().tolist().index(q)
        k_index = data["k"].unique().tolist().index(k)

        assert q_index < Q and q_index >= 0
        assert k_index < K and k_index >= 0

        index = (3 * q_index + k_index // NUM_COLS) * NUM_COLS + k_index % NUM_COLS

        assert index < NUM_ROWS * NUM_COLS

        ax = axes[index]
        ax.set_title(f"k={k} q={q}")

        group = group.loc[group["items_per_thread"] >= 4]

        # plot values for given k
        for alg, subgroup in group.groupby("algorithm"):
            mean_time = subgroup.groupby('items_per_thread')['time'].mean().to_numpy()

            min_time_index = np.argmin(mean_time)
            min_time = subgroup["items_per_thread"].unique().astype(int)[min_time_index]

            # plot speed-up
            ax.errorbar(
                x=subgroup["items_per_thread"].unique().astype(int),
                y=mean_time,
                linewidth=1,
                capsize=5,
                marker='.',
                label=alg,
                color=colors[alg])

            # get color from the errorbar
            color = ax.get_lines()[-1].get_color()

            # draw vertical line at the maximum speed-up
            ax.axvline(x=min_time, color=color, linestyle='--', linewidth=1)

            # draw horizontal line at the maximum speed-up
            ax.axhline(y=mean_time[min_time_index], color=color, linestyle='--', linewidth=1)

            ax.set_xticks([4, 8, 12, 16])
            ax.set_ylim(bottom=0)
            ax.grid(alpha=0.4, linestyle="--")

    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)

    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0), frameon=False, ncol=2, title=f"bits for n={data['point_count'].iat[0]}, block_size={data['block_size'].iat[0]}, hostname={hostname}, jobid={jobid}")
    fig.set_size_inches(NUM_COLS * 2, NUM_ROWS * 2)

    # create directory if it does not exist
    os.makedirs("figures", exist_ok=True)

    fig.savefig(f"figures/opt-ipt-{hostname}-{jobid}.pdf")

    plt.close(fig)

for file in files:
    hostname, jobid = file.split(".")[-2].split("-")[-2:]
    plot(file, hostname, jobid)
