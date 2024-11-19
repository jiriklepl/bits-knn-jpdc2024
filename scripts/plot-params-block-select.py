#!/usr/bin/env python3

import math
import pandas as pd
import matplotlib.pyplot as plt
import utils
import glob
import os

# the file name is data/params-block-select-HOSTNAME-JOBID.csv
files = glob.glob("data/params-block-select-*-*.csv")

def plot(file, hostname, jobid):
    data = pd.read_csv(file, sep=',')
    data = data.loc[(data["iteration"] >= utils.WARMUP) &
                    (data["phase"] == "selection") &
                    (data["point_count"] == 1024 * 1024)]
    data["thread_queue"] = data["items_per_thread"].apply(lambda x: int(x.split(',')[0]))

    num_points = data['point_count'].iat[0]
    num_queries = data['query_count'].iat[0]

    # find the best thread queue size
    tunable = data.loc[data.algorithm == 'block-select-tunable']
    shown_q = set()
    for k in tunable['k'].unique():
        best_q = 0
        best_time = math.inf
        for q in tunable['thread_queue'].unique():
            mean_time = tunable.loc[(tunable.k == k) & (tunable.thread_queue == q)]['time'].mean()
            if mean_time < best_time:
                best_time = mean_time
                best_q = q
        print(f"k = {k} -> q = {best_q}")
        shown_q.add(best_q)

    fig, ax = plt.subplots()

    # plot the default configuration
    baseline = data.loc[data.algorithm == 'block-select']
    baseline = baseline.groupby('k')['time'].mean().reset_index()
    baseline['throughput'] = num_points * num_queries / baseline['time']
    ax.errorbar(
        x=data["k"].unique().astype(str),
        y=baseline["throughput"],
        linewidth=1.5,
        capsize=3,
        marker='.',
        linestyle='--',
        color='black',
        label='default')

    # only show the best parameters
    tunable = tunable.loc[tunable.thread_queue.isin(shown_q)]

    # plot the tuned parameters
    for thread_queue, group in tunable.groupby("thread_queue"):
        values = group.groupby('k')['time'].mean().reset_index()
        values['throughput'] = num_points * num_queries / values['time']
        ax.errorbar(
            x=values["k"].astype(str),
            y=values["throughput"],
            linewidth=1.5,
            capsize=3,
            marker='.',
            label=thread_queue)

    fig.set_size_inches(5, 3)
    fig.subplots_adjust(bottom=0.15)

    plt.legend(frameon=False, title="Thread queue:", loc="upper right")
    plt.xlabel("Nearest neighbors --- k")
    plt.ylabel("Throughput [floats/s]")
    plt.grid(alpha=0.4, linestyle="--")

    # create directory if it does not exist
    os.makedirs("figures", exist_ok=True)

    fig.savefig(f"figures/params-block-select-{hostname}-{jobid}.pdf")

    plt.close(fig)

for file in files:
    hostname, jobid = file.split(".")[-2].split("-")[-2:]
    plot(file, hostname, jobid)
