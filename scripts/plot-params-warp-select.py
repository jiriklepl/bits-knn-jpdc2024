#!/usr/bin/env python3

import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import utils
import glob
import os

# the file name is data/params-warp-select-HOSTNAME-JOBID.csv
files = glob.glob("data/params-warp-select-*-*.csv")

def plot(file, hostname, jobid):
    data = pd.read_csv(file, sep=',')
    data = data.loc[(data["iteration"] >= utils.WARMUP) &
                    (data["phase"] == "selection") &
                    (data["point_count"] == 1024 * 1024)]
    data["thread_queue"] = data["items_per_thread"].apply(lambda x: int(x.split(',')[0]))
    data["throughput"] = data["point_count"] * data["query_count"] / data["time"]

    # get baseline
    baseline = data.loc[data["algorithm"] == "warp-select"]

    # filter out the baseline
    data = data.loc[data["algorithm"] == "warp-select-tunable"]

    fig, ax = plt.subplots()

    # find the best thread queue size
    selected_k = set()
    for k in data['k'].unique():
        best_q = 0
        best_time = math.inf
        for q in data['thread_queue'].unique():
            mean_time = data.loc[(data.k == k) & (data.thread_queue == q)]['time'].mean()
            if mean_time < best_time:
                best_time = mean_time
                best_q = q
        print(f"k = {k} -> q = {best_q}")
        selected_k.add(best_q)

    # show only the best results for clarity
    #data = data.loc[data["thread_queue"].isin(selected_k)]

    opt = pd.DataFrame(columns=["thread_queue", "k", "harm_mean", "harm_std"])
    for (num_points, num_queries, k), group in data.groupby(["point_count", "query_count", "k"]):
        values = (group.groupby(["thread_queue"])["throughput"]
            .agg([utils.harm_mean, utils.harm_std])
            .reset_index())
        values = values.sort_values(by=["harm_mean"], ascending=False)
        print(f"k = {k}")
        print(values)
        row = {
            "thread_queue": int(values["thread_queue"].values[0]),
            "k": int(k),
            "harm_mean": values["harm_mean"].values[0],
            "harm_std": values["harm_std"].values[0]
        }
        if opt.empty:
            opt = pd.DataFrame([row], columns=opt.columns)
        else:
            opt = pd.concat([opt, pd.DataFrame([row])])
        ax.annotate(row["thread_queue"], (k, row["harm_mean"] + 2e9))

    ax.errorbar(
        x=opt["k"].astype(int),
        y=opt["harm_mean"],
        linewidth=1.5,
        capsize=3,
        marker='.',
        label="adjusted")

    # # plot tuned configurations
    # for thread_queue, group in data.groupby("thread_queue"):
    #     group = group.filter(items=["k", "throughput"])
    #     values = (group.groupby(["k"])["throughput"]
    #         .agg([utils.harm_mean, utils.harm_std])
    #         .reset_index())
    #     ax.errorbar(
    #         x=values["k"].astype(int),
    #         y=values["harm_mean"],
    #         yerr=values["harm_std"],
    #         linewidth=1.5,
    #         capsize=3,
    #         marker='.',
    #         label=thread_queue)

    # plot baseline
    values = baseline.groupby(["k"])["throughput"].agg([utils.harm_mean, utils.harm_std]).reset_index()
    ax.errorbar(
        x=values["k"].astype(int),
        y=values["harm_mean"],
        yerr=values["harm_std"],
        linewidth=1.5,
        capsize=3,
        marker='.',
        linestyle='--',
        color='black',
        label="default")

    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xticks(values["k"].astype(int).unique())

    ticks = ax.get_yticks().tolist()
    ticks.append(0)
    ax.set_yticks(ticks, labels=[int(val / 1e10) for val in ticks])

    fig.set_size_inches(5, 3)
    fig.subplots_adjust(bottom=0.15)

    plt.legend(frameon=False, title="Thread queue:", ncol=3)
    plt.xlabel("Nearest neighbors --- k")
    plt.ylabel("Throughput [distances/s] ×$10^{10}$")
    plt.grid(alpha=0.4, linestyle="--")

    # create directory if it does not exist
    os.makedirs("plots", exist_ok=True)

    fig.savefig(f"plots/params-warp-select-{hostname}-{jobid}.pdf")

    plt.close(fig)

for file in files:
    hostname, jobid = file.split(".")[-2].split("-")[-2:]
    plot(file, hostname, jobid)
