#!/usr/bin/env python3

import glob
import sys
import os.path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from pandas import DataFrame
import utils

# The file name is data/kselection-query-HOSTNAME-JOBID.csv or data/kselection-HOSTNAME-JOBID.csv (both can be matched with the same pattern)
kselection_files = glob.glob("data/kselection-*-*.csv")

# The file name is data/fused-HOSTNAME-JOBID.csv
fused_files = glob.glob("data/fused-*-*.csv")


# MEMORY_FLOAT_THROUGHPUT (0: do not plot theoretical throughput)
def genFig(df: DataFrame, ax: Axes, title: str, algorithms: list, max_throughput: float, MEMORY_FLOAT_THROUGHPUT: float, is_last: bool = False):
    i = 0
    for alg in algorithms:
        ax.plot(df.loc[df["algorithm"] == alg]["k"].astype(str), df.loc[df["algorithm"] == alg]["throughput"], utils.SHAPES[i] + '-', label=alg, color=utils.COLORS[i])
        i += 1

    if MEMORY_FLOAT_THROUGHPUT > 0:
        ax.axhline(y=MEMORY_FLOAT_THROUGHPUT, color='black', linestyle='--', label='Throughput limit')

        ax2 = ax.twinx()
        ax2.set_ylim([0.0, 110])
        ax2.plot([], [])
        ax2.set_yticks([0, 25, 50, 75, 100])
        if is_last:
            ax2.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], rotation='vertical', verticalalignment='center')
        else:
            ax2.set_yticklabels([])
        ax2.yaxis.set_label_position("right")

    ax.set_xlabel(f"k ({title})")

    xticks = df["k"].unique()
    log_xticks = np.round(np.log2(xticks)).astype(int)
    ax.set_xticks(ticks=xticks.astype(str))
    ax.set_xticklabels([f"$2^{{{int(x)}}}$" for x in log_xticks])

    if MEMORY_FLOAT_THROUGHPUT > 0:
        ax.set_ylim(0, MEMORY_FLOAT_THROUGHPUT * 1.1)
    else:
        ax.set_ylim(0, max_throughput * 1.1)

    ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))

    exp = np.floor(np.log10(ax.get_ylim()[1]))
    ax.yaxis.get_offset_text().set_text(f"$\\times 10^{{{int(exp)}}}$")

    ax.grid(True, which="both", ls="--", alpha=0.4)

    handles, labels = ax.get_legend_handles_labels()
    return handles, labels


def drawFigInner(file: str, hostname: str, jobid: str, doing_fused: bool, data: pd.DataFrame, filtered: bool):
    if not doing_fused:
        proposed_algorithm = "bits"

        data = data.replace({"algorithm": {
            "bits-prefetch": proposed_algorithm,
            "block-select-tunable": "BlockSelect",
            "warp-select-tunable": "WarpSelect",
            "grid-select": "GridSelect",
            "radik": "RadiK",
            "air-topk": "AIR Top-$K$",
        }})

        data = data.loc[(data["phase"] == "selection")]

        # Ensure that the mean time corresponds to the selection phase
        data["time"] = data["time"]

        # Data have to be moved from the device to SMs
        MEMORY_FLOAT_THROUGHPUT = utils.MEMORY_FLOAT_THROUGHPUT(hostname)
    else:  # if doing_fused:
        proposed_algorithm = "bits-fused"

        data = data.replace({"algorithm": {
            "bits-prefetch": "bits + MAGMA",
            "block-select-tunable": "BlockSelect",
            "fused-cache": proposed_algorithm,
            "fused-regs-tunable": "bits-fused (without MAGMA)",
            "rapidsai-fused": "raftL2-fused",
        }})

        data = data.loc[((data["phase"] == "selection") | (data["phase"] == "distances"))]

        # Ensure that the mean time corresponds to the addition of the selection and distances phases
        data["time"] = data["time"] * 2

        instadist = data.loc[data["algorithm"] == "bits + MAGMA"].copy()
        loc = instadist["phase"] == "distances"

        instadist_point_count = instadist.loc[loc, "point_count"]
        instadist_query_count = instadist.loc[loc, "query_count"]
        instadist_dim = instadist.loc[loc, "dim"]

        # The two-phase algorithm has to load all vectors from the global memory
        instadist_load = (instadist_query_count + instadist_point_count) * instadist_dim / utils.MEMORY_FLOAT_THROUGHPUT(hostname)

        # Computing the compute cost
        # instadist_compute = 2 * instadist_point_count * instadist_query_count * instadist_dim / FLOPS

        # The two-phase algorithm has to store all distances back to the global memory
        instadist_store = instadist_point_count * instadist_query_count / utils.MEMORY_FLOAT_THROUGHPUT(hostname)

        # The theoretical throughput of distance computation
        instadist["algorithm"] = "bits + zero computation"
        instadist.loc[loc, "time"] = instadist_load + instadist_store

        data = pd.concat([data, instadist])
        # data = pd.concat([data, half_dist])

        MEMORY_FLOAT_THROUGHPUT = 0  # do not plot theoretical throughput

    # merge columns "generator" and "preprocessor" into "dataset"
    if "generator" in data.columns and "preprocessor" in data.columns:
        data["dataset"] = data["generator"] + "-" + data["preprocessor"]
        data = data.drop(columns=["generator", "preprocessor"])

        data.replace({"dataset": {
            "uniform-identity": "Uniform",
            "uniform-ascending": "Ascending",
            "uniform-descending": "Descending",
            "normal-identity": "Normal",
        }}, inplace=True)
    else:
        data["dataset"] = "Uniform"

    ROWS = data["dim"].nunique()
    COLS = data["query_count"].nunique()

    algorithms = data["algorithm"].unique()

    LEGEND_COLS = algorithms.size

    if ROWS == 0 or COLS == 0:
        return  # Skip empty plots

    dataset = data["dataset"].unique()
    assert len(dataset) == 1, "Multiple datasets in the same file"
    dataset = dataset[0]

    fig, axes = plt.subplots(nrows=ROWS, ncols=COLS, sharey=True)

    axes = fig.get_axes()

    # aggregate all measurements with the same k, query_count, point_count, and algorithm
    data = data.groupby(["k", "query_count", "point_count", "algorithm", "dim"]).agg({"time": "mean"}).reset_index()

    data["throughput"] = data["point_count"] * data["query_count"] / data["time"]

    max_throughput = data["throughput"].max()

    # to scale the plots uniformly
    MAX = MEMORY_FLOAT_THROUGHPUT if MEMORY_FLOAT_THROUGHPUT > 0 else max_throughput

    row = 0
    for dim in sorted(data["dim"].unique()):
        data_dim = data.loc[data["dim"] == dim]

        col = 0
        for bs in sorted(data_dim["query_count"].unique()):
            data_bs = data_dim.loc[data_dim["query_count"] == bs]
            for N in sorted(data_bs["point_count"].unique()):
                index = row * COLS + col
                n_power = int(np.log2(N))
                data_N = data_bs.loc[data_bs["point_count"] == N]

                if data_N.size == 0:
                    continue

                title = f"q={bs}" + (r' n=$2^{%s}$' %(str(n_power)))
                if data["dim"].nunique() > 1:
                    title += f" d={dim}"

                if index == 0:
                    handles, labels = genFig(data_N, axes[index], title, algorithms, max_throughput, MEMORY_FLOAT_THROUGHPUT, is_last=col == COLS - 1)
                else:
                    genFig(data_N, axes[index], title, algorithms, max_throughput, MEMORY_FLOAT_THROUGHPUT, is_last=col == COLS - 1)

                # continue only if not evaluating the fused algorithms
                #  and not fitering out some algorithms
                if doing_fused or filtered:
                    continue

                # SotA throughput
                sota_throughput = data_N.loc[data_N["algorithm"] != proposed_algorithm].groupby("k").agg({"throughput": "max"}).reset_index()
                proposed_throughput = data_N.loc[data_N["algorithm"] == proposed_algorithm, ["k", "throughput"]]

                speedups = []
                relative_throughputs = []
                k_values = []
                for k in sota_throughput["k"].unique():
                    sota_throughput_k = sota_throughput.loc[sota_throughput["k"] == k, "throughput"]
                    proposed_throughput_k = proposed_throughput.loc[proposed_throughput["k"] == k, "throughput"]

                    if sota_throughput_k.size == 0 or proposed_throughput_k.size == 0:
                        continue

                    sota_throughput_k = sota_throughput_k.iloc[0]
                    proposed_throughput_k = proposed_throughput_k.iloc[0]

                    # speedup: sota_throughput_k < proposed_throughput_k (the proposed algorithm has higher throughput)
                    speedup = proposed_throughput_k / sota_throughput_k

                    speedups.append(speedup)
                    relative_throughputs.append(proposed_throughput_k / MAX)
                    k_values.append(k)

                # print(f"file,dataset,dim,query_count,point_count,situation,k,sota,proposed,speedup")

                if len(speedups) == 0:
                    continue

                # max_speedup
                max_speedup_index = np.argmax(speedups)
                print(f"{file},{dataset},{dim},{bs},{N},max_speedup,{k_values[max_speedup_index]},{sota_throughput.loc[sota_throughput['k'] == k_values[max_speedup_index], 'throughput'].iloc[0]},{proposed_throughput.loc[proposed_throughput['k'] == k_values[max_speedup_index], 'throughput'].iloc[0]},{speedups[max_speedup_index]}")

                # min_speedup
                min_speedup_index = np.argmin(speedups)
                print(f"{file},{dataset},{dim},{bs},{N},min_speedup,{k_values[min_speedup_index]},{sota_throughput.loc[sota_throughput['k'] == k_values[min_speedup_index], 'throughput'].iloc[0]},{proposed_throughput.loc[proposed_throughput['k'] == k_values[min_speedup_index], 'throughput'].iloc[0]},{speedups[min_speedup_index]}")

                avg_speedup = np.mean(speedups)
                print(f"{file},{dataset},{dim},{bs},{N},avg_speedup,0,0,0,{avg_speedup}")

                max_relative_throughput_index = np.argmax(relative_throughputs)
                print(f"{file},{dataset},{dim},{bs},{N},max_relative_throughput,{k_values[max_relative_throughput_index]},{MAX},{proposed_throughput.loc[proposed_throughput['k'] == k_values[max_relative_throughput_index], 'throughput'].iloc[0]},{relative_throughputs[max_relative_throughput_index]}")

                min_relative_throughput_index = np.argmin(relative_throughputs)
                print(f"{file},{dataset},{dim},{bs},{N},min_relative_throughput,{k_values[min_relative_throughput_index]},{MAX},{proposed_throughput.loc[proposed_throughput['k'] == k_values[min_relative_throughput_index], 'throughput'].iloc[0]},{relative_throughputs[min_relative_throughput_index]}")

                avg_relative_throughput = np.mean(relative_throughputs)
                print(f"{file},{dataset},{dim},{bs},{N},avg_relative_throughput,0,{MAX},{avg_relative_throughput * MAX},{avg_relative_throughput}")
            col += 1
        row += 1

        # continue only if not evaluating the fused algorithms
        #  and not fitering out some algorithms
        if doing_fused or filtered:
            continue

        # SotA throughput
        sota_throughput = data_dim.loc[data_dim["algorithm"] != proposed_algorithm].groupby(["k", "query_count", "point_count"]).agg({"throughput": "max"}).reset_index()
        proposed_throughput = data_dim.loc[data_dim["algorithm"] == proposed_algorithm, ["k", "query_count", "point_count", "throughput"]]

        speedups = []
        relative_throughputs = []
        parameters = []
        for k in sota_throughput["k"].unique():
            for bs in sota_throughput["query_count"].unique():
                for N in sota_throughput["point_count"].unique():
                    sota_throughput_k = sota_throughput.loc[(sota_throughput["k"] == k) & (sota_throughput["query_count"] == bs) & (sota_throughput["point_count"] == N), "throughput"]
                    proposed_throughput_k = proposed_throughput.loc[(proposed_throughput["k"] == k) & (proposed_throughput["query_count"] == bs) & (proposed_throughput["point_count"] == N), "throughput"]

                    if sota_throughput_k.size == 0 or proposed_throughput_k.size == 0:
                        continue

                    sota_throughput_k = sota_throughput_k.iloc[0]
                    proposed_throughput_k = proposed_throughput_k.iloc[0]

                    # speedup: sota_throughput_k < proposed_throughput_k (the proposed algorithm has higher throughput)
                    speedup = proposed_throughput_k / sota_throughput_k

                    speedups.append(speedup)
                    relative_throughputs.append(proposed_throughput_k / MAX)
                    parameters.append([k, bs, N])

        # print(f"file,dataset,dim,query_count,point_count,situation,k,sota,proposed,speedup")

        if len(speedups) == 0:
            continue

        # global_max_speedup
        max_speedup_index = np.argmax(speedups)
        max_param = parameters[max_speedup_index]
        print(f"{file},{dataset},{dim},{max_param[1]},{max_param[2]},global_max_speedup,{max_param[0]},{sota_throughput.loc[(sota_throughput['k'] == max_param[0]) & (sota_throughput['query_count'] == max_param[1]) & (sota_throughput['point_count'] == max_param[2]), 'throughput'].iloc[0]},{proposed_throughput.loc[(proposed_throughput['k'] == max_param[0]) & (proposed_throughput['query_count'] == max_param[1]) & (proposed_throughput['point_count'] == max_param[2]), 'throughput'].iloc[0]},{speedups[max_speedup_index]}")

        # min_speedup
        min_speedup_index = np.argmin(speedups)
        min_param = parameters[min_speedup_index]
        print(f"{file},{dataset},{dim},{min_param[1]},{min_param[2]},global_min_speedup,{min_param[0]},{sota_throughput.loc[(sota_throughput['k'] == min_param[0]) & (sota_throughput['query_count'] == min_param[1]) & (sota_throughput['point_count'] == min_param[2]), 'throughput'].iloc[0]},{proposed_throughput.loc[(proposed_throughput['k'] == min_param[0]) & (proposed_throughput['query_count'] == min_param[1]) & (proposed_throughput['point_count'] == min_param[2]), 'throughput'].iloc[0]},{speedups[min_speedup_index]}")

        avg_speedup = np.mean(speedups)
        print(f"{file},{dataset},{dim},0,0,global_avg_speedup,0,0,0,{avg_speedup}")

        max_relative_throughput_index = np.argmax(relative_throughputs)
        max_param = parameters[max_relative_throughput_index]
        print(f"{file},{dataset},{dim},{max_param[1]},{max_param[2]},global_max_relative_throughput,{max_param[0]},{MAX},{proposed_throughput.loc[(proposed_throughput['k'] == max_param[0]) & (proposed_throughput['query_count'] == max_param[1]) & (proposed_throughput['point_count'] == max_param[2]), 'throughput'].iloc[0]},{relative_throughputs[max_relative_throughput_index]}")

        min_relative_throughput_index = np.argmin(relative_throughputs)
        min_param = parameters[min_relative_throughput_index]
        print(f"{file},{dataset},{dim},{min_param[1]},{min_param[2]},global_min_relative_throughput,{min_param[0]},{MAX},{proposed_throughput.loc[(proposed_throughput['k'] == min_param[0]) & (proposed_throughput['query_count'] == min_param[1]) & (proposed_throughput['point_count'] == min_param[2]), 'throughput'].iloc[0]},{relative_throughputs[min_relative_throughput_index]}")

        avg_relative_throughput = np.mean(relative_throughputs)
        print(f"{file},{dataset},{dim},0,0,global_avg_relative_throughput,0,{MAX},{MAX * avg_relative_throughput},{avg_relative_throughput}")

    fig.supylabel("Throughput [distances/s]", x=0.005, y=0.6)
    fig.set_size_inches(4.5, ROWS*3)

    # get size of the x-axis label in figure coordinates
    font_height = axes[0].xaxis.label.get_window_extent().transformed(fig.transFigure.inverted()).height

    legend = fig.legend(handles, labels, ncols=LEGEND_COLS + 1,  frameon=False, loc='lower center')

    # legend size
    try_height = 1.
    while True:  # until the legend fits
        try:
            legend_width = legend.get_window_extent().transformed(fig.transFigure.inverted()).width * 4.5
            plots_width = 3 * data["query_count"].nunique()
            fig.set_size_inches(max(plots_width, legend_width), ROWS*3 + 1 + try_height)
            legend_height = legend.get_window_extent().transformed(fig.transFigure.inverted()).height

            # adjust the plot to make room for the legend
            fig.subplots_adjust(bottom=0.03 + legend_height + font_height * 1.3, top=.99-font_height/2, left=0.06, right=0.99-font_height/4, hspace=0.3, wspace=0.2)
        except ValueError:
            print(f"Legend does not fit, trying with {try_height}", file=sys.stderr)
            try_height += .5
            continue
        break

    # create directory if it does not exist
    os.makedirs("plots", exist_ok=True)

    if filtered:
        fig.savefig(file.replace("data/", "plots/").replace(".csv", f"-{dataset}.pdf"))
    else:
        fig.savefig(file.replace("data/", "plots/extra-").replace(".csv", f"-{dataset}.pdf"))

    with open(file.replace("data/", "plots/"), "w") as f:
        data.to_csv(f, index=False)

    plt.close(fig)


def drawFig(file: str, hostname: str, jobid: str, doing_fused: bool):
    data = pd.read_csv(file)

    data = data.loc[(data["iteration"] >= utils.WARMUP)]

    # remove untuned versions of BlockSelect and WarpSelect
    data = data.loc[(data["algorithm"] != "warp-select") & (data["algorithm"] != "block-select")]

    # remove the bits implementation without prefetching
    data = data.loc[(data["algorithm"] != "bits")]

    drawFigInner(file, hostname, jobid, doing_fused, data.copy(True), False)

    # remove algorithms superseded by SotA
    data = data.loc[(data["algorithm"] != "warp-select-tunable") &
                    (data["algorithm"] != "radik") &
                    (data["algorithm"] != "warpsort") &
                    (data["algorithm"] != "fused-regs-tunable")]
    drawFigInner(file, hostname, jobid, doing_fused, data, True)


def main():
    print("file,dataset,dim,query_count,point_count,situation,k,sota,proposed,speedup")

    for file in kselection_files:
        hostname, jobid = file.split(".")[-2].split("-")[-2:]
        try:
            drawFig(file, hostname, jobid, doing_fused=False)
        except Exception as e:
            print(f"Failed to plot {file}: {e}", file=sys.stderr)

    for file in fused_files:
        if file.startswith("data/fused-params") or file.startswith("data/fused-cache"):
            continue

        hostname, jobid = file.split(".")[-2].split("-")[-2:]
        try:
            drawFig(file, hostname, jobid, doing_fused=True)
        except Exception as e:
            print(f"Failed to plot {file}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
