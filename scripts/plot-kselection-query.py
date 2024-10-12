import pandas as pd
import numpy as np
import os.path
import matplotlib.pyplot as plt
import glob
import utils

# The file name is data/kselection-query-HOSTNAME-JOBID.csv
kselection_files = glob.glob("data/kselection-query-*-*.csv")

# The file name is data/fused-HOSTNAME-JOBID.csv
fused_files = glob.glob("data/fused-*-*.csv")

# MEMORY_FLOAT_THROUGHPUT (0 : do not plot theoretical throughput)
def genFig(df : pd.DataFrame, ax : plt.Axes, title : str, algorithms : list, MEMORY_FLOAT_THROUGHPUT : float):
    i = 0
    for alg in algorithms:
        ax.plot(df.loc[df["algorithm"] == alg]["k"].astype(str), df.loc[df["algorithm"] == alg]["throughput"], utils.SHAPES[i] + '-', label=alg, color=utils.COLORS[i])

        # alg_max_throughput = df.loc[df["algorithm"] == alg]["throughput"].max()
        # ax.axhline(y=alg_max_throughput, linestyle=':', color=utils.COLORS[i], alpha=0.5, label=None)

        i += 1

    if MEMORY_FLOAT_THROUGHPUT > 0:
        ax.axhline(y=MEMORY_FLOAT_THROUGHPUT, color='black', linestyle='--', label='Throughput at memory bandwidth')

        ax2 = ax.twinx()
        ax2.set_ylim([0.0, 110])
        ax2.plot([], [])
        ax2.set_yticks([0, 25, 50, 75, 100])
        ax2.set_yticklabels(["0 %", "25 %", "50 %", "75 %", "100 %"], rotation='vertical', verticalalignment='center')
        # ax2.set_ylabel("Relative to peak throughput (%)")
        ax2.yaxis.set_label_position("right")

    ax.set_xlabel(f"k ({title})")

    xticks = df["k"].unique()
    log_xticks = np.round(np.log2(xticks)).astype(int) 
    ax.set_xticks(ticks = xticks.astype(str))
    ax.set_xticklabels([f"$2^{{{int(x)}}}$" for x in log_xticks])  

    if MEMORY_FLOAT_THROUGHPUT > 0:
        ax.set_ylim(0, MEMORY_FLOAT_THROUGHPUT * 1.1)
    else:
        ax.set_ylim(0, None)

    ax.grid(True, which="both", ls="--", alpha=0.4)

    handles,labels = ax.get_legend_handles_labels()
    return handles,labels

def drawFig(file : str, hostname : str, jobid : str, doing_fused : bool):
    data = pd.read_csv(file)

    data = data.loc[(data["iteration"] >= utils.WARMUP) & (data["algorithm"] != "warp-select") & (data["algorithm"] != "block-select") & (data["algorithm"] != "warp-select-tuned") & (data["algorithm"] != "bits")]

    if not doing_fused:
        data = data.loc[(data["phase"] == "selection")]

        # Ensure that the mean time corresponds to the selection phase
        data["time"] = data["time"]

        # Data have to be moved from the device to SMs
        MEMORY_FLOAT_THROUGHPUT = utils.MEMORY_FLOAT_THROUGHPUT(hostname)

    else:
        data = data.loc[((data["phase"] == "selection") | (data["phase"] == "distances"))]

        # Ensure that the mean time corresponds to the addition of the selection and distances phases
        data["time"] = data["time"] * 2

        instadist = data.loc[data["algorithm"].str.contains("fused") == False].copy()
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
        instadist["algorithm"] = instadist["algorithm"] + "-instadist"
        instadist.loc[loc, "time"] = instadist_load + instadist_store

        data = pd.concat([data, instadist])
        # data = pd.concat([data, half_dist])

        MEMORY_FLOAT_THROUGHPUT = 0

    # merge columns "generator" and "preprocessor" into "dataset"
    if "generator" in data.columns and "preprocessor" in data.columns:
        data["dataset"] = data["generator"] + "-" + data["preprocessor"]
        data = data.drop(columns=["generator", "preprocessor"])

        data.replace({"dataset": {
            "uniform-identity": "Uniform",
            "uniform-ascending": "Ascending",
            "uniform-descending": "Descending",
            "normal-identity": "Normal",
            "radix-adversarial-identity": "Radix adversarial",
        }}, inplace=True)
    else:
        data["dataset"] = "Uniform"

    ROWS=data["dim"].nunique()
    COLS=data["query_count"].nunique()

    algorithms = data["algorithm"].unique()

    LEGEND_COLS = algorithms.size

    if ROWS == 0 or COLS == 0:
        return # Skip empty plots

    dataset = data["dataset"].unique()
    assert len(dataset) == 1, "Multiple datasets in the same file"
    dataset = dataset[0]

    fig, axes = plt.subplots(nrows=ROWS, ncols=COLS, sharex="row")
    # fig.text(0.005, 0.5, "Throughput [distances/s]", va='center', rotation='vertical', fontsize=14)
    # fig.text(0.525, 0.05, r'$log_{2}(K)$', ha='center',fontsize=14)

    axes = fig.get_axes()

    # aggregate all measurements with the same k, query_count, point_count, and algorithm
    data = data.groupby(["k", "query_count", "point_count", "algorithm", "dim"]).agg({"time": "mean"}).reset_index()

    data["throughput"] = data["point_count"] * data["query_count"] / data["time"]

    row = 0
    for dim in sorted(data["dim"].unique()):
        data_dim=data.loc[data["dim"] == dim]

        col = 0
        for bs in sorted(data_dim["query_count"].unique()):
            data_bs=data_dim.loc[data_dim["query_count"] == bs]
            for N in sorted(data_bs["point_count"].unique()):
                index = row * COLS + col
                n_power=int(np.log2(N))
                data_N=data_bs.loc[data_bs["point_count"] == N]

                if data_N.size!=0:
                    title = f"q={bs}" + (r' n=$2^{%s}$' %(str(n_power)))
                    if data["dim"].nunique() > 1:
                        title += f" dim={dim}"

                    if index==0:
                        handles, labels = genFig(data_N,axes[index],title,algorithms,MEMORY_FLOAT_THROUGHPUT)
                    else:
                        genFig(data_N,axes[index],title,algorithms,MEMORY_FLOAT_THROUGHPUT)
            col += 1
        row += 1

    fig.set_size_inches(4.5, ROWS*3)

    # get size of the x-axis label in figure coordinates
    font_height = axes[0].xaxis.label.get_window_extent().transformed(fig.transFigure.inverted()).height

    legend = fig.legend(handles, labels, ncols=LEGEND_COLS + 1,  frameon=False, loc='lower center')

    # legend size
    try_height = 1
    while True: # until the legend fits
        try:
            legend_width = legend.get_window_extent().transformed(fig.transFigure.inverted()).width * 4.5
            plots_width = 3 * data["query_count"].nunique()
            fig.set_size_inches(max(plots_width, legend_width),
                                ROWS*3 + 1 + try_height)
            legend_height = legend.get_window_extent().transformed(fig.transFigure.inverted()).height

            # adjust the plot to make room for the legend
            fig.subplots_adjust(bottom=0.03 + legend_height + font_height * 1.3, top=.99-font_height/2, left=0.03, right=0.97, hspace=0.3)
        except ValueError:
            print(f"Legend does not fit, trying with {try_height}")
            try_height += .5
            continue
        break

    # create directory if it does not exist
    os.makedirs("figures", exist_ok=True)

    fig.savefig(file.replace("data/", "figures/").replace(".csv", f"-{dataset}.pdf"))
    plt.close(fig)

if __name__ == "__main__":
    for file in kselection_files:
        hostname, jobid = file.split(".")[-2].split("-")[-2:]
        try:
            drawFig(file, hostname, jobid, doing_fused=False)
        except Exception as e:
            print(f"Failed to plot {file}: {e}")

    for file in fused_files:
        if file.startswith("data/fused-params") or file.startswith("data/fused-cache"):
            continue

        hostname, jobid = file.split(".")[-2].split("-")[-2:]
        try:
            drawFig(file, hostname, jobid, doing_fused=True)
        except Exception as e:
            print(f"Failed to plot {file}: {e}")
