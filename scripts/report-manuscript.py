#!/usr/bin/env python3
"""Print the manuscript's experimental metrics for the selected recorded runs."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Latest evaluation jobs; tuning lists are disjoint parameter-sweep shards.
# Edit these IDs to report another set of recorded runs. No benchmarks are run.
RUNS = {
    "volta05": {
        "bitonic-sort": [108428], "buffer": [108406, 108410, 108414],
        "kselection": [108425], "distances": [108421], "fused": [108418],
        "opt-bitonic-sort": [108290], "opt-ipt": [108190, 108202, 108205],
        "opt-distances": [108258], "fused-cache-params": [108238, 108242, 108246],
    },
    "ampere01": {
        "bitonic-sort": [108429], "buffer": [108404, 108409, 108415],
        "kselection": [108426], "distances": [108422], "fused": [108419],
        "opt-bitonic-sort": [108293], "opt-ipt": [108187, 108199, 108203],
        "opt-distances": [108256], "fused-cache-params": [108241, 108245, 108248],
    },
    "ampere02": {
        "bitonic-sort": [108430], "buffer": [108405, 108411, 108413],
        "kselection": [108427], "distances": [108420], "fused": [108417],
        "opt-bitonic-sort": [108291], "opt-ipt": [108188, 108200, 108206],
        "opt-distances": [108255], "fused-cache-params": [108240, 108243, 108247],
    },
    "hopper01": {
        "bitonic-sort": [108431], "buffer": [108407, 108408, 108412],
        "kselection": [108424], "distances": [108423], "fused": [108416],
        "opt-bitonic-sort": [101133], "opt-ipt": [108189, 108201, 108204],
        "opt-distances": [101154], "fused-cache-params": [108239, 108244, 108249],
    },
    "bw01": {
        "bitonic-sort": [35760909], "buffer": [35760910, 35760911, 35760912],
        "kselection": [35760913], "distances": [35760914], "fused": [35760915],
        "opt-bitonic-sort": [], "opt-ipt": [],
        "opt-distances": [],
        "fused-cache-params": [],
    },
}
# Same bandwidth constants/convention as utils.py, without importing plotting code.
GPUS = {"volta05": ("V100", 900), "ampere01": ("L40", 864),
        "ampere02": ("A100", 1935), "hopper01": ("H100", 2039),
        "bw01": ("RTX PRO 6000 Blackwell", 1597)}
BANDWIDTH_UNIT_BYTES = 1024**3
FLOAT_BYTES = 4
WARMUP = 10
COMPETITORS = ["air-topk", "grid-select", "block-select-tunable",
               "warp-select-tunable", "warpsort", "radik"]
SELECT_QUERIES = [64, 256, 1024, 4096]
FUSED_QUERIES = [1024, 2048, 4096, 8192]
SELECT_K = [32, 64, 128, 256, 512, 1024, 2048]
FUSED_K = [8, 16, 32, 64, 128, 256]
FUSED_DIMS = [4, 8, 16]
DISTANCE_DIMS = [4, 8, 16, 32, 64, 128, 256]
REPEATS = {"opt-ipt": 30}  # Other families have 20; discard the first WARMUP.
WORKLOAD = ["point_count", "query_count", "dim", "k"]
PARAMS = ["block_size", "items_per_thread", "deg"]
CONFIG = WORKLOAD + ["algorithm", "generator", "preprocessor"] + PARAMS


def in_scope(data, family):
    """The plotted workloads and the paper's separate tuning workloads."""
    if family == "opt-ipt":
        mask = (data.point_count.eq(2**19)
                & data.query_count.isin([256, 512, 1024, 2048, 4096, 8192])
                & data.k.isin([2**i for i in range(1, 12)]) & data.dim.eq(1))
    else:
        mask = (data.point_count * data.query_count).eq(2**30)
        if family in ("distances", "opt-distances"):
            mask &= data.query_count.isin(FUSED_QUERIES) & data.dim.isin(DISTANCE_DIMS)
        elif family in ("fused", "fused-cache-params"):
            mask &= (data.query_count.isin(FUSED_QUERIES)
                     & data.dim.isin(FUSED_DIMS) & data.k.isin(FUSED_K))
        else:
            queries = [q for q in SELECT_QUERIES if q >= 256] if family == "opt-bitonic-sort" else SELECT_QUERIES
            mask &= data.query_count.isin(queries) & data.k.isin(SELECT_K) & data.dim.eq(1)
    mask &= data.generator.eq("uniform")
    if family != "buffer":
        mask &= data.preprocessor.eq("identity")
    return mask


def summarize(data, phases, repeats):
    """Average complete post-warmup iterations, summing phases within each iteration."""
    data = data[data.phase.isin(phases) & data.iteration.between(WARMUP, repeats - 1)]
    sample = CONFIG + ["iteration"]
    if data.empty or data.duplicated(sample + ["phase"]).any():
        raise ValueError("empty input or duplicate timing samples")
    if not np.isfinite(data.time).all() or (data.time < 0).any():
        raise ValueError("invalid timing value")
    times = data.pivot(index=sample, columns="phase", values="time").reindex(columns=phases)
    if times.isna().any().any():
        raise ValueError("incomplete phase pair")
    times = times.sum(axis=1).rename("time").reset_index()
    result = times.groupby(CONFIG).time.agg(["mean", "std", "count"]).reset_index()
    if (result["count"] != repeats - WARMUP).any() or (result["mean"] <= 0).any():
        raise ValueError("incomplete repetitions or nonpositive compute time")
    result["cv"] = result["std"] / result["mean"]
    return result.rename(columns={"mean": "time"})


def load(data_dir, host, family):
    phases = ["distances"] if family in ("distances", "opt-distances") else ["selection"]
    if family == "fused":
        phases = ["selection", "distances"]  # Fused kernels record zero distance-stub time.
    chunks = []
    for job in RUNS[host][family]:
        path = data_dir / f"{family}-{host}-{job}.csv"
        data = pd.read_csv(path, usecols=CONFIG + ["iteration", "phase", "time"])
        data = data[in_scope(data, family) & data.phase.isin(phases)]
        if not data.empty:
            chunks.append(data)
    if not chunks:
        return pd.DataFrame(columns=CONFIG + ["time", "std", "count", "cv"])
    return summarize(pd.concat(chunks, ignore_index=True), phases, REPEATS.get(family, 20))


def matrix(data):
    return data.groupby(WORKLOAD + ["algorithm"]).time.min().unstack("algorithm")


def ratio(table, baseline, candidate):
    return (table.reindex(columns=[baseline])[baseline]
            / table.reindex(columns=[candidate])[candidate]).dropna()


def fixed_parameters(data):
    """Minimax fixed tuple, requiring every workload measured for this algorithm."""
    if data.empty:
        return data
    data = data.copy()
    data["slowdown"] = data.time / data.groupby(WORKLOAD).time.transform("min")
    scores = data.groupby(PARAMS).slowdown.agg(["max", "count"])
    scores = scores[scores["count"] == len(data[WORKLOAD].drop_duplicates())]
    if scores.empty:
        raise ValueError("no parameter tuple covers the complete tuning grid")
    best = scores.sort_index().sort_values("max", kind="stable").index[0]
    return data[(data[PARAMS] == pd.Series(best, index=PARAMS)).all(axis=1)]


def number(value, suffix=""):
    return "—" if pd.isna(value) else f"{value:.2f}{suffix}"


def triple(values, suffix="×"):
    return " / ".join(number(v, suffix) for v in (values.mean(), values.min(), values.max()))


def wins(values, threshold=1):
    return f"{(values > threshold).sum()}/{len(values)}" if len(values) else "—"


def parameters(data):
    if data.empty:
        return "—"
    p = data.iloc[0]
    return f"{p.block_size}; {p.items_per_thread}; {p.deg}"


def witness(values, maximum=True):
    if values.empty:
        return "—"
    n, q, d, k = values.idxmax() if maximum else values.idxmin()
    return f"n={n}, q={q}, d={d}, k={k}"


def report_host(data_dir, host):
    out = {}
    bandwidth = GPUS[host][1] * BANDWIDTH_UNIT_BYTES / FLOAT_BYTES
    data = {family: load(data_dir, host, family) for family in RUNS[host]}
    sel = matrix(data["kselection"])
    speed = (sel.reindex(columns=COMPETITORS).min(axis=1) / sel["bits-prefetch"]).dropna()
    rate = sel.index.get_level_values("point_count") * sel.index.get_level_values("query_count") / sel["bits-prefetch"]
    out["bits speedup vs best competitor: mean / min / max"] = triple(speed)
    out["bits peak-speedup workload"] = witness(speed)
    out["bits minimum-speedup workload"] = witness(speed, maximum=False)
    out["bits speedup at q=64: mean"] = number(speed.xs(64, level="query_count").mean(), "×")
    out["bits wins vs best competitor"] = wins(speed)
    out["bits throughput: peak (distances/s)"] = f"{rate.max():.4e}"
    utilization = rate / bandwidth * 100
    out["bits bandwidth utilization: mean / peak"] = f"{utilization.mean():.2f}% / {utilization.max():.2f}%"
    out["Prefetch speedup gain: mean / min / max"] = triple(100 * (ratio(sel, "bits", "bits-prefetch") - 1), "%")

    sort = matrix(data["bitonic-sort"])
    speed = ratio(sort, "partial-bitonic", "partial-bitonic-regs")
    out["Sort-in-registers speedup: mean / min / max"] = triple(speed)
    out["Sort-in-registers speedup: median"] = number(speed.median(), "×")
    out["Sort-in-registers fastest"] = f"{(sort['partial-bitonic-regs'] == sort.min(axis=1)).sum()}/{len(sort)}"
    for alg in ["partial-bitonic", "partial-bitonic-warp", "partial-bitonic-warp-static", "partial-bitonic-regs"]:
        fixed = fixed_parameters(data["opt-bitonic-sort"].query("algorithm == @alg"))
        if fixed.empty:
            out[f"Fixed {alg}: block; worst slowdown"] = "—"
        else:
            out[f"Fixed {alg}: block; worst slowdown"] = f"{int(fixed.iloc[0].block_size)}; {100 * (fixed.slowdown.max() - 1):.2f}%"
    for order in ("ascending", "identity", "descending"):
        buf = matrix(data["buffer"].query("preprocessor == @order"))
        out[f"Buffer speedup ({order}): mean / min / max"] = triple(ratio(buf, "partial-bitonic-regs", "bits"))

    fixed = fixed_parameters(data["opt-ipt"].query("algorithm == 'bits-prefetch'"))
    out["Fixed bits parameters: block; items; degree"] = parameters(fixed)
    if fixed.empty:
        out["Fixed bits worst slowdown"] = "—"
    else:
        out["Fixed bits worst slowdown"] = number(100 * (fixed.slowdown.max() - 1), "%")

    dist = matrix(data["distances"])
    reference = dist.reindex(columns=["baseline-dist", "cublas-dist"]).min(axis=1)
    out["MAGMA-distance wins vs other plotted kernels"] = wins((reference / dist["magma-part-dist"]).dropna())
    tuning = data["opt-distances"]
    tuning = tuning[tuning.algorithm.isin(["baseline-dist", "cublas-dist", "magma-dist", "magma-part-dist"])]
    if tuning.empty:
        out["Distance parameters fixed per d: worst slowdown"] = "—"
    else:
        fixed_dist = pd.concat([fixed_parameters(group) for _, group in tuning.groupby(["algorithm", "dim"])])
        out["Distance parameters fixed per d: worst slowdown"] = number(100 * (fixed_dist.slowdown.max() - 1), "%")

    fused = matrix(data["fused"])
    raft = ratio(fused, "rapidsai-fused", "fused-cache")
    two_phase = ratio(fused, "bits-prefetch", "fused-cache")
    low_dim = raft[raft.index.get_level_values("dim") <= 8]
    out["bits-fused vs RAFT, d<=8: mean / min / max"] = triple(low_dim)
    out["bits-fused vs RAFT, d<=8: speedup >2×"] = wins(low_dim, 2)
    out["bits-fused wins vs RAFT"] = wins(raft)
    out["bits-fused vs two-phase: mean / min / max"] = triple(two_phase)
    out["bits-fused wins vs two-phase"] = wins(two_phase)
    for d in FUSED_DIMS:
        out[f"bits-fused wins vs two-phase, d={d}"] = wins(two_phase.xs(d, level="dim"))
    raft_k = fused.reindex(columns=["rapidsai-fused"])["rapidsai-fused"].unstack("k")
    loss = (100 * (1 - raft_k[8] / raft_k[64])).dropna()
    out["RAFT throughput loss at k=64 vs k=8: mean / min / max"] = triple(loss, "%")
    fused_k = fused["fused-cache"].unstack("k")
    out["bits-fused throughput loss at k=256 vs k=8: mean / min / max"] = triple(100 * (1 - fused_k[8] / fused_k[256]), "%")

    # Bandwidth is in floats/s, following the existing plots (including bw01).
    n, q, d = (fused.index.get_level_values(c) for c in WORKLOAD[:3])
    raw_selection = load_selection_component(data_dir, host).reindex(fused.index)
    ideal = raw_selection + ((n + q) * d + n * q) / bandwidth
    speed = (ideal / fused["fused-cache"]).dropna()
    for dim in FUSED_DIMS:
        out[f"bits-fused wins vs bits + zero computation, d={dim}"] = wins(speed.xs(dim, level="dim"))
    out["Two-phase distance matrix (GiB)"] = number(float((FLOAT_BYTES * n * q / 2**30).max()))

    fixed = fixed_parameters(data["fused-cache-params"])
    out["Fixed bits-fused parameters: query tile; items tuple; degree"] = parameters(fixed)
    if fixed.empty:
        out["Fixed bits-fused worst slowdown"] = "—"
    else:
        out["Fixed bits-fused worst slowdown"] = number(100 * (fixed.slowdown.max() - 1), "%")
    fixed_time = fixed.set_index(WORKLOAD).time
    overhead = 100 * (fixed_time / fused.reindex(columns=["rapidsai-fused"])["rapidsai-fused"] - 1)
    overhead = overhead.dropna()
    out["Fixed bits-fused vs RAFT: worst overhead (tuning/eval)"] = number(overhead.max(), "%")
    out["Fixed bits-fused vs RAFT: worst workload"] = witness(overhead)
    out["Fixed bits-fused wins vs two-phase (tuning/eval)"] = wins((fused["bits-prefetch"] / fixed_time).dropna())
    return out


def load_selection_component(data_dir, host):
    chunks = []
    for job in RUNS[host]["fused"]:
        frame = pd.read_csv(data_dir / f"fused-{host}-{job}.csv")
        data = frame[in_scope(frame, "fused") & frame.algorithm.eq("bits-prefetch")]
        if not data.empty:
            chunks.append(data)
    if not chunks:
        return pd.DataFrame(columns=WORKLOAD + ["time"])
    means = summarize(pd.concat(chunks), ["selection"], REPEATS.get("fused", 20))
    return means.set_index(WORKLOAD).time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hosts", nargs="+", choices=list(RUNS), default=list(RUNS))
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--output", type=Path, help="write Markdown here instead of stdout")
    parser.add_argument("--show-runs", action="store_true", help="append the selected job IDs")
    args = parser.parse_args()
    try:
        reports = {host: report_host(args.data_dir, host) for host in args.hosts}
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"{error}\n")
    lines = ["Ratios use mean post-warmup times; summaries weight workloads equally.",
             "Bandwidth follows scripts/utils.py. Parameter values are the recorded CSV fields.", "",
             "| Metric | " + " | ".join(f"{GPUS[h][0]} ({h})" for h in args.hosts) + " |",
             "|---|" + "---|" * len(args.hosts)]
    for metric in next(iter(reports.values())):
        lines.append("| " + metric + " | " + " | ".join(reports[h][metric] for h in args.hosts) + " |")
    if args.show_runs:
        lines += ["", "| Run family | " + " | ".join(args.hosts) + " |",
                  "|---|" + "---|" * len(args.hosts)]
        for family in RUNS[args.hosts[0]]:
            lines.append("| " + family + " | " + " | ".join(
                ", ".join(map(str, RUNS[h][family])) for h in args.hosts) + " |")
    report = "\n".join(lines) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
