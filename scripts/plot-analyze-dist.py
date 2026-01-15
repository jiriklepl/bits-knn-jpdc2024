#!/usr/bin/env python3

from typing import Callable
import utils
import glob
import os
import sys

import pandas as pd
import numpy as np

# the file name is data/opt-distances-HOSTNAME-JOBID.csv
files = glob.glob("data/opt-distances-*-*.csv")

gpu_dict = {
    "ampere01": "L40",
    "ampere02": "A100",
    "volta05": "V100",
    "hopper01": "H100",
}

if not files:
    print("No data files found in data/", file=sys.stderr)
    exit(1)

data = None
for file in files:
    hostname, jobid = file.split(".")[-2].split("-")[-2:]

    try:
        more_data = pd.read_csv(file, sep=',')
    except pd.errors.EmptyDataError:
        continue

    more_data["hostname"] = hostname
    if hostname in gpu_dict:
        more_data["GPU"] = gpu_dict[hostname]
    else:
        more_data["GPU"] = "unknown"
    more_data["jobid"] = jobid

    more_data = more_data.loc[(more_data["iteration"] >= utils.WARMUP) &
                              (more_data["phase"] == "distances")]

    def get_nth(n: int) -> Callable[[str], int]:
        def get(x: str) -> int:
            split = x.split(',')
            return int(split[n]) if len(split) > n else 1

        return get

    more_data["items_per_thread2"] = more_data["items_per_thread"].apply(get_nth(1))
    more_data["items_per_thread3"] = more_data["items_per_thread"].apply(get_nth(2))
    more_data["items_per_thread"] = more_data["items_per_thread"].apply(get_nth(0))
    more_data["block_size"] = more_data["block_size"].astype(int)

    more_data["query_count"] = more_data["query_count"].astype(int)
    more_data["point_count"] = more_data["point_count"].astype(int)
    more_data["k"] = more_data["k"].astype(int)

    more_data["time"] = more_data["time"].astype(float)

    data = more_data if data is None else pd.concat([data, more_data])

if data is None:
    print("No valid data found in data/", file=sys.stderr)
    exit(1)

# fine-tuning:

def find_optima(data: pd.DataFrame,
                parameters: list[str],
                optimized_values: list[str],
                criteria: list[tuple[str, str]]) -> pd.DataFrame:
    for opt_val in list(optimized_values):
        if opt_val not in data.columns:
            print(f"Warning: optimized value {opt_val} not found in data columns", file=sys.stderr)
            optimized_values.remove(opt_val)

    crita: list[tuple[str, str]] = []
    for crit, mean in list(criteria):
        if crit not in data.columns:
            print(f"Warning: criterion {crit} not found in data columns", file=sys.stderr)
        else:
            crita.append((crit, mean))

    for param in list(parameters):
        if param not in data.columns:
            print(f"Warning: parameter {param} not found in data columns", file=sys.stderr)
            parameters.remove(param)

    aggregation = {}
    for crit, method in crita:
        aggregation[crit] = method

    ascending = [val == "mean" or val == "min" for _, val in crita]

    optima = None
    # group by the parameters
    for params, parameterized in data.groupby(parameters):
        # aggregate the measurements for the given parameters
        aggregated = parameterized.groupby(optimized_values).aggregate(aggregation)

        # sort the aggregated data by the criteria
        aggregated = aggregated.reset_index().sort_values([crit for crit, _ in crita], ascending=ascending)

        # get the first row
        aggregated = aggregated.head(1)

        # add the parameters
        for iparam, param in enumerate(parameters):
            aggregated[param] = params[iparam]

        # select only the columns of interest
        aggregated = aggregated[parameters + optimized_values + [crit for crit, _ in crita]]

        # append to the optima
        optima = pd.concat([optima, aggregated]) if optima is not None else aggregated

    if optima is None:
        return pd.DataFrame(columns=parameters + optimized_values + [crit for crit, _ in crita])

    optima.sort_values(parameters, inplace=True)

    return optima

# parameters driving the optimization
parameters = ["hostname", "GPU", "algorithm", "point_count", "query_count", "k", "dim"]

# values to optimize
optimized_values = ["block_size", "items_per_thread", "items_per_thread2", "items_per_thread3", "deg"]

# criteria to optimize
criteria = [("time", "mean")]

os.makedirs("scripts", exist_ok=True)
optima = find_optima(data, parameters, optimized_values, criteria)
with open("scripts/optima-dist.csv", "w") as f:
    optima.to_csv(f, index=False)

mean_time = data.groupby(parameters + optimized_values, dropna=False)["time"].mean().reset_index()

best_time = optima[parameters + ["time"]].rename(columns={"time": "best_time"})
mean_time = mean_time.merge(best_time, on=parameters, how="left")
mean_time = mean_time[mean_time["best_time"].notna()].copy()
mean_time["slowdown"] = mean_time["time"] / mean_time["best_time"]
mean_time.drop(columns=["best_time"], inplace=True)
mean_time.sort_values(parameters + ["slowdown"], inplace=True)

with open("scripts/mean-time-dist-with-slowdown.csv", "w") as f:
    mean_time.to_csv(f, index=False)

collapsed_dims = [dim for dim in ["point_count"] if dim in mean_time.columns]

category_dim = "category"
category_params = ["query_count", "k"]
categorization = {
}

collapsed_dims += category_params
mean_time[category_dim] = "other"

for algorithm_name, categories in categorization.items():
    for cat_name, cat_func in categories.items():
        cat_mask = cat_func(mean_time) & (mean_time["algorithm"] == algorithm_name)
        mean_time.loc[cat_mask, category_dim] = cat_name
extra_dims = [category_dim]

group_columns = [param for param in parameters + extra_dims if param not in collapsed_dims]
all_columns = [param for param in parameters + extra_dims + optimized_values if param not in collapsed_dims]

def all_count(values: pd.Series) -> int:
    return values.size

gross = mean_time.groupby(all_columns, dropna=False).agg(
    min_slowdown=("slowdown", "min"),
    max_slowdown=("slowdown", "max"),
    total_count=("slowdown", all_count),
).reset_index()
gross.sort_values(all_columns, inplace=True)

for col in ["max_slowdown"]:
    optima_gross = gross.groupby(group_columns, dropna=False)

    # eliminate those that do not have all_count equal to the maximum in the group to avoid false optima
    max_counts = optima_gross["total_count"].transform("max")
    optima_gross = gross[max_counts == gross["total_count"]].groupby(group_columns, dropna=False)

    optima_gross = optima_gross.apply(
        lambda df: df.nsmallest(1, col),
        include_groups=False
    ).reset_index(level=group_columns).reset_index(drop=True)
    with open(f"scripts/optima-dist-fixed-{col}.csv", "w") as f:
        optima_gross.to_csv(f, index=False)

    # filter data to only these optima
    merged = mean_time.merge(
        optima_gross[group_columns + optimized_values],
        on=group_columns + optimized_values,
        how="inner"
    )
    merged.sort_values(all_columns, inplace=True)
    with open(f"scripts/mean-time-dist-fixed-{col}.csv", "w") as f:
        merged.to_csv(f, index=False)
