#!/usr/bin/env python3

import pandas as pd
import utils
import glob
import os

# the file name is data/opt-distances-HOSTNAME-JOBID.csv
files = glob.glob("data/opt-distances-*-*.csv")

gpu_dict = {
    "ampere01": "L40",
    "ampere02": "A100",
    "volta05": "V100",
    "hopper01": "H100",
}

data = None
for file in files:
    hostname, jobid = file.split(".")[-2].split("-")[-2:]

    more_data = pd.read_csv(file, sep=',')
    more_data["hostname"] = hostname
    if hostname in gpu_dict:
        more_data["GPU"] = gpu_dict[hostname]
    else:
        more_data["GPU"] = "unknown"
    more_data["jobid"] = jobid

    more_data = more_data.loc[(more_data["iteration"] >= utils.WARMUP) &
                              (more_data["phase"] == "distances")]

    def get_nth(n):
        def get(x):
            split = x.split(',')

            if len(split) > n:
                return split[n]

            return 1

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

# parameters driving the optimization
parameters = ["hostname", "GPU", "algorithm", "point_count", "query_count", "k", "dim"]

# values to optimize
optimized_values = ["block_size", "items_per_thread", "items_per_thread2", "items_per_thread3", "deg"]

# criteria to optimize
criteria = {"time" : "min"}

for opt_val in optimized_values:
    if opt_val not in data.columns:
        optimized_values.remove(opt_val)

for crit in criteria.keys():
    if crit not in data.columns:
        criteria.pop(crit)

for param in parameters:
    if param not in data.columns:
        parameters.remove(param)

optima = pd.DataFrame(columns=parameters + optimized_values)

# group by the parameters
for params, parameterized in data.groupby(parameters):
    # aggregate the measurements for the given parameters
    aggregation = {}
    for crit in criteria.keys():
        aggregation[crit] = "mean"

    ascending = [val == "min" for val in criteria.values()]
    aggregated = parameterized.groupby(optimized_values).aggregate(aggregation)

    # sort the aggregated data by the criteria
    aggregated = aggregated.reset_index().sort_values(list(criteria.keys()), ascending=ascending)

    # get the first row
    aggregated = aggregated.head(1)

    # add the parameters
    for param in parameters:
        aggregated[param] = params[parameters.index(param)]

    # select only the columns of interest
    aggregated = aggregated[parameters + optimized_values]

    # append to the optima
    optima = pd.concat([optima, aggregated])

optima.sort_values(parameters, inplace=True)

os.makedirs("scripts", exist_ok=True)
with open("scripts/optima-dist.csv", "w") as f:
    optima.to_csv(f, index=False)
