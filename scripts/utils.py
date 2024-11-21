import math
import numpy as np
import matplotlib.pyplot as plt

# import good colors
import seaborn as sns

WARMUP = 10

SHAPES = ["o", "s", "D", "v", "P", "X", "H", "d", "p", ">", "<", "h", "8", "1", "2", "3", "4", "8", "s", "p", "P", "x", "X", "D", "d", "h", "H", "v", "^", "<", ">", "1", "2", "3", "4"]
COLORS = sns.color_palette("colorblind", 10)

def MEMORY_FLOAT_THROUGHPUT(hostname):
    """The peak Global Memory Bandwidth (in floats) for the given hostname"""
    if hostname.startswith("volta05"):
        # NVIDIA Tesla V100 32GB SXM2: https://images.nvidia.com/content/technologies/volta/pdf/tesla-volta-v100-datasheet-letter-fnl-web.pdf
        return 900.0 * 1024 * 1024 * 1024 / 4
    elif hostname.startswith("ampere01"):
        # NVIDIA L40: https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/support-guide/NVIDIA-L40-Datasheet-January-2023.pdf
        return 864.0 * 1024 * 1024 * 1024 / 4
    elif hostname.startswith("ampere02"):
        # NVIDIA A100 80GB PCIe: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-nvidia-us-2188504-web.pdf
        return 1935.0 * 1024 * 1024 * 1024 / 4
    elif hostname.startswith("hopper01"):
        # NVIDIA H100 PCIe: https://resources.nvidia.com/en-us-tensor-core
        return 2039.0 * 1024 * 1024 * 1024 / 4
    else:
        raise ValueError(f"unknown hostname: {hostname}")

plt.rcParams['axes.autolimit_mode'] = 'round_numbers'
plt.rcParams['pgf.texsystem'] = 'pdflatex'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 14
plt.rcParams['pgf.rcfonts'] = False
# plt.rcParams['text.usetex'] = True

def restricted_size(data, axis):
    if type(axis) is int:
        return data.shape[axis]
    if axis is None:
        axis = range(len(data.shape))
    n = 1
    for a in axis:
        n *= data.shape[a]
    return n

# compute harmonic mean
def harm_mean(data, axis=None):
    n = restricted_size(data, axis)
    return n / np.sum(1.0 / data, axis=axis)

# compute standard deviation of harmonic mean
def harm_std(data, axis=None):
    inv_data = 1.0 / data
    n = restricted_size(data, axis)
    return np.sqrt(
        np.var(inv_data, axis=axis) / (np.mean(inv_data, axis=axis) ** 4) / n
    )
