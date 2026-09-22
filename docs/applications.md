# Application benchmarks

The suite contains nine cases: database top-N, token sampling and gradient compression, each with small, middle and large inputs.

## Inputs

| Application | Operation | Small / middle / large inputs |
| --- | --- | --- |
| Database top-N | Rank rows by FP32 discounted price and gather the top-k rows in descending order | TPC-H SF 0.1 / 1 / 10: 600,572 / 6,001,215 / 59,986,052 rows |
| Token sampling | Select exactly k logits, normalize their probabilities and draw one token per sequence | 8 / 128 / 512 sequences, each with 50,257 logits |
| Gradient compression | Select the k largest magnitudes and gather original signed values and indices | Complete attention / MLP / embedding gradients: 589,824 / 2,359,296 / 38,597,376 elements |

DuckDB generates the TPC-H columns. Model inputs are captured in CPU FP32 from a pinned DistilGPT-2 checkpoint; manifests record the revision, prompts and capture settings. Sampling captures the last nonpadding token's logits. Compression captures gradients of mean next-token loss for `transformer.h.0.attn.c_proj.weight`, `transformer.h.0.mlp.c_fc.weight` and `transformer.wte.weight`, respectively.

## Measurements and scaling

Each configuration uses 10 warmups and 30 measured repetitions, with CPU correctness checks outside timing. Inputs remain resident on the GPU. Full-operator timing includes score transformation, selection and output processing, measured by a CPU wall clock including launch and synchronization costs. Allocation, loading, transfers and model inference/backpropagation are excluded.

Transformation, selection and output are also measured separately. These isolated timings come from separate invocations and must not be added to reconstruct full-operator time. Sampling latency covers the complete batch.

Larger database and gradient inputs lengthen one query; larger sampling batches add independent queries. Ordinary bits uses one block per query, while split bits distributes each query across blocks. At fixed k, the retained fraction decreases as the candidate count grows. Gradient parameters also have different value distributions, so these cases do not isolate size alone or measure training quality.

## Parameters

The applications use k = 32, 64, 128, 256, 512 and 1024. For each k, the batch scripts test the following configurations:

| Backend | Threads per block | Items per thread | Split degree |
| --- | --- | --- | --- |
| bits (`bits-prefetch`) | 128, 256, 512 | 4, 7, 8, 13, 16 | 1 |
| bits (split) (`bits-sq`) | 128, 256, 512 | 4, 7, 8, 13, 16 | 8, 32, 128, 512 |

Split degree is the number of partitions selected independently before merging. Both variants enable prefetch. AIR Top-K, GridSelect and BlockSelect each run once per k.

## Requirements

Run from the repository root with the [CUDA build requirements](../README.md#requirements) and Python 3.11 or later. Install the export and plotting dependencies once:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r scripts/requirements-applications.txt
.venv/bin/python -m pip install -r scripts/requirements.txt

python3 -m venv .venv-model
.venv-model/bin/python -m pip install torch==2.10.0 \
    --index-url https://download.pytorch.org/whl/cpu
.venv-model/bin/python -m pip install -r scripts/requirements-model-applications.txt
```

## Run the benchmarks

Prepare the inputs, build and test the applications, run the benchmarks, then plot the results:

```bash
./local-build.sh "$(hostname)" native applications-prepare
NPROC=4 ./local-build.sh "$(hostname)" native applications-build
./local-build.sh "$(hostname)" native applications-run
scripts/plot-all.sh application-scaling
```

Preparation creates `data/application-inputs/scaling/suite.json` and reuses validated captures. The first capture needs network access for the model and DuckDB extension; append `--cache-directory PATH --local-files-only` to use a populated model cache.

The [Chimera wrappers](used-setup.md) accept the same actions as `local-build.sh`.

## Results and resume

Runs create a directory under `data/application-scaling/`, named by worker and job ID on Slurm or worker, timestamp and PID locally. The coordinator's `data/application-scaling-<node>-<job-id>.err` log records the Slurm output directory. Set `--output-dir PATH` to choose one explicitly.

The plotting command above discovers completed studies recursively and writes their plots under `plots/application-scaling/`. It needs no GPU; append an `index.json` path to plot a single study.

Resume with the original wrapper, options and output directory, adding `--resume`:

```bash
./local-build.sh "$(hostname)" native applications-run \
    --output-dir data/application-scaling/NAME-RUN --resume
```

Resume requires the same inputs, settings, binaries, runner code, host and GPU. Start a new run after any of these change. Use `applications-run --help` for other run options.

## Plots

Each study produces 12 paper PDFs. Every PDF has three panels (database top-N, token sampling, gradient compression), for one size, tuning mode, and measured phase:

| File | Content |
| --- | --- |
| `applications-<size>-paper-operator.pdf` | Full operator, configurations selected per k |
| `applications-<size>-paper-selection.pdf` | Selection alone, using the per-k operator configurations |
| `applications-<size>-paper-global-operator.pdf` | Full operator, one bits variant and configuration across k per application |
| `applications-<size>-paper-global-selection.pdf` | Selection alone, using the global operator winner per application |
| `<paper-stem>-configs.csv` | Each plotted point's backend, degree, block size, items per thread, timings and source CSV |
| `<case>.pdf` | Detailed configurations for one application/size, one page per split degree |
| `<case>.csv` | All median times, quartiles, speedups and selection flags |

`<size>` is `small`, `middle`, or `large`. Per-k selection minimizes median full-operator time independently for bits and bits (split). Global selection maximizes geometric-mean full-operator speedup across k, using configurations measured at every k, then keeps only the winning bits variant. Exact ties prefer ordinary bits. Each application, size, and GPU chooses independently; the three panels need not use the same variant. Selection-only figures carry the operator choices rather than retuning for selection latency.

Paper legends show algorithm names without settings, and the y axis shows only speedup; operator/selection appears in the filename. Every paper PDF has its own configuration CSV. Paper plots omit BlockSelect; the detailed CSV flags `paper_selected` and `paper_global_selected` identify the plotted configurations, including the global variant choice. Detailed database plots show the full operator, while detailed tensor plots include both phases. The study plotter replaces the old individual `*-paper.pdf` and `*-paper-global.pdf` files with the combined figures. Standalone application plotters retain their individual layout and also export paper configuration CSVs.

Speedup is AIR median time divided by backend median time for the same k and phase: above 1 is faster, below 1 is slower. Error bars show backend latency quartiles converted to speedups, holding the AIR median fixed.
