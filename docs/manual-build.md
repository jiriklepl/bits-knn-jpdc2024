# Manual Build for the `knn` benchmarking binary

To build the source code manually, if the process outlined in [README.md](../README.md) does not work, follow these steps:

0. Choose a `NAME` that will be used to distinguish your build from others (e.g., `build-volta`)

1. Clone the repository and update the submodules:

    ```bash
    git clone https://github.com/jiriklepl/bits-knn-jpdc2024.git
    cd bits-knn-jpdc2024
    git submodule update --init --recursive
    ```

2. Configure the build with `cmake` (replace `NAME` with the chosen name):

    ```bash
    cmake -B build-NAME -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="90" -S .
    ```

    (Replace the `CMAKE_CUDA_ARCHITECTURES` with the compute capability of your GPU, and add `-DCMAKE_CXX_COMPILER=g++-13` to explicitly set the compiler to `g++-13`)

3. Build the source code (replace `NAME` with the chosen name):

    ```bash
    cmake --build build-NAME --parallel 16
    ```

    This produces the `build/knn` executable that can be used to run the benchmarks.

    (Replace `--parallel 16` with the number of CPU cores available or reduce the number to avoid overloading the system)

    If you want to build the `knn-minimal` version or just the `test` target, run one of the following commands (replace `NAME` with the chosen name):

    ```bash
    cmake --build build-NAME --target knn-minimal
    cmake --build build-NAME --target test
    ```
