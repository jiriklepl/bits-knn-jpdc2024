#!/bin/bash

find src include tests -type f \( -iname "*.cu" -o -iname "*.cpp" -o -iname "*.hpp" -o -iname "*.cuh" \) -exec clang-format -i {} +
