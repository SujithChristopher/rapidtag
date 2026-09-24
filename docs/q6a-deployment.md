# Radxa Dragon Q6A deployment

The Dragon Q6A uses the Qualcomm QCS6490 ARM64 SoC with four performance and
four efficiency CPU cores. RapidTag has two deployment optimizations for this
layout:

- an opt-in board-native LLVM build, enabling CPU-specific scheduling and
  AArch64/NEON optimization;
- automatic affinity of Rayon detection workers to cores above the slowest
  frequency tier.

Pose estimation itself is single-threaded. Worker affinity accelerates marker
detection but does not directly pin the Python thread that calls PnP.

## Native board build

Build on the Q6A inside the target virtual environment:

```bash
python -m pip install maturin
./scripts/build-board.sh develop
```

The script uses `-C target-cpu=native` when run on AArch64. The result is
specific to that board and must not be redistributed as a portable wheel.

Cross-compilation from another architecture is also supported by the script,
but requires the Rust AArch64 target, an AArch64 linker, and matching Python
target configuration.

## Core affinity

RapidTag reads each online core's `cpuinfo_max_freq`. On a heterogeneous Linux
system, detection workers are pinned to all cores above the slowest tier. The
calling Python/capture thread remains unpinned.

Override automatic selection when needed:

```bash
RAPIDTAG_CORES=4-7 python app.py
RAPIDTAG_CORES=all python app.py
RAYON_NUM_THREADS=4 python app.py
```

Confirm the Q6A's actual CPU numbering before setting an explicit list. `all`
disables RapidTag affinity. `RAYON_NUM_THREADS` changes the pool size without
changing the selected affinity set.

## Frequency governor and thermals

For latency measurements, put the performance cores in the `performance`
governor if the operating image exposes that governor. This is a system-level
setting requiring administrator access. Restore the deployment's desired power
policy after benchmarking and monitor thermal throttling during sustained runs.

## Benchmark on the target

Benchmark detection and pose separately:

```bash
python tests/bench.py
python scripts/bench_ransac_dome.py --samples 1000 --iterations 100
```

Record at least median and p95 latency, success rate, inlier ratio, reprojection
RMSE, CPU governor, RapidTag build flags, and affinity settings. Do not reuse
x86 measurements as Q6A pose numbers.

RapidTag does not currently offload detection or PnP to the Hexagon DSP/NPU or
Adreno GPU. For small PnP problems, dispatch overhead may outweigh accelerator
benefits; profile the CPU path before considering an offload implementation.
