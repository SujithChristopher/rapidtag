#!/usr/bin/env bash
#
# Build a RapidTag wheel tuned for a SPECIFIC ARM64 board — by default the
# Radxa Dragon Q6A (Qualcomm QCS6490 / Kryo 670: Cortex-A78 + A55, ARMv8.2-A).
#
# Why this lives here and NOT in .cargo/config.toml:
#   .cargo/config.toml is shared with the release CI, which builds the *portable*
#   arm64 wheels that must run on ANY ARM chip (Graviton, Apple Silicon, Pi, ...).
#   The flags below tune for one CPU and are NOT portable, so they belong in this
#   opt-in script. Run it once; you never hand-edit flags.
#
# Usage:
#   ./scripts/build-board.sh                # build a wheel  -> target/wheels/
#   ./scripts/build-board.sh develop        # install into the current venv
#   TARGET_CPU=cortex-a76 ./scripts/build-board.sh   # override the CPU tuning
#
set -euo pipefail
cd "$(dirname "$0")/.."

CMD="${1:-build}"          # "build" (default) or "develop"

if [ "$(uname -m)" = "aarch64" ]; then
    # Building natively ON the board: let LLVM detect the exact CPU + features.
    # This is the most reliable option — no CPU string to get wrong.
    export RUSTFLAGS="-C target-cpu=${TARGET_CPU:-native}"
    echo ">> native build on aarch64, RUSTFLAGS=$RUSTFLAGS"
    exec maturin "$CMD" --release
else
    # Cross-compiling FROM x86 FOR the board. Tune for the big cores (A78); the
    # A55 little cores share the ARMv8.2-A ISA, so it's safe across the cluster.
    TRIPLE="aarch64-unknown-linux-gnu"
    export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUSTFLAGS="-C target-cpu=${TARGET_CPU:-cortex-a78}"
    echo ">> cross build for $TRIPLE, target-cpu=${TARGET_CPU:-cortex-a78}"
    echo "   (needs: rustup target add $TRIPLE  +  an aarch64 linker/C toolchain)"
    if [ "$CMD" = "develop" ]; then
        echo "!! 'develop' can't install a cross-built wheel on this host; building instead"
        CMD="build"
    fi
    exec maturin "$CMD" --release --target "$TRIPLE"
fi
