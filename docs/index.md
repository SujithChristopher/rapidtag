# RapidTag documentation

RapidTag is a Python package backed by a pure-Rust implementation of fiducial
marker detection and camera pose estimation. It supports ArUco, AprilTag,
ChArUco, generic chessboards, iterative PnP, RANSAC PnP, and multi-marker rigid
bodies.

## Start here

1. [Install RapidTag](installation.md).
2. Follow [marker detection](marker-detection.md) for images or camera frames.
3. Use [pose estimation](pose-estimation.md) for arbitrary 3D-2D points.
4. Use [rigid bodies](rigid-bodies.md) when several markers have fixed relative
   transforms.
5. Follow the [Dragon Q6A guide](q6a-deployment.md) for ARM64 deployment.

The [Python API reference](python-api.md) records every supported public symbol.
Copyable programs are available in the repository's `examples/` directory.

## Documentation layers

- `README.md` is the short package overview and quick start.
- These guides describe complete user workflows.
- `rapidtag.pyi` is the machine-readable Python typing contract.
- `help(rapidtag)` and `help(rapidtag.<name>)` expose PyO3 docstrings at runtime.
- `cargo doc --no-deps --document-private-items` builds implementation
  documentation for Rust maintainers.

Names beginning with `_` are test hooks and are not supported public API.
