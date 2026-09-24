# Installation

## Python package

RapidTag requires CPython 3.9 or newer. Install the published wheel with:

```bash
python -m pip install rapidtag numpy
```

Prebuilt wheels target Linux x86-64 and ARM64, macOS Intel and Apple Silicon,
and Windows x64. The Python extension uses the stable ABI, so one wheel supports
multiple compatible CPython versions.

Verify the installation:

```bash
python -c "import rapidtag; print(rapidtag.predefined_dictionaries())"
```

## Build from source

Install Rust, Python development headers, and Maturin, then run:

```bash
python -m pip install maturin
maturin develop --release
```

`develop` installs the extension into the active virtual environment. To build a
wheel instead:

```bash
maturin build --release
```

For a non-portable build optimized for the current machine:

```bash
RUSTFLAGS="-C target-cpu=native" maturin develop --release
```

Do not redistribute a `target-cpu=native` wheel: it may contain instructions
that another CPU does not support.

## Development and documentation

Install the test and documentation extras:

```bash
python -m pip install -e ".[dev,docs]"
```

Build the user guide and Rust documentation with:

```bash
mkdocs build --strict
cargo doc --no-deps --document-private-items
```

Run `python scripts/check_docs.py` after installing the current extension to
verify that the runtime API, type stub, and API reference agree.
