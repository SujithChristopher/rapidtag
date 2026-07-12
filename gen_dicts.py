#!/usr/bin/env python3
"""Parse OpenCV predefined_dictionaries.hpp byte tables into a Rust data module.

Each C array is `name[nMarkers][4rotations][nBytes]`. We flatten each marker to a
row of `4*nBytes` bytes (rot0 bytes, rot1 bytes, rot2 bytes, rot3 bytes) which is
exactly the memory layout OpenCV's `Dictionary::bytesList` uses (bytesList.ptr(id)).
"""
import re
import sys

SRC = "opencv/modules/objdetect/src/aruco/predefined_dictionaries.hpp"
SRC_APRILTAG = "opencv/modules/objdetect/src/aruco/apriltag/predefined_dictionaries_apriltag.hpp"

# arrays we need: base 1000 tables + the two originals. Smaller dicts subset these.
ARRAYS = {
    "DICT_4X4_1000_BYTES": 2,
    "DICT_5X5_1000_BYTES": 4,
    "DICT_6X6_1000_BYTES": 5,
    "DICT_7X7_1000_BYTES": 7,
    "DICT_ARUCO_BYTES": 4,
    "DICT_ARUCO_MIP_36h12_BYTES": 5,
}

# AprilTag dictionaries live in a separate header; each is a standalone table.
ARRAYS_APRILTAG = {
    "DICT_APRILTAG_16h5_BYTES": 2,
    "DICT_APRILTAG_25h9_BYTES": 4,
    "DICT_APRILTAG_36h10_BYTES": 5,
    "DICT_APRILTAG_36h11_BYTES": 5,
}


def extract_block(text, name):
    m = re.search(re.escape(name) + r"\[\]\[4\]\[\d+\]\s*=", text)
    if not m:
        raise RuntimeError(f"array {name} not found")
    start = text.index("{", m.end())
    depth = 0
    i = start
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
        i += 1
    raise RuntimeError("unbalanced braces")


def emit(out, text, arrays):
    for name, nbytes in arrays.items():
        block = extract_block(text, name)
        nums = [int(x) for x in re.findall(r"\d+", block)]
        row_len = 4 * nbytes
        assert len(nums) % row_len == 0, (name, len(nums), row_len)
        nmarkers = len(nums) // row_len
        rust_name = name.replace("_BYTES", "")
        out.append(f"// {rust_name}: {nmarkers} markers, {nbytes} bytes/rotation")
        out.append(f"pub static {rust_name}: [[u8; {row_len}]; {nmarkers}] = [")
        for r in range(nmarkers):
            row = nums[r * row_len : (r + 1) * row_len]
            out.append("    [" + ", ".join(str(v) for v in row) + "],")
        out.append("];\n")


def main():
    out = []
    out.append("// @generated from OpenCV predefined_dictionaries.hpp — DO NOT EDIT.")
    out.append("// Layout: each marker is one row of 4*nbytes bytes (rot0|rot1|rot2|rot3).\n")

    with open(SRC) as f:
        emit(out, f.read(), ARRAYS)
    with open(SRC_APRILTAG) as f:
        emit(out, f.read(), ARRAYS_APRILTAG)

    with open("src/dictionaries_data.rs", "w") as f:
        f.write("\n".join(out))
    print(f"wrote src/dictionaries_data.rs ({len(out)} lines)")


if __name__ == "__main__":
    main()
