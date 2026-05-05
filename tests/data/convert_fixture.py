"""Convert OnlineMoments.jl/test/X_data_small.jld2 to X_data_small.npy.

JLD2 is HDF5 with a Julia-specific layout. We read the dataset named "X" via
h5py and store it as a plain float64 NumPy array.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def main() -> None:
    here = Path(__file__).resolve().parent
    repo_root = here.parent.parent
    src = repo_root / "OnlineMoments.jl" / "test" / "X_data_small.jld2"
    dst = here / "X_data_small.npy"
    with h5py.File(src, "r") as f:
        # JLD2 stores the variable at a top-level dataset named "X"
        X = np.asarray(f["X"][:], dtype=np.float64)
    np.save(dst, X)
    print(f"wrote {dst}: shape={X.shape}, dtype={X.dtype}, min={X.min()}, max={X.max()}")


if __name__ == "__main__":
    main()
