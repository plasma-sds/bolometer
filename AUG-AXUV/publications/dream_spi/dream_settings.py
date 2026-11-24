"""Print the ion, radiation and code settings stored in DREAM output files.

Shows what each case was simulated with: settings/eqsys/n_i (species Z,
isotopes, SPI molar fractions), settings/eqsys/spi, ionmeta, the T_cold
radiation type, the info group and the code group (DREAM commit, its date
and the local changes flag). DREAM selects atomic data by Z, so ionmeta Z,
not the species name, tells which element a species was simulated as.

    python publications/dream_spi/dream_settings.py FILE.h5 [FILE.h5 ...]
"""

import argparse

import h5py
import numpy as np

SMALL = 20  # print arrays up to this size, otherwise only the shape


def _value(data):
    """Dataset value for printing; byte arrays become strings."""
    if isinstance(data, bytes):
        return data.decode()
    if isinstance(data, np.ndarray) and data.dtype.kind == "S":
        if data.dtype.itemsize == 1:
            return b"".join(data.ravel()).decode()
        return [x.decode() for x in data.ravel()]
    if np.size(data) > SMALL:
        return f"shape {np.shape(data)}"
    return data


def _group(f, path, recursive=False):
    if path not in f:
        return
    group = f[path]
    if recursive:
        group.visititems(lambda name, obj: isinstance(obj, h5py.Dataset)
                         and print(f"  {path}/{name}:", _value(obj[()])))
        return
    for key, obj in group.items():
        shown = _value(obj[()]) if isinstance(obj, h5py.Dataset) else f"(group) {list(obj.keys())}"
        print(f"  {path}/{key}:", shown)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()
    for path in args.files:
        print(f"=== {path}")
        with h5py.File(path, "r") as f:
            print("  top-level groups:", list(f.keys()))
            for group in ("ionmeta", "settings/eqsys/n_i", "settings/eqsys/spi", "info"):
                _group(f, group)
            if "settings/eqsys/T_cold/type" in f:
                print("  settings/eqsys/T_cold/type:", _value(f["settings/eqsys/T_cold/type"][()]))
            if "settings/other/include" in f:
                print("  settings/other/include:", _value(f["settings/other/include"][()]))
            _group(f, "code", recursive=True)


if __name__ == "__main__":
    main()
