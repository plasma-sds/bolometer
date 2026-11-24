"""File locations in an Open-ADAS tree (the layout of open.adas.ac.uk/download and
of Cherab's _download_cache), and the ions the checks cover."""

from pathlib import Path

# W charge states and couplings checked; B files as (charge, name)
W_CHARGES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50, 60, 73]
W_COUPLINGS = ["ls", "ic", "ca", "cl"]
B_FILES = [(0, "llu#b0"), (1, "llu#b1"), (2, "llu#b2"), (3, "llu#b3"), (3, "llr#b3"), (4, "llr#b4")]
W_PLT = ["plt50", "plt41", "plt42", "plt89"]


def w_pec(root, coupling, charge):
    return Path(root) / "adf15" / "pec40#w" / f"pec40#w_{coupling}#w{charge}.dat"


def b_pec(root, name):
    return Path(root) / "adf15" / "pec93#b" / f"pec93#b_{name}.dat"


def w_plt(root, name):
    return Path(root) / "adf11" / name / f"{name}_w.dat"
