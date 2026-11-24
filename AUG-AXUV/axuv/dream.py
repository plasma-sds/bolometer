"""
axuv.dream
----------
DREAM SPI simulation output for the 1D sightline runs.

A DREAM SPI case is a series of "init" runs (init_0.h5, init_1.h5, ...) and a
"main" run with the enhanced transport. Each file's time starts at 0. The last
step of each init file is repeated as the first step of the next, and main
starts from init_<last> at index settings/init/filetimeindex, the thermal
quench (TQ) onset. ``load_case`` joins them into one time series without the
repeated steps, with t_rel = 0 at the TQ onset.

Species are read from ionmeta (names and Z); ``species_densities`` maps them
to Cherab elements and charge states. D_th, D_fi and D_inj are all deuterium
and are summed per charge state.
"""

from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
from cherab.core.atomic import boron, carbon, deuterium, neon, tungsten

# DREAM species name (before any "_" suffix) to Cherab element
ELEMENTS = {"D": deuterium, "Ne": neon, "W": tungsten, "B": boron, "C": carbon}


@dataclass
class DreamCase:
    """Joined DREAM SPI case. Arrays have time as first axis."""

    path: Path
    t: np.ndarray  # joined time from the start of init_0, s
    t_onset: float  # TQ onset (start of main), s
    t_arrival: float  # first fragment ablation, s (nan when not found)
    source: list  # (file name, index in file) per joined step
    names: list  # ionmeta names
    Z: np.ndarray  # ionmeta Z
    r: np.ndarray  # radial cell centres, m
    r_f: np.ndarray  # radial cell faces, m
    n_i: np.ndarray  # (nt, sum(Z + 1), nr), m^-3
    T_cold: np.ndarray  # (nt, nr), eV
    n_free: np.ndarray  # (nt, nr): n_cold + n_hot + n_re, m^-3
    flags: list = field(default_factory=list)

    @property
    def t_rel(self):
        return self.t - self.t_onset

    def index_at(self, t_rel):
        """Index of the saved step nearest to t_rel (s)."""
        return int(np.argmin(np.abs(self.t_rel - t_rel)))


def load_case(case_dir):
    """Join init_0, init_1, ... and main of a DREAM SPI case.

    :raises ValueError: When a repeated step at a file boundary differs from
        its copy, or the species differ between files.
    """
    case_dir = Path(case_dir)
    inits = sorted(case_dir.glob("init_*.h5"), key=lambda p: int(p.stem.split("_")[1]))
    files = [*inits, case_dir / "main.h5"]
    parts = [_read(path) for path in files]
    for part in parts[1:]:
        if part["names"] != parts[0]["names"] or not np.array_equal(part["Z"], parts[0]["Z"]):
            raise ValueError(f"{case_dir}: species differ between files")

    onset_index = parts[-1]["filetimeindex"]
    pieces = []  # (part, first index, last index exclusive)
    for k, part in enumerate(parts):
        first = 0 if k == 0 else 1  # drop the step repeated from the previous file
        last = onset_index + 1 if k == len(parts) - 2 else len(part["t"])
        pieces.append((part, first, last))
    for (a, _, a_last), (b, _, _) in zip(pieces, pieces[1:]):
        _check_repeat(a, a_last - 1, b, 0, case_dir)

    t, source, offset = [], [], 0.0
    for part, first, last in pieces:
        t.append(part["t"][first:last] + offset)
        source += [(part["file"], i) for i in range(first, last)]
        offset += part["t"][last - 1]
    t = np.concatenate(t)
    n_onset = sum(last - first for _, first, last in pieces[:-1])
    t_onset = t[n_onset - 1]  # last init step == main step 0

    joined = {
        key: np.concatenate([part[key][first:last] for part, first, last in pieces])
        for key in ("n_i", "T_cold", "n_free", "Y_p")
    }
    # first change of any fragment size: ablation starts as a fragment enters the plasma
    changed = (np.abs(np.diff(joined["Y_p"], axis=0)) > 0).any(axis=1)
    if not changed.any():
        raise ValueError(f"{case_dir}: no fragment ablation found")
    t_arrival = t[int(np.argmax(changed)) + 1]

    first = parts[0]
    return DreamCase(
        path=case_dir, t=t, t_onset=float(t_onset), t_arrival=float(t_arrival), source=source,
        names=first["names"], Z=first["Z"], r=first["r"], r_f=first["r_f"],
        n_i=joined["n_i"], T_cold=joined["T_cold"], n_free=joined["n_free"],
    )


def load_file(path):
    """One DREAM output file as a DreamCase, without stitching (t_onset = 0)."""
    part = _read(Path(path))
    return DreamCase(
        path=Path(path), t=part["t"], t_onset=0.0, t_arrival=float("nan"),
        source=[(part["file"], i) for i in range(len(part["t"]))],
        names=part["names"], Z=part["Z"], r=part["r"], r_f=part["r_f"],
        n_i=part["n_i"], T_cold=part["T_cold"], n_free=part["n_free"],
    )


def _read(path):
    with h5py.File(path, "r") as f:
        names = b"".join(f["ionmeta/names"][()]).decode().strip(";").split(";")
        n_free = f["eqsys/n_cold"][()] + f["eqsys/n_hot"][()] + f["eqsys/n_re"][()]
        index = f["settings/init/filetimeindex"][()] if "settings/init/filetimeindex" in f else None
        return {
            "file": path.name,
            "t": f["grid/t"][()],
            "names": names,
            "Z": f["ionmeta/Z"][()].astype(int),
            "r": f["grid/r"][()],
            "r_f": f["grid/r_f"][()],
            "n_i": f["eqsys/n_i"][()],
            "T_cold": f["eqsys/T_cold"][()],
            "n_free": n_free,
            "Y_p": f["eqsys/Y_p"][()][:, :, 0] if "eqsys/Y_p" in f else None,
            "filetimeindex": None if index is None or len(index) == 0 else int(index[0]),
        }


def _check_repeat(a, ia, b, ib, case_dir, rtol=1e-5):
    """The step ia of a and step ib of b are the same plasma state."""
    for key in ("T_cold", "n_free"):
        if not np.allclose(a[key][ia], b[key][ib], rtol=rtol, atol=0):
            raise ValueError(f"{case_dir}: {a['file']}[{ia}] and {b['file']}[{ib}] differ in {key}")
    # ion densities: compare where they matter (above 1e-6 of the largest)
    x, y = a["n_i"][ia], b["n_i"][ib]
    big = np.abs(y) > 1e-6 * np.abs(y).max()
    if not np.allclose(x[big], y[big], rtol=1e-4, atol=0):
        raise ValueError(f"{case_dir}: {a['file']}[{ia}] and {b['file']}[{ib}] differ in n_i")


def sample_grid(cases, step=0.5e-3, before_arrival=2, after_onset=8):
    """Uniform t_rel grid shared by cases of one shot.

    Runs from the grid point at or before the earliest fragment arrival minus
    before_arrival steps, to after_onset steps after the TQ onset (t_rel = 0).

    :raises ValueError: When the grid leaves a case's joined time range.
    """
    arrival = min(case.t_arrival - case.t_onset for case in cases)
    start = (np.floor(arrival / step + 1e-9) - before_arrival) * step
    grid = np.round(np.arange(round(start / step), after_onset + 1) * step, 9)
    for case in cases:
        if grid[0] < case.t_rel[0] - step / 2 or grid[-1] > case.t_rel[-1] + step / 2:
            raise ValueError(f"{case.path}: grid {grid[0]:.4g} to {grid[-1]:.4g} s leaves its time range")
    return grid


def species_densities(case, index, elements=None):
    """Densities per (element, charge) at one joined step.

    Species of the same element (D_th, D_fi, D_inj) are summed per charge.
    When a species has more charge states than its element (the "B" species
    with Z = 6 modelled as boron), the states above the element's atomic
    number are folded into the bare nucleus, and a flag is added to
    case.flags.

    :param elements: Override of ELEMENTS, by species name prefix.
    :return: Dict from (element, charge) to a density profile (nr,), m^-3.
    """
    elements = {**ELEMENTS, **(elements or {})}
    offsets = np.concatenate([[0], np.cumsum(case.Z + 1)])
    densities = {}
    for k, name in enumerate(case.names):
        element = elements[name.split("_")[0]]
        block = case.n_i[index, offsets[k]:offsets[k + 1], :]
        for charge in range(case.Z[k] + 1):
            target = min(charge, element.atomic_number)
            if charge > element.atomic_number:
                flag = (f"{name}: Z = {case.Z[k]} in the file, modelled as {element.name} "
                        f"(Z = {element.atomic_number}); charge {charge} folded into "
                        f"{element.symbol}{element.atomic_number}+")
                if flag not in case.flags:
                    case.flags.append(flag)
            key = (element, target)
            densities[key] = densities.get(key, 0) + block[charge]
    return densities
