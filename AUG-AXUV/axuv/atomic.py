"""
axuv.atomic
-----------
Open-ADAS atomic data for the Cherab plasma models.

The installed Cherab repository (default ``~/.cherab/openadas/repository``) is
the source of truth for which spectral lines exist. ``install_atomic_data``
fills it with the files the synthetic diagnostics need, and ``line_models``
builds one excitation and one recombination model per transition found there
for each species in a plasma.

The repository layout, ``pec/<class>/<element>/<charge>.json`` with one key
``"<upper> -> <lower>"`` per transition, is internal to Cherab. It is read
directly here and holds for the pinned Cherab 1.5.

Install the data once per user and machine:

    python -m axuv.atomic
"""

import argparse
import json
import warnings
from contextlib import contextmanager
from pathlib import Path

from cherab.core import Line
from cherab.core.atomic import Isotope, neon
from cherab.core.model import ExcitationLine, GaussianLine, RecombinationLine
from cherab.openadas import repository as cherab_repository
from cherab.openadas.install import install_adf15
from cherab.openadas.repository import populate
from cherab.openadas.repository.utility import DEFAULT_REPOSITORY_PATH

# ADF15 files installed on top of Cherab's populate(), as
# (element, charge, path relative to the Open-ADAS root).
# populate() installs no Ne PECs; upstream flags the Ne0 and Ne1 files as corrupt.
ADF15_FILES = [(neon, q, f"adf15/pec96#ne/pec96#ne_pju#ne{q}.dat") for q in range(10)]


def install_atomic_data(repository_path=None, adas_path=None):
    """Install Cherab's default Open-ADAS set plus the files in ADF15_FILES.

    Each file is looked up in adas_path first, if given, then in the
    repository's download cache, and downloaded from Open-ADAS otherwise.

    :param repository_path: Repository to install into; defaults to Cherab's
        default path.
    :param adas_path: Local copy of the Open-ADAS file tree (adf11/, adf15/,
        ...), for machines without internet access.
    """
    with _thermal_cx_to(repository_path):
        populate(download=True, repository_path=repository_path, adas_path=adas_path)
        for element, charge, file_path in ADF15_FILES:
            install_adf15(element, charge, file_path, download=True,
                          repository_path=repository_path, adas_path=adas_path)


@contextmanager
def _thermal_cx_to(repository_path):
    """Send thermal CX PECs to repository_path during an install.

    Cherab 1.5 install_adf15 (cherab/openadas/install.py) writes the thermal CX
    PECs found in some ADF15 files (C1, C2, C5, N1 in populate()) to the default
    repository, whatever repository_path says.
    """
    if repository_path is None:
        yield
        return
    original = cherab_repository.update_pec_thermal_cx_rates
    cherab_repository.update_pec_thermal_cx_rates = (
        lambda rates, repository_path_=None: original(rates, repository_path)
    )
    try:
        yield
    finally:
        cherab_repository.update_pec_thermal_cx_rates = original


def available_transitions(element, charge, kind, repository_path=None):
    """Transitions with a PEC of the given kind in the repository.

    :param element: Cherab Element or Isotope. Isotopes use their element's
        data, as cherab.openadas.OpenADAS does.
    :param charge: Charge state of the emitting ion.
    :param kind: "excitation" or "recombination".
    :param repository_path: Repository root; defaults to Cherab's default path.
    :return: List of (upper, lower) level tuples; empty when the charge state
        has no data. Integer levels (hydrogen-like n) are returned as int.
    """
    if isinstance(element, Isotope):
        element = element.element
    root = Path(repository_path or DEFAULT_REPOSITORY_PATH)
    path = root / "pec" / kind / element.symbol.lower() / f"{charge}.json"
    if not path.is_file():
        return []
    with open(path) as f:
        keys = json.load(f)
    return [tuple(_level(level) for level in key.split(" -> ")) for key in keys]


def line_models(plasma, lineshape=GaussianLine):
    """Line emission models for every species in the plasma composition.

    Each species gets an ExcitationLine per transition with an excitation PEC
    in the repository, and a RecombinationLine per transition with a
    recombination PEC when the next charge state, which supplies the
    recombining density, is also in the composition. Species without data
    contribute no models; a warning names each one that still has bound
    electrons, since its line radiation is then missing from the result, and
    each one with excitation but no recombination data while the next charge
    state is present.

    :param plasma: Cherab Plasma with its composition set and an OpenADAS
        instance as atomic_data; the repository path is taken from it.
    :param lineshape: Cherab line shape model class.
    """
    repository_path = plasma.atomic_data.data_path
    present = {(s.element, s.charge) for s in plasma.composition}
    models = []
    for species in plasma.composition:
        element, charge = species.element, species.charge
        excitation = available_transitions(element, charge, "excitation", repository_path)
        if not excitation and charge < element.atomic_number:
            warnings.warn(
                f"No excitation PEC data for {element.symbol}{charge}+ in {repository_path}; "
                "its line radiation is left out. Install the data with: python -m axuv.atomic",
                stacklevel=2,
            )
        for transition in excitation:
            line = Line(element, charge, transition)
            models.append(ExcitationLine(line, lineshape=lineshape))
        if (element, charge + 1) not in present:
            continue
        recombination = available_transitions(element, charge, "recombination", repository_path)
        if excitation and not recombination:
            warnings.warn(
                f"No recombination PEC data for {element.symbol}{charge}+ in {repository_path}; "
                "its recombination line radiation is left out.",
                stacklevel=2,
            )
        for transition in recombination:
            line = Line(element, charge, transition)
            models.append(RecombinationLine(line, lineshape=lineshape))
    return models


def _level(level):
    return int(level) if level.isdigit() else level


def main():
    parser = argparse.ArgumentParser(
        description="Install the Open-ADAS atomic data used by the AXUV synthetic diagnostics."
    )
    parser.add_argument(
        "--repository-path", default=None, metavar="DIR",
        help=f"Cherab repository to install into (default {DEFAULT_REPOSITORY_PATH}).",
    )
    parser.add_argument(
        "--adas-path", default=None, metavar="DIR",
        help="Local copy of the Open-ADAS file tree, searched before downloading.",
    )
    args = parser.parse_args()
    install_atomic_data(repository_path=args.repository_path, adas_path=args.adas_path)


if __name__ == "__main__":
    main()
