"""
axuv.atomic
-----------
Open-ADAS atomic data for the Cherab plasma models.

The installed Cherab repository (default ``~/.cherab/openadas/repository``) is
the source of truth for which spectral lines exist. ``install_atomic_data``
fills it with the files the synthetic diagnostics need, and ``line_models``
builds one excitation and one recombination model per transition found there
for each species in a plasma. When an ion has no data, ``line_models``
installs its files from ADF15_FILES (downloading them if needed) and raises an
error if the data is still missing.

The repository layout, ``pec/<class>/<element>/<charge>.json`` with one key
``"<upper> -> <lower>"`` per transition, is internal to Cherab. It is read
directly here and holds for the pinned Cherab 1.6.

Install the data once per user and machine:

    python -m axuv.atomic
"""

import argparse
import fcntl
import json
import warnings
from contextlib import contextmanager
from pathlib import Path

from cherab.core import Line
from cherab.core.atomic import Isotope, hydrogen, neon
from cherab.core.model import ExcitationLine, GaussianLine, RecombinationLine
from cherab.openadas import install as cherab_install
from cherab.openadas import repository as cherab_repository
from cherab.openadas.install import install_adf15
from cherab.openadas.repository import populate
from cherab.openadas.repository.utility import DEFAULT_REPOSITORY_PATH

# ADF15 files the synthetic diagnostics need, as
# (element, charge, path relative to the Open-ADAS root). line_models installs
# a missing ion from this list. H0 is also in Cherab's populate(); populate()
# installs no Ne PECs, and upstream flags the Ne0 and Ne1 files as corrupt.
ADF15_FILES = [
    (hydrogen, 0, "adf15/pec12#h/pec12#h_pju#h0.dat"),
    *[(neon, q, f"adf15/pec96#ne/pec96#ne_pju#ne{q}.dat") for q in range(10)],
]


def install_atomic_data(repository_path=None, adas_path=None):
    """Install Cherab's default Open-ADAS set plus the files in ADF15_FILES.

    Each file is looked up in adas_path first, if given, then in the
    repository's download cache, and downloaded from Open-ADAS otherwise.

    :param repository_path: Repository to install into; defaults to Cherab's
        default path.
    :param adas_path: Local copy of the Open-ADAS file tree (adf11/, adf15/,
        ...), for machines without internet access.
    """
    with _install_lock(repository_path), _cherab_install_fixes(repository_path):
        # ADF15_FILES first: populate() then writes its NIST wavelengths for H0
        # over the ADF15 file's own, as a plain populate() would
        for element, charge, file_path in ADF15_FILES:
            install_adf15(element, charge, file_path, download=True,
                          repository_path=repository_path, adas_path=adas_path)
        populate(download=True, repository_path=repository_path, adas_path=adas_path)


def _install_missing(composition, repository_path, wavelength_fallback=False):
    """Install data for the ions of the composition that lack it.

    Runs populate(), which also holds the isotope wavelengths, and the
    ADF15_FILES rows of the ions without excitation data. Each step that fails
    is recorded and the others still run.

    :return: Messages for the steps that failed.
    """
    errors = []
    with _install_lock(repository_path), _cherab_install_fixes(repository_path):
        # another process may have installed it while this one waited for the lock
        missing = _ions_without_data(composition, repository_path, wavelength_fallback)
        if not missing:
            return errors
        try:
            populate(download=True, repository_path=repository_path)
        except Exception as error:
            errors.append(f"populate(): {error}")
        for element, charge, file_path in ADF15_FILES:
            if (element.symbol, charge) not in missing:
                continue
            if available_transitions(element, charge, "excitation", repository_path):
                continue
            try:
                install_adf15(element, charge, file_path, download=True,
                              repository_path=repository_path)
            except Exception as error:
                errors.append(f"{file_path}: {error}")
    return errors


@contextmanager
def _install_lock(repository_path):
    """Hold an exclusive lock on the repository while installing.

    Parallel workers that find the same data missing then install it once,
    instead of writing the same JSON files at the same time.
    """
    root = Path(repository_path or DEFAULT_REPOSITORY_PATH)
    root.mkdir(parents=True, exist_ok=True)
    with open(root / ".axuv_install.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


OPENADAS_FILE_URL = "https://open.adas.ac.uk/download/"


@contextmanager
def _cherab_install_fixes(repository_path):
    """Work around two Cherab 1.6 install bugs while installing.

    - Cherab downloads from http://open.adas.ac.uk/download/, which answers 404;
      the same path over https works. Fixed upstream on the development branch,
      not in a release as of Cherab 1.6.0.
    - install_adf15 writes the thermal CX PECs found in some ADF15 files (C1, C2,
      C5, N1 in populate()) to the default repository, whatever repository_path
      says. They are sent to repository_path here.
    """
    original_url = cherab_install.OPENADAS_FILE_URL
    original_cx = cherab_repository.update_pec_thermal_cx_rates
    cherab_install.OPENADAS_FILE_URL = OPENADAS_FILE_URL
    if repository_path is not None:
        cherab_repository.update_pec_thermal_cx_rates = (
            lambda rates, repository_path_=None: original_cx(rates, repository_path)
        )
    try:
        yield
    finally:
        cherab_install.OPENADAS_FILE_URL = original_url
        cherab_repository.update_pec_thermal_cx_rates = original_cx


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


def line_models(plasma, lineshape=GaussianLine, install_missing=True):
    """Line emission models for every species in the plasma composition.

    Each species gets an ExcitationLine per transition with an excitation PEC
    in the repository, and a RecombinationLine per transition with a
    recombination PEC when the next charge state, which supplies the
    recombining density, is also in the composition.

    An ion with bound electrons but no excitation data is installed from
    ADF15_FILES, downloading the files if needed. If its data is still missing
    afterwards (no row in ADF15_FILES, or the download failed), a RuntimeError
    names every such ion. Bare nuclei have no lines and need no data. A species
    with excitation but no recombination data, while the next charge state is
    present, only gets a warning.

    :param plasma: Cherab Plasma with its composition set and an OpenADAS
        instance as atomic_data; the repository path is taken from it.
    :param lineshape: Cherab line shape model class.
    :param install_missing: Try to install missing ions before failing.
    """
    repository_path = plasma.atomic_data.data_path
    fallback = getattr(plasma.atomic_data, "_wavelength_element_fallback", False)
    composition = list(plasma.composition)
    missing = _ions_without_data(composition, repository_path, fallback)
    errors = []
    if missing and install_missing:
        errors = _install_missing(composition, repository_path, fallback)
        missing = _ions_without_data(composition, repository_path, fallback)
    if missing:
        raise RuntimeError(
            f"Atomic data missing in {repository_path}:"
            + "".join(f"\n  {reason}" for _, reason in sorted(missing.items()))
            + "\nAdd the ADF15 file of each ion to ADF15_FILES in axuv/atomic.py, or "
            "install the data with: python -m axuv.atomic"
            + ("\nInstall errors:" + "".join(f"\n  {e}" for e in errors) if errors else "")
        )

    present = {(s.element, s.charge) for s in composition}
    models = []
    for species in composition:
        element, charge = species.element, species.charge
        excitation = available_transitions(element, charge, "excitation", repository_path)
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


def _ions_without_data(composition, repository_path, wavelength_fallback=False):
    """Ions with bound electrons whose excitation PECs or wavelengths are missing.

    PECs are looked up under the element, wavelengths under the species itself
    (the isotope for D and T), as cherab.openadas.OpenADAS does; with
    wavelength_fallback, the element's wavelengths also count.

    :return: Dict from (element symbol, charge) to a description of what is missing.
    """
    root = Path(repository_path or DEFAULT_REPOSITORY_PATH)
    missing = {}
    for species in composition:
        species_element, charge = species.element, species.charge
        element = species_element.element if isinstance(species_element, Isotope) else species_element
        if charge >= element.atomic_number:
            continue
        name = f"{species_element.symbol}{charge}+"
        pec = _keys(root / "pec" / "excitation" / element.symbol.lower() / f"{charge}.json")
        if not pec:
            missing[(element.symbol, charge)] = f"{name}: no excitation PEC data"
            continue
        wavelengths = _keys(root / "wavelength" / species_element.symbol.lower() / f"{charge}.json")
        if wavelength_fallback:
            wavelengths |= _keys(root / "wavelength" / element.symbol.lower() / f"{charge}.json")
        lacking = pec - wavelengths
        if lacking:
            missing[(element.symbol, charge)] = (
                f"{name}: no wavelength for {len(lacking)} of {len(pec)} transitions"
            )
    return missing


def _keys(path):
    if not path.is_file():
        return set()
    with open(path) as f:
        return set(json.load(f))


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
