"""
axuv.atomic
-----------
Open-ADAS atomic data for the Cherab plasma models.

The installed Cherab repository (default ``~/.cherab/openadas/repository``) is
the source of truth for which spectral lines exist. ``install_atomic_data``
fills it with the files the synthetic diagnostics need, and ``line_models``
builds one excitation and one recombination model per transition found there
for each species in a plasma. When an ion has no data, ``line_models``
installs its files from ADF15_FILES or VARIANT_FILES (downloading them if
needed) and raises an error if the data is still missing.

Elements with several PEC data sets in Open-ADAS (VARIANTS: the W couplings,
the B file families) are installed with each set side by side, read by
axuv.adf15. A run picks one set per element through ``line_models(...,
variants=...)``, or ``--pec-variant`` on the run scripts.

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
from cherab.core.atomic import Isotope, boron, hydrogen, neon, tungsten
from cherab.core.model import ExcitationLine, GaussianLine, RecombinationLine
from cherab.openadas import install as cherab_install
from cherab.openadas import repository as cherab_repository
from cherab.openadas.install import install_adf15
from cherab.openadas.repository import populate, update_pec_rates, update_wavelengths
from cherab.openadas.repository.utility import DEFAULT_REPOSITORY_PATH

from axuv.adf15 import VARIANT_SEPARATOR, read_adf15

# ADF15 files the synthetic diagnostics need, as
# (element, charge, path relative to the Open-ADAS root). line_models installs
# a missing ion from this list. H0 is also in Cherab's populate(); populate()
# installs no Ne PECs, and upstream flags the Ne0 and Ne1 files as corrupt.
ADF15_FILES = [
    (hydrogen, 0, "adf15/pec12#h/pec12#h_pju#h0.dat"),
    *[(neon, q, f"adf15/pec96#ne/pec96#ne_pju#ne{q}.dat") for q in range(10)],
]

# PEC data sets per element, read with axuv.adf15 and installed side by side.
#   W, pec40: ic (intermediate coupling), ls (LS coupling), and ca and cl,
#     two line selections from the same configuration-average adf04 data
#     (named in each file's comments). Every charge state has all four.
#   B, pec93: llu (one set per ion, not resolved by metastable) and llr
#     (resolved by metastable; only the blocks driven from the ground state
#     are kept). Only B3+ has both; the other ions have one pec93 file (llu
#     for B0-B2, llr for B4), used in either set.
# The sets differ a lot in summed line power; see "Atomic data" in readme.md.
VARIANTS = {"W": ("ca", "cl", "ic", "ls"), "B": ("llu", "llr")}
# Variant used when a run gives none. B defaults to llu: its B3+ data needs no
# assumption about the metastable population. W has no default, because the
# four sets differ by up to about 100x in summed line power.
DEFAULT_VARIANTS = {"B": "llu"}
_BORON_FILES = {
    "llu": {0: "llu#b0", 1: "llu#b1", 2: "llu#b2", 3: "llu#b3", 4: "llr#b4"},
    "llr": {0: "llu#b0", 1: "llu#b1", 2: "llu#b2", 3: "llr#b3", 4: "llr#b4"},
}

# ADF15 files read with axuv.adf15, as (element, charge, variant, path
# relative to the Open-ADAS root). The variant prefixes every transition key.
VARIANT_FILES = [
    *[
        (tungsten, q, coupling, f"adf15/pec40#w/pec40#w_{coupling}#w{q}.dat")
        for coupling in VARIANTS["W"]
        for q in range(tungsten.atomic_number)
    ],
    *[
        (boron, q, variant, f"adf15/pec93#b/pec93#b_{name}.dat")
        for variant in VARIANTS["B"]
        for q, name in _BORON_FILES[variant].items()
    ],
]


def install_atomic_data(repository_path=None, adas_path=None):
    """Install Cherab's default Open-ADAS set plus ADF15_FILES and VARIANT_FILES.

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
        for element, charge, variant, file_path in VARIANT_FILES:
            _install_variant_file(element, charge, variant, file_path,
                                  repository_path=repository_path, adas_path=adas_path)
        populate(download=True, repository_path=repository_path, adas_path=adas_path)


def _install_variant_file(element, charge, variant, file_path, repository_path=None, adas_path=None):
    """Install one VARIANT_FILES row with axuv.adf15, keys prefixed with the variant.

    The file is looked up as install_adf15 does: in adas_path, then in the
    repository's download cache, then downloaded from Open-ADAS.
    """
    print(f"Installing {file_path} as {element.symbol} {variant}...")
    path = cherab_install._locate_adas_file(file_path, True, adas_path, repository_path)
    rates, wavelengths = read_adf15(path, variant)
    update_pec_rates(
        {cls: {element: {charge: transitions}} for cls, transitions in rates.items() if transitions},
        repository_path,
    )
    update_wavelengths({element: {charge: wavelengths}}, repository_path)


def _install_missing(composition, repository_path, wavelength_fallback=False, variants=None):
    """Install data for the ions of the composition that lack it.

    Runs populate(), which also holds the isotope wavelengths, when an ion
    outside VARIANTS is missing, then the ADF15_FILES rows of the ions without
    excitation data and the VARIANT_FILES rows of the chosen variants. Each step that fails is
    recorded and the others still run.

    :return: Messages for the steps that failed.
    """
    errors = []
    with _install_lock(repository_path), _cherab_install_fixes(repository_path):
        # another process may have installed it while this one waited for the lock
        missing = _ions_without_data(composition, repository_path, wavelength_fallback, variants)
        if not missing:
            return errors
        # populate() holds none of the VARIANTS elements; skip it when only they are missing
        if any(symbol not in VARIANTS for symbol, _ in missing):
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
        for element, charge, variant, file_path in VARIANT_FILES:
            if (element.symbol, charge) not in missing:
                continue
            if variant != (variants or {}).get(element.symbol):
                continue
            try:
                _install_variant_file(element, charge, variant, file_path,
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


def available_transitions(element, charge, kind, repository_path=None, variant=None):
    """Transitions with a PEC of the given kind in the repository.

    :param element: Cherab Element or Isotope. Isotopes use their element's
        data, as cherab.openadas.OpenADAS does.
    :param charge: Charge state of the emitting ion.
    :param kind: "excitation" or "recombination".
    :param repository_path: Repository root; defaults to Cherab's default path.
    :param variant: Keep only the transitions of this VARIANTS data set; all
        transitions when None.
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
        keys = [key for key in json.load(f) if _in_variant(key, variant)]
    return [tuple(_level(level) for level in key.split(" -> ")) for key in keys]


def _in_variant(key, variant):
    """Whether a transition key belongs to a variant ("ic:12 -> ic:1" to ic).

    Every key belongs to variant None.
    """
    if variant is None:
        return True
    prefix = f"{variant}{VARIANT_SEPARATOR}".lower()
    return all(level.startswith(prefix) for level in key.split(" -> "))


def check_variants(composition, variants):
    """Check the variant choice against VARIANTS and the plasma composition.

    :param composition: Species of the plasma.
    :param variants: Dict from element symbol to variant name, or None.
    :return: The choice as a dict, with DEFAULT_VARIANTS filled in.
    :raises ValueError: For an element or variant not in VARIANTS, or an
        element of the composition with neither a choice nor a default. Bare
        nuclei have no lines and need no choice.
    """
    variants = {**DEFAULT_VARIANTS, **(variants or {})}
    for symbol, variant in variants.items():
        if symbol not in VARIANTS:
            raise ValueError(
                f"No PEC variants for {symbol}; elements with variants: {', '.join(VARIANTS)}"
            )
        if variant not in VARIANTS[symbol]:
            raise ValueError(
                f"Unknown PEC variant {variant!r} for {symbol}; choose one of: "
                + ", ".join(VARIANTS[symbol])
            )
    symbols = {
        _element(species.element).symbol
        for species in composition
        if species.charge < _element(species.element).atomic_number
    }
    unchosen = sorted(symbols & set(VARIANTS) - set(variants))
    if unchosen:
        first = unchosen[0]
        example = VARIANTS[first][0]
        raise ValueError(
            "Choose a PEC variant for "
            + "; ".join(f"{s} ({', '.join(VARIANTS[s])})" for s in unchosen)
            + f", for example variants={{'{first}': '{example}'}} or --pec-variant {first}={example}"
        )
    return variants


def _element(element):
    return element.element if isinstance(element, Isotope) else element


def line_models(plasma, lineshape=GaussianLine, install_missing=True, variants=None):
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
    :param variants: PEC data set per element, as {element symbol: variant},
        for example {"W": "ic", "B": "llr"}. Required for each element of the
        composition that is in VARIANTS and not in DEFAULT_VARIANTS; see
        VARIANTS for the choices.
    """
    repository_path = plasma.atomic_data.data_path
    fallback = getattr(plasma.atomic_data, "_wavelength_element_fallback", False)
    composition = list(plasma.composition)
    variants = check_variants(composition, variants)
    missing = _ions_without_data(composition, repository_path, fallback, variants)
    errors = []
    if missing and install_missing:
        errors = _install_missing(composition, repository_path, fallback, variants)
        missing = _ions_without_data(composition, repository_path, fallback, variants)
    if missing:
        raise RuntimeError(
            f"Atomic data missing in {repository_path}:"
            + "".join(f"\n  {reason}" for _, reason in sorted(missing.items()))
            + "\nAdd the ADF15 file of each ion to ADF15_FILES (or VARIANT_FILES) in "
            "axuv/atomic.py, or install the data with: python -m axuv.atomic"
            + ("\nInstall errors:" + "".join(f"\n  {e}" for e in errors) if errors else "")
        )

    present = {(s.element, s.charge) for s in composition}
    models = []
    no_recombination = []
    for species in composition:
        element, charge = species.element, species.charge
        variant = variants.get(_element(element).symbol)
        excitation = available_transitions(element, charge, "excitation", repository_path, variant)
        for transition in excitation:
            line = Line(element, charge, transition)
            models.append(ExcitationLine(line, lineshape=lineshape))
        if (element, charge + 1) not in present:
            continue
        recombination = available_transitions(
            element, charge, "recombination", repository_path, variant
        )
        if excitation and not recombination:
            no_recombination.append(f"{element.symbol}{charge}+")
        for transition in recombination:
            line = Line(element, charge, transition)
            models.append(RecombinationLine(line, lineshape=lineshape))
    if no_recombination:
        warnings.warn(
            f"No recombination PEC data for {', '.join(no_recombination)} in {repository_path}; "
            "their recombination line radiation is left out.",
            stacklevel=2,
        )
    return models


def _ions_without_data(composition, repository_path, wavelength_fallback=False, variants=None):
    """Ions with bound electrons whose excitation PECs or wavelengths are missing.

    PECs are looked up under the element, wavelengths under the species itself
    (the isotope for D and T), as cherab.openadas.OpenADAS does; with
    wavelength_fallback, the element's wavelengths also count. For an element
    with a chosen variant, only that variant's transitions count.

    :return: Dict from (element symbol, charge) to a description of what is missing.
    """
    root = Path(repository_path or DEFAULT_REPOSITORY_PATH)
    missing = {}
    for species in composition:
        species_element, charge = species.element, species.charge
        element = species_element.element if isinstance(species_element, Isotope) else species_element
        if charge >= element.atomic_number:
            continue
        variant = (variants or {}).get(element.symbol)
        name = f"{species_element.symbol}{charge}+" + (f" ({variant})" if variant else "")
        pec = {
            key
            for key in _keys(root / "pec" / "excitation" / element.symbol.lower() / f"{charge}.json")
            if _in_variant(key, variant)
        }
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


def add_variant_argument(parser):
    """Add --pec-variant ELEMENT=VARIANT (repeatable) to an argparse parser.

    The parsed value is a dict for line_models(..., variants=...), or None
    when the option is not given.
    """
    choices = "; ".join(f"{s}: {', '.join(v)}" for s, v in VARIANTS.items())
    parser.add_argument(
        "--pec-variant", action=_VariantAction, default=None, metavar="ELEMENT=VARIANT",
        help=f"PEC data set for an element, repeatable ({choices}). "
        "Needed for W when the plasma contains it; B defaults to llu.",
    )


class _VariantAction(argparse.Action):
    def __call__(self, parser, namespace, value, option_string=None):
        symbol, sep, variant = value.partition("=")
        if not sep or not symbol or not variant:
            parser.error(f"{option_string} expects ELEMENT=VARIANT, got {value!r}")
        try:
            check_variants([], {symbol: variant})
        except ValueError as error:
            parser.error(str(error))
        chosen = dict(getattr(namespace, self.dest) or {})
        if symbol in chosen:
            parser.error(f"{option_string} given twice for {symbol}")
        chosen[symbol] = variant
        setattr(namespace, self.dest, chosen)


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
