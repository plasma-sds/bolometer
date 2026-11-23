"""
axuv.adf15
----------
Reader for ADF15 photon emissivity coefficient (PEC) files that Cherab's
parser cannot read in full: the pec40 tungsten files (Cherab drops lines whose
configuration labels repeat in the ls and ic couplings) and the pec93 boron
files (Cherab cannot read their header, and keeps one metastable block only).

Each block is keyed by the level indices of its transition from the
transition table in the file comments, prefixed with a variant name, for
example ("ic:487", "ic:1") for a W7+ line in intermediate coupling. Several
variants of one ion can then sit side by side in a Cherab repository, and
axuv.atomic.line_models picks one per element for a run.

Units of the returned data follow Cherab's repository: density in m^-3,
temperature in eV, PEC in photon m^3 s^-1 and wavelength in nm.
"""

import re
import warnings

import numpy as np

# Separates the variant name from the level index in a transition key
VARIANT_SEPARATOR = ":"

# First line of a data block: wavelength (Angstrom), number of densities,
# number of temperatures, then "/"-separated key = value fields
_BLOCK_HEADER = re.compile(r"^\s*(\d+\.?\d*)\s*a?\s+(\d+)\s+(\d+)\s*/(.*)$", re.IGNORECASE)
_FIELD = re.compile(r"(isel|type|indm|ispb)\s*=\s*([+\w]+)", re.IGNORECASE)

_TABLE_HEADER = re.compile(r"^c\s+isel\s+wa?ve?len", re.IGNORECASE)
# Row of the transition table in the comments, as in pec40 and pec93 files:
#   C      1   120.310  487(2)3(  3.5)-  1(2)3(  3.5)       excit ...
#   C   1.    2497.5     3(2)0( 0.5)- 1(2)1( 2.5)  EXCIT ...
_TRANSITION_ROW = re.compile(
    r"^c\s+(\d+)\.?\s+(\d+\.?\d*)\s+(\d+)\s*\([^-]*-\s*(\d+)\s*\(", re.IGNORECASE
)

# Block types and the Cherab rate class they belong to
_CLASSES = {"excit": "excitation", "recom": "recombination"}

# Metastable index (INDM) of blocks that are kept: unresolved files give T,
# resolved files 1 for the ground state and +1 for the ground state of the
# recombining ion. Blocks driven by other metastables are skipped, because a
# plasma composition gives only the total density of each charge state.
_GROUND_METASTABLES = {None, "t", "1", "+1"}

# Parent partition (ISPB, pec40 files): t for blocks summed over partitions.
# Partition-resolved blocks are not supported.
_UNRESOLVED_PARTITIONS = {None, "t"}

# Allowed difference between a block's header wavelength and its transition
# table row, relative; the two are printed with the same digits
_WAVELENGTH_MATCH = 1e-5


def read_adf15(path, variant):
    """Read the excitation and recombination PECs of an ADF15 file.

    :param path: Path to the ADF15 file.
    :param variant: Name used as prefix of the level indices in every key.
    :return: (rates, wavelengths). rates maps "excitation" and "recombination"
        to {(upper, lower): {"ne": array, "te": array, "rate": 2D array}};
        wavelengths maps (upper, lower) to the wavelength in nm.
    :raises ValueError: On a file that cannot be read as ADF15 (for example a
        damaged download), a block without a transition table row or with a
        different wavelength there, an unknown block type, a
        partition-resolved block, or two kept blocks with the same transition
        and class.
    """
    with open(path) as f:
        lines = f.read().splitlines()

    levels = _transition_table(lines)
    rates = {"excitation": {}, "recombination": {}}
    wavelengths = {}
    skipped = 0
    try:
        blocks = list(_blocks(lines))
    except (ValueError, IndexError) as error:
        raise ValueError(f"{path}: not a readable ADF15 file ({error})") from error
    for header, values in blocks:
        wavelength, n_density, n_temperature, fields = header
        block_type = fields.get("type")
        if block_type not in _CLASSES:
            raise ValueError(f"{path}: block type {block_type!r} is not supported")
        if fields.get("ispb") not in _UNRESOLVED_PARTITIONS:
            raise ValueError(f"{path}: partition-resolved block (ispb = {fields['ispb']}) is not supported")
        if fields.get("indm") not in _GROUND_METASTABLES:
            skipped += 1
            continue
        isel = int(fields["isel"])
        if isel not in levels:
            raise ValueError(f"{path}: block {isel} is missing from the transition table")
        upper, lower, table_wavelength = levels[isel]
        if abs(table_wavelength - wavelength) > _WAVELENGTH_MATCH * wavelength:
            raise ValueError(
                f"{path}: block {isel} is at {wavelength} A, its transition table row at "
                f"{table_wavelength} A"
            )
        transition = (f"{variant}{VARIANT_SEPARATOR}{upper}", f"{variant}{VARIANT_SEPARATOR}{lower}")
        cls = _CLASSES[block_type]
        if transition in rates[cls]:
            raise ValueError(f"{path}: two {cls} blocks for transition {upper} -> {lower}")

        density = values[:n_density]
        temperature = values[n_density:n_density + n_temperature]
        pec = values[n_density + n_temperature:].reshape(n_density, n_temperature)
        rates[cls][transition] = {
            "ne": density * 1e6,  # cm^-3 to m^-3
            "te": temperature,
            "rate": pec * 1e-6,  # cm^3 s^-1 to m^3 s^-1
        }
        # both classes share the wavelength of the transition
        wavelengths[transition] = wavelength / 10  # Angstrom to nm
    if skipped:
        warnings.warn(
            f"{path}: skipped {skipped} blocks driven from metastables other than the ground state",
            stacklevel=2,
        )
    return rates, wavelengths


def _transition_table(lines):
    """Map block number (isel) to (upper, lower, wavelength in Angstrom).

    Rows are read after the table header ("isel  wavelength ..."), so the
    energy level table above it is never taken for transitions.
    """
    levels = {}
    in_table = False
    for line in lines:
        if _TABLE_HEADER.match(line):
            in_table = True
            continue
        match = _TRANSITION_ROW.match(line) if in_table else None
        if match:
            isel, wavelength, upper, lower = match.groups()
            levels[int(isel)] = (int(upper), int(lower), float(wavelength))
    return levels


def _blocks(lines):
    """Yield ((wavelength, n_density, n_temperature, fields), values) per data block."""
    n_blocks = int(lines[0].split()[0])
    i = 1
    found = 0
    while found < n_blocks:
        if i >= len(lines):
            raise ValueError(f"file ends after {found} of {n_blocks} blocks")
        match = _BLOCK_HEADER.match(lines[i])
        i += 1
        if not match:
            continue
        wavelength, n_density, n_temperature, rest = match.groups()
        n_density, n_temperature = int(n_density), int(n_temperature)
        fields = {key.lower(): value.lower() for key, value in _FIELD.findall(rest)}
        needed = n_density + n_temperature + n_density * n_temperature
        values = []
        while len(values) < needed:
            values += [float(v.replace("D", "E").replace("d", "e")) for v in lines[i].split()]
            i += 1
        if len(values) != needed:
            raise ValueError(f"block {fields.get('isel')}: {len(values)} values, expected {needed}")
        found += 1
        yield (float(wavelength), n_density, n_temperature, fields), np.array(values)
