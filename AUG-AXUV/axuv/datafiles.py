"""
axuv/datafiles.py  –  locate AUG-AXUV input data through plasmasds_utility.

Input data is served by the plasma-sds data server under the client prefix
``bolometer``; AUG-AXUV keys sit below ``aug-axuv/``. Callers pass the key
relative to that folder, e.g. ``data_path("geometry/AXUV_LOS_geom.txt")``,
and get a local path back, downloaded first if it is not on disk yet.

The first folder of a key decides which tree the file lives in (DATA_TREES).
The tree is always passed explicitly, so a public key never asks the private
server and a private key never falls back to public data.

Behaviour is controlled by plasmasds_utility's environment variables:

    PLASMASDS_DATA_DIR       where the local copies live
                             (<dir>/bolometer/{private,public}/aug-axuv/<key>)
    PLASMASDS_SERVER_ACCESS  "all", "public" (default when CI is set) or
                             "local" (no network: local copies only)
    PLASMASDS_QUIET          log the once-per-process notices instead of
                             printing them

Creating the client reads and writes nothing, so importing this module is
free; the configuration and log file are opened on the first data_path call.
"""

from pathlib import Path

from plasmasds_utility import DataClient

PREFIX = "bolometer"
KEY_ROOT = "aug-axuv"

# First folder of a key -> data tree on the server.
DATA_TREES = {
    "geometry": "private",       # diode/aperture geometry, poloidal outline, etendue
    "cad": "private",            # AUG vessel CAD meshes
    "raytransfer": "private",    # Stage 1 geometry matrices
    "jorek": "private",          # JOREK inputs for Stage 1 masking and Stage 2
    "traces": "private",         # frad/ip/wth time traces (AUG data)
    "responsivity": "public",    # AXUV responsivity curves
    "atomic": "public",          # Ne line list for Cherab
    "dream": "public",           # DREAM run for the 1D sightline model
    "emissions": "public",       # dummy Stage 2 outputs
    "test_dataset": "public",    # dummy geometry and reference results for CI
}

_CLIENT = DataClient(PREFIX)


def server_key(key: str) -> str:
    """Return the full server key of an AUG-AXUV *key*, e.g. 'aug-axuv/<key>'."""
    return f"{KEY_ROOT}/{key}"


def is_private(key: str) -> bool:
    """Return whether *key* lives in the private tree.

    :raises KeyError: if the first folder of *key* is not in DATA_TREES.
    """
    group = key.split("/", 1)[0]
    try:
        return DATA_TREES[group] == "private"
    except KeyError:
        raise KeyError(
            f"unknown data key {key!r}: its first folder must be one of "
            f"{', '.join(DATA_TREES)}"
        ) from None


def data_path(key: str, *, update: str = "never") -> Path:
    """Return the local path of the data file *key*, downloading it if needed.

    :param key:    path below the aug-axuv folder, with '/' separators,
                   e.g. "raytransfer/raytransfer_S16_norefl.h5"
    :param update: what to do with a local copy, passed to DataClient.get:
                   "never" (default), "if_newer" or "force"
    :raises KeyError: for a key outside the known folders (DATA_TREES)
    :raises plasmasds_utility.PlasmasdsUtilityError: if the file cannot be
        found or fetched, e.g. a private key with PLASMASDS_SERVER_ACCESS=public
    """
    return _CLIENT.get(server_key(key), private=is_private(key), update=update)


def client_dir() -> Path:
    """Return the local folder of the bolometer client (holds private/ and public/)."""
    return _CLIENT.client_dir()


def local_path(key: str) -> Path:
    """Return where *key* is, or will be, stored locally, without any I/O."""
    return _CLIENT.local_path(server_key(key), private=is_private(key))
