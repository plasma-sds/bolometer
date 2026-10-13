"""
axuv/migrate_data.py  –  move data from the old axuv/data + aug layout into data keys.

Before plasmasds_utility, the input data lived in two hand-copied folders:

    axuv/data/   geometry, responsivity, matrices, DREAM, traces, JOREK inputs
    aug/         AUG CAD meshes

This script places every file of those folders at the local path of its data
key (see axuv.datafiles), i.e. in <data dir>/bolometer/{private,public}/aug-axuv/,
so the code finds it without contacting the server, and so it can be uploaded.

Files are hard-linked by default: no extra disk space, and the modification
times (which the utility's update check relies on) are kept. Across file
systems, or with --copy, they are copied with their modification times.
Existing files at the target are never replaced.

Usage
-----
    # Show what would happen
    python -m axuv.migrate_data --dry-run

    # Link the files into the default data folder (~/.local/share/plasmasds)
    python -m axuv.migrate_data

    # Old folders somewhere else
    python -m axuv.migrate_data --data-dir ~/Data/AUG-AXUV/data --cad-dir ~/Data/AUG-AXUV/aug
"""

import argparse
import errno
import os
import re
import shutil
from pathlib import Path

from axuv.datafiles import client_dir, local_path

_REPO = Path(__file__).resolve().parent.parent

# (pattern on the path relative to the old axuv/data folder, data key template)
_DATA_RULES = [
    (r"(AXUV_LOS_geom\.txt|gc_d_lines\.obj|etendue_\w+\.h5)", r"geometry/\1"),
    (r"(axuv_sensitivity|degraded_sensitivity|degraded_errorbars|worst_estimation)\.csv",
     r"responsivity/\1.csv"),
    (r"ne\.csv", r"atomic/ne.csv"),
    (r"dream_output\.h5", r"dream/dream_output.h5"),
    (r"(raytransfer_[\w-]+\.h5)", r"raytransfer/\1"),
    (r"(step\d+_out\.h5)", r"jorek/\1"),
    (r"(\d+)/(P\d+)/input/([^/]+\.h5)", r"jorek/\1/\2/\3"),
    (r"(\d+)/((?:frad|ip|wth)-(?:exp|jorek)-\d+\.txt)", r"traces/\1/\2"),
]


def data_key(relative: str) -> str | None:
    """Return the data key of a file in the old axuv/data folder, or None.

    :param relative: path relative to axuv/data, with '/' separators
    """
    for pattern, template in _DATA_RULES:
        match = re.fullmatch(pattern, relative)
        if match:
            return match.expand(template)
    return None


def plan(data_dir: Path | None, cad_dir: Path | None):
    """Return ([(source, key)], [unmapped source]) for the two old folders."""
    mapped, unmapped = [], []
    if data_dir is not None:
        for source in sorted(data_dir.rglob("*")):
            if source.is_file():
                key = data_key(source.relative_to(data_dir).as_posix())
                (unmapped.append(source) if key is None
                 else mapped.append((source, key)))
    if cad_dir is not None:
        for source in sorted(cad_dir.rglob("*")):
            if source.is_file():
                mapped.append((source, "cad/" + source.relative_to(cad_dir).as_posix()))
    return mapped, unmapped


def place(source: Path, target: Path, copy: bool = False) -> str:
    """Hard-link (or copy) *source* to *target*; return what was done.

    Returns "linked", "copied", "exists" (same file already there) or
    "conflict" (a different file is already there; it is left alone).
    """
    if target.exists():
        return "exists" if os.path.samefile(source, target) else "conflict"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not copy:
        try:
            os.link(source, target)
            return "linked"
        except OSError as error:
            if error.errno not in (errno.EXDEV, errno.EPERM, errno.EMLINK):
                raise
    shutil.copy2(source, target)
    return "copied"


def _existing_dir(default: Path):
    return default.resolve() if default.is_dir() else None


def main():
    parser = argparse.ArgumentParser(
        description="Place the files of the old axuv/data and aug folders at the "
        "local paths of their data keys."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_existing_dir(_REPO / "axuv" / "data"),
        help="Old data folder. Default: axuv/data (a symlink is followed).",
    )
    parser.add_argument(
        "--cad-dir",
        type=Path,
        default=_existing_dir(_REPO / "aug"),
        help="Old CAD mesh folder. Default: aug (a symlink is followed).",
    )
    parser.add_argument("--copy", action="store_true", help="Copy instead of hard-linking.")
    parser.add_argument("--dry-run", action="store_true", help="Only print what would be done.")
    args = parser.parse_args()
    if args.data_dir is None and args.cad_dir is None:
        parser.error("no old data folder found; give --data-dir and/or --cad-dir")

    mapped, unmapped = plan(args.data_dir, args.cad_dir)
    counts = {}
    for source, key in mapped:
        target = local_path(key)
        if args.dry_run:
            result = "exists" if target.exists() else "new"
            print(f"{result:8} {key}  <-  {source}")
        else:
            result = place(source, target, copy=args.copy)
            if result == "conflict":
                print(f"conflict {key}: a different file is already at {target}, left alone")
        counts[result] = counts.get(result, 0) + 1

    for source in unmapped:
        print(f"skipped  {source}: no data key for this file")
    summary = ", ".join(f"{n} {what}" for what, n in sorted(counts.items()))
    print(f"\n{len(mapped)} file(s): {summary}; {len(unmapped)} skipped")
    print(f"data folder: {client_dir()}")


if __name__ == "__main__":
    main()
