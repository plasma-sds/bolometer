"""
axuv/fetch_data.py  –  download data keys before a run, and print their local paths.

Selects keys from the tracked index (axuv/data_index.json) by folder prefix or
full key, downloads the ones missing locally (axuv.datafiles.data_path), and
optionally prints the local path of each, one per line, for shell scripts.

Fetch once before starting parallel workers: each worker then finds its files
on disk instead of downloading them itself.

Usage
-----
    # Everything Stage 2 needs for one dataset
    python -m axuv.fetch_data jorek/40673/P1 raytransfer/raytransfer_S16_norefl.h5 atomic

    # Local paths of the JOREK inputs of a dataset, for a shell loop
    python -m axuv.fetch_data --paths jorek/40673/P1

    # Download again where the server copy is newer
    python -m axuv.fetch_data --update if_newer geometry responsivity
"""

import argparse
import sys

from plasmasds_utility import PlasmasdsUtilityError

from axuv.data_index import INDEX_FILE, load_index, select
from axuv.datafiles import data_path


def main():
    parser = argparse.ArgumentParser(
        description="Download data keys listed in the index and print their local paths."
    )
    parser.add_argument(
        "prefixes",
        nargs="+",
        metavar="PREFIX",
        help="Key prefixes (folders such as 'jorek/40673/P1') or full keys.",
    )
    parser.add_argument(
        "--paths", action="store_true", help="Print the local path of each key to stdout."
    )
    parser.add_argument(
        "--update",
        choices=["never", "if_newer", "force"],
        default="never",
        help="What to do with a local copy (see DataClient.get).",
    )
    args = parser.parse_args()

    keys = select(load_index(), args.prefixes)
    if not keys:
        raise SystemExit(f"No keys under {', '.join(args.prefixes)} in {INDEX_FILE}.")
    failed = []
    for key in keys:
        try:
            path = data_path(key, update=args.update)
        except PlasmasdsUtilityError as error:
            failed.append(key)
            print(f"cannot fetch {key}: {error}", file=sys.stderr)
            continue
        if args.paths:
            print(path, flush=True)
    if failed:
        raise SystemExit(f"{len(failed)} of {len(keys)} key(s) could not be fetched.")
    if not args.paths:
        print(f"{len(keys)} key(s) available locally.", file=sys.stderr)


if __name__ == "__main__":
    main()
