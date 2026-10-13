"""
axuv/data_index.py  –  the tracked list of data keys on the server.

plasmasds_utility fetches single files and cannot list a server folder, so
the keys that exist are recorded in axuv/data_index.json, with the size and
modification time of each file:

    {"prefix": "bolometer", "root": "aug-axuv",
     "files": {"atomic/ne.csv": {"size": 13997, "mtime": 1760440000}, ...}}

The format follows the manifest planned for plasmasds_utility, which can
replace this file once it exists. axuv.fetch_data reads it to find, e.g.,
every JOREK input of a dataset.

After uploading new data, regenerate the index from the local copies (whose
modification times are the upload times) and commit it:

    python -m axuv.data_index
"""

import json
from pathlib import Path

from axuv.datafiles import KEY_ROOT, PREFIX, local_path

INDEX_FILE = Path(__file__).parent / "data_index.json"


def load_index(path: Path = INDEX_FILE) -> dict:
    """Return {key: {"size": int, "mtime": int}} from the index file."""
    with open(path) as f:
        return json.load(f)["files"]


def select(index: dict, prefixes: list[str]) -> list[str]:
    """Return the sorted keys of *index* under any of *prefixes* (folders or full keys)."""
    return sorted(
        key for key in index
        if any(key == p or key.startswith(p.rstrip("/") + "/") for p in prefixes)
    )


def build_index(keys: list[str]) -> dict:
    """Return the index entries of *keys* from their local copies."""
    files = {}
    for key in keys:
        stat = local_path(key).stat()
        files[key] = {"size": stat.st_size, "mtime": int(stat.st_mtime)}
    return files


def write_index(files: dict, path: Path = INDEX_FILE) -> None:
    """Write *files* to the index file, keys sorted, one entry per line."""
    lines = [f'    {json.dumps(key)}: {json.dumps(files[key])}' for key in sorted(files)]
    with open(path, "w") as f:
        f.write(f'{{"prefix": "{PREFIX}", "root": "{KEY_ROOT}", "files": {{\n')
        f.write(",\n".join(lines))
        f.write("\n}}\n")


def main():
    from axuv.datafiles import DATA_TREES
    from axuv.publish_data import local_keys

    keys, misplaced = local_keys(list(DATA_TREES))
    for path in misplaced:
        print(f"not indexed: {path} is in the wrong tree for its key")
    write_index(build_index(keys))
    print(f"Wrote {len(keys)} keys to {INDEX_FILE}")


if __name__ == "__main__":
    main()
