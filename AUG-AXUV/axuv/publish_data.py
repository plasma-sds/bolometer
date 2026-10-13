"""
axuv/publish_data.py  –  upload local data files to the plasma-sds data server.

Uploads the local copies of data keys (see axuv.datafiles) from
<data dir>/bolometer/{private,public}/aug-axuv/ to the same keys on the server,
each to its tree. Keys are selected by folder prefix; nothing is uploaded
without a selection.

A key that is already on the server is skipped, so an interrupted run can
simply be started again. To replace files on the server, give --replace with a
version label: the current server file is kept as archive/<version>/<key>
first. Other failures are reported and the run continues; a missing SSH
access stops it.

Uploads need SSH access to the server, also for public data
(plasmasds_utility.set_ssh_key).

Usage
-----
    # What would be uploaded
    python -m axuv.publish_data geometry responsivity --dry-run

    # Upload everything except the ray-transfer matrices
    python -m axuv.publish_data geometry responsivity atomic dream traces jorek cad

    # Replace the responsivity curves, keeping the old ones
    python -m axuv.publish_data responsivity --replace 2026-10-14
"""

import argparse

from plasmasds_utility import AlreadyOnServerError, AuthError, PlasmasdsUtilityError

from axuv.datafiles import DATA_TREES, KEY_ROOT, client_dir, is_private, on_server, upload


def local_keys(prefixes: list[str]) -> tuple[list[str], list[str]]:
    """Return (keys, misplaced) of the local files under the given key prefixes.

    *misplaced* lists local files whose tree does not match DATA_TREES (or
    whose first folder is unknown); they are never uploaded.
    """
    keys, misplaced = set(), []
    for tree in ("private", "public"):
        root = client_dir() / tree / KEY_ROOT
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.name.startswith(".") or path.name.endswith(".part"):
                continue
            key = path.relative_to(root).as_posix()
            if not any(key == p or key.startswith(p.rstrip("/") + "/") for p in prefixes):
                continue
            group = key.split("/", 1)[0]
            if DATA_TREES.get(group) != tree:
                misplaced.append(f"{tree}/{KEY_ROOT}/{key}")
            else:
                keys.add(key)
    return sorted(keys), sorted(misplaced)


def publish(keys: list[str], replace: str | None = None) -> dict[str, list[str]]:
    """Upload *keys*; return {"uploaded"|"on server"|"failed": [messages]}."""
    results = {"uploaded": [], "on server": [], "failed": []}
    for i, key in enumerate(keys, 1):
        tree = "private" if is_private(key) else "public"
        print(f"[{i}/{len(keys)}] {tree:7} {key}", flush=True)
        archive_to = None if replace is None else f"archive/{replace}/{key}"
        try:
            upload(key, archive_to=archive_to)
            results["uploaded"].append(key)
        except AlreadyOnServerError:
            results["on server"].append(key)
        except AuthError:
            raise
        except PlasmasdsUtilityError as error:
            results["failed"].append(f"{key}: {error}")
            print(f"    failed: {error}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Upload local data files to the plasma-sds data server."
    )
    parser.add_argument(
        "prefixes",
        nargs="+",
        metavar="PREFIX",
        help="Key prefixes to upload, e.g. 'geometry', 'jorek/40673' or a full key. "
        f"First folders: {', '.join(DATA_TREES)}.",
    )
    parser.add_argument(
        "--replace",
        metavar="VERSION",
        help="Replace files already on the server, keeping each as archive/VERSION/<key>.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list the selected files, and whether each is on the server already.",
    )
    args = parser.parse_args()

    keys, misplaced = local_keys(args.prefixes)
    for path in misplaced:
        print(f"not uploaded: {path} is in the wrong tree for its key (see DATA_TREES)")
    if not keys:
        raise SystemExit(f"No local files under {', '.join(args.prefixes)} in {client_dir()}.")
    if args.dry_run:
        new = 0
        for key in keys:
            try:
                status = "on server" if on_server(key) else "new"
            except PlasmasdsUtilityError as error:
                status = f"unknown ({type(error).__name__})"
            new += status == "new"
            print(f"{'private' if is_private(key) else 'public':7} {status:9} {key}")
        print(f"\n{len(keys)} file(s) selected in {client_dir()}, {new} not on the server")
        return

    try:
        results = publish(keys, args.replace)
    except AuthError as error:
        raise SystemExit(f"No SSH access to the data server: {error}")
    print(
        f"\n{len(results['uploaded'])} uploaded, {len(results['on server'])} already on "
        f"the server, {len(results['failed'])} failed"
    )
    for message in results["failed"]:
        print(f"  {message}")
    if results["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
