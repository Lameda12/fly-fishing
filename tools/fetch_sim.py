#!/usr/bin/env python3
"""Put a local Embodied Fly Lab checkout in vendor/ for the fishing layer to load.

The upstream simulator ships no LICENSE file, so this repository never vendors
its code or its ~91 MB of connectome data. This script fetches (or links) your
own checkout instead, and verifies the files the fishing layer actually reads.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "vendor" / "embodied-fly-lab"
UPSTREAM = "https://github.com/statsleelab/embodied-fly-lab"

# Only the files the fishing layer imports or reads. Anything missing here means a
# stale or partial checkout, which is worth failing on early rather than in the
# middle of a training run.
REQUIRED = (
    "brain-core.mjs",
    "locomotion-controller.mjs",
    "assets/model/fly.xml",
    "assets/model_meta.json",
    "data/manifest.json",
    "data/offsets.u32",
    "data/targets.u32",
    "data/weights.i16",
    "data/groups.u8",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--from",
        dest="source",
        metavar="PATH",
        help="symlink an existing embodied-fly-lab checkout instead of cloning",
    )
    parser.add_argument("--revision", default="main", help="branch, tag, or commit to check out")
    parser.add_argument("--force", action="store_true", help="replace an existing vendor/ checkout")
    parser.add_argument("--check", action="store_true", help="only verify what is already there")
    return parser.parse_args()


def missing_files(base: Path) -> list[str]:
    return [name for name in REQUIRED if not (base / name).exists()]


def report(base: Path) -> int:
    absent = missing_files(base)
    if absent:
        print(f"Incomplete simulator checkout at {base}", file=sys.stderr)
        for name in absent:
            print(f"  missing: {name}", file=sys.stderr)
        return 1
    manifest = base / "data" / "manifest.json"
    size_mb = sum(p.stat().st_size for p in (base / "data").glob("*")) / 1e6
    print(f"Simulator ready: {base}")
    print(f"  connectome data: {size_mb:.0f} MB ({manifest.parent})")
    return 0


def link(source: Path) -> int:
    source = source.expanduser().resolve()
    if missing_files(source):
        print(f"{source} does not look like an embodied-fly-lab checkout", file=sys.stderr)
        return report(source)
    VENDOR.parent.mkdir(parents=True, exist_ok=True)
    if VENDOR.is_symlink() or VENDOR.exists():
        print(f"{VENDOR} already exists; pass --force to replace it", file=sys.stderr)
        return 1
    VENDOR.symlink_to(source, target_is_directory=True)
    print(f"Linked {VENDOR} -> {source}")
    return report(VENDOR)


def clone(revision: str) -> int:
    VENDOR.parent.mkdir(parents=True, exist_ok=True)
    print(f"Cloning {UPSTREAM} ({revision}) into {VENDOR}")
    print("This transfers about 160 MB and can take a few minutes.")
    command = ["git", "clone", "--depth", "1", "--branch", revision, UPSTREAM, str(VENDOR)]
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError:
        print("git is not installed", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as error:
        # --branch rejects raw commit SHAs, so retry as a full clone + checkout.
        print(f"Shallow clone failed ({error.returncode}); retrying without --branch", file=sys.stderr)
        shutil.rmtree(VENDOR, ignore_errors=True)
        try:
            subprocess.run(["git", "clone", UPSTREAM, str(VENDOR)], check=True)
            subprocess.run(["git", "-C", str(VENDOR), "checkout", revision], check=True)
        except subprocess.CalledProcessError as retry_error:
            print(f"Could not fetch the simulator: {retry_error}", file=sys.stderr)
            return 1
    return report(VENDOR)


def main() -> int:
    args = parse_args()
    if args.check:
        if not VENDOR.exists():
            print(f"No simulator checkout at {VENDOR}. Run: python3 tools/fetch_sim.py", file=sys.stderr)
            return 1
        return report(VENDOR)
    if args.force and (VENDOR.is_symlink() or VENDOR.exists()):
        if VENDOR.is_symlink():
            VENDOR.unlink()
        else:
            shutil.rmtree(VENDOR)
    if args.source:
        return link(Path(args.source))
    if VENDOR.exists():
        print(f"{VENDOR} already exists; verifying instead of refetching")
        return report(VENDOR)
    return clone(args.revision)


if __name__ == "__main__":
    raise SystemExit(main())
