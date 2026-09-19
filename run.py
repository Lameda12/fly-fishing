#!/usr/bin/env python3
"""Fly Fishing on top of Embodied Fly Lab: one command for the whole thing.

`python3 run.py` fetches the simulator if it is missing, records the
descending-neuron response cache if it is missing, converts the fly body, trains
the readout, records a pair of episodes, reports the baselines, and then serves
the viewer.

The expensive step is the cache, and only the cache: it is the one thing that
runs the whole-brain network. Everything after it is fast, so re-training or
re-recording costs seconds.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
CACHE = ROOT / "results" / "dn-cache.json"
FLY_GLB = WEB / "public" / "fly.glb"
RECORDINGS = WEB / "public" / "recordings" / "index.json"

MODES = (
    "all", "fetch", "cache", "train", "record", "report", "ablation", "glb",
    "voyage", "record-voyage", "live", "serve", "build", "test",
)


def run(command: list[str], cwd: Path = ROOT) -> int:
    print(f"$ {' '.join(str(part) for part in command)}\n", flush=True)
    try:
        return subprocess.run(command, cwd=cwd).returncode
    except FileNotFoundError:
        print(f"{command[0]} is not installed", file=sys.stderr)
        return 1


def need(tool: str, why: str) -> bool:
    if shutil.which(tool):
        return True
    print(f"{tool} is required {why}", file=sys.stderr)
    return False


def ensure_simulator(auto_fetch: bool) -> bool:
    check = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "fetch_sim.py"), "--check"],
        capture_output=True,
        text=True,
    )
    if check.returncode == 0:
        print(check.stdout.strip())
        return True
    if not auto_fetch:
        print(check.stderr.strip(), file=sys.stderr)
        return False
    print("Simulator missing; fetching it now.")
    return run([sys.executable, str(ROOT / "tools" / "fetch_sim.py")]) == 0


def ensure_web_deps() -> bool:
    if (WEB / "node_modules").is_dir():
        return True
    if not need("npm", "to install the viewer's dependencies"):
        return False
    print("Installing the viewer's dependencies (three, vite, typescript).")
    return run(["npm", "install"], cwd=WEB) == 0


def build_cache(extra: list[str]) -> int:
    return run(["node", str(ROOT / "fishing" / "build-cache.mjs"), *extra])


def train(extra: list[str]) -> int:
    return run(["node", str(ROOT / "fishing" / "train.mjs"), *extra])


def record(extra: list[str]) -> int:
    return run(["node", str(ROOT / "fishing" / "record.mjs"), *extra])


def report(extra: list[str]) -> int:
    return run(["node", str(ROOT / "tools" / "report.mjs"), *extra])


def voyage(extra: list[str]) -> int:
    """Train one readout head per decision stage of a voyage."""
    return run(["node", str(ROOT / "fishing" / "train-voyage.mjs"), *extra])


def record_voyage(extra: list[str]) -> int:
    """Record a voyage pair for the viewer: trained readout against reflex."""
    return run(["node", str(ROOT / "fishing" / "record-voyage.mjs"), *extra])


def ablation(extra: list[str]) -> int:
    """Compare the intact connectome against both ablated ones.

    Needs one recorded cache per arm; the tool says which are missing.
    """
    return run(["node", str(ROOT / "tools" / "ablation.mjs"), *extra])


def live(extra: list[str]) -> int:
    """Stream the live network to the viewer over a local WebSocket.

    Runs at about a ninth of real time; the viewer shows brain time.
    """
    return run(["node", str(ROOT / "fishing" / "live.mjs"), *extra])


def build_glb(extra: list[str]) -> int:
    """Convert the fetched body into web/public/fly.glb (gitignored)."""
    return run([sys.executable, str(ROOT / "tools" / "build_fly_glb.py"), *extra])


def tests() -> int:
    failed = 0
    for path in sorted((ROOT / "tests").glob("*.test.mjs")):
        failed |= run(["node", str(path)])
    return failed


def serve(extra: list[str]) -> int:
    if not ensure_web_deps():
        return 1
    if not RECORDINGS.exists():
        print(
            f"No recordings at {RECORDINGS.parent}; the viewer will say so.\n"
            "Make a pair with: python3 run.py record",
            file=sys.stderr,
        )
    print("Viewer (replay mode; no backend needed once built).")
    return run(["npm", "run", "dev", "--", *extra], cwd=WEB)


def build_site(extra: list[str]) -> int:
    if not ensure_web_deps():
        return 1
    code = run(["npm", "run", "build", "--", *extra], cwd=WEB)
    if code == 0:
        print(f"\nStatic site in {WEB / 'dist'}; it needs no backend.")
    return code


def split_argv(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split this script's own arguments from the sub-command's, at the first `--`.

    argparse.REMAINDER would swallow every option after the positional mode, so
    `run.py train --seed 7` would silently ignore the seed. An explicit split
    keeps both halves unambiguous.
    """
    if "--" in argv:
        cut = argv.index("--")
        return argv[:cut], argv[cut + 1 :]
    return argv, []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="all",
        choices=MODES,
        help="all (default): everything missing, then serve. "
        "fetch / cache / train / record / report / voyage / record-voyage / "
        "ablation / glb / live / serve / build / test run one step.",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="re-record the response cache even if one is already there (slow)",
    )
    parser.add_argument(
        "--no-fetch", action="store_true", help="fail instead of fetching the simulator"
    )
    parser.epilog = (
        "Arguments after -- go to the step, for example:\n"
        "  python3 run.py train -- --seed 7 --episodes 1200\n"
        "  python3 run.py cache -- --ablation weight-shuffle\n"
        "  python3 run.py voyage -- --heads shared\n"
        "\nThe ablation needs all three caches recorded first:\n"
        "  python3 run.py cache\n"
        "  python3 run.py cache -- --ablation weight-shuffle\n"
        "  python3 run.py cache -- --ablation input-shuffle\n"
        "  python3 run.py ablation"
    )
    return parser.parse_args(split_argv(sys.argv[1:])[0])


def main() -> int:
    args = parse_args()
    extra = split_argv(sys.argv[1:])[1]

    if args.mode == "test":
        return tests()
    if args.mode in ("serve", "build"):
        return serve(extra) if args.mode == "serve" else build_site(extra)

    if not ensure_simulator(auto_fetch=not args.no_fetch):
        return 1
    if args.mode == "fetch":
        return 0
    if not need("node", "to run the simulator headlessly (Node 18 or newer)"):
        return 1

    if args.mode == "cache":
        return build_cache(extra)
    if args.mode == "train":
        return train(extra)
    if args.mode == "record":
        return record(extra)
    if args.mode == "report":
        return report(extra)
    if args.mode == "voyage":
        return voyage(extra)
    if args.mode == "record-voyage":
        return record_voyage(extra)
    if args.mode == "ablation":
        return ablation(extra)
    if args.mode == "live":
        return live(extra)
    if args.mode == "glb":
        return build_glb(extra)

    # mode == "all"
    if args.rebuild_cache or not CACHE.exists():
        print(
            "\nRecording the descending-neuron response cache. This is the slow step:\n"
            "it runs the whole-brain network for several minutes of brain time.\n"
        )
        if build_cache([]) != 0:
            return 1
    else:
        print(f"Response cache already at {CACHE.relative_to(ROOT)}; pass --rebuild-cache to redo it.")

    # Cheap, and the viewer falls back to a placeholder without it.
    if not FLY_GLB.exists():
        build_glb([])

    if train([]) != 0:
        return 1
    if record([]) != 0:
        return 1
    if report([]) != 0:
        return 1
    # The voyage is the five-stage loop and it is a separate, much longer
    # training run, so it stays opt-in rather than sitting in the default path.
    print(
        "\nThat was the single-stage fishing task. For the five-stage voyage\n"
        "(bait, fish, row back, cook, eat), which trains one readout head per\n"
        "stage and takes a few thousand voyages:\n"
        "  python3 run.py voyage\n"
        "  python3 run.py record-voyage\n"
    )
    return serve([])


if __name__ == "__main__":
    raise SystemExit(main())
