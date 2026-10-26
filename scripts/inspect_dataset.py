"""Inspect an image dataset folder (no model, no training, nothing is modified).

Assumed layout: one sub-folder per class, e.g.

    my_dataset/
        real/   img001.jpg, img002.jpg, ...
        fake/   img001.png, ...

Images nested deeper still count towards their top-level class folder.
For layouts like train/real and train/fake, use --depth 2 to group by
"train/real", "train/fake", and so on.

Images are opened one at a time and only the header/metadata is read, so
memory use stays small. By default each file gets a quick integrity check
(Image.verify()); add --deep to fully decode every image instead (slower,
catches truncated files that verify() can miss).

Usage:
    python scripts/inspect_dataset.py data/raw/my_dataset
    python scripts/inspect_dataset.py data/raw/my_dataset --depth 2 --deep
    python scripts/inspect_dataset.py data/raw/my_dataset --json outputs/dataset_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    sys.exit("Pillow is not installed. Run: python -m pip install pillow")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}


@dataclass
class Stats:
    """Counters for one group of images (a class folder, or the whole dataset)."""

    images: int = 0
    unreadable: int = 0
    bytes_total: int = 0
    extensions: Counter = field(default_factory=Counter)
    formats: Counter = field(default_factory=Counter)  # real format as seen by Pillow
    modes: Counter = field(default_factory=Counter)  # RGB, RGBA, L, ...
    sizes: Counter = field(default_factory=Counter)  # (width, height) -> count
    widths: list = field(default_factory=list)
    heights: list = field(default_factory=list)


def check_image(path: Path, deep: bool):
    """Return (width, height, format, mode). Raises if the file is not a valid image."""
    with Image.open(path) as img:  # lazy: reads the header only
        info = (img.width, img.height, img.format, img.mode)
        if deep:
            img.load()  # decode pixels of this one image, then free them
        else:
            img.verify()  # quick integrity check; do not use img after this
    return info


def scan(root: Path, depth: int, deep: bool):
    groups: dict[str, Stats] = defaultdict(Stats)
    overall = Stats()
    all_extensions: Counter = Counter()
    bad_files: list[dict] = []
    walk_errors: list[str] = []
    show_progress = sys.stderr.isatty()

    def on_walk_error(err: OSError) -> None:
        walk_errors.append(f"{err.filename}: {err.strerror}")

    for dirpath, dirnames, filenames in os.walk(root, onerror=on_walk_error):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for name in sorted(filenames):
            path = Path(dirpath) / name
            ext = path.suffix.lower() or "(none)"
            all_extensions[ext] += 1
            if ext not in IMAGE_EXTENSIONS:
                continue

            rel = path.relative_to(root)
            group = "/".join(rel.parts[:-1][:depth]) or "(root)"
            targets = (groups[group], overall)

            try:
                size_bytes = path.stat().st_size
            except OSError:
                size_bytes = 0

            for s in targets:
                s.images += 1
                s.bytes_total += size_bytes
                s.extensions[ext] += 1

            try:
                width, height, fmt, mode = check_image(path, deep)
            except Exception as exc:  # corrupt/odd files raise many different errors
                for s in targets:
                    s.unreadable += 1
                bad_files.append(
                    {"path": str(rel), "group": group, "error": f"{type(exc).__name__}: {exc}"}
                )
                continue

            for s in targets:
                s.formats[fmt or "unknown"] += 1
                s.modes[mode] += 1
                s.sizes[(width, height)] += 1
                s.widths.append(width)
                s.heights.append(height)

            if show_progress and overall.images % 500 == 0:
                print(f"\r  checked {overall.images} images...", end="", file=sys.stderr)

    if show_progress:
        print("\r" + " " * 40 + "\r", end="", file=sys.stderr)
    return groups, overall, all_extensions, bad_files, walk_errors


def fmt_counter(counter: Counter, limit: int = 8) -> str:
    if not counter:
        return "-"
    items = [f"{k}={v}" for k, v in counter.most_common(limit)]
    if len(counter) > limit:
        items.append(f"... +{len(counter) - limit} more")
    return ", ".join(items)


def fmt_span(values: list) -> str:
    return f"{min(values):g} / {statistics.median(values):g} / {max(values):g}"


def describe(title: str, s: Stats) -> None:
    print(f"\n[{title}]")
    print(f"  images: {s.images}  (readable: {s.images - s.unreadable}, unreadable: {s.unreadable})")
    print(f"  extensions:  {fmt_counter(s.extensions)}")
    print(f"  real format: {fmt_counter(s.formats)}")
    print(f"  color modes: {fmt_counter(s.modes)}")
    if s.widths:
        print(f"  width  (min / median / max): {fmt_span(s.widths)}")
        print(f"  height (min / median / max): {fmt_span(s.heights)}")
        common = ", ".join(f"{w}x{h} ({n})" for (w, h), n in s.sizes.most_common(5))
        print(f"  most common sizes: {common}")
    if s.images:
        print(f"  average file size: {s.bytes_total / s.images / 1024:.1f} KB")


def stats_to_dict(s: Stats) -> dict:
    out = {
        "images": s.images,
        "unreadable": s.unreadable,
        "extensions": dict(s.extensions),
        "formats": dict(s.formats),
        "modes": dict(s.modes),
        "avg_file_size_kb": round(s.bytes_total / s.images / 1024, 1) if s.images else None,
        "most_common_sizes": {f"{w}x{h}": n for (w, h), n in s.sizes.most_common(10)},
    }
    if s.widths:
        out["width"] = {"min": min(s.widths), "median": statistics.median(s.widths), "max": max(s.widths)}
        out["height"] = {"min": min(s.heights), "median": statistics.median(s.heights), "max": max(s.heights)}
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect an image dataset folder.")
    p.add_argument("dataset_dir", type=Path, help="root folder of the dataset")
    p.add_argument("--depth", type=int, default=1,
                   help="folder levels under the root used to name a class (default: 1)")
    p.add_argument("--deep", action="store_true",
                   help="fully decode every image (slower, stricter corruption check)")
    p.add_argument("--json", type=Path, dest="json_path", metavar="FILE",
                   help="also write the full report to this JSON file")
    p.add_argument("--max-listed", type=int, default=20,
                   help="max unreadable files to list on screen (default: 20)")
    args = p.parse_args()
    if args.depth < 1:
        p.error("--depth must be 1 or more")
    return args


def main() -> int:
    args = parse_args()
    root: Path = args.dataset_dir

    if not root.exists():
        print(f"Error: '{root}' does not exist.", file=sys.stderr)
        return 2
    if not root.is_dir():
        print(f"Error: '{root}' is not a directory.", file=sys.stderr)
        return 2

    mode_label = "deep decode" if args.deep else "quick verify"
    print(f"Inspecting: {root.resolve()}  (check mode: {mode_label}, class depth: {args.depth})")

    try:
        groups, overall, all_ext, bad_files, walk_errors = scan(root, args.depth, args.deep)
    except KeyboardInterrupt:
        print("\nInterrupted; no report produced.", file=sys.stderr)
        return 130

    total_files = sum(all_ext.values())
    print(f"\nFiles found: {total_files}   image files (by extension): {overall.images}")
    print("All extensions: " + fmt_counter(all_ext, limit=15))

    if overall.images == 0:
        print("\nNo image files found. Check the path and that the images have a "
              f"known extension ({', '.join(sorted(IMAGE_EXTENSIONS))}).", file=sys.stderr)
        return 1

    print(f"\nClass folders (groups): {len(groups)}")
    for name in sorted(groups):
        print(f"  {name}: {groups[name].images}")

    describe("OVERALL", overall)
    for name in sorted(groups):
        describe(f"class: {name}", groups[name])

    if bad_files:
        print(f"\nUnreadable images: {len(bad_files)}")
        for item in bad_files[: args.max_listed]:
            print(f"  {item['path']}  ->  {item['error']}")
        if len(bad_files) > args.max_listed:
            print(f"  ... and {len(bad_files) - args.max_listed} more (use --json for the full list)")
    else:
        print("\nUnreadable images: 0")

    if walk_errors:
        print(f"\nFolders that could not be read: {len(walk_errors)}")
        for msg in walk_errors[:10]:
            print(f"  {msg}")

    if args.json_path:
        report = {
            "dataset_dir": str(root.resolve()),
            "check_mode": mode_label,
            "class_depth": args.depth,
            "total_files": total_files,
            "all_extensions": dict(all_ext),
            "overall": stats_to_dict(overall),
            "groups": {name: stats_to_dict(groups[name]) for name in sorted(groups)},
            "unreadable_files": bad_files,
            "folder_errors": walk_errors,
        }
        try:
            args.json_path.parent.mkdir(parents=True, exist_ok=True)
            args.json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"\nJSON report written to {args.json_path}")
        except OSError as exc:
            print(f"\nCould not write JSON report: {exc}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
