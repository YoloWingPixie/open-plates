"""Rebuild every basemap layer for a named region.

Usage:
    uv run python scripts/build_basemap_all.py <region_id>

Example:
    uv run python scripts/build_basemap_all.py spangdahlem

Runs, in order:
 1. ``build_basemap_hillshade``   (SRTM DEM fetch + hillshade PNG — slowest,
    does the big network + CPU work first so a failed OSM retry later
    doesn't throw out the expensive terrain pass)
 2. ``build_basemap_populated``   (Overpass: admin polygons)
 3. ``build_basemap_roads``       (Overpass: highway ways)
 4. ``build_basemap_waterways``   (Overpass: rivers + lakes)
 5. ``build_basemap_obstacles``   (Overpass: tall structures)

All five builders read their bbox from ``data/basemap/regions.json``.
If the region id isn't registered this script fails fast without touching
anything on disk.

Progress is reported per step; a single step's non-zero exit aborts the
whole run so a partial rebuild doesn't leave mismatched sibling files.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Import each builder module so we can call its ``build(...)`` entry
# point directly — keeps stdout streaming and avoids the subprocess
# overhead of shelling out five times.
from build_basemap_hillshade import build as build_hillshade
from build_basemap_obstacles import build as build_obstacles
from build_basemap_populated import build as build_populated
from build_basemap_roads import build as build_roads
from build_basemap_waterways import build as build_waterways

REPO_ROOT = Path(__file__).resolve().parents[1]
REGIONS_PATH = REPO_ROOT / "data" / "basemap" / "regions.json"


def load_region_bbox(region_id: str) -> tuple[float, float, float, float]:
    """Return ``(lat_min, lon_min, lat_max, lon_max)`` for ``region_id``.

    Raises ``SystemExit`` with a human-readable error if the registry is
    missing or the id is unknown — we want the whole multi-step rebuild
    to fail before any builder has run.
    """
    if not REGIONS_PATH.is_file():
        print(f"ERROR: region registry missing: {REGIONS_PATH}", file=sys.stderr)
        sys.exit(2)
    with REGIONS_PATH.open("r", encoding="utf-8") as fh:
        registry = json.load(fh)
    if region_id not in registry:
        known = ", ".join(sorted(registry)) or "<empty>"
        print(
            f"ERROR: region {region_id!r} not in registry "
            f"({REGIONS_PATH}); known regions: {known}",
            file=sys.stderr,
        )
        sys.exit(2)
    bbox = (registry[region_id] or {}).get("bbox")
    if not (isinstance(bbox, list) and len(bbox) == 4):
        print(
            f"ERROR: region {region_id!r} has invalid bbox: {bbox!r} "
            f"(expected [lon_min, lat_min, lon_max, lat_max])",
            file=sys.stderr,
        )
        sys.exit(2)
    lon_min, lat_min, lon_max, lat_max = (float(v) for v in bbox)
    return (lat_min, lon_min, lat_max, lon_max)


# Order matters: terrain first (slowest + biggest network payload), then
# the four Overpass pipelines. Each step's ``build()`` returns a path or
# raises on hard failure.
STEPS: tuple[tuple[str, object], ...] = (
    ("hillshade", build_hillshade),
    ("populated", build_populated),
    ("roads", build_roads),
    ("waterways", build_waterways),
    ("obstacles", build_obstacles),
)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("region_id")
    args = ap.parse_args(argv)

    bbox = load_region_bbox(args.region_id)
    print(
        f"=== building all basemap layers for region={args.region_id!r} "
        f"bbox (lat_min, lon_min, lat_max, lon_max)={bbox} ===",
        flush=True,
    )
    total_t0 = time.perf_counter()
    for name, fn in STEPS:
        print(f"\n--- step: {name} ---", flush=True)
        t0 = time.perf_counter()
        try:
            fn(args.region_id, bbox)
        except SystemExit as exc:
            print(
                f"ERROR: step {name!r} exited with code {exc.code}",
                file=sys.stderr,
            )
            return int(exc.code) if isinstance(exc.code, int) else 1
        except Exception as exc:  # pragma: no cover - defensive
            print(f"ERROR: step {name!r} raised: {exc}", file=sys.stderr)
            return 1
        dt = time.perf_counter() - t0
        print(f"--- step {name!r} done in {dt:.1f}s ---", flush=True)
    total_dt = time.perf_counter() - total_t0
    print(
        f"\n=== all 5 layers built for {args.region_id!r} "
        f"in {total_dt:.1f}s ===",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
