"""Build a road-network line basemap layer from OpenStreetMap.

Usage:
    uv run python scripts/build_basemap_roads.py <region_id> \
        [<lat_min> <lon_min> <lat_max> <lon_max>]

When only ``<region_id>`` is supplied, the bbox is looked up in
``data/basemap/regions.json``. The optional positional lat/lon form stays
supported for one-off builds that don't sit in the region registry.

Example (Spangdahlem, from registry):
    uv run python scripts/build_basemap_roads.py spangdahlem

Example (one-off, custom bbox):
    uv run python scripts/build_basemap_roads.py spangdahlem \
        49.3 5.8 50.5 7.2

The script:
 1. Queries Overpass for ``highway=motorway|trunk|primary|secondary|tertiary``
    ways inside the bbox. Same retry/cache pattern as
    ``build_basemap_populated.py``. One retry after 10 s on HTTP 429 or 5xx.
 2. Extracts each way's geometry (list of ``[lon, lat]`` nodes) from the
    ``geometry`` field returned by ``out body geom;``.
 3. Emits one GeoJSON ``LineString`` feature per way with:
      - ``class`` = ``road_motorway|road_trunk|road_primary|road_secondary|
        road_tertiary`` (from the OSM ``highway`` tag).
      - ``name``  = ``tags.ref`` if present else ``tags.name``.
      - ``rank``  = 1 (motorway/trunk) / 2 (primary) / 3 (secondary) /
        4 (tertiary).
 4. Lightly simplifies each polyline at ~0.0003 deg (~33 m) using the same
    Douglas-Peucker helper as the populated-area script (duplicated here
    so each script is self-contained).
 5. Sorts features deterministically by (rank asc, class asc, name/id asc,
    first-coordinate asc) so repeated builds produce byte-identical output.
 6. Caches the Overpass response under
    ``data/cache/osm/<region>-roads.json`` so subsequent runs stay offline.
 7. Writes ``data/basemap/<region>-roads.json``.

Expected output for the Eifel bbox is several hundred road features.

Data (c) OpenStreetMap contributors, ODbL.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OSM_CACHE_DIR = REPO_ROOT / "data" / "cache" / "osm"
BASEMAP_DIR = REPO_ROOT / "data" / "basemap"
REGIONS_PATH = BASEMAP_DIR / "regions.json"

OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def load_region_bbox(region_id: str) -> tuple[float, float, float, float]:
    """Return ``(lat_min, lon_min, lat_max, lon_max)`` for ``region_id``.

    Looks the id up in ``data/basemap/regions.json``; reorders from the
    GeoJSON ``[lon_min, lat_min, lon_max, lat_max]`` convention into the
    lat-first tuple shape the build scripts consume.
    """
    if not REGIONS_PATH.is_file():
        raise RuntimeError(f"region registry missing: {REGIONS_PATH}")
    with REGIONS_PATH.open("r", encoding="utf-8") as fh:
        registry = json.load(fh)
    if region_id not in registry:
        known = ", ".join(sorted(registry)) or "<empty>"
        raise RuntimeError(
            f"region {region_id!r} not in registry ({REGIONS_PATH}); "
            f"known regions: {known}"
        )
    bbox = (registry[region_id] or {}).get("bbox")
    if not (isinstance(bbox, list) and len(bbox) == 4):
        raise RuntimeError(
            f"region {region_id!r} has invalid bbox: {bbox!r} "
            f"(expected [lon_min, lat_min, lon_max, lat_max])"
        )
    lon_min, lat_min, lon_max, lat_max = (float(v) for v in bbox)
    return (lat_min, lon_min, lat_max, lon_max)

# OSM highway tag -> (class, rank). Rank: 1=motorway/trunk, 2=primary,
# 3=secondary, 4=tertiary. Lower rank = higher visual prominence.
HIGHWAY_MAP: dict[str, tuple[str, int]] = {
    "motorway": ("road_motorway", 1),
    "trunk": ("road_trunk", 1),
    "primary": ("road_primary", 2),
    "secondary": ("road_secondary", 3),
    "tertiary": ("road_tertiary", 4),
}


# ---------------------------------------------------------------------------
# Overpass fetch (with simple retry + on-disk cache)
# ---------------------------------------------------------------------------


def build_query(bbox: tuple[float, float, float, float]) -> str:
    lat_min, lon_min, lat_max, lon_max = bbox
    bb = f"{lat_min},{lon_min},{lat_max},{lon_max}"
    return (
        "[out:json][timeout:90];\n"
        "(\n"
        f'  way["highway"~"^(motorway|trunk|primary|secondary|tertiary)$"]({bb});\n'
        ");\n"
        "out body geom;\n"
    )


def fetch_overpass(region_id: str, query: str) -> dict:
    cache = OSM_CACHE_DIR / f"{region_id}-roads.json"
    if cache.is_file() and cache.stat().st_size > 200:
        print(f"  using cached Overpass response: {cache}", flush=True)
        with cache.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    OSM_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    data = urllib.parse.urlencode({"data": query}).encode("utf-8")
    last_err: Exception | None = None
    for attempt in (1, 2):
        try:
            print(f"  POST {OVERPASS_URL} (attempt {attempt})", flush=True)
            req = urllib.request.Request(
                OVERPASS_URL, data=data,
                headers={
                    "User-Agent": "open-plates/0.1 (build_basemap_roads)",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as r:
                body = r.read()
            cache.write_bytes(body)
            return json.loads(body)
        except urllib.error.HTTPError as e:
            last_err = e
            print(f"    HTTP {e.code}: {e.reason}", flush=True)
            if e.code in (429, 500, 502, 503, 504) and attempt == 1:
                time.sleep(10)
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = e
            print(f"    net error: {e}", flush=True)
            if attempt == 1:
                time.sleep(10)
                continue
            raise
    raise RuntimeError(f"overpass fetch failed: {last_err}")


# ---------------------------------------------------------------------------
# Douglas-Peucker simplification (same algorithm as build_basemap_populated;
# duplicated here so each script stays self-contained)
# ---------------------------------------------------------------------------


def _perp_dist(pt, a, b):
    (x, y), (x1, y1), (x2, y2) = pt, a, b
    dx, dy = x2 - x1, y2 - y1
    if dx == 0.0 and dy == 0.0:
        return math.hypot(x - x1, y - y1)
    num = abs(dy * x - dx * y + x2 * y1 - y2 * x1)
    denom = math.hypot(dx, dy)
    return num / denom


def douglas_peucker(points, tol):
    if len(points) < 3:
        return list(points)
    keep = [False] * len(points)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = points[i], points[j]
        max_d = -1.0
        max_k = -1
        for k in range(i + 1, j):
            d = _perp_dist(points[k], a, b)
            if d > max_d:
                max_d = d
                max_k = k
        if max_d > tol and max_k > 0:
            keep[max_k] = True
            stack.append((i, max_k))
            stack.append((max_k, j))
    return [p for p, flag in zip(points, keep) if flag]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def extract_features(
    osm: dict,
    bbox: tuple[float, float, float, float],
    simplify_tol_deg: float = 0.0003,
) -> list[dict]:
    lat_min, lon_min, lat_max, lon_max = bbox
    elements = osm.get("elements") or []
    features: list[dict] = []
    for el in elements:
        if el.get("type") != "way":
            continue
        tags = el.get("tags") or {}
        hw = (tags.get("highway") or "").lower()
        if hw not in HIGHWAY_MAP:
            continue
        cls, rank = HIGHWAY_MAP[hw]
        geom = el.get("geometry") or []
        pts: list[tuple[float, float]] = []
        for node in geom:
            lon = node.get("lon")
            lat = node.get("lat")
            if lon is None or lat is None:
                continue
            pts.append((float(lon), float(lat)))
        if len(pts) < 2:
            continue
        # Bounding-box reject if the polyline lies entirely outside the bbox.
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        if (max(xs) < lon_min or min(xs) > lon_max
                or max(ys) < lat_min or min(ys) > lat_max):
            continue
        simp = douglas_peucker(pts, simplify_tol_deg)
        if len(simp) < 2:
            continue
        # Drop micro-segments whose full extent is below the simplification
        # tolerance — they read as dots at chart scale and inflate file
        # size. Keep anything whose bbox is at least ``simplify_tol_deg``
        # wide OR tall. ~0.0003 deg at the Eifel bbox ~= 33 m so we drop
        # parking-lot access ramps and dead-end stubs but keep streets.
        sxs = [p[0] for p in simp]
        sys_ = [p[1] for p in simp]
        if (max(sxs) - min(sxs) < simplify_tol_deg
                and max(sys_) - min(sys_) < simplify_tol_deg):
            continue
        name_tag = tags.get("ref") or tags.get("name") or ""
        name = str(name_tag).strip()
        # 5-decimal precision ~ 1.1 m at this latitude — plenty for a
        # chart underlay and cuts the on-disk size ~40%.
        coords = [[round(x, 5), round(y, 5)] for x, y in simp]
        oid = el.get("id")
        feature = {
            "type": "Feature",
            "id": f"way-{oid}",
            "properties": {
                "class": cls,
                "name": name,
                "rank": rank,
            },
            "geometry": {"type": "LineString", "coordinates": coords},
        }
        features.append(feature)

    # Deterministic ordering: rank asc, class asc, name/id asc, first-coord asc.
    def _sort_key(f: dict) -> tuple:
        props = f["properties"]
        coords = f["geometry"]["coordinates"] or [[0.0, 0.0]]
        first = coords[0]
        return (
            int(props.get("rank", 9)),
            str(props.get("class", "")),
            str(props.get("name", "")) or str(f.get("id", "")),
            round(float(first[0]), 6),
            round(float(first[1]), 6),
            str(f.get("id", "")),
        )

    features.sort(key=_sort_key)
    return features


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build(
    region_id: str,
    bbox: tuple[float, float, float, float],
    simplify_tol_deg: float = 0.0003,
) -> Path:
    lat_min, lon_min, lat_max, lon_max = bbox
    print(f"building roads for region={region_id} bbox={bbox}", flush=True)
    query = build_query(bbox)
    try:
        osm = fetch_overpass(region_id, query)
    except Exception as exc:
        print(f"ERROR: Overpass fetch failed: {exc}", file=sys.stderr)
        sys.exit(1)

    elements = osm.get("elements") or []
    print(f"  OSM elements returned: {len(elements)}")

    features = extract_features(osm, bbox, simplify_tol_deg=simplify_tol_deg)
    # Per-class counts for the build log.
    by_class: dict[str, int] = {}
    for f in features:
        cls = str((f.get("properties") or {}).get("class") or "")
        by_class[cls] = by_class.get(cls, 0) + 1
    for cls in sorted(by_class):
        print(f"    {cls}: {by_class[cls]}")
    print(f"  road features emitted: {len(features)}")

    fc = {
        "type": "FeatureCollection",
        "description": (
            f"Auto-generated road-network polylines for region {region_id!r}. "
            f"Built by scripts/build_basemap_roads.py from OpenStreetMap "
            f"(Overpass API). Features tagged class=road_motorway | road_trunk "
            f"| road_primary | road_secondary | road_tertiary; rank 1 = "
            f"motorway/trunk, 2 = primary, 3 = secondary, 4 = tertiary. "
            f"Simplified at ~{simplify_tol_deg:g} deg (~33 m). "
            f"Data (c) OpenStreetMap contributors, ODbL."
        ),
        "region": region_id,
        "bbox": [lat_min, lon_min, lat_max, lon_max],
        "features": features,
    }

    BASEMAP_DIR.mkdir(parents=True, exist_ok=True)
    out_path = BASEMAP_DIR / f"{region_id}-roads.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(fc, fh, indent=0, separators=(",", ":"))
        fh.write("\n")
    print(f"  wrote {out_path} ({out_path.stat().st_size:,} bytes)")
    return out_path


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("region_id")
    ap.add_argument("lat_min", type=float, nargs="?")
    ap.add_argument("lon_min", type=float, nargs="?")
    ap.add_argument("lat_max", type=float, nargs="?")
    ap.add_argument("lon_max", type=float, nargs="?")
    ap.add_argument("--simplify-tol", type=float, default=0.0003)
    args = ap.parse_args(argv)

    positionals = (args.lat_min, args.lon_min, args.lat_max, args.lon_max)
    given = [v for v in positionals if v is not None]
    if len(given) not in (0, 4):
        print(
            "must pass either 0 bbox args (resolve via registry) or all 4 "
            "(lat_min lon_min lat_max lon_max)",
            file=sys.stderr,
        )
        return 2
    if len(given) == 0:
        try:
            bbox = load_region_bbox(args.region_id)
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
    else:
        bbox = (args.lat_min, args.lon_min, args.lat_max, args.lon_max)
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        print("bbox must have lat_max > lat_min and lon_max > lon_min", file=sys.stderr)
        return 2

    build(args.region_id, bbox, simplify_tol_deg=args.simplify_tol)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
