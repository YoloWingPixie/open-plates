"""Build a waterways (rivers, streams, lakes) basemap layer from OpenStreetMap.

Usage:
    uv run python scripts/build_basemap_waterways.py <region_id> \
        [<lat_min> <lon_min> <lat_max> <lon_max>]

When only ``<region_id>`` is supplied, the bbox is looked up in
``data/basemap/regions.json``. The optional positional lat/lon form stays
supported for one-off builds that don't sit in the region registry.

Example (Spangdahlem, from registry):
    uv run python scripts/build_basemap_waterways.py spangdahlem

Example (one-off, custom bbox):
    uv run python scripts/build_basemap_waterways.py spangdahlem \
        49.3 5.8 50.5 7.2

The script:
 1. Queries Overpass for ``waterway=river|canal|stream`` ways and
    relations plus ``natural=water`` ways and relations. Same retry/cache
    pattern as ``build_basemap_populated.py`` / ``build_basemap_roads.py``
    (one retry after 10 s on HTTP 429 or 5xx).
 2. Extracts each river/canal/stream way's geometry and emits a GeoJSON
    ``LineString`` feature:
      - ``class`` = ``river`` (river/canal) or ``stream``.
      - ``name``  = ``tags.name`` with ``name:de``/``name:en`` fallback.
      - ``rank``  = 1 river, 2 canal, 3 stream.
 3. Extracts each lake's outer ring (``natural=water``, filtered by
    ``water=lake|reservoir|basin|pond`` OR projected area > 0.0001 sq deg)
    and emits one GeoJSON ``Polygon`` feature per outer ring:
      - ``class`` = ``lake``.
      - ``rank``  = 1 (>10 km^2), 2 (1-10 km^2), 3 (<1 km^2).
 4. Simplifies line/ring geometry with Douglas-Peucker at ~0.0005 deg
    (~55 m).
 5. Drops streams whose bbox extent is < 0.003 deg in both axes — too
    small to read at plate scale.
 6. Sorts deterministically by (rank asc, name, id).
 7. Caches the Overpass response under
    ``data/cache/osm/<region>-waterways.json`` so subsequent runs stay
    offline.
 8. Writes ``data/basemap/<region>-waterways.json``.

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

# OSM waterway tag -> (class, rank). Rank 1 = river, 2 = canal, 3 = stream.
WATERWAY_MAP: dict[str, tuple[str, int]] = {
    "river": ("river", 1),
    "canal": ("river", 2),
    "stream": ("stream", 3),
}

# water sub-tag values we accept as "lakes" when the polygon area is
# below the geometric threshold. Everything else (riverbank, wastewater,
# pool, fountain, moat, etc.) is dropped unless the polygon is large.
LAKE_WATER_TAGS = {"lake", "reservoir", "basin", "pond"}

# Minimum projected polygon area (sq deg) for unfiltered natural=water
# polygons. ~0.0001 sq deg at mid-latitude ~= 1 km^2. Catches unlabeled
# reservoirs and big water bodies whose OSM ``water`` tag is missing.
LAKE_MIN_AREA_SQDEG = 0.0001

# Stream polyline drop threshold (bbox extent in deg). 0.003 deg ~= 330 m
# at mid-latitude; too short to read on a 1:500 000 plate.
STREAM_MIN_EXTENT_DEG = 0.003


# ---------------------------------------------------------------------------
# Overpass fetch (with simple retry + on-disk cache)
# ---------------------------------------------------------------------------


def build_query(bbox: tuple[float, float, float, float]) -> str:
    lat_min, lon_min, lat_max, lon_max = bbox
    bb = f"{lat_min},{lon_min},{lat_max},{lon_max}"
    return (
        "[out:json][timeout:120];\n"
        "(\n"
        f'  way["waterway"~"^(river|canal|stream)$"]({bb});\n'
        f'  relation["waterway"~"^(river|canal)$"]({bb});\n'
        f'  way["natural"="water"]({bb});\n'
        f'  relation["natural"="water"]({bb});\n'
        ");\n"
        "out body geom;\n"
    )


def fetch_overpass(region_id: str, query: str) -> dict:
    cache = OSM_CACHE_DIR / f"{region_id}-waterways.json"
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
                    "User-Agent": "open-plates/0.1 (build_basemap_waterways)",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=180) as r:
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
# Douglas-Peucker simplification (duplicated from sibling scripts so each
# build script stays self-contained)
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
# Geometry helpers
# ---------------------------------------------------------------------------


def _round_pt(pt: tuple[float, float], places: int = 7) -> tuple[float, float]:
    return (round(pt[0], places), round(pt[1], places))


def _ring_from_way_geom(geom: list[dict]) -> list[tuple[float, float]]:
    """OSM ``geometry`` nodes have ``lat``/``lon``. Return [(lon, lat), ...]."""
    return [
        (float(n["lon"]), float(n["lat"]))
        for n in geom
        if "lon" in n and "lat" in n
    ]


def _stitch_outer_rings(relation: dict) -> list[list[tuple[float, float]]]:
    """Walk a relation's ``outer`` member ways, chaining them into closed
    rings. Greedy endpoint match with way-flipping. Identical to the
    implementation in ``build_basemap_populated.py``; duplicated to keep
    this script self-contained.
    """
    members = relation.get("members") or []
    ways: list[list[tuple[float, float]]] = []
    for m in members:
        if m.get("role") != "outer" or m.get("type") != "way":
            continue
        geom = m.get("geometry") or []
        pts = _ring_from_way_geom(geom)
        if len(pts) >= 2:
            ways.append(pts)

    rings: list[list[tuple[float, float]]] = []
    while ways:
        chain = list(ways.pop(0))
        changed = True
        while changed and _round_pt(chain[0]) != _round_pt(chain[-1]):
            changed = False
            for i in range(len(ways)):
                w = ways[i]
                if _round_pt(w[0]) == _round_pt(chain[-1]):
                    chain.extend(w[1:])
                    ways.pop(i)
                    changed = True
                    break
                if _round_pt(w[-1]) == _round_pt(chain[-1]):
                    chain.extend(reversed(w[:-1]))
                    ways.pop(i)
                    changed = True
                    break
                if _round_pt(w[-1]) == _round_pt(chain[0]):
                    chain[0:0] = w[:-1]
                    ways.pop(i)
                    changed = True
                    break
                if _round_pt(w[0]) == _round_pt(chain[0]):
                    chain[0:0] = list(reversed(w[1:]))
                    ways.pop(i)
                    changed = True
                    break
        if _round_pt(chain[0]) != _round_pt(chain[-1]):
            chain.append(chain[0])
        if len(chain) >= 4:
            rings.append(chain)
    return rings


def _ring_area_abs(ring: list[tuple[float, float]]) -> float:
    """Shoelace unsigned area in sq deg for a closed ring (last==first)."""
    n = len(ring) - 1
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        xi, yi = ring[i]
        xj, yj = ring[i + 1]
        s += xi * yj - xj * yi
    return abs(s) * 0.5


def _ring_overlaps(
    ring: list[tuple[float, float]],
    bbox: tuple[float, float, float, float],
) -> bool:
    lat_min, lon_min, lat_max, lon_max = bbox
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    rx_min, ry_min, rx_max, ry_max = min(xs), min(ys), max(xs), max(ys)
    return not (
        rx_max < lon_min or rx_min > lon_max
        or ry_max < lat_min or ry_min > lat_max
    )


def _lake_rank_from_area(area_sqdeg: float) -> int:
    """Rank a lake polygon by projected area (sq deg).

    Thresholds:
      - rank 1 : area > ~10 km^2 (~0.001 sq deg)
      - rank 2 : area 1-10 km^2 (0.0001-0.001 sq deg)
      - rank 3 : area < 1 km^2
    """
    if area_sqdeg >= 0.001:
        return 1
    if area_sqdeg >= 0.0001:
        return 2
    return 3


def _pick_name(tags: dict) -> str:
    name = tags.get("name")
    if name:
        return str(name).strip()
    for key in ("name:de", "name:en"):
        v = tags.get(key)
        if v:
            return str(v).strip()
    return ""


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def _extract_rivers(
    elements: list[dict],
    bbox: tuple[float, float, float, float],
    simplify_tol_deg: float,
) -> list[dict]:
    lat_min, lon_min, lat_max, lon_max = bbox
    features: list[dict] = []
    for el in elements:
        if el.get("type") != "way":
            continue
        tags = el.get("tags") or {}
        ww = str(tags.get("waterway") or "").lower()
        if ww not in WATERWAY_MAP:
            continue
        cls, rank = WATERWAY_MAP[ww]
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
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        if (max(xs) < lon_min or min(xs) > lon_max
                or max(ys) < lat_min or min(ys) > lat_max):
            continue
        # Drop short streams below the plate-scale threshold.
        if cls == "stream":
            if (max(xs) - min(xs) < STREAM_MIN_EXTENT_DEG
                    and max(ys) - min(ys) < STREAM_MIN_EXTENT_DEG):
                continue
        simp = douglas_peucker(pts, simplify_tol_deg)
        if len(simp) < 2:
            continue
        name = _pick_name(tags)
        coords = [[round(x, 5), round(y, 5)] for x, y in simp]
        oid = el.get("id")
        features.append({
            "type": "Feature",
            "id": f"way-{oid}",
            "properties": {
                "class": cls,
                "name": name,
                "rank": rank,
            },
            "geometry": {"type": "LineString", "coordinates": coords},
        })
    return features


def _extract_lakes(
    elements: list[dict],
    bbox: tuple[float, float, float, float],
    simplify_tol_deg: float,
) -> list[dict]:
    """Extract lake polygons. Emit one Feature per outer ring.

    Sources:
      - ``way[natural=water]``         -> single outer ring.
      - ``relation[natural=water]``    -> one or more stitched outer rings.
    Inner rings (islands) are skipped for v1.
    """
    features: list[dict] = []

    # First: closed ways with natural=water.
    for el in elements:
        if el.get("type") != "way":
            continue
        tags = el.get("tags") or {}
        if tags.get("natural") != "water":
            continue
        water_tag = str(tags.get("water") or "").lower()
        geom = el.get("geometry") or []
        ring = _ring_from_way_geom(geom)
        if len(ring) < 4:
            continue
        if _round_pt(ring[0]) != _round_pt(ring[-1]):
            ring.append(ring[0])
        if not _ring_overlaps(ring, bbox):
            continue
        area = _ring_area_abs(ring)
        keep = water_tag in LAKE_WATER_TAGS or area >= LAKE_MIN_AREA_SQDEG
        if not keep:
            continue
        simp = douglas_peucker(ring, simplify_tol_deg)
        if len(simp) < 4:
            continue
        if simp[0] != simp[-1]:
            simp.append(simp[0])
        coords = [[round(x, 6), round(y, 6)] for x, y in simp]
        oid = el.get("id")
        name = _pick_name(tags)
        rank = _lake_rank_from_area(area)
        features.append({
            "type": "Feature",
            "id": f"way-{oid}",
            "properties": {
                "class": "lake",
                "name": name,
                "rank": rank,
            },
            "geometry": {"type": "Polygon", "coordinates": [coords]},
        })

    # Second: relations with natural=water — stitch outer rings.
    for el in elements:
        if el.get("type") != "relation":
            continue
        tags = el.get("tags") or {}
        if tags.get("natural") != "water":
            continue
        water_tag = str(tags.get("water") or "").lower()
        rings = _stitch_outer_rings(el)
        rings = [r for r in rings if _ring_overlaps(r, bbox)]
        if not rings:
            continue
        name = _pick_name(tags)
        rid = el.get("id")
        for idx, ring in enumerate(rings):
            area = _ring_area_abs(ring)
            keep = water_tag in LAKE_WATER_TAGS or area >= LAKE_MIN_AREA_SQDEG
            if not keep:
                continue
            simp = douglas_peucker(ring, simplify_tol_deg)
            if len(simp) < 4:
                continue
            if simp[0] != simp[-1]:
                simp.append(simp[0])
            coords = [[round(x, 6), round(y, 6)] for x, y in simp]
            rank = _lake_rank_from_area(area)
            features.append({
                "type": "Feature",
                "id": f"rel-{rid}-{idx}",
                "properties": {
                    "class": "lake",
                    "name": name,
                    "rank": rank,
                },
                "geometry": {"type": "Polygon", "coordinates": [coords]},
            })

    return features


def extract_features(
    osm: dict,
    bbox: tuple[float, float, float, float],
    simplify_tol_deg: float = 0.0005,
) -> list[dict]:
    elements = osm.get("elements") or []
    rivers = _extract_rivers(elements, bbox, simplify_tol_deg)
    lakes = _extract_lakes(elements, bbox, simplify_tol_deg)
    features = rivers + lakes

    # Deterministic ordering: rank asc, name, id.
    features.sort(key=lambda f: (
        int((f.get("properties") or {}).get("rank") or 9),
        str((f.get("properties") or {}).get("name") or ""),
        str(f.get("id") or ""),
    ))
    return features


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build(
    region_id: str,
    bbox: tuple[float, float, float, float],
    simplify_tol_deg: float = 0.0005,
) -> Path:
    lat_min, lon_min, lat_max, lon_max = bbox
    print(f"building waterways for region={region_id} bbox={bbox}", flush=True)
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
    print(f"  waterway features emitted: {len(features)}")

    fc = {
        "type": "FeatureCollection",
        "description": (
            f"Auto-generated waterways (rivers, streams, lakes) for region "
            f"{region_id!r}. Built by scripts/build_basemap_waterways.py "
            f"from OpenStreetMap (Overpass API). LineString features tagged "
            f"class=river|stream (rank 1 river, 2 canal, 3 stream). "
            f"Polygon features tagged class=lake (rank 1 >10 km^2, "
            f"2 1-10 km^2, 3 <1 km^2). Simplified at ~{simplify_tol_deg:g} "
            f"deg (~55 m). Data (c) OpenStreetMap contributors, ODbL."
        ),
        "region": region_id,
        "bbox": [lat_min, lon_min, lat_max, lon_max],
        "features": features,
    }

    BASEMAP_DIR.mkdir(parents=True, exist_ok=True)
    out_path = BASEMAP_DIR / f"{region_id}-waterways.json"
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
    ap.add_argument("--simplify-tol", type=float, default=0.0005)
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
