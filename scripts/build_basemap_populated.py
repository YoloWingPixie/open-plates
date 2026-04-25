"""Build a populated-area polygon basemap layer from OpenStreetMap.

Usage:
    uv run python scripts/build_basemap_populated.py <region_id> \
        [<lat_min> <lon_min> <lat_max> <lon_max>]

When only ``<region_id>`` is supplied, the bbox is looked up in
``data/basemap/regions.json``. The optional positional lat/lon form stays
supported for one-off builds that don't sit in the region registry.

Example (Spangdahlem, from registry):
    uv run python scripts/build_basemap_populated.py spangdahlem

Example (one-off, custom bbox):
    uv run python scripts/build_basemap_populated.py spangdahlem \
        49.5 6.0 50.3 7.0

The script:
 1. Sends a single Overpass QL query fetching admin_level=8 relations plus
    place=city|town|village|suburb ways inside the bbox. One retry after
    10 s on HTTP 429 or 5xx.
 2. Stitches relation "outer" member ways into a closed polygon, using OSM
    ``geometry`` member nodes returned by ``out body geom;``.
 3. Emits one GeoJSON Polygon feature per area with ``class=populated_area``,
    ``properties.name`` from ``tags.name``, ``properties.rank`` from
    ``tags.place`` (city=1, town=2, village=3, suburb=4) or 2 if absent.
 4. Simplifies each ring at ~0.0005 deg (~55 m) with pure-Python DP.
 5. Drops polygons wholly outside the bbox; keeps ones that partially overlap.
 6. Caches the Overpass response under ``data/cache/osm/<region>.json`` so
    subsequent runs stay offline.
 7. Writes ``data/basemap/<region>-populated.json``.

If Overpass is unreachable, the script prints an error and exits 1. Do not
ship mock data — the ``--synthetic`` flag is provided only as the
documented last-resort fallback (rectangular city placeholders around
pre-known centers, tagged ``synthetic: true``).
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

PLACE_RANK = {
    "city": 1,
    "town": 2,
    "village": 3,
    "suburb": 4,
    "hamlet": 5,
}

# Hand-authored last-resort city centers (Spangdahlem region only) — used
# only when Overpass is totally unreachable during build. Coords approx.
SYNTHETIC_CITIES = [
    ("Trier", 49.756, 6.641, 0.06, 0.04, 1),
    ("Bitburg", 49.974, 6.528, 0.03, 0.02, 2),
    ("Wittlich", 49.987, 6.893, 0.025, 0.02, 2),
    ("Luxembourg", 49.611, 6.130, 0.045, 0.035, 1),
    ("Daun", 50.193, 6.832, 0.02, 0.015, 3),
]


# ---------------------------------------------------------------------------
# Overpass fetch (with simple retry + on-disk cache)
# ---------------------------------------------------------------------------


def build_query(bbox: tuple[float, float, float, float]) -> str:
    """Overpass QL combining admin_level=8 polygons, closed place=* ways, and
    place=* nodes. The polygons give us "painted" populated areas; the nodes
    are used as a filter — only admin polygons that contain a city/town node
    are retained, keeping the output to a readable 20-60 polygons instead of
    every rural Ortsgemeinde in the bbox.
    """
    lat_min, lon_min, lat_max, lon_max = bbox
    return (
        "[out:json][timeout:90];\n"
        "(\n"
        f'  relation["boundary"="administrative"]["admin_level"~"^(7|8)$"]'
        f"({lat_min},{lon_min},{lat_max},{lon_max});\n"
        f'  way["place"~"^(city|town|village)$"]'
        f"({lat_min},{lon_min},{lat_max},{lon_max});\n"
        f'  node["place"~"^(city|town|village)$"]'
        f"({lat_min},{lon_min},{lat_max},{lon_max});\n"
        ");\n"
        "out body geom;\n"
    )


def fetch_overpass(region_id: str, query: str) -> dict:
    cache = OSM_CACHE_DIR / f"{region_id}.json"
    if cache.is_file() and cache.stat().st_size > 1000:
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
                    "User-Agent": "open-plates/0.1 (build_basemap_populated)",
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
# Geometry stitching
# ---------------------------------------------------------------------------


def _round_pt(pt: tuple[float, float], places: int = 7) -> tuple[float, float]:
    return (round(pt[0], places), round(pt[1], places))


def _ring_from_way_geom(geom: list[dict]) -> list[tuple[float, float]]:
    """OSM geometry entries carry ``lat``/``lon`` per node. Return [lon, lat]."""
    return [(float(n["lon"]), float(n["lat"])) for n in geom if "lon" in n and "lat" in n]


def _stitch_outer_rings(relation: dict) -> list[list[tuple[float, float]]]:
    """Walk a relation's ``outer`` ways, chaining them into closed rings.

    Overpass returns each member way with its own ``geometry``. Ways may be
    in any order and may need to be reversed to make endpoints meet. We do a
    simple greedy chain: start with any unused outer way, keep appending ways
    whose endpoint matches, flipping as needed.
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
        # Close if the ring isn't already closed (common for slightly fuzzy geoms).
        if _round_pt(chain[0]) != _round_pt(chain[-1]):
            chain.append(chain[0])
        if len(chain) >= 4:
            rings.append(chain)
    return rings


def _ring_bbox(ring: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return (min(xs), min(ys), max(xs), max(ys))


def _ring_overlaps(ring: list[tuple[float, float]], bbox: tuple[float, float, float, float]) -> bool:
    lat_min, lon_min, lat_max, lon_max = bbox
    rx_min, ry_min, rx_max, ry_max = _ring_bbox(ring)
    return not (
        rx_max < lon_min or rx_min > lon_max
        or ry_max < lat_min or ry_min > lat_max
    )


def _ring_centroid(ring: list[tuple[float, float]]) -> tuple[float, float]:
    # Unsigned-area weighted centroid (good enough; we won't use it for labels
    # here — labels are computed at render time from polygon coords).
    n = len(ring) - 1
    if n < 3:
        # Degenerate: fall back to mean.
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        return (sum(xs) / len(xs), sum(ys) / len(ys))
    cx = sum(p[0] for p in ring[:-1]) / n
    cy = sum(p[1] for p in ring[:-1]) / n
    return (cx, cy)


# ---------------------------------------------------------------------------
# Douglas-Peucker (same algorithm as the topo script; duplicated so each
# script is self-contained and importable without cross-coupling)
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
# Feature building
# ---------------------------------------------------------------------------


def _point_in_ring(pt: tuple[float, float], ring: list[tuple[float, float]]) -> bool:
    """Ray-cast point-in-polygon (ring is [lon, lat], closed)."""
    x, y = pt
    n = len(ring)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
        ):
            inside = not inside
        j = i
    return inside


def extract_features(
    osm: dict, bbox: tuple[float, float, float, float],
    simplify_tol_deg: float = 0.0005,
) -> list[dict]:
    elements = osm.get("elements") or []
    features: list[dict] = []

    # Collect place=city|town|village nodes — used to FILTER admin polygons
    # down to "notable" ones (otherwise we get every Ortsgemeinde).
    place_nodes: list[tuple[tuple[float, float], str, int]] = []  # (lonlat, name, rank)
    for el in elements:
        if el.get("type") != "node":
            continue
        tags = el.get("tags") or {}
        place_tag = tags.get("place") or ""
        if place_tag not in PLACE_RANK:
            continue
        lat = el.get("lat")
        lon = el.get("lon")
        if lat is None or lon is None:
            continue
        place_nodes.append((
            (float(lon), float(lat)),
            tags.get("name") or "",
            PLACE_RANK[place_tag],
        ))
    # Accept city/town/village nodes as polygon filters. Keeps the output
    # focused on real settlements (not hamlets or suburbs).
    notable_nodes = [n for n in place_nodes if n[2] <= 3]
    print(f"  place nodes: {len(place_nodes)} total, {len(notable_nodes)} city/town/village")

    # First pass: collect all admin polygons with ring data + bbox area.
    # Then per city/town node pick the smallest-area polygon that contains
    # it. This prevents a coarser Kreis (admin_level=6) polygon from
    # swallowing every city in its territory.
    def _ring_area_abs(ring: list[tuple[float, float]]) -> float:
        n = len(ring) - 1
        if n < 3:
            return 0.0
        s = 0.0
        for i in range(n):
            xi, yi = ring[i]
            xj, yj = ring[i + 1]
            s += xi * yj - xj * yi
        return abs(s) * 0.5

    admin_polys: list[tuple[float, str, str, list[list[tuple[float, float]]]]] = []
    for el in elements:
        if el.get("type") != "relation":
            continue
        tags = el.get("tags") or {}
        if tags.get("boundary") != "administrative":
            continue
        admin_level = tags.get("admin_level")
        if admin_level not in ("7", "8"):
            continue
        name = tags.get("name")
        if not name:
            continue
        rings = _stitch_outer_rings(el)
        rings = [r for r in rings if _ring_overlaps(r, bbox)]
        if not rings:
            continue
        total_area = sum(_ring_area_abs(r) for r in rings)
        admin_polys.append((total_area, admin_level, name, rings))

    # For each city/town node, pick the smallest-area admin polygon that
    # contains it. Track the area so we can cap total output below.
    chosen: dict[tuple[str, int], tuple[list[list[tuple[float, float]]], int, float]] = {}
    for pt, nd_name, nd_rank in notable_nodes:
        best: tuple[float, list[list[tuple[float, float]]], str] | None = None
        for area, admin_level, name, rings in admin_polys:
            if not any(_point_in_ring(pt, r) for r in rings):
                continue
            if best is None or area < best[0]:
                best = (area, rings, name)
        if best is None:
            continue
        best_area, rings, polygon_name = best
        key = (polygon_name, nd_rank)
        # Prefer lower node rank (city > town > village) when a polygon is
        # claimed by multiple nodes.
        if key not in chosen or chosen[key][1] > nd_rank:
            chosen[key] = (rings, nd_rank, best_area)

    # Cap total polygons: rank by (node rank asc, area desc). City > town >
    # village, and bigger polygons beat smaller ones at the same rank.
    # Note: build_basemap_populated.py doesn't know the procedure's fix
    # bbox, so we can't filter by proximity to the procedure here — that
    # happens at render time. Just ship a focused top-N by importance.
    MAX_POLYGONS = 30
    ranked = sorted(
        chosen.items(),
        key=lambda kv: (kv[1][1], -kv[1][2]),  # rank asc, then area desc
    )
    chosen = dict(ranked[:MAX_POLYGONS])
    chosen = {k: (v[0], v[1]) for k, v in chosen.items()}
    print(f"  polygons after rank cap (MAX={MAX_POLYGONS}): {len(chosen)}")

    # Emit the chosen polygons as features.
    for (polygon_name, _node_rank), (rings, rank) in chosen.items():
        name = polygon_name
        simplified_rings = []
        for ring in rings:
            simp = douglas_peucker(ring, simplify_tol_deg)
            if len(simp) < 4:
                continue
            # Close if DP dropped the last point accidentally.
            if simp[0] != simp[-1]:
                simp.append(simp[0])
            simplified_rings.append(simp)
        if not simplified_rings:
            continue
        # One Polygon feature per ring — simpler than MultiPolygon and the
        # renderer's polygonal-fill helper handles both.
        for idx, ring in enumerate(simplified_rings):
            coords = [[round(x, 6), round(y, 6)] for x, y in ring]
            slug = "".join(c.lower() if c.isalnum() else "-" for c in name).strip("-")
            fid = f"admin-{slug}-{idx}"
            features.append({
                "type": "Feature",
                "id": fid,
                "properties": {
                    "class": "populated_area",
                    "name": name,
                    "rank": rank,
                    "osm_type": "relation",
                },
                "geometry": {"type": "Polygon", "coordinates": [coords]},
            })

    # Closed ways tagged place=*.
    for el in elements:
        if el.get("type") != "way":
            continue
        tags = el.get("tags") or {}
        place_tag = tags.get("place") or ""
        if place_tag not in PLACE_RANK:
            continue
        name = tags.get("name")
        if not name:
            continue
        geom = el.get("geometry") or []
        ring = _ring_from_way_geom(geom)
        if len(ring) < 4:
            continue
        if _round_pt(ring[0]) != _round_pt(ring[-1]):
            ring.append(ring[0])
        if not _ring_overlaps(ring, bbox):
            continue
        simp = douglas_peucker(ring, simplify_tol_deg)
        if len(simp) < 4:
            continue
        if simp[0] != simp[-1]:
            simp.append(simp[0])
        coords = [[round(x, 6), round(y, 6)] for x, y in simp]
        features.append({
            "type": "Feature",
            "id": f"way-{el.get('id')}",
            "properties": {
                "class": "populated_area",
                "name": name,
                "rank": PLACE_RANK[place_tag],
                "osm_id": el.get("id"),
                "osm_type": "way",
            },
            "geometry": {"type": "Polygon", "coordinates": [coords]},
        })

    # Deterministic order: rank then name then id.
    features.sort(key=lambda f: (
        f["properties"].get("rank", 9),
        f["properties"].get("name", ""),
        f["id"],
    ))
    return features


def synthetic_featurecollection(region_id: str, bbox: tuple[float, float, float, float]) -> dict:
    features = []
    for name, lat, lon, w, h, rank in SYNTHETIC_CITIES:
        ring = [
            [round(lon - w / 2, 6), round(lat - h / 2, 6)],
            [round(lon + w / 2, 6), round(lat - h / 2, 6)],
            [round(lon + w / 2, 6), round(lat + h / 2, 6)],
            [round(lon - w / 2, 6), round(lat + h / 2, 6)],
            [round(lon - w / 2, 6), round(lat - h / 2, 6)],
        ]
        features.append({
            "type": "Feature",
            "id": f"synthetic-{name.lower()}",
            "properties": {
                "class": "populated_area",
                "name": name,
                "rank": rank,
                "synthetic": True,
            },
            "geometry": {"type": "Polygon", "coordinates": [ring]},
        })
    return {
        "type": "FeatureCollection",
        "description": (
            f"SYNTHETIC placeholder populated-area layer for region "
            f"{region_id!r}. Overpass was unreachable at build time; these "
            f"rectangles are rough bounding boxes drawn around hand-known "
            f"city centers (Trier / Bitburg / Wittlich / Luxembourg / Daun). "
            f"Every feature is tagged synthetic=true. Regenerate by running "
            f"scripts/build_basemap_populated.py when Overpass is reachable."
        ),
        "region": region_id,
        "bbox": list(bbox),
        "synthetic": True,
        "features": features,
    }


def build(
    region_id: str, bbox: tuple[float, float, float, float],
    simplify_tol_deg: float = 0.0005,
    allow_synthetic: bool = False,
) -> Path:
    lat_min, lon_min, lat_max, lon_max = bbox
    print(f"building populated areas for region={region_id} bbox={bbox}", flush=True)
    query = build_query(bbox)
    try:
        osm = fetch_overpass(region_id, query)
    except Exception as exc:
        if allow_synthetic:
            print(f"  WARNING: Overpass fetch failed ({exc}); emitting synthetic layer", flush=True)
            fc = synthetic_featurecollection(region_id, bbox)
            BASEMAP_DIR.mkdir(parents=True, exist_ok=True)
            out_path = BASEMAP_DIR / f"{region_id}-populated.json"
            with out_path.open("w", encoding="utf-8") as fh:
                json.dump(fc, fh, indent=0, separators=(",", ":"))
                fh.write("\n")
            print(f"  wrote SYNTHETIC {out_path} ({len(fc['features'])} features)")
            return out_path
        print(f"ERROR: Overpass fetch failed: {exc}", file=sys.stderr)
        sys.exit(1)

    elements = osm.get("elements") or []
    print(f"  OSM elements returned: {len(elements)}")

    features = extract_features(osm, bbox, simplify_tol_deg=simplify_tol_deg)
    print(f"  polygon features emitted: {len(features)}")

    fc = {
        "type": "FeatureCollection",
        "description": (
            f"Auto-generated populated-area polygons for region "
            f"{region_id!r}. Built by scripts/build_basemap_populated.py "
            f"from OpenStreetMap (Overpass API). Features tagged "
            f"class=populated_area; rank 1=city, 2=town, 3=village, 4=suburb. "
            f"Data (c) OpenStreetMap contributors, ODbL."
        ),
        "region": region_id,
        "bbox": [lat_min, lon_min, lat_max, lon_max],
        "features": features,
    }

    BASEMAP_DIR.mkdir(parents=True, exist_ok=True)
    out_path = BASEMAP_DIR / f"{region_id}-populated.json"
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
    ap.add_argument(
        "--allow-synthetic", action="store_true",
        help=(
            "Fall back to a hard-coded rectangular placeholder layer if "
            "Overpass is unreachable. Only for the documented last-resort."
        ),
    )
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

    build(
        args.region_id, bbox,
        simplify_tol_deg=args.simplify_tol,
        allow_synthetic=args.allow_synthetic,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
