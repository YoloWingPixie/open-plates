"""Build a coastline/sea basemap layer from OpenStreetMap coastline ways.

Usage:
    uv run python scripts/build_basemap_coastline.py <region_id>

Example:
    uv run python scripts/build_basemap_coastline.py batumi

The output is ``data/basemap/<region>-coastline.json`` containing:
  * one ``class=sea`` polygon clipped to the region bbox
  * one ``class=coast`` LineString following stitched OSM coastline ways

The sea-polygon builder is intentionally conservative: it assumes the
water body touches the west side of the region bbox unless ``--water-side``
is supplied. That matches the current Batumi/Black Sea region and keeps
the generated layer deterministic without introducing a full polygon
clipping dependency.
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
BASEMAP_DIR = REPO_ROOT / "data" / "basemap"
OSM_CACHE_DIR = REPO_ROOT / "data" / "cache" / "osm"
REGIONS_PATH = BASEMAP_DIR / "regions.json"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def load_region_bbox(region_id: str) -> tuple[float, float, float, float]:
    with REGIONS_PATH.open("r", encoding="utf-8") as fh:
        registry = json.load(fh)
    bbox = (registry.get(region_id) or {}).get("bbox")
    if not (isinstance(bbox, list) and len(bbox) == 4):
        known = ", ".join(sorted(registry))
        raise SystemExit(f"unknown or invalid region {region_id!r}; known: {known}")
    lon_min, lat_min, lon_max, lat_max = (float(v) for v in bbox)
    return lat_min, lon_min, lat_max, lon_max


def overpass_bbox(bbox: tuple[float, float, float, float]) -> str:
    lat_min, lon_min, lat_max, lon_max = bbox
    return f"{lat_min},{lon_min},{lat_max},{lon_max}"


def build_query(bbox: tuple[float, float, float, float]) -> str:
    bb = overpass_bbox(bbox)
    return (
        "[out:json][timeout:90];\n"
        "(\n"
        f'  way["natural"="coastline"]({bb});\n'
        ");\n"
        "out geom;\n"
    )


def fetch_overpass(region_id: str, query: str) -> dict:
    cache = OSM_CACHE_DIR / f"{region_id}-coastline.json"
    if cache.is_file():
        with cache.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    OSM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    data = urllib.parse.urlencode({"data": query}).encode("utf-8")
    req = urllib.request.Request(
        OVERPASS_URL,
        data=data,
        headers={
            "User-Agent": "open-plates/0.1 (build_basemap_coastline)",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    last_err: Exception | None = None
    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                payload = resp.read().decode("utf-8")
            parsed = json.loads(payload)
            with cache.open("w", encoding="utf-8") as fh:
                json.dump(parsed, fh, separators=(",", ":"))
            return parsed
        except urllib.error.HTTPError as exc:
            last_err = exc
            if exc.code not in (429, 500, 502, 503, 504) or attempt:
                break
            time.sleep(10.0)
        except Exception as exc:  # pragma: no cover - network defensive
            last_err = exc
            if attempt:
                break
            time.sleep(10.0)
    raise RuntimeError(f"overpass fetch failed: {last_err}")


def _pt_key(pt: tuple[float, float], scale: float = 1_000_000.0) -> tuple[int, int]:
    return (round(pt[0] * scale), round(pt[1] * scale))


def _dist2(a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _perp_dist_sq(
    p: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
) -> float:
    ax, ay = a
    bx, by = b
    px, py = p
    dx = bx - ax
    dy = by - ay
    denom = dx * dx + dy * dy
    if denom <= 0:
        return _dist2(p, a)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
    q = (ax + t * dx, ay + t * dy)
    return _dist2(p, q)


def simplify_line(
    pts: list[tuple[float, float]],
    tol_deg: float,
) -> list[tuple[float, float]]:
    if len(pts) <= 2:
        return pts
    tol_sq = tol_deg * tol_deg

    def rec(start: int, end: int, keep: set[int]) -> None:
        a = pts[start]
        b = pts[end]
        far_idx = -1
        far_dist = -1.0
        for i in range(start + 1, end):
            d = _perp_dist_sq(pts[i], a, b)
            if d > far_dist:
                far_dist = d
                far_idx = i
        if far_dist > tol_sq and far_idx > 0:
            keep.add(far_idx)
            rec(start, far_idx, keep)
            rec(far_idx, end, keep)

    keep = {0, len(pts) - 1}
    rec(0, len(pts) - 1, keep)
    return [pts[i] for i in sorted(keep)]


def extract_lines(osm: dict) -> list[list[tuple[float, float]]]:
    lines: list[list[tuple[float, float]]] = []
    for el in osm.get("elements") or []:
        tags = el.get("tags") or {}
        if tags.get("natural") != "coastline":
            continue
        geom = el.get("geometry") or []
        pts = [
            (float(p["lon"]), float(p["lat"]))
            for p in geom
            if "lat" in p and "lon" in p
        ]
        if len(pts) >= 2:
            lines.append(pts)
    return lines


def stitch_lines(lines: list[list[tuple[float, float]]]) -> list[tuple[float, float]]:
    """Greedily stitch coastline fragments by matching endpoints."""
    if not lines:
        return []
    remaining = [list(line) for line in lines]
    chain = remaining.pop(max(range(len(remaining)), key=lambda i: len(remaining[i])))
    changed = True
    while remaining and changed:
        changed = False
        head = _pt_key(chain[0])
        tail = _pt_key(chain[-1])
        for idx, line in enumerate(remaining):
            lh = _pt_key(line[0])
            lt = _pt_key(line[-1])
            if lt == head:
                chain = line[:-1] + chain
            elif lh == head:
                chain = list(reversed(line))[1:] + chain
            elif lh == tail:
                chain.extend(line[1:])
            elif lt == tail:
                chain.extend(list(reversed(line))[1:])
            else:
                continue
            remaining.pop(idx)
            changed = True
            break
    return chain


def bbox_boundary_path(
    start: tuple[float, float],
    end: tuple[float, float],
    bbox: tuple[float, float, float, float],
    water_side: str,
) -> list[tuple[float, float]]:
    """Return a coarse bbox edge path closing ``end`` back to ``start``."""
    lat_min, lon_min, lat_max, lon_max = bbox
    if water_side == "west":
        corners = [(lon_min, lat_max), (lon_min, lat_min)]
    elif water_side == "east":
        corners = [(lon_max, lat_min), (lon_max, lat_max)]
    elif water_side == "north":
        corners = [(lon_max, lat_max), (lon_min, lat_max)]
    elif water_side == "south":
        corners = [(lon_min, lat_min), (lon_max, lat_min)]
    else:
        raise ValueError(f"unknown water side: {water_side}")
    return [end, *corners, start]


def build(region_id: str, bbox: tuple[float, float, float, float]) -> Path:
    print(f"building coastline for region={region_id} bbox={bbox}", flush=True)
    osm = fetch_overpass(region_id, build_query(bbox))
    lines = extract_lines(osm)
    print(f"  coastline ways returned: {len(lines)}", flush=True)
    chain = stitch_lines(lines)
    if len(chain) < 2:
        raise RuntimeError("no stitchable coastline geometry returned from OSM")
    chain = simplify_line(chain, tol_deg=0.00025)
    sea_ring = chain + bbox_boundary_path(chain[-1], chain[0], bbox, "west")
    features = [
        {
            "type": "Feature",
            "id": f"{region_id}-sea",
            "properties": {"class": "sea", "name": "BLACK SEA", "rank": 1},
            "geometry": {"type": "Polygon", "coordinates": [sea_ring]},
        },
        {
            "type": "Feature",
            "id": f"{region_id}-coastline",
            "properties": {"class": "coast", "name": "GEORGIAN COAST", "rank": 1},
            "geometry": {"type": "LineString", "coordinates": chain},
        },
    ]
    lat_min, lon_min, lat_max, lon_max = bbox
    doc = {
        "type": "FeatureCollection",
        "description": (
            f"Auto-generated coastline and sea fill for region {region_id!r}. "
            "Built by scripts/build_basemap_coastline.py from OpenStreetMap "
            "natural=coastline ways. Data (c) OpenStreetMap contributors, ODbL."
        ),
        "region": region_id,
        "bbox": [lat_min, lon_min, lat_max, lon_max],
        "features": features,
    }
    out_path = BASEMAP_DIR / f"{region_id}-coastline.json"
    BASEMAP_DIR.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(f"  wrote {out_path.relative_to(REPO_ROOT)}", flush=True)
    return out_path


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("region_id")
    args = ap.parse_args(argv)
    build(args.region_id, load_region_bbox(args.region_id))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
