"""Build a tall-obstacle point basemap layer from OpenStreetMap.

Usage:
    uv run python scripts/build_basemap_obstacles.py <region_id> \
        [<lat_min> <lon_min> <lat_max> <lon_max>] [--min-ft N]

When only ``<region_id>`` is supplied, the bbox is looked up in
``data/basemap/regions.json``. The optional positional lat/lon form stays
supported for one-off builds that don't sit in the region registry.

Example (Spangdahlem, from registry):
    uv run python scripts/build_basemap_obstacles.py spangdahlem

Example (one-off, custom bbox):
    uv run python scripts/build_basemap_obstacles.py spangdahlem \
        49.5 6.0 50.3 7.0

Defaults to a 1000 ft AGL minimum tip-height filter — the threshold that
matches how real charts treat "tall obstacles" as navigation hazards.

The script:
 1. Queries Overpass for ``man_made=tower|mast|communications_tower|chimney``
    nodes and ways, plus ``power=generator`` + ``generator:source=wind``
    nodes. Same retry/cache pattern as build_basemap_populated.py.
 2. Parses a height (meters or feet; m|ft|bare number) from the best tag
    available (``height``, ``tower:height``, ``est_height``) or derives a
    default from ``tower:type`` / wind-turbine tags when a number is absent.
 3. Filters to features whose tip height >= min_ft (default 1000).
 4. Emits one GeoJSON Point per feature with ``class=obstacle_tall``.
 5. Caches the Overpass response under ``data/cache/osm/<region>-obstacles.json``.
 6. Writes ``data/basemap/<region>-obstacles.json``.

If zero features pass the filter, still writes a valid empty FeatureCollection
(with a note describing the result so future readers don't mistake it for a
failed build). Do not invent fake obstacles — an empty set is the right answer
for the Eifel / Luxembourg border region.
"""

from __future__ import annotations

import argparse
import json
import re
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

METERS_PER_FOOT = 0.3048

# tower:type fall-back heights (meters). Only used when no numeric height
# tag is present; picked as conservative typical values, not maxima.
TOWER_TYPE_DEFAULT_M = {
    "broadcasting": 150.0,
    "communication": 100.0,
    "observation": 80.0,
}

# Wind-turbine last-resort default (meters, tip height) when nothing else
# usable is present. Modern onshore mid-2010s+ fleet typical.
WIND_DEFAULT_M = 100.0


# ---------------------------------------------------------------------------
# Overpass fetch
# ---------------------------------------------------------------------------


def build_query(bbox: tuple[float, float, float, float]) -> str:
    lat_min, lon_min, lat_max, lon_max = bbox
    bb = f"{lat_min},{lon_min},{lat_max},{lon_max}"
    return (
        "[out:json][timeout:60];\n"
        "(\n"
        f'  node["man_made"~"^(tower|mast|communications_tower|chimney)$"]({bb});\n'
        f'  node["power"="generator"]["generator:source"="wind"]({bb});\n'
        f'  way["man_made"~"^(tower|mast|communications_tower|chimney)$"]({bb});\n'
        ");\n"
        "out body tags center;\n"
    )


def fetch_overpass(region_id: str, query: str) -> dict:
    cache = OSM_CACHE_DIR / f"{region_id}-obstacles.json"
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
                    "User-Agent": "open-plates/0.1 (build_basemap_obstacles)",
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
# Height parsing
# ---------------------------------------------------------------------------


_HEIGHT_RE = re.compile(
    r"^\s*([0-9]+(?:\.[0-9]+)?)\s*(m|meter|meters|metre|metres|ft|feet|')?\s*$",
    re.IGNORECASE,
)


def parse_height_to_ft(raw: str | None) -> float | None:
    """Parse a height string like '200', '200 m', '200m', '656 ft' -> feet.

    Returns None when the input cannot be parsed. Bare numbers are assumed
    to be meters (OSM convention per the `height=*` wiki).
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    m = _HEIGHT_RE.match(s)
    if not m:
        return None
    value = float(m.group(1))
    unit = (m.group(2) or "").lower()
    if unit in ("ft", "feet", "'"):
        return value
    # Default = meters (OSM `height=*` default unit).
    return value / METERS_PER_FOOT


def _structure_type(tags: dict) -> str:
    mm = (tags.get("man_made") or "").lower()
    if mm in ("tower", "communications_tower"):
        return "tower"
    if mm == "mast":
        return "mast"
    if mm == "chimney":
        return "chimney"
    power = (tags.get("power") or "").lower()
    if power == "generator" and (tags.get("generator:source") or "").lower() == "wind":
        return "wind"
    return "other"


def _default_label(structure_type: str, tags: dict) -> str:
    """A generic caption for features without an OSM name tag."""
    if structure_type == "wind":
        return "WIND TURBINE"
    if structure_type == "tower":
        t = (tags.get("tower:type") or "").lower()
        if t == "broadcasting":
            return "BROADCAST TOWER"
        if t == "communication":
            return "RADIO TOWER"
        if t == "observation":
            return "OBSERVATION TOWER"
        return "TOWER"
    if structure_type == "mast":
        return "MAST"
    if structure_type == "chimney":
        return "CHIMNEY"
    return "OBSTACLE"


def derive_height_ft(tags: dict, structure_type: str) -> float | None:
    """Pick a height in feet from tags, in priority order.

    1. height
    2. tower:height
    3. est_height
    4. For wind: rotor:diameter/2 + hub_height (if both present) -> tip
    5. For towers: tower:type defaults (broadcasting=150m, communication=100m,
       observation=80m)
    6. For wind: 100 m (modern onshore default)
    """
    for key in ("height", "tower:height", "est_height"):
        h = parse_height_to_ft(tags.get(key))
        if h is not None:
            return h

    if structure_type == "wind":
        hub = parse_height_to_ft(tags.get("hub_height")) or parse_height_to_ft(
            tags.get("height:hub")
        )
        rotor = parse_height_to_ft(tags.get("rotor:diameter")) or parse_height_to_ft(
            tags.get("rotor_diameter")
        )
        if hub is not None and rotor is not None:
            return hub + rotor / 2.0
        # Last-resort wind default.
        return WIND_DEFAULT_M / METERS_PER_FOOT

    tower_type = (tags.get("tower:type") or "").lower()
    if tower_type in TOWER_TYPE_DEFAULT_M:
        return TOWER_TYPE_DEFAULT_M[tower_type] / METERS_PER_FOOT

    return None


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def _element_lonlat(el: dict) -> tuple[float, float] | None:
    if el.get("type") == "node":
        lat = el.get("lat")
        lon = el.get("lon")
    else:
        center = el.get("center") or {}
        lat = center.get("lat")
        lon = center.get("lon")
    if lat is None or lon is None:
        return None
    return (float(lon), float(lat))


def _rank_for_height(ft: int) -> int:
    if ft >= 2000:
        return 1
    if ft >= 1500:
        return 2
    return 3


def extract_features(osm: dict, min_ft: int) -> tuple[list[dict], int]:
    """Return (features, raw_hit_count). Features are filtered + sorted."""
    elements = osm.get("elements") or []
    raw = 0
    features: list[dict] = []
    for el in elements:
        el_type = el.get("type")
        if el_type not in ("node", "way"):
            continue
        tags = el.get("tags") or {}
        stype = _structure_type(tags)
        if stype == "other":
            continue
        raw += 1
        lonlat = _element_lonlat(el)
        if lonlat is None:
            continue
        height_ft = derive_height_ft(tags, stype)
        if height_ft is None:
            continue
        if height_ft < min_ft:
            continue
        ft_int = int(round(height_ft))
        name_tag = tags.get("name")
        name = str(name_tag).strip() if name_tag else _default_label(stype, tags)
        fid = f"{el_type}-{el.get('id')}"
        features.append({
            "type": "Feature",
            "id": fid,
            "properties": {
                "class": "obstacle_tall",
                "name": name,
                "height_ft_agl": ft_int,
                "structure_type": stype,
                "rank": _rank_for_height(ft_int),
                "osm_id": el.get("id"),
                "osm_type": el_type,
            },
            "geometry": {
                "type": "Point",
                "coordinates": [round(lonlat[0], 6), round(lonlat[1], 6)],
            },
        })
    # Deterministic: rank asc, height desc, id asc.
    features.sort(key=lambda f: (
        f["properties"]["rank"],
        -f["properties"]["height_ft_agl"],
        f["id"],
    ))
    return features, raw


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build(
    region_id: str, bbox: tuple[float, float, float, float],
    min_ft: int = 1000,
) -> Path:
    lat_min, lon_min, lat_max, lon_max = bbox
    print(
        f"building obstacles for region={region_id} bbox={bbox} min_ft={min_ft}",
        flush=True,
    )
    query = build_query(bbox)
    try:
        osm = fetch_overpass(region_id, query)
    except Exception as exc:
        print(f"ERROR: Overpass fetch failed: {exc}", file=sys.stderr)
        sys.exit(1)

    elements = osm.get("elements") or []
    print(f"  OSM elements returned: {len(elements)}")

    features, raw_hit = extract_features(osm, min_ft=min_ft)
    print(f"  raw obstacle candidates: {raw_hit}")
    print(f"  features passing >= {min_ft} ft filter: {len(features)}")

    if features:
        description = (
            f"Auto-generated tall-obstacle points for region {region_id!r}. "
            f"Built by scripts/build_basemap_obstacles.py from OpenStreetMap "
            f"(Overpass API). Features tagged class=obstacle_tall; "
            f"height_ft_agl is the tip-of-structure height AGL. "
            f"Threshold: {min_ft} ft. Data (c) OpenStreetMap contributors, ODbL."
        )
    else:
        description = (
            f"Auto-generated tall-obstacle layer for region {region_id!r} at "
            f"threshold {min_ft} ft AGL — ZERO features passed the filter. "
            f"No obstacles >= {min_ft} ft AGL in this bbox — this is accurate "
            f"for Eifel / Luxembourg border; expect more in e.g. Munich / "
            f"Berlin regions. Re-run scripts/build_basemap_obstacles.py to "
            f"refresh. Data (c) OpenStreetMap contributors, ODbL."
        )

    fc = {
        "type": "FeatureCollection",
        "description": description,
        "region": region_id,
        "bbox": [lat_min, lon_min, lat_max, lon_max],
        "min_ft_agl": min_ft,
        "features": features,
    }

    BASEMAP_DIR.mkdir(parents=True, exist_ok=True)
    out_path = BASEMAP_DIR / f"{region_id}-obstacles.json"
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
    ap.add_argument(
        "--min-ft", type=int, default=1000,
        help="Minimum tip height AGL in feet to retain (default: 1000).",
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

    build(args.region_id, bbox, min_ft=args.min_ft)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
