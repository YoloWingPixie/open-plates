"""Geographic basemap underlay for plan views.

Loads a hand-authored GeoJSON `FeatureCollection` from
``data/basemap/<region>.json`` and renders it as an SVG fragment that
the plan-view renderer emits *behind* the procedure path.

Design notes
------------
* GeoJSON coordinate order is ``[longitude, latitude]`` per RFC 7946;
  the projector consumes ``(lat, lon)`` so every coordinate is swapped
  at the boundary.
* The basemap must recede visually. Colour tokens are strict tints/shades
  of the existing Swiss palette (COLOR_INK / COLOR_HAIRLINE / COLOR_MUTED /
  COLOR_ACCENT_CYAN / COLOR_PAPER).
* No new runtime dependencies: stdlib ``json`` + string templating.
* Deterministic output — features are sorted by id/name within each
  layer so repeated renders produce byte-identical SVG.

Sibling-file convention
-----------------------
``load_basemap("<region>")`` loads ``<region>.json`` and ALSO merges any
``<region>-populated.json``, ``<region>-obstacles.json``,
``<region>-roads.json``, and ``<region>-waterways.json`` sibling files
found alongside it, concatenating their ``features`` arrays. This keeps
auto-generated layers (OSM populated areas, OSM tall obstacles, OSM road
network, OSM waterways) in their own files for clarity while giving the
caller a single FeatureCollection.

Additionally, if ``<region>-hillshade.json`` exists, its manifest is
stashed under the key ``_hillshade`` on the returned dict — the PNG it
references is embedded as a base64 data-URI at render time. Hillshade has
no vector features; it's a raster underlay keyed by its bbox.

The module exports :func:`render_basemap` and the :class:`BasemapConfig`
dataclass. :func:`load_basemap` looks a region up under
``data/basemap/`` and returns ``None`` if the base file is missing.
"""

from __future__ import annotations

import base64
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Literal

# NB: We import colour/size tokens lazily inside functions that need them
# to avoid a circular dependency with ``render_svg`` (which imports this
# module). Token values are re-declared here only where they must be used
# in dataclass defaults — none of those apply today.

Projector = Callable[[tuple[float, float]], tuple[float, float]]

# ---------------------------------------------------------------------------
# Style tokens — strict tints/shades of the existing palette.
# These are duplicated in ``render_svg`` as well for direct use there.
# ---------------------------------------------------------------------------

COLOR_BASEMAP_SEA = "#A8C9D4"           # Swisstopo-style confident water blue
COLOR_BASEMAP_LAND = "#FFFFFF"          # COLOR_PAPER, no fill (cartographic default)
COLOR_BASEMAP_COAST = "#B8B3A6"         # COLOR_HAIRLINE
COLOR_BASEMAP_RIVER = "#6E9DAD"         # mid-tone ink-mixed cyan; rivers read at plate scale
COLOR_BASEMAP_CITY = "#0A0A0A"          # COLOR_INK
COLOR_BASEMAP_LABEL = "#6E6A61"         # COLOR_MUTED
# Populated-area fill — pale warm tan, quieter than any hairline tone so
# the procedure path stays dominant.
COLOR_BASEMAP_URBAN = "#E0D6C4"         # warm tan, visible over hillshade
# Road-casing fill for atlas-style motorway/trunk rendering. One step
# lighter than COLOR_BASEMAP_URBAN, still inside the Swiss surface
# palette. The only new palette token added for the roads layer; every
# other road colour reuses COLOR_INK / COLOR_MUTED / COLOR_HAIRLINE /
# COLOR_PAPER.
COLOR_BASEMAP_ROAD_FILL = "#E6E2D8"     # urban tan shifted one step lighter

SW_COAST = 0.5
SW_RIVER = 0.4
SW_STREAM = 0.3
SW_LAKE_OUTLINE = 0.4
SW_URBAN_OUTLINE = 0.4
CITY_DOT_PX = 2.5

# Atlas-style road tier visual spec. Each tier draws in two passes: a
# dark casing stroke underneath, a lighter fill stroke on top. Tiers
# with ``fill`` = None draw as a single casing stroke (no fill pass).
# Colours reference the token names; resolved at call time so unit
# tests can monkeypatch the tokens without re-importing the dict.
ROAD_CLASSES = (
    "road_motorway",
    "road_trunk",
    "road_primary",
    "road_secondary",
    "road_tertiary",
)


def _road_styles() -> dict[str, dict[str, object]]:
    """Return the tier-by-tier casing/fill spec.

    Named values are resolved here so the palette tokens stay in a single
    place and future palette tweaks flow through uniformly.
    """
    return {
        "road_motorway": {
            "casing_w": 2.4, "fill_w": 1.5,
            "casing": "#0A0A0A",              # COLOR_INK
            "fill": COLOR_BASEMAP_ROAD_FILL,
        },
        "road_trunk": {
            "casing_w": 2.0, "fill_w": 1.2,
            "casing": "#0A0A0A",              # COLOR_INK
            "fill": COLOR_BASEMAP_ROAD_FILL,
        },
        "road_primary": {
            "casing_w": 1.5, "fill_w": 0.9,
            "casing": "#6E6A61",              # COLOR_MUTED
            "fill": "#FFFFFF",                # COLOR_PAPER
        },
        "road_secondary": {
            "casing_w": 1.0, "fill_w": 0.5,
            "casing": "#B8B3A6",              # COLOR_HAIRLINE
            "fill": None,                     # single-stroke
        },
        "road_tertiary": {
            "casing_w": 0.5, "fill_w": 0.0,
            "casing": "#B8B3A6",              # COLOR_HAIRLINE
            "fill": None,                     # single hairline
        },
    }

# Spot-elevation triangle dimensions (screen px). Small upward-pointing
# triangle so a cluster of elevations reads as terrain callouts without
# competing with fix symbols (which are ~10 px circles/triangles).
_SPOT_TRI_H = 4.0
_SPOT_TRI_W = 5.0
# Caption offset east of the triangle apex.
_SPOT_LABEL_DX = 3.5
_SPOT_LABEL_DY = 1.0

# Hillshade layer appearance. Opacity of 0.35 with multiply blending lands
# at a subtle "tan beige terrain" tone over the paper — enough to hint at
# topography without drowning the procedure path. If mix-blend-mode causes
# cairosvg issues we rely on opacity alone (close enough).
_HILLSHADE_OPACITY = 0.18

# Matches render_svg.SIZE_MICRO; duplicated here so the module remains
# free of circular imports.
_SIZE_MICRO = 6.5
_TRACK = "0.08em"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


LabelDensity = Literal["sparse", "normal", "dense"]


@dataclass(frozen=True)
class BasemapConfig:
    """Knobs controlling which basemap layers render and how densely."""

    show_sea: bool = True
    show_land: bool = True
    show_coastline: bool = True
    show_roads: bool = True
    show_rivers: bool = True
    show_streams: bool = False
    show_lakes: bool = True
    show_cities: bool = True
    show_city_labels: bool = True
    show_populated_areas: bool = True
    show_obstacles: bool = True
    show_hillshade: bool = True
    show_spot_elevations: bool = True
    city_max_rank: int = 3
    # Default excludes tertiary (rank 4) to keep the plan readable; bump
    # to 4 via the YAML ``basemap:`` block for a denser atlas look.
    road_max_rank: int = 3
    # Waterway rank cap — default 2 excludes streams (rank 3) even if
    # ``show_streams`` is enabled. Bump to 3 to include streams.
    waterway_max_rank: int = 2
    obstacle_min_ft_agl: int = 1000
    label_density: LabelDensity = "normal"

    @classmethod
    def from_mapping(cls, data: dict | None) -> "BasemapConfig":
        """Build a config from a YAML/JSON mapping; unknown keys ignored."""
        if not data:
            return cls()
        known = {
            "show_sea", "show_land", "show_coastline",
            "show_roads",
            "show_rivers", "show_streams", "show_lakes",
            "show_cities", "show_city_labels",
            "show_populated_areas", "show_obstacles",
            "show_hillshade", "show_spot_elevations",
            "city_max_rank", "road_max_rank", "waterway_max_rank",
            "obstacle_min_ft_agl",
            "label_density",
        }
        kwargs = {k: v for k, v in data.items() if k in known}
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def basemap_path(region: str, data_root: Path | None = None) -> Path:
    """Resolve the filesystem path for a region id under ``data/basemap/``."""
    if data_root is None:
        # src/open_plates/basemap.py -> repo root is parents[2]
        data_root = Path(__file__).resolve().parents[2] / "data" / "basemap"
    # Region is a filename-safe slug; reject separators defensively.
    if "/" in region or "\\" in region or ".." in region:
        raise ValueError(f"invalid region id: {region!r}")
    return data_root / f"{region}.json"


def find_covering_region(
    plan_bbox: tuple[float, float, float, float],
    data_root: Path | None = None,
) -> str | None:
    """Find the smallest pre-built region that fully covers *plan_bbox*.

    Parameters
    ----------
    plan_bbox
        ``(lat_min, lon_min, lat_max, lon_max)`` — lat-first, matching
        :func:`plan_world_bbox` return format.
    data_root
        Directory containing ``regions.json``. Falls back to
        ``data/basemap/`` relative to the repo root (same convention as
        :func:`basemap_path`).

    Returns the region id string of the tightest-fitting region, or
    ``None`` when no region covers the bbox (or ``regions.json`` is
    missing / unparseable).
    """
    if data_root is None:
        data_root = Path(__file__).resolve().parents[2] / "data" / "basemap"
    regions_file = data_root / "regions.json"
    if not regions_file.is_file():
        return None
    try:
        with regions_file.open("r", encoding="utf-8") as fh:
            regions = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(regions, dict):
        return None

    lat_min, lon_min, lat_max, lon_max = plan_bbox
    best_id: str | None = None
    best_area: float = float("inf")

    for rid, meta in regions.items():
        bbox = meta.get("bbox")
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        try:
            # regions.json uses GeoJSON order: [lon_min, lat_min, lon_max, lat_max]
            r_lon_min, r_lat_min, r_lon_max, r_lat_max = (float(v) for v in bbox)
        except (TypeError, ValueError):
            continue
        # Full containment check.
        if (
            r_lat_min <= lat_min
            and r_lon_min <= lon_min
            and r_lat_max >= lat_max
            and r_lon_max >= lon_max
        ):
            area = (r_lat_max - r_lat_min) * (r_lon_max - r_lon_min)
            if area < best_area:
                best_area = area
                best_id = rid

    return best_id


def _sibling_path(base: Path, suffix: str) -> Path:
    return base.with_name(f"{base.stem}-{suffix}{base.suffix}")


def _sibling_bbox_geographic(
    doc: dict, suffix: str,
) -> tuple[float, float, float, float] | None:
    """Extract the sibling's declared geographic bbox, normalised to
    ``(lat_min, lon_min, lat_max, lon_max)``.

    Sibling files disagree on bbox ordering:
      - hillshade manifests carry GeoJSON ``[lon_min, lat_min, lon_max, lat_max]``.
      - populated / roads / waterways / obstacles carry the lat-first
        tuple the build scripts emit: ``[lat_min, lon_min, lat_max, lon_max]``.

    We infer the convention from which coordinate makes geographic sense
    — the bbox that fits ``[-90, 90]`` for the "lat" slot wins. If
    neither parse passes, return ``None`` so the caller skips the check.
    """
    raw = doc.get("bbox")
    if not (isinstance(raw, list) and len(raw) == 4):
        return None
    try:
        vals = tuple(float(v) for v in raw)
    except (TypeError, ValueError):
        return None
    # Hillshade -> GeoJSON order.
    if suffix == "hillshade":
        lon_min, lat_min, lon_max, lat_max = vals
    else:
        lat_min, lon_min, lat_max, lon_max = vals
    if (
        -90.0 <= lat_min <= 90.0 and -90.0 <= lat_max <= 90.0
        and -180.0 <= lon_min <= 180.0 and -180.0 <= lon_max <= 180.0
        and lat_max > lat_min and lon_max > lon_min
    ):
        return (lat_min, lon_min, lat_max, lon_max)
    return None


def _plan_bbox_escapes(
    plan_bbox: tuple[float, float, float, float],
    region_bbox: tuple[float, float, float, float],
    tol_deg: float = 1e-6,
) -> bool:
    """Return True if ``plan_bbox`` extends outside ``region_bbox``.

    Both tuples are ``(lat_min, lon_min, lat_max, lon_max)``. The tiny
    ``tol_deg`` absorbs floating-point round-trips through the projector
    so a plan view that exactly matches the region edge doesn't trip the
    warning.
    """
    lat_lo_p, lon_lo_p, lat_hi_p, lon_hi_p = plan_bbox
    lat_lo_r, lon_lo_r, lat_hi_r, lon_hi_r = region_bbox
    return (
        lat_lo_p < lat_lo_r - tol_deg
        or lon_lo_p < lon_lo_r - tol_deg
        or lat_hi_p > lat_hi_r + tol_deg
        or lon_hi_p > lon_hi_r + tol_deg
    )


def load_basemap(
    region: str,
    data_root: Path | None = None,
    plan_bbox: tuple[float, float, float, float] | None = None,
) -> dict | None:
    """Load a basemap GeoJSON for ``region``; return ``None`` if missing.

    Also merges any sibling ``<region>-populated.json`` and
    ``<region>-obstacles.json`` files found alongside the base file. Each
    sibling's ``features`` array is concatenated onto the base collection's.
    If no base file exists, returns ``None`` even if siblings are present
    (the base file declares the region; siblings are additive layers).

    If a ``<region>-hillshade.json`` sidecar exists, its manifest is
    stashed under the key ``_hillshade`` on the returned dict. The
    manifest carries a bbox + image filename used by the renderer at
    draw time; the PNG bytes themselves are read lazily during render.

    ``plan_bbox`` is the renderer's effective geographic extent for the
    plan view, ordered ``(lat_min, lon_min, lat_max, lon_max)``. When
    provided, every loaded sibling's declared bbox is compared against
    it; if the plan bbox escapes any sibling we ``warnings.warn`` that
    the basemap region is undersized for this procedure. Rendering
    proceeds unchanged — the warning is a regression detector, not a
    hard failure.
    """
    p = basemap_path(region, data_root)
    if not p.is_file():
        return None
    with p.open("r", encoding="utf-8") as fh:
        base = json.load(fh)
    base_features = list(base.get("features") or [])
    merged_sources: list[str] = []
    sibling_bboxes: list[tuple[str, tuple[float, float, float, float]]] = []
    for suffix in ("populated", "obstacles", "roads", "waterways"):
        sib = _sibling_path(p, suffix)
        if not sib.is_file():
            continue
        with sib.open("r", encoding="utf-8") as fh:
            sib_doc = json.load(fh)
        sib_features = sib_doc.get("features") or []
        base_features.extend(sib_features)
        merged_sources.append(suffix)
        sb = _sibling_bbox_geographic(sib_doc, suffix)
        if sb is not None:
            sibling_bboxes.append((suffix, sb))
    if merged_sources:
        base = {**base, "features": base_features, "_merged": merged_sources}

    # Hillshade sidecar: attach the manifest + its on-disk directory so
    # the renderer can resolve the PNG path. Does NOT read the PNG here;
    # bytes are read lazily in ``_render_hillshade``.
    hs_sib = _sibling_path(p, "hillshade")
    if hs_sib.is_file():
        with hs_sib.open("r", encoding="utf-8") as fh:
            hs_meta = json.load(fh)
        hs_meta = {**hs_meta, "_dir": str(p.parent)}
        base = {**base, "_hillshade": hs_meta}
        sb = _sibling_bbox_geographic(hs_meta, "hillshade")
        if sb is not None:
            sibling_bboxes.append(("hillshade", sb))

    # Regression detector: if the renderer's plan view extends outside any
    # basemap sibling's declared bbox, warn. A generous region bbox (see
    # ``data/basemap/regions.json``) is supposed to bracket every plausible
    # plan view; this warning fires only when a new procedure wants more
    # extent than the region registry promises.
    if plan_bbox is not None and sibling_bboxes:
        escaped: list[str] = [
            suffix for suffix, bb in sibling_bboxes
            if _plan_bbox_escapes(plan_bbox, bb)
        ]
        if escaped:
            warnings.warn(
                (
                    f"basemap region {region!r} is undersized for plan "
                    f"bbox {plan_bbox}: escapes siblings "
                    f"{sorted(set(escaped))}. Rebuild the region with a "
                    f"larger bbox in data/basemap/regions.json."
                ),
                UserWarning,
                stacklevel=2,
            )

    return base


# ---------------------------------------------------------------------------
# SVG emission helpers (self-contained, no render_svg imports)
# ---------------------------------------------------------------------------


def _fmt(n: float, places: int = 2) -> str:
    s = f"{n:.{places}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s or "0"


def _esc(s: object) -> str:
    t = "" if s is None else str(s)
    return (
        t.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# ---------------------------------------------------------------------------
# Placement integration — basemap labels route through the shared engine
# when a caller supplies a placement context.
# ---------------------------------------------------------------------------


def _submit_basemap_label(
    ctx,
    *,
    feature_id: str,
    anchor: tuple[float, float],
    text: str,
    size_px: float,
    family_mono: bool,
    fill: str,
    preferred: str = "NE",
    offset_px: float = 4.0,
    halo: float = 2.5,
) -> None:
    """Queue a basemap label through the plan-view placement context.

    The ctx is a ``open_plates.render_svg._PlanLabelCtx``; we duck-type
    so this module stays import-free of ``render_svg``. Labels default to
    ``suppress`` fallback — basemap labels are meant to recede, so losing
    one to a crowded plan view is always preferable to scribbling leaders.
    """
    # Local import to avoid circular dependency at module load.
    from .placement import (
        LabelCandidate, PlacementRequest, PlacementTier,
        eight_position_candidates,
    )
    w = len(text) * size_px * 0.58
    h = size_px * 1.25
    candidates = eight_position_candidates(
        preferred=preferred, offset_px=offset_px,
    )

    def _render(placed):
        bx = placed.bbox.x
        by = placed.bbox.y + size_px * 0.85
        family_attr = (
            f'font-family:{_esc("monospace")};' if family_mono else ""
        )
        common = (
            f'x="{_fmt(bx)}" y="{_fmt(by)}" text-anchor="start" '
            f'style="font-size:{_fmt(size_px)}px;font-weight:bold;'
            f'{family_attr}letter-spacing:{_TRACK}"'
        )
        return (
            f'<text {common} fill="none" stroke="#FFFFFF" '
            f'stroke-width="{_fmt(halo)}" stroke-linejoin="round">'
            f'{_esc(text)}</text>\n'
            f'<text {common} fill="{fill}">{_esc(text)}</text>\n'
        )

    ctx.requests.append(
        PlacementRequest(
            tier=PlacementTier.LABEL if family_mono else PlacementTier.CULTURE,
            anchor=anchor,
            content_size=(w, h),
            candidates=candidates,
            fallback="suppress",
            feature_id=feature_id,
            required_clearance_px=1.0,
        )
    )
    ctx.render_fns.append(_render)


def collect_basemap_label_requests(*args, **kwargs):
    """Placeholder export so ``render_svg`` can import the symbol.

    Basemap label requests are submitted directly via
    :func:`_submit_basemap_label` inside the existing rendering
    functions when ``render_basemap`` is called with a
    ``placement_ctx``. This function exists so external callers can
    discover the integration surface; it returns an empty list.
    """
    return []


def _ring_to_path(ring: list[list[float]], projector: Projector) -> str:
    """Convert a GeoJSON linear ring ([lon, lat] pairs) to an SVG subpath."""
    if not ring:
        return ""
    parts: list[str] = []
    for i, pt in enumerate(ring):
        lon, lat = float(pt[0]), float(pt[1])
        x, y = projector((lat, lon))
        cmd = "M" if i == 0 else "L"
        parts.append(f"{cmd}{_fmt(x)},{_fmt(y)}")
    parts.append("Z")
    return "".join(parts)


def _polygon_path_d(coords: list, projector: Projector) -> str:
    """Render a Polygon's coordinates (list of rings) to a path ``d`` string."""
    return "".join(_ring_to_path(ring, projector) for ring in coords)


def _multipolygon_path_d(coords: list, projector: Projector) -> str:
    """Render a MultiPolygon's coordinates to a path ``d`` string."""
    return "".join(_polygon_path_d(poly, projector) for poly in coords)


def _chaikin(points: list[tuple[float, float]], iterations: int) -> list[tuple[float, float]]:
    """Chaikin corner-cutting. Each iteration replaces every vertex with two
    points at 1/4 and 3/4 of each incident segment, rounding the corner.
    Endpoints are preserved. N iterations ≈ 2^N × points (we cap at 4 iter).
    """
    if iterations <= 0 or len(points) < 3:
        return points
    for _ in range(min(iterations, 4)):
        out: list[tuple[float, float]] = [points[0]]
        for i in range(len(points) - 1):
            ax, ay = points[i]
            bx, by = points[i + 1]
            out.append((0.75 * ax + 0.25 * bx, 0.75 * ay + 0.25 * by))
            out.append((0.25 * ax + 0.75 * bx, 0.25 * ay + 0.75 * by))
        out.append(points[-1])
        points = out
    return points


def _linestring_path_d(coords: list, projector: Projector, smooth: int = 0) -> str:
    if not coords:
        return ""
    pts: list[tuple[float, float]] = []
    for pt in coords:
        lon, lat = float(pt[0]), float(pt[1])
        pts.append(projector((lat, lon)))
    if smooth > 0:
        pts = _chaikin(pts, smooth)
    parts: list[str] = []
    for i, (x, y) in enumerate(pts):
        cmd = "M" if i == 0 else "L"
        parts.append(f"{cmd}{_fmt(x)},{_fmt(y)}")
    return "".join(parts)


def _multilinestring_path_d(coords: list, projector: Projector, smooth: int = 0) -> str:
    return "".join(_linestring_path_d(ls, projector, smooth) for ls in coords)


# ---------------------------------------------------------------------------
# Feature iteration
# ---------------------------------------------------------------------------


def _iter_features(geojson: dict) -> Iterable[dict]:
    feats = geojson.get("features") or []
    for f in feats:
        if isinstance(f, dict) and f.get("type") == "Feature":
            yield f


def _feature_class(feature: dict) -> str:
    return str((feature.get("properties") or {}).get("class") or "")


def _feature_rank(feature: dict, default: int = 3) -> int:
    r = (feature.get("properties") or {}).get("rank")
    try:
        return int(r)
    except (TypeError, ValueError):
        return default


def _feature_name(feature: dict) -> str:
    return str((feature.get("properties") or {}).get("name") or "")


def _feature_id(feature: dict) -> str:
    return str(feature.get("id") or "")


def _polygon_coords(feature: dict) -> list[list[list[float]]]:
    """Return outer-ring coordinates list for Polygon or MultiPolygon."""
    geom = feature.get("geometry") or {}
    gt = geom.get("type")
    coords = geom.get("coordinates") or []
    if gt == "Polygon":
        return coords
    if gt == "MultiPolygon":
        # Flatten: return the biggest outer ring for labeling purposes.
        return coords[0] if coords else []
    return []


def _polygon_centroid(outer: list[list[float]]) -> tuple[float, float] | None:
    """Unsigned centroid of a polygon outer ring ([lon, lat] pairs, closed)."""
    if not outer or len(outer) < 4:
        return None
    xs = [float(p[0]) for p in outer[:-1]]
    ys = [float(p[1]) for p in outer[:-1]]
    n = len(xs)
    if n == 0:
        return None
    return (sum(xs) / n, sum(ys) / n)


# ---------------------------------------------------------------------------
# Layer rendering
# ---------------------------------------------------------------------------


def _render_polygonal_fill(
    features: list[dict], projector: Projector, fill: str, stroke: str = "none",
    sw: float = 0.0,
) -> str:
    """Emit filled Polygon/MultiPolygon features as a single <path>."""
    d_parts: list[str] = []
    for f in features:
        geom = f.get("geometry") or {}
        gtype = geom.get("type")
        coords = geom.get("coordinates") or []
        if gtype == "Polygon":
            d_parts.append(_polygon_path_d(coords, projector))
        elif gtype == "MultiPolygon":
            d_parts.append(_multipolygon_path_d(coords, projector))
    d = "".join(d_parts)
    if not d:
        return ""
    sw_attr = f' stroke-width="{_fmt(sw)}"' if stroke != "none" else ""
    return (
        f'<path d="{d}" fill="{fill}" fill-rule="evenodd" '
        f'stroke="{stroke}"{sw_attr}/>\n'
    )


def _render_linear(
    features: list[dict], projector: Projector, stroke: str, sw: float,
    dash: str | None = None, smooth: int = 0,
) -> str:
    d_parts: list[str] = []
    for f in features:
        geom = f.get("geometry") or {}
        gtype = geom.get("type")
        coords = geom.get("coordinates") or []
        if gtype == "LineString":
            d_parts.append(_linestring_path_d(coords, projector, smooth))
        elif gtype == "MultiLineString":
            d_parts.append(_multilinestring_path_d(coords, projector, smooth))
        elif gtype == "Polygon":
            if coords:
                d_parts.append(_ring_to_path(coords[0], projector))
    d = "".join(d_parts)
    if not d:
        return ""
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<path d="{d}" fill="none" stroke="{stroke}" '
        f'stroke-width="{_fmt(sw)}" stroke-linejoin="round" '
        f'stroke-linecap="round"{dash_attr}/>\n'
    )


def _render_populated_areas(
    features: list[dict], projector: Projector, show_labels: bool,
    label_size: float,
) -> str:
    """Filled populated-area polygons with paper-halo centroid labels.

    Kept for back-compat; ``render_basemap`` now calls
    :func:`_render_populated_area_fills` and
    :func:`_render_populated_area_labels` separately so the fills and
    labels can live on opposite sides of the procedure-bbox mask.
    """
    fills = _render_populated_area_fills(features, projector)
    if not show_labels:
        return fills
    return fills + _render_populated_area_labels(features, projector, label_size)


def _bbox_overlaps(proj_pts: list[tuple[float, float]], rect: tuple[float, float, float, float]) -> bool:
    if not proj_pts:
        return False
    xs = [p[0] for p in proj_pts]
    ys = [p[1] for p in proj_pts]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    rx0, ry0, rx1, ry1 = rect
    return not (x1 < rx0 or x0 > rx1 or y1 < ry0 or y0 > ry1)


def _populated_in_view(
    features: list[dict], projector: Projector,
    plan_rect: tuple[float, float, float, float] | None, max_visible: int = 5,
) -> list[dict]:
    """Keep only populated-area polygons whose projected bbox overlaps the
    plan rect, then rank by (node rank asc, area desc) and cap at
    ``max_visible``. At plate scale we want a handful of relevant
    settlements, not every rural Gemeinde that happens to live in the
    same 1°×1° OSM query window.
    """
    if plan_rect is None:
        return features
    kept: list[tuple[int, float, dict]] = []
    for f in features:
        coords = _polygon_coords(f)
        if not coords:
            continue
        ring = coords[0] if coords and isinstance(coords[0][0], list) else coords
        proj = [projector((float(pt[1]), float(pt[0]))) for pt in ring]
        if not _bbox_overlaps(proj, plan_rect):
            continue
        # Approximate area in projected px space for ranking.
        xs = [p[0] for p in proj]
        ys = [p[1] for p in proj]
        area_px = (max(xs) - min(xs)) * (max(ys) - min(ys))
        kept.append((_feature_rank(f, 9), area_px, f))
    kept.sort(key=lambda t: (t[0], -t[1]))
    return [t[2] for t in kept[:max_visible]]


def _render_populated_area_fills(
    features: list[dict], projector: Projector,
    plan_rect: tuple[float, float, float, float] | None = None,
) -> str:
    """Polygon fills for populated areas (safe to render under procedure)."""
    if not features:
        return ""
    visible = _populated_in_view(features, projector, plan_rect)
    if not visible:
        return ""
    feats = sorted(
        visible,
        key=lambda f: (
            _feature_rank(f, 9), _feature_name(f).upper(), _feature_id(f),
        ),
    )
    return _render_polygonal_fill(
        feats, projector,
        fill=COLOR_BASEMAP_URBAN, stroke=COLOR_BASEMAP_COAST,
        sw=SW_URBAN_OUTLINE,
    )


def _render_populated_area_labels(
    features: list[dict], projector: Projector, label_size: float,
    plan_rect: tuple[float, float, float, float] | None = None,
    placement_ctx: object | None = None,
) -> str:
    """Centroid city-name labels (masked when they overlap procedure)."""
    if not features:
        return ""
    visible = _populated_in_view(features, projector, plan_rect)
    if not visible:
        return ""
    feats = sorted(
        visible,
        key=lambda f: (
            _feature_rank(f, 9), _feature_name(f).upper(), _feature_id(f),
        ),
    )
    out = ""
    for f in feats:
        name = _feature_name(f).upper()
        if not name:
            continue
        outer = _polygon_coords(f)
        if not outer:
            continue
        ring = outer[0] if outer and isinstance(outer[0][0], list) else outer
        c = _polygon_centroid(ring)
        if c is None:
            continue
        lon, lat = c
        x, y = projector((lat, lon))
        if placement_ctx is None:
            common = (
                f'x="{_fmt(x)}" y="{_fmt(y)}" text-anchor="middle" '
                f'style="font-size:{_fmt(label_size)}px;font-weight:bold;'
                f'letter-spacing:{_TRACK}"'
            )
            out += (
                f'<text {common} fill="none" stroke="#FFFFFF" '
                f'stroke-width="2.5" stroke-linejoin="round">{_esc(name)}</text>\n'
            )
            out += f'<text {common} fill="{COLOR_BASEMAP_LABEL}">{_esc(name)}</text>\n'
        else:
            _submit_basemap_label(
                placement_ctx,
                feature_id=f"pop:{_feature_id(f) or name}",
                anchor=(x, y),
                text=name,
                size_px=label_size,
                family_mono=False,
                fill=COLOR_BASEMAP_LABEL,
                preferred="N",
                offset_px=2.0,
            )
    return out


def _render_hillshade(
    hillshade_meta: dict, projector: Projector,
) -> str:
    """Emit a single <image> element with the hillshade PNG as a data URI.

    The PNG is placed so its bbox corners project to the correct screen-
    space rect. ``preserveAspectRatio="none"`` lets the image stretch to
    whatever aspect the projector produces — the projection is already
    near-conformal at chart scale, so the image distortion is negligible
    and the feature/raster alignment stays correct.

    Uses opacity 0.35 + mix-blend-mode=multiply to read as "tan beige
    terrain" over the paper background. If mix-blend-mode causes cairosvg
    to choke, the opacity alone still lands close to the intended tone.
    """
    bbox = hillshade_meta.get("bbox")
    if not bbox or len(bbox) != 4:
        return ""
    image_name = hillshade_meta.get("image")
    if not image_name:
        return ""
    # Resolve PNG path against the directory we loaded the manifest from.
    # Falls back to the default basemap dir if _dir is missing (safety net
    # for tests that synthesise a manifest by hand).
    dir_str = hillshade_meta.get("_dir")
    if dir_str:
        png_path = Path(dir_str) / image_name
    else:
        png_path = Path(__file__).resolve().parents[2] / "data" / "basemap" / image_name
    if not png_path.is_file():
        return ""
    lon_min, lat_min, lon_max, lat_max = (float(v) for v in bbox)
    # Project NW and SE corners; the raster rect spans them.
    x0, y0 = projector((lat_max, lon_min))
    x1, y1 = projector((lat_min, lon_max))
    # Normalise to (top-left, size) — projections might invert one axis.
    rx = min(x0, x1)
    ry = min(y0, y1)
    rw = abs(x1 - x0)
    rh = abs(y1 - y0)
    if rw <= 0 or rh <= 0:
        return ""
    png_bytes = png_path.read_bytes()
    b64 = base64.b64encode(png_bytes).decode("ascii")
    href = f"data:image/png;base64,{b64}"
    return (
        f'<image x="{_fmt(rx)}" y="{_fmt(ry)}" '
        f'width="{_fmt(rw)}" height="{_fmt(rh)}" '
        f'href="{href}" preserveAspectRatio="none" '
        f'opacity="{_fmt(_HILLSHADE_OPACITY)}" '
        f'style="image-rendering:smooth;mix-blend-mode:multiply"/>\n'
    )


def _render_spot_elevations(
    features: list[dict], projector: Projector, label_size: float,
    placement_ctx: object | None = None,
) -> str:
    """Render spot_elevation Point features as triangle + elevation caption.

    Each feature must be a Point whose ``name`` is the elevation in feet
    (e.g. ``"2221"``). The triangle is a small upward-pointing filled
    arrow; the caption sits east of the apex in mono-bold with a paper
    halo so it stays readable when it sits over any other basemap layer.

    When ``placement_ctx`` is a ``_PlanLabelCtx`` (duck-typed), the
    triangle still draws immediately but the caption routes through the
    placement engine so it avoids fix IDs, course labels, and any other
    plan-view label.
    """
    if not features:
        return ""
    # Deterministic ordering: by rank then name then id.
    feats = sorted(
        features,
        key=lambda f: (
            _feature_rank(f, 9), _feature_name(f), _feature_id(f),
        ),
    )
    tri_d: list[str] = []
    labels_svg = ""
    half_w = _SPOT_TRI_W / 2.0
    for f in feats:
        geom = f.get("geometry") or {}
        if geom.get("type") != "Point":
            continue
        coords = geom.get("coordinates") or []
        if len(coords) < 2:
            continue
        name = _feature_name(f).strip()
        if not name:
            continue
        lon, lat = float(coords[0]), float(coords[1])
        x, y = projector((lat, lon))
        # Triangle: apex at (x, y - tri_h), base centred on (x, y).
        apex_y = y - _SPOT_TRI_H
        tri_d.append(
            f"M{_fmt(x - half_w)},{_fmt(y)}"
            f"L{_fmt(x + half_w)},{_fmt(y)}"
            f"L{_fmt(x)},{_fmt(apex_y)}Z"
        )
        caption = f"{name}'"

        if placement_ctx is None:
            lx = x + _SPOT_LABEL_DX
            ly = y - _SPOT_LABEL_DY
            common = (
                f'x="{_fmt(lx)}" y="{_fmt(ly)}" text-anchor="start" '
                f'style="font-size:{_fmt(label_size)}px;font-weight:bold;'
                f'font-family:{_esc("monospace")};letter-spacing:{_TRACK}"'
            )
            labels_svg += (
                f'<text {common} fill="none" stroke="#FFFFFF" '
                f'stroke-width="2.5" stroke-linejoin="round">'
                f'{_esc(caption)}</text>\n'
            )
            labels_svg += (
                f'<text {common} fill="{COLOR_BASEMAP_CITY}">'
                f'{_esc(caption)}</text>\n'
            )
        else:
            _submit_basemap_label(
                placement_ctx,
                feature_id=f"spot:{_feature_id(f) or name}:{_fmt(x)}",
                anchor=(x, apex_y),
                text=caption,
                size_px=label_size,
                family_mono=True,
                fill=COLOR_BASEMAP_CITY,
                preferred="E",
                offset_px=4.0,
            )

    if not tri_d:
        return ""
    out = (
        f'<path d="{"".join(tri_d)}" fill="{COLOR_BASEMAP_CITY}" '
        f'stroke="none"/>\n'
    )
    out += labels_svg
    return out


def _polyline_midpoint(
    coords: list, projector: Projector,
) -> tuple[float, float] | None:
    """Return the screen-space midpoint of a projected polyline.

    Walks the projected polyline picking the vertex closest to the half-way
    arc-length position — cheap, deterministic, and always lands on a real
    vertex (keeps route-shield labels off numerical tangent seams).
    """
    pts: list[tuple[float, float]] = []
    for pt in coords:
        if not pt or len(pt) < 2:
            continue
        lon, lat = float(pt[0]), float(pt[1])
        pts.append(projector((lat, lon)))
    if not pts:
        return None
    if len(pts) == 1:
        return pts[0]
    # Cumulative screen-space length.
    import math
    lengths = [0.0]
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i - 1][0]
        dy = pts[i][1] - pts[i - 1][1]
        lengths.append(lengths[-1] + math.hypot(dx, dy))
    total = lengths[-1]
    if total <= 0:
        return pts[len(pts) // 2]
    half = total / 2.0
    # Pick the vertex whose cumulative length is closest to half.
    best_i = 0
    best_d = abs(lengths[0] - half)
    for i, L in enumerate(lengths):
        d = abs(L - half)
        if d < best_d:
            best_d = d
            best_i = i
    return pts[best_i]


def _render_roads(
    features_by_class: dict[str, list[dict]],
    projector: Projector,
    max_rank: int,
    label_size: float,
) -> str:
    """Atlas-style road rendering — casing pass + fill pass per tier.

    Draws tiers from LOWEST visual prominence (tertiary) UP to highest
    (motorway) so major roads read on top at every intersection. Within a
    tier the casing stroke is emitted for ALL features as a single
    ``<path>``, then the lighter fill stroke is emitted as a second
    ``<path>`` on top — the classic cartographic double-stroke.

    Route-shield labels (``ref`` / ``name``) are emitted for rank<=2
    features only (motorway / trunk / primary) using the mono
    ``SIZE_MICRO`` face with a paper halo.
    """
    styles = _road_styles()
    # Draw order: tertiary, secondary, primary, trunk, motorway — so
    # higher-tier casings paint on top of lower-tier casings and the
    # motorway's white fill "cuts" through every crossing road.
    draw_order = list(reversed(ROAD_CLASSES))
    out = ""
    label_feats: list[dict] = []
    for cls in draw_order:
        rank_threshold = styles[cls]
        feats_all = features_by_class.get(cls, [])
        if not feats_all:
            continue
        feats = [f for f in feats_all if _feature_rank(f, 9) <= max_rank]
        if not feats:
            continue
        # Deterministic: sort by name/id.
        feats_sorted = sorted(
            feats,
            key=lambda f: (_feature_rank(f, 9), _feature_name(f), _feature_id(f)),
        )
        # Build the combined path ``d`` once — casing and fill reuse it.
        d_parts: list[str] = []
        for f in feats_sorted:
            geom = f.get("geometry") or {}
            gt = geom.get("type")
            coords = geom.get("coordinates") or []
            if gt == "LineString":
                d_parts.append(_linestring_path_d(coords, projector))
            elif gt == "MultiLineString":
                d_parts.append(_multilinestring_path_d(coords, projector))
        d = "".join(d_parts)
        if not d:
            continue
        casing = str(rank_threshold["casing"])
        casing_w = float(rank_threshold["casing_w"])  # type: ignore[arg-type]
        fill = rank_threshold["fill"]
        fill_w = float(rank_threshold["fill_w"])  # type: ignore[arg-type]
        out += (
            f'<path d="{d}" fill="none" stroke="{casing}" '
            f'stroke-width="{_fmt(casing_w)}" '
            f'stroke-linejoin="round" stroke-linecap="round"/>\n'
        )
        if fill is not None and fill_w > 0:
            out += (
                f'<path d="{d}" fill="none" stroke="{fill}" '
                f'stroke-width="{_fmt(fill_w)}" '
                f'stroke-linejoin="round" stroke-linecap="round"/>\n'
            )
    # Route-shield labels intentionally omitted — the lines carry the
    # atlas-style hierarchy on their own, and OSM's per-way fragmentation
    # made the shields visually noisy.
    return out


def _render_cities(
    features: list[dict], projector: Projector, show_labels: bool,
    label_size: float, placement_ctx: object | None = None,
) -> str:
    """Emit city dots (square) and optional UPPERCASE tracked labels."""
    out = ""
    sz = CITY_DOT_PX
    half = sz / 2.0
    # Dots first (they're small markers); labels on top with a paper halo
    # so any overlap with other linework / MSA bezel stays readable.
    for f in features:
        geom = f.get("geometry") or {}
        if geom.get("type") != "Point":
            continue
        coords = geom.get("coordinates") or []
        if len(coords) < 2:
            continue
        lon, lat = float(coords[0]), float(coords[1])
        x, y = projector((lat, lon))
        out += (
            f'<rect x="{_fmt(x - half)}" y="{_fmt(y - half)}" '
            f'width="{_fmt(sz)}" height="{_fmt(sz)}" '
            f'fill="{COLOR_BASEMAP_CITY}"/>\n'
        )
    if show_labels:
        for f in features:
            geom = f.get("geometry") or {}
            if geom.get("type") != "Point":
                continue
            coords = geom.get("coordinates") or []
            if len(coords) < 2:
                continue
            lon, lat = float(coords[0]), float(coords[1])
            x, y = projector((lat, lon))
            name = _feature_name(f).upper()
            if not name:
                continue
            if placement_ctx is None:
                lx = x + 4.0
                ly = y - 3.0
                common = (
                    f'x="{_fmt(lx)}" y="{_fmt(ly)}" text-anchor="start" '
                    f'style="font-size:{_fmt(label_size)}px;font-weight:bold;'
                    f'letter-spacing:{_TRACK}"'
                )
                out += (
                    f'<text {common} fill="none" stroke="#FFFFFF" '
                    f'stroke-width="2.25" stroke-linejoin="round">'
                    f'{_esc(name)}</text>\n'
                )
                out += (
                    f'<text {common} fill="{COLOR_BASEMAP_LABEL}">'
                    f'{_esc(name)}</text>\n'
                )
            else:
                _submit_basemap_label(
                    placement_ctx,
                    feature_id=f"city:{_feature_id(f) or name}",
                    anchor=(x, y),
                    text=name,
                    size_px=label_size,
                    family_mono=False,
                    fill=COLOR_BASEMAP_LABEL,
                    preferred="NE",
                    offset_px=4.0,
                )
    return out


_OBSTACLE_TYPE_LABEL = {
    "tower": "TWR",
    "mast": "MAST",
    "chimney": "CHI",
    "wind": "WIND",
}

# Symbol dimensions (screen px). Chart-scale "tower silhouette": a thin
# vertical line capped by a small upward-pointing triangle.
_OBS_STEM_H = 8.0
_OBS_STEM_W = 0.75
_OBS_TRI_H = 3.0
_OBS_TRI_W = 4.0
# Caption offset from the triangle apex (NE placement).
_OBS_LABEL_DX = 3.5
_OBS_LABEL_DY = 0.5


def _feature_int_prop(feature: dict, key: str) -> int | None:
    v = (feature.get("properties") or {}).get(key)
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _render_obstacles(
    features: list[dict], projector: Projector,
    min_ft_agl: int, micro_size: float,
    placement_ctx: object | None = None,
) -> str:
    """Emit tall-obstacle symbols: vertical stem + triangle cap + caption.

    Filters to ``height_ft_agl >= min_ft_agl`` so the caller can tune the
    threshold lower than the build-time filter if desired.
    """
    if not features:
        return ""
    # Deterministic ordering: same key as the build script so render-time
    # iteration order matches file order.
    feats = sorted(
        features,
        key=lambda f: (
            _feature_rank(f, 9),
            -(_feature_int_prop(f, "height_ft_agl") or 0),
            _feature_id(f),
        ),
    )
    out = ""
    # Accumulate symbol path data (stems + triangles) so all obstacles
    # share a single filled/stroked <path>, keeping SVG compact.
    stem_d: list[str] = []
    tri_d: list[str] = []
    labels_svg = ""
    label_size = micro_size
    sub_size = max(micro_size - 1.0, 5.0)
    for f in feats:
        height = _feature_int_prop(f, "height_ft_agl")
        if height is None or height < min_ft_agl:
            continue
        geom = f.get("geometry") or {}
        if geom.get("type") != "Point":
            continue
        coords = geom.get("coordinates") or []
        if len(coords) < 2:
            continue
        lon, lat = float(coords[0]), float(coords[1])
        x, y = projector((lat, lon))
        # Stem: thin filled rectangle from the base point up by _OBS_STEM_H.
        # Rendered via <path> with stroke rather than <rect> to keep the
        # element list flat.
        stem_top_y = y - _OBS_STEM_H
        stem_d.append(
            f"M{_fmt(x)},{_fmt(y)}L{_fmt(x)},{_fmt(stem_top_y)}"
        )
        # Triangle: apex above stem top, base at stem top width = _OBS_TRI_W.
        apex_y = stem_top_y - _OBS_TRI_H
        half_w = _OBS_TRI_W / 2.0
        tri_d.append(
            f"M{_fmt(x - half_w)},{_fmt(stem_top_y)}"
            f"L{_fmt(x + half_w)},{_fmt(stem_top_y)}"
            f"L{_fmt(x)},{_fmt(apex_y)}Z"
        )
        # Caption: height in feet with apostrophe, NE of the apex, paper halo.
        caption = f"{height}'"
        stype = str((f.get("properties") or {}).get("structure_type") or "")
        sub = _OBSTACLE_TYPE_LABEL.get(stype)
        if placement_ctx is None:
            lx = x + _OBS_LABEL_DX
            ly = apex_y + _OBS_LABEL_DY
            common = (
                f'x="{_fmt(lx)}" y="{_fmt(ly)}" text-anchor="start" '
                f'style="font-size:{_fmt(label_size)}px;font-weight:bold;'
                f'font-family:{_esc("monospace")};letter-spacing:{_TRACK}"'
            )
            labels_svg += (
                f'<text {common} fill="none" stroke="#FFFFFF" '
                f'stroke-width="2.5" stroke-linejoin="round">'
                f'{_esc(caption)}</text>\n'
            )
            labels_svg += (
                f'<text {common} fill="{COLOR_BASEMAP_CITY}">'
                f'{_esc(caption)}</text>\n'
            )
            # Optional sub-label: structure type (WIND/TWR/MAST/CHI). Skip
            # for unknown types so we never print literal "OTHER".
            if sub:
                sy = ly + sub_size + 0.5
                sub_common = (
                    f'x="{_fmt(lx)}" y="{_fmt(sy)}" text-anchor="start" '
                    f'style="font-size:{_fmt(sub_size)}px;'
                    f'letter-spacing:{_TRACK}"'
                )
                labels_svg += (
                    f'<text {sub_common} fill="none" stroke="#FFFFFF" '
                    f'stroke-width="2" stroke-linejoin="round">'
                    f'{_esc(sub)}</text>\n'
                )
                labels_svg += (
                    f'<text {sub_common} fill="{COLOR_BASEMAP_LABEL}">'
                    f'{_esc(sub)}</text>\n'
                )
        else:
            _submit_basemap_label(
                placement_ctx,
                feature_id=f"obs:{_feature_id(f) or caption}:{_fmt(x)}",
                anchor=(x, apex_y),
                text=caption,
                size_px=label_size,
                family_mono=True,
                fill=COLOR_BASEMAP_CITY,
                preferred="NE",
                offset_px=4.0,
            )
            if sub:
                _submit_basemap_label(
                    placement_ctx,
                    feature_id=f"obsub:{_feature_id(f) or caption}:{_fmt(x)}",
                    anchor=(x, apex_y + sub_size + 0.5),
                    text=sub,
                    size_px=sub_size,
                    family_mono=False,
                    fill=COLOR_BASEMAP_LABEL,
                    preferred="NE",
                    offset_px=4.0,
                )
    if not stem_d and not tri_d:
        return ""
    if stem_d:
        out += (
            f'<path d="{"".join(stem_d)}" fill="none" '
            f'stroke="{COLOR_BASEMAP_CITY}" '
            f'stroke-width="{_fmt(_OBS_STEM_W)}" '
            f'stroke-linecap="butt"/>\n'
        )
    if tri_d:
        out += (
            f'<path d="{"".join(tri_d)}" fill="{COLOR_BASEMAP_CITY}" '
            f'stroke="none"/>\n'
        )
    out += labels_svg
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def render_basemap(
    geojson: dict,
    cfg: BasemapConfig,
    projector: Projector,
    clip_path_id: str | None = None,
    line_mask_id: str | None = None,
    plan_rect: tuple[float, float, float, float] | None = None,
    placement_ctx: object | None = None,
) -> str:
    """Render a basemap GeoJSON to an SVG fragment.

    Layer order (bottom -> top):
      1. hillshade raster (PNG, data-URI)            (unmasked, BELOW fills)
      2. lake polygons                               (unmasked fills)
      3. sea polygons                                (unmasked fills)
      4. populated-area polygons                     (unmasked fills)
      5. rivers + streams                            (masked)
      6. coastline                                   (masked)
      7. roads — tier-by-tier (motorway on top)      (masked)
      8. city dots + labels                          (masked)
      9. populated-area name labels                  (masked)
     10. spot-elevation triangles + altitude labels  (masked)
     11. tall-obstacle symbols                       (masked)

    When ``line_mask_id`` is provided, masked layers are wrapped in a
    child ``<g mask="url(#<id>)">`` so the caller's procedure-bbox mask
    hides any linework / label that would overlap a procedure element.
    The hillshade and big fills stay outside the mask — the procedure
    path draws on top of them anyway.
    """
    if not geojson:
        return ""

    features = list(_iter_features(geojson))

    # Bucket by class.
    by_class: dict[str, list[dict]] = {}
    for f in features:
        by_class.setdefault(_feature_class(f), []).append(f)

    # Label size scales with label_density. Normal = 7.5 px so basemap
    # labels read without squinting but still sit below procedure labels.
    if cfg.label_density == "sparse":
        label_size = 7.0
    elif cfg.label_density == "dense":
        label_size = 8.5
    else:
        label_size = 7.5

    # Micro label size for callouts (obstacle + spot elevations).
    micro_size = _SIZE_MICRO

    # --- Unmasked fills (hillshade raster, big background polygons) --------
    fills_body = ""
    # Hillshade first so land fills (and everything else) sit on top of it.
    hillshade_meta = geojson.get("_hillshade") if isinstance(geojson, dict) else None
    if cfg.show_hillshade and isinstance(hillshade_meta, dict):
        fills_body += _render_hillshade(hillshade_meta, projector)
    # Lakes go BEFORE sea + populated-area fills: landscape-layer semantic
    # (hillshade -> lakes -> sea -> cities). Sharing COLOR_BASEMAP_SEA
    # keeps water consistent across inland lakes and coastal ocean.
    if cfg.show_lakes:
        lake_feats = [
            f for f in by_class.get("lake", [])
            if _feature_rank(f, 9) <= cfg.waterway_max_rank
        ]
        lake_feats.sort(
            key=lambda f: (
                _feature_rank(f, 9), _feature_name(f), _feature_id(f),
            ),
        )
        fills_body += _render_polygonal_fill(
            lake_feats, projector,
            fill=COLOR_BASEMAP_SEA, stroke=COLOR_BASEMAP_COAST,
            sw=SW_LAKE_OUTLINE,
        )
    if cfg.show_sea:
        fills_body += _render_polygonal_fill(
            by_class.get("sea", []), projector, fill=COLOR_BASEMAP_SEA,
        )
    if cfg.show_populated_areas:
        fills_body += _render_populated_area_fills(
            by_class.get("populated_area", []), projector,
            plan_rect=plan_rect,
        )

    # --- Masked linework + labels -----------------------------------------
    lines_body = ""

    # 5. Rivers + streams (below coastline so the coast clips them visually).
    if cfg.show_rivers:
        river_feats = [
            f for f in by_class.get("river", [])
            if _feature_rank(f, 9) <= cfg.waterway_max_rank
        ]
        lines_body += _render_linear(
            river_feats, projector,
            stroke=COLOR_BASEMAP_RIVER, sw=SW_RIVER,
        )
    if cfg.show_streams:
        stream_feats = [
            f for f in by_class.get("stream", [])
            if _feature_rank(f, 9) <= cfg.waterway_max_rank
        ]
        lines_body += _render_linear(
            stream_feats, projector,
            stroke=COLOR_BASEMAP_RIVER, sw=SW_STREAM,
        )

    # 5. Coastline.
    if cfg.show_coastline:
        lines_body += _render_linear(
            by_class.get("coast", []), projector,
            stroke=COLOR_BASEMAP_COAST, sw=SW_COAST,
        )

    # 7. Roads — atlas-style double-stroke, drawn lowest tier first so
    # motorways cut through crossings at the top. Route-shield labels
    # (A1, B51) go out in the same group; the SVG mask hides any label
    # that would overlap the procedure.
    if cfg.show_roads:
        lines_body += _render_roads(
            by_class, projector, max_rank=cfg.road_max_rank,
            label_size=micro_size,
        )

    # 8. Cities (labels, then dots on top).
    if cfg.show_cities:
        cities = [
            f for f in by_class.get("city", [])
            if _feature_rank(f) <= cfg.city_max_rank
        ]
        lines_body += _render_cities(
            cities, projector,
            show_labels=cfg.show_city_labels,
            label_size=label_size,
            placement_ctx=placement_ctx,
        )

    # 9. Populated-area name labels — masked so they disappear where they'd
    # overlap a procedure element. (The fills for the same features were
    # already emitted in the unmasked fills group.)
    if cfg.show_populated_areas and cfg.show_city_labels:
        lines_body += _render_populated_area_labels(
            by_class.get("populated_area", []), projector, micro_size,
            plan_rect=plan_rect,
            placement_ctx=placement_ctx,
        )

    # Spot-elevation triangles + callouts. They live in the masked layer so
    # a callout that overlaps a procedure element gets hidden; triangles
    # are small enough that the masking rarely cuts into the symbol itself.
    if cfg.show_spot_elevations:
        lines_body += _render_spot_elevations(
            by_class.get("spot_elevation", []), projector, micro_size,
            placement_ctx=placement_ctx,
        )

    # 10. Tall obstacles (above every other basemap layer so they read as
    # hazards / foreground; still below the procedure overlay since the
    # basemap group as a whole is drawn first at the caller.)
    if cfg.show_obstacles:
        lines_body += _render_obstacles(
            by_class.get("obstacle_tall", []), projector,
            min_ft_agl=cfg.obstacle_min_ft_agl,
            micro_size=micro_size,
            placement_ctx=placement_ctx,
        )

    if not fills_body and not lines_body:
        return ""

    clip_attr = f' clip-path="url(#{clip_path_id})"' if clip_path_id else ""
    inner = ""
    if fills_body:
        inner += f'<g data-layer="basemap-fills">\n{fills_body}</g>\n'
    if lines_body:
        mask_attr = (
            f' mask="url(#{line_mask_id})"' if line_mask_id else ""
        )
        inner += (
            f'<g data-layer="basemap-lines"{mask_attr}>\n{lines_body}</g>\n'
        )
    return f'<g data-layer="basemap"{clip_attr}>\n{inner}</g>\n'


__all__ = [
    "BasemapConfig",
    "LabelDensity",
    "Projector",
    "COLOR_BASEMAP_SEA",
    "COLOR_BASEMAP_LAND",
    "COLOR_BASEMAP_COAST",
    "COLOR_BASEMAP_RIVER",
    "COLOR_BASEMAP_CITY",
    "COLOR_BASEMAP_LABEL",
    "COLOR_BASEMAP_URBAN",
    "COLOR_BASEMAP_ROAD_FILL",
    "SW_LAKE_OUTLINE",
    "SW_RIVER",
    "SW_STREAM",
    "ROAD_CLASSES",
    "basemap_path",
    "find_covering_region",
    "collect_basemap_label_requests",
    "load_basemap",
    "render_basemap",
]
