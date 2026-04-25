"""Local equirectangular projection for the plan view.

Cheap-and-cheerful: for a chart spanning a few tens of NM we scale
longitude by `cos(center_lat)` so 1° of lon ≈ 1° of lat (in NM terms) at
the plan-view center. Aspect is preserved; letterboxing handles any
mismatch between the plotted extent and the plan-view rectangle.

Exposes `plan_projector`, which takes the procedure + fix table + plan
rectangle and returns a LatLon -> (x_px, y_px) function. The projector
keeps `true-north up` (y increases downward in SVG, so we flip lat).
"""

from __future__ import annotations

from math import cos, radians
from typing import Callable, Iterable

from .geodesy import LatLon
from .geometry import Arc, Hold, Primitive, Segment

Projector = Callable[[LatLon], tuple[float, float]]

BBox = tuple[float, float, float, float]  # (min_lat, min_lon, max_lat, max_lon)
PxRect = tuple[float, float, float, float]  # (x0, y0, w, h)

DEFAULT_PAD_FRAC: float = 0.10


def _collect_points(
    procedure: dict,
    fixes: dict[str, LatLon],
    primitives: Iterable[Primitive] | None = None,
) -> list[LatLon]:
    """Return every LatLon the plan view needs to contain.

    Covers what actually gets drawn: fixes referenced by legs (matches the
    renderer's fix-symbol pass), primitive endpoints, and runway thresholds.
    The MSA ring is NOT sampled — it draws in the bezel, not the plan view,
    so including it would only bloat the bbox and squash the procedure.
    """
    points: list[LatLon] = []

    used_ids: set[str] = set()
    for t in procedure.get("transitions") or []:
        for leg in t.get("legs") or []:
            for key in ("from_fix_id", "to_fix_id"):
                v = leg.get(key)
                if v:
                    used_ids.add(v)
    for leg in procedure.get("common_legs") or []:
        for key in ("from_fix_id", "to_fix_id"):
            v = leg.get(key)
            if v:
                used_ids.add(v)
    for leg in (procedure.get("missed_approach") or {}).get("legs") or []:
        for key in ("from_fix_id", "to_fix_id"):
            v = leg.get(key)
            if v:
                used_ids.add(v)
    msa_center = (procedure.get("msa") or {}).get("center_navaid_id")
    if msa_center:
        used_ids.add(msa_center)
    for fid in used_ids:
        if fid in fixes:
            points.append(fixes[fid])

    for rw in procedure.get("runways") or []:
        for key in ("low_threshold_lat_lon", "high_threshold_lat_lon"):
            tl = rw.get(key) or {}
            if "lat" in tl and "lon" in tl:
                points.append((float(tl["lat"]), float(tl["lon"])))

    if primitives:
        from .geodesy import destination
        for p in primitives:
            if isinstance(p, Segment):
                points.append(p.start)
                points.append(p.end)
            elif isinstance(p, Arc):
                # Sample 8 points around the arc's circle so the bbox
                # actually contains the arc's extent — not just its center.
                # Matters for Dubins CS turns where the arc can span >300°
                # and extend a full diameter past the center.
                for brg in (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0):
                    points.append(destination(p.center, brg, p.radius_nm))
            elif isinstance(p, Hold):
                points.append(p.fix)

    return points


def _bbox(points: list[LatLon]) -> BBox:
    lats = [p[0] for p in points]
    lons = [p[1] for p in points]
    return (min(lats), min(lons), max(lats), max(lons))


def _pad_bbox(bbox: BBox, frac: float) -> BBox:
    min_lat, min_lon, max_lat, max_lon = bbox
    dlat = max_lat - min_lat
    dlon = max_lon - min_lon
    # Avoid zero span.
    if dlat == 0:
        dlat = 0.01
    if dlon == 0:
        dlon = 0.01
    return (
        min_lat - dlat * frac,
        min_lon - dlon * frac,
        max_lat + dlat * frac,
        max_lon + dlon * frac,
    )


def plan_projector(
    procedure: dict,
    fixes: dict[str, LatLon],
    plan_bounds_px: PxRect,
    primitives: Iterable[Primitive] | None = None,
    pad_frac: float = DEFAULT_PAD_FRAC,
) -> Projector:
    """Return a LatLon -> (x_px, y_px) function.

    The projector maps into `plan_bounds_px = (x0, y0, w, h)`, preserves
    aspect (letterboxing if the bbox and rect don't match), and keeps
    north-up.
    """
    points = _collect_points(procedure, fixes, primitives)
    if not points:
        # Nothing to project — return a degenerate projector that maps
        # everything to the rect center. Keeps the renderer robust on
        # malformed input instead of crashing.
        x0, y0, w, h = plan_bounds_px
        cx, cy = x0 + w / 2, y0 + h / 2

        def _null(_ll: LatLon) -> tuple[float, float]:
            return (cx, cy)

        return _null

    bbox = _pad_bbox(_bbox(points), pad_frac)
    min_lat, min_lon, max_lat, max_lon = bbox
    center_lat = (min_lat + max_lat) / 2.0
    lon_scale = cos(radians(center_lat))

    # Work in a local planar coordinate: x_deg = (lon - min_lon) * lon_scale,
    # y_deg = (max_lat - lat). Both are non-negative with max at the far
    # corner of the bbox.
    world_w = (max_lon - min_lon) * lon_scale
    world_h = (max_lat - min_lat)
    if world_w <= 0:
        world_w = 1e-9
    if world_h <= 0:
        world_h = 1e-9

    x0, y0, w_px, h_px = plan_bounds_px

    # Uniform scale so both axes fit; letterbox the shorter dimension.
    scale = min(w_px / world_w, h_px / world_h)
    # Center the plotted content in the rect.
    plotted_w = world_w * scale
    plotted_h = world_h * scale
    pad_x = (w_px - plotted_w) / 2.0
    pad_y = (h_px - plotted_h) / 2.0

    def project(ll: LatLon) -> tuple[float, float]:
        lat, lon = ll
        x = (lon - min_lon) * lon_scale * scale + x0 + pad_x
        y = (max_lat - lat) * scale + y0 + pad_y
        return (x, y)

    return project


def plan_world_bbox(
    procedure: dict,
    fixes: dict[str, LatLon],
    primitives: Iterable[Primitive] | None = None,
    pad_frac: float = DEFAULT_PAD_FRAC,
) -> BBox | None:
    """Return the padded geographic bbox the plan projector fits to.

    Matches the bbox ``plan_projector`` computes internally from
    ``_collect_points`` + ``_pad_bbox`` with the same ``pad_frac``.
    Returns ``None`` when there are no points to project (degenerate
    procedure). The tuple is ``(min_lat, min_lon, max_lat, max_lon)``.

    Used by ``load_basemap`` to compare the plan extent against each
    loaded basemap sibling's declared bbox.
    """
    points = _collect_points(procedure, fixes, primitives)
    if not points:
        return None
    return _pad_bbox(_bbox(points), pad_frac)


def projector_scale_px_per_nm(projector: Projector, anchor: LatLon) -> float:
    """Pixels per NM at `anchor`, measured along an east-going offset.

    Useful for scale bars and projected arc radii.
    """
    from .geodesy import destination

    a = projector(anchor)
    b = projector(destination(anchor, 90.0, 1.0))
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return (dx * dx + dy * dy) ** 0.5


__all__ = [
    "Projector",
    "plan_projector",
    "plan_world_bbox",
    "projector_scale_px_per_nm",
]
