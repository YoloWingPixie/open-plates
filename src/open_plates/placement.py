"""Deterministic label + fixed-geometry placement engine.

Industry-aligned, Imhof-derived greedy resolver for the plan view.
Fixed-geometry elements (procedure primitives, LOC feather, navaid
symbols, fix symbols, runways, basemap raster) register axis-aligned
bounding boxes at a high tier; label requests then walk candidate
positions in a fixed preference order, taking the first that does not
collide with any previously reserved bbox.

This module is intentionally stdlib-only. The resolver is O(n^2) on
bbox-vs-bbox overlap. Measured budget: < 5 ms for 500 requests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Literal, Sequence

__all__ = [
    "PlacementTier",
    "BBox",
    "LabelCandidate",
    "PlacementRequest",
    "Placed",
    "PlacementEngine",
    "bbox_intersects",
    "bbox_grow",
    "bbox_contains",
    "eight_position_candidates",
    "four_position_candidates",
]


# ---------------------------------------------------------------------------
# Tier taxonomy
# ---------------------------------------------------------------------------


class PlacementTier(IntEnum):
    """Rendering priority tiers. Higher numbers paint last / win conflicts.

    The resolver walks requests in DESCENDING tier order: top tiers reserve
    their space first, lower tiers adapt around them. This matches the
    convention in every published chart style guide (FAA AeroNav style
    manual, ICAO Annex 4 appendix 2, DFS CS-AD): page furniture and
    reference frames are inviolate; procedure geometry is inviolate;
    labels move to avoid both.
    """

    FRAME = 0        # page paper / corner brackets
    FILL = 1         # hillshade, sea, lake, populated-area polygons
    TERRAIN = 2      # coastline, rivers, spot-elevation triangles
    CULTURE = 3      # roads, city dots, populated-area labels, obstacles
    AIRSPACE = 4     # MSA bezel, restricted areas, compass rose
    NAVAID = 5       # navaid symbols (VOR/VOR-DME/NDB), runways, LOC feather
    PROCEDURE = 6    # procedure path primitives, hold racetracks
    FIX = 7          # fix symbols (fly-by / fly-over markers)
    LABEL = 8        # fix-id, role, altitude box, course, DME tick labels
    REFERENCE = 9    # scale bar, north arrow, notes sidebar, PLAN VIEW tag
    PAGE = 10        # plate code, page title (already outside the plan rect)


# ---------------------------------------------------------------------------
# Geometry primitives
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BBox:
    """Axis-aligned bounding box in screen-space pixels."""

    x: float
    y: float
    w: float
    h: float

    @property
    def x1(self) -> float:
        return self.x + self.w

    @property
    def y1(self) -> float:
        return self.y + self.h


def bbox_intersects(a: BBox, b: BBox) -> bool:
    """True iff ``a`` and ``b`` overlap (edge-touching counts as disjoint)."""
    if a.x1 <= b.x or b.x1 <= a.x:
        return False
    if a.y1 <= b.y or b.y1 <= a.y:
        return False
    return True


def bbox_grow(a: BBox, px: float) -> BBox:
    """Return ``a`` expanded by ``px`` on every side."""
    return BBox(a.x - px, a.y - px, a.w + 2 * px, a.h + 2 * px)


def bbox_contains(a: BBox, p: tuple[float, float]) -> bool:
    """True iff the point ``p`` lies inside ``a`` (right/bottom edges excluded)."""
    return a.x <= p[0] < a.x1 and a.y <= p[1] < a.y1


# ---------------------------------------------------------------------------
# Request / response dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LabelCandidate:
    """One proposed position for a label.

    ``offset_x`` / ``offset_y`` are pixels from the anchor point. Positive
    x is east, positive y is south (screen-space).

    ``weight`` is an integer preference score (lower = more preferred).
    The resolver iterates candidates in their provided order; ``weight``
    is only used to break ties when multiple candidates share a position
    (rare; present mainly for user-supplied preference hints).
    """

    offset_x: float
    offset_y: float
    weight: int = 0


Fallback = Literal["displace", "leader", "suppress"]


@dataclass(frozen=True)
class PlacementRequest:
    """A single placement request — either a label or a fixed bbox.

    ``fixed_bbox`` switches the request into "reservation only" mode: the
    resolver skips candidate iteration and simply records the bbox at the
    request's tier so later lower-tier placements avoid it. Geometry
    elements (procedure path, fix symbols, navaid hexagons, LOC feather)
    use this path.

    For label requests, ``candidates`` lists the preferred positions in
    order. If every candidate collides, ``fallback`` decides the outcome:

    * ``"displace"`` — re-try each candidate with the offsets doubled.
      Cheap, gives breathing room on slightly crowded plates.
    * ``"leader"`` — emit the label along the nearest plan-view margin
      with a thin line from the anchor to the label box. Jeppesen
      callout convention.
    * ``"suppress"`` — omit the label and record the ``feature_id`` on
      the engine's ``omitted`` list for the caller to surface.
    """

    tier: PlacementTier
    anchor: tuple[float, float]
    content_size: tuple[float, float]
    candidates: tuple[LabelCandidate, ...]
    fallback: Fallback
    feature_id: str
    required_clearance_px: float = 2.0
    fixed_bbox: BBox | None = None


@dataclass(frozen=True)
class Placed:
    """Resolver output for one request.

    * ``bbox`` is where the element lands (always populated — for fixed
      reservations, equal to ``request.fixed_bbox``).
    * ``offset_used`` is the (dx, dy) of the chosen candidate, or ``(0, 0)``
      for fixed-bbox reservations.
    * ``displaced`` flags whether the ``displace`` fallback fired.
    * ``leader_line`` is ``((ax, ay), (lx, ly))`` when the ``leader``
      fallback fired, else ``None``.
    """

    request: PlacementRequest
    bbox: BBox
    offset_used: tuple[float, float]
    displaced: bool = False
    leader_line: tuple[tuple[float, float], tuple[float, float]] | None = None


# ---------------------------------------------------------------------------
# Candidate presets
# ---------------------------------------------------------------------------


_COMPASS_ORDER: tuple[str, ...] = ("NE", "E", "SE", "S", "SW", "W", "NW", "N")

# Pixel offsets for each compass direction at unit radius. Y grows DOWN in
# screen space so N is negative, S is positive.
_COMPASS_UNIT: dict[str, tuple[float, float]] = {
    "E":  (1.0,  0.0),
    "NE": (0.707, -0.707),
    "N":  (0.0,  -1.0),
    "NW": (-0.707, -0.707),
    "W":  (-1.0, 0.0),
    "SW": (-0.707, 0.707),
    "S":  (0.0, 1.0),
    "SE": (0.707, 0.707),
}


def _rotate(seq: Sequence[str], start: str) -> list[str]:
    """Return ``seq`` rotated so ``start`` is first; unchanged if absent."""
    if start not in seq:
        return list(seq)
    i = list(seq).index(start)
    return list(seq[i:]) + list(seq[:i])


def eight_position_candidates(
    preferred: Literal["NE", "E", "SE", "S", "SW", "W", "NW", "N"] = "NE",
    offset_px: float = 8.0,
) -> tuple[LabelCandidate, ...]:
    """Imhof-style 8-position candidate set.

    Primary offset is ``offset_px`` from the anchor in each compass
    direction. Preference order begins at ``preferred`` and rotates
    clockwise — this is the Jeppesen plan-view convention when a fix
    label has no course-geometry preference.
    """
    order = _rotate(_COMPASS_ORDER, preferred)
    out: list[LabelCandidate] = []
    for rank, direction in enumerate(order):
        ux, uy = _COMPASS_UNIT[direction]
        out.append(
            LabelCandidate(
                offset_x=ux * offset_px,
                offset_y=uy * offset_px,
                weight=rank,
            )
        )
    return tuple(out)


def four_position_candidates(
    preferred: Literal["N", "S", "E", "W"] = "N",
    offset_px: float = 8.0,
) -> tuple[LabelCandidate, ...]:
    """N/S/E/W candidate set — for fixes along a course where diagonals
    collide with the course line (altitude-box anchors, DME tick labels).
    """
    order = _rotate(("N", "E", "S", "W"), preferred)
    out: list[LabelCandidate] = []
    for rank, direction in enumerate(order):
        ux, uy = _COMPASS_UNIT[direction]
        out.append(
            LabelCandidate(
                offset_x=ux * offset_px,
                offset_y=uy * offset_px,
                weight=rank,
            )
        )
    return tuple(out)


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------


def _candidate_bbox(req: PlacementRequest, cand: LabelCandidate) -> BBox:
    """Screen-space bbox for a candidate offset."""
    ax, ay = req.anchor
    w, h = req.content_size
    return BBox(ax + cand.offset_x, ay + cand.offset_y, w, h)


def _rect_covers(plan_rect: BBox, inner: BBox) -> bool:
    """True iff ``inner`` is fully contained in ``plan_rect``."""
    return (
        inner.x >= plan_rect.x
        and inner.y >= plan_rect.y
        and inner.x1 <= plan_rect.x1
        and inner.y1 <= plan_rect.y1
    )


@dataclass
class PlacementEngine:
    """Deterministic greedy label + geometry placement resolver."""

    plan_rect: BBox
    _omitted: list[str] = field(default_factory=list)

    @property
    def omitted(self) -> list[str]:
        """Feature ids that were suppressed. Populated after ``place``."""
        return list(self._omitted)

    def place(self, requests: list[PlacementRequest]) -> list[Placed]:
        """Resolve all requests and return ``Placed`` in input order.

        Order of evaluation:

        1. Every request that supplies a ``fixed_bbox`` (geometry
           reservations) reserves its space first, in ``(tier DESC,
           input_index ASC, feature_id ASC)`` order. Fixed reservations
           never move, so the order among them is purely for determinism.
        2. Remaining (label) requests then iterate candidates in
           ``(tier DESC, input_index ASC, feature_id ASC)`` order: higher
           tiers get first pick at their preferred positions, and within
           a tier the caller-controlled insertion order decides who
           reserves first. ``feature_id`` breaks ties when two requests
           land at the same input index (not possible in the normal
           path; present for robustness).

        The split ensures labels always see ALL fixed geometry when
        choosing a candidate, which matches the industry pattern
        (procedure primitives + navaid symbols are inviolate; labels
        route around them).
        """
        # Tag each request with its original input index so we can restore
        # that ordering on the way out (callers map request -> SVG frag by
        # index; shuffling would break them).
        indexed = list(enumerate(requests))
        fixed_reqs = [t for t in indexed if t[1].fixed_bbox is not None]
        label_reqs = [t for t in indexed if t[1].fixed_bbox is None]
        # Sort: primary = -tier (desc), secondary = input_index (asc),
        # tertiary = feature_id (asc) — pure determinism tiebreak.
        fixed_reqs.sort(key=lambda t: (-t[1].tier, t[0], t[1].feature_id))
        label_reqs.sort(key=lambda t: (-t[1].tier, t[0], t[1].feature_id))

        results: list[Placed | None] = [None] * len(requests)
        reserved: list[BBox] = []
        self._omitted = []

        for idx, req in fixed_reqs + label_reqs:
            placed = self._place_one(req, reserved)
            results[idx] = placed
            # Only extend reservations for elements that actually landed.
            # Suppressed labels contribute no bbox.
            if placed is not None:
                reserved.append(placed.bbox)
            else:
                self._omitted.append(req.feature_id)

        # ``None`` only appears for suppressed labels — substitute a zero
        # bbox Placed record with displaced=False and a sentinel leader
        # so callers iterating in input order still get a value back.
        out: list[Placed] = []
        for i, r in enumerate(results):
            if r is not None:
                out.append(r)
                continue
            req = requests[i]
            # Sentinel record with an empty bbox at the anchor; callers
            # can detect the suppression via the ``omitted`` list.
            out.append(
                Placed(
                    request=req,
                    bbox=BBox(req.anchor[0], req.anchor[1], 0.0, 0.0),
                    offset_used=(0.0, 0.0),
                    displaced=False,
                    leader_line=None,
                )
            )
        return out

    # --- internal --------------------------------------------------------

    def _place_one(
        self, req: PlacementRequest, reserved: list[BBox],
    ) -> Placed | None:
        # Fixed reservation — no candidate iteration.
        if req.fixed_bbox is not None:
            return Placed(
                request=req,
                bbox=req.fixed_bbox,
                offset_used=(0.0, 0.0),
                displaced=False,
                leader_line=None,
            )

        # Candidate iteration (preference order).
        clearance = req.required_clearance_px
        for cand in req.candidates:
            cb = _candidate_bbox(req, cand)
            if not self._candidate_ok(cb, reserved, clearance):
                continue
            return Placed(
                request=req,
                bbox=cb,
                offset_used=(cand.offset_x, cand.offset_y),
            )

        # All candidates collided -> fallback cascade.
        if req.fallback == "displace":
            for cand in req.candidates:
                doubled = LabelCandidate(
                    offset_x=cand.offset_x * 2.0,
                    offset_y=cand.offset_y * 2.0,
                    weight=cand.weight,
                )
                cb = _candidate_bbox(req, doubled)
                if not self._candidate_ok(cb, reserved, clearance):
                    continue
                return Placed(
                    request=req,
                    bbox=cb,
                    offset_used=(doubled.offset_x, doubled.offset_y),
                    displaced=True,
                )
            # Displacement exhausted — fall through to leader behaviour so
            # the label still appears somewhere. This matches Jeppesen's
            # "if you can't nudge it, call it out" convention.
            return self._leader_placement(req, reserved)

        if req.fallback == "leader":
            return self._leader_placement(req, reserved)

        # fallback == "suppress": drop it.
        return None

    def _candidate_ok(
        self, cb: BBox, reserved: list[BBox], clearance: float,
    ) -> bool:
        if not _rect_covers(self.plan_rect, cb):
            return False
        # Compare the candidate grown by its clearance against each reserved
        # bbox. We grow the candidate, not the reservations, so a large
        # piece of geometry doesn't visually repel every label by its
        # padding (only the label needs breathing room around itself).
        grown = bbox_grow(cb, clearance) if clearance > 0 else cb
        for r in reserved:
            if bbox_intersects(grown, r):
                return False
        return True

    def _leader_placement(
        self, req: PlacementRequest, reserved: list[BBox],
    ) -> Placed | None:
        """Emit the label in a free slot along the nearest plan margin,
        with a leader line from the anchor to the label's edge.

        Search pattern: walk each of the four margins clockwise from the
        one nearest to the anchor, scanning every ``content.w + 4`` px
        slot until one clears all reservations. If all four margins are
        saturated, suppress the label and record the feature id.
        """
        ax, ay = req.anchor
        w, h = req.content_size
        margin_pad = 4.0

        plan = self.plan_rect
        # Distances to each margin for priority ordering.
        dists = {
            "N": ay - plan.y,
            "S": plan.y1 - ay,
            "W": ax - plan.x,
            "E": plan.x1 - ax,
        }
        order = sorted(dists.items(), key=lambda kv: kv[1])
        for side, _ in order:
            slot = self._first_free_margin_slot(
                side, w, h, margin_pad, reserved,
            )
            if slot is None:
                continue
            lx, ly = slot
            cb = BBox(lx, ly, w, h)
            # Leader line from the anchor to the label box edge nearest
            # the anchor. Hairline ``0.5`` stroke width is set by the
            # caller at SVG-emit time.
            leader_end = _closest_edge_point(cb, (ax, ay))
            return Placed(
                request=req,
                bbox=cb,
                offset_used=(lx - ax, ly - ay),
                displaced=True,
                leader_line=((ax, ay), leader_end),
            )
        # All margins full — suppress.
        return None

    def _first_free_margin_slot(
        self,
        side: Literal["N", "S", "E", "W"],
        w: float,
        h: float,
        pad: float,
        reserved: list[BBox],
    ) -> tuple[float, float] | None:
        """Scan a margin strip for the first slot that clears reservations."""
        plan = self.plan_rect
        if side == "N":
            y = plan.y + pad
            x = plan.x + pad
            step = w + pad
            while x + w <= plan.x1 - pad:
                cb = BBox(x, y, w, h)
                if self._candidate_ok(cb, reserved, 0.0):
                    return (x, y)
                x += step
        elif side == "S":
            y = plan.y1 - pad - h
            x = plan.x + pad
            step = w + pad
            while x + w <= plan.x1 - pad:
                cb = BBox(x, y, w, h)
                if self._candidate_ok(cb, reserved, 0.0):
                    return (x, y)
                x += step
        elif side == "W":
            x = plan.x + pad
            y = plan.y + pad
            step = h + pad
            while y + h <= plan.y1 - pad:
                cb = BBox(x, y, w, h)
                if self._candidate_ok(cb, reserved, 0.0):
                    return (x, y)
                y += step
        else:  # E
            x = plan.x1 - pad - w
            y = plan.y + pad
            step = h + pad
            while y + h <= plan.y1 - pad:
                cb = BBox(x, y, w, h)
                if self._candidate_ok(cb, reserved, 0.0):
                    return (x, y)
                y += step
        return None


def _closest_edge_point(
    box: BBox, anchor: tuple[float, float],
) -> tuple[float, float]:
    """Return the point on ``box``'s perimeter nearest to ``anchor``."""
    ax, ay = anchor
    cx = min(max(ax, box.x), box.x1)
    cy = min(max(ay, box.y), box.y1)
    # If anchor is inside the box (rare), clamp to nearest edge.
    if box.x < cx < box.x1 and box.y < cy < box.y1:
        # Pick the nearest edge.
        d_left = cx - box.x
        d_right = box.x1 - cx
        d_top = cy - box.y
        d_bot = box.y1 - cy
        m = min(d_left, d_right, d_top, d_bot)
        if m == d_left:
            cx = box.x
        elif m == d_right:
            cx = box.x1
        elif m == d_top:
            cy = box.y
        else:
            cy = box.y1
    return (cx, cy)
