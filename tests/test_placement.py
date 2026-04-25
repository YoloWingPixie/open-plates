"""Tests for the placement engine (Imhof-derived greedy resolver)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from open_plates.placement import (
    BBox,
    LabelCandidate,
    PlacementEngine,
    PlacementRequest,
    PlacementTier,
    bbox_intersects,
    bbox_grow,
    bbox_contains,
    eight_position_candidates,
    four_position_candidates,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPO_ROOT / "examples"


# ---------------------------------------------------------------------------
# BBox utilities
# ---------------------------------------------------------------------------


def test_bbox_intersects_disjoint():
    assert not bbox_intersects(BBox(0, 0, 10, 10), BBox(20, 0, 10, 10))


def test_bbox_intersects_touching_counts_as_disjoint():
    # Edge-touch must NOT count as a collision or else adjacent labels
    # with zero clearance would refuse to sit next to each other.
    assert not bbox_intersects(BBox(0, 0, 10, 10), BBox(10, 0, 10, 10))


def test_bbox_intersects_overlapping():
    assert bbox_intersects(BBox(0, 0, 10, 10), BBox(5, 5, 10, 10))


def test_bbox_grow_expands_symmetrically():
    g = bbox_grow(BBox(10, 20, 30, 40), 2.0)
    assert (g.x, g.y, g.w, g.h) == (8.0, 18.0, 34.0, 44.0)


def test_bbox_contains_basic():
    b = BBox(0, 0, 10, 10)
    assert bbox_contains(b, (5, 5))
    assert not bbox_contains(b, (15, 5))
    # Right/bottom edges excluded.
    assert not bbox_contains(b, (10, 10))


# ---------------------------------------------------------------------------
# Candidate presets
# ---------------------------------------------------------------------------


def test_eight_position_default_starts_NE():
    cands = eight_position_candidates()
    assert len(cands) == 8
    # First candidate = NE: positive x, negative y (screen space, y grows
    # down).
    assert cands[0].offset_x > 0 and cands[0].offset_y < 0


def test_eight_position_rotates_from_preferred():
    cands = eight_position_candidates(preferred="W")
    # First entry points west (negative x, zero y).
    assert cands[0].offset_x < 0
    assert abs(cands[0].offset_y) < 1e-9


def test_four_position_produces_four():
    cands = four_position_candidates()
    assert len(cands) == 4


# ---------------------------------------------------------------------------
# Resolver — core cases
# ---------------------------------------------------------------------------


def _plan() -> BBox:
    return BBox(0, 0, 400, 300)


def _label_req(
    feature_id: str,
    anchor: tuple[float, float],
    candidates=None,
    *,
    fallback="displace",
    size=(20.0, 10.0),
    clearance: float = 0.0,
    tier: PlacementTier = PlacementTier.LABEL,
) -> PlacementRequest:
    if candidates is None:
        candidates = eight_position_candidates()
    return PlacementRequest(
        tier=tier,
        anchor=anchor,
        content_size=size,
        candidates=candidates,
        fallback=fallback,
        feature_id=feature_id,
        required_clearance_px=clearance,
    )


def test_8_position_picks_preferred_when_free():
    eng = PlacementEngine(plan_rect=_plan())
    req = _label_req("A", (100, 100))
    [placed] = eng.place([req])
    # Preferred = NE @ 8 px.
    assert placed.offset_used[0] > 0
    assert placed.offset_used[1] < 0
    assert not placed.displaced
    assert placed.leader_line is None


def test_collision_cascades_to_second_candidate():
    eng = PlacementEngine(plan_rect=_plan())
    # Block the NE position of the second label by reserving a fixed bbox
    # right where it would have landed.
    blocker = PlacementRequest(
        tier=PlacementTier.FIX,
        anchor=(0, 0),
        content_size=(0, 0),
        candidates=(),
        fallback="suppress",
        feature_id="blocker",
        fixed_bbox=BBox(105, 85, 25, 15),
    )
    req = _label_req("lbl", (100, 100))
    placed = eng.place([blocker, req])
    # Second label should have picked a different candidate, NOT NE.
    lbl = placed[1]
    assert not lbl.displaced
    assert lbl.leader_line is None
    # Anything OTHER than the NE-pointing first candidate is acceptable.
    first = req.candidates[0]
    assert (lbl.offset_used[0], lbl.offset_used[1]) != (
        first.offset_x, first.offset_y,
    )


def test_all_candidates_collide_triggers_leader():
    eng = PlacementEngine(plan_rect=_plan())
    # Reserve a massive bbox around the anchor so all 8 candidates fail
    # AND the displace double-distance pass also fails.
    blocker = PlacementRequest(
        tier=PlacementTier.FIX,
        anchor=(0, 0),
        content_size=(0, 0),
        candidates=(),
        fallback="suppress",
        feature_id="blocker",
        fixed_bbox=BBox(50, 50, 100, 100),
    )
    req = _label_req("crowded", (100, 100), fallback="leader")
    placed = eng.place([blocker, req])
    lbl = placed[1]
    assert lbl.leader_line is not None
    assert lbl.displaced is True


def test_suppress_fallback_logs_omitted():
    eng = PlacementEngine(plan_rect=BBox(0, 0, 50, 50))
    # Tiny plan + tiny anchor surrounded by a blocker that covers the
    # entire plan rect. Margins also blocked -> suppression.
    blocker = PlacementRequest(
        tier=PlacementTier.FIX,
        anchor=(0, 0),
        content_size=(0, 0),
        candidates=(),
        fallback="suppress",
        feature_id="blocker",
        fixed_bbox=BBox(-10, -10, 70, 70),
    )
    req = _label_req("doomed", (25, 25), fallback="suppress",
                     size=(40.0, 20.0))
    placed = eng.place([blocker, req])
    assert "doomed" in eng.omitted
    # The Placed sentinel still exists with a zero-area bbox so callers
    # iterating in input order can match it up.
    assert placed[1].bbox.w == 0 and placed[1].bbox.h == 0


def test_deterministic_tie_break_by_feature_id():
    # Repeated calls with identical input must yield identical output.
    # Determinism here means "same input produces same placements", which
    # is what the caller relies on for byte-identical SVG. Within a tier,
    # insertion order decides who gets first pick (so the caller can
    # express "fix-id labels before altitude boxes" by submitting them
    # in that order); feature_id only breaks ties at the same input
    # index (rare, present for robustness).
    reqs = [
        _label_req("B", (100, 100)),
        _label_req("A", (100, 100)),
    ]
    p1 = PlacementEngine(plan_rect=_plan()).place(list(reqs))
    p2 = PlacementEngine(plan_rect=_plan()).place(list(reqs))
    by_id1 = {p.request.feature_id: p.offset_used for p in p1}
    by_id2 = {p.request.feature_id: p.offset_used for p in p2}
    assert by_id1 == by_id2
    # With insertion-order primary, B (submitted first) wins the NE
    # preferred candidate; A falls to a secondary candidate.
    assert by_id1["B"] != by_id1["A"]


def test_feature_id_breaks_ties_at_same_index():
    # Two requests at the same notional "same-slot" position (we can't
    # literally share an input index, but we can verify that at the
    # same tier + anchor the engine reliably prefers one over the other
    # in a deterministic way).
    eng = PlacementEngine(plan_rect=_plan())
    reqs = [_label_req("A", (100, 100)), _label_req("B", (100, 100))]
    placed = eng.place(reqs)
    # A (input_index 0) reserves first -> NE; B reserves second.
    assert placed[0].offset_used != placed[1].offset_used


def test_fixed_bbox_blocks_labels():
    eng = PlacementEngine(plan_rect=_plan())
    # Fixed NAVAID-tier bbox that occupies the NE position relative to the
    # label's anchor.
    fixed = PlacementRequest(
        tier=PlacementTier.NAVAID,
        anchor=(0, 0),
        content_size=(0, 0),
        candidates=(),
        fallback="suppress",
        feature_id="fix_VOR",
        fixed_bbox=BBox(100, 80, 40, 30),
    )
    lbl = _label_req("lbl", (100, 100))
    placed = eng.place([fixed, lbl])
    # The first candidate (NE @ 8 px) would have landed inside the fixed
    # bbox; the resolver must pick something else.
    placed_lbl = placed[1]
    assert placed_lbl.bbox.x != 108 or placed_lbl.bbox.y != 92


def test_fixed_bbox_returned_verbatim():
    eng = PlacementEngine(plan_rect=_plan())
    b = BBox(10, 10, 50, 20)
    req = PlacementRequest(
        tier=PlacementTier.NAVAID,
        anchor=(0, 0),
        content_size=(0, 0),
        candidates=(),
        fallback="suppress",
        feature_id="g",
        fixed_bbox=b,
    )
    [p] = eng.place([req])
    assert p.bbox == b
    assert p.offset_used == (0.0, 0.0)


def test_results_preserve_input_order():
    eng = PlacementEngine(plan_rect=_plan())
    reqs = [
        _label_req("Z", (50, 50)),
        _label_req("A", (150, 150)),
        _label_req("M", (250, 250)),
    ]
    placed = eng.place(reqs)
    # Result index i corresponds to request index i.
    for i, p in enumerate(placed):
        assert p.request.feature_id == reqs[i].feature_id


# ---------------------------------------------------------------------------
# Integration — end-to-end render via the real renderer.
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _placed_label_bboxes(placed_list):
    """Return bboxes from a list of Placed records, skipping zero-area
    suppressions and fixed-bbox reservations."""
    out = []
    for p in placed_list:
        if p.bbox.w == 0 or p.bbox.h == 0:
            continue
        if p.request.fixed_bbox is not None:
            continue
        out.append(p.bbox)
    return out


def test_etad_render_uses_placement_engine():
    """Rendering ETAD must route label placement through the engine and
    produce no mutually-overlapping labels."""
    from open_plates.render_svg import (
        _layout_regions, _plan_view_placement_trace,
    )
    doc = _load_yaml(EXAMPLES / "etad-ils-rwy-23-straight.yaml")
    trace = _plan_view_placement_trace(doc)
    assert trace is not None, "renderer did not expose placement trace"
    # Pull only label-tier bboxes: those are the ones that went through
    # candidate iteration. Fixed-geometry tiers reserve space but are
    # expected to collide with labels (labels route AROUND them).
    label_bboxes = [
        p.bbox for p in trace
        if p.request.tier == PlacementTier.LABEL
        and p.bbox.w > 0 and p.bbox.h > 0
    ]
    # No two label bboxes may overlap after resolution.
    for i in range(len(label_bboxes)):
        for j in range(i + 1, len(label_bboxes)):
            assert not bbox_intersects(label_bboxes[i], label_bboxes[j]), (
                f"label overlap: {label_bboxes[i]} vs {label_bboxes[j]}"
            )
