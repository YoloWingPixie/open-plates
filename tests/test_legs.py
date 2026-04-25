"""Tests for the ARINC 424 leg dispatcher.

Post-refactor (April 2026): the dispatcher emits simple primitives
(Segments joining waypoints, Arcs for AF, Holds for HM) for callers
that need the structural view — bbox computation, procedure-mask
rendering, and the projector. The rendered polyline is built
separately via :func:`compile_flight_states` + the trajectory builder.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from open_plates.geodesy import destination, great_circle_distance_nm
from open_plates.geometry import Arc, Hold, Segment
from open_plates.legs import build_fix_table, compile_legs

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "ugsb-ils-rwy-13.yaml"


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def test_ugsb_compiles() -> None:
    """End-to-end: UGSB ILS 13 must compile to a non-empty primitive list
    with no NotImplementedError, and must include at least one Hold for
    the HM leg."""
    proc = _load(EXAMPLE)
    fixes = build_fix_table(proc)
    prims = compile_legs(proc, fixes)

    assert prims
    assert any(isinstance(p, Hold) for p in prims)
    assert any(isinstance(p, Segment) for p in prims)


def test_ugsb_hold_at_btm() -> None:
    proc = _load(EXAMPLE)
    fixes = build_fix_table(proc)
    prims = compile_legs(proc, fixes)
    holds = [p for p in prims if isinstance(p, Hold)]
    assert len(holds) == 1
    btm = fixes["BTM"]
    assert holds[0].fix == btm


def test_unknown_terminator_raises() -> None:
    """Stubbed terminators must raise NotImplementedError by name."""
    proc = {
        "fixes": [
            {"id": "X", "type": "named_fix", "lat": 0.0, "lon": 0.0},
        ],
        "common_legs": [
            {"terminator": "VA", "transition_type": "common",
             "heading_deg": 90.0,
             "altitude_constraint": {"altitude_description": "at_or_above",
                                     "altitude_1_ft": 2000}},
        ],
    }
    fixes = build_fix_table(proc)
    with pytest.raises(NotImplementedError, match="VA"):
        compile_legs(proc, fixes)


def test_tf_segment_endpoints() -> None:
    """IF -> TF produces one Segment from the IF fix to the TF fix."""
    start = (0.0, 0.0)
    end = destination(start, 90.0, 10.0)

    proc = {
        "fixes": [
            {"id": "A", "type": "named_fix", "lat": start[0], "lon": start[1]},
            {"id": "B", "type": "named_fix", "lat": end[0], "lon": end[1]},
        ],
        "common_legs": [
            {"terminator": "IF", "transition_type": "common", "to_fix_id": "A"},
            {"terminator": "TF", "transition_type": "common",
             "from_fix_id": "A", "to_fix_id": "B", "course_deg": 90.0},
        ],
    }
    fixes = build_fix_table(proc)
    prims = compile_legs(proc, fixes)

    assert len(prims) == 1
    seg = prims[0]
    assert isinstance(seg, Segment)
    assert seg.start == start
    assert seg.end == end
    assert great_circle_distance_nm(seg.start, seg.end) == pytest.approx(
        10.0, rel=1e-6
    )


def test_cf_chain_emits_vertex_joined_segments() -> None:
    """IF -> CF east -> CF north: the leg dispatcher emits Segments that
    meet at the turn waypoint (no fillet). The smooth rendered path is
    produced by :func:`compile_flight_states` separately.
    """
    a = (0.0, 0.0)
    b = destination(a, 90.0, 10.0)
    c = destination(b, 0.0, 10.0)
    proc = {
        "fixes": [
            {"id": "A", "type": "named_fix", "lat": a[0], "lon": a[1]},
            {"id": "B", "type": "named_fix", "lat": b[0], "lon": b[1]},
            {"id": "C", "type": "named_fix", "lat": c[0], "lon": c[1]},
        ],
        "common_legs": [
            {"terminator": "IF", "transition_type": "common", "to_fix_id": "A"},
            {"terminator": "CF", "transition_type": "common",
             "from_fix_id": "A", "to_fix_id": "B", "course_deg": 90.0},
            {"terminator": "CF", "transition_type": "common",
             "from_fix_id": "B", "to_fix_id": "C", "course_deg": 0.0},
        ],
    }
    fixes = build_fix_table(proc)
    prims = compile_legs(proc, fixes)

    segs = [p for p in prims if isinstance(p, Segment)]
    assert len(segs) == 2
    assert segs[0].start == a
    assert segs[0].end == b
    assert segs[1].start == b
    assert segs[1].end == c


def test_df_emits_direct_segment() -> None:
    """DF from a starting position emits a single Segment to the target
    fix. Under the chart-symbology model, a DF is NOT a Dubins loop —
    it's a direct straight with a cosmetic bend at the origin (added by
    the trajectory builder, not the dispatcher).
    """
    rw = (49.988, 6.713)
    spa = (49.973, 6.692)

    proc = {
        "fixes": [
            {"id": "RW", "type": "runway_threshold",
             "lat": rw[0], "lon": rw[1], "elevation_ft": 1196},
            {"id": "SPA", "type": "navaid", "name": "SPA",
             "navaid_type": "tacan",
             "lat": spa[0], "lon": spa[1]},
        ],
        "common_legs": [
            {"terminator": "IF", "transition_type": "common", "to_fix_id": "RW"},
        ],
        "missed_approach": {
            "text_description": "Climb SW then direct SPA.",
            "icon_sequence": ["climb", "direct"],
            "legs": [
                {"terminator": "CA", "transition_type": "missed_approach",
                 "from_fix_id": "RW", "course_deg": 225.0,
                 "altitude_constraint": {"altitude_description": "at_or_above",
                                         "altitude_1_ft": 3000}},
                {"terminator": "DF", "transition_type": "missed_approach",
                 "to_fix_id": "SPA", "turn_direction": "right",
                 "altitude_constraint": {"altitude_description": "at",
                                         "altitude_1_ft": 3000}},
            ],
        },
    }
    fixes = build_fix_table(proc)
    prims = compile_legs(proc, fixes)

    segs = [p for p in prims if isinstance(p, Segment)]
    # CA emits 1 Segment, DF emits 1 Segment ending at SPA.
    assert segs
    assert any(s.end == fixes["SPA"] for s in segs), (
        "DF must terminate at the target fix"
    )
    # No fillet arcs — the dispatcher is pure structural geometry.
    arcs = [p for p in prims if isinstance(p, Arc)]
    assert not arcs, (
        "dispatcher should NOT emit Arc primitives for CF/DF joins; "
        "smoothing is the trajectory builder's job"
    )


def test_hm_leg_derives_length_from_time() -> None:
    """HM with leg_time_sec but no leg_distance_nm should compute
    leg_length_nm from the default TAS (210 kt)."""
    proc = {
        "fixes": [
            {"id": "BTM", "type": "named_fix", "lat": 41.6102, "lon": 41.5997},
        ],
        "common_legs": [
            {"terminator": "HM", "transition_type": "common",
             "to_fix_id": "BTM",
             "hold": {"inbound_course_deg": 307.0,
                      "turn_direction": "right",
                      "leg_time_sec": 60.0}},
        ],
    }
    fixes = build_fix_table(proc)
    prims = compile_legs(proc, fixes)

    holds = [p for p in prims if isinstance(p, Hold)]
    assert len(holds) == 1
    assert holds[0].leg_length_nm == pytest.approx(3.5, rel=1e-6)
    assert holds[0].leg_time_sec == 60.0
    assert holds[0].turn_direction == "right"
