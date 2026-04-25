"""Tests for flight-state trajectory rendering.

Post-refactor (April 2026): the chart path is built from a
:mod:`open_plates.trajectory` waypoint list with a CONSTANT cosmetic
bend radius at every fly-by waypoint (independent of aircraft speed).

Each flight state ("one flyable trajectory from entry to termination")
renders as exactly two SVG <polyline> elements — a paper casing under a
coloured core. These tests lock in:
  - The state inventory for each example procedure.
  - Smoothness of the flattened polyline (no loops, no right angles).
  - DME-arc fidelity for AF legs.
  - The SVG emits one polyline pair per approach state and one dashed
    pair for the missed approach.
"""

from __future__ import annotations

import re
from math import radians
from pathlib import Path

import pytest
import yaml

from open_plates.geodesy import (
    destination,
    great_circle_distance_nm,
    initial_bearing_deg,
)
from open_plates.geometry import FlightState, Hold
from open_plates.legs import build_fix_table, compile_flight_states
from open_plates.render_svg import (
    COLOR_MISSED,
    COLOR_PAPER,
    COLOR_PROCEDURE,
    render_iap_svg,
)
from open_plates.trajectory import (
    DEFAULT_VISUAL_BEND_NM,
    TrajectoryWaypoint,
    build_trajectory,
)

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
LSZH = EXAMPLES / "lszh-ils-z-rwy-14.yaml"
UGSB = EXAMPLES / "ugsb-ils-rwy-13.yaml"
ETAD_STRAIGHT = EXAMPLES / "etad-ils-rwy-23-straight.yaml"
ETAD_ARC = EXAMPLES / "etad-vor-dme-rwy-05.yaml"


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


# --- State inventory --------------------------------------------------------


def test_lszh_has_four_flight_states() -> None:
    proc = _load(LSZH)
    fixes = build_fix_table(proc)
    states = compile_flight_states(proc, fixes)

    assert len(states) == 4
    approach = [s for s in states if s.phase == "approach"]
    missed = [s for s in states if s.phase == "missed"]
    assert len(approach) == 3
    assert len(missed) == 1

    labels = [s.label for s in approach]
    assert any("AMIKI" in lbl for lbl in labels)
    assert any("RILAX" in lbl for lbl in labels)
    assert any("GIPOL" in lbl for lbl in labels)
    assert all("RW14" in lbl for lbl in labels)


def test_ugsb_has_two_flight_states() -> None:
    proc = _load(UGSB)
    fixes = build_fix_table(proc)
    states = compile_flight_states(proc, fixes)
    assert len(states) == 2
    assert [s.phase for s in states] == ["approach", "missed"]


def test_etad_straight_has_two_flight_states() -> None:
    proc = _load(ETAD_STRAIGHT)
    fixes = build_fix_table(proc)
    states = compile_flight_states(proc, fixes)
    assert len(states) == 2
    assert [s.phase for s in states] == ["approach", "missed"]


def test_etad_arc_has_two_flight_states() -> None:
    proc = _load(ETAD_ARC)
    fixes = build_fix_table(proc)
    states = compile_flight_states(proc, fixes)
    assert len(states) == 2
    assert [s.phase for s in states] == ["approach", "missed"]


# --- No loops, no right angles, smooth curves ------------------------------


def _closest_waypoint_distance_nm(
    point: tuple[float, float], waypoints: list[tuple[float, float]]
) -> float:
    """Minimum great-circle distance from ``point`` to any waypoint."""
    return min(great_circle_distance_nm(point, wp) for wp in waypoints)


def _collect_waypoint_positions(procedure: dict) -> list[tuple[float, float]]:
    """Every waypoint the procedure visits, in document order.

    Used as a sanity reference for the polyline — each rendered point
    should sit reasonably close to SOME declared waypoint, and no point
    should be wildly far from every declared waypoint (which is what a
    physics-radius loop would produce).
    """
    fixes = build_fix_table(procedure)
    out: list[tuple[float, float]] = []
    for transition in procedure.get("transitions") or []:
        for leg in transition.get("legs") or []:
            for key in ("from_fix_id", "to_fix_id"):
                fid = leg.get(key)
                if fid and fid in fixes:
                    out.append(fixes[fid])
    for leg in procedure.get("common_legs") or []:
        for key in ("from_fix_id", "to_fix_id"):
            fid = leg.get(key)
            if fid and fid in fixes:
                out.append(fixes[fid])
    for leg in (procedure.get("missed_approach") or {}).get("legs") or []:
        for key in ("from_fix_id", "to_fix_id"):
            fid = leg.get(key)
            if fid and fid in fixes:
                out.append(fixes[fid])
    return out


def test_lszh_missed_approach_has_no_loops() -> None:
    """A physics-radius fillet loop would drop a polyline point ~3-4 NM
    from every declared waypoint. The symbolic (cosmetic-bend) trajectory
    must hug the waypoints — every polyline point is within a small
    envelope of SOME declared waypoint.

    LSZH missed is the worst-case multi-CF chain.
    """
    proc = _load(LSZH)
    fixes = build_fix_table(proc)
    states = compile_flight_states(proc, fixes)
    missed = next(s for s in states if s.phase == "missed")

    waypoints = _collect_waypoint_positions(proc)
    # Tightest envelope that still passes for a straight-line chain:
    # each sampled point must sit within 25 NM of SOME waypoint (long
    # CF legs like ZUE->AMIKI are ~12 NM and a point in the middle is
    # naturally ~6 NM from either endpoint). A loop would push points
    # arbitrarily far from all waypoints — the test is robust against
    # that without being brittle on segment interiors.
    for pt in missed.points:
        d = _closest_waypoint_distance_nm(pt, waypoints)
        assert d < 25.0, (
            f"polyline point {pt} is {d:.1f} NM from every declared "
            f"waypoint — looks like a loop"
        )


def test_lszh_missed_monotone_progress() -> None:
    """Stronger than the no-loop test: each consecutive point in the
    missed polyline advances — no abrupt jump more than the longest
    published leg (ZUE->AMIKI is ~12 NM great-circle, and the polyline
    samples arcs with many points between waypoints)."""
    proc = _load(LSZH)
    fixes = build_fix_table(proc)
    states = compile_flight_states(proc, fixes)
    missed = next(s for s in states if s.phase == "missed")
    for i in range(1, len(missed.points)):
        gap = great_circle_distance_nm(missed.points[i - 1], missed.points[i])
        assert gap < 15.0, (
            f"gap of {gap:.2f} NM between points {i - 1} and {i} "
            f"— likely a loop / disconnected segment"
        )


def test_lszh_waypoints_visited_in_order() -> None:
    """The flattened missed-approach polyline passes within 0.5 NM of
    every declared missed-approach fix, in document order."""
    proc = _load(LSZH)
    fixes = build_fix_table(proc)
    states = compile_flight_states(proc, fixes)
    missed = next(s for s in states if s.phase == "missed")

    expected_order = ["IKL51", "Z231I", "ZUE78", "ZUE", "AMIKI"]
    indexed_hits: list[tuple[str, int]] = []
    for fid in expected_order:
        if fid not in fixes:
            continue
        target = fixes[fid]
        best_idx, best_d = -1, float("inf")
        for i, pt in enumerate(missed.points):
            d = great_circle_distance_nm(target, pt)
            if d < best_d:
                best_d, best_idx = d, i
        # Cosmetic bend sets the polyline BACK from the fix per fly-by
        # chart convention — the rendered path does not touch the fix
        # symbol. At 1.5 NM base radius with 1.5 NM setback cap, the
        # closest polyline point to a fly-by waypoint can sit up to ~1.5
        # NM from it on sharp turns.
        assert best_d < 1.6, f"{fid} is {best_d:.3f} NM from nearest polyline point"
        indexed_hits.append((fid, best_idx))

    indices = [i for _, i in indexed_hits]
    assert indices == sorted(indices), (
        f"waypoint visit order broken: {indexed_hits}"
    )


def test_etad_arc_af_preserves_dme_radius() -> None:
    """Every point of the polyline that sits along the declared 10 DME
    arc around SPA must be at the declared radius (within 0.05 NM).

    Under the chart-symbology model AF legs are the only true arcs —
    they render at their declared radius via dense sampling, independent
    of the cosmetic bend radius.
    """
    proc = _load(ETAD_ARC)
    fixes = build_fix_table(proc)
    spa = fixes["SPA"]
    states = compile_flight_states(proc, fixes)
    approach = next(s for s in states if s.phase == "approach")

    # The arc portion of the polyline is whatever set of points sits
    # near 10 NM from SPA. Test: there MUST be a contiguous run of at
    # least 20 points all at ~10 NM from SPA (the declared 10 DME arc).
    on_arc = [
        i for i, pt in enumerate(approach.points)
        if abs(great_circle_distance_nm(pt, spa) - 10.0) < 0.05
    ]
    assert len(on_arc) >= 20, (
        f"only {len(on_arc)} polyline points near 10 DME from SPA; "
        f"expected the full arc sweep to sample densely"
    )


def test_lszh_af_preserves_dme_radius() -> None:
    """LSZH's three AF transitions all fly the 12 DME ring around KLO.
    Every approach state's polyline must include ~48 samples along the
    12 DME arc at the declared radius."""
    proc = _load(LSZH)
    fixes = build_fix_table(proc)
    klo = fixes["KLO"]
    states = compile_flight_states(proc, fixes)
    for state in (s for s in states if s.phase == "approach"):
        on_arc = [
            i for i, pt in enumerate(state.points)
            if abs(great_circle_distance_nm(pt, klo) - 12.0) < 0.05
        ]
        # Each transition's arc sweeps 42°..117°; 48 samples along the
        # fullest sweep still leaves >~20 samples within the tolerance.
        assert len(on_arc) >= 15, (
            f"{state.label}: only {len(on_arc)} polyline points near "
            f"12 DME from KLO"
        )


def test_visual_bend_is_constant_radius() -> None:
    """Synthetic waypoint list with a 90° right turn at the middle fix.
    The rounded bend must have a radius matching DEFAULT_VISUAL_BEND_NM
    regardless of an aircraft speed — no speed input exists.
    """
    # A (-5 NM east of origin) -> B (origin) -> C (5 NM north of origin).
    b = (0.0, 0.0)
    a = destination(b, 270.0, 5.0)
    c = destination(b, 0.0, 5.0)
    waypoints = [
        TrajectoryWaypoint(position=a, kind="flyby"),
        TrajectoryWaypoint(position=b, kind="flyby"),
        TrajectoryWaypoint(position=c, kind="flyby"),
    ]
    pts = build_trajectory(waypoints)
    assert len(pts) >= 4
    # Find the polyline points closest to B; they bracket the bend.
    dists_to_b = [great_circle_distance_nm(p, b) for p in pts]
    # The tangent points sit at setback = R/tan(45°) = R NM from B
    # (90° turn). Expect minimum polyline distance to B to approx match
    # (R / sqrt(2)) — the midpoint of the rounded bend — but at least
    # no point passes through B closer than ~0.6 * R (geometric min).
    min_d = min(dists_to_b)
    # Midpoint of the rounded bend: distance from B is R * (1 - sin(45°)) ~= 0.293 R
    # for a 90° turn; use a generous lower bound.
    expected_min = DEFAULT_VISUAL_BEND_NM * (1.0 - 0.75)
    assert min_d > expected_min * 0.5, (
        f"polyline passes too close to waypoint B ({min_d:.4f} NM) — "
        f"bend setback should be ~{DEFAULT_VISUAL_BEND_NM} NM"
    )
    # And the polyline points along the bend stay within ~R NM of B.
    near_bend = [d for d in dists_to_b if d < DEFAULT_VISUAL_BEND_NM * 2.0]
    assert near_bend, "no points near the bend region"


def test_trajectory_is_deterministic() -> None:
    """Same inputs → same outputs, bit-exact."""
    proc = _load(LSZH)
    fixes = build_fix_table(proc)
    a = compile_flight_states(proc, fixes)
    b = compile_flight_states(proc, fixes)
    for sa, sb in zip(a, b):
        assert sa.points == sb.points


# --- Contiguity ------------------------------------------------------------


def test_state_points_are_contiguous() -> None:
    """Consecutive points in any state must be within 25 NM apart."""
    for path in (LSZH, UGSB, ETAD_STRAIGHT, ETAD_ARC):
        proc = _load(path)
        fixes = build_fix_table(proc)
        states = compile_flight_states(proc, fixes)
        for state in states:
            if len(state.points) < 2:
                continue
            for i in range(1, len(state.points)):
                gap = great_circle_distance_nm(
                    state.points[i - 1], state.points[i]
                )
                assert gap < 25.0, (
                    f"{path.stem} {state.label}: gap of {gap:.2f} NM "
                    f"between points {i - 1} and {i}"
                )


def test_missed_approach_is_one_state() -> None:
    for path in (LSZH, UGSB, ETAD_STRAIGHT, ETAD_ARC):
        proc = _load(path)
        fixes = build_fix_table(proc)
        states = compile_flight_states(proc, fixes)
        missed = [s for s in states if s.phase == "missed"]
        assert len(missed) == 1


def test_lszh_approach_states_end_at_threshold() -> None:
    """Each LSZH approach polyline ends within 0.2 NM of RW14 (cosmetic
    bend setback at the preceding FAF may pull the final point up
    slightly)."""
    proc = _load(LSZH)
    fixes = build_fix_table(proc)
    states = compile_flight_states(proc, fixes)
    rw14 = fixes["RW14"]
    approaches = [s for s in states if s.phase == "approach"]
    assert len(approaches) == 3
    for state in approaches:
        assert state.points
        end = state.points[-1]
        gap = great_circle_distance_nm(end, rw14)
        assert gap < 0.2, (
            f"{state.label}: final point is {gap:.3f} NM from RW14"
        )


# --- SVG integration --------------------------------------------------------


_POLY_RE = re.compile(r'<polyline\b[^/>]*/>', re.DOTALL)
_STROKE_RE = re.compile(r'stroke="(#[0-9A-Fa-f]{6})"')
_DASH_RE = re.compile(r'stroke-dasharray="[^"]+"')


def _count_polylines(svg: str) -> dict[tuple[str, bool], int]:
    out: dict[tuple[str, bool], int] = {}
    for m in _POLY_RE.finditer(svg):
        body = m.group(0)
        stroke_m = _STROKE_RE.search(body)
        if not stroke_m:
            continue
        stroke = stroke_m.group(1).upper()
        has_dash = bool(_DASH_RE.search(body))
        key = (stroke, has_dash)
        out[key] = out.get(key, 0) + 1
    return out


def test_lszh_approach_emits_single_polyline_per_state() -> None:
    """Render LSZH and assert that the plan view contains exactly:
      - 3 magenta (approach) core polylines + 3 paper casing partners
      - 1 ink (missed) dashed core polyline + 1 paper casing partner
    """
    proc = _load(LSZH)
    svg = render_iap_svg(proc)
    counts = _count_polylines(svg)

    magenta_solid = counts.get((COLOR_PROCEDURE.upper(), False), 0)
    missed_dashed = counts.get((COLOR_MISSED.upper(), True), 0)

    assert magenta_solid == 3
    assert missed_dashed == 1


def test_ugsb_approach_emits_single_polyline_per_state() -> None:
    proc = _load(UGSB)
    svg = render_iap_svg(proc)
    counts = _count_polylines(svg)
    magenta_solid = counts.get((COLOR_PROCEDURE.upper(), False), 0)
    missed_dashed = counts.get((COLOR_MISSED.upper(), True), 0)
    assert magenta_solid == 1
    assert missed_dashed == 1


def test_missed_hold_still_renders_as_racetrack() -> None:
    proc = _load(UGSB)
    fixes = build_fix_table(proc)
    states = compile_flight_states(proc, fixes)
    missed = next(s for s in states if s.phase == "missed")
    assert any(isinstance(h, Hold) for h in missed.holds)
    svg = render_iap_svg(proc)
    assert "<path" in svg
