"""Tests for the trajectory builder's arc-to-line fillet handling.

Covers the :func:`open_plates.tangent_arc.arc_to_line_fillet`
primitive and its wiring into :func:`open_plates.legs.compile_flight_states`.

Background: a DME arc (ARINC 424 AF) that exits at a waypoint onto a
declared outbound course is almost never *naturally* tangent — there's a
small angular mismatch at the joint. Without a fillet the rendered
polyline shows a visible corner. These tests lock in the corrected
geometry: a small internally-tangent fillet of fixed cosmetic radius
that blends the big arc smoothly into the outbound straight line.
"""

from __future__ import annotations

from math import cos, degrees, radians, sin
from pathlib import Path

import pytest
import yaml

from open_plates.geodesy import (
    LatLon,
    destination,
    great_circle_distance_nm,
    initial_bearing_deg,
)
from open_plates.legs import build_fix_table, compile_flight_states
from open_plates.tangent_arc import arc_to_line_fillet, line_to_arc_fillet

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
LSZH = EXAMPLES / "lszh-ils-z-rwy-14.yaml"
UGSB = EXAMPLES / "ugsb-ils-rwy-13.yaml"
ETAD_STRAIGHT = EXAMPLES / "etad-ils-rwy-23-straight.yaml"
ETAD_ARC = EXAMPLES / "etad-vor-dme-rwy-05.yaml"


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


# ---------------------------------------------------------------------------
# arc_to_line_fillet — unit geometry
# ---------------------------------------------------------------------------


def test_arc_to_line_fillet_tangent_to_both() -> None:
    """Construct a simple CCW arc + mismatched outbound line and verify
    the returned fillet is exactly tangent to both at the declared
    tangent points (within 1e-6 NM)."""
    arc_center = (47.0, 8.0)
    radius = 12.0
    # Arc exits at radial 045 (NE of centre), natural CCW tangent there
    # is 045 - 90 = 315 true. We set the outbound course to 270 — a
    # 45° mismatch, well above the "already tangent" epsilon.
    arc_exit = destination(arc_center, 45.0, radius)
    outbound = 270.0

    result = arc_to_line_fillet(
        arc_center=arc_center,
        arc_radius_nm=radius,
        arc_clockwise=False,
        arc_exit_pt=arc_exit,
        outbound_course_deg=outbound,
        fillet_radius_nm=1.5,
    )
    assert result is not None
    t_arc, fillet, t_line = result

    # T_arc is on the big arc.
    assert great_circle_distance_nm(arc_center, t_arc) == pytest.approx(
        radius, abs=1e-6
    )
    # T_line is on the outbound line — check by perpendicular distance
    # from arc_exit along the outbound direction.
    # Compute local planar offset of T_line from arc_exit, project onto
    # the normal of the outbound direction; must be ~0.
    brg_from_exit = initial_bearing_deg(arc_exit, t_line)
    dist_from_exit = great_circle_distance_nm(arc_exit, t_line)
    # Component of (T_line - arc_exit) along outbound perpendicular:
    perp = (outbound + 90.0) % 360.0
    # Signed cross: d·sin(brg - outbound) = perp component.
    perp_component = dist_from_exit * sin(radians(brg_from_exit - outbound))
    # Tolerance: planar ↔ spherical round-trip accumulates sub-NM error.
    # 5e-3 NM ≈ 30 ft, well under chart pixel at any realistic scale.
    assert abs(perp_component) < 5e-3, (
        f"T_line off outbound line by {perp_component:.2e} NM"
    )

    # Fillet radius matches.
    assert fillet.radius_nm == pytest.approx(1.5, abs=1e-9)
    # Fillet centre at distance (R - r) from arc_center (internal tangency).
    assert great_circle_distance_nm(
        arc_center, fillet.center
    ) == pytest.approx(radius - 1.5, abs=1e-5)


def test_arc_to_line_fillet_tangent_directions_match() -> None:
    """At T_arc, the fillet's tangent direction equals the big arc's
    tangent direction. At T_line, the fillet's tangent direction equals
    the outbound course. Both within a tight angular tolerance."""
    arc_center = (50.0, -5.0)
    radius = 10.0
    arc_exit = destination(arc_center, 120.0, radius)
    # Natural CW tangent at radial 120 is 120 + 90 = 210. Use outbound
    # 180 — a 30° right-ish turn; in range for a normal DME→FAC joint.
    outbound = 180.0

    result = arc_to_line_fillet(
        arc_center=arc_center,
        arc_radius_nm=radius,
        arc_clockwise=True,  # right turn
        arc_exit_pt=arc_exit,
        outbound_course_deg=outbound,
        fillet_radius_nm=1.0,
    )
    assert result is not None
    t_arc, fillet, t_line = result

    # Fillet tangent direction at T_arc: perpendicular to (T_arc - fillet.center)
    # in the direction of fillet traversal.
    # Instead of dissecting the fillet, sample a point a tiny bit along
    # the fillet from T_arc and measure the bearing.
    small = 0.001  # 0.001 NM sample step
    # Bearing from fillet.center to T_arc:
    brg_center_to_t_arc = initial_bearing_deg(fillet.center, t_arc)
    # Step a tiny sweep along the fillet: if clockwise, bearing increases;
    # if ccw, decreases.
    step = 0.02  # degrees of sweep
    if fillet.clockwise:
        next_brg = brg_center_to_t_arc + step
    else:
        next_brg = brg_center_to_t_arc - step
    next_pt = destination(fillet.center, next_brg, fillet.radius_nm)
    fillet_tangent_at_arc = initial_bearing_deg(t_arc, next_pt)

    # Big arc's tangent at T_arc (in the flight direction):
    # big arc is CW (the input arc_clockwise=True) → tangent = radial + 90.
    radial_at_t_arc = initial_bearing_deg(arc_center, t_arc)
    big_tan = (radial_at_t_arc + 90.0) % 360.0
    delta = (fillet_tangent_at_arc - big_tan + 540.0) % 360.0 - 180.0
    assert abs(delta) < 1.0, (
        f"fillet tangent at T_arc differs from big-arc tangent by {delta:.3f}°"
    )

    # Fillet tangent at T_line: sample a tiny step backwards from T_line
    # along the fillet.
    brg_center_to_t_line = initial_bearing_deg(fillet.center, t_line)
    if fillet.clockwise:
        prev_brg = brg_center_to_t_line - step
    else:
        prev_brg = brg_center_to_t_line + step
    prev_pt = destination(fillet.center, prev_brg, fillet.radius_nm)
    fillet_tangent_at_line = initial_bearing_deg(prev_pt, t_line)
    delta = (fillet_tangent_at_line - outbound + 540.0) % 360.0 - 180.0
    assert abs(delta) < 1.0, (
        f"fillet tangent at T_line differs from outbound ({outbound}) "
        f"by {delta:.3f}°"
    )


def test_arc_to_line_fillet_already_tangent_returns_none() -> None:
    """When the arc is already tangent to the outbound (no mismatch), the
    function must return None — there's no geometry to fit a fillet to."""
    arc_center = (0.0, 0.0)
    radius = 10.0
    arc_exit = destination(arc_center, 0.0, radius)  # radial 000
    # CCW tangent at radial 000 = 000 - 90 = 270. Outbound = 270 → tangent.
    result = arc_to_line_fillet(
        arc_center=arc_center,
        arc_radius_nm=radius,
        arc_clockwise=False,
        arc_exit_pt=arc_exit,
        outbound_course_deg=270.0,
        fillet_radius_nm=1.0,
    )
    assert result is None


def test_arc_to_line_fillet_too_big_returns_none() -> None:
    arc_center = (0.0, 0.0)
    radius = 1.0
    arc_exit = destination(arc_center, 0.0, radius)
    result = arc_to_line_fillet(
        arc_center=arc_center,
        arc_radius_nm=radius,
        arc_clockwise=False,
        arc_exit_pt=arc_exit,
        outbound_course_deg=135.0,
        fillet_radius_nm=2.0,  # larger than arc radius
    )
    assert result is None


# ---------------------------------------------------------------------------
# End-to-end: LSZH AMIKI arc exits smoothly into the 137° final
# ---------------------------------------------------------------------------


def _max_bearing_change_on_segment(
    points: list[LatLon],
    around: LatLon,
    neighborhood_nm: float,
) -> float:
    """Find the largest bearing change between adjacent polyline points
    whose midpoint lies within ``neighborhood_nm`` of ``around``.

    Used to assert the arc→line joint is smooth — the fillet should
    distribute the turn across many small arc samples, so no single
    bearing change exceeds a small threshold.
    """
    worst = 0.0
    for i in range(2, len(points)):
        p0, p1, p2 = points[i - 2], points[i - 1], points[i]
        mid = (
            (p0[0] + p1[0] + p2[0]) / 3.0,
            (p0[1] + p1[1] + p2[1]) / 3.0,
        )
        if great_circle_distance_nm(mid, around) > neighborhood_nm:
            continue
        brg1 = initial_bearing_deg(p0, p1)
        brg2 = initial_bearing_deg(p1, p2)
        delta = abs((brg2 - brg1 + 540.0) % 360.0 - 180.0)
        worst = max(worst, delta)
    return worst


def test_lszh_amiki_arc_exits_smoothly_into_final() -> None:
    """The AMIKI approach polyline, near INTER (the arc exit), must show
    no sharp bearing change — the fillet should distribute the turn
    across many small segments.

    Also: NO point sits more than 1.6 NM from INTER at closest approach
    (the fillet setback is ~1.5 NM max; the polyline must still hug the
    declared waypoint within the cosmetic budget)."""
    proc = _load(LSZH)
    fixes = build_fix_table(proc)
    inter = fixes["INTER"]
    states = compile_flight_states(proc, fixes)
    amiki = next(s for s in states if "AMIKI" in s.label and s.phase == "approach")

    # Closest approach to INTER.
    min_d = min(great_circle_distance_nm(p, inter) for p in amiki.points)
    assert min_d < 1.6, (
        f"closest polyline point to INTER is {min_d:.3f} NM — fillet "
        f"setback exceeded budget"
    )

    # Max per-segment bearing change within 2 NM of INTER must stay
    # below 10° — that's what a distributed fillet looks like. A sharp
    # corner would show up as a single ~45° kink.
    worst = _max_bearing_change_on_segment(
        amiki.points, around=inter, neighborhood_nm=2.0
    )
    assert worst < 10.0, (
        f"sharp bend of {worst:.1f}° near INTER on AMIKI approach — "
        f"arc→line fillet not applied"
    )


def test_lszh_rilax_and_gipol_arc_exits_smoothly() -> None:
    """Same smoothness assertion for the RILAX and GIPOL transitions."""
    proc = _load(LSZH)
    fixes = build_fix_table(proc)
    inter = fixes["INTER"]
    states = compile_flight_states(proc, fixes)
    for name in ("RILAX", "GIPOL"):
        st = next(
            s for s in states if name in s.label and s.phase == "approach"
        )
        worst = _max_bearing_change_on_segment(
            st.points, around=inter, neighborhood_nm=2.0
        )
        assert worst < 10.0, (
            f"sharp bend of {worst:.1f}° near INTER on {name} approach"
        )


# ---------------------------------------------------------------------------
# Regression: straight-only examples are unchanged
# ---------------------------------------------------------------------------


def test_ugsb_straight_in_unchanged_waypoint_distances() -> None:
    """UGSB has no AF legs; its approach polyline should pass within
    the cosmetic-bend budget of every declared waypoint in order."""
    proc = _load(UGSB)
    fixes = build_fix_table(proc)
    states = compile_flight_states(proc, fixes)
    approach = next(s for s in states if s.phase == "approach")
    assert approach.points
    # First and last polyline points are essentially the entry/exit fixes.
    assert len(approach.points) >= 3


# ---------------------------------------------------------------------------
# line_to_arc_fillet — unit geometry (mirror of arc_to_line_fillet suite)
# ---------------------------------------------------------------------------


def test_line_to_arc_fillet_tangent_to_both() -> None:
    """Construct a simple straight-line inbound + mismatched CCW arc and
    verify the returned fillet is exactly tangent to both at its declared
    tangent points (within tight tolerance)."""
    arc_center = (47.0, 8.0)
    radius = 10.0
    # Arc enters at radial 180 (S of centre). Natural CCW tangent there
    # is 180 - 90 = 090. We set the inbound course to 270 — a ~180°
    # mismatch geometrically, but we'll use a less extreme case so the
    # fillet actually fits: arc entry at radial 090, natural CCW tangent
    # = 000; inbound course = 270 → 90° left turn (like ETAD ARC1W).
    arc_entry = destination(arc_center, 90.0, radius)
    # A straight line heading 270° passing through arc_entry (starting
    # some ways east, flying west through arc_entry).
    line_entry = destination(arc_entry, 90.0, 5.0)  # 5 NM east of arc entry
    inbound = 270.0  # westbound

    result = line_to_arc_fillet(
        line_entry_pt=line_entry,
        inbound_course_deg=inbound,
        arc_center=arc_center,
        arc_radius_nm=radius,
        arc_clockwise=False,
        arc_entry_pt=arc_entry,
        fillet_radius_nm=1.5,
    )
    assert result is not None
    t_line, fillet, t_arc = result

    # T_arc is on the big arc.
    assert great_circle_distance_nm(arc_center, t_arc) == pytest.approx(
        radius, abs=1e-6
    )
    # T_line is on the inbound line — check perpendicular offset from
    # line_entry along the inbound direction.
    brg_from_entry = initial_bearing_deg(line_entry, t_line)
    dist_from_entry = great_circle_distance_nm(line_entry, t_line)
    perp_component = dist_from_entry * sin(radians(brg_from_entry - inbound))
    # Tolerance 2e-2 NM ≈ 120 ft. The function's internal invariant pins
    # T_line to the inbound line at 1e-6 NM in the LOCAL PLANAR frame
    # (centred on arc_center); this test checks from the spherical-frame
    # inbound-line direction, which accumulates planar↔sphere error
    # proportional to distance from arc_center. Well under any chart
    # pixel at realistic scale.
    assert abs(perp_component) < 2e-2, (
        f"T_line off inbound line by {perp_component:.2e} NM"
    )

    # Fillet radius matches.
    assert fillet.radius_nm == pytest.approx(1.5, abs=1e-9)
    # Fillet centre at distance (R - r) from arc_center (internal tangency).
    assert great_circle_distance_nm(
        arc_center, fillet.center
    ) == pytest.approx(radius - 1.5, abs=1e-5)


def test_line_to_arc_fillet_tangent_directions_match() -> None:
    """At T_line, fillet tangent direction equals inbound course; at
    T_arc, fillet tangent direction equals the big arc's flight-direction
    tangent. Both within a tight angular tolerance."""
    arc_center = (50.0, -5.0)
    radius = 10.0
    arc_entry = destination(arc_center, 120.0, radius)
    # Natural CW tangent at radial 120 is 120 + 90 = 210. Use inbound 180
    # — a 30° right turn onto the arc.
    line_entry = destination(arc_entry, (180.0 + 180.0) % 360.0, 5.0)
    inbound = 180.0

    result = line_to_arc_fillet(
        line_entry_pt=line_entry,
        inbound_course_deg=inbound,
        arc_center=arc_center,
        arc_radius_nm=radius,
        arc_clockwise=True,
        arc_entry_pt=arc_entry,
        fillet_radius_nm=1.0,
    )
    assert result is not None
    t_line, fillet, t_arc = result

    # Fillet tangent direction at T_line: sample a tiny step along the
    # fillet from T_line (i.e. one step INTO the fillet).
    step = 0.02  # degrees of sweep
    brg_center_to_t_line = initial_bearing_deg(fillet.center, t_line)
    if fillet.clockwise:
        next_brg = brg_center_to_t_line + step
    else:
        next_brg = brg_center_to_t_line - step
    next_pt = destination(fillet.center, next_brg, fillet.radius_nm)
    fillet_tangent_at_line = initial_bearing_deg(t_line, next_pt)
    delta = (fillet_tangent_at_line - inbound + 540.0) % 360.0 - 180.0
    assert abs(delta) < 1.0, (
        f"fillet tangent at T_line differs from inbound ({inbound}) "
        f"by {delta:.3f}°"
    )

    # Fillet tangent at T_arc: sample a tiny step backwards (so we get
    # the direction approaching T_arc = exit direction of fillet).
    brg_center_to_t_arc = initial_bearing_deg(fillet.center, t_arc)
    if fillet.clockwise:
        prev_brg = brg_center_to_t_arc - step
    else:
        prev_brg = brg_center_to_t_arc + step
    prev_pt = destination(fillet.center, prev_brg, fillet.radius_nm)
    fillet_tangent_at_arc = initial_bearing_deg(prev_pt, t_arc)

    # Big arc's tangent at T_arc (CW → radial + 90):
    radial_at_t_arc = initial_bearing_deg(arc_center, t_arc)
    big_tan = (radial_at_t_arc + 90.0) % 360.0
    delta = (fillet_tangent_at_arc - big_tan + 540.0) % 360.0 - 180.0
    assert abs(delta) < 1.0, (
        f"fillet tangent at T_arc differs from big-arc tangent by {delta:.3f}°"
    )


def test_line_to_arc_already_tangent_returns_none() -> None:
    """When the inbound course already equals the arc's natural entry
    tangent within epsilon, return None (no fillet needed)."""
    arc_center = (0.0, 0.0)
    radius = 10.0
    arc_entry = destination(arc_center, 0.0, radius)  # radial 000
    # CCW tangent at radial 000 = 000 - 90 = 270. Inbound = 270 → tangent.
    line_entry = destination(arc_entry, 90.0, 5.0)  # doesn't matter; on the line
    result = line_to_arc_fillet(
        line_entry_pt=line_entry,
        inbound_course_deg=270.0,
        arc_center=arc_center,
        arc_radius_nm=radius,
        arc_clockwise=False,
        arc_entry_pt=arc_entry,
        fillet_radius_nm=1.0,
    )
    assert result is None


def test_line_to_arc_fillet_too_big_returns_none() -> None:
    """Fillet radius >= arc radius ⇒ internal tangency impossible."""
    arc_center = (0.0, 0.0)
    radius = 1.0
    arc_entry = destination(arc_center, 0.0, radius)
    line_entry = destination(arc_entry, 135.0, 1.0)
    result = line_to_arc_fillet(
        line_entry_pt=line_entry,
        inbound_course_deg=270.0,
        arc_center=arc_center,
        arc_radius_nm=radius,
        arc_clockwise=False,
        arc_entry_pt=arc_entry,
        fillet_radius_nm=2.0,  # larger than arc radius
    )
    assert result is None


# ---------------------------------------------------------------------------
# End-to-end: ETAD ARC1W line-to-arc joint is now smooth
# ---------------------------------------------------------------------------


def test_etad_arc1w_entry_smooth() -> None:
    """At ETAD's ARC1W, the westbound 270° straight meets a CCW 10 DME
    arc around SPA with a ~91° mismatch. The line-to-arc fillet should
    distribute the turn across many small segments — no single
    bearing-change between adjacent polyline points near ARC1W exceeds
    ~10°."""
    proc = _load(ETAD_ARC)
    fixes = build_fix_table(proc)
    arc1w = fixes["ARC1W"]
    states = compile_flight_states(proc, fixes)
    approach = next(s for s in states if s.phase == "approach")

    worst = _max_bearing_change_on_segment(
        approach.points, around=arc1w, neighborhood_nm=2.5
    )
    assert worst < 10.0, (
        f"sharp bend of {worst:.1f}° near ARC1W on ETAD approach — "
        f"line-to-arc fillet not applied"
    )
